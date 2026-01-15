import os
import time
import asyncio
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv

from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace

from openai import AzureOpenAI
from azure.cosmos import CosmosClient


TOP_K = 5

def setup_logging():
    """Configure logging levels for the application and suppress verbose third-party logs."""
    logging.basicConfig(level=logging.INFO)
    # Suppress verbose Azure SDK logs
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    # Reduce OpenTelemetry noise
    logging.getLogger("opentelemetry").setLevel(logging.WARNING)


def setup_observability():
    """Initialize Azure Monitor telemetry for distributed tracing."""
    conn = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    configure_azure_monitor(connection_string=conn)


def build_sources_context(docs: List[Dict[str, Any]]) -> str:
    """Format retrieved documents into numbered source context for the LLM.
    
    Args:
        docs: List of document dictionaries with 'title' and 'content' fields
        
    Returns:
        Formatted string with numbered sources
    """
    chunks = []
    for i, d in enumerate(docs, start=1):
        title = d.get("title", "")
        content = d.get("content", "")

        # Truncate long content to prevent context overflow
        if len(content) > 1200:
            content = content[:1200] + "..."

        chunks.append(f"[{i}] {title}\n{content}")
    return "\n\n".join(chunks)


async def main():
    load_dotenv()

    setup_logging()
    setup_observability()

    logger = logging.getLogger("app")
    tracer = trace.get_tracer("cosmos-rag-app")

    # Azure OpenAI Client
    aoai = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    EMB_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

    # Cosmos Client
    cosmos = CosmosClient(os.getenv("COSMOS_ENDPOINT"), credential=os.getenv("COSMOS_KEY"))
    db = cosmos.get_database_client(os.getenv("COSMOS_DATABASE"))
    container = db.get_container_client(os.getenv("COSMOS_CONTAINER"))

    PK_VALUE = os.getenv("COSMOS_PARTITION_KEY_VALUE", "docs")
    VECTOR_FIELD = os.getenv("COSMOS_VECTOR_FIELD", "embedding")

    logger.info(f"[Retriever Backend] COSMOS_VECTOR | pk={PK_VALUE} | vector_field={VECTOR_FIELD}")

    # Stateful conversation memory (LLM messages)
    # Note: SOURCES are injected dynamically per turn, not stored in persistent memory
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant.\n"
                "You may use conversation history for context (follow-ups), but factual claims MUST be supported by SOURCES.\n"
                "Use ONLY the provided SOURCES for answering. If the answer is not in SOURCES, say you don't know.\n"
                "When answering, cite sources like [1], [2]."
            ),
        }
    ]

    def embed(text: str) -> List[float]:
        """Generate embedding vector for given text using Azure OpenAI."""
        resp = aoai.embeddings.create(
            model=EMB_DEPLOYMENT,
            input=text,
        )
        return resp.data[0].embedding

    def retrieve(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant documents using Cosmos DB vector search.
        
        Args:
            query: User query text
            k: Number of documents to retrieve
            
        Returns:
            List of document dictionaries with title and content
        """
        qvec = embed(query)

        # Execute Cosmos DB vector similarity search
        sql = f"""
        SELECT TOP @k c.title, c.content
        FROM c
        WHERE c.pk = @pk
        ORDER BY VectorDistance(c.{VECTOR_FIELD}, @qvec)
        """

        params = [
            {"name": "@k", "value": k},
            {"name": "@pk", "value": PK_VALUE},
            {"name": "@qvec", "value": qvec},
        ]

        items = list(
            container.query_items(
                query=sql,
                parameters=params,
                enable_cross_partition_query=True,
            )
        )
        return items

    def ask(user_text: str) -> Dict[str, Any]:
        """Process user query using RAG pattern: retrieve relevant docs, then generate answer.
        
        Args:
            user_text: User's question or query
            
        Returns:
            Dictionary with answer, usage stats, and performance metrics
        """
        # Step 1: Retrieve relevant documents from Cosmos DB
        t0 = time.time()
        docs = retrieve(user_text, TOP_K)
        t_retrieval = time.time() - t0

        if not docs:
            return {
                "answer": "I don't know. (No relevant content retrieved from Cosmos DB.)",
                "usage": None,
                "retrieval_sec": t_retrieval,
                "docs_count": 0,
            }

        sources = build_sources_context(docs)

        # Step 2: Generate answer using LLM with injected sources (context for this turn only)
        turn_messages = messages + [
            {
                "role": "user",
                "content": f"SOURCES:\n{sources}\n\nQuestion: {user_text}",
            }
        ]

        t1 = time.time()
        resp = aoai.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=turn_messages,
            temperature=0.2,
        )
        t_llm = time.time() - t1

        answer = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)

        # Step 3: Update conversation history with clean user/assistant turns
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "usage": usage,
            "retrieval_sec": t_retrieval,
            "llm_sec": t_llm,
            "docs_count": len(docs),
        }

    turn = 0

    while True:
        user_input = input("User > ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        turn += 1

        start_time = time.time()

        try:
            with tracer.start_as_current_span("chat.turn") as span:
                span.set_attribute("turn.id", turn)
                span.set_attribute("user.input_len", len(user_input))
                span.set_attribute("retriever.backend", "COSMOS_VECTOR")

                result = ask(user_input)

                elapsed = time.time() - start_time

                print("Assistant >", result["answer"])

                # Record telemetry metrics
                span.set_attribute("docs.count", int(result.get("docs_count", 0)))
                span.set_attribute("perf.elapsed_sec", float(elapsed))
                span.set_attribute("perf.retrieval_sec", float(result.get("retrieval_sec", 0.0)))
                span.set_attribute("perf.llm_sec", float(result.get("llm_sec", 0.0)))

                # Record token usage metrics
                usage = result.get("usage")
                info_parts = []
                if usage:
                    total_tokens = getattr(usage, "total_tokens", 0)
                    prompt_tokens = getattr(usage, "prompt_tokens", 0)
                    completion_tokens = getattr(usage, "completion_tokens", 0)

                    info_parts.append(f"{total_tokens} tokens (prompt: {prompt_tokens}, completion: {completion_tokens})")
                    span.set_attribute("tokens.total", int(total_tokens))
                    span.set_attribute("tokens.prompt", int(prompt_tokens))
                    span.set_attribute("tokens.completion", int(completion_tokens))

                info_parts.append(f"time: {elapsed:.2f}s")
                info_parts.append(f"retrieval: {result.get('retrieval_sec', 0.0):.2f}s")
                if "llm_sec" in result:
                    info_parts.append(f"llm: {result.get('llm_sec', 0.0):.2f}s")

                print(" | ".join(info_parts), "\n")

        except Exception as e:
            elapsed = time.time() - start_time
            logger.exception("Error during chat turn")
            print(f"Error: {e} | time: {elapsed:.2f}s\n")


if __name__ == "__main__":
    asyncio.run(main())
