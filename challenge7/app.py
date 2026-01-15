import asyncio
import re
from pathlib import Path

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments

# Blocked SQL keywords for safety
BLOCKED_KEYWORDS = {"insert", "update", "delete", "merge", "drop", "alter", 
                    "create", "truncate", "grant", "revoke", "exec", "execute"}

SYSTEM_PROMPT = """You are a senior database engineer.
Convert user requests into ONE valid T-SQL query for SQL Server.

Rules:
- Output ONLY the SQL query. No explanation, markdown, or backticks.
- Use only SELECT or CTE (WITH ... SELECT).
- Never use INSERT/UPDATE/DELETE/MERGE/DROP/ALTER/CREATE/TRUNCATE/EXEC.
- Use TOP 50 unless user specifies a different limit.
- Use schema-qualified names (e.g., HumanResources.Employee).

DATABASE SCHEMA:
{schema}"""


def load_schema(path: str = "dbschema.txt") -> str:
    """Load database schema from file."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return file.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return file.read_text(encoding="utf-8", errors="replace")


def clean_sql(text: str) -> str:
    """Remove markdown code blocks from LLM output."""
    text = re.sub(r"^```sql\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    return re.sub(r"\s*```$", "", text).strip()


def is_safe_query(sql: str) -> bool:
    """Check if SQL is a safe SELECT-only query."""
    normalized = sql.strip().lower()
    if not (normalized.startswith("select") or normalized.startswith("with")):
        return False
    return not any(re.search(rf"\b{kw}\b", normalized) for kw in BLOCKED_KEYWORDS)


async def main():
    load_dotenv()
    
    # Setup kernel with Azure OpenAI
    kernel = Kernel()
    kernel.add_service(AzureChatCompletion(service_id="aoai"))
    
    settings = kernel.get_prompt_execution_settings_from_service_id("aoai")
    settings.temperature = 0
    
    # Create agent with schema injected into prompt
    schema = load_schema("dbschema.txt")
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="NL2SQL",
        instructions=SYSTEM_PROMPT.format(schema=schema),
        arguments=KernelArguments(settings=settings),
    )
    
    # Thread for conversation memory
    thread = ChatHistoryAgentThread()
    
    print("\nNL2SQL Agent ready. Type 'exit' to quit.\n")
    
    while True:
        user_input = input("User> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        
        # Stream response from agent
        response = ""
        async for update in agent.invoke_stream(messages=user_input, thread=thread):
            if update.message:
                chunk = update.message.content
                print(chunk, end="", flush=True)
                response += chunk
        print("\n")
        
        # Validate generated SQL
        sql = clean_sql(response)
        if not is_safe_query(sql):
            print("Unsafe SQL detected. Please rephrase your query.\n")


if __name__ == "__main__":
    asyncio.run(main())
