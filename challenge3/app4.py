import os
from dotenv import load_dotenv
load_dotenv()

from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential


# Manual RAG with Azure OpenAI + Azure AI Search

# Azure OpenAI Client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
EMB_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

# Azure AI Search config
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY")
SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

search = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX,
    credential=AzureKeyCredential(SEARCH_KEY),
)

# Index field mappings
CONTENT_FIELD = os.getenv("AZURE_SEARCH_CONTENT_FIELD")
TITLE_FIELD = os.getenv("AZURE_SEARCH_TITLE_FIELD")
VECTOR_FIELD = os.getenv("AZURE_SEARCH_VECTOR_FIELD")

# retrieve k relevant docs from Azure Search
def retrieve(query: str, k: int = 5):
    # 1- Query embedding
    emb = client.embeddings.create(
        model=EMB_DEPLOYMENT,
        input=query
    ).data[0].embedding

    # 2- Vector + Keyword (hybrid) arama
    results = search.search(
        search_text=query,
        vector_queries=[
            VectorizedQuery(
                vector=emb,
                k_nearest_neighbors=k,
                fields=VECTOR_FIELD
            )
        ],
        select=[CONTENT_FIELD, TITLE_FIELD],
        top=k,
    )

    docs = []
    for r in results:
        docs.append({
            "title": r.get(TITLE_FIELD, ""),
            "content": r.get(CONTENT_FIELD, ""),
        })

    return docs

# Ask function with context injection
def ask(q: str) -> str:
    docs = retrieve(q, k=5)

    if not docs:
        return "Bilmiyorum. (Index'ten ilgili içerik gelmedi.)"

    context = "\n\n".join(
        [f"[{i+1}] {d['title']}\n{d['content']}" for i, d in enumerate(docs)]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant.\n"
                "Use ONLY the provided context. If the answer is not in the context, say you don't know."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {q}",
        },
    ]

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        temperature=0.2,
    )

    return resp.choices[0].message.content or ""

if __name__ == "__main__":
    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        print("AI:", ask(q))
