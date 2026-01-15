import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# Azure OpenAI “On Your Data” / data_sources (retrieve + injection by Azure)

# Azure OpenAI Client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

# Azure AI Search config
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY")
SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

# Index field mappings
CONTENT_FIELD = os.getenv("AZURE_SEARCH_CONTENT_FIELD")
TITLE_FIELD = os.getenv("AZURE_SEARCH_TITLE_FIELD")
VECTOR_FIELD = os.getenv("AZURE_SEARCH_VECTOR_FIELD")

# Vector/Hybrid için embedding deployment
EMB_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

def ask(q: str) -> str:
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": q},
        ],
        temperature=0.2,
        extra_body={
            "data_sources": [
                {
                    "type": "azure_search",
                    "parameters": {
                        "endpoint": SEARCH_ENDPOINT,
                        "index_name": SEARCH_INDEX,
                        "authentication": {"type": "api_key", "key": SEARCH_KEY},

                        # En iyi pratik: hybrid + vector
                        "query_type": "vector_simple_hybrid",
                        "embedding_dependency": {
                            "type": "deployment_name",
                            "deployment_name": EMB_DEPLOYMENT,
                        },

                        # Index mapping
                        "fields_mapping": {
                            "content_fields": [CONTENT_FIELD],
                            "title_field": TITLE_FIELD,
                            "vector_fields": [VECTOR_FIELD],
                        },

                        # Retrieval ayarları
                        "top_n_documents": 5,
                        "strictness": 3,
                        "in_scope": True,
                    }
                }
            ]
        },
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
