import os
import json
import uuid
import time
from typing import List, Dict

from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError

DATA_PATH = "data/secret_kb.jsonl"
PK_VALUE = "docs"
EMBED_FIELD = "embedding"
BATCH_SIZE = 8
MAX_RETRIES = 5
RETRY_BASE = 1.5


def require_env(keys: List[str]):
    """Validate that all required environment variables are set.
    
    Args:
        keys: List of environment variable names to check
        
    Raises:
        ValueError: If any required variables are missing
    """
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise ValueError(f"Missing env vars: {missing}")


def chunked(lst: List[Dict], n: int):
    """Split a list into batches of size n.
    
    Args:
        lst: List to split into chunks
        n: Batch size
        
    Yields:
        Batches of the original list
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def retry_with_backoff(func, max_retries=MAX_RETRIES, base=RETRY_BASE):
    """Execute function with exponential backoff retry logic.
    
    Args:
        func: Callable function to execute with retry logic
        max_retries: Maximum number of retry attempts
        base: Base for exponential backoff calculation
        
    Returns:
        Result of the function call if successful
        
    Raises:
        RuntimeError: If all retry attempts fail
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed after {max_retries} retries: {e}")
            wait = base ** attempt
            print(f"Error: {e} | retry {attempt + 1}/{max_retries} in {wait:.1f}s")
            time.sleep(wait)


def main():
    """Load documents from JSONL file, generate embeddings, and ingest into Cosmos DB."""
    load_dotenv()

    require_env([
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
        "COSMOS_ENDPOINT",
        "COSMOS_KEY",
        "COSMOS_DATABASE",
        "COSMOS_CONTAINER",
    ])

    aoai = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    EMB_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

    cosmos = CosmosClient(os.getenv("COSMOS_ENDPOINT"), credential=os.getenv("COSMOS_KEY"))
    db = cosmos.get_database_client(os.getenv("COSMOS_DATABASE"))
    container = db.get_container_client(os.getenv("COSMOS_CONTAINER"))

    docs: List[Dict] = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))

    print(f"Loaded {len(docs)} docs from {DATA_PATH}")

    inserted = 0

    for batch in chunked(docs, BATCH_SIZE):
        texts = [f"{d.get('title','')}\n\n{d.get('content','')}" for d in batch]

        # Generate embeddings for batch with retry logic
        embeddings = retry_with_backoff(
            lambda: [x.embedding for x in aoai.embeddings.create(model=EMB_DEPLOYMENT, input=texts).data]
        )

        # Insert documents with embeddings into Cosmos DB
        for d, emb in zip(batch, embeddings):
            item = {
                "id": str(uuid.uuid4()),
                "pk": PK_VALUE,
                "title": d.get("title", ""),
                "content": d.get("content", ""),
                EMBED_FIELD: emb,
            }

            retry_with_backoff(lambda: container.upsert_item(item))
            inserted += 1

        print(f"Progress: {inserted}/{len(docs)} inserted")
        time.sleep(0.2)

    print(f"\nDONE! Inserted {inserted} items into Cosmos.")

if __name__ == "__main__":
    main()
