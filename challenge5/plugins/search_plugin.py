import os
from typing import List, Dict
from semantic_kernel.functions import kernel_function

from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery


class SearchPlugin:
    def __init__(self):
        # Azure OpenAI client (for embeddings)
        self.aoai = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        self.emb_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

        # Azure Search client
        self.search = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=os.getenv("AZURE_SEARCH_INDEX"),
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY")),
        )

        # Field mappings
        self.content_field = os.getenv("AZURE_SEARCH_CONTENT_FIELD", "content")
        self.title_field = os.getenv("AZURE_SEARCH_TITLE_FIELD", "title")
        self.vector_field = os.getenv("AZURE_SEARCH_VECTOR_FIELD")

    def _embed(self, text: str) -> List[float]:
        if not self.emb_deployment:
            raise ValueError("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT is missing in .env")

        resp = self.aoai.embeddings.create(
            input=text,
            model=self.emb_deployment,
        )
        return resp.data[0].embedding

    def _retrieve(self, query: str, k: int = 5) -> List[Dict]:
        # Hybrid search if vector field exists, otherwise keyword search
        if self.vector_field:
            emb = self._embed(query)

            results = self.search.search(
                search_text=query,
                vector_queries=[
                    VectorizedQuery(
                        vector=emb,
                        k_nearest_neighbors=k,
                        fields=self.vector_field,
                    )
                ],
                select=[self.content_field, self.title_field],
                top=k,
            )
        else:
            results = self.search.search(
                search_text=query,
                select=[self.content_field, self.title_field],
                top=k,
            )

        docs = []
        for r in results:
            title = str(r.get(self.title_field, "")).strip()
            content = str(r.get(self.content_field, "")).strip()

            if len(content) > 1200:
                content = content[:1200] + "..."

            docs.append({"title": title, "content": content})

        return docs

    @kernel_function(
        name="search_docs",
        description="Hybrid/vector search on Azure AI Search. Returns SOURCES with titles + passages."
    )
    def search_docs(self, query: str, top_k: int = 5) -> str:
        docs = self._retrieve(query, k=top_k)

        if not docs:
            return "No results."

        chunks = []
        for i, d in enumerate(docs):
            chunks.append(f"[{i+1}] {d['title']}\n{d['content']}")

        return "SOURCES:\n" + "\n\n".join(chunks)
