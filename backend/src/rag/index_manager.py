import os
import logging
import google.generativeai as genai
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class GeminiEmbeddings:
    """LangChain-compatible embeddings wrapper using models/gemini-embedding-001."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model_name = "models/gemini-embedding-001"
        genai.configure(api_key=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        res = genai.embed_content(model=self.model_name, content=texts)
        return res['embedding']

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        res = genai.embed_content(model=self.model_name, content=text)
        return res['embedding']

    def __call__(self, text: str) -> List[float]:
        """Make the object callable for FAISS compatibility."""
        return self.embed_query(text)


class IndexManager:
    """Manages the FAISS vector store for RAG."""

    def __init__(self, index_path: str = "index", api_key: Optional[str] = None):
        self.index_path = index_path
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.embeddings = GeminiEmbeddings(api_key=self.api_key)
        self.vector_store = self._load_or_create_index()
        logger.info("IndexManager initialized successfully")

    def _load_or_create_index(self) -> FAISS:
        """Load existing index from disk or create a new empty one."""
        faiss_path = os.path.join(self.index_path, "index.faiss")
        if os.path.exists(faiss_path):
            try:
                return FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}. Creating new one.")

        dummy = Document(
            page_content="Mechanical AI Assistant Documentation Index",
            metadata={"source": "system"},
        )
        return FAISS.from_documents([dummy], self.embeddings)

    def add_documents(self, documents: List[Document]):
        """Add documents to the index and persist to disk."""
        if not documents:
            return
        self.vector_store.add_documents(documents)
        self._save()

    def _save(self):
        os.makedirs(self.index_path, exist_ok=True)
        self.vector_store.save_local(self.index_path)

    def search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search."""
        return self.vector_store.similarity_search(query, k=k)

    def as_retriever(self, k: int = 4):
        """Return vector store as a LangChain retriever."""
        return self.vector_store.as_retriever(search_kwargs={"k": k})
