from constants import *
from pydantic import BaseModel
from typing import Optional, List, Literal
from datetime import datetime
import requests, faiss
import numpy as np
from logging import Logger


class MemoryItem(BaseModel):
    text: str
    type: Literal["preference", "tool_output", "fact", "query", "system"] = "fact"
    timestamp: Optional[str] = datetime.now().isoformat()
    tool_name: Optional[str] = None
    user_query: Optional[str] = None
    tags: List[str] = []
    session_id: Optional[str] = None


class Memory:
    """
    The Memory class manages the state and history of the agent's execution.
    It forms the second step in the Perception -> Memory -> Decision -> Action framework.

    The Memory class:
    1. Stores user preferences and session-specific information
    2. Maintains a history of tool executions and their results
    3. Provides methods to store and recall information
    4. Supports the decision-making process with historical context

    Attributes:
        embedding_model_url (str): URL of the embedding model service
        model_name (str): Name of the embedding model to use
        index (faiss.Index): FAISS index for similarity search
        data (List[MemoryItem]): List of stored memory items
        embeddings (List[np.ndarray]): List of embeddings for memory items
        logger (Logger): Logger instance for logging operations
    """

    def __init__(
        self,
        logger: Logger,
        embedding_model_url=OLLAMA_URL,
        model_name=EMBEDING_MODEL_NAME,
    ):
        """
        Initialize the Memory class with necessary dependencies.

        Args:
            logger (Logger): Logger instance for logging
            embedding_model_url (str): URL of the embedding model service
            model_name (str): Name of the embedding model to use
        """
        self.embedding_model_url = embedding_model_url
        self.model_name = model_name
        self.index = None
        self.data: List[MemoryItem] = []
        self.embeddings: List[np.ndarray] = []
        self.logger = logger

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get the embedding vector for a given text using the embedding model.

        Args:
            text (str): The text to embed

        Returns:
            np.ndarray: The embedding vector for the text

        Raises:
            requests.exceptions.RequestException: If the embedding request fails
        """
        response = requests.post(
            self.embedding_model_url, json={"model": self.model_name, "prompt": text}
        )
        response.raise_for_status()
        self.logger.info(f"Embedded the text: {text} using {self.model_name}")
        return np.array(response.json()["embedding"], dtype=np.float32)

    def add(self, item: MemoryItem):
        """
        Add a new memory item to the memory store.

        This method:
        1. Generates an embedding for the memory item
        2. Adds the embedding to the FAISS index
        3. Stores the memory item and its embedding

        Args:
            item (MemoryItem): The memory item to add
        """
        emb = self._get_embedding(item.text)
        self.embeddings.append(emb)
        self.data.append(item)

        # Initialize or add to index
        if self.index is None:
            self.index = faiss.IndexFlatL2(len(emb))
        self.index.add(np.stack([emb]))
        self.logger.info(
            f"Added embedding for {item.text} to the index using {self.model_name}"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        type_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
        session_filter: Optional[str] = None,
    ) -> List[MemoryItem]:
        """
        Retrieve relevant memory items based on a query.

        This method:
        1. Generates an embedding for the query
        2. Searches the FAISS index for similar items
        3. Filters results based on type, tags, and session
        4. Returns the top-k most relevant items

        Args:
            query (str): The query to search for
            top_k (int): Number of items to retrieve
            type_filter (Optional[str]): Filter by memory item type
            tag_filter (Optional[List[str]]): Filter by memory item tags
            session_filter (Optional[str]): Filter by session ID

        Returns:
            List[MemoryItem]: List of relevant memory items
        """
        if not self.index or len(self.data) == 0:
            return []

        query_vec = self._get_embedding(query).reshape(1, -1)
        D, I = self.index.search(query_vec, top_k * 2)  # Overfetch to allow filtering

        results = []
        for idx in I[0]:
            if idx >= len(self.data):
                continue
            item = self.data[idx]

            # Filter by type
            if type_filter and item.type != type_filter:
                continue

            # Filter by tags
            if tag_filter and not any(tag in item.tags for tag in tag_filter):
                continue

            # Filter by session
            if session_filter and item.session_id != session_filter:
                continue

            results.append(item)
            if len(results) >= top_k:
                break
        self.logger.info(
            f"Retrieved {len(results)} relevant memories for the query {query}"
        )

        return results

    def bulk_add(self, items: List[MemoryItem]):
        """
        Add multiple memory items in bulk.

        Args:
            items (List[MemoryItem]): List of memory items to add
        """
        for item in items:
            self.add(item)

        self.logger.info(f"Bulk added {len(items)} to the faiss index")
