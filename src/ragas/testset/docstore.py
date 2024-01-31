from __future__ import annotations

import heapq
import logging
import typing as t
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt
from langchain.text_splitter import TextSplitter
from langchain_core.documents import Document as LCDocument
from langchain_core.pydantic_v1 import Field

from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.executor import Executor
from ragas.llms.base import BaseRagasLLM
from ragas.llms.json_load import json_loader
from ragas.testset.prompts import keyphrase_extraction_prompt
from ragas.testset.utils import rng

if t.TYPE_CHECKING:
    from llama_index.readers.schema import Document as LlamaindexDocument

Embedding = t.Union[t.List[float], npt.NDArray[np.float64]]
logger = logging.getLogger(__name__)


class Document(LCDocument):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: t.Optional[str] = None
    embedding: t.Optional[t.List[float]] = Field(default=None, repr=False)

    @classmethod
    def from_langchain_document(cls, doc: LCDocument):
        doc_id = str(uuid.uuid4())
        if doc.metadata.get("filename"):
            filename = doc.metadata["filename"]
        else:
            logger.info(
                "Document [ID: %s] has no filename. Using doc_id as filename.", doc_id
            )
            filename = doc_id
        return cls(
            page_content=doc.page_content,
            metadata=doc.metadata,
            doc_id=doc_id,
            filename=filename,
        )

    @classmethod
    def from_llamaindex_document(cls, doc: LlamaindexDocument):
        doc_id = str(uuid.uuid4())
        if doc.metadata.get("filename"):
            filename = doc.metadata["filename"]
        else:
            logger.info(
                "Document [ID: %s] has no filename. Using doc_id as filename.", doc_id
            )
            filename = doc_id
        return cls(
            page_content=doc.text,
            metadata=doc.metadata,
            doc_id=doc_id,
            filename=filename,
        )


class Node(Document):
    keyphrases: t.List[str] = Field(default_factory=list, repr=False)


class Direction(str, Enum):
    """
    Direction for getting adjascent nodes.
    """

    NEXT = "next"
    PREV = "prev"
    UP = "up"
    DOWN = "down"


class DocumentStore(ABC):
    def __init__(self):
        self.documents = {}

    @abstractmethod
    def add_documents(self, docs: t.Sequence[Document], show_progress=True):
        ...

    @abstractmethod
    def add_nodes(self, nodes: t.Sequence[Node], show_progress=True):
        ...

    @abstractmethod
    def get_node(self, node_id: str) -> Node:
        ...

    @abstractmethod
    def get_random_nodes(self, k=1) -> t.List[Node]:
        ...

    @abstractmethod
    def get_similar(
        self, node: Node, threshold: float = 0.7, top_k: int = 3
    ) -> t.Union[t.List[Document], t.List[Node]]:
        ...

    @abstractmethod
    def get_adjacent(
        self, node: Node, direction: Direction = Direction.NEXT
    ) -> t.Optional[Node]:
        ...


class SimilarityMode(str, Enum):
    """Modes for similarity/distance."""

    DEFAULT = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


def similarity(
    embedding1: Embedding,
    embedding2: Embedding,
    mode: SimilarityMode = SimilarityMode.DEFAULT,
) -> float:
    """Get embedding similarity."""
    if mode == SimilarityMode.EUCLIDEAN:
        # Using -euclidean distance as similarity to achieve same ranking order
        return -float(np.linalg.norm(np.array(embedding1) - np.array(embedding2)))
    elif mode == SimilarityMode.DOT_PRODUCT:
        return np.dot(embedding1, embedding2)
    else:
        product = np.dot(embedding1, embedding2)
        norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        return product / norm


default_similarity_fns = similarity


def get_top_k_embeddings(
    query_embedding: Embedding,
    embeddings: t.List[Embedding],
    similarity_fn: t.Optional[t.Callable[..., float]] = None,
    similarity_top_k: t.Optional[int] = None,
    embedding_ids: t.Optional[t.List] = None,
    similarity_cutoff: t.Optional[float] = None,
) -> t.Tuple[t.List[float], t.List]:
    """
    Get top nodes by similarity to the query.
    returns the scores and the embedding_ids of the nodes
    """
    if embedding_ids is None:
        embedding_ids = list(range(len(embeddings)))

    similarity_fn = similarity_fn or default_similarity_fns

    embeddings_np = np.array(embeddings)
    query_embedding_np = np.array(query_embedding)

    similarity_heap: t.List[t.Tuple[float, t.Any]] = []
    for i, emb in enumerate(embeddings_np):
        similarity = similarity_fn(query_embedding_np, emb)
        if similarity_cutoff is None or similarity > similarity_cutoff:
            heapq.heappush(similarity_heap, (similarity, embedding_ids[i]))
            if similarity_top_k and len(similarity_heap) > similarity_top_k:
                heapq.heappop(similarity_heap)
    result_tups = sorted(similarity_heap, key=lambda x: x[0], reverse=True)

    result_similarities = [s for s, _ in result_tups]
    result_ids = [n for _, n in result_tups]

    return result_similarities, result_ids


@dataclass
class InMemoryDocumentStore(DocumentStore):
    splitter: TextSplitter
    llm: t.Optional[BaseRagasLLM] = field(default=None, repr=False)
    embeddings: t.Optional[BaseRagasEmbeddings] = field(default=None, repr=False)
    nodes: t.List[Node] = field(default_factory=list)
    node_embeddings_list: t.List[Embedding] = field(default_factory=list)
    node_map: t.Dict[str, Node] = field(default_factory=dict)

    def _embed_items(self, items: t.Union[t.Sequence[Document], t.Sequence[Node]]):
        ...

    def add_documents(self, docs: t.Sequence[Document], show_progress=True):
        """
        Add documents in batch mode.
        """
        assert self.embeddings is not None, "Embeddings must be set"

        # split documents with self.splitter into smaller nodes
        nodes = [
            Node.from_langchain_document(d)
            for d in self.splitter.transform_documents(docs)
        ]

        self.add_nodes(nodes, show_progress=show_progress)

    def add_nodes(
        self, nodes: t.Sequence[Node], show_progress=True, desc: str = "embedding nodes"
    ):
        assert self.embeddings is not None, "Embeddings must be set"
        assert self.llm is not None, "LLM must be set"

        # NOTE: Adds everything in async mode for now.
        nodes_to_embed = {}
        nodes_to_extract = {}

        # get embeddings for the docs
        executor = Executor(
            desc="embedding nodes",
            keep_progress_bar=False,
            is_async=True,
            raise_exceptions=True,
        )
        result_idx = 0
        for i, n in enumerate(nodes):
            if n.embedding is None:
                nodes_to_embed.update({i: result_idx})
                executor.submit(
                    self.embeddings.aembed_query,
                    n.page_content,
                    name=f"embed_node_task[{i}]",
                )
                result_idx += 1

            if n.keyphrases == []:
                nodes_to_extract.update({i: result_idx})
                executor.submit(
                    self.llm.agenerate_text,
                    keyphrase_extraction_prompt.format(text=n.page_content),
                    name=f"keyphrase-extraction[{i}]",
                )
                result_idx += 1

        results = executor.results()
        for i, n in enumerate(nodes):
            if i in nodes_to_embed.keys():
                n.embedding = results[nodes_to_embed[i]]
            if i in nodes_to_extract.keys():
                result = results[nodes_to_extract[i]]
                keyphrase_dict = json_loader._safe_load(
                    result.generations[0][0].text,
                    llm=self.llm,
                )
                n.keyphrases = keyphrase_dict.get("keyphrases", [])

            if n.embedding is not None and n.keyphrases != []:
                self.nodes.append(n)
                self.node_map[n.doc_id] = n
                assert isinstance(
                    n.embedding, (list, np.ndarray)
                ), "Embedding must be list or np.ndarray"
                self.node_embeddings_list.append(n.embedding)

    def get_node(self, node_id: str) -> Node:
        return self.node_map[node_id]

    def get_document(self, doc_id: str) -> Node:
        raise NotImplementedError

    def get_random_nodes(self, k=1) -> t.List[Node]:
        return rng.choice(np.array(self.nodes), size=k).tolist()

    def get_similar(
        self, node: Node, threshold: float = 0.7, top_k: int = 3
    ) -> t.Union[t.List[Document], t.List[Node]]:
        items = []
        doc = node
        if doc.embedding is None:
            raise ValueError("Document has no embedding.")
        scores, doc_ids = get_top_k_embeddings(
            query_embedding=doc.embedding,
            embeddings=self.node_embeddings_list,
            similarity_fn=similarity,
            similarity_cutoff=threshold,
            # we need to return k+1 docs here as the top result is the input doc itself
            similarity_top_k=top_k + 1,
        )
        # remove the query doc itself from results
        scores, doc_ids = scores[1:], doc_ids[1:]
        items = [self.nodes[doc_id] for doc_id in doc_ids]
        return items

    def get_adjacent(
        self, node: Node, direction: Direction = Direction.NEXT
    ) -> t.Optional[Node]:
        # linear search for doc_id of doc in documents_list
        index = self.nodes.index(node)

        if direction == Direction.NEXT:
            if len(self.nodes) > index + 1:
                next_doc = self.nodes[index + 1]
                if next_doc.filename == node.filename:
                    return next_doc
                else:
                    return None
            else:
                return None
        if direction == Direction.PREV:
            if index > 0:
                prev_doc = self.nodes[index - 1]
                if prev_doc.filename == node.filename:
                    return prev_doc
                else:
                    return None
            else:
                return None
