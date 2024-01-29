import asyncio
from typing import List

import httpx
from ragas.embeddings.base import BaseRagasEmbeddings

class GatewayEmbeddings(BaseRagasEmbeddings):
    def __init__(
            self,
            address: str = "http://172.16.23.86:30392/batch_embedding",
            embedding_type: str = "bge_zh",
            instruction: str = "",
            normalize_embeddings: bool = False,
    ):
        self.address = address
        self.embedding_type = embedding_type
        self.instruction = instruction
        self.normalize_embeddings = normalize_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        with httpx.Client(verify=False) as client:
            try:
                response = client.post(
                    url=self.address,
                    json={
                        "queries": texts,
                        "embedding_type": self.embedding_type,
                        "instruction": self.instruction,
                        "normalize_embeddings": self.normalize_embeddings,
                    }
                )
                return response.json()
            except Exception as e:
                return []

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[-1]


async def main():
    embeddings = GatewayEmbeddings()
    embedding_result = await embeddings.embed_texts(["你好"], is_async=False)
    print(embedding_result)


if __name__ == '__main__':
    asyncio.run(main())
