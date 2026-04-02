"""Main RAG system that orchestrates the entire pipeline."""

from typing import Any, Dict, List, Tuple

from openai import OpenAI

from .config import CHAT_MODEL, OPENAI_API_KEY, SIMILARITY_THRESHOLD, TOP_K_RESULTS
from .document_processor import DocumentProcessor
from .embedding_service import EmbeddingService
from .vector_store import VectorStore


class RAGSystem:
    def __init__(self):
        self.vector_store = VectorStore()
        self.embedding_service = EmbeddingService()
        self.document_processor = DocumentProcessor()
        self.chat_client = OpenAI(api_key=OPENAI_API_KEY)

    def ingest_document(self, source: str):
        chunks = self.document_processor.process_document(source)
        chunks_with_embeddings = self.embedding_service.add_embeddings_to_chunks(chunks)

        print("Storing chunks in vector database...")
        self.vector_store.add_documents(chunks_with_embeddings)

        stats = self.document_processor.get_chunk_stats(chunks)
        print("\nIngestion complete!")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
        print(f"Total documents in store: {self.vector_store.get_document_count()}")

    def retrieve_context(
        self,
        query: str,
        k: int = None,
        threshold: float = None,
        is_hybrid: bool = False,
        metadata_filter: dict = None,
    ) -> List[Dict[str, Any]]:

        if is_hybrid:
            return self.vector_store.hybrid_search(
                query,
                semantic_weight=1.0,
                full_text_weight=1.0,
                keyword_mode="flexible",
                limit=k or TOP_K_RESULTS,
                metadata_filter=metadata_filter,
            )
        else:
            query_embedding = self.embedding_service.create_embedding(query)

            return self.vector_store.similarity_search(
                query_embedding=query_embedding,
                k=k or TOP_K_RESULTS,
                threshold=threshold or SIMILARITY_THRESHOLD,
            )

    def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        is_hybrid: bool = False,
    ) -> str:

        context_texts = []
        for i, doc in enumerate(context, 1):
            if is_hybrid:
                context_texts.append(
                    f"[Document {i}] "
                    f"(FTS: {doc['full_text_score']:.3f}, "
                    f"Semantic: {doc['semantic_score']:.3f}, "
                    f"Combined: {doc['combined_score']:.3f})\n"
                    f"{doc['content']}"
                )
            else:
                context_texts.append(
                    f"[Document {i}] (Similarity: {doc['similarity']:.3f})\n{doc['content']}"
                )

        combined_context = "\n\n---\n\n".join(context_texts)

        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
        Use the context to answer the user's question accurately and comprehensively. 
        If the context doesn't contain enough information to fully answer the question, say so.
        Always cite which document(s) you're using for your answer."""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{combined_context}\n\nQuestion: {query}",
            },
        ]

        response = self.chat_client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, temperature=0.7, max_tokens=1000
        )

        return response.choices[0].message.content

    def query(
        self,
        question: str,
        k: int = None,
        threshold: float = None,
        show_context: bool = False,
        is_hybrid: bool = False,
        metadata_filter: dict = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:

        print("Searching for relevant context...")
        context = self.retrieve_context(
            question,
            k=k,
            threshold=threshold,
            is_hybrid=is_hybrid,
            metadata_filter=metadata_filter,
        )

        if not context:
            return (
                "I couldn't find any relevant information to answer your question.",
                [],
            )

        print(f"Found {len(context)} relevant chunks")

        print("Generating response...")
        response = self.generate_response(question, context, is_hybrid=is_hybrid)

        return response, context

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": self.vector_store.get_document_count(),
            "embedding_model": self.embedding_service.model,
            "embedding_dimensions": self.embedding_service.dimensions,
            "chat_model": CHAT_MODEL,
        }
