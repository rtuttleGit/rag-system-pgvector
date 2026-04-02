"""Document processor for extracting and chunking documents."""

from typing import List, Dict, Any
import tiktoken
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

from .config import EMBEDDING_MODEL, MAX_TOKENS_PER_CHUNK


class DocumentProcessor:
    def __init__(self):
        # Set up tokenizer for chunking
        tiktoken_encoder = tiktoken.encoding_for_model(EMBEDDING_MODEL)
        self.tokenizer = OpenAITokenizer(
            tokenizer=tiktoken_encoder, max_tokens=MAX_TOKENS_PER_CHUNK
        )
        self.chunker = HybridChunker(tokenizer=self.tokenizer)
        self.converter = DocumentConverter()

    def process_document(self, source: str) -> List[Dict[str, Any]]:
        """
        Process a document from URL or file path.
        Returns list of chunks with metadata.
        """
        # Convert document
        print(f"Extracting document from: {source}")
        doc = self.converter.convert(source).document

        # Create chunks
        print("Creating chunks...")
        chunks = list(self.chunker.chunk(dl_doc=doc))

        # Process chunks with contextualization
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Get contextualized text (includes headings/context)
            contextualized_text = self.chunker.contextualize(chunk=chunk)

            # Extract page numbers from chunk metadata
            page_numbers = sorted(
                set(
                    prov.page_no
                    for item in chunk.meta.doc_items
                    for prov in item.prov
                    if hasattr(prov, "page_no")
                )
            )

            # Extract headings from chunk metadata
            headings = chunk.meta.headings if hasattr(chunk.meta, "headings") else []

            # Create metadata
            metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": source,
                "page_numbers": page_numbers,
                "headings": headings,
            }

            processed_chunks.append(
                {"content": contextualized_text, "metadata": metadata}
            )

        print(f"Created {len(processed_chunks)} chunks")
        return processed_chunks

    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the processed chunks."""
        total_tokens = 0

        for chunk in chunks:
            tokens = self.tokenizer.tokenizer.encode(chunk["content"])
            total_tokens += len(tokens)

        return {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": total_tokens / len(chunks) if chunks else 0,
        }
