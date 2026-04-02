"""Interactive script to query the RAG system."""

import sys
from rag import RAGSystem


def print_separator():
    print("\n" + "=" * 60 + "\n")


def main():
    print("=== Docling RAG Query System ===\n")

    # Initialize RAG system
    print("Initializing RAG system...")
    is_hybrid = True

    try:
        rag = RAGSystem()
    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")
        sys.exit(1)

    # Check if database has documents
    doc_count = rag.vector_store.get_document_count()
    if doc_count == 0:
        print("No documents found in database. Please run build_vectordb.py first.")
        sys.exit(1)

    print(f"✅ RAG system ready! ({doc_count} documents in database)")

    # ✅ RESTORED: Example questions + instructions
    print("\nExample questions you can ask:")
    print("- What is Docling and what problems does it solve?")
    print("- How does Docling handle PDF parsing?")
    print("- What are the key features of the HybridChunker?")
    print("- Explain Docling's architecture")
    print("- What document formats does Docling support?")
    print("\nType 'quit' or 'exit' to stop, 'stats' for system statistics.")

    # Interactive query loop
    while True:
        print_separator()
        query = input("Your question: ").strip()

        if query.lower() in ["quit", "exit"]:
            print("\nGoodbye! 👋")
            break

        # ✅ RESTORED: stats command
        if query.lower() == "stats":
            stats = rag.get_stats()
            print("\nSystem Statistics:")
            print(f"- Total documents: {stats['total_documents']}")
            print(f"- Embedding model: {stats['embedding_model']}")
            print(f"- Embedding dimensions: {stats['embedding_dimensions']}")
            print(f"- Chat model: {stats['chat_model']}")
            continue

        if not query:
            print("Please enter a question.")
            continue

        try:
            # Query the system
            print("\nProcessing your query...")

            # ✅ FIX: get BOTH response + context (no re-query)
            response, context = rag.query(
                query, show_context=False, is_hybrid=is_hybrid
            )

            print("\nResponse:")
            print(response)

            # Ask if user wants to see the retrieved context
            show_context = input("\nShow retrieved context? (y/n): ").strip().lower()

            if show_context == "y":
                print("\n📚 Retrieved Context:")

                for i, doc in enumerate(context, 1):
                    if is_hybrid:
                        print(
                            f"\n{i}. Chunk {doc['metadata']['chunk_index'] + 1}/"
                            f"{doc['metadata']['total_chunks']} "
                            f"(Combined: {doc['combined_score']:.3f})"
                        )
                    else:
                        print(
                            f"\n{i}. Chunk {doc['metadata']['chunk_index'] + 1}/"
                            f"{doc['metadata']['total_chunks']} "
                            f"(Similarity: {doc['similarity']:.3f})"
                        )

                    print(f"   Preview: {doc['content'][:200]}...")

        except Exception as e:
            print(f"\nError processing query: {str(e)}")


if __name__ == "__main__":
    main()
