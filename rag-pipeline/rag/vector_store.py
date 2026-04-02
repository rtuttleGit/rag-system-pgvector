"""Vector store implementation using PGVector."""

import json
import os
import psycopg
from typing import Any, Dict, List, Optional, Literal
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from openai import OpenAI
from dotenv import load_dotenv
from .config import DATABASE_CONFIG, EMBEDDING_DIMENSIONS

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class VectorStore:
    DEFAULT_LIMIT = 10
    DEFAULT_RRF_K = 50
    EMBEDDING_DIMENSIONS = 1536
    SEARCH_CONFIG = "english"
    MAX_SEARCH_RESULTS = 30

    def __init__(self):
        self.conn = None
        self.connect()
        self.setup_database()

    def connect(self):
        """Establish connection to PostgreSQL database."""
        # Build connection string for psycopg3
        conn_str = (
            f"host={DATABASE_CONFIG['host']} "
            f"port={DATABASE_CONFIG['port']} "
            f"dbname={DATABASE_CONFIG['database']} "
            f"user={DATABASE_CONFIG['user']} "
            f"password={DATABASE_CONFIG['password']}"
        )
        self.conn = psycopg.connect(conn_str)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)

    def setup_database(self):
        """Create the necessary tables and extensions."""
        with self.conn.cursor() as cur:
            # Create documents table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector({EMBEDDING_DIMENSIONS}),
                    metadata JSONB DEFAULT '{{}}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create index for similarity search using HNSW with inner product
            cur.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                ON documents USING hnsw (embedding vector_ip_ops)
            """)

            self.conn.commit()

    def get_embedding(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI."""
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            dimensions=self.EMBEDDING_DIMENSIONS,
        )
        return response.data[0].embedding

    def _build_metadata_filter(
        self, metadata_filter: dict = None
    ) -> tuple[str, str, dict]:
        """Build metadata filter using PostgreSQL's JSON containment operator."""
        if not metadata_filter:
            return "", "", {}

        where_clause = " WHERE metadata::jsonb @> %(metadata_filter)s::jsonb"
        and_clause = " AND metadata::jsonb @> %(metadata_filter)s::jsonb"
        params = {"metadata_filter": json.dumps(metadata_filter)}

        return where_clause, and_clause, params

    def add_document(
        self,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a single document to the vector store."""
        with self.conn.cursor() as cur:
            result = cur.execute(
                """
                INSERT INTO documents (content, embedding, metadata)
                VALUES (%s, %s, %s)
                RETURNING id
            """,
                (content, embedding, json.dumps(metadata or {})),
            )
            doc_id = result.fetchone()[0]
            self.conn.commit()
            return doc_id

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add multiple documents to the vector store in batch."""
        with self.conn.cursor() as cur:
            for doc in documents:
                cur.execute(
                    """
                    INSERT INTO documents (content, embedding, metadata)
                    VALUES (%s, %s, %s)
                """,
                    (
                        doc["content"],
                        doc["embedding"],
                        json.dumps(doc.get("metadata", {})),
                    ),
                )
            self.conn.commit()

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Perform similarity search using inner product."""
        with self.conn.cursor(row_factory=dict_row) as cur:
            # Base query with similarity score using inner product
            query = """
                SELECT 
                    id,
                    content,
                    metadata,
                    -(embedding <#> %s::vector) as similarity,
                    created_at
                FROM documents
            """

            # Add threshold filter if specified
            if threshold is not None:
                query += f" WHERE -(embedding <#> %s::vector) >= {threshold}"

            # Order by similarity and limit (using inner product)
            query += " ORDER BY embedding <#> %s::vector LIMIT %s"

            # Execute query
            if threshold is not None:
                result = cur.execute(
                    query, (query_embedding, query_embedding, query_embedding, k)
                )
            else:
                result = cur.execute(query, (query_embedding, query_embedding, k))

            results = result.fetchall()

            # Convert to list of dicts with proper formatting
            return [
                {
                    "id": r["id"],
                    "content": r["content"],
                    "metadata": r["metadata"],
                    "similarity": float(r["similarity"]),
                    "created_at": r["created_at"].isoformat()
                    if r["created_at"]
                    else None,
                }
                for r in results
            ]

    def hybrid_search(
        self,
        query: str,
        semantic_weight: float = 1.0,
        full_text_weight: float = 1.0,
        keyword_mode: Literal["strict", "flexible"] = "flexible",
        rrf_k: int = DEFAULT_RRF_K,
        limit: int = DEFAULT_LIMIT,
        metadata_filter: dict = None,
    ) -> List[dict]:
        """
        Perform hybrid search combining semantic and keyword search.

        Args:
            query: Search query text
            semantic_weight: Weight for semantic search (0.0 to disable)
            full_text_weight: Weight for keyword search (0.0 to disable)
            keyword_mode: 'strict' (all terms must match) or 'flexible' (any terms can match)
            rrf_k: Smoothing constant for Reciprocal Rank Fusion
            limit: Maximum number of results to return
            metadata_filter: Optional dictionary of metadata key-value pairs to filter by

        Returns:
            List of search results with scores and metadata
        """

        # Use AND logic (all terms must match)
        if keyword_mode == "strict":
            tsquery_func = (
                f"websearch_to_tsquery('{self.SEARCH_CONFIG}', %(fts_query)s)"
            )
            fts_query = query

        # Use OR logic (any terms can match)
        elif keyword_mode == "flexible":
            tsquery_func = f"to_tsquery('{self.SEARCH_CONFIG}', %(fts_query)s)"
            # Convert websearch AND query to OR query inline
            with self.conn.cursor() as cur:
                cur.execute(
                    f"SELECT websearch_to_tsquery('{self.SEARCH_CONFIG}', %s)::text",
                    (query,),
                )
                result = cur.fetchone()[0]
                fts_query = (
                    result.replace(" & ", " | ") if result and result != "''" else "''"
                )
        else:
            raise ValueError("keyword_mode must be 'strict' or 'flexible'")

        query_embedding = self.get_embedding(query)
        metadata_where, metadata_and, metadata_params = self._build_metadata_filter(
            metadata_filter
        )

        with self.conn.cursor() as cur:
            sql_query = f"""
                WITH full_text AS (
                    SELECT 
                        id,
                        ts_rank_cd(fts, {tsquery_func}) as fts_score,
                        row_number() OVER (
                            ORDER BY 
                                ts_rank_cd(fts, {tsquery_func}) DESC,
                                length(content)
                        ) as rank_ix
                    FROM documents
                    WHERE fts @@ {tsquery_func}{metadata_and}
                    ORDER BY rank_ix
                    LIMIT least(%(limit)s, %(max_results)s) * 2
                ),
                semantic AS (
                    SELECT 
                        id,
                        row_number() OVER (ORDER BY embedding <#> %(query_embedding)s::vector) as rank_ix
                    FROM documents{metadata_where}
                    ORDER BY rank_ix
                    LIMIT least(%(limit)s, %(max_results)s) * 2
                )
                SELECT 
                    d.*,
                    ft.fts_score,
                    COALESCE(1.0 / (%(rrf_k)s + ft.rank_ix), 0.0) * %(full_text_weight)s as full_text_score,
                    COALESCE(1.0 / (%(rrf_k)s + s.rank_ix), 0.0) * %(semantic_weight)s as semantic_score,
                    (
                        COALESCE(1.0 / (%(rrf_k)s + ft.rank_ix), 0.0) * %(full_text_weight)s + 
                        COALESCE(1.0 / (%(rrf_k)s + s.rank_ix), 0.0) * %(semantic_weight)s
                    ) as combined_score
                FROM 
                    full_text ft
                    FULL OUTER JOIN semantic s ON ft.id = s.id
                    JOIN documents d ON COALESCE(ft.id, s.id) = d.id{metadata_where}
                ORDER BY combined_score DESC
                LIMIT least(%(limit)s, %(max_results)s)
            """

            params = {
                "fts_query": fts_query,
                "query_embedding": query_embedding,
                "limit": limit,
                "max_results": self.MAX_SEARCH_RESULTS,
                "rrf_k": rrf_k,
                "full_text_weight": full_text_weight,
                "semantic_weight": semantic_weight,
                **metadata_params,
            }

            cur.execute(sql_query, params)

            results = cur.fetchall()

            return [
                {
                    "id": r[0],
                    "content": r[1],
                    "metadata": r[2],
                    "embedding": r[3],
                    "fts": r[4],
                    "created_at": r[5],
                    "fts_raw_score": float(r[6]) if r[6] else 0.0,
                    "full_text_score": float(r[7]),
                    "semantic_score": float(r[8]),
                    "combined_score": float(r[9]),
                }
                for r in results
            ]

    def get_document_count(self) -> int:
        """Get the total number of documents in the store."""
        with self.conn.cursor() as cur:
            result = cur.execute("SELECT COUNT(*) FROM documents")
            return result.fetchone()[0]

    def clear_all_documents(self):
        """Delete all documents from the store."""
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE documents RESTART IDENTITY")
            self.conn.commit()

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
