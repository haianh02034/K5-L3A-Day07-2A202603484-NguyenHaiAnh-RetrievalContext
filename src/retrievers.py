"""Retrieval methods for Day 7 Agentic RAG:
1. DenseRetriever: Real ChromaDB vector store + sentence-transformers embeddings.
2. LexicalRetriever: TF-IDF retrieval over chunk text.
3. HybridRetriever: Reciprocal Rank Fusion (RRF) combining Dense and Lexical.
"""
import math
import re
import unicodedata
from collections import Counter
from typing import Any
import chromadb
from sentence_transformers import SentenceTransformer


def tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


class DenseRetriever:
    """Method 1: Dense Retrieval using ChromaDB and MiniLM-L12-v2 embeddings."""
    def __init__(self, chroma_path: str = r'E:\AI_Vin\K5-L3A-Day06-2A202603484-DataIngestionEngineering\chroma_db',
                 collection_name: str = 'strategy_b',
                 model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.model_name = model_name
        print(f"[DenseRetriever] Initializing ChromaDB client from {chroma_path}...")
        self.client = chromadb.PersistentClient(chroma_path)
        self.collection = self.client.get_collection(collection_name)
        print(f"[DenseRetriever] Loading SentenceTransformer '{model_name}'...")
        self.embedder = SentenceTransformer(model_name)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        query_vec = self.embedder.encode([query], normalize_embeddings=True).tolist()
        where_filter = None
        if filters:
            # Chroma where filter
            if len(filters) == 1:
                k, v = next(iter(filters.items()))
                where_filter = {k: v}
            elif len(filters) > 1:
                where_filter = {"$and": [{k: v} for k, v in filters.items()]}

        res = self.collection.query(
            query_embeddings=query_vec,
            n_results=top_k,
            where=where_filter
        )

        results = []
        if res and res['ids'] and res['ids'][0]:
            ids = res['ids'][0]
            docs = res['documents'][0]
            metas = res['metadatas'][0]
            dists = res['distances'][0] if 'distances' in res and res['distances'] else [0.0]*len(ids)
            for cid, doc, meta, dist in zip(ids, docs, metas, dists):
                meta_dict = dict(meta) if meta else {}
                score = round(1.0 - float(dist), 4) if dist is not None else 1.0
                results.append({
                    'id': cid,
                    'text': doc,
                    'source_id': meta_dict.get('document_id', 'handbook-sv-2025'),
                    'source': meta_dict.get('source', 'data/handbook_sv_2025.pdf').replace('\\', '/'),
                    'page': int(meta_dict.get('page', 0)),
                    'metadata': meta_dict,
                    'score': score,
                    'strategy': 'dense'
                })
        return results


class LexicalRetriever:
    """Method 2: Lexical Retrieval using TF-IDF."""
    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        self.chunk_by_id = {c['id']: c for c in chunks}
        counts = [Counter(tokenize(c['text'])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {term: math.log((1 + len(chunks)) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts: Counter) -> dict[str, float]:
        vector = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1
        return {term: value / norm for term, value in vector.items()}

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        query_vec = self._vector(Counter(tokenize(query)))
        results = []
        for chunk, vec in zip(self.chunks, self.vectors):
            if filters:
                match = True
                for fk, fv in filters.items():
                    if chunk['metadata'].get(fk) != fv:
                        match = False
                        break
                if not match:
                    continue
            score = sum(value * vec.get(term, 0) for term, value in query_vec.items())
            if score > 0:
                results.append({
                    **chunk,
                    'score': round(score, 4),
                    'strategy': 'tfidf'
                })
        return sorted(results, key=lambda c: (-c['score'], c['id']))[:top_k]


class HybridRetriever:
    """Method 3: Hybrid Retrieval using Reciprocal Rank Fusion (RRF)."""
    def __init__(self, dense_retriever: DenseRetriever, lexical_retriever: LexicalRetriever, rrf_k: int = 60):
        self.dense = dense_retriever
        self.lexical = lexical_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3, candidate_k: int = 10) -> list[dict[str, Any]]:
        # Retrieve candidates from both systems
        dense_results = self.dense.search(query, filters=filters, top_k=candidate_k)
        lexical_results = self.lexical.search(query, filters=filters, top_k=candidate_k)

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(dense_results):
            cid = item['id']
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, item in enumerate(lexical_results):
            cid = item['id']
            if cid not in chunk_map:
                chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: (-rrf_scores[cid], cid))[:top_k]
        fused = []
        for cid in sorted_ids:
            chunk = dict(chunk_map[cid])
            chunk['score'] = round(rrf_scores[cid], 5)
            chunk['strategy'] = 'hybrid_rrf'
            fused.append(chunk)
        return fused
