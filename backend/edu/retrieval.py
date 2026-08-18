"""
backend/edu/retrieval.py
────────────────────────────────────────────────────────────
Route-aware chunk retrieval for Edu-VQAGuider.

Retrieval strategy varies by route:
    concept   → top-3 by transcript similarity
    procedure → top-3 by transcript similarity
    temporal  → top-1 + temporal neighbors (before/after)
    visual    → top-3, re-ranked by CLIP question-frame similarity
    summary   → top-5 for broader coverage

Public API:
    build_chunk_index()     — embed all chunks, build FAISS index
    retrieve_chunks()       — route-aware retrieval
    select_best_frame()     — CLIP-based frame selection per chunk

Dependencies (deferred loading — safe to import without models):
    sentence-transformers   — for transcript embeddings
    faiss-cpu               — for similarity search
    CLIP                    — for frame re-ranking (optional at MVP)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from backend.edu.chunking import ChunkMeta
from backend.edu.prompts import ROUTE_RETRIEVAL_CONFIG
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk retrieved as evidence, with relevance metadata."""
    chunk: ChunkMeta
    transcript_score: float        # cosine sim from sentence-transformer
    clip_score: Optional[float]    # CLIP question-frame sim (if computed)
    combined_score: float          # final score after route-aware adjustments
    selected_frame: Optional[str]  # path to best frame (CLIP-selected)


# ══════════════════════════════════════════════════════════════
# Transcript Embedding Index
# ══════════════════════════════════════════════════════════════

class ChunkIndex:
    """
    In-memory index over chunk transcript embeddings.

    Uses numpy cosine similarity for MVP. Can be upgraded to FAISS
    for larger indices without changing the public API.

    Attributes:
        chunks:     List of ChunkMeta in index order
        embeddings: (N, embed_dim) numpy array of transcript embeddings
        embed_dim:  Dimensionality of the embeddings
    """

    def __init__(self):
        self.chunks: list[ChunkMeta] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embed_dim: int = 0
        self._is_built: bool = False

    @property
    def is_built(self) -> bool:
        return self._is_built

    def build(
        self,
        chunks: list[ChunkMeta],
        embeddings: np.ndarray,
    ) -> None:
        """
        Build the index from pre-computed embeddings.

        Args:
            chunks:     List of ChunkMeta objects (same order as embeddings)
            embeddings: (N, embed_dim) float32 array, L2-normalized
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Chunk count ({len(chunks)}) != embedding count ({embeddings.shape[0]})"
            )

        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)
        self.embed_dim = embeddings.shape[1]
        self._is_built = True

        logger.info(
            "ChunkIndex built: %d chunks, %d-dim embeddings",
            len(chunks), self.embed_dim,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[tuple[int, float]]:
        """
        Find the top-k most similar chunks to a query embedding.

        Args:
            query_embedding: (embed_dim,) float32 array, L2-normalized
            top_k:           Number of results to return

        Returns:
            List of (chunk_index, cosine_similarity) tuples, descending
        """
        if not self._is_built or self.embeddings is None:
            raise RuntimeError("ChunkIndex not built. Call build() first.")

        # Cosine similarity (embeddings are L2-normalized)
        query = query_embedding.astype(np.float32).reshape(1, -1)
        scores = (self.embeddings @ query.T).squeeze()  # (N,)

        # Top-k indices
        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:k]

        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        return results


# ══════════════════════════════════════════════════════════════
# Embedding Functions (lazy-loaded)
# ══════════════════════════════════════════════════════════════

_sentence_model = None


def _get_sentence_model():
    """Lazy-load sentence-transformers model."""
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded sentence-transformers/all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for retrieval. "
                "Install with: pip install sentence-transformers"
            )
    return _sentence_model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using sentence-transformers.

    Args:
        texts: List of strings to embed

    Returns:
        (N, 384) float32 array, L2-normalized
    """
    model = _get_sentence_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 20,
    )
    return embeddings.astype(np.float32)


def embed_query(question: str) -> np.ndarray:
    """Embed a single question string. Returns (384,) float32 array."""
    model = _get_sentence_model()
    embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.astype(np.float32)


# ══════════════════════════════════════════════════════════════
# Index Building
# ══════════════════════════════════════════════════════════════

def build_chunk_index(chunks: list[ChunkMeta]) -> ChunkIndex:
    """
    Build a searchable index from chunk transcripts and visual summaries.

    Embeds combined transcript + visual description texts using
    sentence-transformers and stores them in a ChunkIndex for cosine
    similarity search. This enables retrieval based on BOTH spoken
    content and visual content from lecture frames.

    Args:
        chunks: List of ChunkMeta with transcript_text (and optionally
                visual_summary) populated

    Returns:
        Built ChunkIndex ready for search()
    """
    texts = []
    for c in chunks:
        parts = []
        if c.transcript_text and c.transcript_text.strip():
            parts.append(c.transcript_text)
        if c.visual_summary and c.visual_summary.strip():
            parts.append(f"[Visual Content]: {c.visual_summary}")
        texts.append("\n\n".join(parts) if parts else "")

    # Warn about empty chunks
    empty_count = sum(1 for t in texts if not t.strip())
    if empty_count > 0:
        logger.warning(
            "%d / %d chunks have no transcript or visual content",
            empty_count, len(chunks),
        )

    visual_count = sum(1 for c in chunks if c.visual_summary and c.visual_summary.strip())
    logger.info(
        "Building index: %d chunks (%d with visual summaries)",
        len(chunks), visual_count,
    )

    embeddings = embed_texts(texts)

    index = ChunkIndex()
    index.build(chunks, embeddings)
    return index


# ══════════════════════════════════════════════════════════════
# Route-Aware Retrieval
# ══════════════════════════════════════════════════════════════

def retrieve_chunks(
    index: ChunkIndex,
    question: str,
    route: str,
    clip_model=None,
    clip_preprocess=None,
    device: str = "cpu",
) -> list[RetrievedChunk]:
    """
    Retrieve relevant chunks using route-aware strategy.

    Route behavior:
        concept   → top-3 transcript similarity
        procedure → top-3 transcript similarity
        temporal  → top-1 + neighboring chunks (before/after)
        visual    → top-3, boosted by CLIP frame similarity
        summary   → top-5 for broader coverage

    Args:
        index:          Built ChunkIndex
        question:       User's question text
        route:          Route label from planner
        clip_model:     CLIP model for visual route (optional)
        clip_preprocess: CLIP preprocessing (optional)
        device:         Device for CLIP inference

    Returns:
        List of RetrievedChunk objects, sorted by combined_score desc
    """
    if not index.is_built:
        raise RuntimeError("ChunkIndex not built")

    config = ROUTE_RETRIEVAL_CONFIG.get(route, ROUTE_RETRIEVAL_CONFIG["concept"])
    top_k = config["top_k"]
    use_clip_boost = config["use_clip_boost"]
    use_temporal_neighbors = config["use_temporal_neighbors"]

    # Step 1: Embed question and search
    query_emb = embed_query(question)
    results = index.search(query_emb, top_k=top_k)

    # Step 2: Temporal neighbor expansion
    if use_temporal_neighbors and results:
        # Get the top-1 match, then add its neighbors
        top_idx = results[0][0]
        neighbor_indices = set()
        if top_idx > 0:
            neighbor_indices.add(top_idx - 1)
        neighbor_indices.add(top_idx)
        if top_idx < len(index.chunks) - 1:
            neighbor_indices.add(top_idx + 1)

        # Rebuild results with neighbors, keeping original scores
        all_scores = {idx: score for idx, score in results}
        for ni in neighbor_indices:
            if ni not in all_scores:
                # Compute score for neighbor
                score = float(
                    index.embeddings[ni] @ query_emb.T
                ) if index.embeddings is not None else 0.0
                all_scores[ni] = score

        results = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

    # Step 3: Build RetrievedChunk objects
    retrieved: list[RetrievedChunk] = []
    for chunk_idx, transcript_score in results:
        chunk = index.chunks[chunk_idx]

        # CLIP frame selection (for all routes — picks the best frame)
        selected_frame = None
        clip_score = None

        if chunk.frame_paths and clip_model is not None:
            selected_frame, clip_score = select_best_frame(
                question, chunk.frame_paths,
                clip_model, clip_preprocess, device,
            )

        # Combined score
        combined = transcript_score
        if use_clip_boost and clip_score is not None:
            # Blend: 70% transcript + 30% CLIP
            combined = 0.7 * transcript_score + 0.3 * clip_score

        retrieved.append(RetrievedChunk(
            chunk=chunk,
            transcript_score=transcript_score,
            clip_score=clip_score,
            combined_score=combined,
            selected_frame=selected_frame or (
                chunk.frame_paths[0] if chunk.frame_paths else None
            ),
        ))

    # Re-sort by combined score
    retrieved.sort(key=lambda r: r.combined_score, reverse=True)

    logger.info(
        "Retrieved %d chunks for route=%s (top score=%.3f)",
        len(retrieved), route,
        retrieved[0].combined_score if retrieved else 0.0,
    )
    return retrieved


# ══════════════════════════════════════════════════════════════
# CLIP Frame Selection
# ══════════════════════════════════════════════════════════════

def select_best_frame(
    question: str,
    frame_paths: list[str],
    clip_model,
    clip_preprocess,
    device: str = "cpu",
) -> tuple[str, float]:
    """
    Select the frame most relevant to the question using CLIP.

    Args:
        question:       The user's question
        frame_paths:    List of paths to frame images
        clip_model:     Loaded CLIP model
        clip_preprocess: CLIP preprocessing transform
        device:          Device for inference

    Returns:
        (best_frame_path, similarity_score)
    """
    import torch
    from PIL import Image
    import clip

    # Encode question
    text_tokens = clip.tokenize([question], truncate=True).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Encode frames
    best_path = frame_paths[0]
    best_score = -1.0

    for fpath in frame_paths:
        try:
            image = clip_preprocess(Image.open(fpath)).unsqueeze(0).to(device)
            with torch.no_grad():
                img_features = clip_model.encode_image(image)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            sim = (text_features @ img_features.T).item()
            if sim > best_score:
                best_score = sim
                best_path = fpath
        except Exception as e:
            logger.warning("Failed to process frame %s: %s", fpath, e)

    return best_path, best_score
