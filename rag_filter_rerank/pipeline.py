"""
Pipeline orchestrator for filter-rerank RAG.

Coordinates retrieval, filtering, reranking, and answering with deterministic output.
"""
import time
import logging
from typing import List, Dict, Any, Optional
from .schemas import Chunk, PipelineResult
from .cache import Cache, get_cache
from .filter_engine import FilterEngine
from .reranker import Reranker, build_items_for_rerank
from .evidence import build_evidence_from_chunks, call_answerer
from .retriever import RetrieverAdapter

logger = logging.getLogger(__name__)


class FilterRerankPipeline:
    """Main pipeline coordinating all stages."""
    
    def __init__(
        self,
        retriever: RetrieverAdapter,
        settings,
        cache: Optional[Cache] = None,
        filter_engine: Optional[FilterEngine] = None,
        reranker: Optional[Reranker] = None
    ):
        """
        Initialize pipeline.
        
        Args:
            retriever: Retriever adapter
            settings: Configuration settings
            cache: Optional cache (will create if not provided)
            filter_engine: Optional filter engine (will create if not provided)
            reranker: Optional reranker (will create if not provided)
        """
        self.retriever = retriever
        self.settings = settings
        
        # Initialize components
        self.cache = cache or get_cache(settings)
        self.filter_engine = filter_engine or FilterEngine(self.cache, settings)
        self.reranker = reranker or Reranker(self.cache, settings)
        
        logger.info(f"Pipeline initialized (mode={settings.RAG_FILTER_RERANK})")
    
    def run(self, query: str, trace: bool = False) -> PipelineResult:
        """
        Execute full pipeline.
        
        Args:
            query: User query
            trace: Enable trace logging
        
        Returns:
            PipelineResult with answer and metadata
        """
        timing = {}
        start_total = time.time()
        
        # Stage 0: Retrieval
        logger.info(f"[Stage 0] Retrieving top-{self.settings.PIPELINE_TOPN_RECALL} chunks")
        start = time.time()
        
        try:
            chunks = self.retriever.search(query, self.settings.PIPELINE_TOPN_RECALL)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            chunks = []
        
        timing['retrieval_ms'] = int((time.time() - start) * 1000)
        logger.info(f"[Stage 0] Retrieved {len(chunks)} chunks ({timing['retrieval_ms']}ms)")
        
        if trace:
            self._trace_chunks("After Retrieval", chunks[:10])
        
        # Fallback if no chunks
        if not chunks:
            return self._empty_result(query, timing)
        
        # Stage 1: Filter
        logger.info(f"[Stage 1] Filtering with LLM (threshold>={self.settings.FILTER_THRESHOLD})")
        start = time.time()
        
        try:
            filtered = self.filter_engine.filter_chunks(query, chunks)
        except Exception as e:
            logger.error(f"Filtering failed: {e}, using all chunks")
            # Fallback: use all chunks
            from .schemas import FilterResult
            filtered = [FilterResult(chunk=c, score=0, why="filter-error") for c in chunks]
        
        timing['filter_ms'] = int((time.time() - start) * 1000)
        logger.info(f"[Stage 1] Filtered to {len(filtered)} chunks ({timing['filter_ms']}ms)")
        
        if trace:
            self._trace_filtered("After Filter", filtered[:10])
        
        # Ensure minimum chunks
        if len(filtered) < self.settings.FILTER_MIN_SURVIVORS:
            logger.warning(f"Only {len(filtered)} survivors, using baseline")
            from .schemas import FilterResult
            filtered = [FilterResult(chunk=c, score=1, why="baseline") for c in chunks[:self.settings.FILTER_MIN_SURVIVORS]]
        
        # Stage 2: Rerank
        if self.settings.DISABLE_RERANK or self.settings.RAG_FILTER_RERANK == "off":
            logger.info("[Stage 2] Reranking disabled, skipping")
            timing['rerank_ms'] = 0
            # Use filter scores for ordering
            reranked_chunks = [f.chunk for f in filtered]
        else:
            logger.info(f"[Stage 2] Reranking {len(filtered)} chunks")
            start = time.time()
            
            try:
                items = build_items_for_rerank(filtered)
                scores = self.reranker.rerank(query, items)
                
                # Map scores back to chunks
                score_map = {s.id: s.score for s in scores}
                for f in filtered:
                    f.chunk.rerank_score = score_map.get(f.chunk.id, 0.0)
                
                # Sort by rerank score
                filtered.sort(key=lambda f: (-f.chunk.rerank_score, -f.score, f.chunk.provenance_id))
                reranked_chunks = [f.chunk for f in filtered]
                
            except Exception as e:
                logger.error(f"Reranking failed: {e}, using filter order")
                reranked_chunks = [f.chunk for f in filtered]
            
            timing['rerank_ms'] = int((time.time() - start) * 1000)
            logger.info(f"[Stage 2] Reranked ({timing['rerank_ms']}ms)")
        
        if trace:
            self._trace_chunks("After Rerank", reranked_chunks[:10])
        
        # Stage 3: Take top-K for evidence
        k = self.settings.PIPELINE_TOPK_EVIDENCE
        top_k = reranked_chunks[:k]
        
        logger.info(f"[Stage 3] Selected top-{k} for evidence")
        
        # Stage 4: Build evidence and call answerer
        logger.info("[Stage 4] Building evidence and calling answerer")
        start = time.time()
        
        try:
            # Convert chunks to dicts for evidence rendering
            chunk_dicts = [self._chunk_to_dict(c) for c in top_k]
            evidence = build_evidence_from_chunks(chunk_dicts, k=k, max_code_chars=10000)
            
            # Call answerer
            answer = call_answerer(query, evidence, self.settings)
        except Exception as e:
            logger.error(f"Answering failed: {e}")
            answer = f"Error generating answer: {e}"
            evidence = ""
        
        timing['answer_ms'] = int((time.time() - start) * 1000)
        timing['total_ms'] = int((time.time() - start_total) * 1000)
        
        logger.info(f"[Stage 4] Answer generated ({timing['answer_ms']}ms)")
        logger.info(f"Pipeline complete (total={timing['total_ms']}ms)")
        
        # Build reranked metadata with deterministic ordering
        reranked_meta = []
        for i, chunk in enumerate(top_k):
            reranked_meta.append({
                "rank": i + 1,
                "id": chunk.id,
                "provenance_id": chunk.provenance_id,
                "rel_path": chunk.rel_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "filter_score": chunk.filter_score,
                "rerank_score": chunk.rerank_score,
            })
        
        # Sort deterministically for reproducibility
        reranked_meta.sort(key=lambda x: (-(x.get("rerank_score") or 0), -(x.get("filter_score") or 0), 
                                          len(chunk_dicts[x["rank"]-1].get("text", "")), x["provenance_id"]))
        
        # Cache statistics
        cache_stats = self.cache.stats()
        
        # Build result
        result = PipelineResult(
            query=query,
            recall_n=len(chunks),
            filtered_m=len(filtered),
            reranked=reranked_meta,
            answer=answer,
            evidence_k=len(top_k),
            timing_ms=timing,
            cache_stats=cache_stats,
            flags={
                "rag_filter_rerank": self.settings.RAG_FILTER_RERANK,
                "disable_rerank": self.settings.DISABLE_RERANK,
                "filter_threshold": self.settings.FILTER_THRESHOLD,
            }
        )
        
        return result
    
    def _chunk_to_dict(self, chunk: Chunk) -> Dict[str, Any]:
        """Convert Chunk to dict for evidence rendering."""
        return {
            "id": chunk.id,
            "repo_id": chunk.repo_id,
            "rel_path": chunk.rel_path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "language": chunk.language,
            "text": chunk.text,
            "summary_en": chunk.summary_en,
            "primary_symbol": "",
            "primary_kind": "",
        }
    
    def _empty_result(self, query: str, timing: Dict[str, int]) -> PipelineResult:
        """Build empty result for no chunks case."""
        return PipelineResult(
            query=query,
            recall_n=0,
            filtered_m=0,
            reranked=[],
            answer="No relevant chunks found to answer your question.",
            evidence_k=0,
            timing_ms=timing,
            cache_stats=self.cache.stats(),
            flags={
                "rag_filter_rerank": self.settings.RAG_FILTER_RERANK,
                "disable_rerank": self.settings.DISABLE_RERANK,
            }
        )
    
    def _trace_chunks(self, stage: str, chunks: List[Chunk]):
        """Trace chunk info for debugging."""
        logger.info(f"\n=== {stage} ===")
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"  [{i}] {chunk.rel_path}:{chunk.start_line}-{chunk.end_line}")
            if chunk.filter_score is not None:
                logger.info(f"      filter={chunk.filter_score}")
            if chunk.rerank_score is not None:
                logger.info(f"      rerank={chunk.rerank_score:.3f}")
    
    def _trace_filtered(self, stage: str, filtered: List):
        """Trace filtered results."""
        logger.info(f"\n=== {stage} ===")
        for i, result in enumerate(filtered, 1):
            logger.info(f"  [{i}] score={result.score} why={result.why}")
            logger.info(f"      {result.chunk.rel_path}:{result.chunk.start_line}-{result.chunk.end_line}")

