"""
最终混合检索系统 - 包含重排
"""
import concurrent.futures

# from .semantic_boundary_detector import SimpleMedicalBoundaryDetector
from .information_entropy import calculate_text_entropy
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import CrossEncoder
import torch

class FinalHybridRetrieval:
    """最终混合检索系统 - 包含重排"""
    
    def __init__(self, 
                vector_retriever: Any,
                bm25_retriever: Any,
                reranker_model: str = "D:\\Desktop\\company\\Agent-rag\\agentic-rag-for-dummies\\models\\bge-reranker-base",
                device: str = "cpu"):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        
        # 初始化重排器
        try:
            self.reranker = CrossEncoder(
                reranker_model,
                device=device,
                trust_remote_code=True
            )
            print(f"✅ 重排器加载成功: {reranker_model}")
        except Exception as e:
            print(f"❌ 重排器加载失败: {e}")
            self.reranker = None

    def search(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """真·并行混合检索"""
        # === 核心优化：使用多线程并行查询，极大降低双路检索耗时 ===
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_vector = executor.submit(self.vector_retriever.search, query, k * 2, filters)
            # 这里的 filters 完美传给了 BM25
            future_bm25 = executor.submit(self.bm25_retriever.search, query, k * 2, filters)

            vector_results = future_vector.result()
            bm25_results = future_bm25.result()

        # 2. 使用 RRF 算法融合
        fused_results = self._fuse_results_rrf(vector_results, bm25_results)

        # 3. BGE 重排
        if self.reranker:
            return self._rerank_with_bge(query, fused_results, k)
        return fused_results[:k]

    def _fuse_results_rrf(self, vector_results: List[Dict], bm25_results: List[Dict], k: int = 60) -> List[Dict]:
        """
        核心优化：使用 Reciprocal Rank Fusion (RRF) 倒数排名融合
        公式: 1 / (k + rank)
        完美解决 Vector 和 BM25 分数尺度不一致的问题
        """
        rrf_scores = {}
        merged_docs = {}

        # 处理向量排名
        for rank, doc in enumerate(vector_results):
            doc_id = doc['id']
            if doc_id not in merged_docs:
                merged_docs[doc_id] = doc
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)

        # 处理BM25排名
        for rank, doc in enumerate(bm25_results):
            doc_id = doc['id']
            if doc_id not in merged_docs:
                merged_docs[doc_id] = doc
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            # 记录来源
            if merged_docs[doc_id].get('source') == 'vector':
                merged_docs[doc_id]['source'] = 'hybrid'

        # 根据 RRF 分数重新排序
        sorted_docs = []
        for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            doc = merged_docs[doc_id]
            doc['rrf_score'] = score
            sorted_docs.append(doc)

        return sorted_docs
    
    def _rerank_with_bge(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
        """使用BGE-Reranker重排"""
        if not results:
            return results
        
        # 构建查询-文档对
        pairs = [[query, result.get('document', '')] for result in results]
        
        # 批量推理
        scores = self.reranker.predict(pairs)
        
        # 添加重排分数
        for i, (result, score) in enumerate(zip(results, scores)):
            result['rerank_score'] = float(score)
        
        # 排序
        return sorted(results, key=lambda x: x.get('rerank_score', 0.0), reverse=True)
    
    def _simple_rerank(self, results: List[Dict], query: str) -> List[Dict]:
        """简单重排备用方案"""
        # 基于关键词匹配
        for result in results:
            content = result.get('document', '')
            keywords = query.split()
            match_score = sum(1 for kw in keywords if kw in content) / len(keywords) if keywords else 0
            result['rerank_score'] = match_score
        
        return sorted(results, key=lambda x: x.get('rerank_score', 0.0), reverse=True)