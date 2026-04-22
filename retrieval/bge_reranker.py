"""
BGE-Reranker重排器 - 开源专业版
"""

from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
import torch

class BGEReranker:
    """BGE重排器 - 医学领域优化"""
    
    def __init__(self, model_name: str = "D:\\Desktop\\company\\Agent-rag\\agentic-rag-for-dummies\\models\\bge-reranker-base", device: str = "cpu"):
        """
        初始化BGE重排器
        model_name: "BAAI/bge-reranker-base" 或 "BAAI/bge-reranker-v2-m3"
        """
        self.model_name = model_name
        self.device = device
        
        # 加载模型
        try:
            self.model = CrossEncoder(
                model_name,
                device=device,
                trust_remote_code=True
            )
            print(f"✅ BGE-Reranker加载成功: {model_name} ({device})")
        except Exception as e:
            print(f"❌ BGE-Reranker加载失败: {e}")
            # 备用：使用简单重排器
            self.model = None
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        重排文档列表
        返回: [{'document': str, 'score': float, 'rerank_score': float}, ...]
        """
        if not self.model:
            # 备用方案
            return self._simple_rerank(query, documents, top_k)
        
        try:
            # 构建查询-文档对
            pairs = [[query, doc] for doc in documents]
            
            # 批量推理
            scores = self.model.predict(pairs)
            
            # 创建结果
            results = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                results.append({
                    'document': doc,
                    'original_score': score,
                    'rerank_score': float(score),
                    'source': 'bge_reranker'
                })
            
            # 排序
            results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            print(f"❌ BGE重排失败: {e}")
            return self._simple_rerank(query, documents, top_k)
    
    def _simple_rerank(self, query: str, documents: List[str], top_k: int) -> List[Dict[str, Any]]:
        """简单重排备用方案"""
        results = []
        for i, doc in enumerate(documents):
            # 简单关键词匹配
            keyword_score = sum(1 for word in query.split() if word in doc) / len(query.split()) if query else 0
            results.append({
                'document': doc,
                'original_score': keyword_score,
                'rerank_score': keyword_score,
                'source': 'simple'
            })
        
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        return results[:top_k]

# 全局实例（在main.py中初始化）
bge_reranker = BGEReranker(device="cuda" if torch.cuda.is_available() else "cpu")