"""
轻量级重排模型 - 医学领域专用
"""

from typing import List, Dict, Any
import math

class SimpleReranker:
    """简单重排器 - 医学场景优化"""
    
    def __init__(self):
        # 医学术语权重
        self.medical_keywords = {
            "诊断": 2.0,
            "手术": 2.0,
            "药物": 1.8,
            "禁忌症": 3.0,
            "ICD": 2.5,
            "指南": 1.5,
            "随访": 1.2,
            "康复": 1.0
        }
    
    def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        重排结果
        基于：向量相似度 + 关键词匹配 + 医学术语权重
        """
        if not results:
            return results
        
        # 计算查询关键词
        query_keywords = self._extract_keywords(query)
        
        reranked = []
        for i, result in enumerate(results):
            # 原始分数
            base_score = result.get('score', 0.0) or result.get('distance', 0.0)
            if 'distance' in result:
                base_score = 1.0 / (1.0 + base_score)  # 转换为相似度
            
            # 关键词匹配分数
            keyword_score = self._calculate_keyword_score(query_keywords, result.get('document', ''))
            
            # 医学术语权重
            medical_score = self._calculate_medical_weight(result.get('document', ''))
            
            # 综合分数
            final_score = (
                0.6 * base_score +
                0.3 * keyword_score +
                0.1 * medical_score
            )
            
            result['rerank_score'] = final_score
            reranked.append(result)
        
        # 排序
        reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取查询关键词"""
        # 简单分词
        words = text.split()
        return [word for word in words if len(word) > 1]
    
    def _calculate_keyword_score(self, query_keywords: List[str], document: str) -> float:
        """计算关键词匹配分数"""
        if not query_keywords:
            return 0.0
        
        matches = sum(1 for kw in query_keywords if kw in document)
        return min(1.0, matches / len(query_keywords))
    
    def _calculate_medical_weight(self, document: str) -> float:
        """计算医学术语权重"""
        weight = 0.0
        for term, score in self.medical_keywords.items():
            if term in document:
                weight += score
        
        # 归一化到0-1
        max_weight = sum(self.medical_keywords.values())
        return min(1.0, weight / max_weight)