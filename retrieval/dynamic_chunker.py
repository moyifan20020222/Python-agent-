"""
动态分块器 - 基于信息熵和业务规则
"""

from .information_entropy import assess_content_quality
from typing import List, Dict, Tuple, Optional
import re
from datetime import datetime

class DynamicMedicalChunker:
    """医学领域动态分块器"""
    
    def __init__(self):
        # 配置参数
        self.config = {
            'min_chunk_size': 300,
            'max_chunk_size': 2500,
            'small_doc_threshold': 800,  # 小文档阈值
            'medium_doc_threshold': 1500,  # 中等文档阈值
            'entropy_threshold': 0.6,     # 信息熵阈值
        }
    
    def chunk_document(self, 
                      title: str,
                      category: str,
                      disease: str,
                      content: str,
                      source: str) -> Tuple[List[Dict], List[Dict]]:
        """
        动态分块主方法
        根据文档类型、来源、信息熵自动调整分块策略
        """
        # 1. 评估内容质量
        quality_info = assess_content_quality(content, source, category)
        
        # 2. 确定分块策略
        chunk_strategy = self._determine_chunk_strategy(
            category, source, content, quality_info
        )
        
        # 3. 执行分块
        if chunk_strategy == "small":
            parent_chunks, child_chunks = self._small_chunking(content, quality_info)
        elif chunk_strategy == "medium":
            parent_chunks, child_chunks = self._medium_chunking(content, quality_info)
        else:  # "large"
            parent_chunks, child_chunks = self._large_chunking(content, quality_info)
        
        # 4. 添加元数据
        for i, chunk in enumerate(parent_chunks):
            chunk['metadata'] = {
                'title': title,
                'category': category,
                'disease': disease,
                'source': source,
                'chunk_type': chunk_strategy,
                'quality_score': quality_info['quality_score'],
                'text_length': len(content),
                'entropy_score': quality_info['entropy_score'],
                'importance_weight': quality_info.get('importance_weight', 1.0),
                'created_at': datetime.now().isoformat()
            }
        
        return parent_chunks, child_chunks
    
    def _determine_chunk_strategy(self, 
                                category: str, 
                                source: str, 
                                content: str,
                                quality_info: Dict[str, float]) -> str:
        """
        确定分块策略
        规则：
        - 饮食/运动：小块（语义明确）
        - 心理/药物：中等块
        - 综合：大块（内容全面）
        - 信息熵高 + 来源可靠：可适当增大块大小
        """
        text_length = len(content)
        
        # 基础策略
        if category in ["饮食", "运动"]:
            strategy = "small"
        elif category in ["心理", "药物"]:
            strategy = "medium"
        elif category == "综合":
            strategy = "large"
        else:
            strategy = "medium"
        
        # 调整策略
        entropy_score = quality_info.get('entropy_score', 0.5)
        importance_weight = quality_info.get('importance_weight', 1.0)
        
        # 信息熵高且重要性高 → 增大块大小
        if entropy_score > 0.7 and importance_weight > 1.1:
            if strategy == "small":
                strategy = "medium"
            elif strategy == "medium":
                strategy = "large"
        
        # 文档很小 → 强制小块
        if text_length < 500:
            strategy = "small"
        
        return strategy
    
    def _small_chunking(self, content: str, quality_info: Dict) -> Tuple[List[Dict], List[Dict]]:
        """小文档分块：300-500字符"""
        chunks = []
        start = 0
        chunk_size = 400
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            
            # 确保在句子边界结束
            while end < len(content) and content[end] not in ['\n', '。', '！', '？', '；', '：']:
                end += 1
            
            chunk_content = content[start:end].strip()
            if chunk_content:
                chunks.append({
                    'page_content': chunk_content,
                    'metadata': {}
                })
            start = end + 1
        
        return self._create_child_chunks(chunks, 200)
    
    def _medium_chunking(self, content: str, quality_info: Dict) -> Tuple[List[Dict], List[Dict]]:
        """中等文档分块：500-800字符"""
        chunks = []
        start = 0
        chunk_size = 600
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            
            # 确保在句子边界结束
            while end < len(content) and content[end] not in ['\n', '。', '！', '？', '；', '：']:
                end += 1
            
            chunk_content = content[start:end].strip()
            if chunk_content:
                chunks.append({
                    'page_content': chunk_content,
                    'metadata': {}
                })
            start = end + 1
        
        return self._create_child_chunks(chunks, 300)
    
    def _large_chunking(self, content: str, quality_info: Dict) -> Tuple[List[Dict], List[Dict]]:
        """大文档分块：1500-2500字符"""
        chunks = []
        start = 0
        chunk_size = 2000
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            
            # 确保在段落边界结束
            paragraphs = content[start:end].split('\n\n')
            if len(paragraphs) > 1:
                # 保留完整段落
                end = start + len('\n\n'.join(paragraphs[:-1]))
            
            chunk_content = content[start:end].strip()
            if chunk_content:
                chunks.append({
                    'page_content': chunk_content,
                    'metadata': {}
                })
            start = end + 1
        
        return self._create_child_chunks(chunks, 500)
    
    def _create_child_chunks(self, parent_chunks: List[Dict], child_size: int) -> Tuple[List[Dict], List[Dict]]:
        """创建子块"""
        all_parent_chunks = []
        all_child_chunks = []
        
        for i, parent_chunk in enumerate(parent_chunks):
            parent_content = parent_chunk['page_content']
            parent_metadata = parent_chunk.get('metadata', {})
            
            # 创建父块
            parent_chunk_obj = {
                'page_content': parent_content,
                'metadata': {
                    **parent_metadata,
                    'parent_index': i,
                    'total_parents': len(parent_chunks)
                }
            }
            all_parent_chunks.append(parent_chunk_obj)
            
            # 创建子块
            start = 0
            while start < len(parent_content):
                end = min(start + child_size, len(parent_content))
                
                # 句子边界
                while end < len(parent_content) and parent_content[end] not in ['\n', '。', '！', '？', '；', '：']:
                    end += 1
                
                child_content = parent_content[start:end].strip()
                if child_content:
                    child_chunk = {
                        'page_content': child_content,
                        'metadata': {
                            **parent_metadata,
                            'child_index': len(all_child_chunks),
                            'total_children': 0,
                            'parent_id': f"parent_{i}"
                        }
                    }
                    all_child_chunks.append(child_chunk)
                start = end + 1
        
        # 更新总子块数量
        for chunk in all_child_chunks:
            chunk['metadata']['total_children'] = len([c for c in all_child_chunks 
                                                    if c['metadata'].get('parent_id') == chunk['metadata'].get('parent_id')])
        
        return all_parent_chunks, all_child_chunks