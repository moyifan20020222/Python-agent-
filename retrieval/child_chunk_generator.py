"""
子chunk生成器 - 动态分块策略
基于文档类型、来源、信息密度生成最优子chunk
"""

from .information_entropy import assess_content_quality
from typing import List, Dict, Tuple, Optional
import re
from datetime import datetime


class DynamicChildChunkGenerator:
    """动态子chunk生成器"""

    def __init__(self):
        # 配置参数
        self.config = {
            'min_child_size': 200,  # 最小字块大小
            'max_child_size': 600,  # 最大字块大小
            'small_doc_threshold': 800,  # 小文档阈值
            'medium_doc_threshold': 1500,  # 中等文档阈值
            'entropy_threshold': 0.6,  # 信息熵阈值
        }

    def generate_child_chunks(self,
                              parent_content: str,
                              title: str,
                              category: str,
                              disease: str,
                              source: str) -> List[Dict]:
        """
        根据文档特征生成最优子chunk
        返回：子chunk列表（用于存入ChromaDB）
        """
        # 1. 评估内容质量
        quality_info = assess_content_quality(parent_content, source, category)

        # 2. 确定子块大小策略
        child_size = self._determine_child_size(
            category, source, parent_content, quality_info
        )

        # 3. 生成子chunk
        child_chunks = self._create_child_chunks(
            parent_content, child_size, quality_info
        )

        # 4. 添加元数据
        for i, chunk in enumerate(child_chunks):
            chunk['metadata'].update({
                'title': title,
                'category': category,
                'disease': disease,
                'source': source,
                'child_index': i,
                'total_children': len(child_chunks),
                'child_size': child_size,
                'quality_score': quality_info['quality_score'],
                'entropy_score': quality_info['entropy_score'],
                'importance_weight': quality_info.get('importance_weight', 1.0),
                'created_at': datetime.now().isoformat()
            })

        return child_chunks

    def _determine_child_size(self,
                              category: str,
                              source: str,
                              content: str,
                              quality_info: Dict[str, float]) -> int:
        """
        确定子块大小
        规则：
        - 饮食/运动：小块（200-400字符）→ 语义明确，需要精准检索
        - 心理/药物：中等块（300-500字符）→ 理论性强
        - 综合：大块（400-600字符）→ 内容全面，需要上下文
        - 信息熵高 + 来源可靠：可适当增大块大小
        """
        text_length = len(content)

        # 基础大小
        if category in ["饮食", "运动"]:
            base_size = 300
        elif category in ["心理", "药物"]:
            base_size = 400
        elif category == "综合":
            base_size = 500
        else:
            base_size = 400

        # 调整因子
        entropy_score = quality_info.get('entropy_score', 0.5)
        importance_weight = quality_info.get('importance_weight', 1.0)

        # 信息熵高且重要性高 → 增大块大小
        if entropy_score > 0.7 and importance_weight > 1.1:
            size_factor = 1.2
        elif entropy_score < 0.4:
            size_factor = 0.8
        else:
            size_factor = 1.0

        # 文档大小影响
        if text_length < 500:
            size_factor *= 1.1
        elif text_length > 2000:
            size_factor *= 0.9

        # 计算最终大小
        final_size = int(base_size * size_factor)

        # 约束范围
        return max(self.config['min_child_size'],
                   min(self.config['max_child_size'], final_size))

    def _create_child_chunks(self,
                             content: str,
                             child_size: int,
                             quality_info: Dict) -> List[Dict]:
        """创建子chunk"""
        chunks = []
        start = 0

        while start < len(content):
            end = min(start + child_size, len(content))

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

        return chunks
