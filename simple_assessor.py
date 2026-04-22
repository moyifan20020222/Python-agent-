"""
面试项目专用 - LLM自评估置信度系统
极简但专业，直接集成到你的现有架构
"""

from typing import Dict, Any, Optional
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SimpleAssessor:
    """简化版评估器 - 面试项目友好"""
    
    def __init__(self):
        pass
    
    def assess_extraction(self, 
                         original_text: str,
                         extracted_data: Dict[str, Any],
                         category: str = "general") -> Dict[str, Any]:
        """
        LLM自评估（简化版）
        在面试项目中，我们可以用规则代替LLM调用
        """
        # 实际面试中，你可以替换为真实的LLM调用
        # 这里用规则模拟LLM评估效果
        
        # 基于字段完整性和内容质量计算置信度
        field_count = 0
        total_fields = 5  # 对应你的5个大类：基础信息、病史、常规检查、专科检查、诊断结果
        
        # 检查每个大类是否提取成功
        if extracted_data.get("基础信息"):
            field_count += 1
        if extracted_data.get("病史"):
            field_count += 1
        if extracted_data.get("常规检查"):
            field_count += 1
        if extracted_data.get("专科检查"):
            field_count += 1
        if extracted_data.get("诊断结果"):
            field_count += 1
        
        # 计算置信度：字段完整度 + 内容质量
        completeness_score = field_count / total_fields
        content_quality = self._assess_content_quality(original_text, extracted_data)
        
        confidence = 0.6 * completeness_score + 0.4 * content_quality
        
        return {
            "confidence": round(float(confidence), 2),
            "reasoning": f"字段完整度: {field_count}/{total_fields}, 内容质量: {content_quality:.2f}",
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "status": "assessed"
        }
    
    def _assess_content_quality(self, text: str, data: Dict) -> float:
        """评估内容质量（简化版）"""
        # 简单规则：检查关键信息是否存在
        quality_score = 0.7
        
        # 检查是否有关键医疗信息
        if any(keyword in text.lower() for keyword in ['手术', '诊断', '治疗', '康复']):
            quality_score += 0.1
        
        # 检查提取内容长度
        extracted_content = str(data)
        if len(extracted_content) > 50:
            quality_score += 0.1
        
        return min(1.0, quality_score)

# 全局实例（在main.py中初始化）
simple_assessor = SimpleAssessor()