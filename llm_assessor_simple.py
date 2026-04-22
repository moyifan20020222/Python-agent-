"""
简化版LLM自评估系统 - 面试项目友好版
直接集成到你的现有架构
"""

from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SimpleLLMAssessor:
    """简化版LLM置信度评估器"""
    
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm = llm_client
    
    def assess_extraction(self, 
                         original_text: str,
                         extracted_data: Dict[str, Any],
                         category: str = "general") -> Dict[str, Any]:
        """
        LLM自评估单次提取
        参数：
        - original_text: 原始文本
        - extracted_data: 提取结果
        - category: 提取类别（对应extractor.py的大类）
        """
        if not self.llm:
            # 没有LLM时返回默认评估
            return self._default_assessment(extracted_data, category)
        
        try:
            # 构建简单prompt
            prompt = self._build_simple_prompt(original_text, extracted_data, category)
            
            # 调用LLM
            response = self.llm.generate(prompt, max_tokens=200)
            
            # 解析结果
            return self._parse_simple_response(response)
            
        except Exception as e:
            logger.error(f"LLM assessment error: {e}")
            return self._default_assessment(extracted_data, category)
    
    def _build_simple_prompt(self, text: str, data: Dict, category: str) -> str:
        """构建简单prompt"""
        return f"""你是一名临床专家，请评估以下提取结果的质量：

【原始文本片段】
{text[:300]}...

【提取结果】
{json.dumps(data, ensure_ascii=False, indent=2)}

【提取类别】
{category}

请按以下格式回答：
{{"confidence": 0.85, "reasoning": "提取准确，但缺少部分细节", "critical_issues": []}}

只输出JSON，不要其他内容。"""

    def _parse_simple_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            return json.loads(response)
        except:
            return {"confidence": 0.7, "reasoning": "LLM响应解析失败", "critical_issues": []}
    
    def _default_assessment(self, data: Dict, category: str) -> Dict[str, Any]:
        """默认评估（无LLM时）"""
        # 基于字段完整性计算置信度
        field_count = len([v for v in data.values() if v])
        total_fields = len(data)
        
        confidence = 0.6 + (field_count / max(1, total_fields)) * 0.4
        
        return {
            "confidence": round(float(confidence), 2),
            "reasoning": f"字段完整度: {field_count}/{total_fields}",
            "critical_issues": [],
            "category": category
        }