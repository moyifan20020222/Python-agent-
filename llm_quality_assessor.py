"""
精准LLM质量评估器 - 基于你的schema.py字段定义
严格检查：1) 是否符合提取规则 2) 信息是否在原文中存在
"""

from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PreciseLLMAssessor:
    """精准LLM质量评估器"""
    
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm = llm_client
        # 你的字段提取规则（从schema.py映射）
        self.extraction_rules = {
            "基础信息": {
                "fields": ["患者姓名", "性别", "年龄", "住院号", "入院时间"],
                "extraction_rule": "提取患者基本身份信息，必须是明确提到的内容",
                "critical": True
            },
            "病史": {
                "fields": ["现病史", "既往史", "家族史", "过敏史"],
                "extraction_rule": "提取疾病发展过程和相关历史，必须有原文依据",
                "critical": True
            },
            "常规检查": {
                "fields": ["血常规", "尿常规", "肝功能", "肾功能", "凝血功能"],
                "extraction_rule": "提取实验室检查结果，必须包含具体数值和单位",
                "critical": False
            },
            "专科检查": {
                "fields": ["影像学检查", "心电图", "神经功能检查", "关节活动度"],
                "extraction_rule": "提取专科检查发现，必须有明确描述",
                "critical": True
            },
            "诊断结果": {
                "fields": ["最终诊断", "诊断依据", "鉴别诊断", "ICD-10编码"],
                "extraction_rule": "提取最终临床诊断，必须有明确诊断结论",
                "critical": True
            }
        }
    
    def assess_extraction(self, 
                         original_text: str,
                         extracted_data: Dict[str, Any],
                         category: str = "general") -> Dict[str, Any]:
        """
        精准LLM评估：检查1) 规则符合性 2) 原文存在性
        """
        if not self.llm:
            raise ValueError("LLM客户端未初始化，请在main.py中设置llm_assessor.llm = your_llm_client")
        
        try:
            # 构建精准评估prompt
            prompt = self._build_precise_prompt(original_text, extracted_data, category)
            
            # 调用LLM
            response = self.llm.generate(prompt, max_tokens=300)
            
            # 解析结果
            assessment = self._parse_precise_response(response)
            
            # 计算综合置信度
            confidence_score = self._calculate_confidence(assessment)
            
            return {
                "confidence_score": float(confidence_score),
                "detailed_assessment": assessment,
                "rule_compliance": self._check_rule_compliance(assessment),
                "evidence_verification": self._verify_evidence(assessment),
                "timestamp": datetime.now().isoformat(),
                "category": category
            }
            
        except Exception as e:
            logger.error(f"LLM评估失败: {e}")
            raise
    
    def _build_precise_prompt(self, text: str, data: Dict, category: str) -> str:
        """构建精准评估prompt"""
        rules_str = json.dumps(self.extraction_rules, ensure_ascii=False, indent=2)
        
        return f"""你是一名资深临床医学专家，请严格按照以下要求评估提取结果的质量：

【评估任务】
1. 检查提取内容是否符合指定的提取规则
2. 验证提取的信息是否在原始文本中有明确依据
3. 对每个字段给出置信度评分（0.0-1.0）

【提取规则】
{rules_str}

【原始文本】
{text[:2000]}...  # 限制长度

【提取结果】
{json.dumps(data, ensure_ascii=False, indent=2)}

【评估要求】
- 对每个字段评估两个维度：
  a) 规则符合性：是否按照提取规则提取
  b) 原文存在性：信息是否在原文中有明确依据
- 置信度计算：规则符合性×0.6 + 原文存在性×0.4
- 输出格式必须为JSON：
{{
  "field_assessments": {{
    "字段名": {{
      "confidence": 0.85,
      "rule_compliance": true/false,
      "evidence_exists": true/false,
      "reasoning": "详细说明",
      "evidence_span": [start, end]
    }}
  }},
  "overall_confidence": 0.82,
  "critical_issues": ["问题描述"]
}}

只输出JSON，不要其他内容。"""

    def _parse_precise_response(self, response: str) -> Dict:
        """解析LLM响应"""
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            # 尝试提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            raise ValueError(f"LLM响应格式错误: {e}")
    
    def _calculate_confidence(self, assessment: Dict) -> float:
        """计算综合置信度"""
        field_assessments = assessment.get("field_assessments", {})
        
        if not field_assessments:
            return 0.5
        
        confidence_scores = []
        for field_name, eval_result in field_assessments.items():
            if isinstance(eval_result, dict):
                rule_score = 1.0 if eval_result.get("rule_compliance", False) else 0.0
                evidence_score = 1.0 if eval_result.get("evidence_exists", False) else 0.0
                confidence = rule_score * 0.6 + evidence_score * 0.4
                confidence_scores.append(confidence)
        
        return float(sum(confidence_scores) / len(confidence_scores)) if confidence_scores else 0.5
    
    def _check_rule_compliance(self, assessment: Dict) -> Dict[str, bool]:
        """检查规则符合性"""
        field_assessments = assessment.get("field_assessments", {})
        return {
            field: eval_result.get("rule_compliance", False)
            for field, eval_result in field_assessments.items()
            if isinstance(eval_result, dict)
        }
    
    def _verify_evidence(self, assessment: Dict) -> Dict[str, bool]:
        """验证证据存在性"""
        field_assessments = assessment.get("field_assessments", {})
        return {
            field: eval_result.get("evidence_exists", False)
            for field, eval_result in field_assessments.items()
            if isinstance(eval_result, dict)
        }

# 全局实例（在你的main.py中初始化）
llm_assessor = PreciseLLMAssessor()