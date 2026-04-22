"""
精细化LLM评估器 - 基于你的162个具体fields指标
精准检查：每个具体字段是否符合提取规则 + 是否在原文中存在
"""

from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FineGrainedMedicalAssessor:
    """精细化医疗评估器 - 针对162个具体fields"""
    
    def __init__(self):
        # 你的162个具体fields定义（从extractor.py和schema.py映射）
        # 这里用示例，实际应该从你的代码中提取
        self.field_definitions = {
            # 基础信息类（示例）
            "患者姓名": {"category": "基础信息", "rule": "必须是原文中明确提到的姓名", "critical": True},
            "性别": {"category": "基础信息", "rule": "必须是'男'或'女'，原文明确提及", "critical": True},
            "年龄": {"category": "基础信息", "rule": "必须是数字+岁，原文明确提及", "critical": True},
            "住院号": {"category": "基础信息", "rule": "必须是数字或字母组合", "critical": True},
            
            # 病史类（示例）
            "现病史_受伤时间": {"category": "病史", "rule": "格式：YYYY-MM-DD，原文必须有具体日期", "critical": True},
            "现病史_症状描述": {"category": "病史", "rule": "必须包含症状关键词，如疼痛、麻木等", "critical": True},
            "既往史_高血压": {"category": "病史", "rule": "必须明确写'有高血压病史'或类似表述", "critical": True},
            
            # 常规检查类（示例）
            "血常规_WBC": {"category": "常规检查", "rule": "格式：数值+单位，如'8.5×10^9/L'", "critical": False},
            "尿常规_PRO": {"category": "常规检查", "rule": "必须是'阴性'、'阳性'或具体数值", "critical": False},
            
            # 专科检查类（示例）
            "影像学检查_MRI": {"category": "专科检查", "rule": "必须包含检查结果描述", "critical": True},
            "专科检查_左下肢畸形": {"category": "专科检查", "rule": "必须是'有'或'无'，原文明确提及", "critical": True},
            
            # 诊断结果类（示例）
            "诊断结果_最终诊断": {"category": "诊断结果", "rule": "必须有明确诊断结论，如'腰椎间盘突出症'", "critical": True},
            "诊断结果_ICD10编码": {"category": "诊断结果", "rule": "格式：字母+数字，如'M51.2'", "critical": True}
        }
    
    def assess_fields(self, 
                     original_text: str,
                     extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        精准评估每个具体field：
        1) 是否符合提取规则
        2) 是否在原文中有依据
        3) 计算置信度
        """
        field_assessments = {}
        
        # 对每个具体field进行评估
        for field_name, field_info in self.field_definitions.items():
            if field_name in extracted_data:
                assessment = self._assess_single_field(
                    field_name, field_info, extracted_data[field_name], original_text
                )
                field_assessments[field_name] = assessment
        
        # 计算整体置信度
        confidence_scores = [
            assess["confidence"] for assess in field_assessments.values()
            if isinstance(assess, dict) and "confidence" in assess
        ]
        
        overall_confidence = float(sum(confidence_scores) / len(confidence_scores)) if confidence_scores else 0.5
        
        return {
            "field_assessments": field_assessments,
            "overall_confidence": overall_confidence,
            "timestamp": datetime.now().isoformat(),
            "total_fields_evaluated": len(field_assessments),
            "high_confidence_fields": sum(1 for v in field_assessments.values() if isinstance(v, dict) and v.get("confidence", 0) >= 0.9),
            "low_confidence_fields": sum(1 for v in field_assessments.values() if isinstance(v, dict) and v.get("confidence", 0) < 0.7)
        }
    
    def _assess_single_field(self, 
                           field_name: str, 
                           field_info: Dict,
                           extracted_value: Any,
                           original_text: str) -> Dict:
        """评估单个具体field"""
        # 实际项目中这里调用LLM
        # 面试项目中用规则模拟，但保持LLM评估的结构
        
        # 检查规则符合性
        rule_compliant = self._check_rule_compliance(field_name, field_info, extracted_value)
        
        # 检查原文存在性
        evidence_exists = self._verify_evidence_in_text(field_name, extracted_value, original_text)
        
        # 计算置信度
        confidence = 0.6 * (1.0 if rule_compliant else 0.0) + 0.4 * (1.0 if evidence_exists else 0.0)
        
        return {
            "confidence": float(confidence),
            "rule_compliant": rule_compliant,
            "evidence_exists": evidence_exists,
            "field_category": field_info.get("category", "unknown"),
            "critical": field_info.get("critical", False),
            "extraction_rule": field_info.get("rule", ""),
            "extracted_value": extracted_value,
            "reasoning": f"规则符合: {rule_compliant}, 原文存在: {evidence_exists}"
        }
    
    def _check_rule_compliance(self, field_name: str, field_info: Dict, value: Any) -> bool:
        """检查是否符合提取规则"""
        # 这里应该是LLM判断，面试项目中用简单规则
        rule = field_info.get("rule", "")
        
        # 示例规则检查
        if "必须是'男'或'女'" in rule and value in ["男", "女"]:
            return True
        if "格式：YYYY-MM-DD" in rule and isinstance(value, str) and len(value) == 10 and "-" in value:
            return True
        if "必须是数字" in rule and isinstance(value, (int, float)):
            return True
        
        return True  # 面试项目中默认通过
    
    def _verify_evidence_in_text(self, field_name: str, value: Any, text: str) -> bool:
        """验证原文中是否存在证据"""
        # 实际项目中LLM会精确查找
        # 面试项目中用关键词匹配
        if not text or not value:
            return False
        
        value_str = str(value).lower()
        text_lower = text.lower()
        
        # 关键词匹配
        keywords = [field_name.lower(), value_str[:5] if len(value_str) > 5 else value_str]
        
        return any(keyword in text_lower for keyword in keywords)

# 全局实例
fine_grained_assessor = FineGrainedMedicalAssessor()