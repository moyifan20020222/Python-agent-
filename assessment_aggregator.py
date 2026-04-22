"""
评估汇总器 - 将5个大类的LLM评估结果汇总为最终质量报告
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AssessmentAggregator:
    """评估结果汇总器"""
    
    def __init__(self):
        self.category_weights = {
            "基础信息": 0.2,
            "病史": 0.2,
            "常规检查": 0.15,
            "专科检查": 0.25,
            "诊断结果": 0.2
        }
    
    def aggregate_assessments(self, 
                            category_assessments: Dict[str, Dict]) -> Dict[str, Any]:
        """
        汇总5个大类的LLM评估结果
        参数：category_assessments = {
            "基础信息": {评估结果},
            "病史": {评估结果},
            ...
        }
        """
        if not category_assessments:
            return self._default_report()
        
        # 计算各维度分数
        overall_confidence = 0.0
        total_weight = 0.0
        
        # 按类别加权计算
        for category, assessment in category_assessments.items():
            weight = self.category_weights.get(category, 0.2)
            confidence = assessment["confidence_score"]
            
            overall_confidence += confidence * weight
            total_weight += weight
        
        # 归一化
        overall_confidence = overall_confidence / max(1.0, total_weight)
        
        # 收集关键问题
        critical_issues = []
        for category, assessment in category_assessments.items():
            issues = assessment.get("critical_issues", [])
            if issues:
                critical_issues.extend([f"[{category}] {issue}" for issue in issues])
        
        # 汇总详细信息
        detailed_summary = {}
        for category, assessment in category_assessments.items():
            detailed_summary[category] = {
                "confidence": assessment.get("confidence_score", 0.5),
                "field_count": assessment.get("total_fields_evaluated", 0),
                "high_confidence": assessment.get("high_confidence_fields", 0),
                "low_confidence": assessment.get("low_confidence_fields", 0)
            }
        
        return {
            "report_id": f"agg_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "overall_confidence": float(round(overall_confidence, 3)),
            "category_scores": detailed_summary,
            "critical_issues": critical_issues,
            "total_fields_evaluated": sum(
                ass.get("total_fields_evaluated", 0) 
                for ass in category_assessments.values()
            ),
            "high_confidence_fields": sum(
                ass.get("high_confidence_fields", 0) 
                for ass in category_assessments.values()
            ),
            "low_confidence_fields": sum(
                ass.get("low_confidence_fields", 0) 
                for ass in category_assessments.values()
            ),
            "quality_level": self._determine_quality_level(overall_confidence),
            "recommendations": self._generate_recommendations(critical_issues)
        }
    
    def _default_report(self) -> Dict[str, Any]:
        """默认报告"""
        return {
            "report_id": f"agg_default_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "overall_confidence": 0.5,
            "category_scores": {},
            "critical_issues": ["无评估数据"],
            "total_fields_evaluated": 0,
            "high_confidence_fields": 0,
            "low_confidence_fields": 0,
            "quality_level": "unknown",
            "recommendations": ["请提供各分类的LLM评估结果"]
        }
    
    def _determine_quality_level(self, confidence: float) -> str:
        """确定质量等级"""
        if confidence >= 0.9:
            return "优秀"
        elif confidence >= 0.8:
            return "良好"
        elif confidence >= 0.7:
            return "合格"
        elif confidence >= 0.6:
            return "需改进"
        else:
            return "不合格"
    
    def _generate_recommendations(self, critical_issues: List[str]) -> List[str]:
        """生成改进建议"""
        if not critical_issues:
            return ["提取质量良好，无需改进"]
        
        recommendations = []
        if any("诊断" in issue for issue in critical_issues):
            recommendations.append("加强诊断结果的准确性验证")
        if any("病史" in issue for issue in critical_issues):
            recommendations.append("完善病史信息的完整性检查")
        if any("检查" in issue for issue in critical_issues):
            recommendations.append("强化检查结果的数值准确性验证")
        
        if len(recommendations) == 0:
            recommendations.append("建议人工复核关键字段")
        
        return recommendations

# 全局实例
assessment_aggregator = AssessmentAggregator()
