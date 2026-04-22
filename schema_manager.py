"""
企业级Schema版本管理系统
- 支持JSON Schema版本演进
- 提供向后兼容的迁移能力
- 集成质量评估指标
"""

from typing import Dict, Any, Optional, List, Union, Callable

import numpy as np
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, validator
from datetime import datetime
import json
import logging
from project.prompts.schema import SCHEMA
from project.rag_agent.extractor import extract_section, merge_results
from project.rehab_core.assessment_aggregator import assessment_aggregator
from project.api.main import save_case_to_db
logger = logging.getLogger(__name__)

OPENAI_API_KEY = 'sk-d2cf853e524c4f5fb8c604906a781faa'
# 初始化模型
# llm_client = OpenAI(api_key='sk-d2cf853e524c4f5fb8c604906a781faa',
#                             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
# LLM_MODEL_NAME_small = "deepseek-r1-distill-llama-8b"
LLM = ChatOpenAI(
    model="deepseek-r1-distill-llama-8b",
    api_key=OPENAI_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

class SchemaVersion(BaseModel):
    """Schema版本定义"""
    version: str = Field(..., description="版本号，如 '1.0', '2.0'")
    name: str = Field(..., description="版本名称")
    description: str = Field(..., description="版本描述")
    created_at: datetime = Field(default_factory=datetime.now)
    is_current: bool = Field(default=False, description="是否为当前版本")
    compatibility: List[str] = Field(default_factory=list, description="兼容的旧版本")

class ExtractionResult(BaseModel):
    """标准化提取结果结构"""
    schema_version: str = Field(..., description="使用的schema版本")
    extracted_at: str = Field(..., description="提取时间戳")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="整体置信度")
    extraction_id: str = Field(..., description="提取任务ID")
    
    # 核心字段（强制）
    diagnosis: Dict[str, Any] = Field(..., description="诊断结果")
    surgery_date: Dict[str, Any] = Field(..., description="手术日期")
    rehab_stage: Dict[str, Any] = Field(..., description="康复阶段")
    
    # 可选字段（按版本扩展）
    contraindications: List[Dict[str, Any]] = Field(default_factory=list, description="禁忌症")
    icd10_code: str = Field("", description="ICD-10编码")
    snomed_ct_codes: List[str] = Field(default_factory=list, description="SNOMED CT编码")
    clinical_rules_violated: List[str] = Field(default_factory=list, description="违反的临床规则")
    
    @validator('confidence_score')
    def validate_confidence(cls, v):
        if v < 0 or v > 1:
            raise ValueError('confidence_score must be between 0 and 1')
        return v

class SchemaManager:
    """Schema版本管理器"""
    
    def __init__(self):
        self.versions: Dict[str, Dict] = {
            "1.0": {
                "name": "基础提取",
                "description": "基础字段提取",
                "fields": ["基础信息", "病史", "常规检查", "专科检查"],
                "required": ["基础信息", "病史", "常规检查", "专科检查"]
            },
            "2.0": {
                "name": "增强版",
                "description": "添加诊断结果，用于康复指南生成的参考",
                "fields": ["基础信息", "病史", "常规检查", "专科检查", "诊断结果"],
                "required": ["基础信息", "病史", "常规检查", "专科检查", "诊断结果"],
                "compatible_with": ["1.0"]
            },
            "3.0": {
                "name": "医疗专业版",
                "description": "添加关键词字段和提取内容限制，确保查询结果准确",
                "fields": ["基础信息", "病史", "常规检查", "专科检查", "诊断结果"],
                "required": ["基础信息", "病史", "常规检查", "专科检查", "诊断结果"],
                "compatible_with": ["2.0", "1.0"]
            }
        }
        self.current_version = "3.0"
    
    def get_schema(self, version: str = None) -> Dict:
        """获取指定版本的schema定义"""
        if version is None:
            version = self.current_version
        
        if version not in self.versions:
            raise ValueError(f"Schema version {version} not found")
        
        return self.versions[version]
    
    def validate_extraction(self, extraction: Dict[str, Any], version: str = None) -> Dict[str, Any]:
        """验证提取结果是否符合指定版本规范"""
        if version is None:
            version = self.current_version
        
        schema_def = self.get_schema(version)
        errors = []
        
        # 检查必需字段
        for field in schema_def.get("required", []):
            if field not in extraction:
                errors.append(f"Missing required field: {field}")

        # 检查字段类型
        # if "confidence_score" in extraction:
        #     if not isinstance(extraction["confidence_score"], (int, float)):
        #         errors.append("confidence_score must be numeric")
        #     elif extraction["confidence_score"] < 0 or extraction["confidence_score"] > 1:
        #         errors.append("confidence_score must be between 0 and 1")
        
        # 返回验证结果
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "version": version,
            "timestamp": datetime.now().isoformat()
        }
    
    def migrate_extraction(self, extraction: Dict[str, Any], target_version: str) -> Dict[str, Any]:
        """迁移提取结果到目标版本"""
        current_version = extraction.get("schema_version", "1.0")
        
        if current_version == target_version:
            return extraction
        
        logger.info(f"Migrating extraction from {current_version} to {target_version}")
        
        # 简单迁移逻辑（实际项目中应更复杂）
        migrated = extraction.copy()
        migrated["schema_version"] = target_version
        
        # 添加新字段（如果目标版本需要）
        target_schema = self.get_schema(target_version)
        
        if "icd10_code" in target_schema["fields"] and "icd10_code" not in migrated:
            migrated["icd10_code"] = ""
        
        if "contraindications" in target_schema["fields"] and "contraindications" not in migrated:
            migrated["contraindications"] = []
        
        return migrated

# 全局实例
schema_manager = SchemaManager()

# from project.rehab_core.llm_quality_assessor import PreciseLLMAssessor

# 全局LLM评估器实例（在main.py中初始化）
# llm_assessor = PreciseLLMAssessor()


class LLMConfidenceAssessor:
    """LLM自评估置信度评估器 - 与你的schema体系集成"""

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm = llm_client
        # 你的162个指标的字段定义（从你的schema中提取）
        # self.field_definitions = self._load_field_definitions(section)

    def get_section_fields(self, section: str) -> List[Dict]:
        """获取指定模块的字段定义"""
        return [f for f in SCHEMA["fields"] if f["section"] == section]

    def _load_field_definitions(self, section) -> str:
        """加载你的162个指标定义"""
        # 这里应该从你的实际schema中加载
        # 示例：只展示部分关键字段
        fields = self.get_section_fields(section)

        if fields == "专科检查":
            field_rules = "\n".join([
                f"- |{f['key']}|{f['rules']}|{f['default']}"
                for f in fields
            ])
        else:
            field_rules = "\n".join([
                f"- |{f['key']}| {f['rules']}|"
                for f in fields
            ])
        return field_rules

    def assess_confidence_batch(self, sections,
                                extraction: Dict,
                                text: str) -> List[Dict]:
        """
        批量LLM评估 - 企业级性能优化
        参数：
        - extractions: 提取结果列表
        - original_texts: 原始文本列表
        返回：每个提取结果的LLM评估报告
        """
        assessments = []
        for section in sections:
            try:
                # 构建评估请求
                assessment = self._build_assessment_request(section, extraction, text)

                # 调用LLM（使用你的现有LLM接口）
                if self.llm:
                    response = LLM.invoke([
                        {"role": "system", "content": "你是一个严谨的医疗数据抽取专家，只输出纯JSON。"},
                        {"role": "user", "content": assessment["prompt"]}
                    ])
                    assessment_result = self._parse_llm_response(response)
                # else:
                #         # 安全降级：基于规则的评估
                #     assessment_result = self._rule_based_assessment(extraction, text)

                assessments.append({
                        "llm_assessment": assessment_result,
                        "timestamp": datetime.now().isoformat(),
                        "status": "completed",
                        "section": section
                })
            except Exception as e:
                logger.error(f"Batch assessment error for item : {e}")
                assessments.append({
                    "llm_assessment": "LLM评估服务不可用",
                    "error": str(e),
                    "status": "failed",
                    "section": section
                })

        return assessments

    def _build_assessment_request(self, section, extraction: Dict, text: str) -> Dict:
        """构建LLM评估请求"""
        # 根据你的162个指标构建专业评估prompt
        critical_fields = self._load_field_definitions(section)

        prompt = f"""
                你是一名资深临床医学专家，请严格根据以下原则评估提取结果的质量：
                【评估要求】
                1. 仅基于提供的原始文本进行判断
                2. 对每个关键字段评估：
                   - 准确性：提取值是否在原文中有明确依据，且提取的指标信息是从对应的区域获取的。
                   - 完整性：是否遗漏了原文中的关键信息
                   - 合理性：提取结果是否符合每个字段的提取规则
                【关键字段】
                {critical_fields}
                【提取结果】
                {json.dumps(extraction, ensure_ascii=False, indent=2)}
                【原始文本】
                {text}  # 限制长度避免token超限
                请按以下JSON格式输出评估结果：
                {{
                  "field_assessments": {{
                    "field_name": {{
                      "confidence": 0.0,
                      "reasoning": "简要说明",
                      "evidence_span": [start, end],
                      "clinical_validity": true/false
                    }},
                    ...
                  }},
                  "overall_quality": "high/medium/low",
                  "critical_issues": ["问题1", "问题2"]
                }}
                """

        return {"prompt": prompt, "critical_fields": critical_fields}

    def _parse_llm_response(self, response: str) -> Dict:
        """解析LLM响应"""
        try:
            return json.loads(response)
        except:
            # 解析失败时返回默认结构
            return {
                "field_assessments": {},
                "overall_quality": "medium",
                "critical_issues": []
            }

    # def _rule_based_assessment(self, extraction: Dict, text: str) -> Dict:
    #     """基于规则的评估（LLM不可用时的降级方案）"""
    #     field_assessments = {}
    #
    #     for field_name, field_info in self.field_definitions.items():
    #         if field_name in extraction:
    #             # 简单规则：检查字段是否存在、长度是否合理
    #             value = extraction[field_name]
    #             confidence = 0.7
    #
    #             if isinstance(value, dict) and "value" in value:
    #                 content = value["value"]
    #                 if content and len(content) > 2:
    #                     confidence = 0.8 + min(0.2, len(content) / 50)
    #
    #             field_assessments[field_name] = {
    #                 "confidence": float(confidence),
    #                 "reasoning": "基于规则评估",
    #                 "evidence_span": [0, 0],
    #                 "clinical_validity": True
    #             }
    #
    #     return {
    #         "field_assessments": field_assessments,
    #         "overall_quality": "medium",
    #         "critical_issues": []
    #     }
    #
    # def _fallback_assessment(self, extraction: Dict) -> Dict:
    #     """安全降级评估"""
    #     return {
    #         "field_assessments": {
    #             k: {"confidence": 0.5, "reasoning": "LLM评估不可用", "evidence_span": [0, 0], "clinical_validity": True}
    #             for k in self.field_definitions.keys()},
    #         "overall_quality": "unknown",
    #         "critical_issues": ["LLM评估服务不可用"]
    #     }

llm_assessor = LLMConfidenceAssessor(LLM)

def extract_and_assess_medical_info(text: str, patient_id: int) -> dict:
    """
    医疗信息提取和评估主函数
    保持你原有的5段式提取流程
    """
    # 1. 分5个大类提取
    categories = ["基础信息", "病史", "常规检查", "专科检查", "诊断结果"]
    extracted_sections = {}
    llm_assessments = {}

    for category in categories:
        # 原有提取逻辑
        section_data = extract_section(text, category)
        extracted_sections[category] = section_data
        print("section_data", section_data)
        # 2. 对每个大类进行LLM评估
        try:
            # 这里调用你的LLM评估函数
            assessment = llm_assessor.assess_confidence_batch(categories, extracted_sections, text)
            llm_assessments[category] = assessment
            print("评估指标", assessment)
        except Exception as e:
            logger.warning(f"类别 {category} 评估失败: {e}")
            llm_assessments[category] = {
                "confidence_score": 0.7,
                "critical_issues": [f"评估失败: {str(e)}"],
                "total_fields_evaluated": 0
            }

    # 3. 合并结果
    merged_result = merge_results(list(extracted_sections.values()))

    # 4. 汇总评估报告
    final_assessment = assessment_aggregator.aggregate_assessments(llm_assessments)

    # 5. 添加到结果中
    merged_result["quality_assessment"] = final_assessment
    merged_result["confidence_score"] = final_assessment["overall_confidence"]

    return merged_result

# 使用示例
if __name__ == "__main__":
    # 示例数据
    patient_id = 22
    # 打开文件并读取所有内容
    with open('D:\\Desktop\\company\\Agent-rag\\agentic-rag-for-dummies\\7806738.txt', 'r', encoding='utf-8') as file:
        content = file.read()
        print(content)

    sample_data = content
    result = extract_and_assess_medical_info(sample_data, patient_id)
    save_case_to_db(result, patient_id, "髋部骨折")
    print(json.dumps(result.dict(), ensure_ascii=False, indent=2))

