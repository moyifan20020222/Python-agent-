"""
Prompt A/B测试系统 - 面试友好版
使用LLM-as-Judge方法评估不同Prompt版本
"""

from typing import List, Dict, Any, Optional
import json
import random
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PromptABTest:
    """Prompt A/B测试系统"""
    
    def __init__(self):
        # 测试数据集（200条标注数据）
        self.test_cases = self._load_test_data()
        
        # Prompt版本定义
        self.prompt_versions = {
            "v1_basic": {
                "name": "基础版",
                "description": "简单的自然语言描述",
                "prompt": "请从以下病历中提取信息：{text}"
            },
            "v2_structured": {
                "name": "结构化版",
                "description": "明确字段要求",
                "prompt": "请提取以下字段：患者姓名、年龄、诊断结果、手术日期。格式：JSON"
            },
            "v3_medical": {
                "name": "医学专业版",
                "description": "添加医学规则约束",
                "prompt": "请提取医疗信息，必须符合临床规范：{text}。输出格式：JSON，确保诊断结果有ICD编码"
            }
        }
    
    def _load_test_data(self) -> List[Dict[str, str]]:
        """加载测试数据集（200条）"""
        # 实际项目中从文件加载
        # 这里生成示例数据
        test_cases = []
        
        diseases = ["腰椎间盘突出症", "高血压", "糖尿病", "冠心病", "脑梗死"]
        categories = ["饮食", "运动", "心理", "药物", "综合"]
        
        for i in range(200):
            case_id = f"test_{i:03d}"
            disease = random.choice(diseases)
            category = random.choice(categories)
            
            # 生成示例文本
            text = f"患者张三，男，65岁，因{disease}入院。主诉：腰痛伴右下肢放射痛3月。2024-02-10行L4/5椎间盘切除术。术后第1天：卧床休息。诊断：{disease}。"
            
            test_cases.append({
                "case_id": case_id,
                "text": text,
                "disease": disease,
                "category": category,
                "ground_truth": {
                    "患者姓名": "张三",
                    "年龄": "65岁",
                    "诊断结果": disease,
                    "手术日期": "2024-02-10"
                }
            })
        
        return test_cases
    
    def evaluate_prompt_version(self, 
                              prompt_version: str,
                              llm_client: Any) -> Dict[str, Any]:
        """
        评估单个Prompt版本
        返回：准确率、完整性、临床合理性等指标
        """
        version_info = self.prompt_versions[prompt_version]
        results = []
        
        for case in self.test_cases[:50]:  # 测试前50条（资源有限）
            try:
                # 构建完整prompt
                full_prompt = version_info["prompt"].format(text=case["text"])
                
                # 调用LLM
                response = llm_client.generate(full_prompt, max_tokens=500)
                
                # LLM-as-Judge评估
                assessment = self._judge_response(response, case["ground_truth"], prompt_version)
                results.append(assessment)
                
            except Exception as e:
                logger.error(f"评估失败 {case['case_id']}: {e}")
                results.append({
                    "case_id": case["case_id"],
                    "accuracy": 0.0,
                    "completeness": 0.0,
                    "clinical_validity": 0.0,
                    "error": str(e)
                })
        
        # 计算统计指标
        accuracy_scores = [r.get("accuracy", 0.0) for r in results]
        completeness_scores = [r.get("completeness", 0.0) for r in results]
        clinical_scores = [r.get("clinical_validity", 0.0) for r in results]
        
        return {
            "version": prompt_version,
            "name": version_info["name"],
            "description": version_info["description"],
            "total_cases": len(results),
            "accuracy_avg": sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0,
            "completeness_avg": sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0,
            "clinical_validity_avg": sum(clinical_scores) / len(clinical_scores) if clinical_scores else 0.0,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _judge_response(self, response: str, ground_truth: Dict[str, str], version: str) -> Dict[str, Any]:
        """
        LLM-as-Judge评估
        评分标准：
        - 准确性：提取值是否与ground_truth一致
        - 完整性：是否提取了所有关键字段
        - 临床合理性：是否符合医学规范
        """
        # 简化版：基于关键词匹配
        accuracy = 0.0
        completeness = 0.0
        clinical_validity = 1.0
        
        # 关键字段检查
        key_fields = ["患者姓名", "年龄", "诊断结果", "手术日期"]
        extracted_fields = {}
        
        # 简单解析（实际项目中用LLM判断）
        for field in key_fields:
            if field in response:
                extracted_fields[field] = "extracted"
        
        # 准确性计算
        correct_count = 0
        for field in key_fields:
            if field in ground_truth and field in extracted_fields:
                correct_count += 1
        accuracy = correct_count / len(key_fields)
        
        # 完整性计算
        completeness = len(extracted_fields) / len(key_fields)
        
        # 临床合理性（简化）
        if "诊断结果" in extracted_fields and "手术日期" in extracted_fields:
            clinical_validity = 0.9
        elif "诊断结果" in extracted_fields:
            clinical_validity = 0.8
        
        return {
            "case_id": "test_case",
            "accuracy": accuracy,
            "completeness": completeness,
            "clinical_validity": clinical_validity,
            "response_preview": response[:100],
            "version": version
        }
    
    def run_ab_test(self, llm_client: Any) -> Dict[str, Any]:
        """运行A/B测试"""
        print("🚀 开始Prompt A/B测试...")
        
        results = {}
        
        for version in self.prompt_versions.keys():
            print(f"📝 测试版本: {self.prompt_versions[version]['name']}")
            results[version] = self.evaluate_prompt_version(version, llm_client)
        
        # 生成汇总报告
        summary = self._generate_summary(results)
        
        return {
            "summary": summary,
            "detailed_results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_summary(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """生成A/B测试汇总报告"""
        versions = list(results.keys())
        
        # 提取指标
        accuracy_scores = [results[v]["accuracy_avg"] for v in versions]
        completeness_scores = [results[v]["completeness_avg"] for v in versions]
        clinical_scores = [results[v]["clinical_validity_avg"] for v in versions]
        
        # 找出最佳版本
        best_idx = max(range(len(versions)), key=lambda i: (
            accuracy_scores[i] * 0.4 + 
            completeness_scores[i] * 0.3 + 
            clinical_scores[i] * 0.3
        ))
        
        return {
            "best_version": versions[best_idx],
            "best_score": accuracy_scores[best_idx],
            "metrics": {
                "accuracy": {
                    v: results[v]["accuracy_avg"] for v in versions
                },
                "completeness": {
                    v: results[v]["completeness_avg"] for v in versions
                },
                "clinical_validity": {
                    v: results[v]["clinical_validity_avg"] for v in versions
                }
            },
            "recommendation": f"推荐使用 {self.prompt_versions[versions[best_idx]]['name']} 版本"
        }

# 全局实例
prompt_ab_tester = PromptABTest()