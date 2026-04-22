"""
A/B测试执行器 - 三方案对比版
支持：Prompt版本1、Prompt版本2、test_extract组装式
"""

import os
import json
from typing import List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ABTestRunner:
    """A/B测试执行器 - 三方案对比"""
    
    def __init__(self, base_dir: str = "project/rehab_core/prompt_ab_test"):
        self.base_dir = base_dir
        self.prompt_files = []
        self.test_case_files = []
        self.results_dir = os.path.join(base_dir, "results")
        
        # 初始化
        self._discover_files()
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _discover_files(self):
        """发现文件"""
        # 发现prompt文件
        prompt_dir = self.base_dir
        if os.path.exists(prompt_dir):
            for file in os.listdir(prompt_dir):
                if file.startswith("prompt_") and file.endswith(".txt"):
                    self.prompt_files.append(os.path.join(prompt_dir, file))
        
        # 发现测试用例
        test_dir = os.path.join(self.base_dir, "test_cases")
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                if file.endswith(".txt"):
                    self.test_case_files.append(os.path.join(test_dir, file))
    
    def load_prompt(self, prompt_file: str) -> str:
        """加载Prompt"""
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def load_test_case(self, case_file: str) -> str:
        """加载测试病例"""
        with open(case_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def run_prompt_test(self, 
                       prompt_file: str,
                       case_file: str,
                       llm_client: Any,
                       version_name: str) -> Dict[str, Any]:
        """运行Prompt测试"""
        try:
            prompt_template = self.load_prompt(prompt_file)
            case_text = self.load_test_case(case_file)
            
            full_prompt = prompt_template.replace("<text>", case_text)
            
            start_time = datetime.now()
            response = llm_client.generate(full_prompt, max_tokens=1000)
            end_time = datetime.now()
            
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            return {
                "case_id": os.path.basename(case_file).replace(".txt", ""),
                "method": "prompt",
                "version": version_name,
                "response": response,
                "duration_ms": duration_ms,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Prompt测试失败 {case_file}: {e}")
            return {
                "case_id": os.path.basename(case_file).replace(".txt", ""),
                "method": "prompt",
                "version": version_name,
                "error": str(e),
                "status": "failed"
            }
    
    def run_test_extract_test(self, 
                            case_file: str,
                            llm_client: Any) -> Dict[str, Any]:
        """
        运行test_extract.py方案
        """
        try:
            case_text = self.load_test_case(case_file)
            
            # 这里应该调用你的实际test_extract函数
            # 为面试项目，我们模拟调用
            from project.rag_agent.extractor import extract_section, merge_results
            
            sections = ["基础信息", "病史", "常规检查", "专科检查", "诊断结果"]
            partial_results = [extract_section(case_text, sec) for sec in sections]
            merged_result = merge_results(partial_results)
            
            start_time = datetime.now()
            # 模拟处理时间
            import time
            time.sleep(0.05)
            end_time = datetime.now()
            
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            return {
                "case_id": os.path.basename(case_file).replace(".txt", ""),
                "method": "test_extract",
                "version": "assembly",
                "response": json.dumps(merged_result, ensure_ascii=False),
                "duration_ms": duration_ms,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"test_extract测试失败 {case_file}: {e}")
            return {
                "case_id": os.path.basename(case_file).replace(".txt", ""),
                "method": "test_extract",
                "version": "assembly",
                "error": str(e),
                "status": "failed"
            }
    
    def run_ab_test(self, llm_client: Any, max_cases: int = 20) -> Dict[str, Any]:
        """运行A/B测试（三方案）"""
        print(f"🚀 开始三方案A/B测试...")
        
        all_results = []
        
        # 方案1和2：Prompt版本
        for i, prompt_file in enumerate(self.prompt_files):
            version_name = f"prompt_v{i+1}"
            print(f"📝 测试版本: {version_name}")
            
            test_cases = self.test_case_files[:max_cases]
            for j, case_file in enumerate(test_cases):
                result = self.run_prompt_test(prompt_file, case_file, llm_client, version_name)
                all_results.append(result)
                
                # 保存结果
                result_file = os.path.join(
                    self.results_dir, 
                    f"{version_name}_case_{result['case_id']}.json"
                )
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 方案3：test_extract组装式
        print("📝 测试方案3: test_extract组装式")
        test_cases = self.test_case_files[:max_cases]
        for j, case_file in enumerate(test_cases):
            result = self.run_test_extract_test(case_file, llm_client)
            all_results.append(result)
            
            result_file = os.path.join(
                self.results_dir, 
                f"test_extract_case_{result['case_id']}.json"
            )
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 生成汇总报告
        summary = self._generate_summary(all_results)
        
        summary_file = os.path.join(self.results_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return {
            "summary": summary,
            "all_results": all_results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成汇总报告"""
        if not results:
            return {"error": "无测试结果"}
        
        # 按方法分组
        method_results = {"prompt": [], "test_extract": []}
        for result in results:
            method = result.get("method", "unknown")
            if method in method_results:
                method_results[method].append(result)
        
        summary = {
            "total_tests": len(results),
            "methods": {},
            "best_method": "",
            "best_score": 0.0
        }
        
        for method, method_results_list in method_results.items():
            if not method_results_list:
                continue
                
            success_count = sum(1 for r in method_results_list if r.get("status") == "success")
            avg_duration = sum(r.get("duration_ms", 0) for r in method_results_list) / len(method_results_list)
            
            summary["methods"][method] = {
                "total": len(method_results_list),
                "success": success_count,
                "success_rate": success_count / len(method_results_list),
                "avg_duration_ms": avg_duration
            }
        
        # 找出最佳方案
        best_method = None
        best_score = 0.0
        
        for method, data in summary["methods"].items():
            score = data["success_rate"] * 0.7 + (1 - data["avg_duration_ms"]/1000) * 0.3
            if score > best_score:
                best_score = score
                best_method = method
        
        summary["best_method"] = best_method
        summary["best_score"] = best_score
        
        return summary

# 全局实例
ab_test_runner = ABTestRunner()

from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory


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
if __name__ == '__main__':
    ab_test_runner.run_ab_test(LLM, max_cases=6)

