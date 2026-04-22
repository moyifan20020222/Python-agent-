"""
快速测试模块 - 一次性执行，不循环5次
"""

from project.rag_agent.extractor import extract_section, merge_results
from typing import List, Dict, Any
import time

def quick_test_extract(text: str, sections: List[str] = None) -> Dict[str, Any]:
    """
    快速提取测试 - 一次性执行
    不循环，直接返回结果
    """
    if sections is None:
        sections = ["基础信息", "病史", "常规检查", "专科检查", "诊断结果"]
    
    # 直接执行一次
    start_time = time.time()
    
    partial_results = [extract_section(text, sec) for sec in sections]
    result = merge_results(partial_results)
    
    end_time = time.time()
    
    return {
        "result": result,
        "duration_ms": (end_time - start_time) * 1000,
        "sections_processed": len(sections),
        "status": "success"
    }

# 示例使用
if __name__ == "__main__":
    sample_text = "患者张三，男，65岁，因腰痛伴右下肢放射痛3月入院。2024-02-10行L4/5椎间盘切除术。术后第1天：卧床休息。诊断：腰椎间盘突出症。"
    
    result = quick_test_extract(sample_text)
    print(f"✅ 快速测试完成: {result['duration_ms']:.2f}ms")
    print(f"结果: {result['result']}")