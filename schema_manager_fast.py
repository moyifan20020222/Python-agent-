"""
快速Schema测试模块 - 一次性执行
"""

from .schema_manager import schema_manager, create_extraction_result
from typing import Dict, Any
import time

def quick_schema_test(raw_data: Dict[str, Any], original_text: str = "") -> Dict[str, Any]:
    """
    快速Schema测试 - 一次性执行
    不循环，直接返回结果
    """
    start_time = time.time()
    
    try:
        # 直接创建提取结果（一次调用）
        result = create_extraction_result(
            raw_data=raw_data,
            original_text=original_text,
            version="3.0"
        )
        
        end_time = time.time()
        
        return {
            "extraction_result": result.dict(),
            "duration_ms": (end_time - start_time) * 1000,
            "status": "success"
        }
    except Exception as e:
        end_time = time.time()
        return {
            "error": str(e),
            "duration_ms": (end_time - start_time) * 1000,
            "status": "failed"
        }

# 示例使用
if __name__ == "__main__":
    sample_data = {
        "患者姓名": {"value": "张三", "confidence": 0.98},
        "年龄": {"value": "65岁", "confidence": 0.99},
        "诊断结果": {"value": "腰椎间盘突出症", "confidence": 0.95},
        "手术日期": {"value": "2024-02-10", "confidence": 0.99}
    }
    
    result = quick_schema_test(sample_data, "患者张三，男，65岁...")
    print(f"✅ 快速Schema测试完成: {result['duration_ms']:.2f}ms")