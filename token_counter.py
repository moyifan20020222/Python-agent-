"""
专业级Token计数器 - 兼顾面试演示和生产环境
支持多种模型的精确Token计算
"""

import os
import re
from typing import Optional, Union
from datetime import datetime

# 缓存已加载的tokenizer，避免重复初始化
_tokenizers = {}

def count_tokens(text: str, model_name: Optional[str] = None) -> int:
    """
    精确计算文本Token数量
    支持多种模型，按优先级选择计算方法：
    1. tiktoken（OpenAI/Anthropic模型）
    2. HuggingFace Transformers（本地模型）
    3. 智能启发式算法（降级方案）
    
    Args:
        text: 要计算的文本
        model_name: 模型名称（用于选择合适的tokenizer）
    
    Returns:
        int: Token数量
    """
    if not text or not isinstance(text, str):
        return 0
    
    # 方法1: tiktoken（优先级最高）
    tokens = _try_tiktoken(text, model_name)
    if tokens is not None:
        return tokens
    
    # 方法2: HuggingFace Transformers
    tokens = _try_hf_tokenizer(text, model_name)
    if tokens is not None:
        return tokens
    
    # 方法3: 智能启发式算法（面试项目友好）
    return _heuristic_token_count(text)


def _try_tiktoken(text: str, model_name: Optional[str]) -> Optional[int]:
    """尝试使用tiktoken计算"""
    try:
        import tiktoken
        
        # 根据模型名选择编码器
        if model_name:
            model_name = model_name.lower()
            if "gpt" in model_name or "openai" in model_name:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            elif "claude" in model_name:
                encoding = tiktoken.encoding_for_model("cl100k_base")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    
    except ImportError:
        return None
    except Exception as e:
        # 日志记录但不中断
        _log_warning(f"tiktoken计算失败: {e}")
        return None


def _try_hf_tokenizer(text: str, model_name: Optional[str]) -> Optional[int]:
    """尝试使用HuggingFace tokenizer计算"""
    try:
        from transformers import AutoTokenizer
        
        # 使用预配置的模型路径
        if model_name and "bge" in model_name.lower():
            tokenizer_name = "BAAI/bge-m3"
        elif model_name and "m3e" in model_name.lower():
            tokenizer_name = "m3e-base"
        else:
            tokenizer_name = "BAAI/bge-m3"
        
        # 缓存tokenizer
        if tokenizer_name not in _tokenizers:
            _tokenizers[tokenizer_name] = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True,
                use_fast=True
            )
        
        tokenizer = _tokenizers[tokenizer_name]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    
    except ImportError:
        return None
    except Exception as e:
        _log_warning(f"HF tokenizer计算失败: {e}")
        return None


def _heuristic_token_count(text: str) -> int:
    """
    智能启发式Token计算（面试项目友好版）
    基于中文和英文字符的统计规律
    """
    if not text:
        return 0
    
    # 统计不同类型的字符
    chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    english_words = len(re.findall(r'\b\w+\b', text))  # 英文单词数
    english_chars = len([c for c in text if c.isalnum() and not ('\u4e00' <= c <= '\u9fff')])
    other_chars = len(text) - chinese_chars - english_chars
    
    # 经验公式（基于主流LLM的Token化规则）
    # 中文：2字符 ≈ 1 Token
    # 英文：1单词 ≈ 1 Token，或 4字符 ≈ 1 Token
    # 符号：2字符 ≈ 1 Token
    
    tokens_from_chinese = chinese_chars // 2
    tokens_from_english = max(english_words, english_chars // 4)
    tokens_from_other = other_chars // 2
    
    total_tokens = tokens_from_chinese + tokens_from_english + tokens_from_other
    return max(1, total_tokens)


def _log_warning(message: str):
    """警告日志（面试项目中静默）"""
    # 面试项目中可以注释掉这行
    # print(f"⚠️ Token计算降级: {message}")
    pass


def get_token_info(text: str, model_name: Optional[str] = None) -> dict:
    """
    获取详细的Token信息（用于性能监控）
    
    Returns:
        dict: 包含详细统计信息
    """
    tokens = count_tokens(text, model_name)
    
    # 字符统计
    char_count = len(text)
    chinese_count = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    english_count = len([c for c in text if c.isalnum() and not ('\u4e00' <= c <= '\u9fff')])
    
    return {
        "total_tokens": tokens,
        "character_count": char_count,
        "chinese_characters": chinese_count,
        "english_characters": english_count,
        "tokenization_method": "heuristic" if not _has_tiktoken() else "tiktoken",
        "timestamp": datetime.now().isoformat()
    }


def _has_tiktoken() -> bool:
    """检查是否安装了tiktoken"""
    try:
        import tiktoken
        return True
    except ImportError:
        return False


# 全局实例（用于性能监控）
token_counter = type('TokenCounter', (), {
    'count': lambda self, text, model_name=None: count_tokens(text, model_name),
    'info': lambda self, text, model_name=None: get_token_info(text, model_name)
})()

# 使用示例
if __name__ == "__main__":
    sample_text = "患者张三，男，65岁，因腰痛伴右下肢放射痛3月入院。2024-02-10行L4/5椎间盘切除术。"
    
    print(f"原文长度: {len(sample_text)} 字符")
    print(f"Token数量1: {token_counter.count(sample_text)}")

    print(f"Token数量: {count_tokens(sample_text)}")
    print(f"详细信息: {get_token_info(sample_text)}")