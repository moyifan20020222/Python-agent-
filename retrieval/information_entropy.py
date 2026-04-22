"""
信息熵计算模块 - 用于评估文本信息密度
"""

import math
import re
from typing import List, Dict, Any

import jieba
import numpy as np

def calculate_text_entropy(text: str) -> float:
    """
    计算文本信息熵（香农熵）
    基于字符频率分布，值越大表示信息越丰富
    """
    if not text:
        return 0.0
    
    # 预处理：去除空白字符，转换为字符列表
    chars = list(re.sub(r'\s+', '', text))
    if not chars:
        return 0.0
    
    # 计算字符频率
    char_freq = {}
    for char in chars:
        char_freq[char] = char_freq.get(char, 0) + 1
    
    # 计算概率分布
    total_chars = len(chars)
    probabilities = [freq / total_chars for freq in char_freq.values()]
    
    # 计算香农熵
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    
    # 归一化到0-1范围（中文文本最大熵约12.0）
    normalized_entropy = min(1.0, entropy / 12.0)
    
    return normalized_entropy


def calculate_word_entropy(text: str,
                                 use_stopwords: bool = False,
                                 stopwords_path: str = None) -> float:
    """
    使用 jieba 分词计算词级信息熵

    参数:
        text: 待计算的中文文本
        use_stopwords: 是否使用停用词过滤
        stopwords_path: 停用词文件路径

    返回:
        归一化的信息熵值 (0.0-1.0)
    """
    if not text or not text.strip():
        return 0.0

    # 预处理：去除多余空白字符
    text = re.sub(r'\s+', ' ', text.strip())

    # 1. 使用 jieba 分词
    if use_stopwords and stopwords_path:
        # 使用停用词过滤
        words = [word for word in jieba.lcut(text)
                 if word.strip() and len(word) > 1]  # 过滤单字
    else:
        # 不过滤停用词
        words = [word for word in jieba.lcut(text)
                 if word.strip()]

    if not words:
        return 0.0

    # 2. 计算词频
    word_freq = {}
    for word in words:
        # 可选：进一步过滤标点符号
        if re.match(r'^[^\w\u4e00-\u9fff]+$', word):
            continue
        word_freq[word] = word_freq.get(word, 0) + 1

    if not word_freq:
        return 0.0

    # 3. 计算信息熵
    total_words = sum(word_freq.values())
    entropy = 0.0

    for count in word_freq.values():
        p = count / total_words
        entropy -= p * math.log2(p)

    # 4. 归一化
    # 中文的最大熵理论值约为 13-15 bits (基于汉字数量)
    # 但词级熵会更低，约为 8-10 bits
    max_entropy_jieba = 10.0  # 经验值，可根据实际数据调整
    normalized_entropy = min(1.0, entropy / max_entropy_jieba)

    return normalized_entropy


def assess_content_quality(text: str, source: str, category: str) -> Dict[str, float]:
    """
    综合内容质量评估
    """
    # 1. 信息熵计算
    char_entropy = calculate_text_entropy(text)
    word_entropy = calculate_word_entropy(text)
    entropy_score = 0.3 * char_entropy + 0.7 * word_entropy
    
    # 2. 来源重要性
    source_weight = 1.0
    if source in ["医生", "专家", "论文", "指南"]:
        source_weight = 1.2
    elif source in ["网络", "自媒体", "用户录入"]:
        source_weight = 0.8
    
    # 3. 类别重要性
    category_weight = 1.0
    if category in ["药物", "综合"]:
        category_weight = 1.1
    elif category in ["饮食", "运动"]:
        category_weight = 0.9
    
    # 4. 文档大小影响
    length = len(text)
    size_factor = 1.0
    if length < 500:  # 小文档
        size_factor = 0.8
    elif length > 2000:  # 大文档
        size_factor = 1.2
    
    # 综合评分
    quality_score = entropy_score * source_weight * category_weight * size_factor
    quality_score = min(1.0, max(0.1, quality_score))
    
    return {
        "entropy_score": entropy_score,
        "source_weight": source_weight,
        "category_weight": category_weight,
        "size_factor": size_factor,
        "quality_score": quality_score,
        "text_length": length
    }