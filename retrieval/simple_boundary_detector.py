"""
轻量级语义边界检测器 - 医学领域专用
"""

import re
from typing import List

class SimpleMedicalBoundaryDetector:
    """轻量级医学语义边界检测器"""
    
    def __init__(self):
        # 医学术语边界（简化版）
        self.boundary_terms = [
            # 时间阶段
            "术后第", "入院第", "住院第", "急性期", "亚急性期", "恢复期",
            # 临床决策
            "诊断依据", "鉴别诊断", "治疗原则", "推荐方案", "注意事项", "禁忌症",
            # 症状描述
            "主诉", "现病史", "既往史", "体格检查", "辅助检查",
            # 手术操作
            "手术步骤", "操作流程", "技术要点", "第一步", "第二步", "第三步"
        ]
    
    def detect_boundaries(self, text: str) -> List[int]:
        """检测文本中的语义边界位置"""
        boundaries = set()
        
        # 检测关键词边界
        for term in self.boundary_terms:
            start = 0
            while True:
                pos = text.find(term, start)
                if pos == -1:
                    break
                boundaries.add(pos)
                boundaries.add(pos + len(term))
                start = pos + 1
        
        # 添加段落边界
        paragraph_breaks = [i for i, char in enumerate(text) if char == '\n' and i > 0 and text[i-1] == '\n']
        boundaries.update(paragraph_breaks)
        
        return sorted(list(boundaries))