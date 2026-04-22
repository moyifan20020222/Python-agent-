"""
自定义性能监控器 - 修复版（Token统计正确）
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
import sqlite3

from langchain_core.messages.utils import count_tokens_approximately

from .token_counter import token_counter
from project.rehab_core.config import SQLITE_DB_PATH

class CustomPerformanceMonitor:
    """自定义性能监控器 - 修复版"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self.node_timings: Dict[str, float] = {}
        self.node_tokens: Dict[str, int] = {}  # 确保是int类型
        self.start_times: Dict[str, datetime] = {}
        self.current_session_id: str = ""
        self.total_duration: float = 0.0
        self.total_tokens: int = 0
    
    def start_node(self, node_name: str, session_id: str = ""):
        """开始节点计时"""
        self.current_session_id = session_id
        self.start_times[node_name] = datetime.now()
    
    def end_node(self, node_name: str, result_content: str = ""):
        """结束节点计时并记录Token（修复版）"""
        if node_name in self.start_times:
            duration_ms = (datetime.now() - self.start_times[node_name]).total_seconds() * 1000
            
            # 计算Token（安全版本）
            # try:
            #     tokens = token_counter.count(result_content)
            #     # 确保tokens是数字
            #     if tokens is None or not isinstance(tokens, (int, float)):
            #         tokens = len(result_content) // 4 if result_content else 0
            # except Exception as e:
            #     # 安全降级：简单计算
            #     tokens = len(result_content) // 4 if result_content else 0
            #
            # # 确保tokens是整数
            # tokens = int(tokens) if tokens is not None else 0
            #
            tokens = 0

            # 方法1：直接处理 AIMessage 的 content
            if hasattr(result_content, 'content'):
                content = result_content.content

                # 处理不同类型的 content
                if isinstance(content, str):
                    # 简单估算：1个token ≈ 4个字符（英文）
                    tokens = len(content) // 4
                elif isinstance(content, list):
                    # 处理多模态内容（如 [{'type': 'text', 'text': '...'}, ...]）
                    text_content = ""
                    for item in content:
                        if isinstance(item, str):
                            text_content += item
                        elif isinstance(item, dict):
                            # 处理字典格式的内容
                            if 'text' in item:
                                text_content += str(item['text'])
                            elif 'content' in item:
                                text_content += str(item['content'])
                        elif hasattr(item, 'text'):
                            # 处理有 text 属性的对象
                            text_content += str(item.text)

                    if text_content:
                        tokens = len(text_content) // 4

            # 方法2：如果 result 有 usage_metadata，使用它
            elif hasattr(result_content, 'usage_metadata') and result_content.usage_metadata:
                usage = result_content.usage_metadata
                if isinstance(usage, dict):
                    if 'total_tokens' in usage:
                        tokens = usage['total_tokens']
                    elif 'input_tokens' in usage and 'output_tokens' in usage:
                        tokens = usage.get('input_tokens', 0) + usage.get('output_tokens', 0)

            # 计算Token（简单算法）
            # chinese_chars = len([c for c in result_content if '\u4e00' <= c <= '\u9fff'])
            # english_chars = len([c for c in result_content if c.isalnum() and not ('\u4e00' <= c <= '\u9fff')])
            # tokens = (chinese_chars // 4) + (english_chars // 1)
            # tokens = max(1, tokens)

            # 记录（现在肯定是数字）
            self.node_timings[node_name] = duration_ms
            self.node_tokens[node_name] = tokens
            
            # 累计
            self.total_duration += duration_ms
            self.total_tokens += tokens
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            "session_id": self.current_session_id,
            "node_timings": self.node_timings.copy(),
            "node_tokens": self.node_tokens.copy(),  # ✅ 现在都是int
            "total_duration_ms": self.total_duration,
            "total_tokens": self.total_tokens,
            "node_count": len(self.node_timings),
            "timestamp": datetime.now().isoformat()
        }
    
    def clear(self):
        """清空数据"""
        self.node_timings.clear()
        self.node_tokens.clear()
        self.start_times.clear()
        self.total_duration = 0.0
        self.total_tokens = 0
    
    def save_to_database(self):
        """保存性能数据到数据库"""
        if not self.db_path:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        total_duration_ms REAL NOT NULL,
                        total_tokens INTEGER NOT NULL,
                        node_count INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS node_details (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        node_name TEXT NOT NULL,
                        duration_ms REAL NOT NULL,
                        tokens INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 插入主记录
                cursor.execute("""
                    INSERT INTO performance_logs 
                    (session_id, total_duration_ms, total_tokens, node_count)
                    VALUES (?, ?, ?, ?)
                """, (
                    self.current_session_id,
                    self.total_duration,
                    self.total_tokens,
                    len(self.node_timings)
                ))
                
                # 插入节点详情
                for node_name, duration in self.node_timings.items():
                    tokens = self.node_tokens.get(node_name, 0)  # ✅ 现在肯定是int
                    cursor.execute("""
                        INSERT INTO node_details 
                        (session_id, node_name, duration_ms, tokens)
                        VALUES (?, ?, ?, ?)
                    """, (self.current_session_id, node_name, duration, tokens))
                
                conn.commit()
                print(f"✅ 性能数据已保存到数据库: session_id={self.current_session_id}")
                
        except Exception as e:
            print(f"❌ 保存性能数据失败: {e}")

# 全局实例
custom_performance_monitor = CustomPerformanceMonitor(db_path=SQLITE_DB_PATH)