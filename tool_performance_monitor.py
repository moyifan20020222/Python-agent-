"""
工具级性能监控器 - 专门监控每个工具的调用时间
"""

from datetime import datetime
from typing import Dict, Any, Optional
import sqlite3

class ToolPerformanceMonitor:
    """工具级性能监控器"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self.tool_timings: Dict[str, float] = {}  # tool_name -> duration_ms
        self.tool_tokens: Dict[str, int] = {}    # tool_name -> tokens
        self.start_times: Dict[str, datetime] = {}
        self.current_session_id: str = ""
        self.total_duration: float = 0.0
        self.total_tokens: int = 0
    
    def start_tool(self, tool_name: str, session_id: str = ""):
        """开始工具调用计时"""
        self.current_session_id = session_id
        self.start_times[tool_name] = datetime.now()
    
    def end_tool(self, tool_name: str, result_content: str = ""):
        """结束工具调用计时并记录Token"""
        if tool_name in self.start_times:
            duration_ms = (datetime.now() - self.start_times[tool_name]).total_seconds() * 1000
            
            # 计算Token（简单算法）
            chinese_chars = len([c for c in result_content if '\u4e00' <= c <= '\u9fff'])
            english_chars = len([c for c in result_content if c.isalnum() and not ('\u4e00' <= c <= '\u9fff')])
            tokens = (chinese_chars // 4) + (english_chars // 1)
            tokens = max(1, tokens)
            
            # 记录
            self.tool_timings[tool_name] = duration_ms
            self.tool_tokens[tool_name] = tokens
            
            # 累计
            self.total_duration += duration_ms
            self.total_tokens += tokens
    
    def get_summary(self) -> Dict[str, Any]:
        """获取工具性能摘要"""
        return {
            "session_id": self.current_session_id,
            "tool_timings": self.tool_timings.copy(),
            "tool_tokens": self.tool_tokens.copy(),
            "total_duration_ms": self.total_duration,
            "total_tokens": self.total_tokens,
            "tool_count": len(self.tool_timings),
            "timestamp": datetime.now().isoformat()
        }
    
    def clear(self):
        """清空数据"""
        self.tool_timings.clear()
        self.tool_tokens.clear()
        self.start_times.clear()
        self.total_duration = 0.0
        self.total_tokens = 0
    
    def save_to_database(self):
        """保存工具性能数据到数据库"""
        if not self.db_path:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tool_performance_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        total_duration_ms REAL NOT NULL,
                        total_tokens INTEGER NOT NULL,
                        tool_count INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tool_details (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        duration_ms REAL NOT NULL,
                        tokens INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 插入主记录
                cursor.execute("""
                    INSERT INTO tool_performance_logs 
                    (session_id, total_duration_ms, total_tokens, tool_count)
                    VALUES (?, ?, ?, ?)
                """, (
                    self.current_session_id,
                    self.total_duration,
                    self.total_tokens,
                    len(self.tool_timings)
                ))
                
                # 插入工具详情
                for tool_name, duration in self.tool_timings.items():
                    tokens = self.tool_tokens.get(tool_name, 0)
                    cursor.execute("""
                        INSERT INTO tool_details 
                        (session_id, tool_name, duration_ms, tokens)
                        VALUES (?, ?, ?, ?)
                    """, (self.current_session_id, tool_name, duration, tokens))
                
                conn.commit()
                print(f"✅ 工具性能数据已保存: session_id={self.current_session_id}")
                
        except Exception as e:
            print(f"❌ 保存工具性能数据失败: {e}")

# 全局实例
tool_performance_monitor = ToolPerformanceMonitor(
    db_path="D:/Desktop/company/Agent-rag/agentic-rag-for-dummies/project/db/rehab.db"
)