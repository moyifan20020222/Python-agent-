"""
LangGraph官方Callbacks监听器 - 性能监控专用
"""
from langchain_core.callbacks import BaseCallbackHandler
from datetime import datetime
from typing import Dict, Any, Optional
# from langgraph.graph import Graph
import sqlite3
from .token_counter import token_counter

class PerformanceCallback(BaseCallbackHandler):
    """性能监控回调监听器"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self.node_timings: Dict[str, float] = {}
        self.node_tokens: Dict[str, int] = {}
        self.start_times: Dict[str, datetime] = {}
        self.current_session_id: str = ""
    
    def on_chain_start(self, inputs: Dict[str, Any], **kwargs) -> None:
        """整个链开始时调用"""
        self.current_session_id = inputs.get("session_id", "unknown")
        self.node_timings.clear()
        self.node_tokens.clear()
        self.start_times.clear()
        print(f"🔄 链开始: session_id={self.current_session_id}")
    
    def on_node_start(self, node_name: str, state: Dict[str, Any]) -> None:
        """节点开始时调用"""
        self.start_times[node_name] = datetime.now()
        print(f"▶️ 节点开始: {node_name} (session: {self.current_session_id})")
    
    def on_node_end(self, node_name: str, state: Dict[str, Any], result: Dict[str, Any]) -> None:
        """节点结束时调用"""
        if node_name in self.start_times:
            duration_ms = (datetime.now() - self.start_times[node_name]).total_seconds() * 1000
            
            # 提取结果内容
            content = ""
            if isinstance(result, dict):
                # 尝试多种方式获取内容
                for key in ["response", "final_answer", "answer", "tool_result"]:
                    if key in result:
                        content = str(result[key])
                        break
                if not content and "messages" in result:
                    content = str(result["messages"][-1].content) if result["messages"] else ""
            
            # 计算Token
            tokens = token_counter.count(content)
            
            # 记录性能
            self.node_timings[node_name] = duration_ms
            self.node_tokens[node_name] = tokens
            
            print(f"⏹️ 节点结束: {node_name} | {duration_ms:.2f}ms | {tokens} tokens")
    
    def on_chain_end(self, outputs: Dict[str, Any]) -> None:
        """整个链结束时调用"""
        total_duration = sum(self.node_timings.values())
        total_tokens = sum(self.node_tokens.values())
        
        print(f"✅ 链结束: session_id={self.current_session_id}")
        print(f"📊 总耗时: {total_duration:.2f}ms | 总Token: {total_tokens}")
        
        # 可选：保存到数据库
        if self.db_path:
            self._save_to_database()
    
    def _save_to_database(self):
        """保存性能数据到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 主记录
                cursor.execute("""
                    INSERT INTO performance_logs 
                    (session_id, total_duration_ms, total_tokens, node_count, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    self.current_session_id,
                    sum(self.node_timings.values()),
                    sum(self.node_tokens.values()),
                    len(self.node_timings),
                    datetime.now().isoformat()
                ))
                
                # 节点详情
                for node_name, duration in self.node_timings.items():
                    tokens = self.node_tokens.get(node_name, 0)
                    cursor.execute("""
                        INSERT INTO node_details 
                        (session_id, node_name, duration_ms, tokens)
                        VALUES (?, ?, ?, ?)
                    """, (self.current_session_id, node_name, duration, tokens))
                
                conn.commit()
                print(f"💾 性能数据已保存到数据库")
                
        except Exception as e:
            print(f"❌ 保存性能数据失败: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            "session_id": self.current_session_id,
            "node_timings": self.node_timings.copy(),
            "node_tokens": self.node_tokens.copy(),
            "total_duration_ms": sum(self.node_timings.values()),
            "total_tokens": sum(self.node_tokens.values()),
            "node_count": len(self.node_timings),
            "timestamp": datetime.now().isoformat()
        }