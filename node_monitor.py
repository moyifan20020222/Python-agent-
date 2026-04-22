# project/rehab_core/node_monitor.py
from datetime import datetime
from typing import Dict, Any


class NodeMonitor:
    """节点监控器 - 使用检查点存储"""

    def __init__(self):
        self.node_timings = {}  # {node_name: duration_ms}
        self.node_tokens = {}  # {node_name: tokens}
        self.start_times = {}  # {node_name: start_time}

    def start_node(self, node_name: str):
        """开始节点计时"""
        self.start_times[node_name] = datetime.now()

    def end_node(self, node_name: str, result_content: str = ""):
        """结束节点计时并记录Token"""
        if node_name in self.start_times:
            duration_ms = (datetime.now() - self.start_times[node_name]).total_seconds() * 1000

            # 计算Token（使用你的token_counter）
            from .token_counter import token_counter
            tokens = token_counter.count(result_content)

            self.node_timings[node_name] = duration_ms
            self.node_tokens[node_name] = tokens

    def get_summary(self) -> Dict[str, Any]:
        """获取节点执行摘要"""
        total_duration = sum(self.node_timings.values())
        total_tokens = sum(self.node_tokens.values())

        return {
            "node_timings": self.node_timings,
            "node_tokens": self.node_tokens,
            "total_duration_ms": total_duration,
            "total_tokens": total_tokens,
            "node_count": len(self.node_timings),
            "timestamp": datetime.now().isoformat()
        }

    def clear(self):
        """清空监控数据"""
        self.node_timings.clear()
        self.node_tokens.clear()
        self.start_times.clear()