"""
测试LangGraph Callbacks监听器
"""

from langgraph.graph import StateGraph, START, END
from typing import Dict, Any

class TestCallback:
    """测试回调监听器"""
    
    def __init__(self):
        self.calls = []
    
    def on_chain_start(self, inputs: Dict[str, Any]):
        self.calls.append("chain_start")
        print("✅ chain_start called")
    
    def on_node_start(self, node_name: str, state: Dict[str, Any]):
        self.calls.append(f"node_start_{node_name}")
        print(f"✅ node_start_{node_name} called")
    
    def on_node_end(self, node_name: str, state: Dict[str, Any], result: Dict[str, Any]):
        self.calls.append(f"node_end_{node_name}")
        print(f"✅ node_end_{node_name} called")
    
    def on_chain_end(self, outputs: Dict[str, Any]):
        self.calls.append("chain_end")
        print("✅ chain_end called")

# 创建简单图测试
def simple_node(state: Dict[str, Any]) -> Dict[str, Any]:
    return {"result": "test"}

graph = StateGraph(dict)
graph.add_node("test", simple_node)
graph.add_edge(START, "test")
graph.add_edge("test", END)

compiled_graph = graph.compile()

# 测试回调
callback = TestCallback()
result = compiled_graph.invoke(
    {"input": "test"},
    callbacks=[callback]  # 关键：这里传递回调
)

print(f"回调调用顺序: {callback.calls}")