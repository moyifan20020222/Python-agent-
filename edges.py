from typing import Literal
from project.rehab_core.state import AgentState


def route_after_decision(state: AgentState) -> str:
    """决策节点后的路由"""
    return state.get("next_action", "orchestrator")


def route_after_question(state: AgentState) -> str:
    """提问节点后的路由"""
    # 提问后应该由doctor_speaker输出问题
    return "doctor_speaker"


# 保留你原有的路由函数
def route_start(state: AgentState) -> str:
    """现在入口改为intent_analysis，这个函数可能不再需要"""
    return "intent_analysis"


def route_after_orchestrator_call(state: AgentState) -> Literal["tools", "fallback_response", "collect_answer"]:
    if state["iteration_count"] >= 5 or state["tool_call_count"] > 5:
        return "fallback_response"
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "doctor_speaker"
    # if not getattr(last_msg, "tool_calls", []):
    #     return "collect_answer"
    # return "tools"


def should_compress_context(state: AgentState) -> str:
    current_tokens = len(str(state["messages"])) + len(state["context_summary"])
    if current_tokens > 30000:
        return "compress_context"
    return "orchestrator"


def route_start(state: AgentState):
    """根据动作判断是去聊天还是去结案"""
    if state.get("action") == "finalize":
        return "generate_final_plan"
    elif state.get("action") == "Recheck":
        return
    # return "orchestrator"
    # 多添加一步自查询功能
    return "query_analyzer"
