# 在状态定义中添加会话相关字段
from typing import List, Annotated, Set, Optional, Dict, Any, TypedDict, Literal
from langgraph.graph import MessagesState
import operator
from datetime import datetime
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired


def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    if new and any(item.get('__reset__') for item in new):
        return []
    return existing + new


def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    return a | b


class State(MessagesState):
    """State for main agent graph"""
    questionIsClear: bool = False
    conversation_summary: str = ""
    originalQuery: str = ""
    rewrittenQuestions: List[str] = []
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []

    # 新增：会话管理字段
    session_id: str = ""  # OpenClaw提供的会话ID
    external_session_data: Optional[Dict] = None  # 外部会话数据
    user_context: Dict[str, Any] = {}  # 用户上下文
    created_at: str = ""  # 创建时间
    last_updated: str = ""  # 最后更新时间

    # 新增：检索历史
    search_history: Annotated[List[Dict], accumulate_or_reset] = []
    retrieved_parents: Annotated[Set[str], set_union] = set()


class AgentState(TypedDict):
    """State for individual agent subgraph"""
    # question: str = ""
    # question_index: int = 0
    # context_summary: str = ""
    # retrieval_keys: Annotated[Set[str], set_union] = set()
    # final_answer: str = ""
    # agent_answers: List[dict] = []
    # tool_call_count: Annotated[int, operator.add] = 0
    # iteration_count: Annotated[int, operator.add] = 0
    # context_summary: str      # 当前对话的压缩摘要（短期）
    # past_rehab_plans: str     # 历史康复方案（长期） <--- 新增
    # action: str  # 用于指示当前对话是否结束，是否可以存储成会话信息了
    # # 从State继承的会话相关字段
    # session_id: str = ""
    # user_context: Dict[str, Any] = {}
    # search_history: List[Dict] = []
    # retrieved_parents: Set[str] = set()

    # 短期记忆：LangGraph 自动维护，依靠 add_messages 合并
    messages: Annotated[List[BaseMessage], add_messages]

    # 外部输入变量
    question: str  # 用于记录外部调用输入的信息
    session_id: str  # 用于 LangGraph Checkpointer 的 thread_id (单次就诊周期)
    patient_id: str  # 用于查询长期记忆的 患者ID (跟随用户终生)
    action: str  # "chat" 或 "finalize"
    # # 内部对话流转需要的信息
    user_context: Dict[str, Any]
    # dialogue_phase: Literal[
    #     "initial_greeting",  # 初始问候
    #     "symptom_collection",  # 症状收集
    #     "clarification",  # 澄清细节
    #     "analysis",  # 分析诊断
    #     "treatment_advice",  # 治疗建议
    #     "follow_up_planning",  # 随访规划
    #     "closing"  # 结束对话
    # ]
    # 3. 医疗信息（结构化）
    # 新增：对话流程控制字段
    dialogue_phase: NotRequired[str]  # 对话阶段
    next_action: NotRequired[str]  # 下一步行动
    has_existing_plan: NotRequired[bool]  # 是否有现有康复指南
    pending_clarifications: NotRequired[List[str]]  # 待澄清问题
    current_symptoms: NotRequired[List[Dict]]  # 当前症状
    differential_diagnosis: NotRequired[List[str]]  # 鉴别诊断
    treatment_plan: NotRequired[str]  # 治疗计划
    intent_result: NotRequired[Dict]  # 意图分析结果

    # # 4. 控制状态
    # next_action: Literal[
    #     "ask_question",  # 需要提问
    #     "analyze_symptoms",  # 需要分析
    #     "give_advice",  # 给出建议
    #     "wait_for_input",  # 等待输入
    #     "end_conversation"  # 结束对话
    # ]

    # 内部流转变量
    asked_questions: List[Any]  # 记录由LLM给出的基于病例的询问
    context_summary: str  # 当前会话的压缩摘要
    current_case_plan: str  # 【新增】当前选中病例已经存在的康复方案（如果是复诊会有值）
    other_historical_plans: str  # 【修改】患者其他部位/其他时间的病历方案（仅在最终结案时参考）
    current_illness: str  # 记录当前选择的病例信息
    final_answer: str  # 最终输出的回答或报告
    iteration_count: int  # 记录子图中的循环运行次数
    tool_call_count: int  # 记录子图中使用的工具，也就是查询父chunk的次数
    # 💡 新增：专门用于存放当前会话的检索过滤器
    search_filters: Dict[str, Any]
    # 记录提取的原始信息和提纯信息，避免过多文档信息丢入LLM的上下文中
    raw_retrieved_docs: str
    retrieved_docs: str
    raw_retrieved_faq: str
    retrieved_faq: str

