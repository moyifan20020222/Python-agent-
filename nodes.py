# from langchain_ollama import ChatOllama
import tiktoken
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, RemoveMessage
from datetime import datetime

from langgraph.types import Command

# from session_manager import SessionManager
from project.rehab_core.PatientRecordManager import patient_db
from project.rehab_core.state import State, AgentState
from project.rehab_core.tools import ToolFactory
from project.rehab_core.prompts import get_context_compression_prompt, get_orchestrator_prompt, \
    generate_final_plan_prompt, get_Intent_prompt, get_self_query_parser_prompt, get_fallback_response_prompt
import json
import sqlite3
# ✅ 真实 LLM 实例
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from typing import Set, Dict, Literal
from project.rehab_core.config import BASE_TOKEN_THRESHOLD, TOKEN_GROWTH_FACTOR
from project.rehab_core.performance_monitor import custom_performance_monitor

OPENAI_API_KEY = 'sk-d2cf853e524c4f5fb8c604906a781faa'
# 初始化模型
# llm_client = OpenAI(api_key='sk-d2cf853e524c4f5fb8c604906a781faa',
#                             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
# LLM_MODEL_NAME_small = "deepseek-r1-distill-llama-8b"
LLM = ChatOpenAI(
    model="deepseek-r1-distill-llama-8b",
    api_key=OPENAI_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MODEL_PATH = "D:/Desktop/company/Agent-rag/agentic-rag-for-dummies/models/Hu0922/BGE_Medical"
DB_PATH = "D:\\Desktop\\company\\Agent-rag\\agentic-rag-for-dummies\\project\\db\\rehab.db"
CHROMA_PATH = "D:\\Desktop\\company\\Agent-rag\\agentic-rag-for-dummies\\project\\rehab_core\\data\\chroma_db"


# tool_factory = ToolFactory(MODEL_PATH, DB_PATH)
# llm_with_tools = LLM.bind_tools([tool_factory.search_rehab_guidelines, tool_factory.get_case_data])

def estimate_context_tokens(messages: list) -> int:
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoding.encode(str(msg.content))) for msg in messages if hasattr(msg, 'content') and msg.content)


#
# def initialize_session(state: State) -> Dict:
#     """初始化会话"""
#     session_id = state.get("session_id", "")
#
#     if not session_id:
#         # 如果没有提供session_id，返回空结果
#         return {}
#
#     # 创建或加载会话
#     external_data = state.get("external_session_data", {})
#     user_context = state.get("user_context", {})
#
#     session_data = session_manager.create_or_load_session(session_id, external_data)
#     past_plans = session_manager.get_historical_summary(session_id)
#     # 更新用户上下文
#     if user_context:
#         session_manager.update_session(session_id, {
#             "user_context": {**session_data.get("user_context", {}), **user_context}
#         })
#         session_data["user_context"] = user_context
#
#     # 记录对话开始
#     if state.get("messages"):
#         last_message = state["messages"][-1]
#         if isinstance(last_message, HumanMessage):
#             session_manager.add_conversation(session_id, "user", last_message.content)
#
#     return {
#         "session_id": session_id,
#         "user_context": session_data.get("user_context", {}),
#         "created_at": session_data.get("created_at", datetime.now().isoformat()),
#         "last_updated": session_data.get("last_updated", datetime.now().isoformat()),
#         "search_history": session_data.get("search_history", []),
#         "retrieved_parents": set(session_data.get("retrieved_parents", [])),
#         "past_rehab_plans": past_plans
#     }


# def summarize_history(state: State, llm):
#     """总结对话历史"""
#     # 初始化会话
#     # session_info = initialize_session(state)
#     # if session_info:
#     #     state.update(session_info)
#     session_id = state.get("session_id", "")
#
#     if not session_id or len(state["messages"]) < 4:
#         return {"conversation_summary": ""}
#
#     historical_summary = session_manager.get_historical_summary(session_id)
#
#     if historical_summary:
#         return {
#             "conversation_summary": historical_summary,
#             "agent_answers": [{"__reset__": True}]
#         }
#
#     relevant_msgs = [
#         msg for msg in state["messages"][:-1]
#         if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
#     ]
#
#     if not relevant_msgs:
#         return {"conversation_summary": ""}
#
#     # 如果有会话上下文，添加到对话中
#     session_context = ""
#     if state.get("user_context"):
#         user_context = state["user_context"]
#         session_context = f"\n用户信息: {user_context}\n"
#
#     conversation = f"对话上下文:{session_context}\n对话历史:\n"
#     for msg in relevant_msgs[-3:]:
#         role = "用户" if isinstance(msg, HumanMessage) else "助手"
#         conversation += f"{role}: {msg.content}\n"
#
#     summary_response = llm.with_config(temperature=0.2).invoke([
#         SystemMessage(content=get_conversation_summary_prompt()),
#         HumanMessage(content=conversation)
#     ])
#
#     return {
#         "conversation_summary": summary_response.content,
#         "agent_answers": [{"__reset__": True}]
#     }

def extract_and_compress_docs(state: AgentState, llm):
    """文档提纯节点：从长篇大论中提取只跟患者问题相关的一两句话"""
    raw_docs = state.get("raw_retrieved_docs", "")
    question = state.get("question", "")
    raw_faq = state.get("raw_retrieved_faq", "")
    last_docs_content = state.get("retrieved_docs", "")
    last_faq_content = state.get("retrieved_faq","")
    tool_call_count = state.get("tool_call_count", "")
    if not raw_docs:
        return {}  # 没有检索文档，直接跳过

    extraction_prompt = f"""你是一个高效的医疗文献提取员。
                        患者的问题是："{question}"
                        以下是系统检索到的长篇医学指南：
                        ---
                        {raw_docs}
                        ---
                        任务：请从上述长篇指南中，提取出【直接回答患者问题】的客观医学结论、数据（如天数、角度）和禁忌症。
                        要求：不要废话，极度精简，控制在 300 字以内。如果指南里完全没提到患者问的内容，请直接输出“未提及”，并要求用户重新输入。
                        """
    extraction_prompt_faq = f"""以下的信息是系统检索到的医患FAQ信息：
                        ---
                        {raw_faq}
                        ---
                        任务：请从上述的医患对话中，提取出与用户问题相似的医患对话信息中的能【直接回答患者问题】的医生回复。
                        要求：保留医生的口吻，只留下医生回复部分并确保回答患者问题，如果FAQ文档中完全没提到患者问的内容，请直接输出“未提及”，并要求用户重新输入。
                        """
    # 这一步甚至可以用一个更便宜的 LLM 实例去跑，省钱！
    response = llm.invoke([HumanMessage(content=extraction_prompt)])
    compressed_info = response.content
    response = llm.invoke([HumanMessage(content=extraction_prompt_faq)])
    compressed_info_faq = response.content
    print(f"🗜️ [文档提纯节点] 已将万字长文压缩为核心金块: {compressed_info[:300]}...")
    print(f"🗜️ [faq提纯节点] 用于回复患者提问的信息: {compressed_info[:50]}...")
    return {
        "retrieved_docs": last_docs_content + compressed_info,  # 把精华存入正式变量 可能他要多次查，就先这么着
        "raw_retrieved_docs": "",  # 阅后即焚，清空原始长文，释放内存
        "retrieved_faq": last_faq_content + compressed_info_faq,
        "raw_retrieved_faq": "",
        "tool_call_count": tool_call_count + 1
    }


# 多添加一个结点，输入RAG系统的自查询部分，用于显式提取用户的输入意图，提高检索的精度。
def query_analyzer(state: AgentState, llm):
    """显式自查询节点：解析意图并对齐标准词表"""

    # 1. 组装输入信息
    illness = state.get("current_illness", "未知病历")
    recent_messages = state.get("messages", [])[-3:]  # 取最近的三句话
    chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_messages])

    user_context = f"【患者病历】:\n{illness}\n\n【近期对话】:\n{chat_history}"

    # 2. 调用 LLM（使用强制 JSON 输出能力）
    # 注意：最新的 LangChain 支持 llm.with_structured_output()，更稳。这里展示通用写法：
    messages = [
        SystemMessage(content=get_self_query_parser_prompt()),
        HumanMessage(content=user_context)
    ]
    session_id = state.get("session_id", "Session_01")
    custom_performance_monitor.start_node("query_analyzer", session_id)
    response = llm.invoke(messages)
    custom_performance_monitor.end_node("query_analyzer", response)
    # 3. 解析 JSON (加上基础的容错)
    try:
        parsed_filters = json.loads(response.content.strip("```json\n").strip("```"))
    except json.JSONDecodeError:
        # 兜底逻辑
        parsed_filters = {"disease": "综合", "category": "综合", "optimized_query": state.get("question")}

    print(f"🔍 [Self-Query 节点] 提取出标准过滤器: {parsed_filters}")

    # 4. 更新 State
    return {"search_filters": parsed_filters}


# 再一次拆分调度器的负责的工作，把生成的任务也拆分出去
def get_doctor_speaker_prompt() -> str:
    return """你是一位坐诊于三甲医院康复科的【资深主治医师】。
你的任务是：根据系统后台为你提供的【检索提纯资料】（包含客观指南提纯与医患FAQ的医生回答），结合患者的上下文，给患者生成几句温和、专业、充满同理心的回复。

【表达准则】：
1. 人设要求：语气要温暖、有耐心（像真实的医生面对面交流）。不要生硬地罗列“123点”，要把知识揉进对话里。
2. 数据依赖：
   - 绝不凭空捏造医学指标。所有具体的数字（角度、天数、频次）必须来源于后台的【客观医学指南】。
   - 利用后台的【病友经验(FAQ)】来安抚患者（例如：“其实很多半月板术后的病友在这个阶段都会觉得大腿酸，这是正常的...”）。
3. 引导互动：每次回答结束后，最多只附带【一个】跟进问题（如询问当下的疼痛度），引导患者继续提供出具报告所需的指标。
4. 篇幅控制：保持简短精炼，微信聊天体，字数控制在 100-200 字左右。
"""


def doctor_speaker(state: AgentState, llm):
    """对话生成节点：将冰冷的检索数据转化为充满同理心的医生回复"""

    # 1. 提取后台准备好的物料
    retrieved_docs = state.get("retrieved_docs", "")
    faq_docs = state.get("retrieved_faq", "")  # 假设 FAQ 结果存在这里
    question = state.get("question", "")
    print(f"👨‍⚕️ [retrieved_docs 节点] 生成回复: {retrieved_docs[:50]}...")
    print(f"👨‍⚕️ [faq_docs 节点] 生成回复: {faq_docs[:50]}...")

    # 2. 组装医生专用的提示词
    sys_msg = get_doctor_speaker_prompt()
    if retrieved_docs or faq_docs:
        sys_msg += "\n\n【系统后台为你准备的医学资料】:\n"
        if retrieved_docs: sys_msg += f"📄 客观指南提纯:\n{retrieved_docs}\n"
        if faq_docs: sys_msg += f"💬 医患FAQ的医生回答:\n{faq_docs}\n"

    # 3. 将对话历史和提示词送给 LLM 渲染文本
    messages = [SystemMessage(content=sys_msg)] + state["messages"]

    # 这一步模型不需要挂载工具，纯文本生成！
    response = llm.invoke(messages)

    print(f"👨‍⚕️ [Doctor-Speaker 节点] 生成回复: {response.content[:50]}...")

    # 返回并更新最后的消息，同时清空临时物料避免污染下一轮
    return {
        "messages": [response],
        "retrieved_docs": "",
        "faq_docs": ""
    }


def orchestrator(state: AgentState, llm):
    """
    核心：多诊断指南整合
    """
    context_summary = state.get("context_summary", "").strip()
    current_illness = state.get("current_illness", {})
    # past_plans = state.get("past_rehab_plans", "").strip()  # 获取历史方案
    current_plan = state.get("current_case_plan", "")  # 获取当前病例的方案
    # 定义的是系统提示
    # 构建系统提示
    sys_msg_content = get_orchestrator_prompt()
    session_id = state.get("session_id", "Session_01")
    filters = state.get("search_filters", {})
    disease = filters.get("disease", "综合")
    category = filters.get("category", "综合")
    custom_performance_monitor.start_node("orchestrator", session_id)
    current_iteration_count = state.get("iteration_count", "0")  # 获取当前病例的方案
    # 添加用户信息
    # if user_context:
    #     patient_info = []
    #     for key, value in user_context.items():
    #         if value:
    #             patient_info.append(f"{key}: {value}")
    #     if patient_info:
    sys_msg_content += f"\n患者病例信息：{current_illness}"

    # 添加上下文摘要
    # if context_summary:
    #     sys_msg_content += f"\n\n先前研究摘要：\n{context_summary}"
    if current_plan:
        sys_msg_content += f"""
            【重要提示】：当前病例已有正在执行的康复方案如下：
            {current_plan}

            【你的任务】：患者现在是复诊或要求调整方案。请不要急于给出全新方案，而是：
            1. 询问患者对现有方案的执行情况、疼痛反馈。
            2. 基于患者的新反馈，调用检索工具寻找调整依据，并给出微调建议。
            """
    else:
        if context_summary:
            sys_msg_content += "【当前状态】：这是一个全新的病例，已经有基础的康复指南。请主动询问症状细节并查阅指南和FAQ文档作为你的回答。"
        else:
            sys_msg_content += "【当前状态】：这是一个全新的病例，尚未出具方案。请先查阅指南和FAQ文档生成初步的康复指南。"
    sys_msg_content += f"\n\n【系统预检信息】：\n系统已判定当前患者对应的标准病种为：【{disease}】,需要搜索的文档类型为[{category}]。当调用搜索工具时，请务必将此词汇填入 `disease`和`category`参数中，以保证检索精准度！"
    retrieved_docs = state.get("retrieved_docs", "")
    if retrieved_docs:
        sys_msg_content += f"\n\n【系统后台刚刚为您检索到的基于用户提问和病例得到的绝密参考资料】:\n{retrieved_docs}"

    retrieved_faq = state.get("retrieved_faq", "")
    if retrieved_faq:
        sys_msg_content += f"\n\n【基于用户提问和病例得到的医患对话信息】:\n{retrieved_faq}"
    sys_msg = SystemMessage(content=sys_msg_content)
    # sys_msg = SystemMessage(content=get_mcp_orchestrator_prompt())
    # 定义的是之前查询的结果
    summary_injection = (
        [HumanMessage(content=f"[当前会话的先前研究摘要]\n\n{context_summary}")]
        if context_summary else []
    )
    # 每一轮调用需要输入系统指令 + 前文信息 ， 第一轮需要单独用户原始输入和要求
    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"])
        force_search = HumanMessage(
            content="你必须在作出回答前先使用'search_medical_guidelines_tool'和'search_patient_faq_tool'两个工具寻找相关信息")
        response = llm.invoke([sys_msg] + summary_injection + [human_msg, force_search])
        custom_performance_monitor.end_node("orchestrator", response)
        return {"messages": [human_msg, response], "tool_call_count": len(response.tool_calls or []),
                "iteration_count": 1}

    response = llm.invoke([sys_msg] + summary_injection + state["messages"])
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    custom_performance_monitor.end_node("orchestrator", response)
    return {"messages": [response],
            # "tool_call_count": len(tool_calls) if tool_calls else 0,
            "iteration_count": current_iteration_count + 1}


# Token 计数器
def estimate_tokens(messages: list) -> int:
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        return sum(
            len(encoding.encode(str(msg.content))) for msg in messages if hasattr(msg, 'content') and msg.content)
    except Exception:
        return sum(len(str(msg.content)) // 4 for msg in messages if hasattr(msg, 'content'))


def intent_analysis_node(state: AgentState, llm):
    """分析用户意图，针对医疗场景优化"""
    last_message = state["messages"][-1]
    user_input = last_message.content.lower()

    # 基于当前病例信息
    current_illness = state.get("current_illness", "")
    has_plan = state.get("has_existing_plan", False)
    dialogue_phase = state.get("dialogue_phase", "initial")

    # 构建意图分析提示
    intent_prompt = f"""
    你是一个医疗助手，正在与患者进行对话。请分析患者的最后一条消息的意图。

    【病例信息】
    当前诊断：{current_illness}
    是否有现有康复指南：{"是" if has_plan else "否"}
    当前对话阶段：{dialogue_phase}

    【用户输入】
    {user_input}

    【可能的意图】
    1. 描述症状 - 患者描述新的症状或症状变化
    2. 回答提问 - 患者回答你之前提出的问题
    3. 询问进展 - 询问治疗进展、康复情况
    4. 请求解释 - 请求解释诊断、治疗方案
    5. 报告问题 - 报告副作用、并发症或新问题
    6. 确认计划 - 确认或询问下一步计划
    7. 结束对话 - 表示结束、感谢、再见
    8. 其他话题 - 与当前医疗不直接相关的话题

    【输出格式】
    请以JSON格式返回(只返回置信度最高的一种类型)：
    {{
        "primary_intent": "意图类别",
        "confidence": 0.0-1.0,
        "requires_clarification": true/false,
        "suggested_next": "ask_question|analyze|give_advice|end",
        "reasoning": "分析理由"
    }}
    """

    try:
        response = llm.invoke(intent_prompt)
        import json
        intent_result = json.loads(response.content)
        state["intent_result"] = intent_result
        print("意图识别", intent_result)
    except:
        # 如果解析失败，使用基于规则的后备
        state["intent_result"] = fallback_intent_analysis(user_input, has_plan)

    return state


def fallback_intent_analysis(user_input: str, has_plan: bool) -> Dict:
    """基于规则的后备意图分析"""
    # 结束意图关键词
    end_keywords = ["谢谢", "再见", "结束", "好了", "不用了", "下次"]
    if any(keyword in user_input for keyword in end_keywords):
        return {
            "primary_intent": "结束对话",
            "confidence": 0.9,
            "requires_clarification": False,
            "suggested_next": "end",
            "reasoning": "用户表达了结束对话的意愿"
        }

    # 问题关键词
    question_keywords = ["为什么", "怎么", "如何", "什么原因", "怎么办", "建议"]
    if any(keyword in user_input for keyword in question_keywords):
        return {
            "primary_intent": "请求解释" if has_plan else "询问进展",
            "confidence": 0.8,
            "requires_clarification": False,
            "suggested_next": "give_advice",
            "reasoning": "用户在询问解释或建议"
        }

    # 症状描述关键词
    symptom_keywords = ["痛", "痒", "肿", "热", "晕", "吐", "咳", "烧"]
    if any(keyword in user_input for keyword in symptom_keywords):
        return {
            "primary_intent": "描述症状",
            "confidence": 0.85,
            "requires_clarification": True,
            "suggested_next": "ask_question",
            "reasoning": "用户在描述症状，需要进一步澄清"
        }

    # 默认：假设是回答提问
    return {
        "primary_intent": "回答提问",
        "confidence": 0.7,
        "requires_clarification": False,
        "suggested_next": "analyze",
        "reasoning": "默认假设用户在回答之前的问题"
    }


def dialogue_decision_node(state: AgentState, llm):
    """基于意图和当前状态决定下一步行动"""
    intent_result = state.get("intent_result", {})
    has_plan = state.get("has_existing_plan", False)
    dialogue_phase = state.get("dialogue_phase", "initial")

    # 如果是结束对话意图
    if intent_result.get("primary_intent") == "结束对话":
        state["next_action"] = "generate_final_plan"
        return state

    # 根据是否有现有康复指南采取不同策略
    if has_plan:
        # 复诊流程：重点关注症状变化和治疗反应
        return follow_up_decision_logic(state, intent_result, llm)
    else:
        # 初诊流程：重点关注症状收集和初步诊断
        return initial_decision_logic(state, intent_result, llm)


def follow_up_decision_logic(state: AgentState, intent_result: Dict, llm):
    """复诊决策逻辑"""
    current_symptoms = state.get("current_symptoms", [])
    pending_clarifications = state.get("pending_clarifications", [])

    # 规则1：如果有待澄清问题，先提问
    if pending_clarifications:
        state["next_action"] = "ask_question"
        return state

    # 规则2：如果是描述症状意图，需要收集完整信息
    if intent_result.get("primary_intent") == "描述症状":
        if not current_symptoms:
            # 第一次描述症状，进入症状收集
            state["next_action"] = "ask_question"
            state["dialogue_phase"] = "symptom_collection"
        elif len(current_symptoms) >= 2:  # 收集了至少2个症状
            # 症状收集足够，进入分析
            state["next_action"] = "analyze_symptoms"
            state["dialogue_phase"] = "analysis"
        else:
            # 需要更多症状信息
            state["next_action"] = "ask_question"

    # 规则3：如果是询问进展或请求解释，直接给建议
    elif intent_result.get("primary_intent") in ["询问进展", "请求解释"]:
        state["next_action"] = "give_advice"

    # 规则4：默认使用意图建议的下一步
    else:
        suggested_next = intent_result.get("suggested_next", "analyze")
        if suggested_next == "ask_question":
            state["next_action"] = "ask_question"
        elif suggested_next == "analyze":
            state["next_action"] = "analyze_symptoms"
        elif suggested_next == "give_advice":
            state["next_action"] = "give_advice"
        else:
            state["next_action"] = "orchestrator"  # 回退到原有流程

    return state


def ask_question_node(state: AgentState, llm):
    """提问节点：基于当前状态生成问题"""
    pending_clarifications = state.get("pending_clarifications", [])
    current_symptoms = state.get("current_symptoms", [])
    has_plan = state.get("has_existing_plan", False)
    current_illness = state.get("current_illness", "")

    if pending_clarifications:
        # 有待澄清问题，取第一个
        question = pending_clarifications.pop(0)
        # 2. 立即从列表中移除（关键修复！）
        state["pending_clarifications"] = pending_clarifications[1:]

        # 3. 记录已提问的问题，避免重复
        asked_questions = state.get("asked_questions", [])
        asked_questions.append({
            "question": question,
            "time": datetime.now().isoformat(),
            "user_response": None  # 等待用户回答
        })
        state["asked_questions"] = asked_questions[-10:]  # 只保留最近10个
    else:
        # 生成新的问题
        if has_plan:
            # 复诊：关注症状变化
            question_prompt = f"""
            患者正在进行复诊，原有诊断：{current_illness}
            已有康复指南：{state.get('current_case_plan', '无')[:200]}

            请生成一个自然的问题，询问患者：
            1. 原有症状是否有变化
            2. 是否出现新症状
            3. 对治疗的反应

            问题示例：
            - "最近原来的症状有什么变化吗？"
            - "有没有出现什么新的不适？"
            - "之前建议的治疗方法感觉怎么样？"

            请生成一个自然、关切的医疗问询。
            """
        else:
            # 初诊：收集症状
            if not current_symptoms:
                question = f"请描述一下您的主要症状是什么？"
            else:
                # 基于已有症状生成澄清问题
                symptoms_text = "; ".join([s.get("description", "") for s in current_symptoms])
                question_prompt = f"""
                患者已描述的症状：{symptoms_text}

                请从医疗角度生成一个澄清问题，帮助更准确诊断。
                可能的方向：
                1. 症状的具体性质（刺痛/胀痛/隐痛）
                2. 持续时间
                3. 加重或缓解因素
                4. 伴随症状

                生成一个简洁、专业的问题。
                """

        response = llm.invoke(question_prompt)
        question = response.content.strip()
        asked_questions = state.get("asked_questions", [])
        asked_questions.append({
            "question": question,
            "time": datetime.now().isoformat(),
            "user_response": None  # 等待用户回答
        })
        state["asked_questions"] = asked_questions[-10:]  # 只保留最近10个

    # 添加到状态
    state["pending_clarifications"] = pending_clarifications

    # 注意：这里不直接添加到messages，由doctor_speaker处理
    return state


def analyze_symptoms_node(state: AgentState, llm):
    """分析症状节点：调用你的检索系统"""
    current_symptoms = state.get("current_symptoms", [])
    has_plan = state.get("has_existing_plan", False)

    if not current_symptoms:
        # 没有症状，回退到提问
        state["next_action"] = "ask_question"
        return state

    # 准备检索查询
    symptoms_text = current_symptoms

    if has_plan:
        # 复诊分析：关注症状变化
        search_query = f"{symptoms_text} 症状变化评估 治疗调整"
    else:
        # 初诊分析：关注鉴别诊断
        search_query = f"{symptoms_text} 鉴别诊断 初步评估"

    # 这里会调用你的检索工具
    # 简化示例，实际应该调用你的tools节点
    state["question"] = search_query
    state["next_action"] = "orchestrator"  # 进入原有流程进行检索

    return state


def give_advice_node(state: AgentState, llm):
    """给建议节点：基于诊断生成建议"""
    has_plan = state.get("has_existing_plan", False)
    current_illness = state.get("current_illness", "")
    differential_diagnosis = state.get("differential_diagnosis", [])

    if has_plan and differential_diagnosis:
        # 复诊建议：基于症状变化调整方案
        advice_prompt = f"""
        患者原有诊断：{current_illness}
        原有康复指南：{state.get('current_case_plan', '')[:500]}

        当前症状分析：{state.get('retrieved_docs', '')[:1000]}

        请给出复诊建议，包括：
        1. 对症状变化的评估
        2. 原有治疗方案是否需要调整
        3. 新的注意事项
        4. 随访建议

        语气要专业、关切。
        """
    elif differential_diagnosis:
        # 初诊建议
        advice_prompt = f"""
        初步诊断考虑：{', '.join(differential_diagnosis)}

        请给出初步建议，包括：
        1. 立即措施（如果需要）
        2. 一般建议
        3. 就医建议
        4. 注意事项

        注意：这不是最终诊断，建议咨询医生确认。
        """
    else:
        # 没有诊断信息，回退
        state["next_action"] = "analyze_symptoms"
        return state

    response = llm.invoke(advice_prompt)
    state["treatment_plan"] = response.content

    return state


def initial_decision_logic(state: AgentState, intent_result: Dict, llm):
    """初诊决策逻辑"""
    current_symptoms = state.get("current_symptoms", [])
    pending_clarifications = state.get("pending_clarifications", [])

    # 规则1：如果有待澄清问题，先提问
    if pending_clarifications:
        state["next_action"] = "ask_question"
        return state

    # 规则2：初始阶段，主动询问症状
    if not current_symptoms:
        state["next_action"] = "ask_question"
        state["dialogue_phase"] = "symptom_collection"
        return state

    # 规则3：已有症状但未分析
    if current_symptoms and not state.get("differential_diagnosis"):
        if len(current_symptoms) >= 3:  # 收集了至少3个症状
            state["next_action"] = "analyze_symptoms"
            state["dialogue_phase"] = "analysis"
        else:
            # 需要更多症状信息
            state["next_action"] = "ask_question"
        return state

    # 规则4：已有诊断但无建议
    if state.get("differential_diagnosis") and not state.get("treatment_plan"):
        state["next_action"] = "give_advice"
        state["dialogue_phase"] = "treatment_advice"
        return state

    # 默认回退
    state["next_action"] = "orchestrator"
    return state


def compress_context(state: AgentState, llm):
    # 作用：当对话历史过长时，启动“内存压缩”。将多轮 Assistant思考、Tool结果和之前的摘要，
    # 压缩成一个简洁、信息密集的新摘要。
    # 关键优化：压缩后的摘要会附加上本轮已执行过的检索操作列表，并提示模型“不要重复执行”，避免了冗余检索
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()
    session_id = state.get("session_id", "Session_01")
    custom_performance_monitor.start_node("compress_context", session_id)
    if not messages:
        return {}

    conversation_text = f"用户的输入信息:\n{state.get('question')}\n"
    if existing_summary:
        conversation_text += f"[先前上下文的摘要]\n{existing_summary}\n\n"
    # 记录工具调用信息
    for msg in messages[1:]:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            tool_calls = []
            for tc in msg.tool_calls:
                if tc["name"] == "search_medical_guidelines_tool":
                    tool_calls.append(f"搜索专业文档：{tc['args'].get('query', '')}")
                elif tc["name"] == "search_patient_faq_tool":
                    tool_calls.append(f"检索FAQ文档：{tc['args'].get('query', '')}")
            if tool_calls:
                conversation_text += f"执行操作：{'; '.join(tool_calls)}\n\n"
        elif isinstance(msg, ToolMessage):
            content = msg.content
            if len(content) > 1000:
                content = content[:1000] + "..."
            conversation_text += f"结果：{content}\n\n"
    # 基于用户提问，压缩工具调用结果的内容 TODO：这里的Prompt也记得改。
    summary_response = llm.invoke(
        [SystemMessage(content=get_context_compression_prompt()), HumanMessage(content=conversation_text)])
    new_summary = summary_response.content
    # 记录这各个工具调用的结果，压缩的时候纪要压缩ai回答的信息，也要压缩RAG检索得到的信息，
    # 因为检索信息后，LLM就会使用，那么后面就不需要再次调用检索了
    # retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    # if retrieved_ids:
    #     search_queries = sorted(r.replace("search_rehab_guidelines", "") for r in retrieved_ids if r.startswith("search_rehab_guidelines"))
    #     block = "\n\n---\n**Already executed (do NOT repeat):**\n"
    #     if search_queries:
    #         block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
    #     new_summary += block
    custom_performance_monitor.end_node("compress_context", new_summary)
    return {"context_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages[1:]]}


def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
    """判断是否需要压缩上下文"""
    messages = state["messages"]

    # 计算令牌数
    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])
    current_tokens = current_token_messages + current_token_summary

    # 动态阈值
    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)

    # 决定下一步
    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
    return Command(goto=goto)


def fallback_response(state: AgentState, llm):
    """兜底响应"""
    last_docs_content = state.get("retrieved_docs", "")
    last_faq_content = state.get("retrieved_faq","")
    prompt = get_fallback_response_prompt()
    tool_results = []
    # 构建输入
    context_parts = []
    last_docs_content = state.get("retrieved_docs", "")
    last_faq_content = state.get("retrieved_faq", "")
    if state["context_summary"]:
        context_parts.append(f"""[压缩上下文]\n{state["context_summary"]}""")
    if state["messages"]:
        last_msg = state["messages"][-1]
        for msg in state["messages"]:
            if isinstance(msg, ToolMessage):
                tool_results.append(msg.content)
        if hasattr(last_msg, 'content') and last_msg.content:
            context_parts.append(f"[最后的消息]\n{last_msg.content}")
    context_parts.append(f"【文档结果】：{last_docs_content}")
    context_parts.append(f"【医患对话结果】：{last_faq_content}")
    input_text = "\n\n".join(context_parts) or "无上下文"

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"""用户的询问: {state["question"]}\n\n{input_text}""")
    ])

    return {
        "messages": [AIMessage(content=response.content)],
        "final_answer": response.content
    }


# def generate_final_plan(state: AgentState, llm):
#     """【新增】当接收到 finalize 指令时，生成正式的最终报告"""
#     prompt = generate_final_plan_prompt()
#
#     context_summary = state.get("context_summary", "")
#     current_plan = state.get("current_case_plan", "")  # 获取当前病例的方案
#
#     context_text = f"【当前就诊记录摘要】\n{context_summary}\n\n"
#     if current_plan:
#         context_text += f"【当前病例已有正在执行的康复方案如下：】\n{current_plan}\n\n"
#
#     messages = [
#         SystemMessage(content=prompt),
#         HumanMessage(content=f"{context_text}\n请输出最终报告。")
#     ]
#
#     response = llm.invoke(messages)
#
#     # 将生成的报告作为回答返回
#     return {"messages": [response], "final_answer": response.content}

def generate_final_plan(state: AgentState, llm):
    """当接收到 finalize 指令时，整合所有信息生成正式报告"""
    prompt = generate_final_plan_prompt()
    session_id = state.get("session_id", "Session_01")
    custom_performance_monitor.start_node("generate_final_plan", session_id)
    context_summary = state.get("context_summary", "").strip()
    current_plan = state.get("current_case_plan", "").strip()
    other_plans = state.get("other_historical_plans", "").strip()
    retrieved_docs = state.get("retrieved_docs", "")
    # ==========================================
    # 【核心修复】：提取当前还没来得及压缩的最新聊天记录
    # ==========================================
    recent_conversation = ""
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage) and msg.content:
            recent_conversation += f"患者: {msg.content}\n"
        elif isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            recent_conversation += f"医生: {msg.content}\n"
        elif isinstance(msg, ToolMessage):
            # 也可以选择把工具检索到的原文献给LLM作最终参考（防截断）
            content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
            recent_conversation += f"[系统检索到文献]: {content}\n"

    # ==========================================
    # 组装超级上下文 (包含：已压缩的历史 + 刚聊的最新记录 + 各种病历)
    # ==========================================
    context_text = ""

    if context_summary:
        context_text += f"【前期对话摘要】\n{context_summary}\n\n"

    if recent_conversation:
        context_text += f"【本轮最新对话与检索记录】\n{recent_conversation}\n\n"

    if current_plan:
        context_text += f"【当前病例已有正在执行的康复方案（待调整）】\n{current_plan}\n\n"

    if other_plans:
        context_text += f"⚠️【患者其他部位病历禁忌】\n{other_plans}\n\n"
    if retrieved_docs:
        context_text += f"\n\n【检索到的基于用户提问和病例得到的康复指南参考资料】:\n{retrieved_docs}"
    retrieved_faq = state.get("retrieved_faq", "")
    if retrieved_faq:
        context_text += f"\n\n【基于用户提问和病例得到的医患FAQ对话信息】:\n{retrieved_faq}"
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"{context_text}\n请综合以上所有信息，为患者输出最新的最终康复报告。")
    ]

    response = llm.invoke(messages)
    custom_performance_monitor.end_node("generate_final_plan", response)
    return {"messages": [response], "final_answer": response.content}


def review_and_adjust(state: AgentState, llm):
    """当接收到 finalize 指令时，整合所有信息生成正式报告"""
    prompt = generate_final_plan_prompt()
    session_id = state.get("session_id", "Session_01")
    custom_performance_monitor.start_node("generate_final_plan", session_id)
    context_summary = state.get("context_summary", "").strip()
    current_plan = state.get("current_case_plan", "").strip()
    other_plans = state.get("other_historical_plans", "").strip()

    # ==========================================
    # 【核心修复】：提取当前还没来得及压缩的最新聊天记录
    # ==========================================
    recent_conversation = ""
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage) and msg.content:
            recent_conversation += f"患者: {msg.content}\n"
        elif isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            recent_conversation += f"医生: {msg.content}\n"
        elif isinstance(msg, ToolMessage):
            # 也可以选择把工具检索到的原文献给LLM作最终参考（防截断）
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            recent_conversation += f"[系统检索到文献]: {content}\n"

    # ==========================================
    # 组装超级上下文 (包含：已压缩的历史 + 刚聊的最新记录 + 各种病历)
    # ==========================================
    context_text = ""

    if context_summary:
        context_text += f"【前期对话摘要】\n{context_summary}\n\n"

    if recent_conversation:
        context_text += f"【本轮最新对话与检索记录】\n{recent_conversation}\n\n"

    if current_plan:
        context_text += f"【当前病例已有正在执行的康复方案（待调整）】\n{current_plan}\n\n"

    if other_plans:
        context_text += f"⚠️【患者其他部位病历禁忌】\n{other_plans}\n\n"

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"{context_text}\n请综合以上所有信息，为患者输出最新的最终康复报告。")
    ]

    response = llm.invoke(messages)
    custom_performance_monitor.end_node("generate_final_plan", response)
    return {"messages": [response], "final_answer": response.content}


def collect_answer(state: AgentState):
    """结果收集与落库"""
    action = state.get("action", "chat")
    patient_id = state.get("patient_id")
    # session_id = state.get("session_id")

    # 如果是结束指令，且有患者ID，进行长期记忆落库
    if action == "finalize" and patient_id:
        answer = state.get("final_answer", "")
        # if answer:
        #     patient_db.save_final_plan(patient_id, session_id, answer)
        return {"final_answer": answer}

    # 否则只是普通返回当前 LLM 的回答
    last_message = state["messages"][-1]
    return {"final_answer": last_message.content}


# def collect_answer(state: AgentState, llm=None):
#     """
#     收集答案并保存康复方案
#     增强版的collect_answer节点
#     """
#     session_id = state.get("session_id", "")
#     user_context = state.get("user_context", {})
#     action = state.get("action", "chat")
#
#     # 获取最后一个LLM输出
#     last_message = state["messages"][-1]
#     is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
#
#     if not is_valid:
#         # 如果没有有效的LLM输出，从工具结果生成
#         answer = generate_answer_from_tools(state, llm)
#     else:
#         answer = last_message.content
#
#     if action == "finalize" and session_id and len(answer) >= 30:
#         try:
#             session_manager.save_rehabilitation_plan(
#                 session_id=session_id,
#                 question=state["question"],
#                 answer=answer,  # 这是最终生成的总结性方案
#                 user_context=user_context
#             )
#             print(f"✅ 会话 {session_id} 明确结束，最终方案已存档。")
#         except Exception as e:
#             print(f"保存康复方案失败：{e}")
#     else:
#         # 检查答案质量
#         if not answer or len(answer.strip()) < 30 or "无法生成" in answer or "Unable" in answer:
#             answer = "基于现有信息，暂时无法提供完整的康复方案。建议咨询专业医生获取个性化指导。"
#     return {
#         "final_answer": answer,
#         "agent_answers": [{
#             "index": state.get("question_index", 0),
#             "question": state["question"],
#             "answer": answer
#         }]
#     }


def generate_answer_from_tools(state: AgentState, llm) -> str:
    """从工具结果生成答案"""
    # 收集工具结果
    tool_results = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            tool_results.append(msg.content)

    if not tool_results:
        return "未能检索到相关康复指南。"

    # 构建生成提示
    user_context = state.get("user_context", {})
    context_text = "\n---\n".join(tool_results)

    prompt = f"""
        患者信息：{user_context}
        
        检索到的康复指南：
        {context_text}
        
        问题：{state["question"]}
        
        请基于以上信息，生成一个结构化的康复方案，包括：
        1. 康复目标
        2. 具体建议
        3. 注意事项
        4. 随访计划
        
        康复方案：
        """

    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        return response.content
    except:
        # 如果生成失败，返回原始结果
        return f"基于以下信息：\n{context_text}"
