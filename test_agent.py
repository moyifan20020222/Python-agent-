import os
import sqlite3
import uuid
from datetime import datetime

import chromadb
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tracers import LangChainTracer
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from project.rehab_core.guide_chunker import EmbeddingHierarchicalIndexer
from project.rehab_core.langgraph_callbacks import PerformanceCallback
from project.rehab_core.node_monitor import NodeMonitor
from project.rehab_core.performance_monitor_tool import tool_performance_monitor
from project.rehab_core.retrieval.hybrid_retriever import VectorRetriever, BM25Retriever
from project.rehab_core.retrieval.hybrid_retriever_final import FinalHybridRetrieval
from project.rehab_core.state import AgentState
from project.rehab_core.tool_factory import ToolFactory
from project.rehab_core import config
from project.rehab_core.PatientRecordManager import patient_db
from project.rehab_core import graph
from project.rehab_core.performance_monitor import custom_performance_monitor
# ==========================================
# 1. 挂载测试用的 Checkpointer（图的编译端）
# （这里假设你已经定义好了 agent_builder）
# ==========================================
from project.db.chroma_loader import ChromaFinalBuilder
from chromadb.config import Settings
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from project.rehab_core import config
from project.rehab_core.parent_store_manager_updated import ParentStoreManager
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
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "rehab-agent-monitor"
os.environ["LANGSMITH_TRACING_V2"] = "true"
from langsmith import Client

# 初始化客户端
langsmith_client = Client(
    api_key=os.getenv("LANGSMITH_API_KEY"),
    api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
)

# 测试连接
try:
    projects = langsmith_client.list_projects()
except Exception as e:
    print(f"❌ LangSmith连接失败: {e}")
# memory = MemorySaver()
# rehab_agent_graph = agent_builder.compile(checkpointer=memory)
MODEL_PATH = "D:/Desktop/company/Agent-rag/agentic-rag-for-dummies/models/Hu0922/BGE_Medical"
DB_PATH = config.SQLITE_DB_PATH
CHROMA_PATH = config.CHROMA_PATH

# 创建构建器 这个是我们更新了参考的指南后，才调用的，平常直接获取就好
# builder = ChromaFinalBuilder(MODEL_PATH, DB_PATH, CHROMA_PATH)
#
# client, collection = builder.build(8, 16)
client = chromadb.PersistentClient(
                path=str(CHROMA_PATH),
                settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
collection = client.get_or_create_collection(
                name="rehab_guidelines",
                metadata={"hnsw:space": "cosine"}
            )

indexer = EmbeddingHierarchicalIndexer(
    db_path="D:\\Desktop\\company\\Agent-rag\\agentic-rag-for-dummies\\project\\db\\rehab.db",
    parent_store_path="./data/parent_docs_embedding",
    chroma_path="./chroma_embedding",
    parent_size=2000,
    parent_overlap=200,
    child_size=400,
    child_overlap=80,
    embedding_model="m3e-base",
    embedding_device="cpu"
)
# 两个索引器的初始化构建，
result_faq_message = indexer.index_faq_to_chroma()
result = indexer.index_to_chroma()
print("内容", indexer.faq_collection)
faq_data = indexer.faq_collection.get(include=['documents', 'metadatas'])
bm25_docs = []
if faq_data and faq_data.get('ids'):
    for i in range(len(faq_data['ids'])):
        bm25_docs.append({
            'page_content': faq_data['documents'][i],
            'metadata': faq_data['metadatas'][i],
            'id': faq_data['ids'][i]  # 确保有 ID 字段用于混合融合
        })

# 2. 实例化两个检索器
bm25Retriever = BM25Retriever(documents=bm25_docs)
vectorRetriever = VectorRetriever(
    collection=indexer.faq_collection,  # 传入 FAQ 向量库
    embedding_model=indexer.embedding_function  # 传入 embedding 模型
)

# 3. 实例化混合检索
hybrid_retrieval = FinalHybridRetrieval(
    vector_retriever=vectorRetriever,
    bm25_retriever=bm25Retriever
    # 注意：如果你本地没有下载 bge-reranker-base，这里可以传入 reranker_model=None 降级为单纯的 RRF 融合
)
print("✅ 混合检索器装载完成！\n")

guide_data = indexer.collection.get(include=['documents', 'metadatas'])
guide_docs = []
if guide_data and guide_data.get('ids'):
    for i in range(len(guide_data['ids'])):
        guide_docs.append({
            'page_content': guide_data['documents'][i],
            'metadata': guide_data['metadatas'][i],
            'id': guide_data['ids'][i]
        })

guide_bm25 = BM25Retriever(documents=guide_docs)
guide_vector = VectorRetriever(indexer.collection, indexer.embedding_function)
# 复用同一个 Reranker 模型对象，节省显存
guide_hybrid_retriever = FinalHybridRetrieval(guide_vector, guide_bm25)

parent_store_manager = ParentStoreManager(config.PARENT_STORE_PATH)
tool = ToolFactory(guide_hybrid=guide_hybrid_retriever, faq_hybrid=hybrid_retrieval, parent_store_manager=parent_store_manager)
tool_list = tool.create_tools()
conn = sqlite3.connect("./data/langgraph_checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)
rehab_agent_graph = graph.build_graph(LLM, tool_list, memory)
db_path = config.SQLITE_DB_PATH

# ==========================================
# 2. 模拟 OpenClaw / 外部 MCP 的调度函数
# ==========================================

# def simulate_openclaw_call(patient_id, selected, session_id, user_input, action="chat"):
#     # 1. 拆分获取两种不同的病历状态
#     case_id = selected['case_id']
#     current_illness = selected['diagnosis']
#     current_plan = patient_db.get_current_case_plan(case_id)
#     other_plans = patient_db.get_other_historical_plans(patient_id, case_id)
#
#     initial_state = {
#         "question": user_input,
#         "session_id": session_id,
#         "patient_id": patient_id,
#         "action": action,
#         "current_illness": current_illness,
#         "current_case_plan": current_plan,  # 当前的放进来给 orchestrator
#         "other_historical_plans": other_plans,  # 其他的放进来给 generate_final_plan 备用
#         "messages": [HumanMessage(content=user_input)] if user_input.strip() else [],
#         "retrieved_docs": None,
#         "retrieved_faq": None,
#     }
#     callback = PerformanceCallback(db_path=db_path)
#     config = {"configurable": {"thread_id": session_id}}
#     result = rehab_agent_graph.invoke(initial_state, config=config, callbacks=[callback])
#     if action == "finalize":
#         final_report = result.get("final_answer", "")
#         print("最终结果", final_report)
#         patient_db.save_final_plan(case_id, session_id, final_report)
#         return {"会话已结束，最终康复方案已生成并归档。\n" + final_report}
#     print("对话后产生的结点信息", initial_state)
#     return result["messages"][-1].content

def simulate_openclaw_call(patient_id, selected, session_id, user_input, action="chat", current_state=None, langsmith_config=None):
    """增强的模拟调用函数，支持状态保持"""
    # 1. 获取病例信息
    case_id = selected['case_id']
    current_illness = selected['diagnosis']
    current_plan = patient_db.get_current_case_plan(case_id)
    other_plans = patient_db.get_other_historical_plans(patient_id, case_id)

    # 2. 判断是否有现有康复指南
    has_plan = bool(current_plan and current_plan != "测试病历")
    callbacks = [PerformanceCallback(db_path=db_path)]
    if langsmith_config and langsmith_config.get("enabled"):
        tracer = LangChainTracer(
            project_name=langsmith_config["project_name"],
            client=langsmith_config["client"],
            tags=[
                f"patient:{patient_id}",
                f"case:{selected['case_id']}",
                f"session:{session_id}",
                f"action:{action}"
            ]
        )
        callbacks.append(tracer)
    # 3. 准备初始状态
    if current_state is None:
        # 首次调用，创建新状态
        initial_state = {
            "messages": [HumanMessage(content=user_input)] if user_input.strip() else [],
            "question": user_input,
            "session_id": session_id,
            "patient_id": patient_id,
            "action": action,
            "user_context": {},
            "context_summary": "",
            "current_case_plan": current_plan,
            "other_historical_plans": other_plans,
            "current_illness": current_illness,
            "final_answer": "",
            "iteration_count": 0,
            "tool_call_count": 0,
            "search_filters": {},
            "raw_retrieved_docs": "",
            "retrieved_docs": "",
            "raw_retrieved_faq": "",
            "retrieved_faq": "",

            # 新增字段
            "has_existing_plan": has_plan,
            "dialogue_phase": "follow_up" if has_plan else "initial",
            "next_action": "await_decision",
            "pending_clarifications": [],
            "current_symptoms": [],
            "differential_diagnosis": [],
            "treatment_plan": current_plan if has_plan else "",
            "intent_result": {},
            "is_first_turn": True
        }
    else:
        # 非首次调用，更新现有状态
        initial_state = current_state.copy()
        initial_state["messages"].append(HumanMessage(content=user_input))
        initial_state["question"] = user_input
        initial_state["action"] = action
        initial_state["is_first_turn"] = False

    # 4. 特殊处理：如果用户输入为空（finalize动作），直接生成最终计划
    if action == "finalize":
        # 设置结束标志
        initial_state["next_action"] = "generate_final_plan"
        config = {"configurable": {"thread_id": session_id}}
        result = rehab_agent_graph.invoke(initial_state, config=config)

        final_report = result.get("final_answer", "")
        print("最终结果", final_report)
        patient_db.save_final_plan(case_id, session_id, final_report)

        return {"会话已结束，最终康复方案已生成并归档。\n" + final_report}, None

    # 5. 调用增强的Agent图
    callback = PerformanceCallback(db_path=db_path)
    config = {"configurable": {"thread_id": session_id}}
    result = rehab_agent_graph.invoke(initial_state, config=config,
                                      # callbacks=[callback]
                                      callbacks=callbacks
                                      )

    if langsmith_config and langsmith_config.get("enabled"):
        try:
            langsmith_config["client"].create_run(
                name=f"agent_{action}_{session_id}",
                run_type="chain",
                inputs={
                    "user_input": user_input,
                    "patient_id": patient_id,
                    "case_id": selected['case_id'],
                    "action": action
                },
                outputs={
                    "ai_response": result.get("messages", [])[-1].content[:500] if result.get("messages") else "",
                    "next_action": result.get("next_action", ""),
                    "symptoms_count": len(result.get("current_symptoms", []))
                },
                project_name=langsmith_config["project_name"],
                tags=["agent_execution", f"phase:{result.get('dialogue_phase', 'unknown')}"],
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "iteration": result.get("iteration_count", 0)
                }
            )
        except Exception as e:
            print(f"⚠️ LangSmith记录失败: {e}")

    # 6. 获取响应
    if action == "chat":
        # 提取AI的最后一条消息
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        if ai_messages:
            answer = ai_messages[-1].content
        else:
            answer = "抱歉，我没有收到回复。"

        print("对话后产生的状态信息:", {
            "dialogue_phase": result.get("dialogue_phase"),
            "next_action": result.get("next_action"),
            "current_symptoms": len(result.get("current_symptoms", [])),
            "pending_clarifications": len(result.get("pending_clarifications", []))
        })

        return answer, result

    return "", result


def prepare_initial_state(
        patient_id: str,
        case_id: str,
        current_illness: str,
        current_plan: str,
        session_id: str,
        first_user_input: str
) -> AgentState:
    """准备初始状态"""
    has_existing_plan = bool(current_plan and current_plan != "测试病历")

    return {
        "messages": [
            {"role": "user", "content": first_user_input}
        ],
        "question": first_user_input,
        "session_id": session_id,
        "patient_id": patient_id,
        "action": "chat",
        "user_context": {},
        "context_summary": "",
        "current_case_plan": current_plan,
        "other_historical_plans": "",
        "current_illness": current_illness,
        "final_answer": "",
        "iteration_count": 0,
        "tool_call_count": 0,
        "search_filters": {},
        "raw_retrieved_docs": "",
        "retrieved_docs": "",
        "raw_retrieved_faq": "",
        "retrieved_faq": "",

        # 新增字段初始化
        "has_existing_plan": has_existing_plan,
        "dialogue_phase": "initial" if not has_existing_plan else "follow_up",
        "next_action": "await_decision",  # 初始等待决策
        "pending_clarifications": [],
        "current_symptoms": [],
        "differential_diagnosis": [],
        "treatment_plan": current_plan if has_existing_plan else "",
        "intent_result": {}
    }

import asyncio
import sys
async def interactive_stream_test(patient_id: str, selected, session_id: str, user_input: str,
                                  action: str = "chat"):
    """
    本地终端流式测试：完美复现 Agent 的内部思考和打字机输出
    """
    global rehab_agent_graph, patient_db

    # 1. 组装输入状态 (和你原来一样)
    case_id = selected['case_id']
    current_illness = selected['diagnosis']
    current_illness = patient_db.get_case(case_id) if hasattr(patient_db, 'get_case') else "测试病历"
    current_plan = patient_db.get_current_case_plan(case_id) if hasattr(patient_db, 'get_current_case_plan') else ""

    other_plans = patient_db.get_other_historical_plans(patient_id, case_id)

    initial_state = {
        "question": user_input,
        "session_id": session_id,
        "patient_id": patient_id,
        "action": action,
        "current_illness": current_illness,
        "current_case_plan": current_plan,
        "other_historical_plans": other_plans,
        "messages": [("user", user_input)] if user_input.strip() else []
    }

    config = {"configurable": {"thread_id": session_id}}

    print(f"\n👤 患者: {user_input}")
    print(f"🤖 助手: ", end="")
    sys.stdout.flush()  # 刷新缓冲区，保证立刻输出

    try:
        # 2. 监听 LangGraph 事件流
        async for event in rehab_agent_graph.astream_events(initial_state, config, version="v2"):
            kind = event["event"]

            # 【节点状态可视】当进入某些重要节点时打印
            if kind == "on_chain_start" and event["name"] in ["query_analyzer", "orchestrator", "generate_final_plan"]:
                print(f"\n[🔄 系统状态: 正在进入 {event['name']} 节点...]")

            # 【工具调用可视】当大模型决定调用搜索工具时
            elif kind == "on_tool_start":
                tool_name = event["name"]
                args = event["data"].get("input", {})
                print(f"\n[🔍 检索工具启动: {tool_name} | 参数: {args}]")
                print("🤖 助手: ", end="")  # 工具打印完，恢复打字机开头

            # 【打字机效果】实时流式输出大模型的 Token
            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    # sys.stdout.write 不会像 print 那样自动换行
                    sys.stdout.write(chunk.content)
                    sys.stdout.flush()

        print("\n\n✅ (本轮对话流式生成结束)\n")

    except Exception as e:
        print(f"\n❌ 流式生成发生错误: {e}\n")

# ==========================================
# 3. 控制台交互式测试主循环
# ==========================================
# 1. 初始化LangSmith
def init_langsmith():
    """初始化LangSmith监控"""
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("⚠️ 未找到LANGSMITH_API_KEY，LangSmith监控将禁用")
        return None

    try:
        client = Client()
        project_name = os.getenv("LANGSMITH_PROJECT", "rehab-agent-test")

        print(f"✅ LangSmith监控已启用 - 项目: {project_name}")

        return {
            "client": client,
            "project_name": project_name,
            "enabled": True
        }
    except Exception as e:
        print(f"❌ LangSmith初始化失败: {e}")
        return None


async def main():
    print("🏥 [康复指导智能系统] 测试终端启动...")
    current_patient = "22"
    current_case = None
    current_session = None
    langsmith_config = init_langsmith()
    while True:
        # 如果当前没有选中病例，强制进行病例选择 (模拟 OpenClaw 前端逻辑)
        if not current_case:
            print("\n" + "=" * 50)
            # cases = patient_db.get_unresolved_cases(current_patient)
            cases = patient_db.get_all_cases(current_patient)
            if not cases:
                print("🎉 太棒了，该患者目前没有待处理的病例！")
                break

            print(f"📋 患者 {current_patient} 发现以下就诊记录：")
            for i, c in enumerate(cases):
                print(
                    f" [{i}] [{c['status']}] 病例号: {c['case_id']} | 诊断: {c['diagnosis']} | 病例创建时间: {c['created_at']}"
                    f" | 康复指南创建时间: {c['guide_created_at']}")

            choice = input("👉 请输入序号选择要处理的病例 (输入 q 退出): ")
            if choice.lower() == 'q':
                break

            try:
                selected = cases[int(choice)]
                current_case = selected['case_id']
                current_plan = patient_db.get_current_case_plan(current_case)

                print(f"\n✅ 已锁定病例: {selected['diagnosis']}")
                # 这个历史档案，用于最终生成我们的康复指南
                # 而当前需要的只是对应于当前病例的康复指南，
                if current_plan:
                    current_session = selected['session_id']
                    print(f"\n✅ 正在开启复诊流程。系统已加载该病例之前的方案...")
                    # OpenClaw 可以自动帮用户发一条隐式消息启动
                    answer = await interactive_stream_test(current_patient, selected, current_session,
                                                           "你好，我想针对我之前的方案进行复查。")
                    print(f"🤖 助手: {answer}")
                else:
                    current_session = f"session_{uuid.uuid4().hex[:6]}"  # 为这个新病例生成专属会话ID
                    print(f"\n✅ 正在开启初诊流程。该病例尚未生成过方案。")
                    print("💬 请描述您的具体症状，输入 '/end' 出具报告。")
                # print(f"🔄 正在拉取患者历史档案...")
                # history = patient_db.get_patient_history(current_patient, current_case)
                # if history:
                #     print(history)
                # else:
                #     print("（无其他历史病历）")
                # print("💬 您现在可以开始与 AI 对话了。输入 '/end' 生成最终报告并结案归档。")
            except (ValueError, IndexError):
                print("❌ 输入无效，请重新选择。")
                continue

        # 开始聊天循环
        user_input = input(f"\n🧑 用户 ({current_case}): ")

        if user_input.strip() == "/quit":

            custom_performance_monitor.save_to_database()
            break

        elif user_input.strip() == "/end":
            print("⏳ 正在生成最终康复指导报告并落库...")
            report = await interactive_stream_test(current_patient, selected, current_session, "", action="finalize")
            custom_performance_monitor.save_to_database()
            tool_performance_monitor.save_to_database()
            print("\n【最终正式报告】:")
            print(report)
            # 结案后清空当前上下文，下一轮循环会重新要求选择病例
            current_case = None
            current_session = None
            continue

        # 正常对话发送
        answer = await interactive_stream_test(current_patient, selected, current_session, user_input, action="chat")
        print(f"🤖 助手: {answer}")
if __name__ == "__main__":
    # asyncio.run(main())  # 用 asyncio 驱动 这个不能用Sqlite 只能用PostGreSQL，这个我们先放一下，先保证可行
    langsmith_config = init_langsmith()
    # print("🏥 [康复指导智能系统] 测试终端启动...")
    current_patient = "22"
    current_case = None
    current_session = None
    current_state = None  # 用于在多轮对话中保持状态
    selected = None  # 当前选中的病例

    while True:
        # 如果当前没有选中病例，强制进行病例选择
        if not current_case:
            print("\n" + "=" * 50)
            cases = patient_db.get_all_cases(current_patient)
            if not cases:
                print("🎉 太棒了，该患者目前没有待处理的病例！")
                break

            print(f"📋 患者 {current_patient} 发现以下就诊记录：")
            for i, c in enumerate(cases):
                print(
                    f" [{i}] [{c['status']}] 病例号: {c['case_id']} | 诊断: {c['diagnosis']} | 病例创建时间: {c['created_at']}"
                    f" | 康复指南创建时间: {c['guide_created_at']}")

            choice = input("👉 请输入序号选择要处理的病例 (输入 q 退出): ")
            if choice.lower() == 'q':
                break

            try:
                selected = cases[int(choice)]
                current_case = selected['case_id']
                current_plan = patient_db.get_current_case_plan(current_case)

                print(f"\n✅ 已锁定病例: {selected['diagnosis']}")

                if current_plan:
                    # 复诊流程
                    current_session = selected['session_id']
                    print(f"\n✅ 正在开启复诊流程。系统已加载该病例之前的方案...")

                    # 自动发送初始消息
                    initial_message = "你好，我想针对我之前的方案进行复查。"
                    print(f"👤 用户: {initial_message}")
                    answer, current_state = simulate_openclaw_call(
                        current_patient, selected, current_session,
                        initial_message, action="chat",
            langsmith_config=langsmith_config
                    )
                    print(f"🤖 助手: {answer}")
                else:
                    # 初诊流程
                    current_session = f"session_{uuid.uuid4().hex[:6]}"
                    print(f"\n✅ 正在开启初诊流程。该病例尚未生成过方案。")
                    print("💬 请描述您的具体症状，输入 '/end' 出具报告。")
                    # 不自动发送消息，等待用户输入
                    current_state = None

            except (ValueError, IndexError):
                print("❌ 输入无效，请重新选择。")
                continue

        # 开始聊天循环
        user_input = input(f"\n🧑 用户 ({current_case}): ")

        if user_input.strip() == "/quit":
            if langsmith_config:
                langsmith_config["client"].create_run(
                    name="session_ended",
                    run_type="chain",
                    inputs={"reason": "user_quit"},
                    project_name=langsmith_config["project_name"],
                    tags=["session_end", "user_initiated"]
                )
            custom_performance_monitor.save_to_database()
            break

        elif user_input.strip() == "/end":
            print("⏳ 正在生成最终康复指导报告并落库...")
            report, _ = simulate_openclaw_call(
                current_patient, selected, current_session,
                "", action="finalize", current_state=current_state,
            langsmith_config=langsmith_config
            )

            if isinstance(report, dict) and "会话已结束" in str(report):
                print(report)
            else:
                print("\n【最终正式报告】:")
                print(report)

            custom_performance_monitor.save_to_database()
            tool_performance_monitor.save_to_database()

            # 结案后清空当前上下文
            current_case = None
            current_session = None
            current_state = None
            selected = None
            continue

        # 正常对话
        answer, current_state = simulate_openclaw_call(
            current_patient, selected, current_session,
            user_input, action="chat", current_state=current_state,
            langsmith_config=langsmith_config
        )
        print(f"🤖 助手: {answer}")
    # current_patient = "22"
    # current_case = None
    # current_session = None
    #
    # while True:
    #     # 如果当前没有选中病例，强制进行病例选择 (模拟 OpenClaw 前端逻辑)
    #     if not current_case:
    #         print("\n" + "=" * 50)
    #         # cases = patient_db.get_unresolved_cases(current_patient)
    #         cases = patient_db.get_all_cases(current_patient)
    #         if not cases:
    #             print("🎉 太棒了，该患者目前没有待处理的病例！")
    #             break
    #
    #         print(f"📋 患者 {current_patient} 发现以下就诊记录：")
    #         for i, c in enumerate(cases):
    #             print(
    #                 f" [{i}] [{c['status']}] 病例号: {c['case_id']} | 诊断: {c['diagnosis']} | 病例创建时间: {c['created_at']}"
    #                 f" | 康复指南创建时间: {c['guide_created_at']}")
    #
    #         choice = input("👉 请输入序号选择要处理的病例 (输入 q 退出): ")
    #         if choice.lower() == 'q':
    #             break
    #
    #         try:
    #             selected = cases[int(choice)]
    #             current_case = selected['case_id']
    #             current_plan = patient_db.get_current_case_plan(current_case)
    #
    #             print(f"\n✅ 已锁定病例: {selected['diagnosis']}")
    #             # 这个历史档案，用于最终生成我们的康复指南
    #             # 而当前需要的只是对应于当前病例的康复指南，
    #             if current_plan:
    #                 current_session = selected['session_id']
    #                 print(f"\n✅ 正在开启复诊流程。系统已加载该病例之前的方案...")
    #                 # OpenClaw 可以自动帮用户发一条隐式消息启动
    #                 answer = simulate_openclaw_call(current_patient, selected, current_session,
    #                                                        "你好，我想针对我之前的方案进行复查。")
    #                 print(f"🤖 助手: {answer}")
    #             else:
    #                 current_session = f"session_{uuid.uuid4().hex[:6]}"  # 为这个新病例生成专属会话ID
    #                 print(f"\n✅ 正在开启初诊流程。该病例尚未生成过方案。")
    #                 print("💬 请描述您的具体症状，输入 '/end' 出具报告。")
    #             # print(f"🔄 正在拉取患者历史档案...")
    #             # history = patient_db.get_patient_history(current_patient, current_case)
    #             # if history:
    #             #     print(history)
    #             # else:
    #             #     print("（无其他历史病历）")
    #             # print("💬 您现在可以开始与 AI 对话了。输入 '/end' 生成最终报告并结案归档。")
    #         except (ValueError, IndexError):
    #             print("❌ 输入无效，请重新选择。")
    #             continue
    #
    #     # 开始聊天循环
    #     user_input = input(f"\n🧑 用户 ({current_case}): ")
    #
    #     if user_input.strip() == "/quit":
    #         custom_performance_monitor.save_to_database()
    #
    #         break
    #
    #     elif user_input.strip() == "/end":
    #         print("⏳ 正在生成最终康复指导报告并落库...")
    #         report = simulate_openclaw_call(current_patient, selected, current_session, "", action="finalize")
    #         custom_performance_monitor.save_to_database()
    #         tool_performance_monitor.save_to_database()
    #         print("\n【最终正式报告】:")
    #         print(report)
    #         # 结案后清空当前上下文，下一轮循环会重新要求选择病例
    #         current_case = None
    #         current_session = None
    #         continue
    #
    #     # 正常对话发送
    #     answer = simulate_openclaw_call(current_patient, selected, current_session, user_input, action="chat")
    #     print(f"🤖 助手: {answer}")
# if __name__ == "__main__":
#     print("🚀 康复指南 MCP 交互测试环境启动！")
#     print("=============================================")
#
#     current_patient = "p_001"  # 固定一个患者
#     current_session = "session_A"  # 第一次来就诊的会话 ID
#
#     while True:
#         print("-" * 50)
#         print(f"🟢 当前状态 -> 患者: {current_patient} | 会话: {current_session}")
#         print("💡 指令提示:")
#         print("   直接输入内容 = 正常对话")
#         print("   输入 '/end' = 模拟 OpenClaw 发送结束指令，生成正式报告")
#         print("   输入 '/new' = 模拟一个月后患者复诊，开启全新会话")
#         print("   输入 '/quit' = 退出测试")
#
#         user_input = input("🧑 用户输入: ")
#
#         if user_input.strip() == "/quit":
#             break
#
#         elif user_input.strip() == "/new":
#             new_session = input("🔄 请输入新的会话 ID (例如 session_B): ")
#             current_session = new_session.strip()
#             print("✅ 已开启新就诊流程（记忆已清空，但历史档案保留）。")
#             continue
#
#         elif user_input.strip() == "/end":
#             # 触发结案生成报告的流程
#             answer = simulate_openclaw_call(current_patient, current_session, "", action="finalize")
#             print("\n" + "=" * 40)
#             print(f"📝[正式出具的康复报告]:\n{answer}")
#             print("=" * 40 + "\n")
#             continue
#
#         # 正常对话聊天 (action="chat")
#         answer = simulate_openclaw_call(current_patient, current_session, user_input, action="chat")
#         print(f"\n🤖 [助手回复]:\n{answer}")
