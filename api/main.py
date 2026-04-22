#!/usr/bin/env python3
# project/api/main.py
# MCP 2.0 兼容服务：/mcp/execute + /mcp/tools（必选！）
# 保留你所有原有代码，仅新增 /mcp/tools 路由
import json
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn

from project.prompts.schema import SCHEMA
from project.rag_agent.extractor import merge_results, extract_section

import os
import json
import sqlite3
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 导入我们刚刚写的 Agent 调度和数据库管理器
from project.rehab_core.PatientRecordManager import PatientRecordManager
# from project.rehab_core.test_agent import simulate_openclaw_call
from project.rehab_core.config import SQLITE_DB_PATH

import uuid
import chromadb
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from project.rehab_core.tools import ToolFactory
from project.rehab_core import config
from project.rehab_core.PatientRecordManager import patient_db
from project.rehab_core import graph
# ==========================================
# 1. 挂载测试用的 Checkpointer（图的编译端）
# （这里假设你已经定义好了 agent_builder）
# ==========================================
from project.db.chroma_loader import ChromaFinalBuilder
from chromadb.config import Settings
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
# from project.rehab_core.kafka_producer import KafkaTaskProducer
# ==========================================
# 1. 全局变量声明 (在启动时被赋值，在请求时被使用)
# ==========================================
db_manager = None
rehab_agent_graph = None
memory_conn = None

MODEL_PATH = "D:/Desktop/company/Agent-rag/agentic-rag-for-dummies/models/Hu0922/BGE_Medical"
DB_PATH = config.SQLITE_DB_PATH
CHROMA_PATH = config.CHROMA_PATH
OPENAI_API_KEY = 'sk-d2cf853e524c4f5fb8c604906a781faa'
# ==========================================
# 2. 服务器生命周期管理 (只在启动时运行一次)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_manager, rehab_agent_graph, memory_conn

    print("🚀 [系统启动] 正在初始化基础设施...")

    # 初始化Kafka生产者
    # kafka_producer = KafkaTaskProducer()

    # 2.1 初始化病例数据库管理器
    db_manager = PatientRecordManager(db_path=config.SQLITE_DB_PATH)

    # 2.2 初始化 ChromaDB 向量库
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

    # 2.3 实例化工具工厂
    tool = ToolFactory(collection)
    tool_list = tool.create_tools()

    # 2.4 初始化 LLM (带上绕过本地代理的配置，防止 EOF 报错)
    custom_http_client = httpx.Client(proxies={"http://": None, "https://": None}, verify=False, timeout=60.0)
    llm = ChatOpenAI(
        model="deepseek-r1-distill-llama-8b",
        api_key=OPENAI_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        http_client=custom_http_client
    )

    # 2.5 实例化 LangGraph 的持久化记忆 (SQLite Checkpointer)
    # 只要 session_id 没变，聊多久都能接上
    conn = sqlite3.connect("./data/langgraph_checkpoints.db", check_same_thread=False)
    memory = SqliteSaver(conn)

    # 2.6 编译构建出 全局唯一 的 Agent Graph 引擎
    rehab_agent_graph = graph.build_graph(llm, tool_list, memory)
    print("✅ [系统启动完毕] MCP 服务已就绪，等待 OpenClaw 调用。")

    yield  # 交出控制权，开始处理外部发来的 HTTP 请求

    # --- 下面的代码在服务器关闭时执行 ---
    print("🛑 [系统关闭] 正在清理资源...")
    if conn:
        conn.close()

# === 1. 定义 MCP 工具清单 (OpenClaw 会读取这里) ===
MCP_TOOLS = [
    {
        "name": "medical_extract",
        "description": "从术后病程记录中抽取规定的信息指标，并自动在数据库中为您建立新的就诊病例。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "完整病历文本"},
                "patient_id": {"type": "string", "description": "患者的唯一ID"},
                "department_hint": {"type": "string", "description": "科室提示", "default": "康复科"}
            },
            "required": ["text", "patient_id"]
        }
    },
    {
        "name": "rehab_consultation",
        "description": """
        术后康复方案交互助手。
        【使用规范】：
        1. 当用户从列表中刚选择好一个病例时，请调用此工具，传入选中的 case_id，并设置 query='用户已锁定此病例，请开始问诊'。底层的 Agent 会自动识别是初诊还是复诊，并向用户发问。
        2. 在随后的多轮对话中，将用户说的话原封不动地放入 query 参数，action 保持 'chat'。
        3. 当你主动询问用户是否还有补充，且用户表示‘没有了’，或者明确要求出具最终报告时，必须传入 action='finalize' 结束会话并生成最终报告。
        """,
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "患者的唯一ID"},
                "case_id": {"type": "string", "description": "当前正在处理的病例ID（可从上文获取）"},
                "session_id": {"type": "string", "description": "本次对话的会话ID"},
                "query": {"type": "string", "description": "患者当前的话。如果是finalize，可传空。"},
                "action": {"type": "string", "enum": ["chat", "finalize"],
                           "description": "chat: 继续对话; finalize: 患者确认无其他问题，生成正式报告并落库。"}
            },
            "required": ["patient_id", "case_id", "session_id", "query", "action"]
        }
    },
    {
        "name": "query_patient_cases",
        "description": "查询患者的病例列表、诊断结果及康复方案状态（已出方案/未出方案）。在开始诊断或是说构建康复指南前可调用此工具让用户选择病例。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "患者的唯一ID"}
            },
            "required": ["patient_id"]
        }
    },
]


# === 2. 适配新表的病例保存函数 ===
def save_case_to_db(extracted: dict, patient_id: int, department: str = "髋部骨折") -> dict:
    """
    修改后的保存逻辑：直接存入 SQLite，而不是存 JSON
    """
    case_id = f"case_{uuid.uuid4().hex[:8]}"

    # 假设你的抽取模型提取出了诊断结果
    diagnosis = extracted.get("诊断结果_最终诊断", "未知诊断(需补充)")
    age = extracted.get("基础信息_年龄")
    sex = extracted.get("基础信息_性别")

    query = """
        INSERT INTO cases (case_id, patient_id, diagnosis, department, age, sex, vitals)
        VALUES (?, ?, ?, ?, ?)
    """
    try:
        with sqlite3.connect(config.SQLITE_DB_PATH) as conn:
            conn.cursor().execute(query, (case_id, patient_id, diagnosis, department, age, sex, extracted))
            conn.commit()

        return {
            "case_id": case_id,
            "patient_id": patient_id,
            "message": f"病例提取成功并已入库！诊断: {diagnosis}。请保存此 case_id 用于后续康复咨询。"
        }
    except Exception as e:
        raise Exception(f"数据库保存失败: {str(e)}")


# === 数据库增强配置 ===
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import sqlite3

# 检测数据库类型
DB_URL = os.getenv("DATABASE_URL", config.SQLITE_DB_PATH)

if DB_URL.startswith("postgresql://"):
    # PostgreSQL配置（高并发关键）
    engine = create_engine(
        DB_URL,
        poolclass=QueuePool,
        pool_size=20,           # 高并发连接池
        max_overflow=10,        # 最大溢出连接
        pool_recycle=3600,      # 连接回收（防连接泄漏）
        pool_pre_ping=True,     # 连接前检测
        echo=False
    )
    print("✅ 使用PostgreSQL数据库（高并发支持）")
else:
    # SQLite回退
    engine = create_engine(f"sqlite:///{DB_URL}", echo=False)
    print("⚠️  使用SQLite数据库（开发模式）")

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# def get_db():
#     """依赖注入：数据库会话"""
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# === FastAPI App 设置 ===
app = FastAPI(
    title="MCP Clinical Agent", 
    version="1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

import json
import asyncio
from fastapi.responses import StreamingResponse


# === 新增：流式生成器 用于前端内容的流式输出 ===
async def stream_agent_execution(patient_id: str, case_id: str, session_id: str, user_input: str, action: str):
    """LangGraph 异步事件流生成器"""
    global rehab_agent_graph, db_manager

    # 1. 组装状态 (和原来一样)
    current_illness = db_manager.get_case(case_id)
    current_plan = db_manager.get_current_case_plan(case_id)
    other_plans = db_manager.get_other_historical_plans(patient_id, case_id)

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

    try:
        # 2. 使用 astream_events 监听图中发生的所有事件 (关键点：version="v2")
        async for event in rehab_agent_graph.astream_events(initial_state, config, version="v2"):
            kind = event["event"]

            # 拦截 1：工具调用开始（推送给前端显示：正在查询指南...）
            if kind == "on_tool_start":
                tool_name = event["name"]
                payload = {"type": "status", "content": f"\n🔍 正在调用工具: [{tool_name}]...\n"}
                yield f"data: {json.dumps(payload)}\n\n"

            # 拦截 2：大模型打字机输出 Token
            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    payload = {"type": "token", "content": chunk.content}
                    yield f"data: {json.dumps(payload)}\n\n"

        # 3. 如果是 finalize 动作，确保结束后将结果落库
        if action == "finalize":
            # 注意：流式输出结束后，可以通过 get_state 拿到最终图状态来落库
            final_state = rehab_agent_graph.get_state(config).values
            final_report = final_state.get("final_answer", "")
            db_manager.save_final_plan(case_id, session_id, final_report)

            yield f"data: {json.dumps({'type': 'status', 'content': '✅ 报告已生成并归档。'})}\n\n"

        # 告诉前端流结束了
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


def simulate_openclaw_call(patient_id, case_id, session_id, user_input, action="chat"):
    """
    现在这个函数不需要自己去实例化图了，直接用全局的 rehab_agent_graph
    """
    global rehab_agent_graph, db_manager

    # 获取业务数据
    current_illness = db_manager.get_case(case_id)
    current_plan = db_manager.get_current_case_plan(case_id)
    other_plans = db_manager.get_other_historical_plans(patient_id, case_id)

    # 构造当前这【一帧】的状态
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

    # 依靠 thread_id，LangGraph 会自动去 agent_memory.db 里把过去的消息捞出来并入 state
    # 因为我们的每轮信息或者说每次结束对话，对话激活总结，那这样每次对话开启，只需要获取之前的康复指南就足够了，
    # 当然这里就需要我们调用这个Agent的部分，需要有超过一段时间执行总结的，这样反而可以利用外部API调用的池子，
    # 断开链接，或是说需要给其他人用，就最后传一个finalize
    config = {"configurable": {"thread_id": session_id}}

    # 驱动全局引擎运行
    result = rehab_agent_graph.invoke(initial_state, config=config)

    if action == "finalize":
        final_report = result.get("final_answer", "")
        db_manager.save_final_plan(case_id, session_id, final_report)
        return final_report

    return result["messages"][-1].content

    # 如果我们用Kafka架构处理同步与异步，
    # if action == "chat":
    #     # 同步模式：实时对话
    #     result = rehab_agent_graph.invoke(initial_state, config=config)
    #     return result["messages"][-1].content
    #
    # elif action == "finalize":
    #     # 异步模式：总结生成
    #     try:
    #         # 1. 获取上下文数据
    #         context_data = {
    #             "current_plan": current_plan,
    #             "other_plans": other_plans,
    #             "messages": initial_state["messages"],
    #             "session_id": session_id,
    #             "case_id": case_id,
    #             "patient_id": patient_id
    #         }
    #
    #         # 2. 发送Kafka消息（面试项目中可模拟）
    #         execution_id = kafka_producer.send_finalize_task(
    #             session_id=session_id,
    #             case_id=case_id,
    #             patient_id=patient_id,
    #             context_data=context_data,
    #             priority="high"
    #         )
    #
    #         # 3. 返回异步任务ID
    #         return {
    #             "status": "async_processing",
    #             "execution_id": execution_id,
    #             "message": "康复指南正在生成中，请稍后查看结果",
    #             "estimated_time": "30-60秒"
    #         }
    #
    #     except Exception as e:
    #         # 备用方案：降级为同步处理
    #         result = rehab_agent_graph.invoke(initial_state, config=config)
    #         final_report = result.get("final_answer", "")
    #         db_manager.save_final_plan(case_id, session_id, final_report)
    #         return final_report


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]


# === 3. 路由定义 ===

@app.get("/mcp/tools")
async def get_mcp_tools():
    """你的 TypeScript 前端会调用这个接口拉取工具列表"""
    return MCP_TOOLS


@app.post("/mcp/execute")
async def execute_mcp_tool(call: ToolCall):
    """
    前端解析出需要调用的工具和参数后，会 POST 这个接口
    根据工具名进行分发
    """
    try:
        args = call.arguments

        # 1. 病历提取建档工具
        if call.name == "medical_extract":
            text = args.get("text")
            patient_id = args.get("patient_id")
            dept = args.get("department_hint", "康复科")

            # 【这里放你的真实抽取逻辑，此处用假数据模拟】
            sections = ["基础信息", "病史", "常规检查", "专科检查", "诊断结果"]
            partial_results = [extract_section(text, sec) for sec in sections]
            # 3. 组装
            result = merge_results(partial_results)

            # 保存到 SQLite
            result_info = save_case_to_db(result, patient_id, dept)
            # 兼容 TS 前端 expecting {"result": ...}
            return {"result": result_info, "status": "success"}

            # === 工具 2: 新增的病历查询工具 ===
        elif call.name == "query_patient_cases":
            patient_id = args.get("patient_id")
            if not patient_id:
                raise HTTPException(400, "需要 patient_id")

            # 直接调用我们之前写的数据库管理器方法
            cases = db_manager.get_all_cases(patient_id)

            return {
                "status": "success",
                "result": {
                    "patient_id": patient_id,
                    "total_cases": len(cases),
                    "cases": cases  # 这里面自带 case_id, diagnosis, status 等字段
                }
            }

        # 2. 交互式康复指南 RAG 代理
        elif call.name == "rehab_consultation":
            patient_id = args.get("patient_id")
            case_id = args.get("case_id")
            session_id = args.get("session_id")
            query = args.get("query", "")
            action = args.get("action", "chat")

            # 异常校验
            if not all([patient_id, case_id, session_id]):
                raise HTTPException(400, "缺少必要的标识符 (patient_id, case_id, session_id)")

            print(f"📥 收到 MCP 调用 -> 动作:{action} | 病患:{patient_id} | 病例:{case_id}")

            # 异步调用工具部分
            return StreamingResponse(
                stream_agent_execution(patient_id, case_id, session_id, query, action),
                media_type="text/event-stream"
            )
            # 同步调用工具部分，
            # # 调用我们完善好的核心 Agent
            # answer = simulate_openclaw_call(patient_id, case_id, session_id, query, action)
            #
            # # 返回给前端展示（如果是 finalize，返回的是长篇大报告）
            # return {"result": answer, "status": "success"}

        else:
            raise HTTPException(404, f"未知的工具名称: {call.name}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"执行失败: {str(e)}")


# 启动命令： uvicorn main:app --host 0.0.0.0 --port 8000

@app.get("/health")
async def health():
    return {"status": "healthy", "tools_count": len(MCP_TOOLS)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)