"""
企业级RESTful API - 医疗场景专用
基于FastAPI的标准化API设计
"""

from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
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

logger = logging.getLogger(__name__)

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

# API版本控制
API_VERSION = "v1"

app = FastAPI(
    title="Medical Case Management API",
    description="术后康复指南智能提取与管理API",
    version="1.0",
    docs_url=f"/{API_VERSION}/docs",
    redoc_url=f"/{API_VERSION}/redoc"
)

# 全局依赖（模拟数据库连接）
def get_db():
    # 这里应该是你的实际数据库连接
    return "mock_db_connection"

# 响应模型
class APIResponse(BaseModel):
    """标准API响应格式"""
    code: int = Field(200, description="状态码")
    message: str = Field("success", description="消息")
    data: Optional[Any] = Field(None, description="返回数据")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class CaseCreateRequest(BaseModel):
    """创建病例请求"""
    patient_id: str = Field(..., description="患者ID")
    raw_text: str = Field(..., description="原始病历文本")
    department: str = Field("康复科", description="科室")
    category: str = Field("general", description="病例类别")

class CaseResponse(BaseModel):
    """病例响应"""
    case_id: str = Field(..., description="病例ID")
    patient_id: str = Field(..., description="患者ID")
    diagnosis: str = Field(..., description="诊断结果")
    extraction_confidence: float = Field(..., description="提取置信度")
    created_at: str = Field(..., description="创建时间")
    status: str = Field("extracted", description="状态")

class ExtractionAssessment(BaseModel):
    """提取评估结果"""
    overall_confidence: float = Field(..., description="整体置信度")
    quality_level: str = Field(..., description="质量等级")
    critical_issues: List[str] = Field(default_factory=list, description="关键问题")
    category_scores: Dict[str, float] = Field(default_factory=dict, description="各分类得分")

# 路由定义
@app.post(f"/{API_VERSION}/cases", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def create_case(
    request: CaseCreateRequest,
    db: str = Depends(get_db)
):
    """
    创建新病例（支持批量）
    POST /v1/cases
    """
    try:
        # 这里调用你的现有提取逻辑
        from project.rehab_core.extractor import extract_medical_info
        from project.rehab_core.schema_manager import create_extraction_result
        
        # 1. 提取信息
        extracted_data = extract_medical_info(request.raw_text, request.patient_id)
        
        # 2. 创建标准化提取结果
        extraction_result = create_extraction_result(
            raw_data=extracted_data,
            original_text=request.raw_text,
            version="3.0",
            category=request.category
        )
        
        # 3. 保存到数据库（你的原有逻辑）
        # save_case_to_db(extracted_data, request.patient_id, request.department)
        
        # 4. 构建响应
        response_data = {
            "case_id": extraction_result.extraction_id,
            "patient_id": request.patient_id,
            "diagnosis": extracted_data.get("诊断结果_最终诊断", "未知"),
            "extraction_confidence": extraction_result.confidence_score,
            "created_at": extraction_result.extracted_at,
            "status": "extracted"
        }
        
        return APIResponse(
            code=201,
            message="病例创建成功",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"创建病例失败: {e}")
        return APIResponse(
            code=500,
            message=f"创建失败: {str(e)}",
            data=None
        )

@app.get(f"/{API_VERSION}/cases/{case_id}", response_model=APIResponse)
async def get_case(case_id: str, db: str = Depends(get_db)):
    """
    获取单个病例详情
    GET /v1/cases/{case_id}
    """
    try:
        # 这里调用你的数据库查询逻辑
        from project.rehab_core.PatientRecordManager import patient_db
        
        case_data = patient_db.get_case_detail(case_id)
        
        return APIResponse(
            code=200,
            message="获取成功",
            data=case_data
        )
        
    except Exception as e:
        return APIResponse(
            code=404,
            message=f"病例不存在: {str(e)}",
            data=None
        )

@app.post(f"/{API_VERSION}/cases/batch", response_model=APIResponse)
async def batch_create_cases(
    requests: List[CaseCreateRequest],
    db: str = Depends(get_db)
):
    """
    批量创建病例
    POST /v1/cases/batch
    """
    results = []
    for i, request in enumerate(requests):
        try:
            # 复用单个创建逻辑
            result = await create_case(request, db)
            results.append({
                "index": i,
                "case_id": result.data.get("case_id") if result.data else None,
                "status": "success" if result.code == 201 else "failed",
                "message": result.message
            })
        except Exception as e:
            results.append({
                "index": i,
                "status": "failed",
                "message": str(e)
            })
    
    return APIResponse(
        code=200,
        message=f"批量处理完成: {len([r for r in results if r['status']=='success'])}/{len(results)}",
        data={"results": results}
    )

@app.get(f"/{API_VERSION}/health")
async def health_check():
    """健康检查"""
    return APIResponse(
        code=200,
        message="API服务正常运行",
        data={
            "version": API_VERSION,
            "timestamp": datetime.now().isoformat(),
            "services": ["database", "llm", "retrieval"]
        }
    )

# 添加OpenAPI规范
@app.get(f"/{API_VERSION}/openapi.json")
async def get_openapi_json():
    """获取OpenAPI规范"""
    return app.openapi()

# 启动配置
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)