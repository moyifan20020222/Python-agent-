from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
from project.rag_agent.extractor import merge_results, extract_section

app = FastAPI(title="MCP Medical Extractor", version="0.1")


class MCPRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]


class MCPResponse(BaseModel):
    result: Dict[str, Any]
    status: str


@app.post("/mcp/v1/call", response_model=MCPResponse)
async def mcp_call(req: MCPRequest):
    if req.tool_name != "medical_extract":
        raise HTTPException(status_code=404, detail=f"Tool '{req.tool_name}' not found")

    text: str = req.arguments.get("text")
    department_hint: str = req.arguments.get("department_hint", "demo")

    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' in arguments")

    # === 核心：调用你的抽取引擎 ===
    # 实际应替换为：partial_results = [extract_section(text, sec) for sec in sections]
    # 此处为演示（你后续替换为真实LLM调用）
    partial_results = [
        {"现病史_受伤时间": "2024-09-10", "既往史_高血压": "有高血压病10余年"},
        {"体格检查_体温": "36.5℃"},
        {"专科检查-左下肢畸形": "有", "专科检查-肢端血运": "良好"}
    ]
    extracted = merge_results(partial_results)

    return {
        "result": {
            "extracted": extracted,
            "department": department_hint,
            "timestamp": "2024-09-10T16:24:00Z"
        },
        "status": "success"
    }