# session_manager.py
import json
import os
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import config


class SessionManager:
    """会话管理器"""

    def __init__(self, storage_path: str = config.SESSION_STORAGE_PATH):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def create_or_load_session(self, session_id: str, external_data: Optional[Dict] = None) -> Dict:
        """
        创建或加载会话

        Args:
            session_id: OpenClaw提供的会话ID
            external_data: OpenClaw的会话数据

        Returns:
            会话数据
        """
        session_file = self.storage_path / f"{session_id}.json"

        if session_file.exists():
            # 加载现有会话
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            # 更新时间戳
            session_data["last_updated"] = datetime.now().isoformat()

            # 如果提供了外部数据，更新外部数据部分
            if external_data:
                session_data["external_data"] = external_data

            print(f"📁 加载现有会话: {session_id}")
        else:
            # 创建新会话
            session_data = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "external_data": external_data or {},
                "user_context": {},
                "conversation_history": [],
                "search_history": [],
                "retrieved_parents": [],
                "agent_answers": [],
                "context_summary": ""
            }
            print(f"🆕 创建新会话: {session_id}")

        # 保存会话
        self._save_session(session_data)

        return session_data

    def save_rehabilitation_plan(self, session_id: str, question: str,
                                 answer: str, user_context: Dict):
        """保存康复方案"""
        session_data = self._load_or_create_session(session_id, user_context)

        # 创建方案记录
        plan = {
            "id": f"plan_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "user_context": user_context
        }

        # 添加到历史
        if "rehabilitation_plans" not in session_data:
            session_data["rehabilitation_plans"] = []

        session_data["rehabilitation_plans"].append(plan)

        # 限制数量
        if len(session_data["rehabilitation_plans"]) > 20:
            session_data["rehabilitation_plans"] = session_data["rehabilitation_plans"][-20:]

        session_data["last_updated"] = datetime.now().isoformat()
        self._save_session(session_data)

    def get_historical_summary(self, session_id: str) -> str:
        """获取历史方案摘要"""
        session_data = self._load_session(session_id)
        if not session_data or "rehabilitation_plans" not in session_data:
            return ""

        plans = session_data["rehabilitation_plans"]
        if not plans:
            return ""

        # 只返回最近3个方案的摘要
        recent_plans = plans[-3:]
        summary = "历史康复方案：\n"

        for i, plan in enumerate(recent_plans, 1):
            date = plan.get("timestamp", "").split("T")[0]
            question = plan.get("question", "")[:50]
            summary += f"{i}. [{date}] {question}...\n"

        return summary

    def update_session(self, session_id: str, updates: Dict):
        """更新会话"""
        session_data = self._load_session(session_id)
        if session_data:
            session_data.update(updates)
            session_data["last_updated"] = datetime.now().isoformat()
            self._save_session(session_data)
            return session_data
        return None

    def add_conversation(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """添加对话记录"""
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,  # "user" 或 "assistant"
            "content": content,
            "metadata": metadata or {}
        }

        session_data = self._load_session(session_id)
        if session_data:
            session_data["conversation_history"].append(conversation_entry)
            # 只保留最近50条对话记录
            if len(session_data["conversation_history"]) > 50:
                session_data["conversation_history"] = session_data["conversation_history"][-50:]

            session_data["last_updated"] = datetime.now().isoformat()
            self._save_session(session_data)

    def add_search(self, session_id: str, query: str, results: List[Dict], filters: Optional[Dict] = None):
        """添加搜索记录"""
        search_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "filters": filters or {},
            "result_count": len(results),
            "parent_ids": [r.get("parent_id") for r in results if r.get("parent_id")],
            "results_preview": [r.get("content", "")[:100] for r in results[:3]]  # 前3个结果预览
        }

        session_data = self._load_session(session_id)
        if session_data:
            session_data["search_history"].append(search_entry)

            # 更新已检索的父chunk
            parent_ids = search_entry["parent_ids"]
            for pid in parent_ids:
                if pid and pid not in session_data["retrieved_parents"]:
                    session_data["retrieved_parents"].append(pid)

            session_data["last_updated"] = datetime.now().isoformat()
            self._save_session(session_data)

    def get_session_summary(self, session_id: str) -> Dict:
        """获取会话摘要"""
        session_data = self._load_session(session_id)
        if not session_data:
            return {}

        return {
            "session_id": session_data["session_id"],
            "created_at": session_data["created_at"],
            "last_updated": session_data["last_updated"],
            "conversation_count": len(session_data.get("conversation_history", [])),
            "search_count": len(session_data.get("search_history", [])),
            "retrieved_parent_count": len(session_data.get("retrieved_parents", [])),
            "user_context": session_data.get("user_context", {}),
            "recent_conversations": session_data.get("conversation_history", [])[-3:],
            "recent_searches": session_data.get("search_history", [])[-3:]
        }

    def cleanup_old_sessions(self, days: int = 30):
        """清理过期会话"""
        cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)

        for session_file in self.storage_path.glob("*.json"):
            file_mtime = session_file.stat().st_mtime
            if file_mtime < cutoff_time:
                try:
                    session_file.unlink()
                    print(f"🗑️ 清理过期会话: {session_file.name}")
                except Exception as e:
                    print(f"❌ 清理会话失败 {session_file.name}: {e}")

    def _load_session(self, session_id: str) -> Optional[Dict]:
        """加载会话"""
        session_file = self.storage_path / f"{session_id}.json"

        if session_file.exists():
            with open(session_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_session(self, session_data: Dict):
        """保存会话"""
        session_file = self.storage_path / f"{session_data['session_id']}.json"

        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

    def _load_or_create_session(self, session_id: str, user_context: Dict) -> Dict:
        """加载或创建会话"""
        session_data = self._load_session(session_id)

        if not session_data:
            session_data = {
                "session_id": session_id,
                "user_context": user_context,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }

        return session_data

# 全局会话管理器实例
session_manager = SessionManager()
