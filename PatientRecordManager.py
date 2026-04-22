import sqlite3
import json
import uuid
import os
from datetime import datetime
from typing import List, Dict, Optional, Union
import psycopg2

class PatientRecordManager:
    """专门用于管理患者长期病历和历史最终方案的管理器"""
    
    def __init__(self, db_path: str = None, db_url: str = None):
        """
        初始化数据库连接
        支持两种模式：
        - db_path: SQLite路径（开发模式）
        - db_url: PostgreSQL连接字符串（生产模式）
        """
        if db_url and db_url.startswith("postgresql://"):
            # PostgreSQL模式
            self.db_type = "postgresql"
            self.conn = psycopg2.connect(db_url)
            self.cursor = self.conn.cursor()
            print(f"✅ 使用PostgreSQL数据库（高并发支持）")
        else:
            # SQLite模式
            self.db_type = "sqlite"
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
            self.db_path = db_path
            print(f"⚠️ 使用SQLite数据库")
    
    def _execute_query(self, query: str, params: tuple = None):
        """执行查询，兼容两种数据库"""
        try:
            if self.db_type == "postgresql":
                self.cursor.execute(query, params or ())
            else:
                self.cursor.execute(query, params or ())
            return self.cursor
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def _init_output_table(self):
        """初始化用于保存 Agent 最终生成的方案的表"""
        if self.db_type == "postgresql":
            self._execute_query("""
                CREATE TABLE IF NOT EXISTS case_rehab_plans (
                    plan_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    session_id TEXT,
                    final_plan_content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        else:
            self._execute_query("""
                CREATE TABLE IF NOT EXISTS case_rehab_plans (
                    plan_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    session_id TEXT,
                    final_plan_content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        self.conn.commit()
# 长期记忆是我们的每一个患者的病例信息，和他们创建出来的康复指南

class PatientRecordManager:
    """专门用于管理患者长期病历和历史最终方案的管理器"""
    def __init__(self, db_path):
        # 实际使用中可以连接你的 SQLite (rehab.db)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.db_path = db_path

    def _init_output_table(self):
        """初始化用于保存 Agent 最终生成的方案的表"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # 创建方案归档表（如果不存在）
            # 外键 case_id 关联到你截图中的 cases 表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS case_rehab_plans (
                    plan_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    session_id TEXT,
                    final_plan_content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def get_unresolved_cases(self, patient_id: str) -> List[Dict]:
        """
        [给 OpenClaw 使用]
        查询某患者当前【尚未出具康复方案】的病例，供用户选择
        """
        query = """
            SELECT c.case_id, c.created_at, c.diagnosis, c.treatment_summary 
            FROM cases c
            LEFT JOIN case_rehab_plans p ON c.case_id = p.case_id
            WHERE c.patient_id = ? AND p.plan_id IS NULL
            ORDER BY c.created_at DESC
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # 方便按列名获取数据
            cursor = conn.cursor()
            cursor.execute(query, (patient_id,))
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def get_all_cases(self, patient_id: str):
        """[给 OpenClaw 使用] 查询患者的所有病例，并标记是否已有方案"""
        query = """
            SELECT c.case_id, c.diagnosis, c.created_at, p.plan_id, p.created_at as guide_created_at, p.session_id
            FROM cases c
            LEFT JOIN case_rehab_plans p ON c.case_id = p.case_id
            WHERE c.patient_id = ?
            ORDER BY c.created_at DESC
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.cursor().execute(query, (patient_id,)).fetchall()

            result = []
            for r in rows:
                case_info = dict(r)
                # 添加状态标记
                case_info['status'] = "已出方案(可复诊/调整)" if r['plan_id'] else "未出方案(初诊)"
                case_info['guide_created_time'] = r['guide_created_at']
                result.append(case_info)
            return result

    def get_case(self, case_id: str):
        """[给 OpenClaw 使用] 查询患者的特定病例 只需要病例ID即可"""
        query = """
            SELECT c.case_id, c.diagnosis, c.created_at, c.department, c.age, c.sex
            FROM cases c
            where c.case_id = ?
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.cursor().execute(query, (case_id,)).fetchall()

            result = []
            for r in rows:
                case_info = dict(r)
                result.append(case_info)
            return result

    def get_current_case_plan(self, case_id: str) -> str:
        """获取当前正在处理的病例的【最新已有方案】"""
        query = """
            SELECT final_plan_content FROM case_rehab_plans 
            WHERE case_id = ? ORDER BY created_at DESC LIMIT 1
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.cursor().execute(query, (case_id,)).fetchone()
            return row[0] if row else ""

    def get_other_historical_plans(self, patient_id: str, current_case_id: str) -> str:
        """获取该患者【其他病例】的康复方案（用于最终报告的安全排雷）"""
        query = """
            SELECT c.diagnosis, p.final_plan_content 
            FROM cases c
            JOIN case_rehab_plans p ON c.case_id = p.case_id
            WHERE c.patient_id = ? AND c.case_id != ?
            ORDER BY c.created_at DESC
        """
        with sqlite3.connect(self.db_path) as conn:
            records = conn.cursor().execute(query, (patient_id, current_case_id)).fetchall()

        if not records: return ""
        history = ""
        for diag, plan in records:
            history += f"- 历史确诊: {diag} | 核心方案: {plan[:100]}...\n"
        return history

    def get_patient_history(self, patient_id: str, current_case_id: str = None) -> str:
        """
        [给 Agent 长期记忆使用]
        提取该患者所有【历史的】病例及康复方案，喂给 LLM 做参考
        """
        query = """
            SELECT c.created_at, c.diagnosis, p.final_plan_content
            FROM cases c
            JOIN case_rehab_plans p ON c.case_id = p.case_id
            WHERE c.patient_id = ? 
        """
        params = [patient_id]

        # 排除当前正在处理的 case_id（如果有）
        if current_case_id:
            query += " AND c.case_id != ?"
            params.append(current_case_id)

        query += " ORDER BY c.created_at ASC"  # 按时间正序，LLM更好理解时间线

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            history_records = cursor.fetchall()

        if not history_records:
            return "无历史康复记录。"

        # 将历史记录组装成供 LLM 阅读的文本格式
        history_text = "【患者历史康复档案】：\n"
        for i, (created_at, diagnosis, plan) in enumerate(history_records, 1):
            history_text += f"\n--- 历史记录 {i} ({created_at}) ---\n"
            history_text += f"确诊病症：{diagnosis}\n"
            # 摘要一下方案内容，避免 Token 过长（可选）
            plan_preview = plan[:300] + "..." if len(plan) > 300 else plan
            history_text += f"历史康复方案：\n{plan_preview}\n"

        return history_text

    def save_final_plan(self, case_id: str, session_id: str, plan_content: str):
        """[MCP 工具 finalize 动作调用]
        当这一轮对话结束，将正式报告保存并关联到对应的病例
        """
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        query = """
            INSERT INTO case_rehab_plans (plan_id, case_id, session_id, final_plan_content)
            VALUES (?, ?, ?, ?)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (plan_id, case_id, session_id, plan_content))
                conn.commit()
            print(f"✅ [数据库] 成功保存 Case: {case_id} 的最终康复方案！")
        except Exception as e:
            print(f"❌ [数据库] 保存康复方案失败：{e}")

patient_db = PatientRecordManager("D:\\Desktop\\company\\Agent-rag\\agentic-rag-for-dummies\\project\\db\\rehab.db")
