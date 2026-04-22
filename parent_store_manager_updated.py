# project/db/parent_store_manager_updated.py
"""
更新后的父文档管理器
适配层次化索引的存储结构
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import glob


class ParentStoreManager:
    """父文档管理器（更新版）"""

    def __init__(self, parent_store_path: str = "./data/parent_docs_embedding"):
        self.parent_store_path = Path(parent_store_path)

        if not self.parent_store_path.exists():
            self.parent_store_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 创建父文档存储目录: {self.parent_store_path}")

    def load_content(self, parent_id: str) -> Optional[Dict[str, Any]]:
        """根据parent_id加载父文档内容"""
        try:
            # 查找匹配的JSON文件
            pattern = os.path.join(self.parent_store_path, f"*{parent_id}*.json")
            files = glob.glob(pattern)

            if not files:
                # 尝试直接文件名
                direct_path = self.parent_store_path / f"{parent_id}.json"
                if direct_path.exists():
                    files = [str(direct_path)]
                else:
                    # 尝试查找包含parent_id的文件
                    all_files = glob.glob(os.path.join(self.parent_store_path, "*.json"))
                    for file in all_files:
                        with open(file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if data.get('metadata', {}).get('parent_id') == parent_id:
                                files = [file]
                                break

            if not files:
                print(f"⚠️ 未找到父文档: {parent_id}")
                return None

            # 加载第一个匹配的文件
            with open(files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 标准化返回格式
            return {
                "parent_id": data.get('metadata', {}).get('parent_id', parent_id),
                "content": data.get('page_content', ''),
                "metadata": data.get('metadata', {})
            }

        except Exception as e:
            print(f"❌ 加载父文档失败 {parent_id}: {e}")
            return None

    def load_content_many(self, parent_ids: List[str]) -> List[Dict[str, Any]]:
        """批量加载父文档"""
        results = []
        for parent_id in parent_ids:
            content = self.load_content(parent_id)
            if content:
                results.append(content)

        return results

    def search_by_title(self, title_keyword: str) -> List[Dict[str, Any]]:
        """根据标题关键词搜索父文档"""
        results = []

        try:
            all_files = glob.glob(os.path.join(self.parent_store_path, "*.json"))

            for file in all_files:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                title = data.get('metadata', {}).get('title', '')
                if title_keyword.lower() in title.lower():
                    results.append({
                        "parent_id": data.get('metadata', {}).get('parent_id', ''),
                        "title": title,
                        "content_preview": data.get('page_content', '')[:100] + "...",
                        "file_path": file
                    })

        except Exception as e:
            print(f"❌ 标题搜索失败: {e}")

        return results

    def get_all_parents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取所有父文档"""
        results = []

        try:
            all_files = glob.glob(os.path.join(self.parent_store_path, "*.json"))

            for i, file in enumerate(all_files[:limit]):
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                results.append({
                    "parent_id": data.get('metadata', {}).get('parent_id', f"file_{i}"),
                    "title": data.get('metadata', {}).get('title', '未知'),
                    "category": data.get('metadata', {}).get('category', '未知'),
                    "content_length": len(data.get('page_content', ''))
                })

        except Exception as e:
            print(f"❌ 获取所有父文档失败: {e}")

        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            all_files = glob.glob(os.path.join(self.parent_store_path, "*.json"))

            # 计算平均长度
            total_length = 0
            for file in all_files:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    total_length += len(data.get('page_content', ''))

            avg_length = total_length / len(all_files) if all_files else 0

            return {
                "total_parents": len(all_files),
                "storage_path": str(self.parent_store_path),
                "avg_content_length": avg_length
            }

        except Exception as e:
            return {
                "error": str(e),
                "storage_path": str(self.parent_store_path)
            }