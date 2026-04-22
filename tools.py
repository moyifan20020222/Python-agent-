# 在tools.py中修改，集成会话管理
from typing import List, Dict, Optional, Union
from langchain_core.tools import tool
from project.rehab_core.Medical_chunk import ParentChunkStore
# from session_manager import session_manager
# import config
from project.rehab_core.performance_monitor import custom_performance_monitor

class ToolFactory:

    def __init__(self, collection):
        self.collection = collection
        self.parent_store_manager = ParentChunkStore()

    def _search_child_chunks(self, query: str, limit: int = 5, filters: Optional[Dict] = None) -> str:
        """搜索与查询相关的子chunk"""
        try:
            custom_performance_monitor.start_rag_query("search_child_chunks")
            # 构建搜索条件
            where_condition = {k: v for k, v in filters.items() if v} if filters else None

            # 执行搜索
            if where_condition:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=limit,
                    where=where_condition
                )
            else:
                results = self.collection.similarity_search_with_score(query, k=limit)

            if not results or (isinstance(results, dict) and not results.get("documents")):
                return "NO_RELEVANT_CHUNKS"

            # 处理结果
            if isinstance(results, dict):
                # ChromaDB查询结果
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                formatted_results = []

                for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                    formatted_results.append({
                        "content": doc,
                        "metadata": meta,
                        "similarity": results.get("distances", [[1.0]])[0][i] if results.get("distances") else 1.0
                    })
            else:
                # 普通相似性搜索结果
                formatted_results = []
                for doc, score in results:
                    formatted_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity": score
                    })

            # # 记录搜索历史
            # if self.session_id:
            #     session_manager.add_search(
            #         self.session_id,
            #         query,
            #         formatted_results,
            #         filters
            #     )

            # 格式化返回
            if not formatted_results:
                custom_performance_monitor.end_rag_query("search_child_chunks", "NO_RELEVANT_CHUNKS")
                return "NO_RELEVANT_CHUNKS"
            custom_performance_monitor.end_rag_query("search_child_chunks", "\n\n".join([
                f"父ID: {result['metadata'].get('parent_id', '未知')}\n"
                f"来源: {result['metadata'].get('source', '未知')}\n"
                f"标题: {result['metadata'].get('title', '无标题')}\n"
                f"疾病: {result['metadata'].get('disease', '未知')}\n"
                f"类别: {result['metadata'].get('category', '未知')}\n"
                f"相似度: {1 - result['similarity']:.4f}\n"
                f"内容: {result['content'][:300]}...\n"
                f"（注：若需完整内容，请使用 retrieve_parent_chunks 工具检索上述 父ID）"
                for result in formatted_results[:limit]  # 只返回前3个结果
            ]))
            return "\n\n".join([
                f"父ID: {result['metadata'].get('parent_id', '未知')}\n"
                f"来源: {result['metadata'].get('source', '未知')}\n"
                f"标题: {result['metadata'].get('title', '无标题')}\n"
                f"疾病: {result['metadata'].get('disease', '未知')}\n"
                f"类别: {result['metadata'].get('category', '未知')}\n"
                f"相似度: {1 - result['similarity']:.4f}\n"
                f"内容: {result['content'][:300]}...\n"
                f"（注：若需完整内容，请使用 retrieve_parent_chunks 工具检索上述 父ID）"
                for result in formatted_results[:limit]  # 只返回前3个结果
            ])

        except Exception as e:
            custom_performance_monitor.end_rag_query("search_child_chunks", f"RETRIEVAL_ERROR: {str(e)}")
            return f"RETRIEVAL_ERROR: {str(e)}"

    def _retrieve_parent_chunks(self, parent_id: Union[str, List[str]]) -> str:
        """检索完整的父chunk内容"""
        try:
            custom_performance_monitor.start_rag_query("retrieve_parent_chunks")
            # 支持单个ID或ID列表
            if isinstance(parent_id, str):
                parent_ids = [parent_id]
            elif isinstance(parent_id, list):
                parent_ids = parent_id
            else:
                return "INVALID_PARENT_ID_FORMAT"

            # 批量检索父chunk
            # parent 是单纯的内容， metadata是所有相关信息
            raw_parents = []
            for pid in parent_ids:
                parent, metadata = self.parent_store_manager.load_parent_chunk(pid)
                if parent:
                    raw_parents.append({parent, metadata})

            if not raw_parents:
                custom_performance_monitor.end_rag_query("retrieve_parent_chunks", "NO_PARENT_DOCUMENT_FOUND")
                return "NO_PARENT_DOCUMENT_FOUND"
            custom_performance_monitor.end_rag_query("retrieve_parent_chunks", "\n\n".join([
                f"父ID: {metadata.get('parent_id', '未知')}\n"
                f"来源: {metadata.get('source', '未知')}\n"
                f"标题: {metadata.get('title', '无标题')}\n"
                f"全文内容:\n{parent}"
                for parent, metadata in raw_parents
            ]))
            # 格式化返回
            return "\n\n".join([
                f"父ID: {metadata.get('parent_id', '未知')}\n"
                f"来源: {metadata.get('source', '未知')}\n"
                f"标题: {metadata.get('title', '无标题')}\n"
                f"全文内容:\n{parent}"
                for parent, metadata in raw_parents
            ])

        except Exception as e:
            custom_performance_monitor.end_rag_query("retrieve_parent_chunks", f"PARENT_RETRIEVAL_ERROR: {str(e)}")
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"

    def create_tools(self) -> List:
        """创建 langchain 兼容的工具列表"""

        @tool("search_child_chunks")
        def search_child_chunks_tool(query: str, limit: int = 5, filters: Optional[Dict] = None) -> str:
            """当你需要查找康复指南、医学知识时调用此工具。返回的是内容的片段和对应的父ID。"""
            return self._search_child_chunks(query, limit, filters)

        @tool("retrieve_parent_chunks")
        def retrieve_parent_chunks_tool(parent_id: Union[str, List[str]]) -> str:
            """当 search_child_chunks 返回了相关的父ID时，调用此工具获取该指南的完整正文。可传入单个ID或ID列表。"""
            return self._retrieve_parent_chunks(parent_id)

        return [search_child_chunks_tool, retrieve_parent_chunks_tool]

