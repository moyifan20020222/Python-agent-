# project/agents/tool_factory_fixed.py
"""
修复后的ToolFactory，添加了必要的文档字符串
"""
from typing import List, Dict, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
import logging
from langgraph.types import Command
from project.rehab_core.guide_chunker import EmbeddingHierarchicalIndexer
from project.rehab_core.performance_monitor import custom_performance_monitor
from project.rehab_core.retrieval.hybrid_retriever import BM25Retriever, VectorRetriever
from project.rehab_core.retrieval.hybrid_retriever_final import FinalHybridRetrieval
from project.rehab_core.performance_monitor_tool import tool_performance_monitor

logger = logging.getLogger(__name__)


class ToolFactory:
    """修复的工具工厂"""

    def __init__(self, guide_hybrid, faq_hybrid, parent_store_manager):
        """
        初始化

        Args:
            guide_hybrid: 权威指南的混合检索器 (FinalHybridRetrieval)
            faq_hybrid: 患者FAQ的混合检索器 (FinalHybridRetrieval)
            parent_store_manager: 父文档本地 JSON 存储管理器 (Indexer)
        """
        self.guide_hybrid = guide_hybrid
        self.faq_hybrid = faq_hybrid
        self.parent_store_manager = parent_store_manager
        # 整个混合检索需要做到的初始化部分。
        # indexer = EmbeddingHierarchicalIndexer(
        #     db_path="D:\\Desktop\\company\\Agent-rag\\agentic-rag-for-dummies\\project\\db\\rehab.db",
        #     parent_store_path="./data/parent_docs_embedding",
        #     chroma_path="./chroma_embedding",
        #     parent_size=3000,
        #     parent_overlap=200,
        #     child_size=400,
        #     child_overlap=80,
        #     embedding_model="m3e-base",
        #     embedding_device="cpu"
        # )
        # # 两个索引器的初始化构建，直接在构建图的时候调用
        # # result_faq_message = indexer.index_faq_to_chroma()
        # # result = indexer.index_to_chroma()
        # faq_data = indexer.faq_collection.get(include=['documents', 'metadatas'])
        # bm25_docs = []
        # if faq_data and faq_data.get('ids'):
        #     for i in range(len(faq_data['ids'])):
        #         bm25_docs.append({
        #             'page_content': faq_data['documents'][i],
        #             'metadata': faq_data['metadatas'][i],
        #             'id': faq_data['ids'][i]  # 确保有 ID 字段用于混合融合
        #         })
        #
        # # 2. 实例化两个检索器
        # bm25Retriever = BM25Retriever(documents=bm25_docs)
        # vectorRetriever = VectorRetriever(
        #     collection=indexer.faq_collection,  # 传入 FAQ 向量库
        #     embedding_model=indexer.embedding_function  # 传入 embedding 模型
        # )
        #
        # # 3. 实例化混合检索
        # self.hybrid_retrieval = FinalHybridRetrieval(
        #     vector_retriever=vectorRetriever,
        #     bm25_retriever=bm25Retriever
        #     # 注意：如果你本地没有下载 bge-reranker-base，这里可以传入 reranker_model=None 降级为单纯的 RRF 融合
        # )
        #
        # print("✅ 混合检索器装载完成！\n")
        #
        # guide_data = indexer.collection.get(include=['documents', 'metadatas'])
        # guide_docs = []
        # if guide_data and guide_data.get('ids'):
        #     for i in range(len(guide_data['ids'])):
        #         guide_docs.append({
        #             'page_content': guide_data['documents'][i],
        #             'metadata': guide_data['metadatas'][i],
        #             'id': guide_data['ids'][i]
        #         })
        #
        # guide_bm25 = BM25Retriever(documents=guide_docs)
        # guide_vector = VectorRetriever(indexer.collection, indexer.embedding_function)
        # # 复用同一个 Reranker 模型对象，节省显存
        # self.guide_hybrid_retriever = FinalHybridRetrieval(guide_vector, guide_bm25)

    def _search_and_retrieve_guidelines(self, query: str, limit: int = 5, filters: str = None,
                                        session_id: str = "test") -> str:
        """
        【宏工具】一步到位：混合搜索子块 -> 去重父级ID -> 拉取完整父文档
        """
        try:
            # 1. 记录开始时间
            tool_performance_monitor.start_tool("search_medical_guidelines", session_id)
            print(f"[{session_id}] 🛠️ 开始调用工具: search_medical_guidelines | Query: {query}")

            # 2. 混合搜索最相关的 Child Chunks
            results = self.guide_hybrid.search(query, k=limit, filter_Dict=filters)
            print("查询结果", results)
            if not results:
                tool_performance_monitor.end_tool("search_medical_guidelines", "NO_RELEVANT_GUIDELINES")
                return "⚠️ 未找到相关的权威康复指南，请检测参数调用的问题并尝试调整查询词。"

            # 3. 提取并去重 Parent IDs
            parent_ids = []
            seen_parents = set()

            for doc in results:
                # 注意：Hybrid检索返回的是字典，所以用 doc.get()
                metadata = doc.get('metadata', {})
                parent_id = metadata.get('parent_id')

                # 兼容旧版本数据格式
                if not parent_id and 'guideline_id' in metadata:
                    parent_id = f"parent_{metadata['guideline_id']}"

                if parent_id and parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    parent_ids.append(parent_id)

            # 4. 根据去重后的 Parent IDs 组装完整的父文档
            retrieved_parents = []
            for pid in parent_ids:
                parent_content = self.parent_store_manager.load_content(pid)
                if parent_content:
                    title = parent_content.get('metadata', {}).get('title', '未知指南')
                    category = parent_content.get('metadata', {}).get('category', '未知分类')
                    content = parent_content.get('content', '').strip()

                    # 组装极具结构化的提示信息给 LLM
                    retrieved_parents.append(
                        f"📘 【权威指南标题】: {title}\n"
                        f"📁 【疾病分类】: {category}\n"
                        f"🔗 【溯源ID】: {pid}\n"
                        f"📝 【完整正文】:\n{content}"
                    )

            if not retrieved_parents:
                return "⚠️ 找到了相关段落，但因底层文件构建的父Chunk缺失无法加载完整指南。"

            # 5. 格式化最终输出
            final_output = (
                    f"✅ 成功为您找到 {len(retrieved_parents)} 篇高度相关的完整康复指南：\n\n" +
                    "\n\n" + "=" * 60 + "\n\n".join(retrieved_parents) + "\n" + "=" * 60
            )

            # 6. 记录结束时间并返回
            tool_performance_monitor.end_tool("search_medical_guidelines", f"返回了 {len(retrieved_parents)} 篇指南")
            print(f"[{session_id}] ✅ 工具调用结束: 找到 {len(retrieved_parents)} 篇指南")
            return final_output

        except Exception as e:
            logger.error(f"指南检索宏工具失败: {e}")
            return f"🚨 检索系统异常: {str(e)}"

    def _search_patient_faq(self, query: str, limit: int = 4, filters: str = None,
                            session_id: str = "test") -> str:
        """
        搜索患者历史 FAQ
        """
        try:
            tool_performance_monitor.start_tool("search_patient_faq", session_id)
            print(f"[{session_id}] 🛠️ 开始调用工具: search_patient_faq | Query: {query}")

            # 组装过滤条件（如果大模型传入了具体的疾病类型）
            # filters = {"disease": disease_filter} if disease_filter else None

            # 混合检索 FAQ 库
            results = self.faq_hybrid.search(query, k=limit, filters=filters)
            print("查询结果", results)
            if not results:
                tool_performance_monitor.end_tool("search_patient_faq", "NO_FAQ_FOUND")
                return "⚠️ 当前 FAQ 库中没有找到其他患者类似的经验反馈。请检测参数调用的问题并尝试调整查询词"

            formatted_faqs = []
            for doc in results:
                metadata = doc.get('metadata', {})
                disease = metadata.get('disease', '未知')
                intent = metadata.get('intent_type', '经验')
                # 取出 RRF 得分以供调试
                score = doc.get('rrf_score', doc.get('score', 0))

                formatted_faqs.append(
                    f"🏷️ 【病症】: {disease} | 【维度】: {intent} (匹配度: {score:.2f})\n"
                    f"💬 {doc.get('document', '')}"
                )

            final_output = (
                    f"💡 为您找到 {len(formatted_faqs)} 条历史患者的相似经验：\n\n" +
                    "\n" + "-" * 50 + "\n".join(formatted_faqs) + "\n" + "-" * 50
            )

            tool_performance_monitor.end_tool("search_patient_faq", f"返回了 {len(formatted_faqs)} 条经验")
            print(f"[{session_id}] ✅ 工具调用结束: 找到 {len(formatted_faqs)} 条FAQ")
            return final_output

        except Exception as e:
            logger.error(f"FAQ检索工具失败: {e}")
            return f"🚨 FAQ系统异常: {str(e)}"

    def create_tools(self) -> List:
        """注册并返回给 LangGraph 使用的工具列表"""

        # @tool
        # def finish_conversation_and_generate_plan() -> str:
        #     """
        #     当患者明确表示“没有问题了”、“帮我总结一下吧”、“出个报告吧”、“OK了”时，或者患者已经有了康复指南，需要”“调用此工具。
        #     """
        #     return "FINALIZE"

        @tool
        def search_medical_guidelines_tool(query: str, limit: int = 5, disease: str = None,
                                           category: str = None, tool_call_id: str = None) -> Command:
            """
            【强制说明】：当患者询问标准的康复动作、训练周期、负重标准、禁忌症或医学客观规律时，必须调用此工具。
            此工具会自动检索切片并组装完整的、权威的医学指南全文供你参考。

            Args:
                query: 提炼出的医学检索词 (如 "半月板损伤 术后第一周 弯曲角度")
                limit: 需要返回的指南篇数，默认5
                disease: (可选) 如果明确知道患者病种，请传入，如 "半月板损伤" 或 "骨折"
                category: 意图类别如饮食、运动等 (可选)
                tool_call_id: 调用的工具名称
            """
            filters = {}
            # 💡 核心逻辑：如果大模型传进来的是"综合"或者空值，干脆不设这个过滤条件
            if disease and disease != "综合":
                filters["disease"] = disease
            filters["category"] = category
            # 如果字典是空的，直接传 None，触发底层的全库自由检索
            final_filters = filters if filters else None
            # 在实际业务中，你可以通过 LangChain 回调机制获取真实的 session_id
            # return self._search_and_retrieve_guidelines(query, limit, final_filters, session_id="Agent_Call")
            long_docs_str = self._search_and_retrieve_guidelines(query, limit, final_filters, session_id="Agent_Call")
            return Command(
                update={
                    "raw_retrieved_docs": long_docs_str,  # 存入原始长文本
                    "messages": [ToolMessage(content="✅ 检索完毕，等待后台提纯...", tool_call_id=tool_call_id)]
                }
            )

        @tool
        def search_patient_faq_tool(query: str, disease: str = None, category: str = None, tool_call_id:str=None) -> Command:
            """
            【强制说明】：当患者询问“别人会不会也这样痛”、“大腿酸正不正常”、“能不能吃XX补品”等
            主观感受、并发痛感、心理焦虑或过往患者经验时，必须调用此工具。

            Args:
                query: 患者的主观症状或疑问描述 (如 "弯曲膝盖后侧撕扯感")
                disease: (可选) 如果明确知道患者病种，请传入，如 "半月板损伤" 或 "骨折"
                category: 意图类别如饮食、运动等 (可选)
                tool_call_id: 调用的工具名称
            """
            filters = {}
            # 💡 核心逻辑：如果大模型传进来的是"综合"或者空值，干脆不设这个过滤条件
            if disease and disease != "综合":
                filters["disease"] = disease
            filters["intent_type"] = category
            # 如果字典是空的，直接传 None，触发底层的全库自由检索
            final_filters = filters if filters else None
            long_faq = self._search_patient_faq(query, limit=4, filters=final_filters, session_id="Agent_Call")
            # return self._search_patient_faq(query, limit=4, filters=final_filters, session_id="Agent_Call")

            return Command(
                update={
                    "raw_retrieved_faq": long_faq,  # 存入原始长文本
                    "messages": [ToolMessage(content="✅ 检索完毕，等待后台提纯...", tool_call_id=tool_call_id)]
                }
            )

        return [search_medical_guidelines_tool, search_patient_faq_tool]

    # def _search_child_chunks(self, query: str, limit: int = 5, score_threshold: float = 0.3,
    #                          session_id: str = "test") -> str:
    #     """
    #     搜索最相关的子chunk
    #
    #     Args:
    #         query: 搜索查询字符串
    #         limit: 返回结果的最大数量
    #         score_threshold: 相关性分数阈值，0-1之间，越大表示越相关
    #         session_id: 本次在RAG中调用工具的会话Id
    #     Returns:
    #         格式化后的搜索结果字符串
    #     """
    #     try:
    #         tool_performance_monitor.start_tool("search_child_chunks", session_id)
    #
    #         # ✅ 优先使用混合检索（如果初始化成功）
    #         if hasattr(self, 'guide_hybrid_retriever') and self.guide_hybrid_retriever:
    #             results = self.guide_hybrid_retriever.search(query, k=limit)
    #         else:
    #             # ❌ 降级到原有搜索逻辑
    #             results = self.chroma_adapter.similarity_search(query, k=limit)
    #
    #         if not results:
    #             custom_performance_monitor.end_rag_query("search_child_chunks", "NO_RELEVANT_CHUNKS")
    #             return "NO_RELEVANT_CHUNKS"
    #
    #         # 格式化结果
    #         formatted_results = []
    #         for doc in results:
    #             parent_id = doc.metadata.get('parent_id', '')
    #             if not parent_id and 'guideline_id' in doc.metadata:
    #                 parent_id = f"parent_{doc.metadata['guideline_id']}"
    #
    #             formatted_results.append(
    #                 f"📄 标题: {doc.metadata.get('title', '未知')}\n"
    #                 f"🔗 父文档ID: {parent_id}\n"
    #                 f"📁 分类: {doc.metadata.get('category', '未知')}\n"
    #                 f"📝 内容: {doc.page_content[:200].strip()}..."
    #             )
    #
    #         tool_performance_monitor.end_tool("search_child_chunks", "\n\n".join(formatted_results))
    #         return "\n\n" + "=" * 50 + "\n".join(formatted_results)
    #
    #     except Exception as e:
    #         logger.error(f"搜索失败: {e}")
    #         custom_performance_monitor.end_rag_query("search_child_chunks", f"RETRIEVAL_ERROR: {str(e)}")
    #         return f"RETRIEVAL_ERROR: {str(e)}"
    #
    # def _retrieve_parent_chunks(self, parent_id: str, session_id: str = "test") -> str:
    #     """
    #     根据parent_id检索完整的父chunk
    #
    #     Args:
    #         parent_id: 父文档的唯一标识符
    #
    #     Returns:
    #         格式化后的父文档内容
    #     """
    #     try:
    #         tool_performance_monitor.start_tool("retrieve_parent_chunks", session_id)
    #         # 从parent_store_manager加载
    #         parent_content = self.parent_store_manager.load_content(parent_id)
    #
    #         if not parent_content:
    #             # 尝试从ChromaDB中查找相关文档
    #             logger.warning(f"父文档 {parent_id} 未找到，尝试在ChromaDB中搜索...")
    #
    #             # 在ChromaDB中搜索包含此parent_id的文档
    #             all_docs = self.chroma_adapter.get_all_documents(limit=100)
    #             related_docs = []
    #
    #             for doc in all_docs:
    #                 if doc.metadata.get('parent_id') == parent_id:
    #                     related_docs.append(doc)
    #
    #             if related_docs:
    #                 # 合并所有相关的子chunk
    #                 combined_content = []
    #                 for doc in related_docs:
    #                     combined_content.append(doc.page_content)
    #
    #                 return (
    #                     f"🔍 注意: 父文档 {parent_id} 被分割为 {len(related_docs)} 个子chunk\n"
    #                     f"📄 标题: {related_docs[0].metadata.get('title', '未知')}\n"
    #                     f"📁 分类: {related_docs[0].metadata.get('category', '未知')}\n"
    #                     f"📝 合并内容:\n{chr(10).join(combined_content)}"
    #                 )
    #             else:
    #                 return "NO_PARENT_DOCUMENT"
    #         tool_performance_monitor.end_tool("retrieve_parent_chunks", "找到一个对应的父文档信息")
    #         # 正常返回父文档内容
    #         return (
    #             f"📄 标题: {parent_content.get('metadata', {}).get('title', '未知')}\n"
    #             f"📁 分类: {parent_content.get('metadata', {}).get('category', '未知')}\n"
    #             f"🔗 文档ID: {parent_content.get('parent_id', parent_id)}\n"
    #             # f"📊 维度得分: "
    #             # f"饮食:{parent_content.get('metadata', {}).get('diet_score', 0):.2f}/"
    #             # f"心理:{parent_content.get('metadata', {}).get('psych_score', 0):.2f}/"
    #             # f"药物:{parent_content.get('metadata', {}).get('drug_score', 0):.2f}/"
    #             # f"运动:{parent_content.get('metadata', {}).get('exercise_score', 0):.2f}\n"
    #             f"📝 完整内容:\n{parent_content.get('content', '').strip()}"
    #         )
    #
    #     except Exception as e:
    #         logger.error(f"检索父文档失败: {e}")
    #         return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    #
    # def _retrieve_many_parent_chunks(self, parent_ids: List[str], session_id: str = "test") -> str:
    #     """
    #     检索多个父chunk
    #
    #     Args:
    #         parent_ids: 父文档ID列表
    #
    #     Returns:
    #         格式化后的多个父文档内容
    #     """
    #     try:
    #         tool_performance_monitor.end_tool.start_tool("retrieve_many_parent_chunks", session_id)
    #         if not parent_ids:
    #             return "NO_PARENT_IDS_PROVIDED"
    #
    #         # 如果是字符串，转换为列表
    #         if isinstance(parent_ids, str):
    #             parent_ids = [parent_ids]
    #
    #         all_parents_content = []
    #         for parent_id in parent_ids:
    #             parent_content = self._retrieve_parent_chunks(parent_id)
    #             all_parents_content.append(f"【父文档 {parent_id}】\n{parent_content}")
    #         tool_performance_monitor.end_tool("retrieve_parent_chunks", "\n\n" + "=" * 60 + "\n".join(all_parents_content))
    #
    #         return "\n\n" + "=" * 60 + "\n".join(all_parents_content)
    #
    #     except Exception as e:
    #         logger.error(f"批量检索父文档失败: {e}")
    #         return f"BATCH_PARENT_RETRIEVAL_ERROR: {str(e)}"
    #
    # def _search_with_filters(
    #         self,
    #         query: str,
    #         limit: int = 5,
    #         min_diet_score: float = 0.0,
    #         min_psych_score: float = 0.0,
    #         min_drug_score: float = 0.0,
    #         min_exercise_score: float = 0.0
    # ) -> str:
    #     """
    #     带维度过滤的搜索
    #
    #     Args:
    #         query: 搜索查询
    #         limit: 返回结果数量
    #         min_diet_score: 最小饮食得分
    #         min_psych_score: 最小心理得分
    #         min_drug_score: 最小药物得分
    #         min_exercise_score: 最小运动得分
    #
    #     Returns:
    #         过滤后的搜索结果
    #     """
    #     try:
    #         # 注意：原生ChromaDB查询需要手动过滤
    #         # 我们先获取更多结果，然后手动过滤
    #         results = self.chroma_adapter.similarity_search(query, k=limit * 3)
    #
    #         if not results:
    #             return "NO_RELEVANT_CHUNKS"
    #
    #         # 应用维度过滤
    #         filtered_results = []
    #         for doc in results:
    #             metadata = doc.metadata
    #
    #             # 检查维度得分
    #             if (metadata.get('diet_score', 0) >= min_diet_score and
    #                     metadata.get('psych_score', 0) >= min_psych_score and
    #                     metadata.get('drug_score', 0) >= min_drug_score and
    #                     metadata.get('exercise_score', 0) >= min_exercise_score):
    #                 filtered_results.append(doc)
    #
    #             if len(filtered_results) >= limit:
    #                 break
    #
    #         if not filtered_results:
    #             return "NO_CHUNKS_MATCH_FILTERS"
    #
    #         # 格式化结果
    #         formatted_results = []
    #         for doc in filtered_results:
    #             formatted_results.append(
    #                 f"📄 标题: {doc.metadata.get('title', '未知')}\n"
    #                 f"🔗 父文档ID: {doc.metadata.get('parent_id', '')}\n"
    #                 f"📁 分类: {doc.metadata.get('category', '未知')}\n"
    #                 # f"📊 维度得分: "
    #                 # f"饮食:{doc.metadata.get('diet_score', 0):.2f}/"
    #                 # f"心理:{doc.metadata.get('psych_score', 0):.2f}/"
    #                 # f"药物:{doc.metadata.get('drug_score', 0):.2f}/"
    #                 # f"运动:{doc.metadata.get('exercise_score', 0):.2f}\n"
    #                 f"📝 内容: {doc.page_content[:200].strip()}..."
    #             )
    #
    #         return "\n\n" + "=" * 50 + "\n".join(formatted_results)
    #
    #     except Exception as e:
    #         logger.error(f"带过滤搜索失败: {e}")
    #         return f"FILTERED_RETRIEVAL_ERROR: {str(e)}"
    #
    # def create_tools(self) -> List:
    #     """创建和返回工具列表"""
    #
    #     # 创建搜索工具
    #     @tool
    #     def search_child_chunks_tool(query: str, limit: int = 5, score_threshold: float = 0.3) -> str:
    #         """
    #         搜索康复指南中最相关的子chunk
    #
    #         Args:
    #             query: 搜索查询字符串
    #             limit: 返回结果的最大数量，默认5
    #             score_threshold: 相关性分数阈值，0-1之间，越大表示越相关，默认0.3
    #         """
    #         return self._search_child_chunks(query, limit, score_threshold)
    #
    #     # 创建检索工具
    #     @tool
    #     def retrieve_parent_chunks_tool(parent_id: str) -> str:
    #         """
    #         根据父文档ID检索完整的康复指南内容
    #
    #         Args:
    #             parent_id: 父文档的唯一标识符
    #         """
    #         return self._retrieve_parent_chunks(parent_id)
    #
    #     # 创建批量检索工具
    #     @tool
    #     def retrieve_many_parent_chunks_tool(parent_ids: List[str]) -> str:
    #         """
    #         批量检索多个父文档的完整内容
    #
    #         Args:
    #             parent_ids: 父文档ID列表
    #         """
    #         return self._retrieve_many_parent_chunks(parent_ids)
    #
    #     # 创建带过滤的搜索工具
    #     @tool
    #     def search_with_filters_tool(
    #             query: str,
    #             limit: int = 5,
    #             min_diet_score: float = 0.0,
    #             min_psych_score: float = 0.0,
    #             min_drug_score: float = 0.0,
    #             min_exercise_score: float = 0.0
    #     ) -> str:
    #         """
    #         根据康复指南的维度得分进行过滤搜索
    #
    #         Args:
    #             query: 搜索查询
    #             limit: 返回结果数量，默认5
    #             min_diet_score: 最小饮食得分，默认0.0
    #             min_psych_score: 最小心理得分，默认0.0
    #             min_drug_score: 最小药物得分，默认0.0
    #             min_exercise_score: 最小运动得分，默认0.0
    #         """
    #         return self._search_with_filters(
    #             query, limit, min_diet_score, min_psych_score,
    #             min_drug_score, min_exercise_score
    #         )
    #
    #     return [
    #         search_child_chunks_tool,
    #         retrieve_parent_chunks_tool,
    #         retrieve_many_parent_chunks_tool,
    #         search_with_filters_tool
    #     ]
