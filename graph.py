from langgraph.graph import StateGraph, START, END

from project.rehab_core import config
from project.rehab_core.guide_chunker import EmbeddingHierarchicalIndexer
from project.rehab_core.langgraph_callbacks import PerformanceCallback
from project.rehab_core.retrieval.hybrid_retriever import BM25Retriever, VectorRetriever
from project.rehab_core.retrieval.hybrid_retriever_final import FinalHybridRetrieval
from project.rehab_core.state import AgentState
from project.rehab_core.nodes import orchestrator, compress_context, collect_answer, generate_final_plan, \
    query_analyzer, extract_and_compress_docs, fallback_response, doctor_speaker, intent_analysis_node, \
    dialogue_decision_node, ask_question_node, analyze_symptoms_node, give_advice_node
from project.rehab_core.edges import route_after_orchestrator_call, should_compress_context, route_start, \
    route_after_decision
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from project.rehab_core.node_monitor import NodeMonitor


def build_graph(llm, tools, memory):
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
    # # 两个索引器的初始化构建，
    # result_faq_message = indexer.index_faq_to_chroma()
    # result = indexer.index_to_chroma()
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
    # hybrid_retrieval = FinalHybridRetrieval(
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
    # guide_hybrid_retriever = FinalHybridRetrieval(guide_vector, guide_bm25)

    # agent_builder = StateGraph(AgentState)
    #
    # # 添加节点
    # agent_builder.add_node("orchestrator", lambda state: orchestrator(state, llm))
    # agent_builder.add_node("tools", ToolNode(tools))
    # agent_builder.add_node("compress_context", lambda state: compress_context(state, llm))  # 你的原函数
    # agent_builder.add_node("generate_final_plan", lambda state: generate_final_plan(state, llm))
    # agent_builder.add_node("should_compress_context", lambda state: should_compress_context(state))
    # agent_builder.add_node("query_analyzer", lambda state: query_analyzer(state, llm))  # 👈 新增节点
    # agent_builder.add_node("extract_and_compress_docs", lambda state: extract_and_compress_docs(state, llm))
    # agent_builder.add_node("fallback_response", lambda state: fallback_response(state, llm))
    # agent_builder.add_node("doctor_speaker", lambda state: doctor_speaker(state, llm))
    # # 先暂时不考虑这个处理，毕竟我们的调用还是撑得住的。
    # # agent_builder.add_node("fallback_response", lambda state: fallback_response(state, llm))
    # agent_builder.add_node("collect_answer", collect_answer)
    # # agent_builder.add_node("intent_message", intent_node)
    # # 绘制边：入口路由
    # # 在多加一个点， 他读取用户的输入，判断他是要做什么内容，如果是提到要复查或是结束对话的话，就是总结信息生成病例，然后再生成最后的结果的时候
    # # 需要多一个部分，如果是复查，就基于康复指南信息，询问，如果总结就不要在询问了，直接返回结果。
    # # agent_builder.add_edge(START, "intent_message")
    # # agent_builder.add_conditional_edges(
    # #     "intent_message",
    # #     route_start,
    # #     {"generate_final_plan": "generate_final_plan", "orchestrator": "orchestrator"}
    # # )
    # # 我们是这样设计的，只要用户没有显式的输出结束，就不要为他生成最终的康复指南，当然实际使用的时候，如果
    # # 迟迟没有回应可以在5分钟后这样自动执行一次。
    # agent_builder.add_conditional_edges(
    #     START,
    #     route_start,
    #     {"generate_final_plan": "generate_final_plan", "query_analyzer": "query_analyzer"}
    # )
    # agent_builder.add_edge("query_analyzer", "orchestrator")
    # # 终结流程
    # agent_builder.add_edge("generate_final_plan", "collect_answer")
    #
    # # RAG循环流程
    # agent_builder.add_conditional_edges(
    #     "orchestrator",
    #     route_after_orchestrator_call,
    #     {"tools": "tools", "doctor_speaker": "doctor_speaker", "fallback_response": "fallback_response"}
    # )
    # agent_builder.add_edge("tools", "extract_and_compress_docs")
    # agent_builder.add_edge("doctor_speaker", "collect_answer")
    # # 多增加一个结点，用于把提取出来的文档先总结
    # agent_builder.add_conditional_edges("extract_and_compress_docs",
    #                                     should_compress_context,
    #                                     {"compress_context": "compress_context", "orchestrator": "orchestrator"})
    # # agent_builder.add_edge("tools", "should_compress_context")
    # # should_compress_context分支两个部分,调度器和压缩上下文.
    # agent_builder.add_edge("compress_context", "orchestrator")
    # # agent_builder.add_edge("tools", "compress_context")  # 简化了should_compress的条件，你代码里可以保留原样
    # # agent_builder.add_edge("compress_context", "orchestrator")
    # agent_builder.add_edge("fallback_response", END)
    # agent_builder.add_edge("collect_answer", END)
    #
    # # 挂载 SQLite Checkpointer (实现短期记忆天然存储)
    # # 需要注意添加中断配置，比如状态不能保存输出，不然下一次复原的时候会再输出一遍了，
    # return agent_builder.compile(checkpointer=memory, interrupt_before=["collect_answer"])
    workflow = StateGraph(AgentState)

    # 1. 意图识别节点（替代原来的query_analyzer）
    workflow.add_node("intent_analysis", lambda state: intent_analysis_node(state, llm))

    # 2. 决策节点
    workflow.add_node("dialogue_decision", lambda state: dialogue_decision_node(state, llm))

    # 3. 原有的节点（保留）
    workflow.add_node("orchestrator", lambda state: orchestrator(state, llm))
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("compress_context", lambda state: compress_context(state, llm))
    workflow.add_node("generate_final_plan", lambda state: generate_final_plan(state, llm))
    workflow.add_node("query_analyzer", lambda state: query_analyzer(state, llm))
    workflow.add_node("extract_and_compress_docs", lambda state: extract_and_compress_docs(state, llm))
    workflow.add_node("fallback_response", lambda state: fallback_response(state, llm))
    workflow.add_node("doctor_speaker", lambda state: doctor_speaker(state, llm))
    workflow.add_node("collect_answer", collect_answer)

    # 4. 新增的动作执行节点
    workflow.add_node("ask_question", lambda state: ask_question_node(state, llm))
    workflow.add_node("analyze_symptoms", lambda state: analyze_symptoms_node(state, llm))
    workflow.add_node("give_advice", lambda state: give_advice_node(state, llm))

    # 5. 新的路由逻辑
    workflow.add_edge(START, "intent_analysis")
    workflow.add_edge("intent_analysis", "dialogue_decision")

    # 决策节点的条件路由
    workflow.add_conditional_edges(
        "dialogue_decision",
        route_after_decision,
        {
            "ask_question": "ask_question",
            "analyze_symptoms": "analyze_symptoms",
            "give_advice": "give_advice",
            "generate_final_plan": "generate_final_plan",
            "orchestrator": "orchestrator",  # 原有流程
            "doctor_speaker": "doctor_speaker"
        }
    )

    # 动作执行后的路由
    workflow.add_edge("ask_question", "doctor_speaker")
    workflow.add_edge("analyze_symptoms", "query_analyzer")  # 分析后走原有流程
    workflow.add_edge("give_advice", "doctor_speaker")

    workflow.add_edge("query_analyzer", "orchestrator")
    # 原有的边（保留大部分）
    workflow.add_edge("doctor_speaker", "collect_answer")
    workflow.add_edge("generate_final_plan", "collect_answer")
    workflow.add_edge("collect_answer", END)

    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator_call,
        {"tools": "tools", "doctor_speaker": "doctor_speaker", "fallback_response": "fallback_response"}
    )
    workflow.add_edge("tools", "extract_and_compress_docs")
    workflow.add_conditional_edges("extract_and_compress_docs",
                                   should_compress_context,
                                   {"compress_context": "compress_context", "orchestrator": "orchestrator"})
    workflow.add_edge("compress_context", "orchestrator")
    workflow.add_edge("fallback_response", END)

    return workflow.compile(checkpointer=memory, interrupt_before=["collect_answer"])



def monitored_node(node_func):
    """节点监控装饰器"""

    def wrapper(state: dict):
        # 获取或创建监控器
        monitor = state.get("node_monitor")
        if not monitor:
            monitor = NodeMonitor()
            state["node_monitor"] = monitor

        # 开始计时
        node_name = node_func.__name__
        monitor.start_node(node_name)

        try:
            # 执行原节点逻辑
            result = node_func(state)

            # 结束计时
            content = ""
            if isinstance(result, dict) and "response" in result:
                content = result["response"]
            elif hasattr(result, 'content'):
                content = result.content

            monitor.end_node(node_name, content)

            return result

        except Exception as e:
            monitor.end_node(node_name, "")
            raise e

    return wrapper

# 构建 Agent 子图（完全复刻原项目结构）
# agent_builder = StateGraph(AgentState)
# # 添加节点
# agent_builder.add_node("orchestrator", orchestrator)
# agent_builder.add_node("compress_context", compress_context)
# agent_builder.add_node("fallback_response", fallback_response)
# agent_builder.add_node("collect_answer", collect_answer)
# # 工具节点：由 bind_tools 自动 注册，无需显式 add_node
# # （在 nodes.py 中已通过 llm_with_tools.bind_tools 注册）
# # 边定义（100% 复刻原项目流程）
# agent_builder.add_edge(START, "orchestrator")
# agent_builder.add_conditional_edges(
#     "orchestrator",
#     route_after_orchestrator_call,
#     {
#         "tools": "tools",  # 实际由 langchain 的 ToolNode 自动处理
#         "fallback_response": "fallback_response",
#         "collect_answer": "collect_answer"
#     }
# )
# # 注意：LangGraph 中 "tools" 节点是隐式的，由 ToolNode 提供
# # 我们通过以下方式注入：
#
# tool_node = ToolNode([
#     # 已在 tools.py 中绑定的工具
# ])
# agent_builder.add_node("tools", tool_node)
# agent_builder.add_edge("tools", "should_compress_context")
# agent_builder.add_conditional_edges(
#     "should_compress_context",
#     lambda x: x.goto,
#     {
#         "compress_context": "compress_context",
#         "orchestrator": "orchestrator"
#     }
# )
# agent_builder.add_edge("compress_context", "orchestrator")
# agent_builder.add_edge("fallback_response", "collect_answer")
# agent_builder.add_edge("collect_answer", END)
# # 编译
# rehab_agent_graph = agent_builder.compile()
#
