# project/db/hierarchical_indexer_with_embedding.py
"""
集成自定义embedding的层次化索引器 启动后增量更新，
"""
import os
import sqlite3
import json
import uuid
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from project.rehab_core.custom_embedding import LocalEmbeddingFunction, get_embedding_function
from project.rehab_core.chroma_adapter import load_chroma_adapter
from project.rehab_core.parent_store_manager_updated import ParentStoreManager
# from project.rehab_core.tool_factory import ToolFactory
from project.rehab_core.retrieval.hybrid_retriever_final import FinalHybridRetrieval
from project.rehab_core.retrieval.hybrid_retriever import BM25Retriever, VectorRetriever
logger = logging.getLogger(__name__)


from project.rehab_core.chunking.semantic_chunker import SemanticChunker
# from project.rehab_core.schema_manager import schema_manager, create_extraction_result
# 在你的 guide_chunker.py 中
from project.rehab_core.retrieval.child_chunk_generator import DynamicChildChunkGenerator
class EmbeddingHierarchicalIndexer:
    """企业级带embedding的层次化索引器 - 医学领域优化版"""

    def __init__(
            self,
            db_path: str = "D:\\Desktop\\company\\Agent-rag\\agentic-rag-for-dummies\\project\\db\\rehab.db",
            parent_store_path: str = "./data/parent_docs",
            chroma_path: str = "./chroma_embedding",
            parent_size: int = 3000,
            min_parent_size = 2000,
            max_parent_size = 4000,
            parent_overlap: int = 200,
            child_size: int = 400,
            child_overlap: int = 80,
            embedding_model: str = "m3e-base",  # 或 "bge-small-zh"
            embedding_device: str = "cpu"
    ):
        """
        初始化索引器

        Args:
            embedding_model: embedding模型名称或路径
            embedding_device: cpu 或 cuda
        """
        self.db_path = Path(db_path)
        self.parent_store_path = Path(parent_store_path)
        self.chroma_path = Path(chroma_path)

        # 创建存储目录
        self.parent_store_path.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        # 初始化embedding函数
        print(f"🔧 初始化embedding模型: {embedding_model}")
        self.embedding_function = get_embedding_function(
            model_name=embedding_model,
            device=embedding_device
        )

        # 中文友好的分隔符
        separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

        logger.info(f"✂️ 初始化父子分块器：Parent={parent_size}, Child={child_size}")

        # 父文档分割器
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            length_function=len,
            separators=separators
        )

        # 子文档分割器
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            length_function=len,
            separators=separators
        )
        # 初始化子chunk生成器
        self.child_generator = DynamicChildChunkGenerator()
        # 分块参数
        self.parent_size = parent_size
        self.parent_overlap = parent_overlap
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.min_parent_size = min_parent_size
        self.max_parent_size = max_parent_size
        # ChromaDB客户端
        self.chroma_client = None
        self.collection = None
        self.faq_collection = None

    def _init_chroma(self):
        """初始化ChromaDB客户端"""
        try:
            import chromadb
            from chromadb.config import Settings

            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_path),
                settings=Settings(anonymized_telemetry=False)
            )

            return True

        except Exception as e:
            logger.error(f"❌ 初始化ChromaDB失败: {e}")
            return False

    def load_faqs_from_db(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.db_path.exists():
            logger.error(f"数据库不存在: {self.db_path}")
            return []

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            # FAQ的部分
            query_faq = """
                SELECT faq_id, disease, intent_type, question, answer, source_session FROM faq_message
                        """

            if limit:
                query_faq += f" LIMIT {limit}"

            cursor.execute(query_faq)
            rows = cursor.fetchall()
            conn.close()

            faqs = []
            for row in rows:
                faqs.append({
                    "faq_id": row[0],
                    "disease": row[1],
                    "intent_type": row[2],
                    "question": row[3],
                    "answer": row[4],
                    "source_session": row[5],
                })

            logger.info(f"✅ 从数据库加载了 {len(faqs)} 条指南")
            return faqs

        except Exception as e:
            logger.error(f"❌ 加载数据库失败: {e}")
            return []

    def load_guidelines_from_db(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """从数据库加载康复指南"""
        if not self.db_path.exists():
            logger.error(f"数据库不存在: {self.db_path}")
            return []

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            query = """
                SELECT guideline_id, title, category, disease, content, source FROM rehab_guidelines
            """

            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()

            guidelines = []
            for row in rows:
                guidelines.append({
                    "guideline_id": row[0],
                    "title": row[1],
                    "category": row[2],
                    "disease": row[3],
                    "content": row[4],
                    "source": row[5],
                })

            logger.info(f"✅ 从数据库加载了 {len(guidelines)} 条指南")


            return guidelines

        except Exception as e:
            logger.error(f"❌ 加载数据库失败: {e}")
            return []

    def process_single_guide(self, guide: Dict[str, Any]) -> Tuple[List[Document], List[Document]]:
        """
        处理单条指南，生成父子分块

        Returns:
            (parent_chunks, child_chunks)
        """
        content = guide.get("content", "")
        if not content or not content.strip():
            return [], []

        # 基础元数据
        base_metadata = {
            "guideline_id": str(guide["guideline_id"]),
            "title": guide["title"],
            "category": guide["category"],
            "source": guide["source"],
            "disease": guide["disease"],
            "document_type": "rehabilitation_guide"
        }

        # 1. 创建父文档
        parent_doc = Document(
            page_content=content,
            metadata=base_metadata.copy()
        )

        # 2. 对父文档进行分块
        parent_chunks = self.parent_splitter.split_documents([parent_doc])
        merged_parents = self.merge_small_parents(parent_chunks, self.min_parent_size)
        split_parents = self.split_large_parents(merged_parents, self.max_parent_size, self.child_splitter)
        cleaned_parents = self.clean_small_chunks(split_parents, self.min_parent_size)

        # 3. 为每个父chunk生成子chunks
        all_parent_chunks = []
        all_child_chunks = []

        for i, parent_chunk in enumerate(cleaned_parents):
            # 为父chunk添加唯一ID和索引
            parent_id = f"guide_{guide['guideline_id']}_p{i}"
            parent_metadata = parent_chunk.metadata.copy()
            parent_metadata.update({
                "parent_id": parent_id,
                "parent_index": i,
                "total_parents": len(cleaned_parents)
            })

            parent_chunk_with_id = Document(
                page_content=parent_chunk.page_content,
                metadata=parent_metadata
            )
            all_parent_chunks.append(parent_chunk_with_id)

            # 对父chunk进行子分块
            # child_chunks = self.child_splitter.split_documents([parent_chunk_with_id])
            # 子chunk：动态生成
            child_chunks = self.child_generator.generate_child_chunks(
                parent_content=content,
                title=guide.get("title", ""),
                category=guide.get("category", "general"),
                disease=guide.get("disease", ""),
                source=guide.get("source", "unknown")
            )
            # 为子chunk添加父chunk信息
            for j, child_chunk in enumerate(child_chunks):
                child_metadata = child_chunk["metadata"].copy()
                child_metadata.update({
                    "child_id": f"{parent_id}_child_{j}",
                    "parent_id": parent_id,  # 建立父子关系
                    "child_index": j,
                    "total_children": len(child_chunks)
                })

                child_chunk_with_id = Document(
                    page_content=child_chunk["page_content"],
                    metadata=child_metadata
                )
                all_child_chunks.append(child_chunk_with_id)

        return all_parent_chunks, all_child_chunks

    def index_faq_to_chroma(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """将FAQ直接作为独立Chunk存入独立的Chroma集合"""
        print("=" * 60)
        print("🏥 开始层次化索引到ChromaDB（自定义embedding）")
        print("=" * 60)
        # 这个是为了测试中，不做额外操作， 实际使用需要借助Sqlite的时间戳比对
        # if self.faq_collection.count() > 0:
        #     print(f"⚠️ FAQ 集合中已有 {self.faq_collection.count()} 条数据。")
        #     force_reindex = input("是否要强制重新索引并覆盖? (y/n): ")
        #     if force_reindex.lower() != 'y':
        #         return {"status": "skipped", "count": self.faq_collection.count()}
        # 1. 初始化ChromaDB
        if not self._init_chroma():
            return {"error": "ChromaDB初始化失败"}

        # 2. 加载指南
        faq_list = self.load_faqs_from_db(limit)
        if not faq_list:
            return {"error": "没有找到FAQ数据"}

        print("💬 开始索引患者FAQ数据库...")
        faq_collection_name = "patient_faq_embedding"
        try:
            # 尝试获取现有集合
            self.faq_collection = self.chroma_client.get_collection(faq_collection_name)
            print(f"📁 使用现有集合: {faq_collection_name}")
        except:
            # 创建新集合
            self.faq_collection = self.chroma_client.get_or_create_collection(
                name=faq_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"📁 创建新集合: {faq_collection_name}")
        try:
            texts_to_embed = []
            metadatas = []
            ids = []

            for faq in faq_list:
                # 拼接 Q+A 作为 page_content，保证语义完整
                content = f"患者问题: {faq['question']}\n医生回答: {faq['answer']}"
                texts_to_embed.append(content)

                # 提取有价值的元数据
                meta = {
                    "faq_id": faq["faq_id"],
                    "disease": faq.get("disease", ""),
                    "intent_type": faq.get("intent_type", ""),
                    "document_type": "patient_faq"
                }
                metadatas.append(meta)
                ids.append(faq["faq_id"])

            # 生成 Embeddings 并入库 (批量处理类似你的原来逻辑)
            embeddings = self.embedding_function.encode(texts_to_embed)
            self.faq_collection.upsert(
                embeddings=embeddings,
                documents=texts_to_embed,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ 成功存入 {len(faq_list)} 条FAQ。")
            return {"status": "success", "count": len(faq_list)}

        except Exception as e:
            logger.error(f"❌ FAQ入库失败: {e}")
            return {"error": str(e)}

    def index_to_chroma(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        索引到ChromaDB（使用自定义embedding）

        Returns:
            统计信息字典
        """
        print("=" * 60)
        print("🏥 开始层次化索引到ChromaDB（自定义embedding）")
        print("=" * 60)
        # 这个是为了测试中，不做额外操作， 实际使用需要借助Sqlite的时间戳比对
        # if self.collection.count() > 0:
        #     print(f"⚠️ FAQ 集合中已有 {self.collection.count()} 条数据。")
        #     force_reindex = input("是否要强制重新索引并覆盖? (y/n): ")
        #     if force_reindex.lower() != 'y':
        #         return {"status": "skipped", "count": self.collection.count()}
        # 1. 初始化ChromaDB
        if not self._init_chroma():
            return {"error": "ChromaDB初始化失败"}

        # 2. 加载指南
        guidelines = self.load_guidelines_from_db(limit)
        if not guidelines:
            return {"error": "没有找到指南数据"}

        # 3. 创建或获取集合
        collection_name = "rehab_embedding"
        stats = {}
        try:
            # 尝试获取现有集合
            self.collection = self.chroma_client.get_collection(collection_name)
            print(f"📁 使用现有集合: {collection_name}")
        except:
            # 创建新集合
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"📁 创建新集合: {collection_name}")

        # 4. 处理每条指南
            total_processed = 0
            total_parents = 0
            total_children = 0

            for guide in guidelines:
                print(f"📄 处理: {guide['title']}")

                parent_chunks, child_chunks = self.process_single_guide(guide)

                if not child_chunks:
                    print(f"  ⚠️ 没有生成子chunks")
                    continue

                # 5. 为子chunks生成embedding
                print(f"  🔢 生成embedding向量...")
                child_texts = [chunk.page_content for chunk in child_chunks]

                try:
                    embeddings = self.embedding_function.encode(child_texts)

                    # 6. 准备元数据和ID
                    metadatas = []
                    ids = []

                    for chunk in child_chunks:
                        metadatas.append(chunk.metadata)
                        ids.append(chunk.metadata.get("child_id", f"chunk_{uuid.uuid4().hex[:8]}"))

                    # 7. 添加到ChromaDB
                    batch_size = 20
                    for i in range(0, len(child_chunks), batch_size):
                        batch_end = min(i + batch_size, len(child_chunks))

                        self.collection.upsert(
                            embeddings=embeddings[i:batch_end],
                            documents=child_texts[i:batch_end],
                            metadatas=metadatas[i:batch_end],
                            ids=ids[i:batch_end]
                        )

                    total_processed += 1
                    total_parents += len(parent_chunks)
                    total_children += len(child_chunks)

                    print(f"  ✅ 添加 {len(child_chunks)} 个子chunks")

                    # 9. 返回统计
                    stats = {
                        "guides_processed": total_processed,
                        "total_guides": len(guidelines),
                        "parent_chunks": total_parents,
                        "child_chunks": total_children,
                        "collection_size": self.collection.count(),
                        "embedding_model": str(self.embedding_function.model_path),
                        "embedding_dimension": self.embedding_function.get_dimension(),
                        "timestamp": str(uuid.uuid4())[:8]
                    }

                    print(f"\n🎉 索引完成！")
                    print(f"   处理指南: {stats['guides_processed']}/{stats['total_guides']}")
                    print(f"   父文档块: {stats['parent_chunks']}")
                    print(f"   子文档块: {stats['child_chunks']}")
                    print(f"   集合大小: {stats['collection_size']}")
                    print(f"   Embedding模型: {stats['embedding_model']}")
                    print(f"   向量维度: {stats['embedding_dimension']}")

                except Exception as e:
                    print(f"  ❌ 处理失败: {e}")
                    continue

        # 8. 存储父文档
        self._store_parent_chunks_for_all(guidelines)



        return {"stats": stats, "collection": self.collection}

    def _store_parent_chunks_for_all(self, guidelines: List[Dict[str, Any]]):
        """为所有指南存储父chunks"""
        all_parent_chunks = []

        for guide in guidelines:
            parent_chunks, _ = self.process_single_guide(guide)
            all_parent_chunks.extend(parent_chunks)

        if not all_parent_chunks:
            return

        # 清空目录
        for item in self.parent_store_path.glob("*.json"):
            item.unlink()

        # 保存每个父chunk
        saved_count = 0
        for chunk in all_parent_chunks:
            try:
                parent_id = chunk.metadata.get("parent_id", str(uuid.uuid4()))
                filename = f"{parent_id}.json"
                filepath = self.parent_store_path / filename

                chunk_data = {
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                }

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(chunk_data, f, ensure_ascii=False, indent=2)

                saved_count += 1

            except Exception as e:
                logger.error(f"保存父chunk失败: {e}")

        print(f"💾 已保存 {saved_count} 个父文档到 {self.parent_store_path}")

    def query_with_embedding(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """使用自定义embedding查询"""
        if not self.collection:
            print("❌ 集合未初始化")
            return []

        try:
            # 1. 为查询文本生成embedding
            query_embedding = self.embedding_function.encode([query_text])[0]

            # 2. 执行查询
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            # 3. 格式化结果
            formatted_results = []
            if results['documents']:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        "id": results['ids'][0][i],
                        "document": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i],
                        "relevance": 1.0 / (1.0 + results['distances'][0][i])  # 计算相关性分数
                    })

            return formatted_results

        except Exception as e:
            print(f"❌ 查询失败: {e}")
            return []


    def merge_small_parents(self, chunks, min_size):
        if not chunks:
            return []

        merged, current = [], None

        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                current.page_content += "\n\n" + chunk.page_content
                for k, v in chunk.metadata.items():
                    if k in current.metadata:
                        current.metadata[k] = f"{current.metadata[k]} -> {v}"
                    else:
                        current.metadata[k] = v

            if len(current.page_content) >= min_size:
                merged.append(current)
                current = None

        if current:
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                for k, v in current.metadata.items():
                    if k in merged[-1].metadata:
                        merged[-1].metadata[k] = f"{merged[-1].metadata[k]} -> {v}"
                    else:
                        merged[-1].metadata[k] = v
            else:
                merged.append(current)

        return merged

    def _merge_small_parents(self, chunks: List[Document]) -> List[Document]:
        """合并小段"""
        if not chunks:
            return []

        merged, current = [], None

        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                current.page_content += "\n\n" + chunk.page_content
                # 合并特定元数据
                for k, v in chunk.metadata.items():
                    if k in current.metadata and k not in ["source", "title", "doc_id", "original_id"]:
                        if current.metadata[k] != v:
                            current.metadata[k] = f"{current.metadata[k]} | {v}"

            if len(current.page_content) >= self.__min_parent_size:
                merged.append(current)
                current = None

        if current:
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                for k, v in current.metadata.items():
                    if k in merged[-1].metadata and k not in ["source", "title", "doc_id", "original_id"]:
                        if merged[-1].metadata[k] != v:
                            merged[-1].metadata[k] = f"{merged[-1].metadata[k]} | {v}"
            else:
                merged.append(current)

        return merged

    def split_large_parents(self, chunks, max_size, splitter):
        split_chunks = []

        for chunk in chunks:
            if len(chunk.page_content) <= max_size:
                split_chunks.append(chunk)
            else:
                large_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_size,
                    chunk_overlap=splitter._chunk_overlap
                )
                sub_chunks = large_splitter.split_documents([chunk])
                split_chunks.extend(sub_chunks)

        return split_chunks

    def clean_small_chunks(self, chunks, min_size):
        cleaned = []

        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < min_size:
                if cleaned:
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    for k, v in chunk.metadata.items():
                        if k in cleaned[-1].metadata:
                            cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                        else:
                            cleaned[-1].metadata[k] = v
                elif i < len(chunks) - 1:
                    chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    for k, v in chunk.metadata.items():
                        if k in chunks[i + 1].metadata:
                            chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                        else:
                            chunks[i + 1].metadata[k] = v
                else:
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)

        return cleaned

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计"""
        if not self.collection:
            return {"error": "集合未初始化"}

        try:
            return {
                "count": self.collection.count(),
                "name": self.collection.name,
                "metadata": self.collection.metadata
            }
        except Exception as e:
            return {"error": str(e)}



# 主程序
if __name__ == "__main__":
    import sys

    sys.path.append(str(Path(__file__).parent.parent))

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("🏥 康复指南层次化索引系统（自定义Embedding）")
    print("=" * 60)

    # 模型选择
    print("\n🔧 选择embedding模型:")
    print("1. M3E-base (本地模型)")
    print("2. BGE-small-zh (在线下载)")
    print("3. BGE-base-zh (在线下载)")

    model_choice = input("\n请选择模型 (1-3): ").strip()
    model_map = {
        "1": "m3e-base",
        "2": "bge-small-zh",
        "3": "bge-base-zh"
    }

    if model_choice not in model_map:
        print("⚠️ 无效选择，使用默认模型")
        model_name = "m3e-base"
    else:
        model_name = model_map[model_choice]

    # 设备选择
    device = input("使用设备 (cpu/cuda, 默认cpu): ").strip().lower()
    if device not in ["cpu", "cuda"]:
        device = "cpu"
    # 创建索引器
    indexer = EmbeddingHierarchicalIndexer(
        db_path="D:\\Desktop\\company\\Agent-rag\\agentic-rag-for-dummies\\project\\db\\rehab.db",
        parent_store_path="./data/parent_docs_embedding",
        chroma_path="./chroma_embedding",
        parent_size=2000,
        parent_overlap=200,
        child_size=400,
        child_overlap=80,
        embedding_model=model_name,
        embedding_device=device
    )
    while True:
        print("\n选项:")
        print("1. 构建索引到ChromaDB")
        print("2. 测试查询")
        print("3. 查看统计")
        print("4. 退出")
        choice = input("\n请选择 (1-4): ").strip()

        if choice == "1":
            # 询问限制数量
            limit_input = input("处理文档数量限制 (留空为全部): ").strip()
            limit = int(limit_input) if limit_input.isdigit() else None
            result_faq_message = indexer.index_faq_to_chroma()
            result = indexer.index_to_chroma(limit)

            if "stats" in result:
                stats = result["stats"]
                print(f"\n📊 最终统计:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")

        elif choice == "2":
            query = input("输入查询语句: ").strip()
            if not query:
                query = "高血压饮食建议"

            n_results = int(input("返回结果数量 (默认5): ").strip() or "5")

            # ==========================================
            # 💡 核心修复：在这里动态初始化混合检索系统
            # ==========================================
            if indexer.faq_collection is None:
                print("⚠️ 请先执行 1 构建/加载FAQ索引！")
                continue

            print("\n⏳ 正在装载混合检索器数据...")
            # 1. 从 ChromaDB 的 FAQ 集合中拉取所有文档给 BM25
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
            print("✅ 指南混合检索器装载完成！")
            # ==========================================
            # 🚀 执行查询
            # ==========================================
            print("1. 权威指南的纯向量搜索结果：")
            results_guide = indexer.query_with_embedding(query, n_results)
            for i, result in enumerate(results_guide, 1):
                print(f"  [{result['metadata'].get('title', '未知')}] 距离: {result['distance']:.4f}")
            results_guide = guide_hybrid_retriever.search(query, k=n_results)
            for i, result in enumerate(results_guide, 1):
                print(f"\n{i}.[得分: {result.get('rrf_score', result.get('score', 0)):.4f}]")
                print(f"   来源: {result.get('source', 'unknown')}")
                print(f"   标签: {result['metadata'].get('disease')} | {result['metadata'].get('intent_type')}")
                print(f"   内容: {result['document']}")
            print("\n2. FAQ 患者经验的混合搜索结果 (Vector + BM25 + 重排)：")
            # 模拟意图过滤 或者说时病情 或是 对话的方面
            filters = None
            # filters = {"disease": "半月板损伤"} # 你可以解开注释测试精准过滤

            results_faq = hybrid_retrieval.search(query, k=n_results, filters=filters)

            for i, result in enumerate(results_faq, 1):
                print(f"\n{i}.[得分: {result.get('rrf_score', result.get('score', 0)):.4f}]")
                print(f"   来源: {result.get('source', 'unknown')}")
                print(f"   标签: {result['metadata'].get('disease')} | {result['metadata'].get('intent_type')}")
                print(f"   内容: {result['document']}")


        elif choice == "3":
            stats = indexer.get_collection_stats()
            print(f"\n📊 集合统计:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        elif choice == "4":
            print("退出")
            break
        else:
            print("❌ 无效选择")
