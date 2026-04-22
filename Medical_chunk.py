import sqlite3
import json
import os
import uuid

import torch
from sentence_transformers import SentenceTransformer

from project.rehab_core import config
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma

# from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
import hashlib

from project.rehab_core.custom_embedding import get_embedding_function
from project.rehab_core.retrieval.child_chunk_generator import DynamicChildChunkGenerator


class EmbeddingModel:
    """本地Embedding模型封装"""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL_PATH, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"正在加载Embedding模型: {model_name} 在设备: {device} 上...")
        self.model = SentenceTransformer(model_name, device=device)
        print("模型加载完成")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档列表生成embedding"""
        if not texts:
            return []

        embeddings = self.model.encode(texts, normalize_embeddings=config.NORMALIZE_EMBEDDINGS)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """为查询生成embedding"""
        embedding = self.model.encode([query], normalize_embeddings=config.NORMALIZE_EMBEDDINGS)
        return embedding[0].tolist()


class ParentChunkStore:
    """父chunk存储管理"""

    def __init__(self, storage_path: str = config.PARENT_STORAGE_PATH):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save_parent_chunk(self, parent_chunk: Document) -> str:
        """保存父chunk到文件系统"""
        parent_id = parent_chunk.metadata.get("parent_id", "unknown")

        # 创建哈希作为文件名的一部分，避免冲突
        content_hash = hashlib.md5(parent_chunk.page_content.encode()).hexdigest()[:8]
        filename = f"{parent_id}_{content_hash}"

        # 保存内容
        content_path = self.storage_path / f"{filename}.txt"
        with open(content_path, "w", encoding="utf-8") as f:
            f.write(parent_chunk.page_content)

        # 保存元数据
        metadata_path = self.storage_path / f"{filename}_meta.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(parent_chunk.metadata, f, ensure_ascii=False, indent=2)

        return str(content_path)

    def load_parent_chunk(self, parent_id: str) -> Tuple[str, Dict]:
        """加载父chunk"""
        # 查找以parent_id开头的文件
        for file in self.storage_path.glob(f"{parent_id}_*.txt"):
            content_path = file
            metadata_path = file.parent / f"{file.stem}_meta.json"

            with open(content_path, "r", encoding="utf-8") as f:
                content = f.read()

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            return content, metadata

        raise FileNotFoundError(f"Parent chunk {parent_id} not found")

    def get_all_parent_ids(self) -> List[str]:
        """获取所有父chunk的ID"""
        parent_ids = []
        for file in self.storage_path.glob("*_*.txt"):
            # 提取parent_id（第一个下划线之前的部分）
            parent_id = file.name.split("_")[0]
            if parent_id not in parent_ids:
                parent_ids.append(parent_id)
        return parent_ids


class ChromaDBManager:
    """ChromaDB向量存储管理器（直接使用chromadb）"""

    def __init__(self,
                 collection_name: str = config.CHROMA_COLLECTION_NAME,
                 embedding_model_path: str = config.EMBEDDING_MODEL_PATH,
                 persist_directory: str = config.CHROMA_PERSIST_DIR,
                 chroma_server_host: str = config.CHROMA_SERVER_HOST,
                 chroma_server_port: int = config.CHROMA_SERVER_PORT,
                 use_local_persist: bool = config.USE_LOCAL_PERSIST):

        # 初始化本地embedding模型
        self.embedding_function = get_embedding_function(
            model_name=embedding_model_path,
            device=config.EMBEDDING_DEVICE
        )
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name=embedding_model_path,
        #     model_kwargs={'device': config.EMBEDDING_DEVICE},
        #     encode_kwargs={'normalize_embeddings': config.NORMALIZE_EMBEDDINGS}
        # )
        # self.embedding_model = EmbeddingModel(model_name=embedding_model_path)

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chroma_server_host = chroma_server_host
        self.chroma_server_port = chroma_server_port
        self.use_local_persist = use_local_persist

        # 初始化chromadb客户端
        if use_local_persist:
            # 本地持久化模式
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            # 服务器模式
            self.client = chromadb.HttpClient(
                host=chroma_server_host,
                port=chroma_server_port
            )

        self.collection = None

    def _clean_metadata(self, metadata: Dict) -> Dict:
        """清理元数据，确保没有None值，所有值都是字符串、整数、浮点数或布尔值"""
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                # 将None转换为空字符串
                cleaned[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                # 将其他类型转换为字符串
                cleaned[key] = str(value)
        return cleaned

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """生成文本的embedding向量"""
        if not texts:
            return []

        # 使用本地模型生成embedding
        embeddings = self.embeddings.encode(texts)
        return embeddings


    def initialize_collection(self, recreate: bool = False):
        """初始化或加载集合"""
        try:
            # 如果集合已存在且需要重建
            if recreate:
                try:
                    self.client.delete_collection(self.collection_name)
                    print(f"已删除集合: {self.collection_name}")
                except:
                    pass  # 集合不存在

            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )
            print(f"已初始化集合: {self.collection_name}")

        except Exception as e:
            print(f"初始化集合失败: {e}")
            # 尝试创建新集合
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"已创建新集合: {self.collection_name}")

        return self.collection

    def add_documents(self,
                      documents: List[Document],
                      ids: Optional[List[str]] = None,
                      batch_size: int = 100) -> int:
        """添加文档到ChromaDB集合"""
        if not self.collection:
            self.initialize_collection()

        if not documents:
            return 0

        # 准备数据
        texts = [doc.page_content for doc in documents]
        metadatas = [self._clean_metadata(doc.metadata) for doc in documents]

        # 生成ID（如果未提供）
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]

        # 分批处理以避免内存问题
        total_added = 0
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            # 生成embedding
            print(f"正在为批次 {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} 生成embedding...")
            batch_embeddings = self.embedding_function.encode(batch_texts)

            # 添加到集合
            self.collection.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

            total_added += len(batch_texts)
            print(f"已添加 {total_added}/{len(texts)} 个文档")

        return total_added

    def query(self,
              query_text: str,
              n_results: int = 5,
              where: Optional[Dict] = None,
              where_document: Optional[Dict] = None) -> Dict:
        """查询相似文档"""
        if not self.collection:
            self.initialize_collection()

        # 生成查询的embedding
        query_embedding = self.embedding_function.encode(query_text)

        # 执行查询
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
            where_document=where_document
        )

        return results

    def get_collection_info(self) -> Dict:
        """获取集合信息"""
        if not self.collection:
            self.initialize_collection()

        return {
            "name": self.collection.name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata
        }

    def delete_documents(self, ids: List[str]):
        """删除文档"""
        if not self.collection:
            self.initialize_collection()

        self.collection.delete(ids=ids)
        print(f"已删除 {len(ids)} 个文档")

    def update_document(self,
                        document_id: str,
                        document: Document,
                        embedding: Optional[List[float]] = None):
        """更新文档"""
        if not self.collection:
            self.initialize_collection()

        # 如果没有提供embedding，则生成
        if embedding is None:
            embedding = self._generate_embeddings([document.page_content])[0]

        self.collection.update(
            ids=[document_id],
            embeddings=[embedding],
            documents=[document.page_content],
            metadatas=[document.metadata]
        )

    def peek(self, limit: int = 10) -> Dict:
        """查看集合中的部分文档"""
        if not self.collection:
            self.initialize_collection()

        return self.collection.peek(limit=limit)


class DocumentProcessor:
    """文档处理管道"""

    def __init__(self, db_path: str = config.SQLITE_DB_PATH):
        self.db_path = db_path
        self.parent_store = ParentChunkStore()
        self.vector_store = ChromaDBManager()

        # 初始化文本分割器
        self.__parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.PARENT_CHUNK_SIZE,
            chunk_overlap=config.PARENT_CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )

        self.__child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_CHUNK_SIZE,
            chunk_overlap=config.CHILD_CHUNK_OVERLAP
        )
        self.child_generator = DynamicChildChunkGenerator()
        self.__min_parent_size = config.MIN_PARENT_SIZE
        self.__max_parent_size = config.MAX_PARENT_SIZE

    def process_all_documents(self, recreate_vector_store: bool = False):
        """处理所有文档的完整流程"""
        # 1. 初始化向量存储
        self.vector_store.initialize_collection(recreate=recreate_vector_store)

        # 2. 从数据库提取并分块
        parent_chunks, child_chunks = self._create_chunks_from_db()

        # 3. 保存父chunk
        parent_paths = []
        for chunk in parent_chunks:
            path = self.parent_store.save_parent_chunk(chunk)
            parent_paths.append(path)

        print(f"已保存 {len(parent_paths)} 个父chunk到 {self.parent_store.storage_path}")

        # 4. 添加子chunk到向量存储
        # 为每个子chunk生成唯一ID
        child_ids = [f"child_{i}_{hashlib.md5(c.page_content.encode()).hexdigest()[:8]}"
                     for i, c in enumerate(child_chunks)]

        added_count = self.vector_store.add_documents(child_chunks, child_ids)

        return {
            "parent_count": len(parent_chunks),
            "child_count": len(child_chunks),
            "added_to_vector_store": added_count,
            "vector_store_size": self.vector_store.get_collection_info(),
            "parent_storage_dir": str(self.parent_store.storage_path)
        }

    def _read_texts_from_db(self) -> List[dict]:
        """从SQLite读取文本数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 根据您的表结构调整查询
        cursor.execute("""
            SELECT guideline_id, title, category, disease, content, source, updated_at 
            FROM rehab_guidelines
        """)

        texts = []
        for row in cursor.fetchall():
            # 将行转换为字典
            text_dict = {
                "guideline_id": row[0],
                "title": row[1],
                "category": row[2],
                "disease": row[3],
                "content": row[4],
                "source": row[5],
                "updated_time": row[6]
            }
            print(text_dict)
            texts.append(text_dict)

        conn.close()
        return texts

    def _create_chunks_from_db(self) -> Tuple[List[Document], List[Document]]:
        """从数据库创建父子chunk"""
        all_parent_chunks, all_child_chunks = [], []

        texts = self._read_texts_from_db()

        for i, text_data in enumerate(texts):
            parent_chunks, child_chunks = self._create_chunks_single(text_data, doc_id=i)
            all_parent_chunks.extend(parent_chunks)
            all_child_chunks.extend(child_chunks)

        return all_parent_chunks, all_child_chunks

    def _create_chunks_single(self, text_data: dict, doc_id: int = None):
        """处理单个文本"""
        # 创建初始父块
        initial_chunk = Document(
            page_content=text_data["content"],
            metadata={
                "source": text_data.get("source", f"db_doc_{doc_id}"),
                "title": text_data.get("title", ""),
                "doc_id": text_data.get("guideline_id", doc_id),
                "original_id": text_data.get("guideline_id"),
                "category": text_data.get("category", ""),
                "updated_at": text_data.get("updated_at", ""),
                "disease": text_data.get("disease", ""),
                "processed_at": datetime.now().isoformat()
            }
        )

        # 进行父分割
        parent_chunks = self.__parent_splitter.split_documents([initial_chunk])

        # 合并小段、分割大段
        merged_parents = self._merge_small_parents(parent_chunks)
        split_parents = self._split_large_parents(merged_parents)
        cleaned_parents = self._clean_small_chunks(split_parents)

        all_parent_chunks, all_child_chunks = [], []
        self._create_child_chunks(
            all_parent_chunks,
            all_child_chunks,
            cleaned_parents,
            text_data.get("guideline_id", doc_id),
            # text_data
        )

        return all_parent_chunks, all_child_chunks

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

    def _split_large_parents(self, chunks: List[Document]) -> List[Document]:
        """分割大段"""
        split_chunks = []

        for chunk in chunks:
            if len(chunk.page_content) <= self.__max_parent_size:
                split_chunks.append(chunk)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.__max_parent_size,
                    chunk_overlap=config.PARENT_CHUNK_OVERLAP
                )
                sub_chunks = splitter.split_documents([chunk])
                # 为每个子块添加父块ID前缀
                for i, sub_chunk in enumerate(sub_chunks):
                    if "original_parent_id" not in sub_chunk.metadata:
                        sub_chunk.metadata["original_parent_id"] = chunk.metadata.get("original_id", "")
                split_chunks.extend(sub_chunks)

        return split_chunks

    def _clean_small_chunks(self, chunks: List[Document]) -> List[Document]:
        """清理小段"""
        cleaned = []

        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < self.__min_parent_size:
                if cleaned:
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    for k, v in chunk.metadata.items():
                        if k in cleaned[-1].metadata and k not in ["source", "title", "doc_id", "original_id"]:
                            if cleaned[-1].metadata[k] != v:
                                cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} | {v}"
                elif i < len(chunks) - 1:
                    chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    for k, v in chunk.metadata.items():
                        if k in chunks[i + 1].metadata and k not in ["source", "title", "doc_id", "original_id"]:
                            if chunks[i + 1].metadata[k] != v:
                                chunks[i + 1].metadata[k] = f"{v} | {chunks[i + 1].metadata[k]}"
                else:
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)

        return cleaned

    def _create_child_chunks(self, all_parent_chunks, all_child_chunks, parent_chunks, doc_id):
        """创建子块"""
        for i, p_chunk in enumerate(parent_chunks):
            parent_id = f"doc_{doc_id}_parent_{i}"
            p_chunk.metadata["parent_id"] = parent_id

            all_parent_chunks.append(p_chunk)

            # 创建子块
            # child_chunks = self.__child_splitter.split_documents([p_chunk])
            # 子chunk：动态生成
            child_chunks = self.child_generator.generate_child_chunks(
                parent_content=p_chunk["content"],
                title=p_chunk["title"],
                category=p_chunk["category"],
                disease=p_chunk["disease"],
                source=p_chunk["source"]
            )
            for j, child in enumerate(child_chunks):
                child["metadata"].update({
                    "parent_id": parent_id,
                    "doc_id": doc_id,
                    "child_index": j,
                    "total_children": len(child_chunks)
                })
            all_child_chunks.extend(child_chunks)

    def add_incremental_documents(self, new_texts: List[dict]):
        """增量添加新文档"""
        # 处理新文档
        new_parent_chunks, new_child_chunks = [], []

        for i, text_data in enumerate(new_texts):
            start_idx = len(self.parent_store.get_all_parent_ids())
            parent_chunks, child_chunks = self._create_chunks_single(text_data, doc_id=start_idx + i)
            new_parent_chunks.extend(parent_chunks)
            new_child_chunks.extend(child_chunks)

        # 保存新父chunk
        for chunk in new_parent_chunks:
            self.parent_store.save_parent_chunk(chunk)

        # 添加到向量存储
        child_ids = [f"child_{len(new_child_chunks) + i}_{hashlib.md5(c.page_content.encode()).hexdigest()[:8]}"
                     for i, c in enumerate(new_child_chunks)]

        added_count = self.vector_store.add_documents(new_child_chunks, child_ids)

        return {
            "new_parents": len(new_parent_chunks),
            "new_children": len(new_child_chunks),
            "added_to_vector_store": added_count
        }

    def search_documents(self,
                         query_text: str,
                         n_results: int = 5,
                         with_parent: bool = True,
                         return_similarity: bool = True,  # 是否返回相似度分数
                         where: Optional[Dict] = None) -> List[Dict]:
        """搜索文档并返回相关父chunk"""
        # 搜索相似子chunk
        results = self.vector_store.query(
            query_text=query_text,
            n_results=n_results,
            where=where
        )

        if not results.get("documents") or not results["documents"][0]:
            return []

        enhanced_results = []

        for i in range(len(results["documents"][0])):
            doc_content = results["documents"][0][i]

            doc_metadata = {}
            if results.get("metadatas") and results["metadatas"][0] and i < len(results["metadatas"][0]):
                doc_metadata = results["metadatas"][0][i]

            doc_id = None
            if results.get("ids") and results["ids"][0] and i < len(results["ids"][0]):
                doc_id = results["ids"][0][i]

            # 获取距离（余弦距离，越小越相似）
            distance = None
            if results.get("distances") and results["distances"][0] and i < len(results["distances"][0]):
                distance = results["distances"][0][i]

            # 计算相似度分数
            if return_similarity and distance is not None:
                # 余弦距离转换为相似度（1 - 距离）
                similarity = 1 - distance
            else:
                similarity = distance

            parent_content = None
            parent_metadata = {}

            if with_parent:
                parent_id = doc_metadata.get("parent_id")
                if parent_id:
                    try:
                        parent_content, parent_metadata = self.parent_store.load_parent_chunk(parent_id)
                    except FileNotFoundError:
                        parent_content = "父chunk未找到"

            enhanced_results.append({
                "id": doc_id or f"unknown_{i}",
                "child_content": doc_content,
                "child_metadata": doc_metadata,
                "parent_content": parent_content,
                "parent_metadata": parent_metadata,
                "distance": distance,  # 原始距离
                "similarity": similarity  # 相似度分数
            })

        return enhanced_results

if __name__ == '__main__':
    processor = DocumentProcessor(db_path=config.SQLITE_DB_PATH)

    # 1. 首次处理所有文档
    print("开始首次处理所有文档...")
    result = processor.process_all_documents(recreate_vector_store=True)
    print(f"处理完成: {result}")
    #
    # # 2. 增量添加新文档
    # print("\n增量添加新文档...")
    # # 假设从数据库获取新文档
    # new_docs = [
    #     {
    #         "guideline_id": 101,
    #         "title": "新文档标题",
    #         "content": "文档内容_测试部分",
    #         "category": "心理",
    #         "disease": "test",
    #         "source": "新增来源",
    #         "updated_at": "now"
    #     }
    # ]
    # incremental_result = processor.add_incremental_documents(new_docs)
    # print(f"增量添加完成: {incremental_result}")

    # 3. 搜索文档
    # 完整的搜索示例
    query_text = "高血压患者的饮食指南"
    search_results = processor.search_documents(
        query_text=query_text,
        n_results=5,
        with_parent=True,
        return_similarity=True
    )

    print(f"查询: '{query_text}'")
    print(f"找到 {len(search_results)} 个相关结果")
    print("=" * 80)

    for i, result in enumerate(search_results):
        print(f"\n📄 结果 {i + 1}")
        print(f"   ID: {result['id']}")
        print(f"   相似度: {result['similarity']:.4f}")
        print(f"   距离: {result['distance']:.4f}")
        print(f"   标题: {result['child_metadata'].get('title', '无标题')}")
        print(f"   来源: {result['child_metadata'].get('source', '未知来源')}")
        print(f"   疾病: {result['child_metadata'].get('disease', '未知疾病')}")
        print(f"   类别: {result['child_metadata'].get('category', '未知类别')}")

        # 显示子chunk内容
        print(f"\n   子chunk内容:")
        child_content = result['child_content'].strip()
        if len(child_content) > 200:
            print(f"   {child_content[:200]}...")
        else:
            print(f"   {child_content}")

        # 显示父chunk内容
        if result.get('parent_content'):
            print(f"\n   父chunk内容:")
            parent_content = result['parent_content'].strip()
            if len(parent_content) > 300:
                print(f"   {parent_content[:300]}...")
            else:
                print(f"   {parent_content}")

        print(f"\n   元数据:")
        for key, value in result['child_metadata'].items():
            if key not in ['title', 'source', 'disease', 'category']:
                print(f"     {key}: {value}")

        print("-" * 80)
# 配置文件示例
"""
# config.py
import torch

# SQLite配置
SQLITE_DB_PATH = "documents.db"

# 分割配置
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 50
MIN_PARENT_SIZE = 100
MAX_PARENT_SIZE = 3000

# 存储配置
PARENT_STORAGE_PATH = "./data/parent_chunks"

# ChromaDB配置
CHROMA_COLLECTION_NAME = "document_collection"
CHROMA_PERSIST_DIR = "./data/chroma_db"

# Embedding模型配置
EMBEDDING_MODEL_PATH = "/path/to/your/local/model"  # 例如: "BAAI/bge-base-zh"
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORMALIZE_EMBEDDINGS = True
"""