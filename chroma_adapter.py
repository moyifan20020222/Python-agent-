# project/db/chroma_adapter_fixed.py
"""
修复的ChromaDB适配器
"""
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain.vectorstores.base import VectorStore
import chromadb
from chromadb.config import Settings
import numpy as np


class ChromaAdapterFixed(VectorStore):
    """修复的ChromaDB适配器"""

    def __init__(
            self,
            chroma_path: str,
            collection_name: str = "rehab_embedding"
    ):
        """
        初始化适配器

        Args:
            chroma_path: ChromaDB存储路径
            collection_name: 集合名称
        """
        self.chroma_path = chroma_path
        self.collection_name = collection_name

        # 连接ChromaDB
        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # 获取集合
        self.collection = self.client.get_collection(collection_name)

        print(f"✅ 已加载ChromaDB: {chroma_path}")
        print(f"   集合: {collection_name}")
        print(f"   文档数: {self.collection.count()}")

    def similarity_search(
            self,
            query: str,
            k: int = 4,
            score_threshold: Optional[float] = None,
            **kwargs
    ) -> List[Document]:
        """
        相似度搜索，兼容LangChain接口

        Args:
            query: 查询文本
            k: 返回结果数
            score_threshold: 分数阈值
        """
        try:
            # 执行查询
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )

            if not results['documents'] or not results['documents'][0]:
                return []

            # 转换为LangChain Document格式
            documents = []
            for i in range(len(results['documents'][0])):
                # 获取文档内容
                doc_content = results['documents'][0][i]

                # 获取元数据
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}

                # 计算相关性分数
                distance = results['distances'][0][i] if results['distances'] else 0.0
                similarity = 1.0 / (1.0 + distance)

                # 添加分数到元数据
                metadata["score"] = similarity

                # 创建文档
                doc = Document(
                    page_content=doc_content,
                    metadata=metadata
                )
                documents.append(doc)

            # 应用分数阈值过滤
            if score_threshold is not None:
                filtered_docs = []
                for doc in documents:
                    if doc.metadata.get("score", 0.0) >= score_threshold:
                        filtered_docs.append(doc)
                return filtered_docs

            return documents

        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []

    def similarity_search_with_score(
            self,
            query: str,
            k: int = 4,
            **kwargs
    ) -> List[tuple[Document, float]]:
        """
        带分数的相似度搜索

        Returns:
            [(Document, score), ...]
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )

            if not results['documents'] or not results['documents'][0]:
                return []

            documents_with_scores = []
            for i in range(len(results['documents'][0])):
                doc_content = results['documents'][0][i]
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}

                distance = results['distances'][0][i] if results['distances'] else 0.0
                # 将距离转换为相似度分数
                score = 1.0 / (1.0 + distance)

                doc = Document(
                    page_content=doc_content,
                    metadata=metadata
                )
                documents_with_scores.append((doc, score))

            return documents_with_scores

        except Exception as e:
            print(f"❌ 带分数搜索失败: {e}")
            return []

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """根据ID获取文档"""
        try:
            results = self.collection.get(ids=ids)

            documents = []
            for i in range(len(results['documents'])):
                doc = Document(
                    page_content=results['documents'][i],
                    metadata=results['metadatas'][i] if results['metadatas'] else {}
                )
                documents.append(doc)

            return documents

        except Exception as e:
            print(f"❌ 根据ID获取失败: {e}")
            return []

    def get_all_documents(self, limit: int = 100) -> List[Document]:
        """获取所有文档"""
        try:
            results = self.collection.get(limit=limit)

            documents = []
            for i in range(len(results['documents'])):
                doc = Document(
                    page_content=results['documents'][i],
                    metadata=results['metadatas'][i] if results['metadatas'] else {}
                )
                documents.append(doc)

            return documents

        except Exception as e:
            print(f"❌ 获取所有文档失败: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计"""
        return {
            "count": self.collection.count(),
            "name": self.collection.name,
            "path": self.chroma_path
        }

    # VectorStore接口要求的其他方法（简化实现）
    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None, **kwargs):
        """添加文本（简化实现）"""
        print("⚠️  add_texts方法需要自定义embedding函数")
        return []

    def from_texts(self, texts: List[str], embedding, metadatas: Optional[List[dict]] = None, **kwargs):
        """从文本创建（简化实现）"""
        print("⚠️  from_texts方法需要完整实现")
        return self

    @classmethod
    def from_documents(cls, documents: List[Document], embedding, **kwargs):
        """从文档创建（简化实现）"""
        print("⚠️  from_documents方法需要完整实现")
        return cls(**kwargs)


# 工厂函数
def load_chroma_adapter(
        chroma_path: str = "./chroma_embedding",
        collection_name: str = "rehab_embedding"
) -> ChromaAdapterFixed:
    """加载ChromaDB适配器"""
    return ChromaAdapterFixed(chroma_path, collection_name)