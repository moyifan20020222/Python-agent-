import sqlite3
import config
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    def __init__(self, db_path: str = config.SQLITE_DB_PATH):
        self.db_path = db_path
        # 父分割器 - 使用递归分割器处理纯文本
        self.__parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.PARENT_CHUNK_SIZE,  # 建议设置为较大的值，如2000-4000
            chunk_overlap=config.PARENT_CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        # 子分割器
        self.__child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_CHUNK_SIZE,
            chunk_overlap=config.CHILD_CHUNK_OVERLAP
        )
        self.__min_parent_size = config.MIN_PARENT_SIZE
        self.__max_parent_size = config.MAX_PARENT_SIZE

    def create_chunks(self) -> Tuple[List[Document], List[Document]]:
        """从SQLite数据库提取文本并分块"""
        all_parent_chunks, all_child_chunks = [], []

        # 从数据库读取文本
        texts = self._read_texts_from_db()

        for i, text_data in enumerate(texts):
            # text_data应该是包含文本和元数据的字典
            parent_chunks, child_chunks = self.create_chunks_single(text_data, doc_id=i)
            all_parent_chunks.extend(parent_chunks)
            all_child_chunks.extend(child_chunks)

        return all_parent_chunks, all_child_chunks

    def _read_texts_from_db(self) -> List[dict]:
        """从SQLite读取文本数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 根据您的表结构调整查询
        cursor.execute("""
            SELECT guideline_id, title, category, disease, content, source, updated_at 
            FROM documents  rehab_guidelines
            ORDER BY id
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
            texts.append(text_dict)

        conn.close()
        return texts

    def create_chunks_single(self, text_data: dict, doc_id: int = None):
        """处理单个文本"""
        # 创建初始父块
        initial_chunk = Document(
            page_content=text_data["content"],
            metadata={
                "source": text_data.get("source", f"db_doc_{doc_id}"),
                "title": text_data.get("title", ""),
                "doc_id": text_data.get("guideline_id", doc_id)
            }
        )

        # 进行父分割
        parent_chunks = self.__parent_splitter.split_documents([initial_chunk])

        # 合并小段、分割大段
        merged_parents = self.__merge_small_parents(parent_chunks)
        split_parents = self.__split_large_parents(merged_parents)
        cleaned_parents = self.__clean_small_chunks(split_parents)

        all_parent_chunks, all_child_chunks = [], []
        self.__create_child_chunks(
            all_parent_chunks,
            all_child_chunks,
            cleaned_parents,
            text_data.get("guideline_id", doc_id)
        )

        return all_parent_chunks, all_child_chunks

    def __merge_small_parents(self, chunks: List[Document]) -> List[Document]:
        """合并小段（保持原逻辑）"""
        if not chunks:
            return []

        merged, current = [], None

        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                current.page_content += "\n\n" + chunk.page_content
                # 合并元数据
                for k, v in chunk.metadata.items():
                    if k in current.metadata and k != "source" and k != "title":
                        current.metadata[k] = f"{current.metadata[k]} -> {v}"

            if len(current.page_content) >= self.__min_parent_size:
                merged.append(current)
                current = None

        if current:
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                for k, v in current.metadata.items():
                    if k in merged[-1].metadata and k != "source" and k != "title":
                        merged[-1].metadata[k] = f"{merged[-1].metadata[k]} -> {v}"
            else:
                merged.append(current)

        return merged

    def __split_large_parents(self, chunks: List[Document]) -> List[Document]:
        """分割大段（保持原逻辑）"""
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
                split_chunks.extend(sub_chunks)

        return split_chunks

    def __clean_small_chunks(self, chunks: List[Document]) -> List[Document]:
        """清理小段（保持原逻辑）"""
        cleaned = []

        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < self.__min_parent_size:
                if cleaned:
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    for k, v in chunk.metadata.items():
                        if k in cleaned[-1].metadata and k != "source" and k != "title":
                            cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                elif i < len(chunks) - 1:
                    chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    for k, v in chunk.metadata.items():
                        if k in chunks[i + 1].metadata and k != "source" and k != "title":
                            chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                else:
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)

        return cleaned

    def __create_child_chunks(self, all_parent_chunks, all_child_chunks, parent_chunks, doc_id):
        """创建子块（稍作调整）"""
        for i, p_chunk in enumerate(parent_chunks):
            parent_id = f"doc_{doc_id}_parent_{i}"
            p_chunk.metadata["parent_id"] = parent_id

            all_parent_chunks.append(p_chunk)

            # 创建子块
            child_chunks = self.__child_splitter.split_documents([p_chunk])
            for child in child_chunks:
                child.metadata.update({
                    "parent_id": parent_id,
                    "doc_id": doc_id
                })
            all_child_chunks.extend(child_chunks)


