# config.py
# SQLite配置
SQLITE_DB_PATH = "D:/Desktop/company/Agent-rag/agentic-rag-for-dummies/project/db/rehab.db"
CHROMA_PATH = "D:/Desktop/company/Agent-rag/agentic-rag-for-dummies/project/db/chroma_final"
# 父块配置
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200

# 子块配置
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 50

# 最小/最大父块大小
MIN_PARENT_SIZE = 1000
MAX_PARENT_SIZE = 3000

# 存储路径
PARENT_STORAGE_PATH = "./data/parent_chunks"

# ChromaDB配置
CHROMA_COLLECTION_NAME = "medical_document_collection"
CHROMA_PERSIST_DIR = "./data/chroma_db"

# Embedding模型配置
EMBEDDING_MODEL_PATH = "D:\\Desktop\\company\\Agent-rag\\agentic-rag-for-dummies\\models\\m3e-base"  # 例如: "BAAI/bge-base-zh"
EMBEDDING_DEVICE = "cpu"  # 或 "cpu"
NORMALIZE_EMBEDDINGS = True

CHROMA_SERVER_HOST = "127.0.0.1"
CHROMA_SERVER_PORT = "5000"
USE_LOCAL_PERSIST = True

SESSION_STORAGE_PATH = "./data/session"

BASE_TOKEN_THRESHOLD = 2000
TOKEN_GROWTH_FACTOR = 0.9
chroma_path = "./chroma_final"

PARENT_STORE_PATH = "./data/parent_docs_embedding"
BGE_RERANKER_PATH = "D:\\Desktop\\company\\Agent-rag\\agentic-rag-for-dummies\\models\\bge-reranker-base"
