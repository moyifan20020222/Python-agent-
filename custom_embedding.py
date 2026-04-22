# project/db/custom_embeddings.py
"""
自定义embedding模型包装
支持本地M3E/BGE模型
"""
import os
from typing import List, Union
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# 设置离线模式
os.environ.update({
    'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_OFFLINE': '1',
    'TF_CPP_MIN_LOG_LEVEL': '3',
})


class LocalEmbeddingFunction:
    """本地模型embedding函数"""

    def __init__(self, model_path: str, device: str = "cpu", max_length: int = 512):
        """
        初始化本地模型

        Args:
            model_path: 本地模型路径或HuggingFace模型名称
            device: cpu 或 cuda
            max_length: 最大序列长度
        """
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载模型"""
        print(f"🔄 加载本地embedding模型: {self.model_path}")

        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            import torch.nn.functional as F

            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True
            )

            # 加载模型
            self.model = AutoModel.from_pretrained(self.model_path)

            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()

            self.model.eval()

            # 测试模型
            test_text = ["测试文本"]
            test_embedding = self.encode(test_text)

            print(f"✅ 模型加载成功，维度: {len(test_embedding[0])}")
            print(f"   设备: {self.device}")
            print(f"   最大长度: {self.max_length}")

        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            raise

    def encode(self, texts: Union[str, List[str]], **kwargs) -> List[List[float]]:
        """
        编码文本为向量

        Args:
            texts: 文本或文本列表
        Returns:
            向量列表
        """
        import torch
        import torch.nn.functional as F

        if not self.tokenizer or not self.model:
            raise ValueError("模型未加载")

        # 确保texts是列表
        if isinstance(texts, str):
            texts = [texts]

        # 分批处理，避免内存问题
        batch_size = 8
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # 编码
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

                # 使用[CLS] token
                embeddings = outputs.last_hidden_state[:, 0]

                # 归一化
                embeddings = F.normalize(embeddings, p=2, dim=1)

                if self.device == "cuda":
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings.numpy().tolist())

        return all_embeddings

    def __call__(self, input_texts: Union[str, List[str]]) -> List[List[float]]:
        """调用接口，兼容ChromaDB"""
        return self.encode(input_texts)

    def get_dimension(self) -> int:
        """获取向量维度"""
        test_embedding = self.encode(["测试"])
        return len(test_embedding[0])


# 预设模型路径
MODEL_PATHS = {
    "m3e-base": "D:/Desktop/company/Agent-rag/agentic-rag-for-dummies/models/m3e-base",
    "bge-large-zh": "D:/Desktop/company/Agent-rag/agentic-rag-for-dummies/models/bge-large-zh-v1.5",  # 在线下载
    "bge-medical": "D:/Desktop/company/Agent-rag/agentic-rag-for-dummies/models/Hu0922/BGE_Medical",
}


def get_embedding_function(model_name: str = "m3e-base", **kwargs) -> LocalEmbeddingFunction:
    """获取embedding函数"""
    if model_name in MODEL_PATHS:
        model_path = MODEL_PATHS[model_name]
    else:
        model_path = model_name

    return LocalEmbeddingFunction(model_path, **kwargs)


# 测试函数
if __name__ == "__main__":
    # 测试M3E模型
    print("🧪 测试M3E embedding模型...")
    try:
        embedder = get_embedding_function("m3e-base", device="cpu")

        texts = ["高血压饮食建议", "糖尿病康复训练"]
        embeddings = embedder.encode(texts)

        print(f"✅ 测试成功！")
        print(f"   文本数量: {len(texts)}")
        print(f"   向量维度: {len(embeddings[0])}")
        print(f"   模型维度: {embedder.get_dimension()}")

        # 测试相似度
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        if len(embeddings) >= 2:
            sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            print(f"   相似度: {sim:.4f}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()

    # 测试BGE模型
    print("\n🧪 测试BGE embedding模型...")
    try:
        embedder = get_embedding_function("bge-medical", device="cpu")

        texts = ["心脏病注意事项", "骨折康复指南"]
        embeddings = embedder.encode(texts)

        print(f"✅ 测试成功！")
        print(f"   模型维度: {embedder.get_dimension()}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")