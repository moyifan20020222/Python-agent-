"""
企业级混合检索系统
- 支持向量检索 + 关键词检索 + 元数据过滤
- 智能权重动态调整
- 医学领域优化
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime
import logging
from project.rehab_core.retrieval.bge_reranker import BGEReranker
logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25关键词检索器 - 性能与过滤优化版"""

    def __init__(self, documents: List[Dict[str, Any]] = None):
        self.documents = documents or []
        self.term_freqs = {}
        self.doc_freqs = {}
        self.idf = {}
        self.total_docs = 0
        self.avgdl = 0

        # 新增：保存倒排索引和文档长度，避免搜索时重复分词
        self.inverted_index = {}
        self.doc_lengths = []

        if documents:
            self._build_index()

    def _build_index(self):
        """构建真正的倒排索引"""
        medical_terms = {'术后': ['手术后'], '康复': ['恢复'], '禁忌症': ['禁用']}
        total_length = 0

        for i, doc in enumerate(self.documents):
            content = doc.get('page_content', '')
            metadata = doc.get('metadata', {})
            full_text = f"{content} {metadata.get('title', '')} {metadata.get('disease', '')} {metadata.get('intent_type', '')}"

            for term, variants in medical_terms.items():
                for variant in variants:
                    full_text = full_text.replace(variant, term)

            terms = self._tokenize(full_text)
            self.doc_lengths.append(len(terms))
            total_length += len(terms)

            # 构建倒排索引：词 -> {文档ID: 词频}
            doc_term_counts = self._count_terms(terms)
            for term, count in doc_term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][i] = count

                self.term_freqs[term] = self.term_freqs.get(term, 0) + count

        self.total_docs = len(self.documents)
        self.avgdl = total_length / max(1, self.total_docs)

        for term, doc_dict in self.inverted_index.items():
            self.doc_freqs[term] = len(doc_dict)
            self.idf[term] = np.log((self.total_docs - self.doc_freqs[term] + 0.5) /
                                    (self.doc_freqs[term] + 0.5) + 1)

    def _tokenize(self, text: str) -> List[str]:
        import jieba  # 强烈建议换成 jieba
        return [t.lower() for t in jieba.lcut(text) if len(t.strip()) > 1]

    def _count_terms(self, terms: List[str]) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(terms))

    def search(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """
        BM25 搜索算法实现，支持元数据过滤

        Args:
            query: 用户查询字符串
            k: 返回的搜索结果数量，默认为5
            filters: 元数据过滤条件字典，例如 {"category": "news", "year": 2023}

        Returns:
            搜索结果列表，每个结果包含文档内容、元数据和相关性分数
        """

        # 检查文档集合是否为空
        if not self.documents:
            return []

        # 将查询字符串分词处理
        query_terms = self._tokenize(query)

        # 初始化所有文档的得分字典
        # 键：文档索引，值：BM25分数
        scores = {i: 0.0 for i in range(self.total_docs)}

        # ========== BM25 算法核心计算 ==========
        # 遍历查询中的每个词项
        for term in query_terms:
            # 如果词项不在IDF字典中，跳过（意味着这个词在语料库中不存在）
            if term not in self.idf:
                continue

            # 获取该词项的逆文档频率
            idf = self.idf[term]

            # 只遍历包含该词项的文档（利用倒排索引优化性能）
            for doc_idx, tf in self.inverted_index[term].items():
                # 获取当前文档的长度
                dl = self.doc_lengths[doc_idx]

                # 计算BM25分数：
                # 公式: idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))
                # 这里 k1=1.2, b=0.75
                score = idf * (tf * 2.2) / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / self.avgdl))

                # 累加该词项对当前文档的贡献分数
                scores[doc_idx] += score

        # ========== 元数据过滤阶段 ==========
        # 获取所有有分数的文档索引
        valid_indices = scores.keys()

        # 如果有过滤条件，应用元数据过滤
        if filters:
            valid_indices = []  # 重置有效索引列表

            # 遍历所有有分数的文档
            for i in scores.keys():
                # 获取文档的元数据
                doc_meta = self.documents[i].get('metadata', {})
                match = True  # 默认匹配

                # 检查文档元数据是否满足所有过滤条件
                for fk, fv in filters.items():
                    # 💡 防错处理：如果大模型解析出"综合"或空值，代表不限制该维度，跳过过滤
                    if not fv or fv == "综合":
                        continue

                    # 如果不匹配，标记为不匹配并跳出
                    if doc_meta.get(fk) != fv:
                        match = False
                        break

                    # 如果满足所有条件，添加到有效索引列表
                if match:
                    valid_indices.append(i)

        # ========== 结果排序和筛选 ==========
        # 过滤掉分数为0的文档，并按分数降序排序
        # 1. 从有效索引中筛选分数>0的文档
        # 2. 按分数降序排序
        # 3. 取前k个结果
        top_indices = sorted([i for i in valid_indices if scores[i] > 0],
                             key=lambda i: scores[i], reverse=True)[:k]

        # ========== 格式化返回结果 ==========
        results = []
        for i in top_indices:
            # 获取原始文档
            doc = self.documents[i]

            # 构建结果字典，保持与 VectorRetriever 一致的格式
            result = {
                'id': doc.get('metadata', {}).get('child_id', f'bm25_{i}'),  # 文档ID
                'document': doc.get('page_content', ''),  # 文档内容
                'metadata': doc.get('metadata', {}),  # 文档元数据
                'score': scores[i],  # BM25绝对分数
                'source': 'bm25'  # 标识来源
            }
            results.append(result)

        return results


class VectorRetriever:
    """向量检索器 - 医学优化版"""
    
    def __init__(self, collection: Any = None, embedding_model: Any = None):
        self.collection = collection
        self.embedding_model = embedding_model
        self.cache = {}
    
    def search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """向量搜索"""
        try:
            import json
            # 💡 防错 1：清洗 filter_dict，去掉值为"综合"的键，避免 ChromaDB 查不到
            clean_filters = None
            if filter_dict:
                clean_filters = {k: v for k, v in filter_dict.items() if v and v != "综合"}
                # 如果清理后变为空字典，必须设为 None，否则 ChromaDB 会报错
                if not clean_filters:
                    clean_filters = None

            # 💡 防错 2：使用 json.dumps 并按键排序，保证缓存 Key 唯一且稳定
            filter_str = json.dumps(clean_filters, sort_keys=True) if clean_filters else "None"
            cache_key = f"{query}_{k}_{filter_str}"

            if cache_key in self.cache:
                return self.cache[cache_key]

            # 生成查询向量
            query_embedding = self.embedding_model.encode([query])[0]

            # 执行检索 (传入清洗后的 where 条件)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=clean_filters
            )
            
            # 格式化结果
            formatted_results = []
            if results.get('documents') and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    doc = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'score': 1.0 / (1.0 + results['distances'][0][i]),  # 转换为相似度
                        'source': 'vector'
                    }
                    formatted_results.append(doc)
            
            self.cache[cache_key] = formatted_results
            return formatted_results
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []


class HybridRetrievalSystem:
    """混合检索系统主类"""
    
    def __init__(self, vector_retriever: VectorRetriever, bm25_retriever: BM25Retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = BGEReranker()
        self.query_analyzer = QueryAnalyzer()
    
    def search(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """
        混合检索主方法
        企业级特性：
        - 查询类型识别
        - 动态权重调整
        - 结果融合排序
        """
        start_time = datetime.now()

        # 1. 分析查询类型
        query_type = self.query_analyzer.analyze_query(query)
        logger.info(f"Query type: {query_type}")

        # 2. 根据查询类型调整权重
        weights = self._get_weights_by_query_type(query_type)
        vector_results = self.vector_retriever.search(query, k * 2, filters)
        bm25_results = self.bm25_retriever.search(query, k * 2)

        # 2. 融合结果
        fused_results = self._fuse_results(vector_results, bm25_results, weights, query_type)

        # 3. 专业重排
        documents = [result['document'] for result in fused_results]
        reranked = self.reranker.rerank(query, documents, k)

        # 4. 合并元数据
        final_results = []
        for i, rerank_result in enumerate(reranked):
            original_result = next((r for r in fused_results if r['document'] == rerank_result['document']), None)
            if original_result:
                final_result = {
                    'document': rerank_result['document'],
                    'score': rerank_result['rerank_score'],
                    'metadata': original_result.get('metadata', {}),
                    'source': 'hybrid+bge_rerank'
                }
                final_results.append(final_result)

        # 记录性能指标
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Hybrid search completed in {elapsed:.3f}s, returned {len(final_results)} results")
        
        return final_results
    
    def _get_weights_by_query_type(self, query_type: str) -> Dict[str, float]:
        """根据查询类型获取权重"""
        weights_map = {
            'symptom_query': {'vector': 0.4, 'bm25': 0.6},  # 症状查询更依赖关键词
            'principle_query': {'vector': 0.7, 'bm25': 0.3},  # 原则查询更依赖语义
            'procedure_query': {'vector': 0.5, 'bm25': 0.5},  # 操作步骤平衡
            'general': {'vector': 0.6, 'bm25': 0.4}  # 默认
        }
        return weights_map.get(query_type, weights_map['general'])
    
    def _fuse_results(self, 
                     vector_results: List[Dict],
                     bm25_results: List[Tuple[int, float]],
                     weights: Dict[str, float],
                     query_type: str) -> List[Dict]:
        """融合检索结果"""
        # 创建结果映射
        result_map = {}
        
        # 处理向量结果
        for i, result in enumerate(vector_results):
            result_id = result['id']
            if result_id not in result_map:
                result_map[result_id] = {
                    'original': result,
                    'vector_score': result['score'],
                    'bm25_score': 0.0,
                    'rank_vector': i
                }
        
        # 处理BM25结果
        for i, (doc_idx, bm25_score) in enumerate(bm25_results):
            if doc_idx < len(self.bm25_retriever.documents):
                doc_id = self.bm25_retriever.documents[doc_idx].get('metadata', {}).get('child_id')
                if doc_id and doc_id in result_map:
                    result_map[doc_id]['bm25_score'] = bm25_score / max(1, max(score for _, score in bm25_results) if bm25_results else 1)
                elif doc_id:
                    # 新结果
                    result_map[doc_id] = {
                        'original': self.bm25_retriever.documents[doc_idx],
                        'vector_score': 0.0,
                        'bm25_score': bm25_score / max(1, max(score for _, score in bm25_results) if bm25_results else 1),
                        'rank_bm25': i
                    }
        
        # 计算融合分数
        fused_list = []
        for doc_id, data in result_map.items():
            # 归一化分数
            vector_norm = data['vector_score']
            bm25_norm = data['bm25_score']
            
            # 动态权重调整
            if query_type == 'symptom_query':
                # 症状查询：BM25权重更高
                fused_score = weights['vector'] * vector_norm + weights['bm25'] * bm25_norm
            else:
                # 其他查询：向量权重更高
                fused_score = weights['vector'] * vector_norm + weights['bm25'] * bm25_norm
            
            fused_list.append({
                'id': doc_id,
                'score': fused_score,
                'original': data['original'],
                'vector_score': vector_norm,
                'bm25_score': bm25_norm,
                'query_type': query_type
            })
        
        # 排序
        fused_list.sort(key=lambda x: x['score'], reverse=True)
        
        # 返回格式化结果
        return [{
            'id': item['id'],
            'document': item['original'].get('page_content', ''),
            'metadata': item['original'].get('metadata', {}),
            'score': item['score'],
            'vector_score': item['vector_score'],
            'bm25_score': item['bm25_score'],
            'source': 'hybrid',
            'query_type': item['query_type']
        } for item in fused_list]


class QueryAnalyzer:
    """查询分析器 - 医学领域专用"""
    
    def __init__(self):
        self.symptom_keywords = [
            '疼痛', '发热', '呕吐', '头晕', '乏力', '咳嗽', '呼吸困难',
            '症状', '表现', '不适', '检查结果'
        ]
        self.principle_keywords = [
            '原则', '指南', '建议', '推荐', '注意事项', '禁忌症',
            '康复计划', '治疗方案', '护理要点'
        ]
        self.procedure_keywords = [
            '手术', '操作', '步骤', '方法', '流程', '技术',
            '术后', '当天', '第X天', '小时'
        ]
    
    def analyze_query(self, query: str) -> str:
        """分析查询类型"""
        query_lower = query.lower()
        
        symptom_count = sum(1 for kw in self.symptom_keywords if kw in query_lower)
        principle_count = sum(1 for kw in self.principle_keywords if kw in query_lower)
        procedure_count = sum(1 for kw in self.procedure_keywords if kw in query_lower)
        
        # 判断类型
        if symptom_count > max(principle_count, procedure_count) and symptom_count > 1:
            return 'symptom_query'
        elif principle_count > max(symptom_count, procedure_count) and principle_count > 1:
            return 'principle_query'
        elif procedure_count > max(symptom_count, principle_count) and procedure_count > 1:
            return 'procedure_query'
        else:
            return 'general'


# 使用示例
if __name__ == "__main__":
    # 这里演示如何集成到现有系统
    print("Hybrid Retriever system ready for enterprise deployment")