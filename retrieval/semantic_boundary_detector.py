"""
语义边界检测器 - 医学领域专用
用于优化父子Chunk的分割边界
"""

import re
from typing import List, Tuple, Dict, Any, Optional
import json

class MedicalBoundaryDetector:
    """医学文本语义边界检测器"""
    
    def __init__(self):
        # 医学术语边界词典
        self.boundary_patterns = {
            # 时间阶段边界
            "time_phase": [
                r'术后第(\d+)天', r'术后第(\d+)周', r'术后第(\d+)月',
                r'入院第(\d+)天', r'住院第(\d+)天',
                r'急性期', r'亚急性期', r'恢复期', r'康复期'
            ],
            # 临床决策边界
            "clinical_decision": [
                r'诊断依据', r'鉴别诊断', r'治疗原则', r'推荐方案',
                r'注意事项', r'禁忌症', r'适应症', r'随访计划',
                r'出院标准', r'复查时间'
            ],
            # 症状描述边界
            "symptom": [
                r'主诉', r'现病史', r'既往史', r'家族史',
                r'体格检查', r'辅助检查', r'实验室检查', r'影像学检查'
            ],
            # 手术操作边界
            "surgical_procedure": [
                r'手术步骤', r'操作流程', r'技术要点',
                r'第一步', r'第二步', r'第三步', r'首先', r'然后', r'最后',
                r'术中', r'术后', r'麻醉后', r'切开后'
            ]
        }
        
        # 关键医学实体（用于保持完整性）
        self.critical_entities = [
            '手术名称', '药物名称', '剂量', '时间', '部位',
            'ICD-10编码', 'SNOMED CT', '指南编号'
        ]
    
    def detect_boundaries(self, text: str) -> List[int]:
        """检测文本中的语义边界位置"""
        boundaries = set()
        
        # 检测各类型边界
        for boundary_type, patterns in self.boundary_patterns.items():
            for pattern in patterns:
                try:
                    matches = list(re.finditer(pattern, text))
                    for match in matches:
                        boundaries.add(match.start())
                        boundaries.add(match.end())
                except re.error as e:
                    continue
        
        # 添加段落边界
        paragraph_breaks = [m.start() for m in re.finditer(r'\n\s*\n', text)]
        boundaries.update(paragraph_breaks)
        
        # 排序并返回
        return sorted(list(boundaries))
    
    def is_medical_entity_boundary(self, text: str, position: int) -> bool:
        """检查位置是否为医学实体边界"""
        # 检查前后是否有重要医学实体
        context_before = text[max(0, position-20):position]
        context_after = text[position:min(len(text), position+20)]
        
        entity_indicators = ['mg', 'g', 'ml', 'u', 'IU', '次', '天', '周', '月',
                           '左侧', '右侧', '上肢', '下肢', '胸腔', '腹腔']
        
        for indicator in entity_indicators:
            if indicator in context_before or indicator in context_after:
                return True
        
        return False


class DynamicChunkSizer:
    """动态分块尺寸调整器"""
    
    def __init__(self):
        # 配置参数
        self.config = {
            'min_chunk_size': 300,      # 最小块大小（字符数）
            'max_chunk_size': 2000,     # 最大块大小
            'optimal_density': 0.8,     # 理想信息密度
            'density_threshold': 0.6,   # 密度阈值
            'medical_weight': 0.7       # 医学内容权重
        }
    
    def calculate_optimal_size(self, 
                             text: str, 
                             metadata: str,
                             content_type: str = 'general') -> Tuple[int, int]:
        """
        计算最优分块尺寸
        企业级算法：基于信息密度 + 内容类型 + 临床重要性
        """
        # 1. 计算基础信息密度
        words = len(text.split())
        chars = len(text)
        density = words / max(1, chars / 5)  # 中文平均5字/词
        
        # 2. 根据内容类型调整
        type_weights = {
            'surgical_procedure': 1.2,    # 手术步骤：需要更大块保持完整性
            'rehab_principle': 0.8,       # 康复原则：可以较小块
            'contraindication': 0.5,      # 禁忌症：必须单独成块
            'diagnosis': 1.0,             # 诊断：中等大小
            'general': 1.0                # 默认
        }
        
        weight = type_weights.get(content_type, 1.0)
        
        # 3. 根据临床重要性调整
        clinical_importance = self._assess_clinical_importance(metadata)
        
        # 4. 计算最终尺寸
        base_size = self.config['optimal_density'] * self.config['max_chunk_size']
        
        # 密度调整
        if density > self.config['density_threshold']:
            # 高密度内容：增大块大小以保持上下文
            size_factor = 1.0 + (density - self.config['density_threshold']) * 2
        else:
            # 低密度内容：减小块大小提高精度
            size_factor = 1.0 - (self.config['density_threshold'] - density) * 1.5
        
        # 综合调整
        optimal_size = int(base_size * weight * size_factor * 
                         (1 + clinical_importance * self.config['medical_weight']))
        
        # 约束范围
        optimal_size = max(self.config['min_chunk_size'], 
                          min(self.config['max_chunk_size'], optimal_size))
        
        # 子块大小（父块的20-30%）
        child_size = int(optimal_size * 0.25)
        child_size = max(200, min(600, child_size))  # 限制子块范围
        
        return optimal_size, child_size
    
    def _assess_clinical_importance(self, metadata: str) -> float:
        """评估临床重要性"""
        importance = 0.0
        
        # 关键字段存在性
        if metadata.get('diagnosis'):
            importance += 0.3
        if metadata.get('surgery_date'):
            importance += 0.2
        if metadata.get('contraindications'):
            importance += 0.4
        if metadata.get('icd10_code'):
            importance += 0.1
        
        # 内容类型
        content_type = metadata.get('category', '')
        if 'surgical' in content_type.lower() or 'operation' in content_type.lower():
            importance += 0.3
        elif 'rehab' in content_type.lower() or 'recovery' in content_type.lower():
            importance += 0.2
            
        return min(1.0, importance)


class SemanticChunker:
    """语义分块器主类"""
    
    def __init__(self):
        self.boundary_detector = MedicalBoundaryDetector()
        self.chunk_sizer = DynamicChunkSizer()
    
    def chunk_document(self, 
                      document: Dict[str, Any],
                      force_parent_size: Optional[int] = None,
                      force_child_size: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        企业级语义分块主方法
        返回：(parent_chunks, child_chunks)
        """
        text = document.get('content', '')
        metadata = document.get('metadata', {})
        category = document.get('category', '')
        if not text.strip():
            return [], []
        
        # 1. 检测语义边界
        boundaries = self.boundary_detector.detect_boundaries(text)
        
        # 2. 确定分块策略
        content_type = self._infer_content_type(category, text)
        parent_size, child_size = self.chunk_sizer.calculate_optimal_size(
            text, metadata, content_type
        )
        
        if force_parent_size:
            parent_size = force_parent_size
        if force_child_size:
            child_size = force_child_size
        
        # 3. 执行分块
        parent_chunks = self._semantic_split(text, boundaries, parent_size, metadata)
        child_chunks = []
        
        # 4. 为每个父块生成子块
        for i, parent_chunk in enumerate(parent_chunks):
            parent_metadata = parent_chunk['metadata'].copy()
            parent_metadata.update({
                'parent_index': i,
                'total_parents': len(parent_chunks),
                'chunk_strategy': 'semantic_dynamic'
            })
            
            # 生成子块
            sub_child_chunks = self._create_child_chunks(
                parent_chunk, child_size, parent_metadata
            )
            child_chunks.extend(sub_child_chunks)
        
        return parent_chunks, child_chunks
    
    def _infer_content_type(self, metadata: Dict[str, Any], text: str) -> str:
        """推断内容类型"""
        # 基于元数据

        cat = metadata['category'].lower()
        if 'surgical' in cat or '手术' in cat:
            return 'surgical_procedure'
        elif 'rehab' in cat or '康复' in cat:
            return 'rehab_principle'
        elif 'contraindication' in cat or '禁忌' in cat:
            return 'contraindication'
        
        # 基于文本内容
        text_lower = text.lower()
        if any(term in text_lower for term in ['手术步骤', '操作流程', '技术要点']):
            return 'surgical_procedure'
        elif any(term in text_lower for term in ['康复计划', '注意事项', '禁忌症']):
            return 'rehab_principle'
        elif any(term in text_lower for term in ['禁忌', '禁用', '避免']):
            return 'contraindication'
        
        return 'general'
    
    def _semantic_split(self, 
                       text: str, 
                       boundaries: List[int], 
                       chunk_size: int,
                       metadata: Dict[str, Any]) -> List[Dict]:
        """语义分割核心算法"""
        chunks = []
        start = 0
        
        # 添加文档开始和结束边界
        all_boundaries = [0] + boundaries + [len(text)]
        
        # 合并相邻边界（距离太近的合并）
        merged_boundaries = self._merge_close_boundaries(all_boundaries, min_distance=50)
        
        # 创建块
        for i in range(len(merged_boundaries) - 1):
            start_pos = merged_boundaries[i]
            end_pos = merged_boundaries[i + 1]
            
            # 如果块太大，进一步分割
            if end_pos - start_pos > chunk_size * 1.5:
                # 使用递归分割
                sub_chunks = self._recursive_split(text[start_pos:end_pos], chunk_size)
                for sub_chunk in sub_chunks:
                    chunks.append({
                        'page_content': sub_chunk,
                        'metadata': metadata.copy(),
                        'chunk_id': f"parent_{len(chunks)}",
                        'source_span': (start_pos, start_pos + len(sub_chunk)),
                        'chunk_type': 'semantic'
                    })
            else:
                chunk_content = text[start_pos:end_pos].strip()
                if chunk_content:
                    chunks.append({
                        'page_content': chunk_content,
                        'metadata': metadata.copy(),
                        'chunk_id': f"parent_{len(chunks)}",
                        'source_span': (start_pos, end_pos),
                        'chunk_type': 'semantic'
                    })
        
        # 后处理：合并过小的块
        chunks = self._merge_small_chunks(chunks, min_size=300)
        
        return chunks
    
    def _merge_close_boundaries(self, boundaries: List[int], min_distance: int) -> List[int]:
        """合并距离过近的边界"""
        if not boundaries:
            return boundaries
        
        merged = [boundaries[0]]
        for boundary in boundaries[1:]:
            if boundary - merged[-1] < min_distance:
                continue
            merged.append(boundary)
        
        return merged
    
    def _recursive_split(self, text: str, max_size: int) -> List[str]:
        """递归分割大块"""
        if len(text) <= max_size:
            return [text]
        
        # 在中间位置找最佳分割点
        mid = len(text) // 2
        best_split = mid
        
        # 向前搜索最近的边界
        for i in range(mid, max(0, mid-100), -1):
            if text[i] in ['\n', '。', '；', '：']:
                best_split = i
                break
        
        # 向后搜索
        for i in range(mid, min(len(text), mid+100)):
            if text[i] in ['\n', '。', '；', '：']:
                best_split = i
                break
        
        left = text[:best_split].strip()
        right = text[best_split:].strip()
        
        result = []
        if left:
            result.extend(self._recursive_split(left, max_size))
        if right:
            result.extend(self._recursive_split(right, max_size))
        
        return result
    
    def _merge_small_chunks(self, chunks: List[Dict], min_size: int) -> List[Dict]:
        """合并过小的块"""
        if not chunks:
            return chunks
        
        merged = [chunks[0]]
        
        for chunk in chunks[1:]:
            last_chunk = merged[-1]
            combined_length = len(last_chunk['page_content']) + len(chunk['page_content'])
            
            if combined_length < min_size:
                # 合并
                merged[-1]['page_content'] += '\n\n' + chunk['page_content']
                # 合并元数据
                for key, value in chunk['metadata'].items():
                    if key not in ['chunk_id', 'source_span']:
                        if key in last_chunk['metadata'] and last_chunk['metadata'][key] != value:
                            last_chunk['metadata'][key] = f"{last_chunk['metadata'][key]} | {value}"
                        elif key not in last_chunk['metadata']:
                            last_chunk['metadata'][key] = value
                # 更新source_span
                merged[-1]['source_span'] = (
                    last_chunk['source_span'][0],
                    chunk['source_span'][1]
                )
            else:
                merged.append(chunk)
        
        return merged
    
    def _create_child_chunks(self, 
                           parent_chunk: Dict, 
                           child_size: int,
                           parent_metadata: Dict) -> List[Dict]:
        """创建子块"""
        text = parent_chunk['page_content']
        chunks = []
        
        # 简单固定大小分块（可替换为更复杂的语义分块）
        start = 0
        while start < len(text):
            end = min(start + child_size, len(text))
            
            # 确保在句子边界结束
            while end < len(text) and text[end] not in ['\n', '。', '！', '？', '；', '：']:
                end += 1
            
            child_content = text[start:end].strip()
            if child_content:
                chunks.append({
                    'page_content': child_content,
                    'metadata': {
                        **parent_metadata,
                        'child_index': len(chunks),
                        'total_children': 0,  # 后续更新
                        'parent_id': parent_chunk.get('chunk_id'),
                        'chunk_type': 'child_semantic'
                    },
                    'source_span': (start, end)
                })
            start = end + 1
        
        # 更新总子块数量
        for chunk in chunks:
            chunk['metadata']['total_children'] = len(chunks)
        
        return chunks