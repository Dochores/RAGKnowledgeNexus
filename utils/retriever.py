from typing import List, Tuple, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class AdvancedRetriever:
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        """
        高级检索器
        :param model_name: 向量模型名称
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
    
    def add_documents(self, chunks: List[str], metadata: List[Dict] = None):
        """
        添加文档到检索系统
        """
        self.chunks = chunks
        self.chunk_metadata = metadata if metadata else [{} for _ in chunks]
        
        # 生成嵌入
        embeddings = self.model.encode(chunks, normalize_embeddings=True)
        
        # 创建FAISS索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
    
    def hybrid_search(self, query: str, k: int = 3, 
                     use_reranking: bool = True,
                     use_metadata: bool = True) -> List[Tuple[str, float, Dict]]:
        """
        混合检索策略
        1. 向量相似度检索
        2. 可选的重排序
        3. 元数据增强
        """
        # 生成查询向量
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # 向量检索
        distances, indices = self.index.search(query_embedding, k * 2)  # 检索更多候选
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                metadata = self.chunk_metadata[idx]
                score = 1.0 / (1.0 + dist)  # 转换距离为相似度分数
                
                if use_metadata:
                    # 使用元数据增强相关性分数
                    metadata_score = self._calculate_metadata_score(query, metadata)
                    score = 0.7 * score + 0.3 * metadata_score
                
                results.append((chunk, score, metadata))
        
        if use_reranking:
            results = self._rerank_results(query, results)
        
        # 返回前k个结果
        return sorted(results, key=lambda x: x[1], reverse=True)[:k]
    
    def _calculate_metadata_score(self, query: str, metadata: Dict) -> float:
        """
        基于元数据计算相关性分数
        """
        score = 0.0
        if 'keywords' in metadata:
            # 检查查询词与关键词的重叠
            query_words = set(query.lower().split())
            keyword_overlap = len(query_words.intersection(metadata['keywords']))
            score += 0.1 * keyword_overlap
        
        return min(1.0, score)  # 归一化分数
    
    def _rerank_results(self, query: str, results: List[Tuple[str, float, Dict]]) -> List[Tuple[str, float, Dict]]:
        """
        重排序结果
        使用更复杂的相关性计算
        """
        reranked = []
        for chunk, score, metadata in results:
            # 考虑文本长度的惩罚项
            length_penalty = min(1.0, 500 / max(len(chunk), 100))
            
            # 考虑段落数量的奖励项
            para_bonus = min(0.2, 0.05 * metadata.get('paragraphs', 1))
            
            # 调整最终分数
            final_score = score * length_penalty + para_bonus
            
            reranked.append((chunk, final_score, metadata))
        
        return sorted(reranked, key=lambda x: x[1], reverse=True) 