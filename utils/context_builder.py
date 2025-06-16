from typing import List, Tuple, Dict
import numpy as np
from collections import defaultdict


class ContextBuilder:
    def __init__(self, max_tokens: int = 2000):
        """
        上下文构建器
        :param max_tokens: 最大token数量
        """
        self.max_tokens = max_tokens
    
    def build_context(self, 
                     retrieved_chunks: List[Tuple[str, float, Dict]],
                     query: str,
                     chat_history: List[Dict] = None) -> str:
        """
        智能构建上下文
        1. 相关性加权
        2. 信息去重
        3. 历史对话整合
        4. 动态token控制
        """
        # 提取并排序chunks
        chunks_with_scores = [(chunk, score) for chunk, score, _ in retrieved_chunks]
        
        # 去重处理
        unique_chunks = self._remove_duplicates(chunks_with_scores)
        
        # 整合对话历史
        if chat_history:
            history_context = self._integrate_history(chat_history, query)
        else:
            history_context = ""
        
        # 构建最终上下文
        context_parts = []
        current_tokens = self._estimate_tokens(history_context)
        
        # 添加历史上下文
        if history_context:
            context_parts.append(history_context)
        
        # 添加检索内容
        for chunk, score in unique_chunks:
            chunk_tokens = self._estimate_tokens(chunk)
            if current_tokens + chunk_tokens <= self.max_tokens:
                context_parts.append(chunk)
                current_tokens += chunk_tokens
            else:
                break
        
        # 组装最终上下文
        final_context = "\n\n".join(context_parts)
        
        return self._format_context(final_context, query)
    
    def _remove_duplicates(self, 
                         chunks_with_scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        去除重复或高度重叠的内容
        使用简单的文本相似度比较
        """
        unique_chunks = []
        seen_content = set()
        
        for chunk, score in sorted(chunks_with_scores, key=lambda x: x[1], reverse=True):
            # 生成chunk的简化表示（关键词集合）
            chunk_keywords = set(chunk.lower().split())
            
            # 检查是否与已有内容重叠
            is_duplicate = False
            for seen in seen_content:
                seen_keywords = set(seen.lower().split())
                overlap = len(chunk_keywords & seen_keywords) / len(chunk_keywords | seen_keywords)
                if overlap > 0.7:  # 重叠阈值
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append((chunk, score))
                seen_content.add(chunk)
        
        return unique_chunks
    
    def _integrate_history(self, 
                         chat_history: List[Dict],
                         current_query: str) -> str:
        """
        整合对话历史
        只保留最相关的历史对话
        """
        # 选择最近的3轮对话
        recent_history = chat_history[-3:]
        
        # 构建历史对话文本
        history_parts = []
        for turn in recent_history:
            if turn.get("role") == "user":
                history_parts.append(f"用户: {turn['content']}")
            else:
                history_parts.append(f"助手: {turn['content']}")
        
        return "\n".join(history_parts)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        估算文本的token数量
        使用简单的字符数估算
        """
        return len(text) // 2  # 粗略估算中文token数
    
    def _format_context(self, context: str, query: str) -> str:
        """
        格式化上下文，添加提示信息
        """
        return f"""相关上下文信息：

{context}

当前问题：{query}

请基于以上上下文信息，准确回答问题。如果上下文信息不足以回答问题，请明确指出。
""" 