from typing import List, Dict
import re
from nltk.tokenize import sent_tokenize
import nltk

class AdvancedTextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        高级文本分块器
        :param chunk_size: 每个块的目标大小（字符数）
        :param chunk_overlap: 块之间的重叠大小（字符数）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 下载必要的NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def split_text(self, text: str) -> List[str]:
        """
        智能分块策略
        1. 优先按段落分割
        2. 长段落按句子分割
        3. 保持上下文连贯性
        """
        # 按段落分割
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # 如果段落过长，按句子分割
            if len(para) > self.chunk_size:
                sentences = sent_tokenize(para)
                for sent in sentences:
                    if len(current_chunk) + len(sent) <= self.chunk_size:
                        current_chunk += sent + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent + " "
            else:
                if len(current_chunk) + len(para) <= self.chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 添加重叠以保持上下文连贯性
        overlapped_chunks = []
        for i in range(len(chunks)):
            if i > 0:
                # 从前一个块的末尾添加重叠内容
                prev_chunk = chunks[i-1]
                overlap_content = prev_chunk[-self.chunk_overlap:]
                current_chunk = overlap_content + chunks[i]
            else:
                current_chunk = chunks[i]
            overlapped_chunks.append(current_chunk)
        
        return overlapped_chunks

    def get_chunk_metadata(self, chunk: str) -> Dict:
        """
        为每个文本块提取元数据
        """
        return {
            'length': len(chunk),
            'sentences': len(sent_tokenize(chunk)),
            'paragraphs': len([p for p in chunk.split('\n\n') if p.strip()]),
            'keywords': self._extract_keywords(chunk)
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        提取文本块中的关键词
        """
        # 简单的关键词提取策略
        words = re.findall(r'\b\w+\b', text.lower())
        # 过滤停用词和短词
        keywords = [w for w in words if len(w) > 3]
        # 返回出现频率最高的前10个词
        from collections import Counter
        return [word for word, _ in Counter(keywords).most_common(10)] 