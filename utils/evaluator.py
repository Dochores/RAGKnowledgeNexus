from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer


class RAGEvaluator:
    def __init__(self):
        """
        RAG系统评估器
        评估检索质量和回答质量
        """
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def evaluate_retrieval(self,
                         query: str,
                         retrieved_chunks: List[Tuple[str, float, Dict]],
                         ground_truth: str) -> Dict:
        """
        评估检索质量
        1. 相关性评分
        2. 召回率
        3. 检索准确性
        """
        results = {}
        
        # 计算相关性分数
        relevance_scores = [score for _, score, _ in retrieved_chunks]
        results['mean_relevance'] = np.mean(relevance_scores)
        results['max_relevance'] = np.max(relevance_scores)
        
        # 计算召回率（需要ground truth中的关键信息是否被检索）
        retrieved_text = ' '.join([chunk for chunk, _, _ in retrieved_chunks])
        recall_score = self._calculate_recall(retrieved_text, ground_truth)
        results['recall'] = recall_score
        
        # 计算检索准确性
        accuracy = self._calculate_retrieval_accuracy(retrieved_text, ground_truth)
        results['accuracy'] = accuracy
        
        return results
    
    def evaluate_answer(self,
                       generated_answer: str,
                       ground_truth: str,
                       context: str) -> Dict:
        """
        评估回答质量
        1. 答案相关性
        2. 事实准确性
        3. 上下文覆盖度
        """
        results = {}
        
        # 计算ROUGE分数
        rouge_scores = self.rouge_scorer.score(ground_truth, generated_answer)
        results['rouge1'] = rouge_scores['rouge1'].fmeasure
        results['rouge2'] = rouge_scores['rouge2'].fmeasure
        results['rougeL'] = rouge_scores['rougeL'].fmeasure
        
        # 计算答案相关性
        answer_relevance = self._calculate_answer_relevance(generated_answer, ground_truth)
        results['answer_relevance'] = answer_relevance
        
        # 计算上下文覆盖度
        context_coverage = self._calculate_context_coverage(generated_answer, context)
        results['context_coverage'] = context_coverage
        
        return results
    
    def _calculate_recall(self, retrieved_text: str, ground_truth: str) -> float:
        """
        计算召回率
        基于关键词匹配
        """
        ground_truth_words = set(ground_truth.lower().split())
        retrieved_words = set(retrieved_text.lower().split())
        
        if not ground_truth_words:
            return 0.0
        
        matched_words = ground_truth_words.intersection(retrieved_words)
        return len(matched_words) / len(ground_truth_words)
    
    def _calculate_retrieval_accuracy(self, retrieved_text: str, ground_truth: str) -> float:
        """
        计算检索准确性
        使用余弦相似度
        """
        # 简单的词袋表示
        def get_bow(text):
            words = text.lower().split()
            bow = {}
            for word in words:
                bow[word] = bow.get(word, 0) + 1
            return bow
        
        retrieved_bow = get_bow(retrieved_text)
        ground_truth_bow = get_bow(ground_truth)
        
        # 构建向量
        all_words = list(set(retrieved_bow.keys()) | set(ground_truth_bow.keys()))
        retrieved_vector = [retrieved_bow.get(word, 0) for word in all_words]
        ground_truth_vector = [ground_truth_bow.get(word, 0) for word in all_words]
        
        # 计算余弦相似度
        dot_product = sum(a * b for a, b in zip(retrieved_vector, ground_truth_vector))
        retrieved_norm = np.sqrt(sum(a * a for a in retrieved_vector))
        ground_truth_norm = np.sqrt(sum(b * b for b in ground_truth_vector))
        
        if retrieved_norm == 0 or ground_truth_norm == 0:
            return 0.0
        
        return dot_product / (retrieved_norm * ground_truth_norm)
    
    def _calculate_answer_relevance(self, answer: str, ground_truth: str) -> float:
        """
        计算答案相关性
        使用ROUGE-L分数
        """
        rouge_scores = self.rouge_scorer.score(ground_truth, answer)
        return rouge_scores['rougeL'].fmeasure
    
    def _calculate_context_coverage(self, answer: str, context: str) -> float:
        """
        计算上下文覆盖度
        检查答案中的关键信息是否来自上下文
        """
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        if not answer_words:
            return 0.0
        
        covered_words = answer_words.intersection(context_words)
        return len(covered_words) / len(answer_words)
    
    def generate_evaluation_report(self,
                                 retrieval_metrics: Dict,
                                 answer_metrics: Dict) -> str:
        """
        生成评估报告
        """
        report = "RAG系统评估报告\n"
        report += "=" * 50 + "\n\n"
        
        # 检索质量报告
        report += "1. 检索质量评估\n"
        report += "-" * 30 + "\n"
        report += f"平均相关性得分: {retrieval_metrics['mean_relevance']:.3f}\n"
        report += f"最高相关性得分: {retrieval_metrics['max_relevance']:.3f}\n"
        report += f"召回率: {retrieval_metrics['recall']:.3f}\n"
        report += f"检索准确性: {retrieval_metrics['accuracy']:.3f}\n\n"
        
        # 回答质量报告
        report += "2. 回答质量评估\n"
        report += "-" * 30 + "\n"
        report += f"ROUGE-1 F1: {answer_metrics['rouge1']:.3f}\n"
        report += f"ROUGE-2 F1: {answer_metrics['rouge2']:.3f}\n"
        report += f"ROUGE-L F1: {answer_metrics['rougeL']:.3f}\n"
        report += f"答案相关性: {answer_metrics['answer_relevance']:.3f}\n"
        report += f"上下文覆盖度: {answer_metrics['context_coverage']:.3f}\n\n"
        
        # 总体评估
        avg_retrieval = np.mean([
            retrieval_metrics['mean_relevance'],
            retrieval_metrics['recall'],
            retrieval_metrics['accuracy']
        ])
        avg_answer = np.mean([
            answer_metrics['rouge1'],
            answer_metrics['rouge2'],
            answer_metrics['rougeL'],
            answer_metrics['answer_relevance'],
            answer_metrics['context_coverage']
        ])
        
        report += "3. 总体评估\n"
        report += "-" * 30 + "\n"
        report += f"检索质量综合得分: {avg_retrieval:.3f}\n"
        report += f"回答质量综合得分: {avg_answer:.3f}\n"
        report += f"系统总体得分: {(avg_retrieval + avg_answer) / 2:.3f}\n"
        
        return report 