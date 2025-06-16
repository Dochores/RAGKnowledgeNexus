from typing import List, Dict, Optional


class PromptBuilder:
    def __init__(self):
        """
        提示词构建器
        实现高级提示词工程策略
        """
        self.system_prompt = self._get_default_system_prompt()
    
    def build_prompt(self,
                    query: str,
                    context: str,
                    chat_history: Optional[List[Dict]] = None) -> List[Dict]:
        """
        构建优化的提示词
        1. 任务分解
        2. 思维链引导
        3. 自我验证机制
        """
        messages = []
        
        # 添加系统提示词
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # 添加对话历史（如果有）
        if chat_history:
            for msg in chat_history[-3:]:  # 只保留最近3轮
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # 构建当前查询的提示词
        query_prompt = self._build_query_prompt(query, context)
        messages.append({
            "role": "user",
            "content": query_prompt
        })
        
        return messages
    
    def _get_default_system_prompt(self) -> str:
        """
        默认系统提示词
        设定回答的基本原则和框架
        """
        return """你是一个专业的知识助手，请遵循以下原则回答问题：

1. 准确性：只基于提供的上下文信息回答，不要添加未经验证的信息
2. 完整性：确保回答涵盖问题的所有方面
3. 逻辑性：采用结构化的方式组织回答
4. 透明度：如果上下文信息不足，请明确指出

回答时请遵循以下步骤：
1. 仔细分析问题要点
2. 从上下文中提取相关信息
3. 组织信息并形成逻辑连贯的回答
4. 确保回答准确且有依据"""
    
    def _build_query_prompt(self, query: str, context: str) -> str:
        """
        构建查询提示词
        添加思维链引导和自我验证要求
        """
        return f"""请基于以下信息回答问题：

{context}

问题：{query}

请按照以下步骤思考并回答：

1. 问题分析：
- 明确问题的核心要点
- 确定需要从上下文中获取的关键信息

2. 信息提取：
- 从上下文中识别相关信息
- 确认信息的完整性和可靠性

3. 答案构建：
- 组织提取的信息
- 形成逻辑清晰的回答

4. 自我验证：
- 检查回答是否完全基于上下文信息
- 验证是否完整回答了问题
- 确认回答的逻辑性和准确性

请开始你的回答..."""
    
    def build_followup_prompt(self, 
                            original_query: str,
                            followup_query: str,
                            context: str) -> str:
        """
        构建后续问题的提示词
        保持上下文连贯性
        """
        return f"""基于之前的问题"{original_query}"和新的问题，请回答：

上下文信息：
{context}

新问题：{followup_query}

请确保：
1. 考虑前后问题的联系
2. 保持回答的连贯性
3. 必要时参考之前的回答""" 