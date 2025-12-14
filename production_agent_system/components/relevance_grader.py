from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm import LLM
from ..utils.logger import log_info

class RelevanceGrader:
    def __init__(self, llm_service: LLM):
        self.llm = llm_service.llm
        
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
            Output ONLY 'yes' or 'no'."""
        
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )
        
        self.chain = self.prompt | self.llm | StrOutputParser()

        # --- History Relevance Grader ---
        history_system = """You are a grader assessing if a new user question is related to the previous conversation topic.
        
        If the Current Question is a follow-up, refers to entities in the Previous Question, or stays on the same topic, say 'yes'.
        If the Current Question starts a completely new, unrelated topic, say 'no'.
        Output ONLY 'yes' or 'no'."""

        self.history_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", history_system),
                ("human", "Previous Question: {last_question} \n\n Current Question: {current_question}"),
            ]
        )
        self.history_chain = self.history_prompt | self.llm | StrOutputParser()

    async def grade(self, question: str, context: str, config=None) -> str:
        try:
            score = await self.chain.ainvoke({"question": question, "document": context}, config=config)
            score = score.strip().lower()
            log_info(f"Relevance Grade: {score}")
            return score
        except Exception as e:
            log_info(f"Grading failed: {e}. Defaulting to 'yes'")
            return "yes" 

    async def grade_history(self, current_question: str, last_question: str, config=None) -> str:
        try:
            score = await self.history_chain.ainvoke(
                {"current_question": current_question, "last_question": last_question}, 
                config=config
            )
            score = score.strip().lower()
            log_info(f"History Relevance Grade: {score}")
            return score
        except Exception as e:
            log_info(f"History Grading failed: {e}. Defaulting to 'yes'")
            return "yes" 
