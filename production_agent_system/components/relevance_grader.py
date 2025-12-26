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
        
        self.chain = (self.prompt | self.llm | StrOutputParser()).with_config(run_name="DocumentRelevanceGrader")

        # --- History Relevance Grader ---
        history_system = """You are a grader assessing if a new user question is related to the previous conversation context.
        
        You will be given the Chat History (which may include a summary of older messages) and the Current Question.
        
        If the Current Question is a follow-up, refers to entities in the history, or stays on the same topic, say 'yes'.
        If the Current Question starts a completely new, unrelated topic, say 'no'.
        Output ONLY 'yes' or 'no'."""

        self.history_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", history_system),
                ("human", "Chat History:\n{chat_history}\n\nCurrent Question: {current_question}"),
            ]
        )
        self.history_chain = (self.history_prompt | self.llm | StrOutputParser()).with_config(run_name="TopicSwitchGrader")

        # --- Profile Relevance Grader ---
        profile_system = """You are a grader assessing if a user's stored profile information is relevant to their current question.
        
        If the User Profile contains information (preferences, projects, expertise) that is directly related to the Current Question, say 'yes'.
        If the profile is irrelevant to the question, say 'no'.
        Output ONLY 'yes' or 'no'."""

        self.profile_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", profile_system),
                ("human", "User Profile: {profile} \n\n Current Question: {question}"),
            ]
        )
        self.profile_chain = (self.profile_prompt | self.llm | StrOutputParser()).with_config(run_name="LTMProfileGrader")

    async def grade(self, question: str, context: str, config=None) -> str:
        try:
            score = await self.chain.ainvoke({"question": question, "document": context}, config=config)
            score = score.strip().lower()
            log_info(f"Relevance Grade: {score}")
            return score
        except Exception as e:
            log_info(f"Grading failed: {e}. Defaulting to 'yes'")
            return "yes" 

    async def grade_history(self, current_question: str, chat_history: str, config=None) -> str:
        try:
            score = await self.history_chain.ainvoke(
                {"current_question": current_question, "chat_history": chat_history}, 
                config=config
            )
            score = score.strip().lower()
            log_info(f"History Relevance Grade: {score}")
            return score
        except Exception as e:
            log_info(f"History Grading failed: {e}. Defaulting to 'yes'")
            return "yes" 

    async def grade_profile(self, question: str, profile: str, config=None) -> str:
        try:
            score = await self.profile_chain.ainvoke({"question": question, "profile": profile}, config=config)
            score = score.strip().lower()
            log_info(f"Profile Relevance Grade: {score}")
            return score
        except Exception as e:
            log_info(f"Profile grading failed: {e}. Defaulting to 'no'")
            return "no"
