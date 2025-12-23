import os
from langchain_openai import ChatOpenAI
from ..utils.logger import log_info, Colors

# Configuration
# Use OLLAMA_HOST env var, default to localhost for local dev
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LOCAL_LLM_URL = f"{ollama_host}/v1"
MODEL_NAME = "llama3.2"

class LLM:
    def __init__(self):
        self.llm = None
        self._setup()

    def _setup(self):
        """Initialize the LLM (Local or OpenAI)."""
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            log_info(f"Connecting to Local LLM at {LOCAL_LLM_URL}...", Colors.YELLOW)
            self.llm = ChatOpenAI(
                base_url=LOCAL_LLM_URL,
                api_key="ollama",
                model=MODEL_NAME,
                temperature=0.0,
                tiktoken_model_name="gpt-3.5-turbo" 
                # It is only using the GPT dictionary for counting tokens. Just tokenizer
                # llama is the chef, token is the scale
            )
        else:
            log_info("Using Real OpenAI API...", Colors.GREEN)
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

    async def generate_response(self, messages, config=None):
        if not self.llm:
            raise ValueError("LLM not initialized")
        return await self.llm.ainvoke(messages, config=config)
