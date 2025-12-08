from .logger import log_warning

class FallbackHandler:
    @staticmethod
    def handle_error(e: Exception) -> str:
        log_warning(f"Fallback triggered due to error: {e}")
        return "I apologize, but I encountered an error while processing your request. Please try again later."

    @staticmethod
    def handle_no_results() -> str:
        return "I couldn't find any relevant information in my knowledge base to answer your question."
