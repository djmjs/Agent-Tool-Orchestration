import requests
import json
import time

BASE_URL = "http://localhost:8000"

def run_query(query, test_name):
    print(f"\n{'='*20} Test: {test_name} {'='*20}")
    print(f"Query: {query}")
    
    start_time = time.time()
    try:
        response = requests.post(f"{BASE_URL}/query", json={"query": query})
        response.raise_for_status()
        data = response.json()
        duration = time.time() - start_time
        
        print(f"Time: {duration:.2f}s")
        print(f"Answer: {data.get('answer')}")
        
        sources = data.get('sources', [])
        if sources:
            print("Sources:")
            for s in sources:
                print(f" - {s.get('title')} ({s.get('url')})")
        else:
            print("Sources: None (Direct Answer or General Chat)")
            
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None

def check_profile():
    print(f"\n{'='*20} Checking LTM Profile {'='*20}")
    try:
        # Assuming default_user based on code
        response = requests.get(f"{BASE_URL}/profiles/default_user") 
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
        else:
            print("Profile not found or empty.")
    except Exception as e:
        print(f"Error fetching profile: {e}")

def main():
    print("Starting Comprehensive Agent System Test...")
    print("Ensure the agent system is running at http://localhost:8000")

    # 1. Direct Answer (Router: DIRECT_ANSWER)
    # Expectation: Fast response, no sources.
    run_query("Hi, how are you?", "1. Direct Answer")

    # 2. LTM Write (Router: GENERAL/TOOL -> Extraction)
    # Expectation: Agent acknowledges. Background task updates Postgres.
    run_query("My name is Alex and I am a Senior Python Developer working on AI agents.", "2. LTM Write (Profile Extraction)")

    # 3. LTM Read (Router: DIRECT_ANSWER or GENERAL + Context Injection)
    # Expectation: Agent knows name is Alex and job is Python Dev.
    run_query("What is my name and what do I do?", "3. LTM Read (Context Injection)")

    # 4. Vector DB - Technical (Router: DATABASE -> Retrieve -> Grade: Yes)
    # Expectation: Sources from Qdrant (local docs).
    run_query("What is Retrieval Augmented Generation (RAG)?", "4. Vector DB Retrieval")

    # 5. STM Context (Router: GENERAL/DATABASE + Reformulation)
    # Expectation: Reformulates "its" to "RAG's". Returns relevant info.
    run_query("What are its main benefits?", "5. STM Context & Reformulation (Follow-up)")

    # 6. Topic Switch (Router: WEB/GENERAL -> STM Clear)
    # Expectation: Detects switch from RAG to Weather. Ignores RAG context. Uses Web Search.
    run_query("What is the weather in Tokyo right now?", "6. Topic Switch (Web Search)")

    # 7. Web Search - Explicit (Router: WEB)
    # Expectation: Uses Tavily to find news.
    run_query("Search for the latest news about OpenAI.", "7. Explicit Web Search")

    # 8. Retrieval Fallback (Router: DATABASE -> Retrieve -> Grade: No -> Web)
    # Expectation: Tries DB first (technical topic), finds nothing relevant, falls back to Web.
    run_query("Who won the Nobel Prize in Physics in 2024?", "8. Retrieval Fallback to Web (Relevance Grading)")

    # 9. Tool Execution (Router: TOOL)
    # Expectation: Calls update_user_info tool.
    run_query("Update my name to Alexander the Great.", "9. Tool Execution (Explicit Profile Update)")

    # 10. Verify Tool Effect
    # Expectation: Name is now Alexander the Great.
    run_query("Who am I?", "10. Verify Tool Update")
    
    # 11. General Knowledge (Router: GENERAL)
    # Expectation: No sources, just LLM knowledge.
    run_query("Write a hello world function in Python.", "11. General Knowledge")

    # Check actual DB state
    check_profile()

if __name__ == "__main__":
    main()