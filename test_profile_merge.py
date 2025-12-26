import requests
import time
import json

BASE_URL = "http://localhost:8000"

def query_agent(query):
    response = requests.post(f"{BASE_URL}/query", json={"query": query})
    response.raise_for_status()
    return response.json()

def get_profile(user_id="default_user"):
    response = requests.get(f"{BASE_URL}/profiles/{user_id}")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()

def main():
    print("Starting Profile Merge Test...")

    # 1. First Interaction: Mention Python preference
    print("\n--- Interaction 1: Mentioning Python ---")
    q1 = "I am working on a backend project using Python. I prefer clean code."
    print(f"Query: {q1}")
    try:
        query_agent(q1)
    except Exception as e:
        print(f"Query failed: {e}")
        return
    
    # Wait a bit for async extraction
    time.sleep(5)
    
    profile1 = get_profile()
    print("Profile after Interaction 1:")
    print(json.dumps(profile1, indent=2))
    
    if not profile1:
        print("ERROR: Profile not created.")
        # Continue anyway to see if second one works
    
    # 2. Second Interaction: Mention Rust preference
    print("\n--- Interaction 2: Mentioning Rust ---")
    q2 = "I am also starting a new CLI tool using Rust. I need high performance."
    print(f"Query: {q2}")
    try:
        query_agent(q2)
    except Exception as e:
        print(f"Query failed: {e}")
        return
    
    # Wait a bit for async extraction
    time.sleep(5)
    
    profile2 = get_profile()
    print("Profile after Interaction 2:")
    print(json.dumps(profile2, indent=2))

    if not profile2:
        print("ERROR: Profile not found after second interaction.")
        return

    # 3. Verification
    preferences = profile2.get("preferences", [])
    projects = profile2.get("projects", [])
    
    # Flatten lists for searching
    all_text = " ".join(preferences + projects).lower()
    
    has_python = "python" in all_text
    has_rust = "rust" in all_text
    
    if has_python and has_rust:
        print("\nSUCCESS: Both Python and Rust detected in profile!")
    else:
        print("\nFAILURE: Profile missing expected data.")
        print(f"Has Python: {has_python}")
        print(f"Has Rust: {has_rust}")

if __name__ == "__main__":
    main()
