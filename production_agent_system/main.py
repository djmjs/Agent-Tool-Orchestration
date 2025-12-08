import uvicorn
import os
import sys

# Add the parent directory (agents_test) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

if __name__ == "__main__":
    print("Starting Production Agent System...")
    # Run from the project root (agents_test)
    uvicorn.run("production_agent_system.frontend.api:app", host="127.0.0.1", port=8000, reload=True, reload_dirs=[current_dir])
