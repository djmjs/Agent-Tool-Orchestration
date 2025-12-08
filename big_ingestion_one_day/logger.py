from enum import Enum

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    PURPLE = '\033[95m'
    ENDC = '\033[0m'

def log_info(message, color=Colors.CYAN):
    print(f"{color}[INFO] {message}{Colors.ENDC}")

def log_success(message):
    print(f"{Colors.GREEN}[SUCCESS] {message}{Colors.ENDC}")

def log_warning(message):
    print(f"{Colors.YELLOW}[WARNING] {message}{Colors.ENDC}")

def log_error(message):
    print(f"{Colors.RED}[ERROR] {message}{Colors.ENDC}")

def log_header(message):
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== {message} ==={Colors.ENDC}")
