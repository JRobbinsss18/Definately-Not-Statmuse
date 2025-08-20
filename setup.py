import subprocess
import sys
import os
def install_requirements():
    print("[INSTALLING] Python requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
def check_ollama():
    print("\n[CHECKING] Ollama installation...")
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("[SUCCESS] Ollama is installed and working")
            return True
        else:
            print("[ERROR] Ollama is not installed or not working properly")
            return False
    except FileNotFoundError:
        print("[ERROR] Ollama is not installed on this system")
        return False
def setup_ollama_model():
    print("\n[SETUP] Setting up Ollama model...")
    try:
        print("[DOWNLOADING] Pulling llama2 model - this might take a few minutes...")
        subprocess.check_call(["ollama", "pull", "llama2"])
        print("[SUCCESS] llama2 model is ready to use")
        return True
    except subprocess.CalledProcessError:
        print("[ERROR] Failed to download llama2 model")
        return False
def main():
    print("NBA Sports Muse Setup")
    print("=" * 40)
    try:
        install_requirements()
        print("[SUCCESS] Python requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install requirements: {e}")
        return False
    if not check_ollama():
        print("\n[INSTRUCTIONS] Ollama Setup Required:")
        print("1. Install Ollama from: https://ollama.ai/")
        print("2. Run 'ollama pull llama2' to download the model")
        print("3. Run this setup script again")
        return False
    if not setup_ollama_model():
        print("[WARNING] Please manually run: ollama pull llama2")
        return False
    print("\n[COMPLETE] Setup finished successfully!")
    print("\n[INFO] To run the dashboard:")
    print("streamlit run main.py")
    return True
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)