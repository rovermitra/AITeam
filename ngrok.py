#!/usr/bin/env python3
"""
RoverMitra Server & ngrok Tunnel Starter (Robust Version)
---------------------------------------------------------
This script automates running the local server and exposing it via ngrok.
It now includes a function to free up the required port if it's already in use.

Instructions:
1. Make sure 'local_llama_server.py' is in the same directory.
2. Add NGROK_AUTHTOKEN and a valid DATABASE_URL to your .env file.
3. Run this script from your terminal: python run_with_ngrok.py
"""
import os
import uvicorn
from dotenv import load_dotenv

# --- 1. Load Environment Variables ---
print("🔌 Loading environment variables from .env file...")
load_dotenv()
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")
PORT = 8002  # The port your FastAPI app runs on

# --- 2. Port Management ---
# This new section ensures the port is free before we start.
try:
    import psutil
    def free_port(port: int):
        print(f"🔎 Checking if port {port} is in use...")
        # Iterate over all running processes
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                # For each process, check its connections
                for conn in proc.connections(kind='inet'):
                    # Check if the process is listening on the target port
                    if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                        print(f"⚠️ Port {port} is in use by process '{proc.name()}' (PID: {proc.pid}).")
                        print(f"   Terminating process to free up the port.")
                        proc.terminate()
                        proc.wait() # Wait for the process to terminate
                        print(f"✅ Port {port} has been freed.")
                        return
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # This can happen if the process terminates while we are inspecting it
                continue
        print(f"✅ Port {port} is already free.")
    free_port(PORT)
except ImportError:
    print("⚠️ 'psutil' library not found. Cannot automatically free up ports.")
    print("   If you encounter a port-in-use error, you may need to stop the process manually.")
    print("   To enable this feature, run: pip install psutil")

# --- 3. Check for ngrok and Authenticate ---
try:
    from pyngrok import ngrok
except ImportError:
    print("❌ 'pyngrok' library not found.")
    print("   Please install it by running: pip install pyngrok")
    exit(1)

if not NGROK_AUTHTOKEN:
    print("❌ ERROR: NGROK_AUTHTOKEN not found in your .env file.")
    exit(1)

# --- 4. Start the ngrok Tunnel ---
try:
    print(f"🔗 Authenticating and starting ngrok tunnel for port {PORT}...")
    ngrok.set_auth_token(NGROK_AUTHTOKEN)
    public_url = ngrok.connect(PORT).public_url
    print("✅ ngrok tunnel is active!")
    print(f"🌍 Your Public URL is: {public_url}")
    print("   Use this URL in your website's front-end configuration.")

except Exception as e:
    print(f"❌ Failed to start ngrok tunnel: {e}")
    exit(1)

# --- 5. Start the Uvicorn Server ---
if __name__ == "__main__":
    print(f"🚀 Starting FastAPI server on http://localhost:{PORT}...")
    uvicorn.run(
        "local_llama_server:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )

