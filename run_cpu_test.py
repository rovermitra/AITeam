#!/usr/bin/env python3
"""
RoverMitra CPU Performance Test Runner

This script helps you run the CPU-only Llama server and then execute the CPU-only client
to measure performance differences.

Usage:
    python run_cpu_test.py

This will:
1. Start the CPU-only Llama server in the background
2. Wait for it to load the model
3. Run the CPU-only client
4. Show performance metrics
"""

import subprocess
import time
import requests
import sys
from pathlib import Path

def check_server_health(port=8000):
    """Check if server is healthy"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("ok", False)
    except Exception:
        pass
    return False

def wait_for_server(max_wait=300):
    """Wait for server to be ready"""
    print("⏳ Waiting for Llama server to load model...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        # Try multiple ports
        for port in [8000, 8001, 8002, 8003, 8004, 8005]:
            if check_server_health(port):
                print(f"✅ Server is ready on port {port}")
                return True
        
        print(".", end="", flush=True)
        time.sleep(2)
    
    print(f"\n❌ Server did not start within {max_wait} seconds")
    return False

def main():
    print("🚀 RoverMitra CPU Performance Test")
    print("=" * 50)
    
    # Check if required files exist
    server_file = Path("serve_llama_CPU.py")
    client_file = Path("main_CPU.py")
    
    if not server_file.exists():
        print(f"❌ Server file not found: {server_file}")
        return
    
    if not client_file.exists():
        print(f"❌ Client file not found: {client_file}")
        return
    
    print("📋 Starting CPU-only Llama server...")
    
    # Start server in background
    try:
        server_process = subprocess.Popen([
            sys.executable, str(server_file)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("🔄 Server process started")
        
        # Wait for server to be ready
        if not wait_for_server():
            print("❌ Failed to start server")
            server_process.terminate()
            return
        
        print("\n🎯 Running CPU-only client...")
        print("=" * 50)
        
        # Run the client
        client_process = subprocess.run([
            sys.executable, str(client_file)
        ])
        
        print("\n" + "=" * 50)
        print("📊 Performance test completed!")
        
        if client_process.returncode == 0:
            print("✅ Client executed successfully")
        else:
            print(f"⚠️  Client exited with code: {client_process.returncode}")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Clean up server process
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
            print("🔄 Server stopped")
        except Exception:
            print("⚠️  Server cleanup failed")
    
    print("\n💡 Tips:")
    print("   - Compare timing with GPU version")
    print("   - Check memory usage during execution")
    print("   - Server logs are available in the terminal")

if __name__ == "__main__":
    main()
