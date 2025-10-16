#!/bin/bash

# Start RoverMitra Llama API Server
echo "🚀 Starting RoverMitra Llama API Server..."

# Kill any existing server
pkill -f llama_api_server.py 2>/dev/null

# Start the server
cd /data/abdul/RoverMitra
python llama_api_server.py &

# Wait for server to start
echo "⏳ Waiting for server to start..."
sleep 15

# Check if server is running
if curl -s http://localhost:8002/health > /dev/null; then
    echo "✅ RoverMitra Llama API Server is running!"
    echo "📊 Server status:"
    curl -s http://localhost:8002/health | python -m json.tool
else
    echo "❌ Failed to start server"
    exit 1
fi

echo ""
echo "🎯 Now you can run main.py and it will use the API server for fast inference!"
echo "   python main.py"
echo ""
echo "🛑 To stop the server: pkill -f llama_api_server.py"
