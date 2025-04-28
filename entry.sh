#!/bin/bash

# Start Ollama server in the background
ollama serve &

# Wait for Ollama server to be ready
sleep 5

# Pull required Ollama models
ollama pull deepseek-r1:1.5b
ollama pull nomic-embed-text

# Run the specified application
if [ "$1" = "streamlit" ]; then
    streamlit run app.py --server.port 8502
elif [ "$1" = "jupyter" ]; then
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
else
    echo "Invalid command. Use 'streamlit' or 'jupyter'."
    exit 1
fi