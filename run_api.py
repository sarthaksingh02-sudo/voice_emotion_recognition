"""
FastAPI server runner for Emotion Recognition API v1.1
Run this script to start the API server
"""

import uvicorn
from config import API_HOST, API_PORT, DEBUG

if __name__ == "__main__":
    print(f"🚀 Starting Emotion Recognition API v1.1")
    print(f"📍 Server: {API_HOST}:{API_PORT}")
    print(f"🔧 Debug Mode: {DEBUG}")
    print(f"📋 Available endpoints:")
    print(f"   - GET  /health - Health check")
    print(f"   - POST /predict - Emotion prediction")
    print(f"   - GET  /logs - Retrieve logs")
    print(f"   - GET  /docs - API documentation")
    print(f"")
    print(f"🌐 Access API docs at: http://{API_HOST}:{API_PORT}/docs")
    print(f"")
    
    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG,
        access_log=True
    )
