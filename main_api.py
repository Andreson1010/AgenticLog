"""
Ponto de entrada do servidor FastAPI do AgenticLog.

Uso: python main_api.py
"""
import uvicorn

from agenticlog.config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "agenticlog.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )
