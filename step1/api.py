from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import uuid
from datetime import datetime
import re
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 允许跨域请求，适配前端
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 内存存储：文档和会话
documents: Dict[str, Dict] = {}  # {id: {name, content}}
chat_sessions: Dict[str, Dict] = {}  # {id: {summary, timestamp, messages}}

class ChatRequest(BaseModel):
    query: str

# 文档管理
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith((".txt", ".pdf")):
        raise HTTPException(status_code=400, detail="仅支持.txt或.pdf文件")
    
    # 模拟读取文件内容（仅支持txt简化处理）
    content = await file.read()
    if file.filename.endswith(".txt"):
        content = content.decode("utf-8")
    else:
        content = "PDF内容（模拟）"  # 实际需用pdf解析库如PyPDF2
    
    doc_id = str(uuid.uuid4())
    documents[doc_id] = {
        "name": file.filename,
        "content": content
    }
    return {"id": doc_id, "name": file.filename}

@app.get("/api/documents")
async def list_documents():
    return [{"id": k, "name": v["name"]} for k, v in documents.items()]

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="文档不存在")
    del documents[doc_id]
    return {"message": "删除成功"}

# 对话和历史记录
@app.post("/api/chat")
async def chat(request: ChatRequest):
    async def stream_response():
        # 简单RAG：搜索文档中的关键词
        query = request.query.lower()
        relevant_content = ""
        for doc in documents.values():
            if re.search(query, doc["content"].lower()):
                relevant_content += doc["content"][:100] + "... "
        
        # 模拟模型响应
        response = f"基于文档：{relevant_content or '无相关文档'}\n回答：{query} 的回答（模拟）。"
        
        # 流式输出：逐字符发送
        for char in response:
            yield char.encode("utf-8")
            await asyncio.sleep(0.05)  # 模拟延迟

        # 保存会话
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = {
            "summary": query[:30] + "..." if len(query) > 30 else query,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": [
                {"role": "user", "content": query},
                {"role": "bot", "content": response}
            ]
        }

    return StreamingResponse(stream_response(), media_type="text/plain")

@app.get("/api/chat/history")
async def get_chat_history():
    return [
        {
            "id": k,
            "summary": v["summary"],
            "timestamp": v["timestamp"]
        }
        for k, v in chat_sessions.items()
    ]

@app.get("/api/chat/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"messages": chat_sessions[session_id]["messages"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)