# app.py
from fastapi import FastAPI,Request, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import os
import json
import uuid
from datetime import datetime
import asyncio
import sqlite3
from typing import Dict, List
from contextlib import asynccontextmanager
import urllib.parse

from utils.text_splitter import AdvancedTextSplitter
from utils.retriever import AdvancedRetriever
from utils.context_builder import ContextBuilder
from utils.prompt_builder import PromptBuilder

# 创建应用启动上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动前执行
    init()
    init_db()  # 初始化数据库
    load_documents()
    yield
    # 关闭时执行
    save_documents()
    # 可以在这里添加清理代码

# 创建FastAPI应用
app = FastAPI(lifespan=lifespan)
# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# 添加CORS中间件允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 SQLite 数据库
def init_db():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    
    # 创建聊天会话表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id TEXT PRIMARY KEY,
        summary TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 创建消息表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
    )
    ''')
    
    conn.commit()
    conn.close()
    print("数据库初始化完成")

# 全局变量
model = None
index = None
documents = {}
document_to_chunks = {}
chunks_to_document = {}
all_chunks = []
client = None

# 文档和会话存储
chat_sessions = {}

# 初始化组件
text_splitter = AdvancedTextSplitter(chunk_size=500, chunk_overlap=50)
retriever = AdvancedRetriever()
context_builder = ContextBuilder()
prompt_builder = PromptBuilder()

# 保存和加载文档数据
def save_documents():
    # 创建一个可序列化的版本（不包含文件内容以减少文件大小）
    serializable_docs = {}
    for doc_id, doc_data in documents.items():
        serializable_docs[doc_id] = {
            "name": doc_data["name"],
            "path": doc_data["path"],
            "chunks": doc_data["chunks"],
            "metadata": doc_data["metadata"]
        }
    
    with open("docs/documents_index.json", "w", encoding="utf-8") as f:
        json.dump(serializable_docs, f, ensure_ascii=False, indent=2)

def load_documents():
    global documents
    
    index_path = "docs/documents_index.json"
    if not os.path.exists(index_path):
        return
    
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            serialized_docs = json.load(f)
        
        # 加载文档元数据，但不加载全部内容
        for doc_id, doc_data in serialized_docs.items():
            path = doc_data.get("path")
            if path and os.path.exists(path):
                documents[doc_id] = {
                    "name": doc_data["name"],
                    "path": path,
                    "chunks": doc_data["chunks"],
                    "metadata": doc_data["metadata"]
                }
        
        # 重建索引
        rebuild_index()
    except Exception as e:
        print(f"加载文档索引失败: {str(e)}")

# 初始化函数
def init():
    global model, index, client
    
    # 初始化OpenAI客户端
    client = OpenAI(
        api_key="you api key",
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )
    
    # 加载嵌入模型
    local_model_path = 'local_m3e_model'
    if os.path.exists(local_model_path):
        model = SentenceTransformer(local_model_path)
    else:
        model = SentenceTransformer('moka-ai/m3e-base')
        model.save(local_model_path)

# 重新构建索引
def rebuild_index():
    global index, document_to_chunks, chunks_to_document, all_chunks
    
    # 重置数据
    document_to_chunks = {}
    chunks_to_document = {}
    all_chunks = []
    
    # 处理上传的文档
    for doc_id, doc_data in documents.items():
        # 获取文档内容
        content = doc_data.get("content", "")
        path = doc_data.get("path", "")
        
        # 如果是txt文件，直接读取内容
        if path.endswith(".txt") and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(path, "r", encoding="gbk") as f:
                        content = f.read()
                except:
                    pass
        
        # 分块（这里简化处理，实际应该根据文档大小进行更细致的分块）
        

        chunk_id = len(all_chunks)
        all_chunks.append(content)
        document_to_chunks[doc_id] = [chunk_id]
        chunks_to_document[chunk_id] = doc_id
    
    # 如果没有文档，不创建索引
    if not all_chunks:
        index = None
        return
        
    # 生成嵌入
    chunk_embeddings = get_embeddings(all_chunks)
    
    # 初始化FAISS索引
    dimension = chunk_embeddings.shape[1]  # 768 for m3e-base
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)
    
    # 保存索引
    faiss.write_index(index, "m3e_faiss_index.bin")
    
    # 保存映射关系
    mapping_data = {
        'doc_to_chunks': document_to_chunks,
        'chunks_to_doc': chunks_to_document,
        'all_chunks': all_chunks
    }
    np.save("chunks_mapping.npy", mapping_data)

# 获取嵌入向量
def get_embeddings(texts):
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.array(embeddings)

# 检索函数
def retrieve_docs(query, k=3):
    if index is None or not all_chunks:
        return [], []
        
    query_embedding = get_embeddings([query])
    distances, chunk_indices = index.search(query_embedding, k)
    
    # 获取包含这些chunks的原始文档
    retrieved_doc_ids = set()
    retrieved_chunks = []
    
    for chunk_idx in chunk_indices[0]:
        if chunk_idx >= 0 and chunk_idx < len(all_chunks):
            doc_id = chunks_to_document.get(int(chunk_idx))
            if doc_id is not None:
                retrieved_doc_ids.add(doc_id)
                retrieved_chunks.append((doc_id, all_chunks[int(chunk_idx)]))
    
    # 获取原始文档详情
    retrieved_docs = []
    for doc_id in retrieved_doc_ids:
        if doc_id in documents:
            retrieved_docs.append(f"文档: {documents[doc_id]['name']}")
    
    return retrieved_docs, retrieved_chunks

# 文档管理 API
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        doc_id = str(uuid.uuid4())
        
        # 保存文件
        file_path = f"docs/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(content)
        
        # 读取文件内容
        if file.filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            # 处理其他文件类型...
            raise HTTPException(status_code=400, detail="暂不支持该文件类型")
        
        # 文档分块
        chunks = text_splitter.split_text(content)
        chunk_metadata = [text_splitter.get_chunk_metadata(chunk) for chunk in chunks]
        
        # 更新检索器
        retriever.add_documents(chunks, chunk_metadata)
        
        # 保存文档信息
        documents[doc_id] = {
            "name": file.filename,
            "path": file_path,
            "chunks": chunks,
            "metadata": chunk_metadata
        }
        
        return {"message": "文档上传成功", "doc_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def list_documents():
    return [{"id": k, "name": v["name"]} for k, v in documents.items()]

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="文档不存在")
    
    # 删除文件
    file_path = documents[doc_id].get("path")
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"删除文件时出错: {str(e)}")
    
    # 从内存中删除记录
    del documents[doc_id]
    
    # 重建索引
    rebuild_index()
    
    # 保存文档索引
    save_documents()
    
    return {"message": "删除成功"}

@app.post("/api/chat")
async def chat(query: str, session_id: str = None):
    async def stream_response():
        try:
            # 检索相关文档
            retrieved_chunks = retriever.hybrid_search(query, k=3)
            
            # 构建上下文
            chat_history = chat_sessions.get(session_id, {}).get("messages", []) if session_id else None
            context = context_builder.build_context(retrieved_chunks, query, chat_history)
            
            # 构建提示词
            messages = prompt_builder.build_prompt(query, context, chat_history)
            
            # 模拟AI响应（实际项目中替换为真实的LLM调用）
            response = f"基于检索到的内容，为您回答问题：{query}\n\n"
            response += "1. 相关文档：\n"
            for chunk, score, metadata in retrieved_chunks:
                response += f"- 相关度 {score:.2f}: {chunk[:100]}...\n"
            
            # 流式输出
            for char in response:
                yield char.encode("utf-8")
                await asyncio.sleep(0.05)
            
            # 保存会话
            if not session_id:
                session_id = str(uuid.uuid4())
            if session_id not in chat_sessions:
                chat_sessions[session_id] = {
                    "summary": query[:30] + "..." if len(query) > 30 else query,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "messages": []
                }
            chat_sessions[session_id]["messages"].extend([
                {"role": "user", "content": query},
                {"role": "assistant", "content": response}
            ])
            
        except Exception as e:
            yield f"错误: {str(e)}".encode("utf-8")
    
    return StreamingResponse(stream_response(), media_type="text/plain")

# 会话历史记录 API
@app.get("/api/chat/history")
async def get_chat_history():
    try:
        conn = sqlite3.connect('chat_history.db')
        conn.row_factory = sqlite3.Row  # 启用行工厂，使结果可以通过列名访问
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, summary, updated_at  FROM chat_sessions ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        
        # 将行转换为字典
        sessions = [dict(row) for row in rows]
        
        conn.close()
        return sessions
        
    except Exception as e:
        print(f"获取聊天历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取聊天历史失败: {str(e)}")

@app.get("/api/chat/session/{session_id}")
async def get_session(session_id: str):
    try:
        conn = sqlite3.connect('chat_history.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 查询会话是否存在
        cursor.execute("SELECT id FROM chat_sessions WHERE id = ?", (session_id,))
        session = cursor.fetchone()
        
        if not session:
            conn.close()
            raise HTTPException(status_code=404, detail="会话不存在")
        
        # 获取会话中的所有消息
        cursor.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id asc",
            (session_id,)
        )
        messages = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return {"messages": messages}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"获取会话详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取会话详情失败: {str(e)}")

# 删除会话
@app.delete("/api/chat/session/{session_id}")
async def delete_session(session_id: str):
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        
        # 首先删除会话关联的所有消息
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        
        # 然后删除会话本身
        cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        
        if cursor.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail="会话不存在")
        
        conn.commit()
        conn.close()
        
        return {"message": "会话已删除"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"删除会话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")

# 健康检查接口
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# 运行服务
if __name__ == "__main__":
    # 确保必要的目录存在
    os.makedirs("docs", exist_ok=True)
    
    # 启动应用
    uvicorn.run(app, host="0.0.0.0", port=8000)