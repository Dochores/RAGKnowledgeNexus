# app.py
from fastapi import FastAPI,Request, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import os
import json
import uuid
from datetime import datetime
import asyncio
import re
import sqlite3
from typing import List, Dict
from contextlib import asynccontextmanager
 
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

# 添加CORS中间件允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
model = None
index = None
documents = []
document_to_chunks = {}
chunks_to_document = {}
all_chunks = []
client = None

# 文档和会话存储
uploaded_documents: Dict[str, Dict] = {}  # {id: {name, content, path}}
chat_sessions: Dict[str, Dict] = {}  # {id: {summary, updated_at, messages}}

# 保存和加载文档数据
def save_documents():
    # 创建一个可序列化的版本（不包含文件内容以减少文件大小）
    serializable_docs = {}
    for doc_id, doc_data in uploaded_documents.items():
        serializable_docs[doc_id] = {
            "name": doc_data["name"],
            "path": doc_data["path"]
        }
    
    with open("docs/documents_index.json", "w", encoding="utf-8") as f:
        json.dump(serializable_docs, f, ensure_ascii=False, indent=2)

def load_documents():
    global uploaded_documents
    
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
                uploaded_documents[doc_id] = {
                    "name": doc_data["name"],
                    "path": path,
                    "content": f"文件内容已保存到 {path}"  # 不加载实际内容
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
        api_key="your api key",
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
    for doc_id, doc_data in uploaded_documents.items():
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
        if doc_id in uploaded_documents:
            retrieved_docs.append(f"文档: {uploaded_documents[doc_id]['name']}")
    
    return retrieved_docs, retrieved_chunks


 

# 文档管理 API
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith((".txt", ".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="仅支持.txt或.pdf或.docx文件")
    
    # 确保docs目录存在
    os.makedirs("docs", exist_ok=True)
    
    # 保存文件到docs目录
    file_path = os.path.join("docs", file.filename)
    
    # 检查文件名是否重复，如果重复则添加时间戳
    if os.path.exists(file_path):
        filename, extension = os.path.splitext(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_path = os.path.join("docs", f"{filename}_{timestamp}{extension}")
        file_name = f"{filename}_{timestamp}{extension}"
    else:
        file_name = file.filename
        
    # 读取文件内容
    content = await file.read()
    
    # 保存文件到磁盘
    with open(file_path, "wb") as f:
        f.write(content)
    
    # 读取文件内容（用于存储在内存中）
    if file.filename.endswith(".txt"):
        try:
            content_text = content.decode("utf-8")
        except UnicodeDecodeError:
            content_text = content.decode("gbk", errors="ignore")
    else:
        content_text = f"文件内容已保存到 {file_path}"  # 实际需用pdf/docx解析库
    
    doc_id = str(uuid.uuid4())
    uploaded_documents[doc_id] = {
        "name": file_name,
        "content": content_text,
        "path": file_path
    }
    
    # 重建索引
    rebuild_index()
    
    # 保存文档索引
    save_documents()
    
    return {"id": doc_id, "name": file_name}

@app.get("/api/documents")
async def list_documents():
    return [{"id": k, "name": v["name"]} for k, v in uploaded_documents.items()]

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id not in uploaded_documents:
        raise HTTPException(status_code=404, detail="文档不存在")
    
    # 删除文件
    file_path = uploaded_documents[doc_id].get("path")
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"删除文件时出错: {str(e)}")
    
    # 从内存中删除记录
    del uploaded_documents[doc_id]
    
    # 重建索引
    rebuild_index()
    
    # 保存文档索引
    save_documents()
    
    return {"message": "删除成功"}


@app.post("/api/stream")
async def stream_post(request: Request):
    try:
        # 解析请求体中的 JSON 数据
        req_data = await request.json()
        query = req_data.get("query")
        session_id = req_data.get("session_id")  # 获取会话ID
        return await process_stream_request(query, session_id)
    except Exception as e:
        error_msg = str(e)
        print(f"聊天接口错误: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/stream")
async def stream_get(query: str = Query(None), session_id: str = Query(None)):
    try:
        if not query:
            raise HTTPException(status_code=400, detail="Missing query parameter")
        return await process_stream_request(query, session_id)
    except Exception as e:
        error_msg = str(e)
        print(f"聊天接口错误: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

async def process_stream_request(query: str, session_id: str = None):
    # 如果没有提供session_id，创建一个新的
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM chat_sessions WHERE id = ?", (session_id,))
    has_session = cursor.fetchone()
    conn.close()
    
    if not has_session:
        session_id = str(uuid.uuid4())
 

    # 检索相关文档
    retrieved_docs, retrieved_chunks = retrieve_docs(query)
    
    # 构建上下文
    context = "相关文档:\n" + "\n".join(retrieved_docs)
    
    if retrieved_chunks:
        context += "\n\n文档内容片段:\n"
        for i, (doc_id, chunk) in enumerate(retrieved_chunks):
            doc_name = "未知文档"
            if doc_id in uploaded_documents:
                doc_name = uploaded_documents[doc_id]["name"]
            context += f"[文档{i+1}: {doc_name}] {chunk}\n"
    else:
        context += "\n\n没有找到相关的文档内容。"
    
    prompt = f"上下文信息:\n{context}\n\n问题: {query}\n请基于上下文信息回答问题:"
    
    # 用于保存完整响应
    full_response = ""
    
    # 创建stream响应    
    async def generate():
        nonlocal full_response
        
        stream = client.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {"role": "system", "content": "你是一个专业的问答助手。请仅基于提供的上下文信息回答问题，不要添加任何未在上下文中提及的信息。如果没有相关信息，请明确告知用户无法回答该问题。"},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield f"data: {json.dumps({'content': content, 'session_id': session_id})}\n\n"
                print(f"data: {json.dumps({'content': content, 'session_id': session_id})}\n\n")
                await asyncio.sleep(0.01)  # 添加小延迟确保流式输出
                
            if chunk.choices[0].finish_reason is not None:
                yield f"data: {json.dumps({'content': '', 'session_id': session_id, 'done': True})}\n\n"
                break
                
        # 响应完成后，将完整会话保存到数据库
        if has_session:
            await add_message_to_session(session_id, query, full_response)
        else:
            await create_new_chat_session(session_id, query, full_response)
            
                
    return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked"
            }
        )

# 创建新的聊天会话
async def create_new_chat_session(session_id, query, response):
    # 创建会话摘要
    summary = query[:30] + "..." if len(query) > 30 else query
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        
        # 插入会话记录
        cursor.execute(
            "INSERT INTO chat_sessions (id, summary, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, summary, current_time, current_time)
        )
        
        # 插入用户消息
        cursor.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, "user", query, current_time)
        )
        
        # 插入机器人响应
        cursor.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, "bot", response, current_time)
        )
        
        conn.commit()
        conn.close()
        
        print(f"创建新会话 {session_id} 成功")
        return True
    except Exception as e:
        print(f"创建新会话失败: {str(e)}")
        return False

# 向现有会话添加消息
async def add_message_to_session(session_id, query, response):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        
        # 插入用户消息
        cursor.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, "user", query, current_time)
        )
        
        # 插入机器人响应
        cursor.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, "bot", response, current_time)
        )
        
        # 更新会话时间戳，使其保持最新
        cursor.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (current_time, session_id)
        )
        
        conn.commit()
        conn.close()
        
        print(f"向会话 {session_id} 添加消息成功")
        return True
    except Exception as e:
        print(f"向会话添加消息失败: {str(e)}")
        return False

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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)