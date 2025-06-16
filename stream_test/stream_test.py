from fastapi import FastAPI, Request, Query
from fastapi.responses import StreamingResponse
from openai import OpenAI
import json
from fastapi.middleware.cors import CORSMiddleware
import asyncio

app = FastAPI()

# 允许跨域请求，适配前端
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化OpenAI客户端
# client = OpenAI(
#     api_key="your api key",
#     base_url="https://open.bigmodel.cn/api/paas/v4/"
# )


client = OpenAI(
    api_key="your api key",
    base_url="https://api.moonshot.cn/v1"
)

    
@app.post("/stream")
async def stream_post(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "你好，请简单介绍一下自己！")
    return StreamingResponse(
        generate_stream(prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked"
        }
    )

@app.get("/stream")
async def stream_get(prompt: str = Query("你好，请简单介绍一下自己！")):
    return StreamingResponse(
        generate_stream(prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked"
        }
    )

 
 

async def generate_stream(prompt: str):
    try:
        stream = client.chat.completions.create(
            #model="glm-4-plus",
            model="moonshot-v1-8k",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield f"data: {json.dumps({'content': content})}\n\n"
                await asyncio.sleep(0.01)  # 添加小延迟确保流式输出
                
            if chunk.choices[0].finish_reason is not None:
                yield "data: [DONE]\n\n"
                break
                
    except Exception as e:
        print(f"Error generating stream: {e}")
        yield f"data: {json.dumps({'content': f'错误: {str(e)}'})}\n\n"
        yield "data: [DONE]\n\n"

# 运行服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)