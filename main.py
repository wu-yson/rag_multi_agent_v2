from contextlib import asynccontextmanager
from pathlib import Path
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# 固定路径，解决uvicorn导入问题
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# 仅仅引入路由，不写业务逻辑
from src.api.api_root import api_router


@asynccontextmanager
async def lifespan(app):
    yield                                    # 应用运行期间
    from src.mcp.client import close
    await close()

# 创建服务实例
app = FastAPI(title="多智能体服务", lifespan=lifespan)
app.include_router(api_router, prefix="/api")

# 托管前端页面（web/ 目录），访问 http://127.0.0.1:8000/web/
app.mount("/web", StaticFiles(directory="web", html=True), name="web")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )
