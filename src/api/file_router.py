import ctypes
import time
import subprocess
from fastapi import APIRouter

router = APIRouter()

@router.get("/open-folder")
async def open_folder(path: str):
    """ 打开本地文件夹（强制置顶） """
    ctypes.windll.user32.keybd_event(0x12, 0, 0, 0)       # 模拟按下 ALT
    subprocess.Popen(f'explorer "{path}"')
    time.sleep(0.3)
    ctypes.windll.user32.keybd_event(0x12, 0, 2, 0)       # 松开 ALT
    return {"code": 200}