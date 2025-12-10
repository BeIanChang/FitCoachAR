# romp_server.py
from typing import List, Optional
import io
import cv2
import numpy as np
import romp
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="ROMP 3D Pose Server")

# 全局模型实例，进程启动时加载一次
romp_model = None

def create_romp_model():
    settings = romp.main.default_settings
    settings.mode = 'image'       # 我们按帧处理
    settings.show = False         # 不弹窗口
    settings.save_path = ''       # 不在磁盘乱写
    settings.calc_smpl = True
    settings.render_mesh = False
    # 只保留最大的人（可选）
    settings.show_largest = True
    model = romp.ROMP(settings)
    return model

@app.on_event("startup")
def load_model():
    global romp_model
    romp_model = create_romp_model()
    print("ROMP model loaded.")

class RompResult(BaseModel):
    joints_3d: List[List[float]]                      # [J, 3] 必有
    joints_2d: Optional[List[List[float]]] = None     # [J, 2] 可选
    verts: Optional[List[List[float]]] = None         # [V, 3] 可选
    frame_idx: int = 0


@app.post("/infer_image", response_model=RompResult)
async def infer_image(file: UploadFile = File(...)):
    if romp_model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded"})

    content = await file.read()
    image_array = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)  # BGR

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Cannot decode image"})

    outputs = romp_model(img)

    # 先看一下有啥 key（方便你调试）
    print("ROMP outputs keys:", outputs.keys())

    # 1. 3D joints（必需）
    joints_3d_all = outputs["joints"]          # (N, J, 3) 或 (J, 3)
    if isinstance(joints_3d_all, np.ndarray) and joints_3d_all.ndim == 3:
        joints_3d = joints_3d_all[0]          # 取第一个人 (J, 3)
    else:
        joints_3d = joints_3d_all             # 已经是 (J, 3)

    # 2. 2D joints（可选）
    joints_2d = None
    if "pj2d" in outputs:
        pj2d_all = outputs["pj2d"]
        if isinstance(pj2d_all, np.ndarray) and pj2d_all.ndim == 3:
            joints_2d = pj2d_all[0]
        else:
            joints_2d = pj2d_all
    elif "joints2d" in outputs:
        pj2d_all = outputs["joints2d"]
        if isinstance(pj2d_all, np.ndarray) and pj2d_all.ndim == 3:
            joints_2d = pj2d_all[0]
        else:
            joints_2d = pj2d_all
    # else: 保持 None

    # 3. verts（可选）
    verts = None
    if "verts" in outputs:
        verts_all = outputs["verts"]
        if isinstance(verts_all, np.ndarray) and verts_all.ndim == 3:
            verts = verts_all[0]
        else:
            verts = verts_all

    result = RompResult(
        joints_3d=joints_3d.tolist(),
        joints_2d=joints_2d.tolist() if joints_2d is not None else None,
        verts=verts.tolist() if verts is not None else None,
        frame_idx=0,
    )
    return result

@app.get("/health")
def health():
    return {"status": "ok"}
