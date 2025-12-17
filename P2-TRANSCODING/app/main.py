import os
import shutil
import numpy as np
from typing import List, Union
from enum import Enum

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Importem la lògica de negoci
from app.scav_logic import (
    ColorTranslator, 
    RLEConverter, 
    SerpentineScanner, 
    DCTConverter, 
    DWTConverter, 
    ImageEncoder,
    VideoProcessor,
    MonsterTranscoder
)

app = FastAPI(
    title="SCAV Monster API (P1 + S2 + P2)",
    description="API completa: Pràctica 1, Seminari 2 i Pràctica 2.",
    version="3.1.0"
)

# ---------------------------------------------------------------------------
# 1. CONFIGURACIÓ DE SISTEMA I GUI
# ---------------------------------------------------------------------------

SHARED_FOLDER = "/shared"
app.mount("/downloads", StaticFiles(directory=SHARED_FOLDER), name="downloads")

# Configuració de Plantilles (HTML)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# FUNCIÓ AUXILIAR PER NO REPETIR CODI (OPTIMITZACIÓ)
def save_file_to_shared(file: UploadFile) -> str:
    """Guarda el fitxer pujat a la carpeta compartida i retorna el nom."""
    file_path = os.path.join(SHARED_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file.filename

# ---------------------------------------------------------------------------
# 2. ENDPOINT PRINCIPAL (GUI)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------------------------------------------------------------------------
# 3. MODELS DE DADES (Pydantic)
# ---------------------------------------------------------------------------

class RGBInput(BaseModel):
    r: int; g: int; b: int

class YUVInput(BaseModel):
    y: float; u: float; v: float

class RLEInput(BaseModel):
    data: List[int]

class MatrixInput(BaseModel):
    matrix: List[List[float]]

class DWTInput(BaseModel):
    coeffs: List[Union[List[List[float]], List[List[List[float]]]]]

# ---------------------------------------------------------------------------
# 4. ENDPOINTS - PRÀCTICA 1 (Conceptes Bàsics)
# ---------------------------------------------------------------------------

@app.post("/converter/rgb-to-yuv", tags=["P1 - Basics"])
def convert_rgb_to_yuv(color: RGBInput):
    # Validació d'entrada
    if not (0 <= color.r <= 255 and 0 <= color.g <= 255 and 0 <= color.b <= 255):
        raise HTTPException(status_code=400, detail="Els valors RGB han de ser 0-255")
    y, u, v = ColorTranslator().rgb_to_yuv(color.r, color.g, color.b)
    return {"y": y, "u": u, "v": v}

@app.post("/converter/yuv-to-rgb", tags=["P1 - Basics"])
def convert_yuv_to_rgb(color: YUVInput):
    r, g, b = ColorTranslator().yuv_to_rgb(color.y, color.u, color.v)
    return {"r": r, "g": g, "b": b}

@app.post("/converter/rle", tags=["P1 - Basics"])
def run_length_encode(payload: RLEInput):
    return {"encoded": RLEConverter().encode(payload.data)}

@app.post("/scanner/serpentine", tags=["P1 - Basics"])
def serpentine_scan(payload: MatrixInput):
    try:
        result = SerpentineScanner().serpentine(np.array(payload.matrix))
        return {"serpentine": result.tolist()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/converter/dct/forward", tags=["P1 - Basics"])
def dct_forward(payload: MatrixInput):
    return {"dct_coefficients": DCTConverter().forward(np.array(payload.matrix)).tolist()}

@app.post("/converter/dct/inverse", tags=["P1 - Basics"])
def dct_inverse(payload: MatrixInput):
    return {"reconstructed_block": DCTConverter().inverse(np.array(payload.matrix)).tolist()}

@app.post("/converter/dwt/forward", tags=["P1 - Basics"])
def dwt_forward(payload: MatrixInput):
    cA, (cH, cV, cD) = DWTConverter().forward(np.array(payload.matrix))
    return {
        "approximation": cA.tolist(),
        "details": {"horizontal": cH.tolist(), "vertical": cV.tolist(), "diagonal": cD.tolist()}
    }

@app.post("/converter/dwt/inverse", tags=["P1 - Basics"])
def dwt_inverse(payload: DWTInput):
    try:
        ll = np.array(payload.coeffs[0])
        details = [np.array(d) for d in payload.coeffs[1]]
        result = DWTConverter().inverse((ll, tuple(details)))
        return {"reconstructed_image": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Estructura DWT invàlida: {str(e)}")

# ---------------------------------------------------------------------------
# 5. ENDPOINTS - SEMINARI 2 (Imatge i Vídeo)
# ---------------------------------------------------------------------------

@app.post("/image/resize", tags=["S2 - Imatge"])
async def resize_image_endpoint(width: int = None, height: int = None, file: UploadFile = File(...)):
    filename = save_file_to_shared(file)
    output = f"resized_{filename}"
    ImageEncoder().resize_image(filename, output, width, height)
    return FileResponse(f"{SHARED_FOLDER}/{output}")

@app.post("/image/compress-bw", tags=["S2 - Imatge"])
async def compress_bw_endpoint(file: UploadFile = File(...)):
    filename = save_file_to_shared(file)
    output = f"bw_{filename}"
    ImageEncoder().compress_bw(filename, output)
    return FileResponse(f"{SHARED_FOLDER}/{output}")

@app.post("/video/resize", tags=["S2 - Vídeo"])
async def resize_video_endpoint(width: int = None, height: int = None, output_name: str = None, file: UploadFile = File(...)):
    filename = save_file_to_shared(file)
    
    # Determinem el nom de sortida
    if output_name:
        out_file = f"{output_name}.{filename.split('.')[-1]}" if "." not in output_name else output_name
    else:
        out_file = f"resized_{filename}"
    
    try:
        VideoProcessor().resize_video(filename, out_file, width, height)
        return {"message": "Vídeo processat", "output_location": f"{SHARED_FOLDER}/{out_file}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChromaOption(str, Enum):
    YUV_420 = "yuv420p"; YUV_422 = "yuv422p"; YUV_444 = "yuv444p"

@app.post("/video/chroma-subsampling", tags=["S2 - Vídeo"])
async def chroma_subsampling_endpoint(subsampling: ChromaOption, output_name: str = None, file: UploadFile = File(...)):
    filename = save_file_to_shared(file)
    out_file = output_name if output_name else f"{subsampling.value}_{filename}"
    
    VideoProcessor().change_chroma_subsampling(filename, out_file, subsampling.value)
    return {"message": "Submostreig modificat", "output_location": f"{SHARED_FOLDER}/{out_file}"}

@app.post("/video/info", tags=["S2 - Vídeo"])
async def get_video_info_endpoint(file: UploadFile = File(...)):
    filename = save_file_to_shared(file)
    return VideoProcessor().get_video_info(filename)

@app.post("/video/bbb-container", tags=["S2 - Vídeo"])
async def bbb_container_endpoint(output_name: str = None, file: UploadFile = File(...)):
    filename = save_file_to_shared(file)
    out_file = output_name if output_name else f"bbb_container_{filename}"
    VideoProcessor().process_bbb_container(filename, out_file)
    return {"message": "Contenidor BBB creat", "output_location": f"{SHARED_FOLDER}/{out_file}"}

@app.post("/video/count-tracks", tags=["S2 - Vídeo"])
async def count_tracks_endpoint(file: UploadFile = File(...)):
    filename = save_file_to_shared(file)
    return {"filename": filename, "tracks_info": VideoProcessor().count_tracks(filename)}

@app.post("/video/motion-vectors", tags=["S2 - Vídeo"])
async def motion_vectors_endpoint(output_name: str = None, file: UploadFile = File(...)):
    filename = save_file_to_shared(file)
    out_file = output_name if output_name else f"motion_vectors_{filename}"
    VideoProcessor().visualize_motion_vectors(filename, out_file)
    return {"message": "Vectors generats", "output_location": f"{SHARED_FOLDER}/{out_file}"}

@app.post("/video/yuv-histogram", tags=["S2 - Vídeo"])
async def yuv_histogram_endpoint(output_name: str = None, file: UploadFile = File(...)):
    filename = save_file_to_shared(file)
    out_file = output_name if output_name else f"yuv_histogram_{filename}"
    VideoProcessor().yuv_histogram(filename, out_file)
    return {"message": "Histograma generat", "output_location": f"{SHARED_FOLDER}/{out_file}"}

# ---------------------------------------------------------------------------
# 6. ENDPOINTS - PRÀCTICA 2 (Transcoding i Ladder)
# ---------------------------------------------------------------------------

@app.post("/api/transcode", tags=["P2 - Transcoding"])
async def transcode_endpoint(file: UploadFile = File(...), codec: str = Form(...)):
    filename = save_file_to_shared(file)
    
    # Determinem extensió automàtica
    ext = "mkv" if codec == "av1" else "webm" if "vp" in codec else "mp4"
    out_file = f"converted_{codec}_{filename.split('.')[0]}.{ext}"
    output_path = os.path.join(SHARED_FOLDER, out_file)

    try:
        MonsterTranscoder().convert_to_codec(filename, out_file, codec)
        return FileResponse(output_path, filename=out_file)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/ladder", tags=["P2 - Transcoding"])
async def ladder_endpoint(file: UploadFile = File(...)):
    filename = save_file_to_shared(file)
    try:
        files = MonsterTranscoder().create_encoding_ladder(filename)
        return {
            "status": "success",
            "message": "Ladder generat correctament",
            "files": files,
            "download_links": [f"/downloads/{f}" for f in files]
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})