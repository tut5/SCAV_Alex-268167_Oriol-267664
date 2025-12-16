from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Union, Tuple
import numpy as np
import shutil
import os
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from enum import Enum

# Importem la lògica (Afegim MonsterTranscoder al final)
from app.scav_logic import (
    ColorTranslator, 
    RLEConverter, 
    SerpentineScanner, 
    DCTConverter, 
    DWTConverter, 
    ImageEncoder,
    VideoProcessor,
    MonsterTranscoder  # <--- NOVA CLASSE P2
)

app = FastAPI(
    title="SCAV Monster API (P1 + S2 + P2)",
    description="API completa que inclou Pràctica 1, Seminari 2 i Pràctica 2 (Transcoding).",
    version="3.0.0"
)

# Definim la carpeta compartida (que veu el Docker)
SHARED_FOLDER = "/shared"

# Muntem la carpeta per poder descarregar fitxers directament (necessari per a l'Encoding Ladder)
app.mount("/downloads", StaticFiles(directory=SHARED_FOLDER), name="downloads")

# ---------------------------------------------------------------------------
# MODELS DE DADES (Pydantic) - PRÀCTICA 1
# ---------------------------------------------------------------------------

class RGBInput(BaseModel):
    r: int
    g: int
    b: int

class YUVInput(BaseModel):
    y: float
    u: float
    v: float

class RLEInput(BaseModel):
    data: List[int]

class MatrixInput(BaseModel):
    matrix: List[List[float]]  # Matrius 2D per Serpentí i DCT

class DWTInput(BaseModel):
    # Estructura per a la inversa de DWT: (LL, (LH, HL, HH))
    # Simplifiquem l'entrada com una llista de 2 elements: [LL, [LH, HL, HH]]
    coeffs: List[Union[List[List[float]], List[List[List[float]]]]]

# ---------------------------------------------------------------------------
# ENDPOINTS - PRÀCTICA 1 (Basics)
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "SCAV API is fully operational!"}

# 1. COLOR TRANSLATOR
@app.post("/converter/rgb-to-yuv", tags=["P1 - Basics"])
def convert_rgb_to_yuv(color: RGBInput):
    translator = ColorTranslator()
    if not (0 <= color.r <= 255 and 0 <= color.g <= 255 and 0 <= color.b <= 255):
        raise HTTPException(status_code=400, detail="RGB values must be 0-255")
    y, u, v = translator.rgb_to_yuv(color.r, color.g, color.b)
    return {"y": y, "u": u, "v": v}

@app.post("/converter/yuv-to-rgb", tags=["P1 - Basics"])
def convert_yuv_to_rgb(color: YUVInput):
    translator = ColorTranslator()
    r, g, b = translator.yuv_to_rgb(color.y, color.u, color.v)
    return {"r": r, "g": g, "b": b}

# 2. RUN-LENGTH ENCODING
@app.post("/converter/rle", tags=["P1 - Basics"])
def run_length_encode(payload: RLEInput):
    rle = RLEConverter()
    encoded = rle.encode(payload.data)
    return {"encoded": encoded}

# 3. SERPENTINE SCANNER
@app.post("/scanner/serpentine", tags=["P1 - Basics"])
def serpentine_scan(payload: MatrixInput):
    scanner = SerpentineScanner()
    try:
        result = scanner.serpentine(np.array(payload.matrix))
        return {"serpentine": result.tolist()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# 4. DCT (Discrete Cosine Transform)
@app.post("/converter/dct/forward", tags=["P1 - Basics"])
def dct_forward(payload: MatrixInput):
    converter = DCTConverter()
    block = np.array(payload.matrix)
    result = converter.forward(block)
    return {"dct_coefficients": result.tolist()}

@app.post("/converter/dct/inverse", tags=["P1 - Basics"])
def dct_inverse(payload: MatrixInput):
    converter = DCTConverter()
    block = np.array(payload.matrix)
    result = converter.inverse(block)
    return {"reconstructed_block": result.tolist()}

# 5. DWT (Discrete Wavelet Transform)
@app.post("/converter/dwt/forward", tags=["P1 - Basics"])
def dwt_forward(payload: MatrixInput):
    converter = DWTConverter()
    data = np.array(payload.matrix)
    cA, (cH, cV, cD) = converter.forward(data)
    return {
        "approximation": cA.tolist(),
        "details": {
            "horizontal": cH.tolist(),
            "vertical": cV.tolist(),
            "diagonal": cD.tolist()
        }
    }

@app.post("/converter/dwt/inverse", tags=["P1 - Basics"])
def dwt_inverse(payload: DWTInput):
    converter = DWTConverter()
    try:
        ll = np.array(payload.coeffs[0])
        details = [np.array(d) for d in payload.coeffs[1]]
        coeffs_tuple = (ll, tuple(details))
        result = converter.inverse(coeffs_tuple)
        return {"reconstructed_image": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid DWT coefficients structure: {str(e)}")


# ---------------------------------------------------------------------------
# ENDPOINTS - SEMINARI 2 (Image & Video Processing)
# ---------------------------------------------------------------------------

@app.post("/image/resize", tags=["S2 - Image"])
async def resize_image_endpoint(width: int = None, height: int = None, file: UploadFile = File(...)):
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    output_filename = f"resized_{file.filename}"
    encoder = ImageEncoder()
    encoder.resize_image(file.filename, output_filename, width, height)
    
    output_path = f"{SHARED_FOLDER}/{output_filename}"
    return FileResponse(output_path)

@app.post("/image/compress-bw", tags=["S2 - Image"])
async def compress_bw_endpoint(file: UploadFile = File(...)):
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    output_filename = f"bw_{file.filename}"
    encoder = ImageEncoder()
    encoder.compress_bw(file.filename, output_filename)
    
    output_path = f"{SHARED_FOLDER}/{output_filename}"
    return FileResponse(output_path)


# TASK 1: REDIMENSIONAR VÍDEO
@app.post("/video/resize", tags=["S2 - Video"])
async def resize_video_endpoint(
    width: int = None, 
    height: int = None, 
    output_name: str = None,
    file: UploadFile = File(...)
):
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    if output_name:
        output_filename = output_name
        if "." not in output_filename:
            ext = file.filename.split(".")[-1]
            output_filename = f"{output_filename}.{ext}"
    else:
        output_filename = f"resized_{file.filename}"
    
    processor = VideoProcessor()
    try:
        processor.resize_video(file.filename, output_filename, width, height)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    return {
        "message": "Video processat correctament",
        "output_location": f"{SHARED_FOLDER}/{output_filename}",
        "access_on_host": f"./shared_data/{output_filename}",
        "width": width, "height": height
    }

# TASK 2: SUBMOSTREIG DE CROMA
class ChromaOption(str, Enum):
    YUV_420 = "yuv420p"
    YUV_422 = "yuv422p"
    YUV_444 = "yuv444p"

@app.post("/video/chroma-subsampling", tags=["S2 - Video"])
async def chroma_subsampling_endpoint(
    subsampling: ChromaOption, 
    output_name: str = None,
    file: UploadFile = File(...)
):
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    if output_name:
        output_filename = output_name
        if "." not in output_filename:
            ext = file.filename.split(".")[-1]
            output_filename = f"{output_filename}.{ext}"
    else:
        output_filename = f"{subsampling.value}_{file.filename}"
    
    processor = VideoProcessor()
    try:
        processor.change_chroma_subsampling(file.filename, output_filename, subsampling.value)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    return {
        "message": "Submostreig modificat correctament",
        "output_location": f"{SHARED_FOLDER}/{output_filename}",
        "chroma": subsampling.value
    }

# TASK 3: VIDEO INFO
@app.post("/video/info", tags=["S2 - Video"])
async def get_video_info_endpoint(file: UploadFile = File(...)):
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    processor = VideoProcessor()
    try:
        return processor.get_video_info(file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# TASK 4: BBB CONTAINER
@app.post("/video/bbb-container", tags=["S2 - Video"])
async def bbb_container_endpoint(output_name: str = None, file: UploadFile = File(...)):
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    if output_name:
        output_filename = output_name
        if "." not in output_filename:
            ext = file.filename.split(".")[-1]
            output_filename = f"{output_filename}.{ext}"
    else:
        output_filename = f"bbb_container_{file.filename}"
    
    processor = VideoProcessor()
    try:
        processor.process_bbb_container(file.filename, output_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    return {
        "message": "Contenidor BBB creat",
        "output_location": f"{SHARED_FOLDER}/{output_filename}"
    }

# TASK 5: COUNT TRACKS
@app.post("/video/count-tracks", tags=["S2 - Video"])
async def count_tracks_endpoint(file: UploadFile = File(...)):
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    processor = VideoProcessor()
    try:
        result = processor.count_tracks(file.filename)
        return {"filename": file.filename, "tracks_info": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# TASK 6: MOTION VECTORS
@app.post("/video/motion-vectors", tags=["S2 - Video"])
async def motion_vectors_endpoint(output_name: str = None, file: UploadFile = File(...)):
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    if output_name:
        output_filename = output_name
        if "." not in output_filename:
            ext = file.filename.split(".")[-1]
            output_filename = f"{output_filename}.{ext}"
    else:
        output_filename = f"motion_vectors_{file.filename}"
    
    processor = VideoProcessor()
    try:
        processor.visualize_motion_vectors(file.filename, output_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    return {
        "message": "Vectors generats",
        "output_location": f"{SHARED_FOLDER}/{output_filename}"
    }

# TASK 7: YUV HISTOGRAM
@app.post("/video/yuv-histogram", tags=["S2 - Video"])
async def yuv_histogram_endpoint(output_name: str = None, file: UploadFile = File(...)):
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    if output_name:
        output_filename = output_name
        if "." not in output_filename:
            ext = file.filename.split(".")[-1]
            output_filename = f"{output_filename}.{ext}"
    else:
        output_filename = f"yuv_histogram_{file.filename}"
    
    processor = VideoProcessor()
    try:
        processor.yuv_histogram(file.filename, output_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    return {
        "message": "Histograma generat",
        "output_location": f"{SHARED_FOLDER}/{output_filename}"
    }


# ===========================================================================
#  PART 3: PRÀCTICA 2 (TRANSCODING I ENCODING LADDER)
# ===========================================================================

# 1. Endpoint Transcodificació (Exercici 1)
# ---------------------------------------------------------------------------
@app.post("/api/transcode", tags=["P2 - Transcoding"])
async def transcode_endpoint(
    file: UploadFile = File(...), 
    codec: str = Form(...)
):
    """
    Converteix un vídeo a VP8, VP9, H265 o AV1.
    """
    # 1. Guardar el fitxer pujat a la carpeta compartida
    input_full_path = os.path.join(SHARED_FOLDER, file.filename)
    with open(input_full_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Instanciar la nova classe MonsterTranscoder
    transcoder = MonsterTranscoder()
    
    # 3. Determinar extensió de sortida segons el còdec
    ext = "mkv" if codec == "av1" else "webm" if "vp" in codec else "mp4"
    output_filename = f"converted_{codec}_{file.filename.split('.')[0]}.{ext}"
    output_path = os.path.join(SHARED_FOLDER, output_filename)

    try:
        # 4. Executar transcodificació (al contenidor Docker veí)
        transcoder.convert_to_codec(file.filename, output_filename, codec)
        
        # Retornem el fitxer directament
        return FileResponse(output_path, filename=output_filename)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# 2. Endpoint Encoding Ladder (Exercici 2)
# ---------------------------------------------------------------------------
@app.post("/api/ladder", tags=["P2 - Transcoding"])
async def ladder_endpoint(file: UploadFile = File(...)):
    """
    Genera automàticament versions en 1080p, 720p i 480p (Encoding Ladder).
    """
    # 1. Guardar fitxer
    input_full_path = os.path.join(SHARED_FOLDER, file.filename)
    with open(input_full_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    transcoder = MonsterTranscoder()
    
    try:
        # 2. Generar Ladder
        files = transcoder.create_encoding_ladder(file.filename)
        
        # Retornem JSON amb els enllaços de descàrrega
        # Els enllaços apunten al muntatge estàtic que hem definit a l'inici (/downloads)
        return {
            "status": "success",
            "message": "Ladder generat correctament",
            "files": files,
            "download_links": [f"/downloads/{f}" for f in files]
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})