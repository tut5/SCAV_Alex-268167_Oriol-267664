from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Tuple
import numpy as np
import shutil
import os
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
from enum import Enum

# Importem la lògica
from app.scav_logic import (
    ColorTranslator, 
    RLEConverter, 
    SerpentineScanner, 
    DCTConverter, 
    DWTConverter, 
    ImageEncoder,
    VideoProcessor
)

app = FastAPI(
    title="SCAV Practice 1 API",
    description="API lab1 (SCAV)",
    version="2.0.0"
)

# --- MODELS DE DADES (Pydantic) ---

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

# --- ENDPOINTS ---

@app.get("/")
def root():
    return {"message": "SCAV API is fully operational!"}

# 1. COLOR TRANSLATOR
@app.post("/converter/rgb-to-yuv", tags=["Colors"])
def convert_rgb_to_yuv(color: RGBInput):
    translator = ColorTranslator()
    if not (0 <= color.r <= 255 and 0 <= color.g <= 255 and 0 <= color.b <= 255):
        raise HTTPException(status_code=400, detail="RGB values must be 0-255")
    y, u, v = translator.rgb_to_yuv(color.r, color.g, color.b)
    return {"y": y, "u": u, "v": v}

@app.post("/converter/yuv-to-rgb", tags=["Colors"])
def convert_yuv_to_rgb(color: YUVInput):
    translator = ColorTranslator()
    r, g, b = translator.yuv_to_rgb(color.y, color.u, color.v)
    return {"r": r, "g": g, "b": b}

# 2. RUN-LENGTH ENCODING
@app.post("/converter/rle", tags=["Compression"])
def run_length_encode(payload: RLEInput):
    rle = RLEConverter()
    encoded = rle.encode(payload.data)
    return {"encoded": encoded}

# 3. SERPENTINE SCANNER
@app.post("/scanner/serpentine", tags=["Scanners"])
def serpentine_scan(payload: MatrixInput):
    scanner = SerpentineScanner()
    try:
        result = scanner.serpentine(np.array(payload.matrix))
        # Convertim a llista per retornar JSON
        return {"serpentine": result.tolist()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# 4. DCT (Discrete Cosine Transform)
@app.post("/converter/dct/forward", tags=["Transforms"])
def dct_forward(payload: MatrixInput):
    converter = DCTConverter()
    block = np.array(payload.matrix)
    result = converter.forward(block)
    return {"dct_coefficients": result.tolist()}

@app.post("/converter/dct/inverse", tags=["Transforms"])
def dct_inverse(payload: MatrixInput):
    converter = DCTConverter()
    block = np.array(payload.matrix)
    result = converter.inverse(block)
    return {"reconstructed_block": result.tolist()}

# 5. DWT (Discrete Wavelet Transform)
@app.post("/converter/dwt/forward", tags=["Transforms"])
def dwt_forward(payload: MatrixInput):
    converter = DWTConverter()
    data = np.array(payload.matrix)
    # coeffs és una tupla (cA, (cH, cV, cD))
    cA, (cH, cV, cD) = converter.forward(data)
    
    return {
        "approximation": cA.tolist(),
        "details": {
            "horizontal": cH.tolist(),
            "vertical": cV.tolist(),
            "diagonal": cD.tolist()
        }
    }

@app.post("/converter/dwt/inverse", tags=["Transforms"])
def dwt_inverse(payload: DWTInput):
    """
    Nota: Per fer la inversa, cal passar l'estructura exacta:
    coeffs = [ Aproximació(LL), [Horitzontal(LH), Vertical(HL), Diagonal(HH)] ]
    """
    converter = DWTConverter()
    
    try:
        # Reconstruïm l'estructura de tuples que espera PyWavelets
        # payload.coeffs[0] -> LL
        # payload.coeffs[1] -> [LH, HL, HH]
        ll = np.array(payload.coeffs[0])
        details = [np.array(d) for d in payload.coeffs[1]]
        
        coeffs_tuple = (ll, tuple(details))
        
        result = converter.inverse(coeffs_tuple)
        return {"reconstructed_image": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid DWT coefficients structure: {str(e)}")
    

SHARED_FOLDER = "/shared"

@app.post("/image/resize", tags=["FFmpeg"])
async def resize_image_endpoint(width: int = None, height: int = None, file: UploadFile = File(...)):
    # Guardem el fitxer que puja l'usuari a la carpeta compartida
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    # Definim el nom de sortida
    output_filename = f"resized_{file.filename}"
    
    # Cridem al contenidor d'FFmpeg
    encoder = ImageEncoder()
    encoder.resize_image(file.filename, output_filename, width, height)
    
    # Retornem el fitxer processat
    output_path = f"{SHARED_FOLDER}/{output_filename}"
    return FileResponse(output_path)

@app.post("/image/compress-bw", tags=["FFmpeg"])
async def compress_bw_endpoint(file: UploadFile = File(...)):
    # Guardem el fitxer
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    # Definim sortida
    output_filename = f"bw_{file.filename}"
    
    # Executem la conversió al contenidor ffmpeg
    encoder = ImageEncoder()
    encoder.compress_bw(file.filename, output_filename)
    
    # Resultat
    output_path = f"{SHARED_FOLDER}/{output_filename}"
    return FileResponse(output_path)

# ---------------------------------------------------------------------------
# SEMINARI 2 - TASK 1: REDIMENSIONAR VÍDEO (Big Buck Bunny)
# ---------------------------------------------------------------------------
@app.post("/video/resize", tags=["S2 - Video"])
async def resize_video_endpoint(
    width: int = None, 
    height: int = None, 
    output_name: str = None,
    file: UploadFile = File(...)
):
    # Guardem el vídeo a la carpeta compartida
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    # Definim nom de sortida
    if output_name:
        output_filename = output_name
        if "." not in output_filename:
            ext = file.filename.split(".")[-1]
            output_filename = f"{output_filename}.{ext}"
    else:
        output_filename = f"resized_{file.filename}"
    
    # Processem el vídeo 
    processor = VideoProcessor()
    
    try:
        processor.resize_video(file.filename, output_filename, width, height)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el processament de vídeo: {str(e)}")
    
    # Retornem la ruta del video (si retornem el video en si el navegador es satura)
    return {
        "message": "Video processat correctament",
        "output_location": f"{SHARED_FOLDER}/{output_filename}",
        "access_on_host": f"./shared_data/{output_filename}", # Ruta
        "width": width,
        "height": height
    }

# ---------------------------------------------------------------------------
# SEMINARI 2 - TASK 2: MODIFICAR EL SUBMOSTREIG DE CROMA
# ---------------------------------------------------------------------------
# Definim les opcions vàlides per a l'usuari
class ChromaOption(str, Enum):
    YUV_420 = "yuv420p"
    YUV_422 = "yuv422p"
    YUV_444 = "yuv444p"

# Endpoint
@app.post("/video/chroma-subsampling", tags=["S2 - Video"])
async def chroma_subsampling_endpoint(
    subsampling: ChromaOption, 
    output_name: str = None,
    file: UploadFile = File(...)
):
    # Guardar fitxer
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    # Definim nom de sortida
    if output_name:
        output_filename = output_name
        if "." not in output_filename:
            ext = file.filename.split(".")[-1]
            output_filename = f"{output_filename}.{ext}"
    else:
        # Per defecte usem el valor del subsampling (ex: yuv422p_video.mp4)
        output_filename = f"{subsampling.value}_{file.filename}"
    
    # Processar el video
    processor = VideoProcessor()
    try:
        processor.change_chroma_subsampling(
            file.filename, 
            output_filename, 
            subsampling.value
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error canviant chroma: {str(e)}")
    
    # Retornem la ruta del video (si retornem el video en si el navegador es satura)
    return {
        "message": "Submostreig de croma modificat correctament",
        "output_location": f"{SHARED_FOLDER}/{output_filename}",
        "access_on_host": f"./shared_data/{output_filename}",
        "chroma": subsampling.value
    }

# ---------------------------------------------------------------------------
# SEMINARI 2 - TASK 3: VIDEO INFO (ffprobe)
# ---------------------------------------------------------------------------
@app.post("/video/info", tags=["S2 - Video"])
async def get_video_info_endpoint(file: UploadFile = File(...)):
    # Guardar fitxer
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    # Processar
    processor = VideoProcessor()
    try:
        # Cridem al mètode que conté la lògica
        video_data = processor.get_video_info(file.filename)
        return video_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obtenint info: {str(e)}")
    
# ---------------------------------------------------------------------------
# SEMINARI 2 - TASK 4: BBB CONTAINER
# ---------------------------------------------------------------------------

@app.post("/video/bbb-container", tags=["S2 - Video"])
async def bbb_container_endpoint(
    output_name: str = None,
    file: UploadFile = File(...)
):
    # Guardar fitxer
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    # Definim nom de sortida
    if output_name:
        output_filename = output_name
        if "." not in output_filename:
            ext = file.filename.split(".")[-1]
            output_filename = f"{output_filename}.{ext}"
    else:
        output_filename = f"bbb_container_{file.filename}"
    
    # Processar
    processor = VideoProcessor()
    try:
        processor.process_bbb_container(file.filename, output_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creant contenidor: {str(e)}")
    
    # Retornem la ruta del video (si retornem el video en si el navegador es satura)
    return {
        "message": "Contenidor BBB creat correctament (Vídeo tallat + 3 Àudios)",
        "output_location": f"{SHARED_FOLDER}/{output_filename}",
        "access_on_host": f"./shared_data/{output_filename}"
    }


# ---------------------------------------------------------------------------
# SEMINARI 2 - TASK 5: COUNT TRACKS
# ---------------------------------------------------------------------------

@app.post("/video/count-tracks", tags=["S2 - Video"])
async def count_tracks_endpoint(file: UploadFile = File(...)):
    # Guardar fitxer
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    # Processar
    processor = VideoProcessor()
    try:
        result = processor.count_tracks(file.filename)
        return {
            "filename": file.filename,
            "tracks_info": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comptant pistes: {str(e)}")
    
# ---------------------------------------------------------------------------
# SEMINARI 2 - TASK 6: MACROBLOCKS AND MOTION VECTORS
# ---------------------------------------------------------------------------

@app.post("/video/motion-vectors", tags=["S2 - Video"])
async def motion_vectors_endpoint(
    output_name: str = None,
    file: UploadFile = File(...)
):
    # Guardar fitxer
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    # Definim nom de sortida
    if output_name:
        output_filename = output_name
        if "." not in output_filename:
            ext = file.filename.split(".")[-1]
            output_filename = f"{output_filename}.{ext}"
    else:
        output_filename = f"motion_vectors_{file.filename}"
    
    # Processar el vídeo
    processor = VideoProcessor()
    try:
        processor.visualize_motion_vectors(file.filename, output_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generant vectors: {str(e)}")
    
    # Retornem la ruta del video (si retornem el video en si el navegador es satura)
    return {
        "message": "Vídeo amb vectors de moviment generat correctament",
        "output_location": f"{SHARED_FOLDER}/{output_filename}",
        "access_on_host": f"./shared_data/{output_filename}"
    }


# ---------------------------------------------------------------------------
# SEMINARI 2 - TASK 7: YUV HISTOGRAM
# ---------------------------------------------------------------------------

@app.post("/video/yuv-histogram", tags=["S2 - Video"])
async def yuv_histogram_endpoint(
    output_name: str = None,
    file: UploadFile = File(...)
):
    # Guardar fitxer
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    # Definim nom de sortida
    if output_name:
        output_filename = output_name
        if "." not in output_filename:
            ext = file.filename.split(".")[-1]
            output_filename = f"{output_filename}.{ext}"
    else:
        output_filename = f"yuv_histogram_{file.filename}"
    
    # Processar el vídeo
    processor = VideoProcessor()
    try:
        processor.yuv_histogram(file.filename, output_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generant histograma: {str(e)}")
    
    # Retornem la ruta del video (si retornem el video en si el navegador es satura)
    return {
        "message": "Vídeo amb histograma YUV generat correctament",
        "output_location": f"{SHARED_FOLDER}/{output_filename}",
        "access_on_host": f"./shared_data/{output_filename}"
    }