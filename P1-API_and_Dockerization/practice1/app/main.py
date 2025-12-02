from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Tuple
import numpy as np
import shutil
import os
from fastapi import UploadFile, File
from fastapi.responses import FileResponse

# Importem la lògica
from app.scav_logic import (
    ColorTranslator, 
    RLEConverter, 
    SerpentineScanner, 
    DCTConverter, 
    DWTConverter, 
    ImageEncoder
)

app = FastAPI(
    title="SCAV Practice 1 API",
    description="API completa amb els algoritmes del Seminari 1",
    version="1.1.0"
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
    # 1. Guardem el fitxer que puja l'usuari a la carpeta compartida
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    # 2. Definim el nom de sortida
    output_filename = f"resized_{file.filename}"
    
    # 3. Cridem al contenidor d'FFmpeg via la nostra classe lògica
    encoder = ImageEncoder()
    encoder.resize_image(file.filename, output_filename, width, height)
    
    # 4. Retornem el fitxer processat
    output_path = f"{SHARED_FOLDER}/{output_filename}"
    return FileResponse(output_path)

@app.post("/image/compress-bw", tags=["FFmpeg"])
async def compress_bw_endpoint(file: UploadFile = File(...)):
    # 1. Guardem el fitxer
    file_location = f"{SHARED_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    # 2. Definim sortida
    output_filename = f"bw_{file.filename}"
    
    # 3. Executem la conversió al contenidor veí
    encoder = ImageEncoder()
    encoder.compress_bw(file.filename, output_filename)
    
    # 4. Retornem resultat
    output_path = f"{SHARED_FOLDER}/{output_filename}"
    return FileResponse(output_path)