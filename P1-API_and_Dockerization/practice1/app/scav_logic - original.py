import os
import subprocess
import numpy as np
from scipy.fftpack import dct, idct
import pywt

class ColorTranslator:
    def rgb_to_yuv(self, r, g, b):
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = 0.492 * (b - y)
        v = 0.877 * (r - y)
        return y, u, v

    def yuv_to_rgb(self, y, u, v):
        r = y + 1.13983 * v
        g = y - 0.39465 * u - 0.58060 * v
        b = y + 2.03211 * u
        return (max(0, min(255, int(r))), 
                max(0, min(255, int(g))), 
                max(0, min(255, int(b))))

class ImageEncoder:
    """
    Nota: Aquesta classe requereix tenir FFmpeg instal·lat al sistema 
    (o al contenidor Docker) per funcionar.
    """
    def resize_image(self, input_path, output_path, width=None, height=None):
        if width and not height:
            scale_str = f"{width}:-2"
        elif height and not width:
            scale_str = f"-2:{height}"
        elif width and height:
            scale_str = f"{width}:{height}"
        else:
            scale_str = "iw/2:ih/2"

        cmd = [
            'ffmpeg', '-y', '-i', input_path, 
            '-vf', f'scale={scale_str}', output_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def compress_bw(self, input_path, output_path):
        cmd = [
            'ffmpeg', '-y', '-i', input_path, 
            '-vf', 'format=gray', '-q:v', '31', output_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

class SerpentineScanner:
    def serpentine(self, matrix):
        if matrix is None or len(matrix) == 0 or len(matrix[0]) == 0:
            raise ValueError("La matriu d'entrada està buida")
        
        matrix = np.array(matrix) # Assegurem que és numpy array
        rows, cols = matrix.shape
        num_diagonals = rows + cols - 1
        solution = [[] for _ in range(num_diagonals)]
        
        for i in range(rows):
            for j in range(cols):
                sum_indices = i + j
                if (sum_indices % 2 == 0):
                    solution[sum_indices].insert(0, matrix[i][j])
                else:
                    solution[sum_indices].append(matrix[i][j])
        
        result = []
        for diagonal in solution:
            result.extend(diagonal)
        return np.array(result)

class RLEConverter:
    def encode(self, data_bytes):
        if not isinstance(data_bytes, (list, tuple, np.ndarray)):
             # Convertim a llista si ve de numpy
            if isinstance(data_bytes, np.ndarray):
                data_bytes = data_bytes.tolist()
            else:
                raise TypeError("Input must be list, tuple or numpy array")
        
        if not data_bytes:
            return []
            
        encoded = []
        count = 1
        prev = data_bytes[0]
        
        for i in range(1, len(data_bytes)):
            if data_bytes[i] == prev:
                count += 1
            else:
                encoded.append((prev, count))
                prev = data_bytes[i]
                count = 1
        encoded.append((prev, count))
        return encoded

class DCTConverter:
    def forward(self, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def inverse(self, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

class DWTConverter:
    def forward(self, data):
        coeffs = pywt.dwt2(data, 'haar')
        return coeffs # Retorna (LL, (LH, HL, HH))

    def inverse(self, coeffs):
        return pywt.idwt2(coeffs, 'haar')