import os
import subprocess
import numpy as np
from scipy.fftpack import dct, idct
import pywt
import docker
import json


# Adaptació del Seminari 1 
# Hem fet ús de la IA per adaptar el codi
# Ha estat revisat i modificat per nosaltres posteriorment

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
    
###################### - S2-MPEG4_and_MoreEndpoints - ######################

# Separem la lògica que es comunica amb ffmpeg per poder tractar de manera més
# ordenada tant imatges com vídeos

class FFmpegExecutor:
    """
    Classe "base" que gestionarà la connexió amb el contenidor de FFmpeg
    """
    def __init__(self):
        self.client = docker.from_env()
        self.container_name = "ffmpeg-service"

    def _run_ffmpeg_in_docker(self, cmd_list):
        try:
            container = self.client.containers.get(self.container_name)
            cmd_str = " ".join(cmd_list)
            # Executem la comanda dins del contenidor ffmpeg-service
            exec_result = container.exec_run(cmd_str)
            
            if exec_result.exit_code != 0:
                raise Exception(f"FFmpeg Error: {exec_result.output.decode()}")
            return exec_result.output
        except docker.errors.NotFound:
            raise Exception("El contenidor d'FFmpeg no està actiu!")

class ImageEncoder(FFmpegExecutor):
    """
    Lògica de l'encoder d'imatges (del seminari 1, permet redimensionar)
    """
    def resize_image(self, input_filename, output_filename, width=None, height=None):
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"

        if width and not height:
            scale_str = f"{width}:-2" # Manté aspect ratio
        elif height and not width:
            scale_str = f"-2:{height}" # Manté aspect ratio
        elif width and height:
            scale_str = f"{width}:{height}" # Força dimensions
        else:
            scale_str = "iw/2:ih/2" # La meitat per defecte

        cmd = [
            'ffmpeg', '-y', '-i', input_path, 
            '-vf', f'scale={scale_str}', output_path
        ]
        self._run_ffmpeg_in_docker(cmd)

    def compress_bw(self, input_filename, output_filename):
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"
        cmd = [
            'ffmpeg', '-y', '-i', input_path, 
            '-vf', 'format=gray', '-q:v', '31', output_path
        ]
        self._run_ffmpeg_in_docker(cmd)

# CLASSE NOVA SEMINARI 2 PER MODIFICAR LA RESOLUCIÓ D'UN VÍDEO (TASK 1)
class VideoProcessor(FFmpegExecutor):

    # ---------------------------------------------------------------------------
    # SEMINARI 2 - TASK 1: REDIMENSIONAR VÍDEO (Big Buck Bunny)
    # ---------------------------------------------------------------------------
    def resize_video(self, input_filename, output_filename, width, height):
        """
        Gestiona la resolució d'un vídeo fent servir FFmpeg des del docker
        """
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"

        # Definim l'escala. -2 per mantenir l'aspect ratio en cas de no tenir tots els paràmetres
        w_val = width if width else -2
        h_val = height if height else -2
        
        scale_str = f"{w_val}:{h_val}"

        print(f"Processant vídeo: {scale_str}")

        cmd = [
            'ffmpeg', '-y', 
            '-i', input_path, 
            '-vf', f'scale={scale_str}', 
            '-c:a', 'copy', # Copiem l'àudio sense recodificar
            output_path
        ]
        self._run_ffmpeg_in_docker(cmd)

    # ---------------------------------------------------------------------------
    # SEMINARI 2 - TASK 2: MODIFICAR EL SUBMOSTREIG DE CROMA
    # ---------------------------------------------------------------------------
    def change_chroma_subsampling(self, input_filename, output_filename, pixel_format):
        """
        Canvia el submostreig de croma (Chroma Subsampling) d'un vídeo
        
        Paràmetres:
        - pixel_format: 'yuv420p'/ 'yuv422p'/ 'yuv444p'
        """
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"

        print(f"Canviant chroma subsampling a: {pixel_format}")

        cmd = [
            'ffmpeg', '-y', 
            '-i', input_path,
            '-c:v', 'libx264',   # Recodifiquem amb H.264
            '-pix_fmt', pixel_format, # Apliquem el nou format de píxel
            '-c:a', 'copy',      # Copiem l'àudio
            output_path
        ]
        
        self._run_ffmpeg_in_docker(cmd)
    
    # ---------------------------------------------------------------------------
    # SEMINARI 2 - TASK 3: VIDEO INFO (ffprobe)
    # ---------------------------------------------------------------------------
    def get_video_info(self, input_filename):
        """
        Busquem metadades del video fent servir la comanda ffprobe
        """
        input_path = f"/shared/{input_filename}"
        
        # Comanda ffprobe per obtenir sortida en JSON
        # -v quiet: No mostra logs innecessaris
        # -print_format json: Sortida en JSON
        # -show_streams: Mostra informació de cada stream (video, audio)
        # -show_format: Mostra informació general del contenidor
        cmd = [
            'ffprobe', 
            '-v', 'quiet', 
            '-print_format', 'json', 
            '-show_format', 
            '-show_streams', 
            input_path
        ]
        
        # Executem la comanda i capturem la sortida --> string JSON
        json_output = self._run_ffmpeg_in_docker(cmd)
        
        # Convertim l'string JSON a un diccionari Python
        try:
            data = json.loads(json_output)
            
            # Busquem l'stream de vídeo
            video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
            audio_stream = next((s for s in data['streams'] if s['codec_type'] == 'audio'), None)
            
            if not video_stream:
                raise Exception("No s'ha trobat cap stream de vídeo.")

            # Extraiem 5+ dades rellevants
            info = {
                "filename": input_filename,
                "container_format": data['format'].get('format_name', 'N/A'),
                "duration_seconds": float(data['format'].get('duration', 0)),
                "video_codec": video_stream.get('codec_name', 'N/A'),
                "resolution": f"{video_stream.get('width')}x{video_stream.get('height')}",
                "frame_rate": video_stream.get('r_frame_rate', 'N/A'),
                "bitrate": f"{int(data['format'].get('bit_rate', 0)) / 1000} kbps",
                "audio_codec": audio_stream.get('codec_name', 'None') if audio_stream else "No Audio"
            }
            return info
            
        except json.JSONDecodeError:
            raise Exception("Error interpretant la sortida de ffprobe")
        
    # ---------------------------------------------------------------------------
    # SEMINARI 2 - TASK 4: BBB CONTAINER
    # ---------------------------------------------------------------------------

    def process_bbb_container(self, input_filename, output_filename):
        """
        Crea un contenidor MP4 amb:
        - Vídeo tallat a 20s
        - Àudio 1: AAC Mono
        - Àudio 2: MP3 Stereo (Low Bitrate)
        - Àudio 3: AC3
        """
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"

        print(f"Creant contenidor BBB...")

        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            
            # --- TALL TEMPORAL ---
            '-t', '20', # Durada de 20 segons
            
            # --- VIDEO STREAM (Stream 0) ---
            '-map', '0:v',      # Vídeo de l'entrada
            '-c:v', 'copy',     # Copiem el vídeo sense recodificar
            
            # --- AUDIO STREAM 1 (Stream 1): AAC Mono ---
            '-map', '0:a',      # Agafem l'àudio de l'entrada
            '-c:a:0', 'aac',    # Codec AAC
            '-ac:a:0', '1',     # Canals --> 1
            
            # --- AUDIO STREAM 2 (Stream 2): MP3 Stereo Low Bitrate ---
            '-map', '0:a',            # Agafem l'àudio de l'entrada
            '-c:a:1', 'libmp3lame',   # Codec MP3
            '-b:a:1', '64k',          # Bitrate baix (64k)
            '-ac:a:1', '2',           # Canals --> 2
            
            # --- AUDIO STREAM 3 (Stream #3): AC3 ---
            '-map', '0:a',      # Agafem l'àudio de l'entrada
            '-c:a:2', 'ac3',    # Codec AC3
            
            output_path
        ]
        
        self._run_ffmpeg_in_docker(cmd)

    # ---------------------------------------------------------------------------
    # SEMINARI 2 - TASK 5: COUNT TRACKS
    # ---------------------------------------------------------------------------
    def count_tracks(self, input_filename):
        """
        Retorna el nombre de pistes d'un contenidor i què és cadascuna de les pistes
        """
        input_path = f"/shared/{input_filename}"
        
        # Busquem els streams en format JSON fent servir ffprobe
        cmd = [
            'ffprobe', 
            '-v', 'quiet', 
            '-print_format', 'json', 
            '-show_streams', 
            input_path
        ]
        
        json_output = self._run_ffmpeg_in_docker(cmd)
        
        try:
            data = json.loads(json_output)
            streams = data.get('streams', [])
            
            # Nombre de pistes
            count = len(streams)
            
            # Resum per saber què és cadascuna de les pistes
            details = []
            for stream in streams:
                index = stream.get('index')
                codec_type = stream.get('codec_type') # video, audio, subtitle
                codec_name = stream.get('codec_name') # codec
                details.append(f"Track {index}: {codec_type} ({codec_name})")
            
            return {
                "total_tracks": count,
                "breakdown": details
            }
            
        except json.JSONDecodeError:
            raise Exception("Error llegint les dades del contenidor")
        
        
    # ---------------------------------------------------------------------------
    # SEMINARI 2 - TASK 6: MACROBLOCKS AND MOTION VECTORS
    # ---------------------------------------------------------------------------

    def visualize_motion_vectors(self, input_filename, output_filename):
        """
        Genera un vídeo amb els vectors de moviment (Motion Vectors)
        """
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"

        print(f"Generant visualització de vectors de moviment...")

        cmd = [
            'ffmpeg', 
            '-y',
            # Activar la exportació de vectors de moviment al decoder
            '-flags2', '+export_mvs', 
            '-i', input_path,
            
            # Filtre 'codecview' per veure els vectors de moviment
            # mv=pf+bf+bb: 
            # pf = P-frames forward
            # bf = B-frames forward
            # bb = B-frames backward
            '-vf', 'codecview=mv=pf+bf+bb', 
            
            output_path
        ]
        
        self._run_ffmpeg_in_docker(cmd)

    
    # ---------------------------------------------------------------------------
    # SEMINARI 2 - TASK 7: YUV HISTOGRAM
    # ---------------------------------------------------------------------------

    def yuv_histogram(self, input_filename, output_filename):
        """
        Genera un vídeo amb l'histograma YUV
        """
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"

        print(f"Generant histograma YUV...")

        cmd = [
            'ffmpeg',
            '-y',
            '-i', input_path,
            
            # Filtre per superposar l'histograma al vídeo original
            '-vf', "split=2[a][b],[b]histogram,format=yuva444p[hh],[a][hh]overlay",
            
            '-c:a', 'copy', # Mantenim l'àudio
            output_path
        ]
        
        self._run_ffmpeg_in_docker(cmd)