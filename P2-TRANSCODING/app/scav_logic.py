import os
import numpy as np
from scipy.fftpack import dct, idct
import pywt
import docker
import json

# ===========================================================================
# CLASSES AUXILIARS (PRÀCTICA 1)
# ===========================================================================

class ColorTranslator:
    def rgb_to_yuv(self, r, g, b):
        # Conversió estàndard de RGB a YUV
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = 0.492 * (b - y)
        v = 0.877 * (r - y)
        return y, u, v

    def yuv_to_rgb(self, y, u, v):
        # Conversió inversa de YUV a RGB
        r = y + 1.13983 * v
        g = y - 0.39465 * u - 0.58060 * v
        b = y + 2.03211 * u
        # Assegurem que els valors estiguin entre 0 i 255
        return (max(0, min(255, int(r))), max(0, min(255, int(g))), max(0, min(255, int(b))))

class RLEConverter:
    def encode(self, data_bytes):
        # Algorisme Run-Length Encoding per compressió simple
        if not isinstance(data_bytes, (list, tuple, np.ndarray)):
            if isinstance(data_bytes, np.ndarray):
                data_bytes = data_bytes.tolist()
            else:
                raise TypeError("L'entrada ha de ser llista, tupla o array numpy")
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

class SerpentineScanner:
    def serpentine(self, matrix):
        # Escaneig en serpentina (zigzag) d'una matriu
        if matrix is None or len(matrix) == 0:
            raise ValueError("La matriu d'entrada està buida")
        matrix = np.array(matrix)
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

class DCTConverter:
    def forward(self, block):
        # Transformada de Cosinus Discreta (DCT)
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def inverse(self, block):
        # DCT Inversa
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

class DWTConverter:
    def forward(self, data):
        # Transformada Wavelet Discreta (Haar)
        coeffs = pywt.dwt2(data, 'haar')
        return coeffs 
    
    def inverse(self, coeffs):
        # DWT Inversa
        return pywt.idwt2(coeffs, 'haar')

# ===========================================================================
# GESTIÓ DOCKER (OPTIMITZADA)
# ===========================================================================

class FFmpegExecutor:
    # Variable de classe per reutilitzar la connexió (Singleton Pattern)
    _client = None

    def __init__(self):
        # Només connectem a Docker si no ho hem fet abans
        if FFmpegExecutor._client is None:
            try:
                FFmpegExecutor._client = docker.from_env()
            except Exception as e:
                print(f"[AVÍS] No s'ha pogut connectar a Docker: {e}")
        
        self.client = FFmpegExecutor._client
        self.container_name = "ffmpeg-service"

    def _run_ffmpeg_in_docker(self, cmd_list):
        # Executa una comanda dins del contenidor FFmpeg
        try:
            container = self.client.containers.get(self.container_name)
            cmd_str = " ".join(cmd_list)
            print(f"[DOCKER] Executant: {cmd_str}")
            
            exec_result = container.exec_run(cmd_str)
            
            if exec_result.exit_code != 0:
                raise Exception(f"Error FFmpeg: {exec_result.output.decode()}")
            return exec_result.output
        except docker.errors.NotFound:
            raise Exception("El contenidor 'ffmpeg-service' no està actiu.")
        except AttributeError:
            raise Exception("El client de Docker no està inicialitzat.")

# ===========================================================================
# SEMINARI 2 (ImageEncoder & VideoProcessor)
# ===========================================================================

class ImageEncoder(FFmpegExecutor):
    def resize_image(self, input_filename, output_filename, width=None, height=None):
        # Redimensiona una imatge
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"
        
        # Si no s'especifica mida, reduïm a la meitat
        scale = f"{width if width else -2}:{height if height else -2}"
        if not width and not height: scale = "iw/2:ih/2"
        
        self._run_ffmpeg_in_docker(['ffmpeg', '-y', '-i', input_path, '-vf', f'scale={scale}', output_path])

    def compress_bw(self, input_filename, output_filename):
        # Converteix a blanc i negre i comprimeix
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"
        self._run_ffmpeg_in_docker(['ffmpeg', '-y', '-i', input_path, '-vf', 'format=gray', '-q:v', '31', output_path])

class VideoProcessor(FFmpegExecutor):
    def resize_video(self, input_filename, output_filename, width, height):
        # Redimensiona un vídeo (mètode base per a herència)
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"
        w_val = width if width else -2
        h_val = height if height else -2
        
        cmd = ['ffmpeg', '-y', '-i', input_path, '-vf', f'scale={w_val}:{h_val}', '-c:a', 'copy', output_path]
        self._run_ffmpeg_in_docker(cmd)

    def change_chroma_subsampling(self, input_filename, output_filename, pixel_format):
        # Canvia el submostreig de croma (ex: yuv420p, yuv422p)
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"
        cmd = ['ffmpeg', '-y', '-i', input_path, '-c:v', 'libx264', '-pix_fmt', pixel_format, '-c:a', 'copy', output_path]
        self._run_ffmpeg_in_docker(cmd)

    def get_video_info(self, input_filename):
        # Obté metadades del vídeo usant ffprobe
        input_path = f"/shared/{input_filename}"
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', input_path]
        res = self._run_ffmpeg_in_docker(cmd)
        data = json.loads(res)
        
        # Busquem el stream de vídeo
        vid = next((s for s in data.get('streams',[]) if s['codec_type']=='video'), {})
        return {
            "filename": input_filename,
            "duration": float(data['format'].get('duration', 0)),
            "codec": vid.get('codec_name', 'N/A'),
            "resolution": f"{vid.get('width')}x{vid.get('height')}"
        }

    def process_bbb_container(self, input_filename, output_filename):
        # Crea un contenidor complex amb múltiples pistes d'àudio (Exemple BBB)
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"
        cmd = [
            'ffmpeg', '-y', '-i', input_path, '-t', '20',
            '-map', '0:v', '-c:v', 'copy', # Copiem vídeo
            '-map', '0:a', '-c:a:0', 'aac', '-ac:a:0', '1', # Audio 1: AAC Mono
            '-map', '0:a', '-c:a:1', 'libmp3lame', '-b:a:1', '64k', '-ac:a:1', '2', # Audio 2: MP3 Stereo baix bitrate
            '-map', '0:a', '-c:a:2', 'ac3', # Audio 3: AC3
            output_path
        ]
        self._run_ffmpeg_in_docker(cmd)

    def count_tracks(self, input_filename):
        # Compta quantes pistes (streams) té el fitxer
        input_path = f"/shared/{input_filename}"
        res = self._run_ffmpeg_in_docker(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', input_path])
        data = json.loads(res)
        return [{"index": s['index'], "type": s['codec_type'], "codec": s.get('codec_name')} for s in data.get('streams', [])]

    def visualize_motion_vectors(self, input_filename, output_filename):
        # Genera un vídeo amb els vectors de moviment visualitzats
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"
        cmd = ['ffmpeg', '-y', '-flags2', '+export_mvs', '-i', input_path, '-vf', 'codecview=mv=pf+bf+bb', output_path]
        self._run_ffmpeg_in_docker(cmd)

    def yuv_histogram(self, input_filename, output_filename):
        # Superposa un histograma YUV al vídeo
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"
        # Filtre complex: divideix el vídeo, genera histograma i el superposa
        cmd = ['ffmpeg', '-y', '-i', input_path, '-vf', "split=2[a][b],[b]histogram,format=yuva444p[hh],[a][hh]overlay", '-c:a', 'copy', output_path]
        self._run_ffmpeg_in_docker(cmd)

# ===========================================================================
# PRÀCTICA 2 - MONSTER TRANSCODER
# ===========================================================================

class MonsterTranscoder(VideoProcessor):
    def convert_to_codec(self, input_filename, output_filename, codec):
        # Converteix el vídeo als còdecs demanats a la Pràctica 2
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"
        
        cmd = ['ffmpeg', '-y', '-i', input_path]
        
        # Configuració específica per a cada còdec
        if codec == 'vp8': 
            cmd += ['-c:v', 'libvpx', '-b:v', '1M', '-c:a', 'libvorbis', output_path]
        elif codec == 'vp9': 
            cmd += ['-c:v', 'libvpx-vp9', '-b:v', '2M', output_path]
        elif codec == 'h265': 
            cmd += ['-c:v', 'libx265', '-crf', '28', output_path]
        elif codec == 'av1': 
            # AV1 és lent, usem -cpu-used 8 per accelerar proves
            cmd += ['-c:v', 'libaom-av1', '-crf', '30', '-b:v', '0', '-cpu-used', '8', '-strict', 'experimental', output_path]
        else: 
            raise ValueError(f"Còdec {codec} no suportat.")
        
        print(f"Transcodificant a {codec}...")
        self._run_ffmpeg_in_docker(cmd)

    def create_encoding_ladder(self, input_filename):
        # Genera 3 versions del vídeo (Encoding Ladder)
        # Fent ús d'herència cridant al mètode del pare (VideoProcessor)
        resolutions = [("1080p", 1920, 1080), ("720p", 1280, 720), ("480p", 854, 480)]
        files = []
        
        for name, w, h in resolutions:
            out_name = f"ladder_{name}_{input_filename}"
            print(f"Generant esglaó {name} ({w}x{h})...")
            # Cridem al mètode heretat de la classe pare
            self.resize_video(input_filename, out_name, w, h)
            files.append(out_name)
            
        return files