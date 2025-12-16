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
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = 0.492 * (b - y)
        v = 0.877 * (r - y)
        return y, u, v

    def yuv_to_rgb(self, y, u, v):
        r = y + 1.13983 * v
        g = y - 0.39465 * u - 0.58060 * v
        b = y + 2.03211 * u
        return (max(0, min(255, int(r))), max(0, min(255, int(g))), max(0, min(255, int(b))))

class RLEConverter:
    def encode(self, data_bytes):
        if not isinstance(data_bytes, (list, tuple, np.ndarray)):
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

class SerpentineScanner:
    def serpentine(self, matrix):
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
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    def inverse(self, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

class DWTConverter:
    def forward(self, data):
        coeffs = pywt.dwt2(data, 'haar')
        return coeffs 
    def inverse(self, coeffs):
        return pywt.idwt2(coeffs, 'haar')

# ===========================================================================
# GESTIÓ DOCKER (Connexió amb el contenidor FFmpeg)
# ===========================================================================

class FFmpegExecutor:
    def __init__(self):
        self.client = docker.from_env()
        self.container_name = "ffmpeg-service"

    def _run_ffmpeg_in_docker(self, cmd_list):
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

# ===========================================================================
# SEMINARI 2 (ImageEncoder & VideoProcessor)
# ===========================================================================

class ImageEncoder(FFmpegExecutor):
    def resize_image(self, input_filename, output_filename, width=None, height=None):
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"
        scale = f"{width if width else -2}:{height if height else -2}"
        if not width and not height: scale = "iw/2:ih/2"
        self._run_ffmpeg_in_docker(['ffmpeg', '-y', '-i', input_path, '-vf', f'scale={scale}', output_path])

    def compress_bw(self, input_filename, output_filename):
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"
        self._run_ffmpeg_in_docker(['ffmpeg', '-y', '-i', input_path, '-vf', 'format=gray', '-q:v', '31', output_path])

class VideoProcessor(FFmpegExecutor):
    def resize_video(self, input_filename, output_filename, width, height):
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"
        w_val = width if width else -2
        h_val = height if height else -2
        cmd = ['ffmpeg', '-y', '-i', input_path, '-vf', f'scale={w_val}:{h_val}', '-c:a', 'copy', output_path]
        self._run_ffmpeg_in_docker(cmd)

    def change_chroma_subsampling(self, input_filename, output_filename, pixel_format):
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"
        cmd = ['ffmpeg', '-y', '-i', input_path, '-c:v', 'libx264', '-pix_fmt', pixel_format, '-c:a', 'copy', output_path]
        self._run_ffmpeg_in_docker(cmd)

    def get_video_info(self, input_filename):
        input_path = f"/shared/{input_filename}"
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', input_path]
        res = self._run_ffmpeg_in_docker(cmd)
        data = json.loads(res)
        vid = next((s for s in data.get('streams',[]) if s['codec_type']=='video'), {})
        return {
            "filename": input_filename,
            "duration": float(data['format'].get('duration', 0)),
            "codec": vid.get('codec_name', 'N/A'),
            "resolution": f"{vid.get('width')}x{vid.get('height')}"
        }

    def process_bbb_container(self, input_filename, output_filename):
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"
        cmd = [
            'ffmpeg', '-y', '-i', input_path, '-t', '20',
            '-map', '0:v', '-c:v', 'copy',
            '-map', '0:a', '-c:a:0', 'aac', '-ac:a:0', '1',
            '-map', '0:a', '-c:a:1', 'libmp3lame', '-b:a:1', '64k', '-ac:a:1', '2',
            '-map', '0:a', '-c:a:2', 'ac3',
            output_path
        ]
        self._run_ffmpeg_in_docker(cmd)

    def count_tracks(self, input_filename):
        input_path = f"/shared/{input_filename}"
        res = self._run_ffmpeg_in_docker(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', input_path])
        data = json.loads(res)
        return [{"index": s['index'], "type": s['codec_type'], "codec": s.get('codec_name')} for s in data.get('streams', [])]

    def visualize_motion_vectors(self, input_filename, output_filename):
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"
        cmd = ['ffmpeg', '-y', '-flags2', '+export_mvs', '-i', input_path, '-vf', 'codecview=mv=pf+bf+bb', output_path]
        self._run_ffmpeg_in_docker(cmd)

    def yuv_histogram(self, input_filename, output_filename):
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"
        cmd = ['ffmpeg', '-y', '-i', input_path, '-vf', "split=2[a][b],[b]histogram,format=yuva444p[hh],[a][hh]overlay", '-c:a', 'copy', output_path]
        self._run_ffmpeg_in_docker(cmd)

# ===========================================================================
# PRÀCTICA 2 - MONSTER TRANSCODER
# ===========================================================================

class MonsterTranscoder(VideoProcessor):
    def convert_to_codec(self, input_filename, output_filename, codec):
        input_path = f"/shared/{input_filename}"
        output_path = f"/shared/{output_filename}"
        cmd = ['ffmpeg', '-y', '-i', input_path]
        if codec == 'vp8': cmd += ['-c:v', 'libvpx', '-b:v', '1M', '-c:a', 'libvorbis', output_path]
        elif codec == 'vp9': cmd += ['-c:v', 'libvpx-vp9', '-b:v', '2M', output_path]
        elif codec == 'h265': cmd += ['-c:v', 'libx265', '-crf', '28', output_path]
        elif codec == 'av1': cmd += ['-c:v', 'libaom-av1', '-crf', '30', '-b:v', '0', '-cpu-used', '8', '-strict', 'experimental', output_path]
        else: raise ValueError(f"Còdec {codec} no suportat.")
        print(f"Transcodificant a {codec}...")
        self._run_ffmpeg_in_docker(cmd)

    def create_encoding_ladder(self, input_filename):
        resolutions = [("1080p", 1920, 1080), ("720p", 1280, 720), ("480p", 854, 480)]
        files = []
        for name, w, h in resolutions:
            out_name = f"ladder_{name}_{input_filename}"
            print(f"Generant esglaó {name} amb herència...")
            self.resize_video(input_filename, out_name, w, h)
            files.append(out_name)
        return files