import os
import subprocess
import numpy as np
from scipy.fftpack import dct, idct
import unittest
import pywt
from unittest.mock import MagicMock
import sys

# ----------------------------------------------------------------
# UTILITATS VISUALS (Colors per la terminal)
# ----------------------------------------------------------------
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}\n {text}\n{'='*60}{Colors.ENDC}")

def print_step(text):
    print(f"{Colors.OKCYAN}[PAS] {text}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.OKGREEN}[ÈXIT] {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKBLUE}[INFO] {text}{Colors.ENDC}")


TEST_IMAGES = ["test_image.jpg", "test_image2.jpg"]

# ----------------------------------------------------------------
# ----------------------------------------------------------------

"""
EXERCICI 2: Implementació de conversió de color.

ENUNCIAT:
----------------------------------------------------------------
Start a script called 'first_seminar.py'. Then create a class and a method, which is a
translator from 3 values in RGB into the 3 YUV values, plus the opposite operation. 
You can choose the 3 values, or open them from a text file, receive it from command 
line... feel free.  
----------------------------------------------------------------
"""
class ColorTranslator:
    def rgb_to_yuv(self, r, g, b):
        """
        Converteix valors RGB (0-255) a l'espai de color YUV.

        Aquesta implementació utilitza els coeficients estàndard (BT.601) per 
        calcular la luminància (Y) i les crominàncies (U, V).

        Referència
        ----------
        Fórmula (Simon Fraser University): 
        https://www.cs.sfu.ca/mmbook/programming_assignments/additional_notes/rgb_yuv_note/RGB-YUV.pdf
        
        Paràmetres
        ----------
        r : int o float
            Valor del component Vermell (Red) [0, 255].
        g : int o float
            Valor del component Verd (Green) [0, 255].
        b : int o float
            Valor del component Blau (Blue) [0, 255].

        Retorna
        -------
        tuple (float, float, float)
            Una tupla amb els valors convertits (y, u, v):
            - y: Luminància.
            - u: Component de crominància blava.
            - v: Component de crominància vermella.
        """
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = 0.492 * (b - y)
        v = 0.877 * (r - y)
        return y, u, v

    def yuv_to_rgb(self, y, u, v):
        """
        Converteix valors YUV a RGB.
        
        Resultats limitats a [0, 255] perque puguin ser valors RGB vàlids. 

        Paràmetres
        ----------
        y : float
            Valor de la luminància.
        u : float
            Component de crominància blava.
        v : float
            Component de crominància vermella.

        Retorna
        -------
        tuple (int, int, int)
            Una tupla amb els valors convertits (r, g, b):
            - r: Component Vermell [0, 255].
            - g: Component Verd [0, 255].
            - b: Component Blau [0, 255].
        """
        r = y + 1.13983 * v
        g = y - 0.39465 * u - 0.58060 * v
        b = y + 2.03211 * u
        
        # Clamping per aswsugurar-nos que son valors vàlids RGB.
        return (max(0, min(255, int(r))), 
                max(0, min(255, int(g))), 
                max(0, min(255, int(b))))




class ImageEncoder:

    """
    EXERCICI 3: Reduir la qualitat d'una imatge.

    ENUNCIAT:
    ----------------------------------------------------------------
    Use ffmpeg to resize images into lower quality. Use any image you like.
    Now, create a method in previous script to automatise this order.
    ----------------------------------------------------------------
    """

    def resize_image(self, input_path, output_path, width=None, height=None):
        """
        Redimensiona una imatge fent servir ffmpeg. 

        Hem decidit gestionar de manera automàtica la relació d'aspecte en els casos 
        en els que només haguem definit o alçada o amplada:
            - Només width: Calcula height automàticament mantenint la proporció.
            - Només height: Calcula width automàticament mantenint la proporció.
            - Cap: Redueix la imatge a la meitat de la seva mida original.
            - Width & height: Força les dimensions.

        Paràmetres
        ----------
        input_path : str
            Ruta de la imatge d'entrada
        output_path : str
            Ruta on es guardarà la imatge redimensionada.
        width : int, opcional
            Amplada en píxels. Per defecte és None.
        height : int, opcional
            Alçada en píxels. Per defecte és None.
        """
        
        # Lògica per definir l'escala mantenint la proporció

        if width and not height: # Cas només width
            # Alçada automàtica
            scale_str = f"{width}:-2"

        elif height and not width: # Cas només height
            # Amplada automàtica
            scale_str = f"-2:{height}"

        elif width and height: # Cas width & height
            # Definim la mida
            scale_str = f"{width}:{height}"

        else: # Cas en el que no s'indica res
            # Reduïm a la meitat tant width com height
            scale_str = "iw/2:ih/2"

        # FFMpeg
        cmd = [
            'ffmpeg', 
            '-y', # Sobreescriu l'arxiu de sortida
            '-i', input_path, # Ruta de la imatge d'entrada
            '-vf', f'scale={scale_str}', # Video filter per redimensionar
            output_path # Imatge de sortida
        ]
        
        print(f"Executant redimensionament amb escala '{scale_str}'...")
        
        # Executem les comandes a FFMpeg
        subprocess.run(cmd)


    """
    EXERCICI 5: Compressió d'una imatge, b/w fent servir FFMPeg

    ENUNCIAT:
    ----------------------------------------------------------------
    Use FFMPEG to transform the previous image into b/w. 
    Do the hardest compression you can.
    Add everything into a new method and comment the results.
    ----------------------------------------------------------------
    """

    def compress_bw(self, input_path, output_path):
        """
        Transforma una imathe a blanc i negre i aplica el nivell de compressió més alt possible. 

        Paràmetres
        ----------
        input_path : str
            Ruta de la imatge d'entrada.
        output_path : str
            Ruta on es guardarà la imatge comprimida en blanc i negre.
        """
        # Comanda FFmpeg amb els paràmetres de compressió
        cmd = [
            'ffmpeg', 
            '-y', # Sobreescriu l'arxiu de sortida
            '-i', input_path, # Ruta de la imatge d'entrada
            '-vf', 'format=gray', # Video filter per convertir a blanc i negre
            '-q:v', '31', # Factor de qualitat (qscale) per la imatge. Rang [1, 31].
            output_path # Imatge de sortida
        ]
        
        print(f"Executant compressió i conversió a blanc i negre...")
        
        # Executem les comandes a FFMpeg
        subprocess.run(cmd)



"""
EXERCICI 4: Llegir bytes seguint la forma de serpentí

ENUNCIAT:
----------------------------------------------------------------
Create a method called serpentine which should be able to read the bytes of a JPEG file in 
the serpentine way we saw.
----------------------------------------------------------------
"""

class SerpentineScanner:
    def serpentine(self, matrix):
        """
        Llegeix una matriu NxM seguint la forma de serpentí

        Paràmetres
        ----------
        matrix : numpy.ndarray
            Matriu d'entrada de dimensions (Rows x Cols).

        Retorna
        -------
        numpy.ndarray
            Array amb els bytes reordenats seguint el serpentí.
        """

        if matrix is None or len(matrix) == 0 or len(matrix[0]) == 0:
            raise ValueError("La matriu d'entrada està buida")
        
        # Dimensions de la matriu
        rows, cols = matrix.shape
        
        # Les diagonals son (files + columnes) - 1
        num_diagonals = rows + cols - 1
        solution = [[] for _ in range(num_diagonals)]
        
        for i in range(rows):
            for j in range(cols):
                # La suma (i + j) ens diu a quina diagonal pertany l'element
                sum_indices = i + j
                
                # Decidim l'ordre segons si les diagonals son parelles o senars

                if (sum_indices % 2 == 0):
                    # Diagonals parelles al principi (amunt dreta)
                    solution[sum_indices].insert(0, matrix[i][j])

                else:
                    # Diagonals imparelles al final (amunt esquerre)
                    solution[sum_indices].append(matrix[i][j])
        
        # Aplanem la llista de llistes en un únic vector
        result = []
        for diagonal in solution:
            result.extend(diagonal)
                
        return np.array(result)
    

"""
EXERCICI 6: Run-Length Encoding

ENUNCIAT:
----------------------------------------------------------------
Create a method which applies a run-lenght encoding from a series of bytes given.
----------------------------------------------------------------
"""

class RLEConverter:
    def encode(self, data_bytes):
        """
        Apliquem Run-Length Encoding (RLE) als bytes d'entrada. 

        Paràmetres
        ----------
        data_bytes : list o bytes
            Dades d'entrada

        Retorna
        -------
        list of tuples
            Llista de tuples on cada tupla conté (valor, quantitat).
        """

        if not isinstance(data_bytes, (list, tuple)):
            raise TypeError("Input to RLE must be a list or tuple")
        
        # Si la llista d'entrada està buida, retornem una llista buida
        if not data_bytes:
            return []
            
        encoded = []
        count = 1
        prev = data_bytes[0] # Inicialització
        
        # Iterem per cada element de la llista
        for i in range(1, len(data_bytes)):
            if data_bytes[i] == prev:
                # Si hi ha dos elements iguals incrementem el contador
                count += 1
            else:
                # Si l'element canvia, guardem el grup anterior
                encoded.append((prev, count))
                
                # Actualitzem al nou valor i reiniciem el contador
                prev = data_bytes[i]
                count = 1
        
        # Afegim l'últim grup després de sortir del bucle
        encoded.append((prev, count))
        
        return encoded

"""
EXERCICI 7: Encoder - Decoder fent ús de la DCT

ENUNCIAT:
----------------------------------------------------------------
Create a class which can convert, can decode (or both) an input using the DCT. Not necessary a 
JPG encoder or decoder. A class only about DCT is OK too.
----------------------------------------------------------------
"""

class DCTConverter:
    def forward(self, block):
        """
        Aplica la Transformada Discreta del Cosinus (DCT) a un bloc.

        Implementada a partir de la DCT 1D primer sobre les files i després sobre les columnes.

        Paràmetres
        ----------
        block : numpy.ndarray
            Bloc de pixels. 

        Retorna
        -------
        numpy.ndarray
            Bloc de la mateixa mida amb els coeficients de freqüència.
        """
        # 'block.T' --> Transposem per aplicar la DCT a les columnes
        # norm='ortho' --> Aplica la transformada normalitzada.
        # 3. '.T': Tornem a transposar.
        # 4. 'dct(...)': Apliquem la transformada a les files.
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def inverse(self, block):
        """
        Aplica la DCT Inversa (IDCT) en 2D per recuperar el bloc original.
        Mateixa implementació que la forward.

        Paràmetres
        ----------
        block : numpy.ndarray
            Bloc de coeficients de freqüència (DCT).

        Retorna
        -------
        numpy.ndarray
            Bloc reconstruït.
        """
        # 'block.T' --> Transposem per aplicar la DCT a les columnes
        # norm='ortho' --> Aplica la transformada inversa normalitzada.
        # 3. '.T': Tornem a transposar.
        # 4. 'idct(...)': Apliquem la transformada inversa a les files.
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

"""
EXERCICI 8: Encoder - Decoder fent ús de la DWT (Versió PyWavelets)

ENUNCIAT:
----------------------------------------------------------------
Create a class which can convert, can decode (or both) an input using the DWT. 
----------------------------------------------------------------
"""

class DWTConverter:
    """
    Implementació de la DWT fent servir la llibreria PyWavelets (pywt).
    """
    
    def forward(self, data):
        """
        Aplica DWT a les dades d'entrada fent servir la funció 'dwt2' de PyWavelets.
        
        Paràmetres
        ----------
        data : numpy.ndarray
            La imatge o bloc d'entrada (2D).

        Retorna
        -------
        tuple
            Una estructura de coeficients: (LL, (LH, HL, HH))
            - LL: Aproximació (Baixes freqüències).
            - (LH, HL, HH): Detalls (Horitzontal, Vertical, Diagonal).
        """
        # 'haar' és la família de Wavelets
        # pywt.dwt2 retorna els coeficients separats.
        coeffs = pywt.dwt2(data, 'haar')
        return coeffs

    def inverse(self, coeffs):
        """
        Aplica la DWT Inversa (IDCT) 2D fent servir la funció 'idwt2' de PyWavelets.

        Paràmetres
        ----------
        coeffs : tuple
            Els coeficients generats pel mètode forward: (LL, (LH, HL, HH)).

        Retorna
        -------
        numpy.ndarray
            La imatge reconstruïda.
        """
        return pywt.idwt2(coeffs, 'haar')



"""
EXERCICI 9: UNIT TESTS

ENUNCIAT:
----------------------------------------------------------------
Use any AI (YES, you can NOW, you lazy!) to create UNIT TESTS to your code, for each 
method and class.
If the code is too much poor, try to improve it a bit.
----------------------------------------------------------------
El codi a continuació ha estat parcialment generat amb IA (Google Gemini).
Revisió, codi final i modificacions fetes manualment. 
"""

class TestColorTranslator(unittest.TestCase):
    def setUp(self):
        # Configuració inicial per a cada test
        self.translator = ColorTranslator()

    def test_rgb_to_yuv_known(self):
        # Test de conversió RGB a YUV amb valors coneguts (Vermell pur)
        y, u, v = self.translator.rgb_to_yuv(255, 0, 0)
        self.assertAlmostEqual(y, 0.299*255, delta=1e-6)
        self.assertIsInstance(u, float)

    def test_yuv_to_rgb_roundtrip(self):
        # Test d'anada i tornada: RGB -> YUV -> RGB ha de donar el mateix valor
        rgb_samples = [(0, 0, 0), (255, 255, 255), (10, 120, 200)]
        for r, g, b in rgb_samples:
            y, u, v = self.translator.rgb_to_yuv(r, g, b)
            r2, g2, b2 = self.translator.yuv_to_rgb(y, u, v)
            # Permetem un marge d'error d'1 unitat degut a l'arrodoniment
            self.assertTrue(abs(r - r2) <= 1)
            self.assertTrue(abs(g - g2) <= 1)
            self.assertTrue(abs(b - b2) <= 1)

class TestImageEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = ImageEncoder()
        # Fem servir la llista global d'imatges
        self.test_images = TEST_IMAGES
        self.outdir = "test_outputs"

        # Creem el directori de sortida si no existeix
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Verificació prèvia d'existència d'imatges
        missing = [img for img in self.test_images if not os.path.exists(img)]
        if missing:
            print(f"{Colors.WARNING}ATENCIÓ: Falten les imatges: {missing}. Els tests fallaran.{Colors.ENDC}")

    def _check_and_report(self, output_path):
        """ Comprova si l'arxiu existeix i reporta error si no. """
        self.assertTrue(os.path.exists(output_path), f"L'arxiu {output_path} NO s'ha generat!")

    def _get_output_name(self, original_path, suffix):
        """ Funció auxiliar per generar noms únics basats en l'entrada """
        base = os.path.splitext(os.path.basename(original_path))[0]
        return os.path.join(self.outdir, f"{base}_{suffix}")

    def test_resize_only_width_real(self):
        print(f"\n{Colors.OKBLUE}[Prova] Redimensionar Amplada...{Colors.ENDC}")
        for img_path in self.test_images:
            output = self._get_output_name(img_path, "width_320.jpg")
            self.encoder.resize_image(img_path, output, width=320)
            self._check_and_report(output)

    def test_resize_only_height_real(self):
        print(f"\n{Colors.OKBLUE}[Prova] Redimensionar Alçada...{Colors.ENDC}")
        for img_path in self.test_images:
            output = self._get_output_name(img_path, "height_240.jpg")
            self.encoder.resize_image(img_path, output, height=240)
            self._check_and_report(output)

    def test_resize_both_real(self):
        print(f"\n{Colors.OKBLUE}[Prova] Redimensionar 100x200...{Colors.ENDC}")
        for img_path in self.test_images:
            output = self._get_output_name(img_path, "100x200.jpg")
            self.encoder.resize_image(img_path, output, width=100, height=200)
            self._check_and_report(output)

    def test_resize_auto_half_real(self):
        print(f"\n{Colors.OKBLUE}[Prova] Redimensionar a la meitat...{Colors.ENDC}")
        for img_path in self.test_images:
            output = self._get_output_name(img_path, "half.jpg")
            self.encoder.resize_image(img_path, output)
            self._check_and_report(output)

    def test_compress_bw_real(self):
        print(f"\n{Colors.OKBLUE}[Prova] Compressió B/N...{Colors.ENDC}")
        for img_path in self.test_images:
            output = self._get_output_name(img_path, "bw_compressed.jpg")
            self.encoder.compress_bw(img_path, output)
            self._check_and_report(output)


class TestSerpentineScanner(unittest.TestCase):
    def setUp(self):
        self.scanner = SerpentineScanner()

    def test_serpentine_square(self):
        # Test amb una matriu quadrada simple 3x3
        mat = np.array([[1,2,3],
                        [4,5,6],
                        [7,8,9]])
        res = self.scanner.serpentine(mat)
        expected = np.array([1,2,4,7,5,3,6,8,9])
        np.testing.assert_array_equal(res, expected)

    def test_serpentine_empty_raises(self):
        # Verificar que llença un error amb matrius buides
        with self.assertRaises(ValueError):
            empty = np.array([]).reshape((0,0))
            self.scanner.serpentine(empty)


def image_to_matrix_ffmpeg(path):
    """ Converteix imatge -> matriu (gray8) usant ffmpeg. """
    # Obtenir mida de la imatge
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            stdout=subprocess.PIPE, text=True
        )
        if probe.returncode != 0: return None
        w, h = map(int, probe.stdout.strip().split("\n"))

        # Extreure bytes RAW
        raw = subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout

        pixels = list(raw)
        # Convertir llista plana a matriu 2D
        matrix = [pixels[i*w:(i+1)*w] for i in range(h)]
        return np.array(matrix, dtype=np.uint8)
    except Exception as e:
        return None

class TestSerpentineRealImages(unittest.TestCase):
    def setUp(self):
        self.images = TEST_IMAGES
        self.outdir = "test_outputs_serpentine"
        os.makedirs(self.outdir, exist_ok=True)
        self.scanner = SerpentineScanner()

    def test_serpentine_real_images(self):
        print(f"\n{Colors.OKBLUE}[Prova] Escaneig Serpentí (Imatges Reals)...{Colors.ENDC}")
        for img in self.images:
            if not os.path.exists(img):
                print(f"{Colors.WARNING}Saltant {img} (no existeix){Colors.ENDC}")
                continue
                
            mat = image_to_matrix_ffmpeg(img)
            if mat is None:
                print(f"{Colors.FAIL}Error llegint {img}{Colors.ENDC}")
                continue

            serp = self.scanner.serpentine(mat)
            self.assertEqual(len(serp), mat.size)

            name = os.path.splitext(os.path.basename(img))[0]
            # CANVI: Nom del fitxer indicant que només son els primers 500
            out_path = os.path.join(self.outdir, f"{name}_serpentine_first500.txt")
            
            with open(out_path, "w") as f:
                f.write(", ".join(map(str, serp[:500]))) 
            
            print(f"  > Serpentí generat (primers 500) per {name} a {out_path}")


class TestRLEConverter(unittest.TestCase):
    def setUp(self):
        self.rle = RLEConverter()

    def test_rle_empty(self):
        # Test amb llista buida
        self.assertEqual(self.rle.encode([]), [])

    def test_rle_single(self):
        # Test amb un sol valor
        self.assertEqual(self.rle.encode([5]), [(5,1)])

    def test_rle_alternating(self):
        # Test amb valors alterns
        self.assertEqual(self.rle.encode([1,2,1,2,1]), [(1,1),(2,1),(1,1),(2,1),(1,1)])


class TestDCTConverter(unittest.TestCase):
    def setUp(self):
        self.conv = DCTConverter()

    def test_dct_inverse_roundtrip_random(self):
        # Test de transformació inversa amb dades aleatòries
        rng = np.random.RandomState(0)
        block = rng.randint(0, 256, size=(8,8)).astype(float)
        coeffs = self.conv.forward(block)
        recon = self.conv.inverse(coeffs)
        np.testing.assert_allclose(recon, block, atol=1e-6)

class TestDWTConverter(unittest.TestCase):
    def setUp(self):
        self.conv = DWTConverter()

    def test_dwt_inverse_roundtrip(self):
        # Test de transformació inversa Wavelet
        rng = np.random.RandomState(1)
        data = rng.randn(32,32)
        coeffs = self.conv.forward(data)
        recon = self.conv.inverse(coeffs)
        self.assertEqual(recon.shape, data.shape)
        np.testing.assert_allclose(recon, data, atol=1e-6)


# =========================
# Execució Visual dels Tests
# =========================

def run_tests_visual():
    print_header("EXERCICI 9: Unit tests automàtics - Primer Seminari")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Afegim tots els casos de prova
    test_cases = [
        TestColorTranslator,
        TestImageEncoder,
        TestSerpentineScanner,
        TestSerpentineRealImages,
        TestRLEConverter,
        TestDCTConverter,
        TestDWTConverter
    ]

    for tc in test_cases:
        print_step(f"Afegint proves: {tc.__name__}")
        suite.addTests(loader.loadTestsFromTestCase(tc))

    print_info("Executant la suite de tests amb sortida detallada...\n")
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    # Resum final molt visual
    total = result.testsRun
    fails = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total - fails - errors - skipped

    print_header("RESUM DELS TESTS")
    print_info(f"Proves executades : {total}")
    print_info(f"Proves aprovades  : {passed}")
    if skipped:
        print_info(f"Proves saltades   : {skipped}")
    if fails or errors:
        print(f"{Colors.FAIL}Proves fallades   : {fails}{Colors.ENDC}")
        print(f"{Colors.FAIL}Errors            : {errors}{Colors.ENDC}")
        sys.exit(1)
    else:
        print_success("TOTS ELS TESTS HAN PASSAT AMB ÈXIT 🎉")
        sys.exit(0)

if __name__ == '__main__':
    run_tests_visual()