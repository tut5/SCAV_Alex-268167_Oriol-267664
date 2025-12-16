import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from app.scav_logic import (
    ColorTranslator, 
    SerpentineScanner, 
    RLEConverter, 
    DCTConverter, 
    DWTConverter,
    MonsterTranscoder # <-- Importar la nova clase
)

# Adaptació del Seminari 1 
# Hem fet ús de la IA per adaptar el codi
# Ha estat revisat i modificat per nosaltres posteriorment

class TestSCAVLogic(unittest.TestCase):

    def setUp(self):
        self.translator = ColorTranslator()
        self.scanner = SerpentineScanner()
        self.rle = RLEConverter()
        self.dct = DCTConverter()
        self.dwt = DWTConverter()

    # --- TESTS COLOR ---
    def test_rgb_to_yuv_known(self):
        y, u, v = self.translator.rgb_to_yuv(255, 0, 0)
        self.assertAlmostEqual(y, 0.299*255, delta=1e-6)

    def test_yuv_to_rgb_roundtrip(self):
        rgb_samples = [(0, 0, 0), (255, 255, 255), (10, 120, 200)]
        for r, g, b in rgb_samples:
            y, u, v = self.translator.rgb_to_yuv(r, g, b)
            r2, g2, b2 = self.translator.yuv_to_rgb(y, u, v)
            self.assertTrue(abs(r - r2) <= 1)
            self.assertTrue(abs(g - g2) <= 1)
            self.assertTrue(abs(b - b2) <= 1)

    # --- TESTS SERPENTINE ---
    def test_serpentine_square(self):
        mat = np.array([[1,2,3], [4,5,6], [7,8,9]])
        res = self.scanner.serpentine(mat)
        expected = np.array([1,2,4,7,5,3,6,8,9])
        np.testing.assert_array_equal(res, expected)

    # --- TESTS RLE ---
    def test_rle_simple(self):
        data = [1, 1, 1, 2, 2, 3]
        res = self.rle.encode(data)
        self.assertEqual(res, [(1, 3), (2, 2), (3, 1)])

    # --- TESTS DCT ---
    def test_dct_roundtrip(self):
        block = np.random.rand(8, 8)
        coeffs = self.dct.forward(block)
        recon = self.dct.inverse(coeffs)
        np.testing.assert_allclose(block, recon, atol=1e-6)
        
    # --- TESTS DWT ---
    def test_dwt_roundtrip(self):
        data = np.random.rand(32, 32)
        coeffs = self.dwt.forward(data)
        recon = self.dwt.inverse(coeffs)
        np.testing.assert_allclose(data, recon, atol=1e-6)


# ===========================================================================
#  NOUS TESTS - PRÀCTICA 2 (TRANSCODING)
# ===========================================================================

class TestMonsterTranscoder(unittest.TestCase):
    
    def setUp(self):
        # Instanciem la classe
        self.transcoder = MonsterTranscoder()
        
        # MOCKEJEM EL CLIENT DOCKER:
        # Això evita que el test intenti connectar-se realment a un contenidor,
        # ja que en l'entorn de test (o GitHub Actions) potser no tenim Docker.
        self.transcoder.client = MagicMock()
        self.transcoder.container_name = "test-container"

    @patch('app.scav_logic.MonsterTranscoder.resize_video')
    def test_ladder_inheritance(self, mock_resize):
        """
        EXERCICI 4: Test d'Herència.
        Verifica que create_encoding_ladder crida 3 vegades al mètode 
        resize_video de la classe pare (VideoProcessor).
        """
        # Executem la funció
        result_files = self.transcoder.create_encoding_ladder("video_test.mp4")
        
        # Comprovacions
        self.assertEqual(len(result_files), 3) # Ha de generar 3 fitxers
        self.assertEqual(mock_resize.call_count, 3) # Ha de cridar 3 cops al resize
        
        # Comprovem que els noms de sortida són correctes
        self.assertIn("ladder_1080p_video_test.mp4", result_files)
        self.assertIn("ladder_720p_video_test.mp4", result_files)

    def test_codec_command_generation_vp8(self):
        """
        EXERCICI 4: Test de generació de comandes (VP8).
        Comprova que si demanem VP8, es genera la comanda FFmpeg correcta.
        """
        # Simulem el contenidor de Docker
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=b"")
        self.transcoder.client.containers.get.return_value = mock_container

        # Cridem al mètode
        self.transcoder.convert_to_codec("input.mp4", "output.webm", "vp8")
        
        # Recuperem la comanda que s'hauria executat
        args, _ = mock_container.exec_run.call_args
        command_str = args[0]
        
        # Verifiquem que conté els paràmetres clau de VP8
        self.assertIn("ffmpeg", command_str)
        self.assertIn("libvpx", command_str)
        self.assertIn("libvorbis", command_str)

    def test_codec_invalid_raises_error(self):
        """
        Verifica que si passem un còdec inventat, salta un error.
        """
        with self.assertRaises(ValueError):
            self.transcoder.convert_to_codec("in.mp4", "out.mp4", "codec_inventat")

if __name__ == '__main__':
    unittest.main()