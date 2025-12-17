import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from app.scav_logic import (
    ColorTranslator, 
    SerpentineScanner, 
    RLEConverter, 
    DCTConverter, 
    DWTConverter,
    MonsterTranscoder
)

# ===========================================================================
# SUITE DE TESTS AUTOMÀTICS (P1 + P2)
# ===========================================================================

class TestSCAVLogic(unittest.TestCase):

    def setUp(self):
        # Inicialització de les classes bàsiques per a cada test
        self.translator = ColorTranslator()
        self.scanner = SerpentineScanner()
        self.rle = RLEConverter()
        self.dct = DCTConverter()
        self.dwt = DWTConverter()

    # --- TESTS PRÀCTICA 1: BÀSICS ---
    
    def test_rgb_to_yuv_known(self):
        # Cas conegut: Vermell pur
        y, u, v = self.translator.rgb_to_yuv(255, 0, 0)
        self.assertAlmostEqual(y, 0.299*255, delta=1e-6)

    def test_yuv_to_rgb_roundtrip(self):
        # Test d'anada i tornada (RGB -> YUV -> RGB)
        rgb_samples = [(0, 0, 0), (255, 255, 255), (10, 120, 200)]
        for r, g, b in rgb_samples:
            y, u, v = self.translator.rgb_to_yuv(r, g, b)
            r2, g2, b2 = self.translator.yuv_to_rgb(y, u, v)
            # Acceptem un marge d'error de +/- 1 per arrodoniments
            self.assertTrue(abs(r - r2) <= 1)
            self.assertTrue(abs(g - g2) <= 1)
            self.assertTrue(abs(b - b2) <= 1)

    def test_serpentine_square(self):
        # Test matriu quadrada 3x3
        mat = np.array([[1,2,3], [4,5,6], [7,8,9]])
        res = self.scanner.serpentine(mat)
        expected = np.array([1,2,4,7,5,3,6,8,9])
        np.testing.assert_array_equal(res, expected)

    def test_rle_simple(self):
        # Test compressió RLE bàsica
        data = [1, 1, 1, 2, 2, 3]
        res = self.rle.encode(data)
        self.assertEqual(res, [(1, 3), (2, 2), (3, 1)])

    def test_dct_roundtrip(self):
        # Test DCT reversible
        block = np.random.rand(8, 8)
        coeffs = self.dct.forward(block)
        recon = self.dct.inverse(coeffs)
        np.testing.assert_allclose(block, recon, atol=1e-6)
        
    def test_dwt_roundtrip(self):
        # Test DWT reversible (Haar)
        data = np.random.rand(32, 32)
        coeffs = self.dwt.forward(data)
        recon = self.dwt.inverse(coeffs)
        np.testing.assert_allclose(data, recon, atol=1e-6)


# ===========================================================================
#  NOUS TESTS - PRÀCTICA 2 (TRANSCODING)
# ===========================================================================

class TestMonsterTranscoder(unittest.TestCase):
    
    def setUp(self):
        self.transcoder = MonsterTranscoder()
        
        # MOCKEJEM EL CLIENT DOCKER:
        # Sobreescrivim el client real amb un de fals (MagicMock)
        # Això és vital perquè els tests funcionin a GitHub Actions o sense Docker aixecat.
        self.transcoder.client = MagicMock()
        self.transcoder.container_name = "test-container"

    @patch('app.scav_logic.MonsterTranscoder.resize_video')
    def test_ladder_inheritance(self, mock_resize):
        """
        EXERCICI 4: Test d'Herència.
        Verifica que create_encoding_ladder reutilitza codi cridant 3 vegades 
        al mètode resize_video de la classe pare.
        """
        # Executem la funció
        result_files = self.transcoder.create_encoding_ladder("video_test.mp4")
        
        # Comprovacions
        self.assertEqual(len(result_files), 3)        # Ha de generar 3 fitxers
        self.assertEqual(mock_resize.call_count, 3)   # Ha de cridar 3 cops al resize
        
        # Comprovem noms de sortida
        self.assertIn("ladder_1080p_video_test.mp4", result_files)
        self.assertIn("ladder_720p_video_test.mp4", result_files)

    def test_codec_command_generation_vp8(self):
        """
        EXERCICI 4: Test de generació de comandes (VP8).
        Comprova que si demanem VP8, es genera la comanda FFmpeg correcta.
        """
        # Preparem el contenidor fals
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=b"")
        self.transcoder.client.containers.get.return_value = mock_container

        # Cridem al mètode real
        self.transcoder.convert_to_codec("input.mp4", "output.webm", "vp8")
        
        # Inspeccionem quina comanda s'ha enviat
        args, _ = mock_container.exec_run.call_args
        command_str = args[0] # La cadena de text amb la comanda
        
        # Verifiquem paràmetres clau
        self.assertIn("libvpx", command_str)
        self.assertIn("libvorbis", command_str)

    def test_codec_command_generation_av1_optimization(self):
        """
        TEST EXTRA: Comprovem que AV1 utilitza l'optimització de velocitat.
        """
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=b"")
        self.transcoder.client.containers.get.return_value = mock_container

        self.transcoder.convert_to_codec("input.mp4", "output.mkv", "av1")
        
        args, _ = mock_container.exec_run.call_args
        command_str = args[0]
        
        # Ha d'incloure el flag d'optimització que hem posat a scav_logic.py
        self.assertIn("-cpu-used 8", command_str)
        self.assertIn("libaom-av1", command_str)

    def test_codec_invalid_raises_error(self):
        """
        Verifica que si passem un còdec inventat, salta un error controlat.
        """
        with self.assertRaises(ValueError):
            self.transcoder.convert_to_codec("in.mp4", "out.mp4", "codec_inventat")

if __name__ == '__main__':
    unittest.main()