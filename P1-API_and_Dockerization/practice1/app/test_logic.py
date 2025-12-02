import unittest
import numpy as np
from app.scav_logic import ColorTranslator, SerpentineScanner, RLEConverter, DCTConverter, DWTConverter

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

if __name__ == '__main__':
    unittest.main()