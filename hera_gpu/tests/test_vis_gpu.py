import unittest
import hera_gpu.vis as vis
from hera_gpu.vis import vis_gpu
import numpy as np
from numpy.random import rand
from scipy.interpolate import RectBivariateSpline
import time

np.random.seed(0)
NANT = 16
NTIMES = 10
BM_PIX = 31
NPIX = 12 * 16 ** 2

class TestCuda(unittest.TestCase):
    def test_compile(self):
        gpu_code = vis.GPU_TEMPLATE % {
            'NANT': 8,
            'NPIX': 1024,
            'BEAM_PX': 31,
            'BLOCK_PX': 128,
            'DTYPE': 'float',
            'CDTYPE': 'cuFloatComplex',
        }
        gpu_module = vis.compiler.SourceModule(gpu_code)
        bm_interp = gpu_module.get_function('InterpolateBeam')
        meas_eq = gpu_module.get_function('MeasEq')
        bm_texref = gpu_module.get_texref('bm_tex')
    def test_bm_interp_flat(self):
        NANT = 8
        BEAM_PX = 31
        NPIX = 128
        bm_cube = np.ones((NANT, BEAM_PX, BEAM_PX), dtype=np.float32)
        gpu_code = vis.GPU_TEMPLATE % {
            'NANT': NANT,
            'NPIX': NPIX,
            'BEAM_PX': BEAM_PX,
            'BLOCK_PX': NPIX,
            'DTYPE': 'float',
            'CDTYPE': 'cuFloatComplex',
        }
        gpu_module = vis.compiler.SourceModule(gpu_code)
        bm_interp = gpu_module.get_function('InterpolateBeam')
        bm_texref = gpu_module.get_texref('bm_tex')
        bm_texref.set_array(vis.numpy3d_to_array(bm_cube))
        crdtop_gpu = vis.gpuarray.empty(shape=(3,NPIX), dtype=np.float32)
        A_gpu = vis.gpuarray.empty(shape=(NANT,NPIX), dtype=np.float32)
        block = (NPIX, NANT, 1)
        grid = (1, 1)
        stream = vis.driver.Stream()
        # Set z>0 => "above the horizon"
        crdtop = np.zeros((3,NPIX), dtype=np.float32)
        crdtop[2] = 0.9
        crdtop_gpu.set_async(crdtop)
        bm_interp(crdtop_gpu, A_gpu, grid=grid, block=block, stream=stream)
        A = A_gpu.get_async()
        self.assertTrue(np.all(A == 1))
        # Set z<0 => "below the horizon"
        crdtop[2] = -0.9
        crdtop_gpu.set_async(crdtop)
        bm_interp(crdtop_gpu, A_gpu, grid=grid, block=block, stream=stream)
        A = A_gpu.get_async()
        self.assertTrue(np.all(A == 0))
    def test_bm_interp_middle(self):
        NANT = 8
        BEAM_PX = 31
        NPIX = 128
        bm_cube = np.zeros((NANT, BEAM_PX, BEAM_PX), dtype=np.float32)
        bm_cube[:,15,15] = 1.
        gpu_code = vis.GPU_TEMPLATE % {
            'NANT': NANT,
            'NPIX': NPIX,
            'BEAM_PX': BEAM_PX,
            'BLOCK_PX': NPIX,
            'DTYPE': 'float',
            'CDTYPE': 'cuFloatComplex',
        }
        gpu_module = vis.compiler.SourceModule(gpu_code)
        bm_interp = gpu_module.get_function('InterpolateBeam')
        bm_texref = gpu_module.get_texref('bm_tex')
        bm_texref.set_array(vis.numpy3d_to_array(bm_cube))
        crdtop_gpu = vis.gpuarray.empty(shape=(3,NPIX), dtype=np.float32)
        A_gpu = vis.gpuarray.empty(shape=(NANT,NPIX), dtype=np.float32)
        block = (NPIX, NANT, 1)
        grid = (1, 1)
        stream = vis.driver.Stream()
        # Set x,y=0 "center of the beam"
        crdtop = np.zeros((3,NPIX), dtype=np.float32)
        crdtop[2] = 1.0
        crdtop_gpu.set_async(crdtop)
        bm_interp(crdtop_gpu, A_gpu, grid=grid, block=block, stream=stream)
        A = A_gpu.get_async()
        self.assertTrue(np.all(A == 1))
        # Set y off of "center of the beam"
        crdtop[2] = 0.5**0.5
        crdtop[1] = 0.5**0.5
        crdtop_gpu.set_async(crdtop)
        bm_interp(crdtop_gpu, A_gpu, grid=grid, block=block, stream=stream)
        A = A_gpu.get_async()
        self.assertTrue(np.all(A == 0))
    def test_meas_eq(self):
        NANT = 8
        NPIX = 128
        gpu_code = vis.GPU_TEMPLATE % {
            'NANT': NANT,
            'NPIX': NPIX,
            'BEAM_PX': 31,
            'BLOCK_PX': NPIX,
            'DTYPE': 'float',
            'CDTYPE': 'cuFloatComplex',
        }
        gpu_module = vis.compiler.SourceModule(gpu_code)
        meas_eq = gpu_module.get_function('MeasEq')
        block = (NPIX, NANT, 1)
        grid = (1, 1)
        stream = vis.driver.Stream()
        A_gpu = vis.gpuarray.empty(shape=(NANT,NPIX), dtype=np.float32)
        A = np.ones((NANT,NPIX), dtype=np.float32)
        A_gpu.set_async(A)
        I_gpu = vis.gpuarray.empty(shape=(NPIX,), dtype=np.float32)
        I = np.ones((NPIX,), dtype=np.float32)
        I_gpu.set_async(I)
        tau_gpu = vis.gpuarray.empty(shape=(NANT,NPIX), dtype=np.float32)
        tau = np.zeros((NANT,NPIX), dtype=np.float32)
        tau_gpu.set_async(tau)
        v_gpu = vis.gpuarray.empty(shape=(NANT,NPIX), dtype=np.complex64)
        v = np.empty((NANT,NPIX), dtype=np.complex64)
        meas_eq(A_gpu, I_gpu, tau_gpu, np.float32(0.1), v_gpu,
            grid=grid, block=block, stream=stream)
        v = v_gpu.get_async()
        self.assertTrue(np.all(v == 1))
        tau[:,0] = 2 * np.pi * 10
        tau_gpu.set_async(tau)
        meas_eq(A_gpu, I_gpu, tau_gpu, np.float32(0.1), v_gpu,
            grid=grid, block=block, stream=stream)
        v = v_gpu.get_async()
        np.testing.assert_allclose(v, 1, rtol=1e-6)
    def test_cublas(self):
        NANT = 8
        NPIX = 128
        gpu_code = vis.GPU_TEMPLATE % {
            'NANT': NANT,
            'NPIX': NPIX,
            'BEAM_PX': 31,
            'BLOCK_PX': NPIX,
            'DTYPE': 'float',
            'CDTYPE': 'cuFloatComplex',
        }
        gpu_module = vis.compiler.SourceModule(gpu_code)
        meas_eq = gpu_module.get_function('MeasEq')
        block = (NPIX, NANT, 1)
        grid = (1, 1)
        h = vis.cublasCreate()
        stream = vis.driver.Stream()
        crd_eq_gpu = vis.gpuarray.empty(shape=(3,NPIX), dtype=np.float32)
        crd_eq = np.random.normal(size=(3,NPIX)).astype(np.float32)
        crd_eq_gpu.set_async(crd_eq)
        eq2top_gpu = vis.gpuarray.empty(shape=(3,3), dtype=np.float32)
        eq2top = np.array([[1.,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
        eq2top_gpu.set_async(eq2top)
        crdtop_gpu = vis.gpuarray.empty(shape=(3,NPIX), dtype=np.float32)
        vis.cublasSetStream(h, stream.handle)
        vis.cublasSgemm(h, 'n', 'n', NPIX, 3, 3, 1., crd_eq_gpu.gpudata,
            NPIX, eq2top_gpu.gpudata, 3, 0., crdtop_gpu.gpudata, NPIX)
        crdtop = crdtop_gpu.get_async()
        vis.cublasDestroy(h)
        np.testing.assert_allclose(crd_eq, crdtop, rtol=1e-6)

class TestVisGpu(unittest.TestCase):
    def test_shapes(self):
        antpos = np.zeros((NANT, 3))
        eq2tops = np.zeros((NTIMES, 3, 3))
        crd_eq = np.zeros((3, NPIX))
        I_sky = np.zeros(NPIX)
        bm_cube = np.zeros((NANT, BM_PIX, BM_PIX))
        v = vis_gpu(antpos, 0.15, eq2tops, crd_eq, I_sky, bm_cube)
        self.assertEqual(v.shape, (NTIMES, NANT, NANT))
        self.assertRaises(
            AssertionError, vis_gpu, antpos.T, 0.15, eq2tops, crd_eq, I_sky, bm_cube
        )
        self.assertRaises(
            AssertionError, vis_gpu, antpos, 0.15, eq2tops.T, crd_eq, I_sky, bm_cube
        )
        self.assertRaises(
            AssertionError, vis_gpu, antpos, 0.15, eq2tops, crd_eq.T, I_sky, bm_cube
        )
        self.assertRaises(
            AssertionError, vis_gpu, antpos, 0.15, eq2tops, crd_eq, I_sky, bm_cube.T
        )

    def test_dtypes(self):
        antpos = np.zeros((NANT, 3))
        eq2tops = np.zeros((NTIMES, 3, 3))
        crd_eq = np.zeros((3, NPIX))
        I_sky = np.zeros(NPIX)
        bm_cube = np.zeros((NANT, BM_PIX, BM_PIX))
        for precision, cdtype in zip((1,2),(np.complex64, np.complex128)):
            v = vis_gpu(antpos, 0.15, eq2tops, crd_eq, I_sky, bm_cube, 
                        precision=precision)
            self.assertEqual(v.dtype, cdtype)
    def test_values(self):
        antpos = np.ones((NANT, 3))
        eq2tops = np.array([np.identity(3)] * NTIMES)
        crd_eq = np.zeros((3, NPIX))
        crd_eq[2] = 1
        I_sky = np.ones(NPIX)
        bm_cube = np.ones((NANT, BM_PIX, BM_PIX))
        for precision in (1,2):
            # Make sure that a zero in sky or beam gives zero output
            v = vis_gpu(antpos, 1.0, eq2tops, crd_eq, I_sky * 0, bm_cube,
                precision=precision)
            np.testing.assert_equal(v, 0)
            v = vis_gpu(antpos, 1.0, eq2tops, crd_eq, I_sky, bm_cube * 0,
                precision=precision)
            np.testing.assert_equal(v, 0)
            # For co-located ants & sources on sky, answer should be sum of pixels
            v = vis_gpu(antpos, 1.0, eq2tops, crd_eq, I_sky, bm_cube,
                precision=precision)
            np.testing.assert_equal(v, NPIX)
    
        # For co-located ants & two sources separated on sky, answer should still be sum
        crd_eq = np.zeros((3, 2))
        crd_eq[2, 0] = 1
        crd_eq[1, 1] = np.sqrt(0.5)
        crd_eq[2, 1] = np.sqrt(0.5)
        I_sky = np.ones(2)
        for precision in (1,2):
            v = vis_gpu(antpos, 1.0, eq2tops, crd_eq, I_sky, bm_cube,
                precision=precision)
            np.testing.assert_almost_equal(v, 2, 2)
        # For ant[0] at (0,0,1), ant[1] at (1,1,1), src[0] at (0,0,1) and src[1] at (0,.707,.707)
        antpos[0, 0] = 0
        antpos[0, 1] = 0
        for precision in (1,2):
            v_CPU = vis_cpu(antpos, 1.0, eq2tops, crd_eq, I_sky, bm_cube,
                precision=precision)
            v = vis_gpu(antpos, 1.0, eq2tops, crd_eq, I_sky, bm_cube,
                precision=precision)
            if precision == 1:
                np.testing.assert_almost_equal(v, v_CPU, 5)
            else:
                np.testing.assert_almost_equal(v, v_CPU, 15)
    def test_compare_cpu(self):
        for i in range(NTIMES):
            antpos = np.array(rand(NANT, 3), dtype=np.float32)
            eq2tops = np.array(rand(NTIMES, 3, 3), dtype=np.float32)
            crd_eq = np.array(rand(3, NPIX), dtype=np.float32)
            I_sky = np.array(rand(NPIX), dtype=np.float32)
            bm_cube = np.array(rand(NANT, BM_PIX, BM_PIX), dtype=np.float32)
            freq = float(rand(1)[0])
            for precision in (1,2):
                v = vis_gpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube, precision=precision)
                v_CPU = vis_cpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube, precision=precision)
                diff = np.abs(v - v_CPU)
                if precision == 1:
                    self.assertLess(np.median(diff), 1e-4)
                else:
                    self.assertLess(np.median(diff), 1e-2)


def vis_cpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube, precision=1):
    """
    Calculate visibility from an input intensity map and beam model.
    
    Args:
        antpos (array_like, shape: (NANT, 3)): antenna position array.
        freq (float): frequency to evaluate the visibilities at [GHz].
        eq2tops (array_like, shape: (NTIMES, 3, 3)): Set of 3x3 transformation matrices converting equatorial
            coordinates to topocentric at each hour angle (and declination) in the dataset.
        crd_eq (array_like, shape: (3, NPIX)): equatorial coordinates of Healpix pixels.
        I_sky (array_like, shape: (NPIX,)): intensity distribution on the sky, stored as array of Healpix pixels.
        bm_cube (array_like, shape: (NANT, BM_PIX, BM_PIX)): beam maps for each antenna.
        precision (int): 1 for single precision, 2 for doule
    
    Returns:
        array_like, shape(NTIMES, NANTS, NANTS): visibilities
    """
    assert precision in (1,2)
    if precision == 1:
        real_dtype, complex_dtype = np.float32, np.complex64
    else:
        real_dtype, complex_dtype = np.float64, np.complex128
    nant, ncrd = antpos.shape
    assert ncrd == 3, "antpos must have shape (NANTS, 3)"
    ntimes, ncrd1, ncrd2 = eq2tops.shape
    assert ncrd1 == 3 and ncrd2 == 3, "eq2tops must have shape (NTIMES, 3, 3)"
    ncrd, npix = crd_eq.shape
    assert ncrd == 3, "crd_eq must have shape (3, NPIX)"
    assert I_sky.ndim == 1 and I_sky.shape[0] == npix, "I_sky must have shape (NPIX,)"
    bm_pix = bm_cube.shape[-1]
    assert bm_cube.shape == (
        nant,
        bm_pix,
        bm_pix,
    ), "bm_cube must have shape (NANTS, BM_PIX, BM_PIX)"

    # Intensity distribution (sqrt) and antenna positions
    Isqrt = np.sqrt(I_sky).astype(real_dtype)  # XXX does not support negative sky
    antpos = antpos.astype(real_dtype)
    ang_freq = 2 * np.pi * freq

    # Empty arrays: beam pattern, visibilities, delays, complex voltages
    A_s = np.empty((nant, npix), dtype=real_dtype)
    vis = np.empty((ntimes, nant, nant), dtype=complex_dtype)
    tau = np.empty((nant, npix), dtype=real_dtype)
    v = np.empty((nant, npix), dtype=complex_dtype)
    crd_eq = crd_eq.astype(real_dtype)

    bm_pix_x = np.linspace(-1, 1, bm_pix)
    bm_pix_y = np.linspace(-1, 1, bm_pix)

    # Loop over time samples
    for t, eq2top in enumerate(eq2tops.astype(real_dtype)):
        tx, ty, tz = crd_top = np.dot(eq2top, crd_eq)
        for i in range(nant):
            # Linear interpolation of primary beam pattern
            spline = RectBivariateSpline(bm_pix_y, bm_pix_x, bm_cube[i], kx=1, ky=1)
            A_s[i] = spline(ty, tx, grid=False)
        A_s = np.where(tz > 0, A_s, 0)

        # Calculate delays
        np.dot(antpos, crd_top, out=tau)
        np.exp((1.0j * ang_freq) * tau, out=v)

        # Complex voltages
        v *= A_s * Isqrt

        # Compute visibilities (upper triangle only)
        for i in range(len(antpos)):
            np.dot(v[i : i + 1].conj(), v[i:].T, out=vis[t, i : i + 1, i:])

    # Conjugate visibilities
    np.conj(vis, out=vis)

    # Fill in whole visibility matrix from upper triangle
    for i in range(nant):
        vis[:, i + 1 :, i] = vis[:, i, i + 1 :].conj()

    return vis


if __name__ == "__main__":
    unittest.main()
