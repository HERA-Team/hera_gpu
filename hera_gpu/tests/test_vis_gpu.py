import unittest
import vis_gpu as vis
import numpy as np
from numpy.random import rand
from scipy.interpolate import RectBivariateSpline
import time

np.random.seed(0)
NANT = 16
NTIMES = 10
BM_PIX = 31
NPIX = 12 * 16 ** 2

class TestVisGpu(unittest.TestCase):
    def test_shapes(self):
        antpos = np.zeros((NANT, 3))
        eq2tops = np.zeros((NTIMES, 3, 3))
        crd_eq = np.zeros((3, NPIX))
        I_sky = np.zeros(NPIX)
        bm_cube = np.zeros((NANT, BM_PIX, BM_PIX))
        v = vis.vis_gpu(antpos, 0.15, eq2tops, crd_eq, I_sky, bm_cube)
        self.assertEqual(v.shape, (NTIMES, NANT, NANT))
        self.assertRaises(
            AssertionError, vis.vis_gpu, antpos.T, 0.15, eq2tops, crd_eq, I_sky, bm_cube
        )
        self.assertRaises(
            AssertionError, vis.vis_gpu, antpos, 0.15, eq2tops.T, crd_eq, I_sky, bm_cube
        )
        self.assertRaises(
            AssertionError, vis.vis_gpu, antpos, 0.15, eq2tops, crd_eq.T, I_sky, bm_cube
        )
        self.assertRaises(
            AssertionError, vis.vis_gpu, antpos, 0.15, eq2tops, crd_eq, I_sky, bm_cube.T
        )

    def test_dtypes(self):
        for dtype in (np.float32, np.float64):
            antpos = np.zeros((NANT, 3), dtype=dtype)
            eq2tops = np.zeros((NTIMES, 3, 3), dtype=dtype)
            crd_eq = np.zeros((3, NPIX), dtype=dtype)
            I_sky = np.zeros(NPIX, dtype=dtype)
            bm_cube = np.zeros((NANT, BM_PIX, BM_PIX), dtype=dtype)
            v = vis.vis_gpu(antpos, 0.15, eq2tops, crd_eq, I_sky, bm_cube)
            self.assertEqual(v.dtype, np.complex64)
            v = vis.vis_gpu(
                antpos,
                0.15,
                eq2tops,
                crd_eq,
                I_sky,
                bm_cube,
                real_dtype=np.float64,
                complex_dtype=np.complex128,
            )
            self.assertEqual(v.dtype, np.complex128)
    def test_values(self):
        antpos = np.ones((NANT, 3))
        eq2tops = np.array([np.identity(3)] * NTIMES)
        crd_eq = np.zeros((3, NPIX))
        crd_eq[2] = 1
        I_sky = np.ones(NPIX)
        bm_cube = np.ones((NANT, BM_PIX, BM_PIX))
        # Make sure that a zero in sky or beam gives zero output
        v = vis.vis_gpu(antpos, 1.0, eq2tops, crd_eq, I_sky * 0, bm_cube)
        np.testing.assert_equal(v, 0)
        v = vis.vis_gpu(antpos, 1.0, eq2tops, crd_eq, I_sky, bm_cube * 0)
        np.testing.assert_equal(v, 0)
        # For co-located ants & sources on sky, answer should be sum of pixels
        v = vis.vis_gpu(antpos, 1.0, eq2tops, crd_eq, I_sky, bm_cube)
        np.testing.assert_almost_equal(v, NPIX, 2)
        v = vis.vis_gpu(
            antpos,
            1.0,
            eq2tops,
            crd_eq,
            I_sky,
            bm_cube,
            real_dtype=np.float64, #FLOAT64 
            complex_dtype=np.complex128, #COMPLEX128
        )
        np.testing.assert_almost_equal(v, NPIX, 10)


	
        # For co-located ants & two sources separated on sky, answer should still be sum
        crd_eq = np.zeros((3, 2))
        crd_eq[2, 0] = 1
        crd_eq[1, 1] = np.sqrt(0.5)
        crd_eq[2, 1] = np.sqrt(0.5)
        I_sky = np.ones(2)
        v = vis.vis_gpu(antpos, 1.0, eq2tops, crd_eq, I_sky, bm_cube)
        np.testing.assert_almost_equal(v, 2, 2)
        # For ant[0] at (0,0,1), ant[1] at (1,1,1), src[0] at (0,0,1) and src[1] at (0,.707,.707)
        antpos[0, 0] = 0
        antpos[0, 1] = 0

        v = vis.vis_gpu(antpos, 1.0, eq2tops, crd_eq, I_sky, bm_cube)
        v_CPU = vis_cpu(antpos, 1.0, eq2tops, crd_eq, I_sky, bm_cube)
	np.testing.assert_allclose(v, v_CPU, 1e-5)

        v_CPU = vis_cpu(antpos, 1.0, eq2tops, crd_eq, I_sky, bm_cube, real_dtype=np.float64, complex_dtype=np.complex128)
        v = vis.vis_gpu(antpos, 1.0, eq2tops, crd_eq, I_sky, bm_cube, real_dtype=np.float64, complex_dtype=np.complex128)
	np.testing.assert_almost_equal(
	    v, v_CPU, 15
	)

	np.testing.assert_almost_equal(
            v[:, 0, 1], 1 + np.exp(-2j * np.pi * np.sqrt(0.5)), 7
        )

    def test_compare_cpu(self):

    	for i in xrange(NTIMES):
	    antpos = np.array(rand(NANT, 3), dtype=np.float32)
	    eq2tops = np.array(rand(NTIMES, 3, 3), dtype=np.float32)
	    crd_eq = np.array(rand(3, NPIX), dtype=np.float32)
	    I_sky = np.array(rand(NPIX), dtype=np.float32)
	    bm_cube = np.array(rand(NANT, BM_PIX, BM_PIX), dtype=np.float32)
	    freq = float(rand(1)[0])

	    v_gpu = vis.vis_gpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube) #, real_dtype=np.float64, complex_dtype=np.complex128)
	    v_cpu = vis_cpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube)
	    np.testing.assert_allclose(v_gpu, v_cpu, 1e-5)
	    
	    v_gpu = vis.vis_gpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube, real_dtype=np.float64, complex_dtype=np.complex128)
	    v_cpu = vis_cpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube, real_dtype=np.float64, complex_dtype=np.complex128)
	    np.testing.assert_allclose(v_gpu, v_cpu, 1e-9)


"""
def vis_cpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube,
            real_dtype=np.float32, complex_dtype=np.complex64,
            verbose=False):

    crd_eq = crd_eq.astype(real_dtype)



    t_start = time.time()
    nant = len(antpos)
    ntimes = len(eq2tops)
    npix = I_sky.size
    bm_pix = bm_cube.shape[-1]
    Isqrt = np.sqrt(I_sky).astype(real_dtype)
    antpos = antpos.astype(real_dtype)
    A_s = np.empty((nant,npix), dtype=real_dtype)
    vis = np.empty((ntimes,nant,nant), dtype=complex_dtype)
    tau = np.empty((nant,npix), dtype=real_dtype)
    v = np.empty((nant,npix), dtype=complex_dtype)
    bm_pix_x = np.linspace(-1,1,bm_pix)
    bm_pix_y = np.linspace(-1,1,bm_pix)
    for t,eq2top in enumerate(eq2tops.astype(real_dtype)):
        tx,ty,tz = crd_top = np.dot(eq2top, crd_eq)
        for i in xrange(nant):
            spline = RectBivariateSpline(bm_pix_y, bm_pix_x, bm_cube[i], kx=1, ky=1)
            A_s[i] = spline(ty, tx, grid=False)
        A_s = np.where(tz > 0, A_s, 0)

        np.dot(antpos, crd_top, out=tau) #OUT=TAU
        np.exp((1j*freq)*tau, out=v)
        AI_s = A_s * Isqrt
        v *= AI_s
        for i in xrange(len(antpos)):
            # only compute upper triangle
            np.dot(v[i:i+1].conj(), v[i:].T, out=vis[t,i:i+1,i:])
        if verbose:
            print 'TOTAL:', time.time() - t_start
            #print vis[t].conj()
    np.conj(vis, out=vis)
    for i in xrange(nant):
        # fill in whole corr matrix from upper triangle
        vis[:,i+1:,i] = vis[:,i,i+1:].conj()
    return vis
"""

def vis_cpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube, real_dtype=np.float32, complex_dtype=np.complex64):
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
        real_dtype, complex_dtype (dtype, optional): data type to use for real and complex-valued arrays.
    
    Returns:
        array_like, shape(NTIMES, NANTS, NANTS): visibilities
    """
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
