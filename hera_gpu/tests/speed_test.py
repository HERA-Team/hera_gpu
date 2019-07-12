import unittest
import vis_gpu as vis
import numpy as np
from numpy.random import rand
from scipy.interpolate import RectBivariateSpline
import time

np.random.seed(0)
NTIMES = 10
BM_PIX = 31
NPIX = 12 * 16 ** 2


class TestSpeed(unittest.TestCase):
    def test_small(self):

    	NANT = 16
	antpos = np.array(rand(NANT, 3), dtype=np.float32)
	eq2tops = np.array(rand(NTIMES, 3, 3), dtype=np.float32)
	crd_eq = np.array(rand(3, NPIX), dtype=np.float32)
	I_sky = np.array(rand(NPIX), dtype=np.float32)
	bm_cube = np.array(rand(NANT, BM_PIX, BM_PIX), dtype=np.float32)
	freq = np.array(rand(1), dtype=np.float32)[0]


	t = time.time()
	v_gpu = vis.vis_gpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube, real_dtype=np.float64, complex_dtype=np.complex128)
	print "SMALL GPU TIME (FLOAT):", time.time() - t, "s"

	t = time.time()
	v_cpu = vis_cpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube, real_dtype=np.float64, complex_dtype=np.complex128)
	print "SMALL CPU TIME (FLOAT):", time.time() - t, "s"
	print "MAX REL DIFFERENCE", max_rel_diff(v_gpu, v_cpu)


	antpos = np.array(rand(NANT, 3), dtype=np.float64)
	eq2tops = np.array(rand(NTIMES, 3, 3), dtype=np.float64)
	crd_eq = np.array(rand(3, NPIX), dtype=np.float64)
	I_sky = np.array(rand(NPIX), dtype=np.float64)
	bm_cube = np.array(rand(NANT, BM_PIX, BM_PIX), dtype=np.float64)
	freq = np.array(rand(1), dtype=np.float64)[0]

	t = time.time()
	v_gpu = vis.vis_gpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube, real_dtype=np.float64, complex_dtype=np.complex128)
	print "SMALL GPU TIME (DOUBLE):", time.time() - t, "s"

	t = time.time()
	v_cpu = vis_cpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube, real_dtype=np.float64, complex_dtype=np.complex128)
	print "SMALL CPU TIME (DOUBLE):", time.time() - t, "s"
	print "MAX REL DIFFERENCE", max_rel_diff(v_gpu, v_cpu)



    def test_512_ants(self):
    	NANT = 512
	NPIX = 12 * 16 ** 2
	
	antpos = np.array(rand(NANT, 3), dtype=np.float32)
	eq2tops = np.array(rand(NTIMES, 3, 3), dtype=np.float32)
	crd_eq = np.array(rand(3, NPIX), dtype=np.float32)
	I_sky = np.array(rand(NPIX), dtype=np.float32)
	bm_cube = np.array(rand(NANT, BM_PIX, BM_PIX), dtype=np.float32)
	freq = np.array(rand(1), dtype=np.float32)[0]


	t = time.time()
	v_gpu = vis.vis_gpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube)
	print "512 ANTS GPU TIME (FLOAT):", time.time() - t, "s"

	t = time.time()
	v_cpu = vis_cpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube)
	print "512 ANTS CPU TIME (FLOAT):", time.time() - t, "s"
	print "MAX REL DIFFERENCE", max_rel_diff(v_gpu, v_cpu)

	antpos = np.array(rand(NANT, 3), dtype=np.float64)
	eq2tops = np.array(rand(NTIMES, 3, 3), dtype=np.float64)
	crd_eq = np.array(rand(3, NPIX), dtype=np.float64)
	I_sky = np.array(rand(NPIX), dtype=np.float64)
	bm_cube = np.array(rand(NANT, BM_PIX, BM_PIX), dtype=np.float64)
	freq = np.array(rand(1), dtype=np.float64)[0]

	t = time.time()
	v_gpu = vis.vis_gpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube, real_dtype=np.float64, complex_dtype=np.complex128)
	print "512 ANTS GPU TIME (DOUBLE):", time.time() - t, "s"

	t = time.time()
	v_cpu = vis_cpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube, real_dtype=np.float64, complex_dtype=np.complex128)
	print "512 ANTS CPU TIME (DOUBLE):", time.time() - t, "s"
	print "MAX REL DIFFERENCE", max_rel_diff(v_gpu, v_cpu)


    def test_hera_350(self):
    	NANT = 350
	NPIX = int(3.14159 * 10 ** 5)
	
	antpos = np.array(rand(NANT, 3), dtype=np.float32)
	eq2tops = np.array(rand(NTIMES, 3, 3), dtype=np.float32)
	crd_eq = np.array(rand(3, NPIX), dtype=np.float32)
	I_sky = np.array(rand(NPIX), dtype=np.float32)
	bm_cube = np.array(rand(NANT, BM_PIX, BM_PIX), dtype=np.float32)
	freq = np.array(rand(1), dtype=np.float32)[0]


	t = time.time()
	v_gpu = vis.vis_gpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube)
	print "350 ANTS, PI*10^5 PIX GPU TIME (FLOAT):", time.time() - t, "s"

	t = time.time()
	v_cpu = vis_cpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube)
	print "350 ANTS, PI*10^5 PIX CPU TIME (FLOAT):", time.time() - t, "s"
	print "MAX REL DIFFERENCE",max_rel_diff(v_gpu, v_cpu)


	antpos = np.array(rand(NANT, 3), dtype=np.float64)
	eq2tops = np.array(rand(NTIMES, 3, 3), dtype=np.float64)
	crd_eq = np.array(rand(3, NPIX), dtype=np.float64)
	I_sky = np.array(rand(NPIX), dtype=np.float64)
	bm_cube = np.array(rand(NANT, BM_PIX, BM_PIX), dtype=np.float64)
	freq = np.array(rand(1), dtype=np.float64)[0]

	t = time.time()
	v_gpu = vis.vis_gpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube, real_dtype=np.float64, complex_dtype=np.complex128)
	print "350 ANTS, PI*10^5 PIX GPU TIME (DOUBLE):", time.time() - t, "s"

	t = time.time()
	v_cpu = vis_cpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube, real_dtype=np.float64, complex_dtype=np.complex128)
	print "350 ANTS, PI*10^5 CPU TIME (DOUBLE):", time.time() - t, "s"
	print "MAX REL DIFFERENCE", max_rel_diff(v_gpu, v_cpu)


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

    bm_pix_x = np.array(np.linspace(-1, 1, bm_pix)) 
    bm_pix_y = np.array(np.linspace(-1, 1, bm_pix))
    # Loop over time samples
    for t, eq2top in enumerate(eq2tops.astype(real_dtype)):
        tx, ty, tz = crd_top = np.array(np.dot(eq2top, crd_eq))
	for i in range(nant):
            # Linear interpolation of primary beam pattern
            spline = RectBivariateSpline(bm_pix_y, bm_pix_x, bm_cube[i], kx=1, ky=1)
            A_s[i] = spline(ty, tx, grid=False)
        A_s = np.where(tz > 0, A_s, 0)

	if t == ntimes-1:
		print "CPU A_s[0][0]", A_s[0][0], "01", A_s[0][1], "02", A_s[0][2]

        # Calculate delays
        np.dot(antpos, crd_top, out=tau)
        np.exp((1.0j * ang_freq) * tau, out=v)

        # Complex voltages
        v *= A_s * Isqrt

        # Compute visibilities (upper triangle only)
        for i in range(len(antpos)):
            np.dot(v[i : i + 1].conj(), v[i:].T, out=vis[t, i : i + 1, i:])


	if t == ntimes-1:
		print "CPU OUTPUT FOR CORRESPONDING VIS:", v[0][0]
		np.save("CPU", v)

    # Conjugate visibilities
    np.conj(vis, out=vis)

    # Fill in whole visibility matrix from upper triangle
    for i in range(nant):
        vis[:, i + 1 :, i] = vis[:, i, i + 1 :].conj()

    return vis


# MAX ( ABS ( V_GPU - V_CPU ) / V_CPU )
def max_rel_diff(v_gpu, v_cpu):
    return np.max( np.abs( (v_gpu-v_cpu)/v_cpu ) )
    


if __name__ == "__main__":
    unittest.main()
