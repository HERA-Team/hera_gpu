import unittest
import hera_gpu.redcal as redcal
from hera_sim.antpos import linear_array, hex_array
from hera_sim.vis import sim_red_data
import hera_cal.redcal as om

import numpy as np
from numpy.random import rand
import time

np.random.seed(0)
NANTS = 16
NDATA = 2048

class TestRedcalCuda(unittest.TestCase):
    def test_float_compile(self):
        gpu_code = redcal.GPU_TEMPLATE.format(**{
            'NDATA': NDATA,
            'NBLS': NANTS * (NANTS - 1) // 2,
            'NUBLS': 2 * NANTS,
            'NANTS': NANTS,
            'GAIN': 0.3,
            'CMULT': 'cuCmulf',
            'CONJ': 'cuConjf',
            'CSUB': 'cuCsubf',
            'CDIV': 'cuCdivf',
            'DTYPE': 'float',
            'CDTYPE': 'cuFloatComplex',
        })
        gpu_module = redcal.compiler.SourceModule(gpu_code)
        gen_dmdl_cuda = gpu_module.get_function("gen_dmdl")
        calc_chisq_cuda = gpu_module.get_function("calc_chisq")
        calc_dwgts_cuda = gpu_module.get_function("calc_dwgts")
        calc_gu_wgt_cuda = gpu_module.get_function("calc_gu_wgt")
        calc_gu_buf_cuda = gpu_module.get_function("calc_gu_buf")
        update_gains_cuda = gpu_module.get_function("update_gains")
        update_ubls_cuda = gpu_module.get_function("update_ubls")
        calc_conv_gains_cuda = gpu_module.get_function("calc_conv")
    def test_double_compile(self):
        gpu_code = redcal.GPU_TEMPLATE.format(**{
            'NDATA': NDATA,
            'NBLS': NANTS * (NANTS - 1) // 2,
            'NUBLS': 2 * NANTS,
            'NANTS': NANTS,
            'GAIN': 0.3,
            'CMULT': 'cuCmul',
            'CONJ': 'cuConj',
            'CSUB': 'cuCsub',
            'CDIV': 'cuCdiv',
            'DTYPE': 'double',
            'CDTYPE': 'cuDoubleComplex',
        })
        gpu_module = redcal.compiler.SourceModule(gpu_code)
        gen_dmdl_cuda = gpu_module.get_function("gen_dmdl")
        calc_chisq_cuda = gpu_module.get_function("calc_chisq")
        calc_dwgts_cuda = gpu_module.get_function("calc_dwgts")
        calc_gu_wgt_cuda = gpu_module.get_function("calc_gu_wgt")
        calc_gu_buf_cuda = gpu_module.get_function("calc_gu_buf")
        update_gains_cuda = gpu_module.get_function("update_gains")
        update_ubls_cuda = gpu_module.get_function("update_ubls")
        calc_conv_ubls_cuda = gpu_module.get_function("calc_conv")
    def test_already_correct(self):
        nubls = 15
        nbls = NANTS * (NANTS - 1) // 2
        for precision in (1,2):
            gains = np.ones((NDATA, NANTS), dtype=np.complex64)
            ubls  = np.ones((NDATA, nubls,), dtype=np.complex64)
            data  = np.ones((NDATA, nbls, ), dtype=np.complex64)
            wgts  = np.ones((NDATA, nbls, ), dtype=np.float32)
            conv_crit = 1e-4
            maxiter = 100
            check_every, check_after = 2, 1
            ggu_indices = np.array([(i,j,i-j-1) for i in range(NANTS) 
                for j in range(i)], dtype=np.uint)
            info = redcal.omnical(ggu_indices, gains, ubls, data, wgts,
                        conv_crit, maxiter, check_every, check_after,
                        precision=precision)
            np.testing.assert_allclose(info['gains'], 1, 4)
            np.testing.assert_allclose(info['ubls'], 1, 4)
            np.testing.assert_allclose(info['chisq'], 0, 4)
            np.testing.assert_allclose(info['iters'], 1, 4)
            np.testing.assert_allclose(info['conv'], 0, 4)
    def test_calibrate(self):
        nubls = 15
        nbls = NANTS * (NANTS - 1) // 2
        for precision in (1,2):
            gains = 1.1 * np.ones((NDATA, NANTS), dtype=np.complex64)
            ubls = np.ones((NDATA, nubls,), dtype=np.complex64)
            data = np.ones((NDATA, nbls, ), dtype=np.complex64)
            wgts = np.ones((NDATA, nbls, ), dtype=np.float32)
            conv_crit = 1e-3**precision
            maxiter = 100
            check_every, check_after = 2, 1
            ggu_indices = np.array([(i,j,i-j-1) for i in range(NANTS) 
                for j in range(i)], dtype=np.uint)
            info = redcal.omnical(ggu_indices, gains, ubls, data, wgts,
                        conv_crit, maxiter, check_every, check_after,
                        precision=precision)
            np.testing.assert_allclose(info['gains'], 1, 4)
            np.testing.assert_allclose(info['ubls'], 1, 4)
            np.testing.assert_allclose(info['chisq'], 0, atol=10)
            np.testing.assert_array_less(1, info['iters'])
            np.testing.assert_array_less(info['conv'], conv_crit)

class TestOmnicalSolver(unittest.TestCase):
    def test_wrap(self):
        shape = (NDATA, 1)
        antpos = linear_array(NANTS)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        info = redcal.RedundantCalibratorGPU(reds)
        gains, true_vis, d = sim_red_data(reds, gain_scatter=.0099999, shape=shape)
        w = dict([(k, 1.) for k in d.keys()])
        sol0 = dict([(k, np.ones_like(v)) for k, v in gains.items()])
        sol0.update(info.compute_ubls(d, sol0))
        meta, sol = info.omnical_gpu(d, sol0, conv_crit=1e-12, gain=.5, maxiter=500, check_after=30, check_every=6)
        # Leaving a reference to standard (non-GPU) omnical here in
        # case we want to compare GPU vs non-GPU for speed/accuracy.
        #meta, sol = info.omnical(d, sol0, conv_crit=1e-12, gain=.5, maxiter=500, check_after=30, check_every=6)
        for i in range(NANTS):
            assert sol[(i, 'Jxx')].shape == shape
        for bls in reds:
            ubl = sol[bls[0]]
            assert ubl.shape == shape
            for bl in bls:
                d_bl = d[bl]
                mdl = sol[(bl[0], 'Jxx')] * sol[(bl[1], 'Jxx')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), decimal=10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, decimal=10)
    def test_same(self):
        antpos = linear_array(NANTS)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        info = redcal.RedundantCalibratorGPU(reds)
        shape = (NDATA, 1)
        gains, true_vis, d = sim_red_data(reds, gain_scatter=.0099999, shape=shape)
        w = dict([(k, 1.) for k in d.keys()])
        sol0 = dict([(k, np.ones_like(v)) for k, v in gains.items()])
        sol0.update(info.compute_ubls(d, sol0))
        for precision in (1,2):
            conv_crit = 1e-12
            kwargs = {'maxiter': 100, 'check_after': 10,
                'check_every': 4, 'gain': 0.3,
                'conv_crit': conv_crit}
            meta_gpu, sol_gpu = info.omnical_gpu(d, sol0, precision=precision, **kwargs)
            meta_cpu, sol_cpu = info.omnical(d, sol0, **kwargs)
            for k in sol_cpu:
                np.testing.assert_almost_equal(sol_gpu[k], sol_cpu[k], decimal=5*precision)

if __name__ == "__main__":
    unittest.main()
