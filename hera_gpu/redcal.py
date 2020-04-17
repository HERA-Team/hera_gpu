import pycuda.autoinit
from pycuda import compiler, gpuarray, driver
import numpy as np
from math import ceil
import time
from hera_cal.redcal import OmnicalSolver, RedundantCalibrator
import linsolve

GPU_TEMPLATE = """
// CUDA code for interpolating antenna beams and computing "voltage" visibilities 
// [A^1/2 * I^1/2 * exp(-2*pi*i*freq*dot(a,s)/c)]
// === Template Parameters ===
// "CMULT"  : cuCmulf or cuCmul
// "CSUB"  : cuCsubf or cuCsub
// "CDIV"  : cuCdivf or cuCdiv
// "CONJ"  : cuConjf or cuConj
// "DTYPE"  : float or double
// "CDTYPE"  : cuFloatComplex or cuDoubleComplex

#include <cuComplex.h>
#include <pycuda-helpers.hpp>
#include <stdio.h>

// Shared memory for storing baseline order reused among all cells
__shared__ uint sh_buf[{NBLS}*3];

__device__ inline {DTYPE} mag({CDTYPE} a) {{
    return a.x * a.x + a.y * a.y;
}}

// 
__global__ void omnical(uint *ggu_indices, {CDTYPE} *gains, {CDTYPE} *ubls, {CDTYPE} *data, {DTYPE} *wgts, {DTYPE} *chisq, uint *iters, {DTYPE} *conv, float conv_crit, uint maxiter, uint check_every, uint check_after)
{{
    uint i, idx, _iters;
    // Use threads to parallelize over data axis
    const uint tx = threadIdx.x;
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    // Create local buffers
    {CDTYPE} _gains[{NANTS}];
    {CDTYPE} _gains_new[{NANTS}];
    {CDTYPE} _ubls[{NUBLS}];
    {CDTYPE} _ubls_new[{NUBLS}];
    {CDTYPE} _gsum[{NANTS}];
    {DTYPE} _gwgt[{NANTS}];
    {CDTYPE} _usum[{NUBLS}];
    {DTYPE} _uwgt[{NUBLS}];
    {CDTYPE} _dw[{NBLS}];
    {CDTYPE} _data[{NBLS}];
    {DTYPE} _wgts[{NBLS}];
    {CDTYPE} _dmdl[{NBLS}];
    //{DTYPE} _dwgts[{NBLS}];
    {DTYPE} _chisq, _chisq_new, _conv, _conv_sum, _conv_wgt;
    {CDTYPE} wc;
    {DTYPE} wf;

    if (px >= {NDATA}) return;
    if (tx == 0) {{
        // XXX parallelize this better?
        // Copy (gi,gj,uij) indices into shared memory for faster access
        for (int i=0; i < {NBLS} * 3; i++) {{
            sh_buf[i] = ggu_indices[i];
        }}
    }}
    __syncthreads(); // make sure all memory is loaded before computing

    // Copy data into local buffers (XXX global?)
    for (i=0; i < {NANTS}; i++) {{
        _gains[i] = gains[i * {NDATA} + px];
    }}
    for (i=0; i < {NUBLS}; i++) {{
        _ubls[i] = ubls[i * {NDATA} + px];
    }}
    for (i=0; i < {NBLS}; i++) {{
        idx = i * {NDATA};
        _data[i] = data[idx + px];
        _wgts[i] = wgts[idx + px];
    }}

    // Calculate dmdl and chisq for the first time
    _chisq = 0;
    for (i=0; i < {NBLS}; i++) {{
        idx = 3 * i;
        //_dmdl[i] = _gains[sh_buf[idx+0]] * conj(_gains[sh_buf[idx+1]]) * _ubls[sh_buf[idx+2]];
        _dmdl[i] = {CMULT}({CMULT}(_gains[sh_buf[idx+0]], {CONJ}(_gains[sh_buf[idx+1]])), _ubls[sh_buf[idx+2]]);
        //_chisq += pow(abs(_data[i] - _dmdl[i]), 2) * _wgts[i];
        wc = {CSUB}(_data[i], _dmdl[i]);
        _chisq += mag(wc) * _wgts[i];
                //if (px == 0 && i == 0) {{
                //    printf("(%d, %d, %d)\\n", sh_buf[idx], sh_buf[idx+1], sh_buf[idx+2]);
                //    printf("g%d: (%20.18f + 1j %20.18f)\\n", sh_buf[idx], _gains[sh_buf[idx+0]].x, _gains[sh_buf[idx+0]].y);
                //    printf("g%d: (%20.18f + 1j %20.18f)\\n", sh_buf[idx+1], _gains[sh_buf[idx+1]].x, _gains[sh_buf[idx+1]].y);
                //    printf("ubl%d: (%20.18f + 1j %20.18f)\\n", sh_buf[idx+2], _ubls[sh_buf[idx+2]].x, _ubls[sh_buf[idx+2]].y);
                //}}
    }}
    
    // Main OMNICAL loop
    for (_iters=1; _iters <= maxiter; _iters++) {{

    //if (px == 0) printf("%d X2: %.18e\\n", _iters, _chisq);
        // Compute parameter wgts: dwgts = sum(V_mdl^2 * wgts)
        // Don't need to update weighting every iteration
        if (_iters % check_every == 1) {{
            for (i=0; i < {NANTS}; i++) {{
                _gwgt[i] = 0;
            }}
            for (i=0; i < {NUBLS}; i++) {{
                _uwgt[i] = 0;
            }}
            for (i=0; i < {NBLS}; i++) {{
                idx = 3 * i;
                wf = mag(_dmdl[i]) * _wgts[i];
                //if (px == 0 && i == 0) printf("wf: %20.18f\\n", wf);
                _gwgt[sh_buf[idx+0]] += wf;
                _gwgt[sh_buf[idx+1]] += wf;
                _uwgt[sh_buf[idx+2]] += wf;
                _dw[i].x = _data[i].x * wf;
                _dw[i].y = _data[i].y * wf;
                //if (px == 0 && i == 0) printf("dw: (%20.18f + %20.18fj)\\n", _dw[i].x, _dw[i].y);
            }}
        }}

        // Every cycle, compute sum(wgts * V_meas / V_mdl)
        for (i=0; i < {NANTS}; i++) {{
            _gsum[i].x = 0;
            _gsum[i].y = 0;
        }}
        for (i=0; i < {NUBLS}; i++) {{
            _usum[i].x = 0;
            _usum[i].y = 0;
        }}
        for (i=0; i < {NBLS}; i++) {{
            idx = 3 * i;
            //w = _dw[i] / _dmdl[i];
            wc = {CDIV}(_dw[i], _dmdl[i]);
            _gsum[sh_buf[idx+0]].x += wc.x;
            _gsum[sh_buf[idx+0]].y += wc.y;
            _gsum[sh_buf[idx+1]].x += wc.x;
            _gsum[sh_buf[idx+1]].y -= wc.y; // conjugate
            _usum[sh_buf[idx+2]].x += wc.x;
            _usum[sh_buf[idx+2]].y += wc.y;
        }}

        // Calculate dmdl given updated gains and ubls
        // Check if i % check_every is 0, which is purposely one less than 
        // the '1' up at the top of the loop
        if (_iters < maxiter && (_iters < check_after || (_iters % check_every) != 0)) {{
            // Fast branch when we aren't expensively computing convergence/chisq
            for (i=0; i < {NANTS}; i++) {{
                //_gains[i] *= (1 - {GAIN}) + {GAIN} * _gsum[i] / _gwgt[i];
                //_gains[i].x = _gains[i].x * (1 - {GAIN}) + {GAIN} * _gsum[i].x / _gwgt[i];
                //_gains[i].y = _gains[i].y * (1 - {GAIN}) + {GAIN} * _gsum[i].y / _gwgt[i];
                wc.x = (1 - {GAIN}) + {GAIN} * _gsum[i].x / _gwgt[i];
                wc.y = {GAIN} * _gsum[i].y / _gwgt[i];
                _gains[i] = {CMULT}(_gains[i], wc);
            }}
            for (i=0; i < {NUBLS}; i++) {{
                //_ubls[i] *= (1 - {GAIN}) + {GAIN} * _usum[i] / _uwgt[i];
                //_ubls[i].x = _ubls[i].x * (1 - {GAIN}) + {GAIN} * _usum[i].x / _uwgt[i];
                //_ubls[i].y = _ubls[i].y * (1 - {GAIN}) + {GAIN} * _usum[i].y / _uwgt[i];
                wc.x = (1 - {GAIN}) + {GAIN} * _usum[i].x / _uwgt[i];
                wc.y = {GAIN} * _usum[i].y / _uwgt[i];
                _ubls[i] = {CMULT}(_ubls[i], wc);
            }}
            for (i=0; i < {NBLS}; i++) {{
                idx = 3 * i;
                //_dmdl[i] = _gains[sh_buf[idx+0]] * conj(_gains[sh_buf[idx+1]]) * _ubls[sh_buf[idx+2]];
                _dmdl[i] = {CMULT}({CMULT}(_gains[sh_buf[idx+0]], {CONJ}(_gains[sh_buf[idx+1]])), _ubls[sh_buf[idx+2]]);
                //if (px == 0 && i == 0) {{
                //    printf("(%d, %d, %d)\\n", sh_buf[idx], sh_buf[idx+1], sh_buf[idx+2]);
                //    printf("new g%d: (%20.18f + 1j %20.18f)\\n", sh_buf[idx], _gains[sh_buf[idx+0]].x, _gains[sh_buf[idx+0]].y);
                //    printf("new g%d: (%20.18f + 1j %20.18f)\\n", sh_buf[idx+1], _gains[sh_buf[idx+1]].x, _gains[sh_buf[idx+1]].y);
                //    printf("new ubl%d: (%20.18f + 1j %20.18f)\\n", sh_buf[idx+2], _ubls[sh_buf[idx+2]].x, _ubls[sh_buf[idx+2]].y);
                //    printf("new ubl%dsum: (%20.18f + 1j %20.18f)\\n", sh_buf[idx+2], _usum[sh_buf[idx+2]].x, _usum[sh_buf[idx+2]].y);
                //    printf("new ubl%dwgt: %20.18f\\n", sh_buf[idx+2], _uwgt[sh_buf[idx+2]]);
                //}}
            }}

        }} else {{
            // Slow branch when we compute convergence/chisq
            _conv_sum = 0;
            _conv_wgt = 0;
            for (i=0; i < {NANTS}; i++) {{
                //_gains_new[i] = _gains[i] * (1 - {GAIN}) + {GAIN} * _gsum[i] / _gwgt[i];
                wc.x = (1 - {GAIN}) + {GAIN} * _gsum[i].x / _gwgt[i];
                wc.y = {GAIN} * _gsum[i].y / _gwgt[i];
                _gains_new[i] = {CMULT}(_gains[i], wc);
                //_gains_new[i].x = _gains[i].x * (1 - {GAIN}) + {GAIN} * _gsum[i].x / _gwgt[i];
                //_gains_new[i].y = _gains[i].y * (1 - {GAIN}) + {GAIN} * _gsum[i].y / _gwgt[i];
                wc = {CSUB}(_gains_new[i], _gains[i]);
                //_conv_sum += pow(abs(_gains_new[i] - _gains[i]), 2);
                //_conv_wgt += pow(abs(_gains_new[i]), 2);
                _conv_sum += mag(wc);
                _conv_wgt += mag(_gains_new[i]);
            }}
            for (i=0; i < {NUBLS}; i++) {{
                //_ubls_new[i] = _ubls[i] * (1 - {GAIN}) + {GAIN} * _usum[i] / _uwgt[i];
                wc.x = (1 - {GAIN}) + {GAIN} * _usum[i].x / _uwgt[i];
                wc.y = {GAIN} * _usum[i].y / _uwgt[i];
                _ubls_new[i] = {CMULT}(_ubls[i], wc);
                //_ubls_new[i].x = _ubls[i].x * (1 - {GAIN}) + {GAIN} * _usum[i].x / _uwgt[i];
                //_ubls_new[i].y = _ubls[i].y * (1 - {GAIN}) + {GAIN} * _usum[i].y / _uwgt[i];
                wc = {CSUB}(_ubls_new[i], _ubls[i]);
                //_conv_sum += pow(abs(_ubls_new[i] - _ubls[i]), 2);
                //_conv_wgt += pow(abs(_ubls_new[i]), 2);
                _conv_sum += mag(wc);
                _conv_wgt += mag(_gains_new[i]);
            }}
            _chisq_new = 0;
            for (i=0; i < {NBLS}; i++) {{
                idx = 3 * i;
                //_dmdl[i] = _gains_new[sh_buf[idx+0]] * conj(_gains_new[sh_buf[idx+1]]) * _ubls_new[sh_buf[idx+2]];
                _dmdl[i] = {CMULT}({CMULT}(_gains_new[sh_buf[idx+0]], {CONJ}(_gains_new[sh_buf[idx+1]])), _ubls_new[sh_buf[idx+2]]);
                //_chisq_new += pow(abs(_data[i] - _dmdl[i]), 2) * _wgts[i];
                wc = {CSUB}(_data[i], _dmdl[i]);
                _chisq_new += mag(wc) * _wgts[i];
                //if (px == 0 && i == 0) {{
                //    printf("(%d, %d, %d)\\n", sh_buf[idx], sh_buf[idx+1], sh_buf[idx+2]);
                //    printf("new g%d: (%20.18f + 1j %20.18f)\\n", sh_buf[idx], _gains[sh_buf[idx+0]].x, _gains[sh_buf[idx+0]].y);
                //    printf("new g%d: (%20.18f + 1j %20.18f)\\n", sh_buf[idx+1], _gains[sh_buf[idx+1]].x, _gains[sh_buf[idx+1]].y);
                //    printf("new ubl%d: (%20.18f + 1j %20.18f)\\n", sh_buf[idx+2], _ubls[sh_buf[idx+2]].x, _ubls[sh_buf[idx+2]].y);
                //    printf("new ubl%dsum: (%20.18f + 1j %20.18f)\\n", sh_buf[idx+2], _usum[sh_buf[idx+2]].x, _usum[sh_buf[idx+2]].y);
                //    printf("new ubl%dwgt: %20.18f\\n", sh_buf[idx+2], _uwgt[sh_buf[idx+2]]);
                //}}
            }}
            //if (px == 0) printf("Calculated new X2: %.18e\\n", _chisq_new);

            // Check if we've diverged
            if (_chisq_new > _chisq) {{
                //if (px == 0) printf("DIVERGED\\n");
                break;
            }}
            _conv = sqrt(_conv_sum / _conv_wgt);
            //if (px == 0) printf("conv: %.18e\\n", _conv);

            // Check if we've converged
            if (_conv <= conv_crit) {{
                //if (px == 0) printf("CONVERGED\\n");
                break;
            }}

            // Update solution and soldier on
            for (i=0; i < {NANTS}; i++) {{
                _gains[i] = _gains_new[i];
            }}
            for (i=0; i < {NUBLS}; i++) {{
                _ubls[i] = _ubls_new[i];
            }}
            _chisq = _chisq_new;
            //if (px == 0) printf("Updating X2 to %.18e\\n", _chisq_new);
        }}
    }}
    
    // Prepare return values
    chisq[px] = _chisq;
    iters[px] = _iters;
    conv[px] = _conv;
    for (i=0; i < {NANTS}; i++) {{
        gains[i * {NDATA} + px] = _gains[i];
                //if (px == 0) {{
                //    printf("FINAL g%d: (%20.18f + 1j %20.18f) == (%20.18f + 1j %20.18f)\\n", i, _gains[i].x, _gains[i].y, gains[i * {NDATA} + px].x, gains[i * {NDATA} + px].y);
                //}}
    }}
    for (i=0; i < {NUBLS}; i++) {{
        ubls[i * {NDATA} + px] = _ubls[i];
    }}
    __syncthreads(); // make sure everyone used mem before kicking out
}}
"""

NTHREADS = 1024 # make 512 for smaller GPUs
MAX_MEMORY = 2**29 # floats (4B each)
MIN_CHUNK = 8

def omnical(ggu_indices, gains, ubls, data, wgts, 
            conv_crit, maxiter, check_every, check_after,
            nthreads=NTHREADS, max_memory=MAX_MEMORY,
            precision=1, gain=0.3, verbose=False):
    '''
        ggu_indices: (Nbls,3) array of (i,j,k) indices denoting data order
                     as gains[i] * gains[j].conj() * ubl[k]
        gains: (Nants,Ndata) array of estimated complex gains
        ubls: (Nubls, Ndata) array of estimated complex unique baselines
        data: (Nbls, Ndata) array of data to be calibrated
        wgts: (Nbls, Ndata) array of weights for each data
        conv_crit:
        maxiter:
        check_every:
        check_after:
    '''
    nbls = ggu_indices.shape[0]
    assert ggu_indices.shape == (nbls, 3)
    ndata = data.shape[1]
    assert data.shape == (nbls, ndata)
    assert wgts.shape == (nbls, ndata)
    nants = gains.shape[0]
    assert gains.shape == (nants, ndata)
    nubls = ubls.shape[0]
    assert ubls.shape == (nubls, ndata)
    assert(precision in (1,2))
    if precision == 1:
        real_dtype, complex_dtype = np.float32, np.complex64
        DTYPE, CDTYPE = 'float', 'cuFloatComplex'
        CMULT, CONJ, CSUB, CDIV = 'cuCmulf', 'cuConjf', 'cuCsubf', 'cuCdivf'
    else:
        real_dtype, complex_dtype = np.float64, np.complex128
        DTYPE, CDTYPE = 'double', 'cuDoubleComplex'
        CMULT, CONJ, CSUB, CDIV = 'cuCmul', 'cuConj', 'cuCsub', 'cuCdiv'
    assert check_every > 1

    # ensure data types
    ggu_indices = ggu_indices.astype(np.uint32)
    data = data.astype(complex_dtype)
    wgts = wgts.astype(real_dtype)
    gains = gains.astype(complex_dtype)
    ubls = ubls.astype(complex_dtype)
    total_mem_usage = (7 * nants + 7 * nubls + 7 * nbls) * ndata
    max_cores = min(nthreads, 2**int(ceil(np.log2(max_memory / 2 / total_mem_usage))))
    max_cores = 256 # XXX figure out how to get here algorithmically
    # blocks of threads are mapped to (ndata,)
    block = (max_cores, 1, 1)
    grid = (int(ceil(ndata/block[0])),1)
    
    # Choose to use single or double precision CUDA code
    gpu_code = GPU_TEMPLATE.format(**{
            'NDATA': ndata,
            'NBLS': nbls,
            'NUBLS': nubls,
            'NANTS': nants,
            'GAIN': gain,
            'CMULT': CMULT,
            'CONJ': CONJ,
            'CSUB': CSUB,
            'CDIV': CDIV,
            'DTYPE': DTYPE,
            'CDTYPE': CDTYPE,
    })

    gpu_module = compiler.SourceModule(gpu_code)
    omnical_cuda = gpu_module.get_function("omnical")
    total_mem_usage = omnical_cuda.local_size_bytes // 4
    #h = cublasCreate() # handle for managing cublas

    # define GPU buffers and transfer initial values
    ggu_indices_gpu = gpuarray.empty(shape=ggu_indices.shape, dtype=np.uint32)
    gains_gpu = gpuarray.empty(shape=gains.shape, dtype=complex_dtype)
    ubls_gpu = gpuarray.empty(shape=ubls.shape, dtype=complex_dtype)
    data_gpu = gpuarray.empty(shape=data.shape, dtype=complex_dtype)
    wgts_gpu = gpuarray.empty(shape=wgts.shape, dtype=real_dtype)
    chisq_gpu = gpuarray.empty(shape=(ndata,), dtype=real_dtype)
    iters_gpu = gpuarray.empty(shape=(ndata,), dtype=np.uint32)
    conv_gpu = gpuarray.empty(shape=(ndata,), dtype=real_dtype)

    #stream = driver.Stream()
    chisq = np.empty((ndata,), dtype=real_dtype)
    conv = np.empty((ndata,), dtype=real_dtype)
    iters = np.empty((ndata,), dtype=np.uint32)
    
    ggu_indices_gpu.set_async(ggu_indices)
    gains_gpu.set_async(gains)
    ubls_gpu.set_async(ubls)
    data_gpu.set_async(data)
    wgts_gpu.set_async(wgts)
    
    omnical_cuda(ggu_indices_gpu, gains_gpu, ubls_gpu, data_gpu, wgts_gpu,
            chisq_gpu, iters_gpu, conv_gpu, np.float32(conv_crit), 
            np.uint32(maxiter), np.uint32(check_every), np.uint32(check_after),
            grid=grid, block=block)

    gains_gpu.get_async(ary=gains)
    ubls_gpu.get_async(ary=ubls)
    chisq_gpu.get_async(ary=chisq)
    iters_gpu.get_async(ary=iters)
    conv_gpu.get_async(ary=conv)

    # teardown GPU configuration
    #cublasDestroy(h)
    return {'gains':gains, 'ubls':ubls, 'chisq':chisq, 'iters':iters,
            'conv':conv}

class OmnicalSolverGPU(OmnicalSolver):
    def solve_iteratively(self, conv_crit=1e-10, maxiter=50, check_every=4, check_after=1, verbose=False):
        sol = self.sol0
        terms = [(linsolve.get_name(gi), linsolve.get_name(gj), linsolve.get_name(uij))
                  for term in self.all_terms for (gi, gj, uij) in term]
        gain_map = {}
        ubl_map = {}
        for gi,gj,uij in terms:
            if not gi in gain_map:
                gain_map[gi] = len(gain_map)
            if not gj in gain_map:
                gain_map[gj] = len(gain_map)
            if not uij in ubl_map:
                ubl_map[uij] = len(ubl_map)
        ggu_indices = np.array([(gain_map[gi], gain_map[gj], ubl_map[uij]) for (gi, gj, uij) in terms], dtype=np.uint)
        v = sol[gi]
        shape, dtype, ndata = v.shape, v.dtype, v.size
        assert dtype in (np.complex64, np.complex128)
        if dtype == np.complex64:
            precision = 1
        else:
            precision = 2
        gains = np.empty((len(gain_map), ndata), dtype=dtype)
        for k,v in gain_map.items():
            gains[v] = sol[k].flatten()
        ubls = np.empty((len(ubl_map), ndata), dtype=dtype)
        for k,v in ubl_map.items():
            ubls[v] = sol[k].flatten()
        data = np.array([self.data[k].flatten() for k in self.keys])
        wgts = np.array([self.wgts[k].flatten() for k in self.keys])
        if wgts.shape != data.shape:
            wgts = np.resize(wgts, data.shape)
        result = omnical(ggu_indices, gains, ubls, data, wgts, 
            conv_crit, maxiter, check_every, check_after,
            nthreads=NTHREADS, max_memory=MAX_MEMORY,
            precision=precision, gain=self.gain, verbose=verbose)
        for k,v in gain_map.items():
            sol[k] = np.reshape(result['gains'][v], shape)
        for k,v in ubl_map.items():
            sol[k] = np.reshape(result['ubls'][v], shape)
        meta = {
            'iter': np.reshape(result['iters'], shape),
            'chisq': np.reshape(result['chisq'], shape),
            'conv_crit': np.reshape(result['conv'], shape),
        }
        return meta, sol

class RedundantCalibratorGPU(RedundantCalibrator):
    def omnical_gpu(self, data, sol0, wgts={}, gain=.3, conv_crit=1e-10, maxiter=50, check_every=4, check_after=1):
        """Use the Liu et al 2010 Omnical algorithm to linearize equations      and iteratively minimize chi^2.
 
        Args:
            data: visibility data in the dictionary format {(ant1,ant2,pol)     : np.array}
            sol0: dictionary of guess gains and unique model visibilities,      keyed by antenna tuples
                like (ant,antpol) or baseline tuples like. Gains should inc     lude firstcal gains.
            wgts: dictionary of linear weights in the same format as data.      Defaults to equal wgts.
            conv_crit: maximum allowed relative change in solutions to be c     onsidered converged
            maxiter: maximum number of omnical iterations allowed before it      gives up
            check_every: Compute convergence every Nth iteration (saves com     putation).  Default 4.
            check_after: Start computing convergence only after N iteration     s.  Default 1.
            gain: The fractional step made toward the new solution each ite     ration.  Default is 0.3.
                Values in the range 0.1 to 0.5 are generally safe.  Increas     ing values trade speed
                for stability.
 
        Returns:
            meta: dictionary of information about the convergence and chi^2      of the solution
            sol: dictionary of gain and visibility solutions in the {(index     ,antpol): np.array}
                and {(ind1,ind2,pol): np.array} formats respectively
        """
 
        sol0 = {self.pack_sol_key(k): sol0[k] for k in sol0.keys()}
        ls = self._solver(OmnicalSolverGPU, data, sol0=sol0, wgts=wgts, gain=gain)
        meta, sol = ls.solve_iteratively(conv_crit=conv_crit, maxiter=maxiter, check_every=check_every, check_after=check_after)
        sol = {self.unpack_sol_key(k): sol[k] for k in sol.keys()}
        return meta, sol
