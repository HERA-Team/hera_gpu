'''Module for accelerating hera_cal.redcal implementations of Omnical.'''

import pycuda.autoinit
from pycuda import compiler, gpuarray, driver
from skcuda.cublas import cublasCreate, cublasDestroy
from skcuda.cublas import cublasCcopy, cublasZcopy
import numpy as np
from math import ceil, floor
import time
from hera_cal.redcal import OmnicalSolver, RedundantCalibrator
from linsolve import get_name

# GPU parameters
NTHREADS = 1024 # make 512 for smaller GPUs
MAX_MEMORY = 2**29 # floats (4B each), currently unused
MAX_REGISTERS = 2**16 # number of registers, currently unused
MIN_CHUNK = 8
TIME_IT = False

GPU_TEMPLATE = """
// CUDA code for redundant calibration
//
// Arrays should be shaped as (px, NBLS), (px, NANT), etc,
// so offsets are not dependent on pixels, but only on NBLS/NANTS/etc.
// Within kernels, x-axis parallelizes over separate omnical optimizations;
// y-axis parallelizes computations within an omnical calculation.
//
// === Template Parameters ===
// "NBLS": number of baselines in data/dmdl/wgts, etc.
// "NUBLS": number of unique baselines in ubls/ubuf/uwgt, etc.
// "NANTS": number of antenna in gains/gbuf/gwgt, etc.
// "GAIN": fractional step between iterations
// "CMULT"  : cuCmulf or cuCmul
// "CSUB"  : cuCsubf or cuCsub
// "CONJ"  : cuConjf or cuConj
// "DTYPE"  : float or double
// "CDTYPE"  : cuFloatComplex or cuDoubleComplex

#include <cuComplex.h>
#include <pycuda-helpers.hpp>
#include <stdio.h>

__device__ inline {DTYPE} mag2({CDTYPE} a) {{
    return a.x * a.x + a.y * a.y;
}}

__global__ void gen_dmdl(uint *ggu_indices, {CDTYPE} *gains,
                         {CDTYPE} *ubls, {CDTYPE} *dmdl, uint *active)
// Calculate gi * gj.conj * ubl for each baseline in data
{{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint td = blockIdx.y * blockDim.y + threadIdx.y;
    const uint ant_offset = px * {NANTS};
    const uint ubl_offset = px * {NUBLS};
    const uint bls_idx = px * {NBLS} + td;
    uint idx;

    if (td < {NBLS} && active[px]) {{
        idx = 3 * td;
        dmdl[bls_idx] = {CMULT}(
            {CMULT}(
                gains[ant_offset + ggu_indices[idx+0]], 
                {CONJ}(gains[ant_offset + ggu_indices[idx+1]])
            ), ubls[ubl_offset + ggu_indices[idx+2]]
        );
    }}
}}


__global__ void calc_chisq({CDTYPE} *data, {CDTYPE} *dmdl, {DTYPE} *wgts,
                           {DTYPE} *chisq, uint *active)
// Calculate weighted X^2 difference between data and dmdl
{{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint td = blockIdx.y * blockDim.y + threadIdx.y;
    const uint bls_idx = px * {NBLS} + td;
    {CDTYPE} diff;

    if (td < {NBLS} && active[px]) {{
        diff = {CSUB}(data[bls_idx], dmdl[bls_idx]);
        atomicAdd(&chisq[px], mag2(diff) * wgts[bls_idx]);
    }}
}}


__global__ void calc_dwgts({CDTYPE} *dmdl, {DTYPE} *wgts, {DTYPE} *dwgts,
                           uint *active)
// Calculate per-baseline weights as |dmdl|^2 * wgt
{{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint td = blockIdx.y * blockDim.y + threadIdx.y;
    const uint bls_idx = px * {NBLS} + td;

    if (td < {NBLS} && active[px]) {{
        dwgts[bls_idx] = mag2(dmdl[bls_idx]) * wgts[bls_idx];
    }}
}}


__global__ void clear_complex({CDTYPE} *buf, uint len)
// Clear a complex buffer of length 'len'
{{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px < len) {{
        buf[px] = make_{CDTYPE}(0, 0);
    }}
}}


__global__ void clear_real({DTYPE} *buf, uint len, {DTYPE} val)
// Fill a real-valued buffer of length 'len' with value 'val'
{{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px < len) {{
        buf[px] = val;
    }}
}}


__global__ void clear_uint(uint *buf, uint len, uint val)
// Fill a uint-valued buffer of length 'len' with value 'val'
{{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px < len) {{
        buf[px] = val;
    }}
}}
    

__global__ void calc_gu_wgt(uint *ggu_indices, {CDTYPE} *dmdl, 
                            {DTYPE} *dwgts, {DTYPE} *gwgt, 
                            {DTYPE} *uwgt, uint *active)
// Calculate the gain/ubl weights used in denominator of estimates of the
// next gain/ubl values. Not updated every loop. Compute intense.
{{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint td = blockIdx.y * blockDim.y + threadIdx.y;
    const uint ant_offset = px * {NANTS};
    const uint ubl_offset = px * {NUBLS};
    const uint bls_idx = px * {NBLS} + td;
    uint idx;
    {DTYPE} w;

    if (td < {NBLS} && active[px]) {{
        idx = 3 * td;
        w = dwgts[bls_idx];
        atomicAdd(&gwgt[ant_offset + ggu_indices[idx+0]], w);
        atomicAdd(&gwgt[ant_offset + ggu_indices[idx+1]], w);
        atomicAdd(&uwgt[ubl_offset + ggu_indices[idx+2]], w);
    }}
}}


__global__ void calc_gu_buf(uint *ggu_indices, {CDTYPE} *data,
                            {DTYPE} *dwgts, {CDTYPE} *dmdl,
                            {CDTYPE} *gbuf, {CDTYPE} *ubuf, uint *active)
// Calculate the gain/ubl numerator used to estimate next gain/ubl values.
// Updated every loop. Most compute-intensive step by factor of 3.
{{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint td = blockIdx.y * blockDim.y + threadIdx.y;
    const uint ant_offset = px * {NANTS};
    const uint ubl_offset = px * {NUBLS};
    const uint bls_idx = px * {NBLS} + td;
    uint idx, i;
    {CDTYPE} d;
    {DTYPE} w;

    if (td < {NBLS} && active[px]) {{
        idx = 3 * td;
        w = dwgts[bls_idx] / mag2(dmdl[bls_idx]);
        d = {CMULT}(data[bls_idx], {CONJ}(dmdl[bls_idx]));
        d.x = d.x * w;
        d.y = d.y * w;
        i = ant_offset + ggu_indices[idx];
        atomicAdd(&gbuf[i].x,  d.x);
        atomicAdd(&gbuf[i].y,  d.y);
        i = ant_offset + ggu_indices[idx+1];
        atomicAdd(&gbuf[i].x,  d.x);
        atomicAdd(&gbuf[i].y, -d.y); // Note: conjugate happens here.
        i = ubl_offset + ggu_indices[idx+2];
        atomicAdd(&ubuf[i].x,  d.x);
        atomicAdd(&ubuf[i].y,  d.y);
    }}
}}


__global__ void update_gains({CDTYPE} *gbuf, {DTYPE} *gwgt,
                             {CDTYPE} *gains, float gstep, uint *active)
// Given accumated values in gbuf and gwgt and step size gstep, compute
// new gains for next iteration.
{{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint td = blockIdx.y * blockDim.y + threadIdx.y;
    const uint ant_idx = px * {NANTS} + td;
    {CDTYPE} wc;

    if (td < {NANTS} && active[px]) {{
        wc.x = (1 - gstep) + gstep * gbuf[ant_idx].x / gwgt[ant_idx];
        wc.y = gstep * gbuf[ant_idx].y / gwgt[ant_idx];
        gains[ant_idx] = {CMULT}(gains[ant_idx], wc);
    }}
}}


__global__ void update_ubls({CDTYPE} *ubuf, {DTYPE} *uwgt,
                            {CDTYPE} *ubls, float gstep, uint *active)
// Given accumated values in ubuf and uwgt and step size gstep, compute
// new ubls for next iteration.
{{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint td = blockIdx.y * blockDim.y + threadIdx.y;
    const uint ubl_idx = px * {NUBLS} + td;
    {CDTYPE} wc;

    if (td < {NUBLS} && active[px]) {{
        wc.x = (1 - gstep) + gstep * ubuf[ubl_idx].x / uwgt[ubl_idx];
        wc.y = gstep * ubuf[ubl_idx].y / uwgt[ubl_idx];
        ubls[ubl_idx] = {CMULT}(ubls[ubl_idx], wc);
    }}
}}


__global__ void calc_conv({CDTYPE} *new_gains, {CDTYPE} *old_gains,
                          {CDTYPE} *new_ubls, {CDTYPE} *old_ubls,
                          {DTYPE} *conv_sum, {DTYPE} *conv_wgt,
                          uint *active)
// Compare old and new values of gains/ubls to determine convergence.
{{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint td = blockIdx.y * blockDim.y + threadIdx.y;
    const uint ant_idx = px * {NANTS} + td;
    const uint ubl_idx = px * {NUBLS} + td;
    {CDTYPE} wc;

    if (td < {NANTS} && active[px]) {{
        wc = {CSUB}(new_gains[ant_idx], old_gains[ant_idx]);
        atomicAdd(&conv_sum[px], mag2(wc));
        atomicAdd(&conv_wgt[px], mag2(new_gains[ant_idx]));
    }}

    if (td < {NUBLS} && active[px]) {{
        wc = {CSUB}(new_ubls[ubl_idx], old_ubls[ubl_idx]);
        atomicAdd(&conv_sum[px], mag2(wc));
        atomicAdd(&conv_wgt[px], mag2(new_ubls[ubl_idx]));
    }}
}}


__global__ void update_active({CDTYPE} *new_gains, {CDTYPE} *old_gains,
                              {CDTYPE} *new_ubls, {CDTYPE} *old_ubls,
                              {DTYPE} *conv_sum, {DTYPE} *conv_wgt,
                              {DTYPE} *conv, {DTYPE} conv_crit,
                              {DTYPE} *new_chisq, {DTYPE} *chisq,
                              uint *iters, uint i, uint *active)
// Given values of convergence and chisq, copy new gains/ubls into
// the working (old) values, update chisq, record the next iteration,
// and flag off pixels in active that have converged/diverged.
{{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint td = blockIdx.y * blockDim.y + threadIdx.y;
    const uint ant_idx = px * {NANTS} + td;
    const uint ubl_idx = px * {NUBLS} + td;
    
    if (td == 0 && active[px]) {{
        conv[px] = sqrt(conv_sum[px] / conv_wgt[px]);
        if (conv[px] < conv_crit) {{
            active[px] = 0;
        }} else if (new_chisq[px] > chisq[px]) {{
            active[px] = 0;
        }} else {{
            chisq[px] = new_chisq[px];
            iters[px] = i;
        }}
    }}

    __syncthreads(); // wait for active to be updated by master td=0

    if (td < {NANTS} && active[px]) {{
        old_gains[ant_idx] = new_gains[ant_idx];
    }}

    if (td < {NUBLS} && active[px]) {{
        old_ubls[ubl_idx] = new_ubls[ubl_idx];
    }}
}}
"""


def omnical(ggu_indices, gains, ubls, data, wgts, 
            conv_crit, maxiter, check_every, check_after,
            gain=0.3, nthreads=NTHREADS,
            precision=1, verbose=False):
    '''CPU-side function for organizing GPU-acceleration primitives into
    a cohesive omnical algorithm.
        
    Args:
        ggu_indices: (nbls,3) array of (i,j,k) indices denoting data order
                     as gains[i] * gains[j].conj() * ubl[k]
        gains: (ndata, nants) array of estimated complex gains
        ubls: (ndata, nubls) array of estimated complex unique baselines
        data: (ndata, nbls) array of data to be calibrated
        wgts: (ndata, nbls) array of weights for each data
        conv_crit: maximum allowed relative change in solutions to be 
            considered converged
        maxiter: maximum number of omnical iterations allowed before it
            gives up.
        check_every: Compute convergence every Nth iteration (saves 
            computation).  Default 4.
        check_after: Start computing convergence only after N iterations.  
            Default 1.
        gain: The fractional step made toward the new solution each 
            iteration. Values in the range 0.1 to 0.5 are generally safe.  
            Increasing values trade speed for stability.  Default is 0.3.
        nthreads: Number of GPU threads to use. Default NTHREADS=1024
        precision: 1=float32, 2=double (float64). Default 1.
        verbose: If True, print things. Default False.

    Returns:
        info dictionary of {'gains':gains, 'ubls':ubls, 'chisq':chisq, 
            'iters':iters, 'conv':conv}
    '''
    # Sanity check input array dimensions
    nbls = ggu_indices.shape[0]
    assert ggu_indices.shape == (nbls, 3)
    ndata = data.shape[0]
    assert data.shape == (ndata, nbls)
    assert wgts.shape == (ndata, nbls)
    nants = gains.shape[1]
    assert gains.shape == (ndata, nants)
    nubls = ubls.shape[1]
    assert ubls.shape == (ndata, nubls)
    assert precision in (1,2)
    assert check_every > 1
    if verbose:
        print('PRECISION:', precision)
        print('NDATA:', ndata)
        print('NANTS:', nants)
        print('NUBLS:', nubls)
        print('NBLS:', nbls)

    # Choose between float/double primitives
    if precision == 1:
        real_dtype, complex_dtype = np.float32, np.complex64
        DTYPE, CDTYPE = 'float', 'cuFloatComplex'
        CMULT, CONJ, CSUB = 'cuCmulf', 'cuConjf', 'cuCsubf'
        COPY = cublasCcopy
    else:
        real_dtype, complex_dtype = np.float64, np.complex128
        DTYPE, CDTYPE = 'double', 'cuDoubleComplex'
        CMULT, CONJ, CSUB = 'cuCmul', 'cuConj', 'cuCsub'
        COPY = cublasZcopy

    # ensure data types
    ggu_indices = ggu_indices.astype(np.uint32)
    data = data.astype(complex_dtype)
    wgts = wgts.astype(real_dtype)
    gains = gains.astype(complex_dtype)
    ubls = ubls.astype(complex_dtype)
    
    # px=64, nants=350, precision=1, nthreads//16 -> 15.5 s
    # px=128, nants=350, precision=1, nthreads//16 -> 15.3 s
    # px=256, nants=350, precision=1, nthreads//16 -> 16.3 s
    # px=512, nants=350, precision=1, nthreads//16 -> 18.1 s
    # px=1024, nants=350, precision=1, nthreads//16 -> 21.5 s
    # px=2048, nants=350, precision=1, nthreads//16 -> 28.0 s
    # (killed after that for memory)
    # Model: 15.3 s + ndata/140 => px=8192 in 75 s on Quadro T2000
    
    # px=2000, nants=126, precision=2
    #chunk_size = min(nthreads // 2, ndata) # 8.5 s
    #chunk_size = min(nthreads // 4, ndata) # 6.7 s
    #chunk_size = min(nthreads // 8, ndata) # 5.6 s
    #chunk_size = min(nthreads // 16, ndata) # 5.2 s
    #chunk_size = min(nthreads // 32, ndata) # 5.4 s
    #chunk_size = min(nthreads // 64, ndata) # 5.9 s
    # px=2000, nants=54, precision=2
    #chunk_size = min(nthreads // 2, ndata) # 1.5 s
    #chunk_size = min(nthreads // 4, ndata) # 1.3 s
    #chunk_size = min(nthreads // 8, ndata) # 1.3 s
    #chunk_size = min(nthreads // 16, ndata) # 1.5 s
    #chunk_size = min(nthreads // 32, ndata) # 1.9 s
    #chunk_size = min(nthreads // 64, ndata) # 3.0 s
    # px=2000, nants=180, precision=2
    #chunk_size = min(nthreads // 2, ndata) # 16.8 s
    #chunk_size = min(nthreads // 4, ndata) # 13.8 s
    #chunk_size = min(nthreads // 8, ndata) # 11.1 s
    #chunk_size = min(nthreads // 16, ndata) # 9.9 s
    #chunk_size = min(nthreads // 32, ndata) # 10.0 s
    #chunk_size = min(nthreads // 64, ndata) # 10.14 s
    # px=2000, nants=180, precision=1
    #chunk_size = min(nthreads // 2, ndata) # 7.6 s
    #chunk_size = min(nthreads // 4, ndata) # 8.3 s
    #chunk_size = min(nthreads // 8, ndata) # 7.4 s
    #chunk_size = min(nthreads // 16, ndata) # 7.3 s
    #chunk_size = min(nthreads // 32, ndata) # 7.4 s
    #chunk_size = min(nthreads // 64, ndata) # 7.5 s

    # Build the CUDA code
    gpu_code = GPU_TEMPLATE.format(**{
            'NBLS': nbls,
            'NUBLS': nubls,
            'NANTS': nants,
            'GAIN': gain,
            'CMULT': CMULT,
            'CONJ': CONJ,
            'CSUB': CSUB,
            'DTYPE': DTYPE,
            'CDTYPE': CDTYPE,
    })
    # Extract functions from CUDA, suffix _cuda indicates GPU operation
    gpu_module = compiler.SourceModule(gpu_code)
    gen_dmdl_cuda = gpu_module.get_function("gen_dmdl")
    calc_chisq_cuda = gpu_module.get_function("calc_chisq")
    calc_dwgts_cuda = gpu_module.get_function("calc_dwgts")
    calc_gu_wgt_cuda = gpu_module.get_function("calc_gu_wgt")
    calc_gu_buf_cuda = gpu_module.get_function("calc_gu_buf")
    clear_complex_cuda = gpu_module.get_function("clear_complex")
    clear_real_cuda = gpu_module.get_function("clear_real")
    clear_uint_cuda = gpu_module.get_function("clear_uint")
    update_gains_cuda = gpu_module.get_function("update_gains")
    update_ubls_cuda = gpu_module.get_function("update_ubls")
    calc_conv_cuda = gpu_module.get_function("calc_conv")
    update_active_cuda = gpu_module.get_function("update_active")

    h = cublasCreate() # handle for managing cublas, used for buffer copies

    # define GPU buffers, suffix _g indicates GPU buffer
    #data_chunks = int(ceil(ndata / chunk_size))
    chunk_size = min(nthreads // 16, ndata)
    data_chunks = 1
    ANT_SHAPE = (chunk_size, nants)
    UBL_SHAPE = (chunk_size, nubls)
    BLS_SHAPE = (chunk_size, nbls)
    block = (chunk_size, int(floor(nthreads/chunk_size)), 1)
    ant_grid  = (data_chunks, int(ceil(nants/block[1])))
    ubl_grid  = (data_chunks, int(ceil(nubls/block[1])))
    bls_grid  = (data_chunks, int(ceil( nbls/block[1])))
    conv_grid = (data_chunks, int(ceil(max(nants,nubls)/block[1])))
    if verbose:
        print('GPU block:', block)
        print('ANT grid:', ant_grid)
        print('UBL grid:', ubl_grid)
        print('BLS grid:', bls_grid)

    ggu_indices_g = gpuarray.empty(shape=ggu_indices.shape, dtype=np.uint32)
    active_g = gpuarray.empty(shape=(chunk_size,), dtype=np.uint32)
    iters_g  = gpuarray.empty(shape=(chunk_size,), dtype=np.uint32)
    
    gains_g     = gpuarray.empty(shape=ANT_SHAPE, dtype=complex_dtype)
    new_gains_g = gpuarray.empty(shape=ANT_SHAPE, dtype=complex_dtype)
    gbuf_g      = gpuarray.empty(shape=ANT_SHAPE, dtype=complex_dtype)
    gwgt_g      = gpuarray.empty(shape=ANT_SHAPE, dtype=real_dtype)

    ubls_g     = gpuarray.empty(shape=UBL_SHAPE, dtype=complex_dtype)
    new_ubls_g = gpuarray.empty(shape=UBL_SHAPE, dtype=complex_dtype)
    ubuf_g     = gpuarray.empty(shape=UBL_SHAPE, dtype=complex_dtype)
    uwgt_g     = gpuarray.empty(shape=UBL_SHAPE, dtype=real_dtype)

    data_g  = gpuarray.empty(shape=BLS_SHAPE, dtype=complex_dtype)
    dmdl_g  = gpuarray.empty(shape=BLS_SHAPE, dtype=complex_dtype)
    wgts_g  = gpuarray.empty(shape=BLS_SHAPE, dtype=real_dtype)
    dwgts_g = gpuarray.empty(shape=BLS_SHAPE, dtype=real_dtype)

    chisq_g     = gpuarray.empty(shape=(chunk_size,), dtype=real_dtype)
    new_chisq_g = gpuarray.empty(shape=(chunk_size,), dtype=real_dtype)
    conv_sum_g  = gpuarray.empty(shape=(chunk_size,), dtype=real_dtype)
    conv_wgt_g  = gpuarray.empty(shape=(chunk_size,), dtype=real_dtype)
    conv_g      = gpuarray.empty(shape=(chunk_size,), dtype=real_dtype)

    # Define return buffers
    chisq = np.empty((ndata,), dtype=real_dtype)
    conv  = np.empty((ndata,), dtype=real_dtype)
    iters = np.empty((ndata,), dtype=np.uint32)
    active = np.empty((chunk_size,), dtype=np.uint32)
    
    # upload data indices
    ggu_indices_g.set_async(ggu_indices)

    # initialize structures used to time code
    event_order = ('start', 'upload', 'dmdl', 'calc_chisq', 'loop_top', 'calc_gu_wgt', 'calc_gu_buf', 'copy_gains', 'copy_ubls', 'update_gains', 'update_ubls', 'dmdl2', 'chisq2', 'calc_conv', 'update_active', 'get_active', 'end')
    event_pairs = list((event_order[i],event_order[i+1])
                        for i in range(len(event_order[:-1])))
    cum_time = {}
    t0 = time.time()

    # Loop over chunks of parallel omnical problems
    for px in range(0,ndata,chunk_size):
        events = {e:driver.Event() for e in event_order}
        events['start'].record()

        end = min(ndata, px + chunk_size)
        beg = end - chunk_size
        offset = px - beg
        gains_g.set_async(gains[beg:end])
        ubls_g.set_async(ubls[beg:end])
        data_g.set_async(data[beg:end])
        wgts_g.set_async(wgts[beg:end])
        active = np.ones((chunk_size,), dtype=np.uint32)
        if offset > 0:
            active[:offset] = 0
        active_g.set_async(active)
        clear_real_cuda(conv_sum_g, np.uint32(chunk_size), real_dtype(0),
                        block=(chunk_size,1,1), grid=(1,1))
        clear_real_cuda(conv_wgt_g, np.uint32(chunk_size), real_dtype(0),
                        block=(chunk_size,1,1), grid=(1,1))
        events['upload'].record()

        gen_dmdl_cuda(ggu_indices_g, gains_g, ubls_g, dmdl_g, active_g,
                      block=block, grid=bls_grid)
        events['dmdl'].record()

        clear_real_cuda(chisq_g, np.uint32(chunk_size), real_dtype(0),
                        block=(chunk_size,1,1), grid=(1,1))
        calc_chisq_cuda(data_g, dmdl_g, wgts_g, chisq_g, active_g,
                        block=block, grid=bls_grid)
        events['calc_chisq'].record()

        if TIME_IT:
            events['calc_chisq'].synchronize()
            for (e1,e2) in event_pairs[:3]:
                cum_time[(e1,e2)] = cum_time.get((e1,e2), 0) + \
                                    events[e2].time_since(events[e1])

        # Loop over iterations within an omnical problem
        for i in range(1,maxiter+1):
            events['loop_top'].record()

            if (i % check_every) == 1:
                # Per standard omnical algorithm, only update gwgt/uwgt
                # every few iterations to save compute.
                calc_dwgts_cuda(dmdl_g, wgts_g, dwgts_g,
                    active_g, block=block, grid=bls_grid)
                clear_real_cuda(gwgt_g, np.uint32(nants*chunk_size),
                    real_dtype(0), block=(nthreads,1,1),
                    grid=(int(ceil(nants * chunk_size / nthreads)),1))
                clear_real_cuda(uwgt_g, np.uint32(nubls*chunk_size),
                    real_dtype(0), block=(nthreads,1,1),
                    grid=(int(ceil(nubls * chunk_size / nthreads)),1))
                calc_gu_wgt_cuda(ggu_indices_g, dmdl_g, dwgts_g, 
                    gwgt_g, uwgt_g, active_g,
                    block=block, grid=bls_grid)
            events['calc_gu_wgt'].record()

            clear_complex_cuda(gbuf_g, np.uint32(nants*chunk_size),
                block=(nthreads,1,1),
                grid=(int(ceil(nants * chunk_size / nthreads)),1))
            clear_complex_cuda(ubuf_g, np.uint32(nubls*chunk_size),
                block=(nthreads,1,1),
                grid=(int(ceil(nubls * chunk_size / nthreads)),1))
            # This is 75% of the compute load
            calc_gu_buf_cuda(ggu_indices_g, 
                data_g, dwgts_g, dmdl_g, 
                gbuf_g, ubuf_g, active_g, 
                block=block, grid=bls_grid)
            events['calc_gu_buf'].record()

            if (i < maxiter) and (i<check_after or (i%check_every != 0)):
                # Fast branch: don't check convergence/divergence
                events['copy_gains'].record()
                events['copy_ubls'].record()

                update_gains_cuda(gbuf_g, gwgt_g, gains_g, 
                    np.float32(gain), active_g, 
                    block=block, grid=ant_grid)
                events['update_gains'].record()

                update_ubls_cuda(ubuf_g, uwgt_g, ubls_g, 
                    np.float32(gain), active_g, 
                    block=block, grid=ubl_grid)
                events['update_ubls'].record()

                gen_dmdl_cuda(ggu_indices_g, gains_g, ubls_g, 
                    dmdl_g, active_g, 
                    block=block, grid=bls_grid)
                events['dmdl2'].record()

                events['chisq2'].record()
                events['calc_conv'].record()
                events['update_active'].record()
                events['get_active'].record()

            else:
                # Slow branch: check convergence/divergence
                COPY(h, nants*chunk_size,
                    gains_g.gpudata, 1,
                    new_gains_g.gpudata, 1)
                events['copy_gains'].record()

                COPY(h, nubls*chunk_size,
                    ubls_g.gpudata, 1,
                    new_ubls_g.gpudata, 1)
                events['copy_ubls'].record()

                update_gains_cuda(gbuf_g, gwgt_g, new_gains_g, 
                    np.float32(gain), active_g, 
                    block=block, grid=ant_grid)
                events['update_gains'].record()

                update_ubls_cuda(ubuf_g, uwgt_g, new_ubls_g, 
                    np.float32(gain), active_g, 
                    block=block, grid=ubl_grid)
                events['update_ubls'].record()

                gen_dmdl_cuda(ggu_indices_g, new_gains_g, new_ubls_g,
                    dmdl_g, active_g,
                    block=block, grid=bls_grid)
                events['dmdl2'].record()

                clear_real_cuda(new_chisq_g, np.uint32(chunk_size),
                    real_dtype(0), block=(chunk_size,1,1), grid=(1,1))
                calc_chisq_cuda(data_g, dmdl_g, wgts_g,
                    new_chisq_g, active_g,
                    block=block, grid=bls_grid)
                events['chisq2'].record()

                clear_real_cuda(conv_sum_g, np.uint32(chunk_size),
                    real_dtype(0), block=(chunk_size,1,1), grid=(1,1))
                clear_real_cuda(conv_wgt_g, np.uint32(chunk_size),
                    real_dtype(0), block=(chunk_size,1,1), grid=(1,1))
                calc_conv_cuda(new_gains_g, gains_g,
                    new_ubls_g, ubls_g,
                    conv_sum_g, conv_wgt_g, active_g,
                    block=block, grid=conv_grid)
                events['calc_conv'].record()

                update_active_cuda(new_gains_g, gains_g,
                    new_ubls_g, ubls_g,
                    conv_sum_g, conv_wgt_g,
                    conv_g, real_dtype(conv_crit),
                    new_chisq_g, chisq_g,
                    iters_g, np.uint32(i),
                    active_g,
                    block=block, grid=conv_grid)
                events['update_active'].record()

                active_g.get_async(ary=active)
                events['get_active'].record()

                if not np.any(active):
                    break
            events['end'].record()

            if TIME_IT:
                events['end'].synchronize()
                for (e1,e2) in event_pairs[4:]:
                    cum_time[(e1,e2)] = cum_time.get((e1,e2), 0) + \
                                        events[e2].time_since(events[e1])

        # Download final answers into buffers returned to user
        # use offset to trim off parts of chunk that were never active
        _chisq = np.empty((chunk_size,), dtype=real_dtype)
        chisq_g.get_async(ary=_chisq)
        chisq[px:end] = _chisq[offset:]
        _iters = np.empty((chunk_size,), dtype=np.uint32)
        iters_g.get_async(ary=_iters)
        iters[px:end] = _iters[offset:]
        _conv = np.empty((chunk_size,), dtype=real_dtype)
        conv_g.get_async(ary=_conv)
        conv[px:end] = _conv[offset:]
        _gains = np.empty(ANT_SHAPE, dtype=complex_dtype)
        gains_g.get_async(ary=_gains)
        gains[px:end,:] = _gains[offset:,:]
        _ubls = np.empty(UBL_SHAPE, dtype=complex_dtype)
        ubls_g.get_async(ary=_ubls)
        ubls[px:end,:] = _ubls[offset:,:]

    t1 = time.time()
    if TIME_IT:
        print('Final, nthreads=%d' % nthreads)
        for (e1,e2) in event_pairs:
            try:
                print('%6.3f' % cum_time[(e1,e2)], e1, e2)
            except(KeyError):
                pass
        print(t1 - t0)
        print()
    
    cublasDestroy(h) # teardown GPU configuration

    return {'gains':gains, 'ubls':ubls, 'chisq':chisq, 'iters':iters,
            'conv':conv}


class OmnicalSolverGPU(OmnicalSolver):
    def solve_iteratively(self, conv_crit=1e-10, maxiter=50,
            check_every=4, check_after=1, precision=None, verbose=False):
        """Repeatedly solves and updates solution until convergence or 
        maxiter is reached.  Returns a meta-data about the solution and 
        the solution itself. THIS IS A DROP_IN REPLACEMENT FOR
        hera_cal.redcal.OmnicalSolver.solve_iteratively

        Args:
            conv_crit: A convergence criterion (default 1e-10) below which 
                to stop iterating.  Converegence is measured L2-norm of 
                the change in the solution of all the variables divided by 
                the L2-norm of the solution itself.
            maxiter: An integer maximum number of iterations to perform 
                before quitting. Default 50.
            check_every: Compute convergence and updates weights every Nth 
                iteration (saves computation). Default 4.
            check_after: Start computing convergence and updating weights 
                after the first N iterations.  Default 1.

        Returns: meta, sol
            meta: a dictionary with metadata about the solution, including
                iter: the number of iterations taken to reach convergence 
                    (or maxiter), with dimensions of the data.
                chisq: the chi^2 of the solution produced by the final 
                    iteration, with dimensions of the data.
                conv_crit: the convergence criterion evaluated at the final
                    iteration, with dimensions of the data.
            sol: a dictionary of complex solutions with variables as keys, 
                    with dimensions of the data.
        """
        sol = self.sol0
        terms = [(get_name(gi), get_name(gj), get_name(uij))
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
        ggu_indices = np.array([(gain_map[gi], gain_map[gj], ubl_map[uij]) 
                            for (gi, gj, uij) in terms], dtype=np.uint)
        v = sol[gi]
        shape, dtype, ndata = v.shape, v.dtype, v.size
        ngains = len(gain_map)
        nubls = len(ubl_map)
        nbls = len(self.keys)
        assert dtype in (np.complex64, np.complex128)
        if precision is None:
            if dtype == np.complex64:
                precision = 1
            else:
                precision = 2
        if precision == 1:
            real_dtype = np.float32
        else:
            real_dtype = np.float64
        gains = np.empty((ndata, ngains), dtype=dtype)
        for k,v in gain_map.items():
            gains[:,v] = sol[k].flatten()
        ubls = np.empty((ndata, nubls), dtype=dtype)
        for k,v in ubl_map.items():
            ubls[:,v] = sol[k].flatten()
        data = np.empty((ndata, nbls), dtype=dtype)
        wgts = np.empty((ndata, nbls), dtype=real_dtype)
        for i,k in enumerate(self.keys):
            data[:,i] = self.data[k].flatten()
            wgts[:,i] = self.wgts[k].flatten()
        #data = np.array([self.data[k].flatten() for k in self.keys])
        #wgts = np.array([self.wgts[k].flatten() for k in self.keys])
        if wgts.shape != data.shape:
            wgts = np.resize(wgts, data.shape)
        result = omnical(ggu_indices, gains, ubls, data, wgts, 
            conv_crit, maxiter, check_every, check_after,
            nthreads=NTHREADS, precision=precision, gain=self.gain, 
            verbose=verbose)
        for k,v in gain_map.items():
            sol[k] = np.reshape(result['gains'][:,v], shape)
        for k,v in ubl_map.items():
            sol[k] = np.reshape(result['ubls'][:,v], shape)
        meta = {
            'iter': np.reshape(result['iters'], shape),
            'chisq': np.reshape(result['chisq'], shape),
            'conv_crit': np.reshape(result['conv'], shape),
        }
        return meta, sol

class RedundantCalibratorGPU(RedundantCalibrator):
    def omnical_gpu(self, data, sol0, wgts={}, gain=.3, conv_crit=1e-10, maxiter=50, check_every=4, check_after=1, precision=None):
        """Use the Liu et al 2010 Omnical algorithm to linearize 
        equations and iteratively minimize chi^2. THIS IS A DROP-IN
        REPLACEMENT FOR hera_cal.redcal.RedundantCalibrator.omnical
 
        Args:
            data: visibility data in the dictionary format 
                {(ant1,ant2,pol): np.array}
            sol0: dictionary of guess gains and unique model visibilities,
                keyed by antenna tuples like (ant,antpol) or baseline 
                tuples like. Gains should include firstcal gains.
            wgts: dictionary of linear weights in the same format as data.
                Defaults to equal wgts.
            gain: The fractional step made toward the new solution each 
                iteration. Values in the range 0.1 to 0.5 are generally 
                safe. Increasing values trade speed for stability.  
                Default is 0.3.
            conv_crit: maximum allowed relative change in solutions to be 
                considered converged
            maxiter: maximum number of omnical iterations allowed before it
                gives up
            check_every: Compute convergence every Nth iteration (saves 
                computation).  Default 4.
            check_after: Start computing convergence only after N 
                iterations.  Default 1.
            precision: 1=float32, 2=double (float64), None=determined from 
                data. Default None.
 
        Returns:
            meta: dictionary of information about the convergence and 
                chi^2 of the solution
            sol: dictionary of gain and visibility solutions in the 
                {(index, antpol): np.array} and 
                {(ind1,ind2,pol): np.array} formats respectively
        """
 
        sol0 = {self.pack_sol_key(k): sol0[k] for k in sol0.keys()}
        ls = self._solver(OmnicalSolverGPU, data, sol0=sol0, wgts=wgts,
                          gain=gain)
        meta, sol = ls.solve_iteratively(conv_crit=conv_crit, 
                            maxiter=maxiter, check_every=check_every,
                            check_after=check_after, precision=precision)
        sol = {self.unpack_sol_key(k): sol[k] for k in sol.keys()}
        return meta, sol
