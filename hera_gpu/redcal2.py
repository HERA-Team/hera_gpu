import pycuda.autoinit
from pycuda import compiler, gpuarray, driver
from skcuda.cublas import cublasCreate, cublasSetStream, cublasDestroy, cublasCcopy, cublasZcopy
import numpy as np
from math import ceil, floor
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

__device__ inline {DTYPE} mag2({CDTYPE} a) {{
    return a.x * a.x + a.y * a.y;
}}

// Arrays should be shaped as (px, NBLS), (px, NANT), etc,
// so offsets are not dependent on pixels, but only on NBLS/NANTS/etc

__global__ void gen_dmdl(uint *ggu_indices, {CDTYPE} *gains, {CDTYPE} *ubls, {CDTYPE} *dmdl, uint *active)
{{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint bl = blockIdx.y * blockDim.y + threadIdx.y;
    const uint ant_offset = px * {NANTS};
    const uint ubl_offset = px * {NUBLS};
    const uint bls_idx = px * {NBLS} + bl;
    uint idx;

    //printf("px:%d, bl:%d\\n", px, bl);
    //printf("bls_idx:%d, (%d, %d)\\n", bls_idx, bls_idx/{NBLS}, bls_idx % {NBLS});
    if (bl < {NBLS} && active[px]) {{
        idx = 3 * bl;
        dmdl[bls_idx] = {CMULT}(
            {CMULT}(
                gains[ant_offset + ggu_indices[idx+0]], 
                {CONJ}(gains[ant_offset + ggu_indices[idx+1]])
            ), ubls[ubl_offset + ggu_indices[idx+2]]
        );
    }}
}}


__global__ void calc_chisq({CDTYPE} *data, {CDTYPE} *dmdl, {DTYPE} *wgts, {DTYPE} *chisq, uint *active) {{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint bl = blockIdx.y * blockDim.y + threadIdx.y;
    const uint bls_idx = px * {NBLS} + bl;
    {CDTYPE} diff;

    //if (bl == 0 && active[px]) {{
    //    chisq[px] = 0;
    //}}

    //__syncthreads(); // Make sure chisq is zeroed before continuing

    if (bl < {NBLS} && active[px]) {{
        diff = {CSUB}(data[bls_idx], dmdl[bls_idx]);
        atomicAdd(&chisq[px], mag2(diff) * wgts[bls_idx]);
    }}
}}

//
__global__ void calc_dwgts({CDTYPE} *dmdl, {DTYPE} *wgts, {DTYPE} *dwgts, uint *active) {{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint bl = blockIdx.y * blockDim.y + threadIdx.y;
    const uint bls_idx = px * {NBLS} + bl;

    //printf("px:%d, bl:%d\\n", px, bl);
    //printf("bls_idx:%d, (%d, %d)\\n", bls_idx, bls_idx/{NBLS}, bls_idx % {NBLS});
    if (bl < {NBLS} && active[px]) {{
        dwgts[bls_idx] = mag2(dmdl[bls_idx]) * wgts[bls_idx];
    }}
}}

__global__ void clear_complex({CDTYPE} *buf, uint len) {{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px < len) {{
        buf[px] = make_{CDTYPE}(0, 0);
    }}
}}

__global__ void clear_real({DTYPE} *buf, uint len, {DTYPE} val) {{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px < len) {{
        buf[px] = val;
    }}
}}

__global__ void clear_uint(uint *buf, uint len, uint val) {{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px < len) {{
        buf[px] = val;
    }}
}}
    

//
__global__ void calc_gu_wgt(uint *ggu_indices, {CDTYPE} *dmdl, {DTYPE} *dwgts, {DTYPE} *gwgt, {DTYPE} *uwgt, uint *active) {{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint bl = blockIdx.y * blockDim.y + threadIdx.y;
    const uint ant_offset = px * {NANTS};
    const uint ubl_offset = px * {NUBLS};
    const uint bls_idx = px * {NBLS} + bl;
    uint idx;
    {DTYPE} w;

    //// Initialize the buffers we will integrate into
    //if (bl < {NANTS} && active[px]) {{
    //    gwgt[ant_offset + bl] = 0;
    //}}
    //if (bl < {NUBLS} && active[px]) {{
    //    uwgt[ubl_offset + bl] = 0;
    //}}

    //__syncthreads(); // make sure buffers are cleared

    if (bl < {NBLS} && active[px]) {{
        idx = 3 * bl;
        //w = mag2(dmdl[px]) * wgts[px];
        w = dwgts[bls_idx];
        atomicAdd(&gwgt[ant_offset + ggu_indices[idx+0]], w);
        atomicAdd(&gwgt[ant_offset + ggu_indices[idx+1]], w);
        atomicAdd(&uwgt[ubl_offset + ggu_indices[idx+2]], w);
    }}
}}

// 
__global__ void calc_gu_buf(uint *ggu_indices, {CDTYPE} *data, {DTYPE} *dwgts, {CDTYPE} *dmdl, {CDTYPE} *gbuf, {CDTYPE} *ubuf, uint *active) {{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint bl = blockIdx.y * blockDim.y + threadIdx.y;
    const uint ant_offset = px * {NANTS};
    const uint ubl_offset = px * {NUBLS};
    const uint bls_idx = px * {NBLS} + bl;
    uint idx;
    {CDTYPE} d;
    {DTYPE} w;

    //// Initialize the buffers we will integrate into
    //if (bl < {NANTS} && active[px]) {{
    //    gbuf[ant_offset + bl] = make_{CDTYPE}(0, 0);
    //}}
    //if (bl < {NUBLS} && active[px]) {{
    //    ubuf[ubl_offset + bl] = make_{CDTYPE}(0, 0);
    //}}

    //__syncthreads(); // make sure buffers are cleared

    if (bl < {NBLS} && active[px]) {{
        idx = 3 * bl;
        w = dwgts[bls_idx] / mag2(dmdl[bls_idx]);
        d = {CMULT}(data[bls_idx], {CONJ}(dmdl[bls_idx]));
        d.x = d.x * w;
        d.y = d.y * w;
        atomicAdd(&gbuf[ant_offset + ggu_indices[idx+0]].x,  d.x);
        atomicAdd(&gbuf[ant_offset + ggu_indices[idx+0]].y,  d.y);
        atomicAdd(&gbuf[ant_offset + ggu_indices[idx+1]].x,  d.x);
        atomicAdd(&gbuf[ant_offset + ggu_indices[idx+1]].y, -d.y);
        atomicAdd(&ubuf[ubl_offset + ggu_indices[idx+2]].x,  d.x);
        atomicAdd(&ubuf[ubl_offset + ggu_indices[idx+2]].y,  d.y);
    }}
}}


__global__ void update_gains({CDTYPE} *gbuf, {DTYPE} *gwgt, {CDTYPE} *gains, float gstep, uint *active) {{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint aa = blockIdx.y * blockDim.y + threadIdx.y;
    const uint ant_idx = px * {NANTS} + aa;
    {CDTYPE} wc;

    if (aa < {NANTS} && active[px]) {{
        wc.x = (1 - gstep) + gstep * gbuf[ant_idx].x / gwgt[ant_idx];
        wc.y = gstep * gbuf[ant_idx].y / gwgt[ant_idx];
        gains[ant_idx] = {CMULT}(gains[ant_idx], wc);
    }}
}}

__global__ void update_ubls({CDTYPE} *ubuf, {DTYPE} *uwgt, {CDTYPE} *ubls, float gstep, uint *active) {{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint uu = blockIdx.y * blockDim.y + threadIdx.y;
    const uint ubl_idx = px * {NUBLS} + uu;
    {CDTYPE} wc;

    if (uu < {NUBLS} && active[px]) {{
        wc.x = (1 - gstep) + gstep * ubuf[ubl_idx].x / uwgt[ubl_idx];
        wc.y = gstep * ubuf[ubl_idx].y / uwgt[ubl_idx];
        ubls[ubl_idx] = {CMULT}(ubls[ubl_idx], wc);
    }}
}}


__global__ void calc_conv({CDTYPE} *new_gains, {CDTYPE} *old_gains, {CDTYPE} *new_ubls, {CDTYPE} *old_ubls, {DTYPE} *conv_sum, {DTYPE} *conv_wgt, uint *active) {{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint bl = blockIdx.y * blockDim.y + threadIdx.y;
    const uint ant_idx = px * {NANTS} + bl;
    const uint ubl_idx = px * {NUBLS} + bl;
    {CDTYPE} wc;

    //if (bl == 0 && active[px]) {{
    //    conv_sum[px] = 0;
    //    conv_wgt[px] = 0;
    //}}

    //__syncthreads(); // Make sure conv buffers are cleared
    
    if (bl < {NANTS} && active[px]) {{
        wc = {CSUB}(new_gains[ant_idx], old_gains[ant_idx]);
        atomicAdd(&conv_sum[px], mag2(wc));
        atomicAdd(&conv_wgt[px], mag2(new_gains[ant_idx]));
    }}

    if (bl < {NUBLS} && active[px]) {{
        wc = {CSUB}(new_ubls[ubl_idx], old_ubls[ubl_idx]);
        atomicAdd(&conv_sum[px], mag2(wc));
        atomicAdd(&conv_wgt[px], mag2(new_ubls[ubl_idx]));
    }}
}}

__global__ void update_active({CDTYPE} *new_gains, {CDTYPE} *old_gains, {CDTYPE} *new_ubls, {CDTYPE} *old_ubls, {DTYPE} *conv_sum, {DTYPE} *conv_wgt, {DTYPE} *conv, {DTYPE} conv_crit, {DTYPE} *new_chisq, {DTYPE} *chisq, uint *iters, uint i, uint *active) {{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint bl = blockIdx.y * blockDim.y + threadIdx.y;
    const uint ant_idx = px * {NANTS} + bl;
    const uint ubl_idx = px * {NUBLS} + bl;
    
    if (bl == 0 && active[px]) {{
        //printf("conv[%d]: %f %f\\n", px, conv_sum[px], conv_wgt[px]);
        conv[px] = sqrt(conv_sum[px] / conv_wgt[px]);
        //conv_sum[px] = 0;
        //conv_wgt[px] = 0;
        if (conv[px] < conv_crit) {{
            active[px] = 0;
        }} else if (new_chisq[px] > chisq[px]) {{
            active[px] = 0;
        }} else {{
            chisq[px] = new_chisq[px];
            iters[px] = i;
        }}
    }}

    __syncthreads();

    if (bl < {NANTS} && active[px]) {{
        old_gains[ant_idx] = new_gains[ant_idx];
    }}

    if (bl < {NUBLS} && active[px]) {{
        old_ubls[ubl_idx] = new_ubls[ubl_idx];
    }}
}}
"""

NTHREADS = 1024 # make 512 for smaller GPUs
MAX_MEMORY = 2**29 # floats (4B each)
MAX_REGISTERS = 2**16 # number of registers
MIN_CHUNK = 8

def omnical(ggu_indices, gains, ubls, data, wgts, 
            conv_crit, maxiter, check_every, check_after,
            nthreads=NTHREADS, max_memory=MAX_MEMORY, 
            max_registers=MAX_REGISTERS,
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
    print('PRECISION:', precision)
    print('NDATA:', ndata)
    if precision == 1:
        real_dtype, complex_dtype = np.float32, np.complex64
        DTYPE, CDTYPE = 'float', 'cuFloatComplex'
        CMULT, CONJ, CSUB, CDIV = 'cuCmulf', 'cuConjf', 'cuCsubf', 'cuCdivf'
        COPY = cublasCcopy
    else:
        real_dtype, complex_dtype = np.float64, np.complex128
        DTYPE, CDTYPE = 'double', 'cuDoubleComplex'
        CMULT, CONJ, CSUB, CDIV = 'cuCmul', 'cuConj', 'cuCsub', 'cuCdiv'
        COPY = cublasZcopy
    assert check_every > 1

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

    chunk_size = min(nthreads // 16, ndata)

    # Build the CUDA code
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
    h = cublasCreate() # handle for managing cublas

    # define GPU buffers and transfer initial values
    #data_chunks = int(ceil(ndata / chunk_size))
    data_chunks = 1
    ANT_SHAPE = (chunk_size, nants)
    UBL_SHAPE = (chunk_size, nubls)
    BLS_SHAPE = (chunk_size, nbls)
    block = (chunk_size, int(floor(nthreads/chunk_size)), 1)
    print('NANTS:', nants)
    print('NUBLS:', nubls)
    print('NBLS:', nbls)
    print('chunk_size:', chunk_size)
    print('block:', block)
    ant_grid = (data_chunks, int(ceil(nants/block[1])))
    ubl_grid = (data_chunks, int(ceil(nubls/block[1])))
    bls_grid = (data_chunks, int(ceil( nbls/block[1])))

    ggu_indices_gpu = gpuarray.empty(shape=ggu_indices.shape, dtype=np.uint32)
    active_gpu = gpuarray.empty(shape=(chunk_size,), dtype=np.uint32)
    iters_gpu = gpuarray.empty(shape=(chunk_size,), dtype=np.uint32)
    
    gains_gpu     = gpuarray.empty(shape=ANT_SHAPE, dtype=complex_dtype)
    new_gains_gpu = gpuarray.empty(shape=ANT_SHAPE, dtype=complex_dtype)
    gbuf_gpu      = gpuarray.empty(shape=ANT_SHAPE, dtype=complex_dtype)
    gwgt_gpu      = gpuarray.empty(shape=ANT_SHAPE, dtype=real_dtype)

    ubls_gpu     = gpuarray.empty(shape=UBL_SHAPE, dtype=complex_dtype)
    new_ubls_gpu = gpuarray.empty(shape=UBL_SHAPE, dtype=complex_dtype)
    ubuf_gpu     = gpuarray.empty(shape=UBL_SHAPE, dtype=complex_dtype)
    uwgt_gpu     = gpuarray.empty(shape=UBL_SHAPE, dtype=real_dtype)

    data_gpu  = gpuarray.empty(shape=BLS_SHAPE, dtype=complex_dtype)
    dmdl_gpu  = gpuarray.empty(shape=BLS_SHAPE, dtype=complex_dtype)
    wgts_gpu  = gpuarray.empty(shape=BLS_SHAPE, dtype=real_dtype)
    dwgts_gpu = gpuarray.empty(shape=BLS_SHAPE, dtype=real_dtype)

    chisq_gpu     = gpuarray.empty(shape=(chunk_size,), dtype=real_dtype)
    new_chisq_gpu = gpuarray.empty(shape=(chunk_size,), dtype=real_dtype)
    conv_sum_gpu  = gpuarray.empty(shape=(chunk_size,), dtype=real_dtype)
    conv_wgt_gpu  = gpuarray.empty(shape=(chunk_size,), dtype=real_dtype)
    conv_gpu      = gpuarray.empty(shape=(chunk_size,), dtype=real_dtype)

    #stream = driver.Stream()
    chisq = np.empty((ndata,), dtype=real_dtype)
    conv  = np.empty((ndata,), dtype=real_dtype)
    iters = np.empty((ndata,), dtype=np.uint32)
    active = np.empty((chunk_size,), dtype=np.uint32)
    
    ggu_indices_gpu.set_async(ggu_indices)

    event_order = ('start', 'upload', 'dmdl', 'calc_chisq', 'loop_top', 'calc_gu_wgt', 'calc_gu_buf', 'copy_gains', 'copy_ubls', 'update_gains', 'update_ubls', 'dmdl2', 'chisq2', 'calc_conv', 'update_active', 'get_active', 'end')
    event_pairs = list((event_order[i],event_order[i+1]) for i in range(len(event_order[:-1])))

    import time
    cum_time = {}
    t0 = time.time()
    for px in range(0,ndata,chunk_size):
        events = {e:driver.Event() for e in event_order}
        events['start'].record()
        end = min(ndata, px + chunk_size)
        beg = end - chunk_size
        offset = px - beg
        _gains = gains[:,beg:end].T.copy()
        _ubls   = ubls[:,beg:end].T.copy()
        _data   = data[:,beg:end].T.copy()
        _wgts   = wgts[:,beg:end].T.copy()
        gains_gpu.set_async(_gains)
        ubls_gpu.set_async(_ubls)
        data_gpu.set_async(_data)
        wgts_gpu.set_async(_wgts)
        active = np.ones((chunk_size,), dtype=np.uint32)
        if offset > 0:
            active[:offset] = 0
        active_gpu.set_async(active)
        #clear_uint_cuda(active_gpu, np.uint32(chunk_size), np.uint32(1), 
        #                block=(chunk_size,1,1), grid=(1,1))
        clear_real_cuda(conv_sum_gpu, np.uint32(chunk_size), real_dtype(0),
                        block=(chunk_size,1,1), grid=(1,1))
        clear_real_cuda(conv_wgt_gpu, np.uint32(chunk_size), real_dtype(0),
                        block=(chunk_size,1,1), grid=(1,1))
        events['upload'].record()

        gen_dmdl_cuda(ggu_indices_gpu, gains_gpu, ubls_gpu, dmdl_gpu, active_gpu, block=block, grid=bls_grid)
        #print(px)
        #print(dmdl_gpu.shape)
        #print(dmdl_gpu)
        events['dmdl'].record()
        clear_real_cuda(chisq_gpu, np.uint32(chunk_size), real_dtype(0),
            block=(chunk_size,1,1), grid=(1,1))
        calc_chisq_cuda(data_gpu, dmdl_gpu, wgts_gpu,
            chisq_gpu, active_gpu,
            block=block, grid=bls_grid)
        #print(chisq_gpu.shape)
        #print(chisq_gpu)
        events['calc_chisq'].record()
        #chisq_gpu.get_async(ary=_chisq)
        if False:
            time.sleep(.01)
            for (e1,e2) in event_pairs[:3]:
                cum_time[(e1,e2)] = cum_time.get((e1,e2), 0) + events[e2].time_since(events[e1])

        for i in range(1,maxiter+1):
            #print(i, px, ndata, chunk_size, time.time() - t0)
            #print('ACTIVE:', active_gpu)
            #print('CHISQ:', chisq_gpu)
            #print('CONV:', conv_gpu)
            events['loop_top'].record()
            if (i % check_every) == 1:
                calc_dwgts_cuda(dmdl_gpu, wgts_gpu, dwgts_gpu,
                    active_gpu, block=block, grid=bls_grid)
                #print(dwgts_gpu.shape)
                #print(dwgts_gpu)
                clear_real_cuda(gwgt_gpu, np.uint32(nants*chunk_size),
                    real_dtype(0), block=(nthreads,1,1),
                    grid=(int(ceil(nants * chunk_size / nthreads)),1))
                clear_real_cuda(uwgt_gpu, np.uint32(nubls*chunk_size),
                    real_dtype(0), block=(nthreads,1,1),
                    grid=(int(ceil(nubls * chunk_size / nthreads)),1))
                calc_gu_wgt_cuda(ggu_indices_gpu, dmdl_gpu, dwgts_gpu, 
                    gwgt_gpu, uwgt_gpu, active_gpu,
                    block=block, grid=bls_grid)
                #print(gwgt_gpu.shape)
                #print(gwgt_gpu)
                #print(uwgt_gpu.shape)
                #print(uwgt_gpu)
            events['calc_gu_wgt'].record()
            clear_complex_cuda(gbuf_gpu, np.uint32(nants*chunk_size),
                block=(nthreads,1,1),
                grid=(int(ceil(nants * chunk_size / nthreads)),1))
            clear_complex_cuda(ubuf_gpu, np.uint32(nubls*chunk_size),
                block=(nthreads,1,1),
                grid=(int(ceil(nubls * chunk_size / nthreads)),1))
            calc_gu_buf_cuda(ggu_indices_gpu, 
                data_gpu, dwgts_gpu, dmdl_gpu, 
                gbuf_gpu, ubuf_gpu, active_gpu, 
                block=block, grid=bls_grid)
            events['calc_gu_buf'].record()
            #print(gbuf_gpu.shape)
            #print(gbuf_gpu)
            #print(ubuf_gpu.shape)
            #print(ubuf_gpu)
            if (i < maxiter) and (i < check_after or (i % check_every != 0)):
                # Fast branch
                #print('FAST')
                events['copy_gains'].record()
                events['copy_ubls'].record()
                update_gains_cuda(gbuf_gpu, gwgt_gpu, gains_gpu, 
                    np.float32(gain), active_gpu, 
                    block=block, grid=ant_grid)
                #print("GBUF:", gbuf_gpu)
                #print("GWGT:", gwgt_gpu)
                #print("GAIN:", gains_gpu)
                events['update_gains'].record()
                update_ubls_cuda(ubuf_gpu, uwgt_gpu, ubls_gpu, 
                    np.float32(gain), active_gpu, 
                    block=block, grid=ubl_grid)
                #print("UBUF:", ubuf_gpu)
                #print("UWGT:", uwgt_gpu)
                #print("UBLS:", ubls_gpu)
                events['update_ubls'].record()
                gen_dmdl_cuda(ggu_indices_gpu, gains_gpu, ubls_gpu, 
                    dmdl_gpu, active_gpu, 
                    block=block, grid=bls_grid)
                events['dmdl2'].record()
                events['chisq2'].record()
                events['calc_conv'].record()
                events['update_active'].record()
                events['get_active'].record()

                if False:
                    print('Fast Iteration:', i)
                    for (e1,e2) in event_pairs[4:]:
                        print('%5.3f' % events[e2].time_since(events[e1]), e1, e2)
                    e1,e2 = 'loop_top', 'end'
                    print('%5.3f' % events[e2].time_since(events[e1]), e1, e2)
                    print()
            else:
                #print('SLOW')
                COPY(h, nants*chunk_size,
                    gains_gpu.gpudata, 1,
                    new_gains_gpu.gpudata, 1)
                #print(new_gains_gpu)
                events['copy_gains'].record()
                COPY(h, nubls*chunk_size,
                    ubls_gpu.gpudata, 1,
                    new_ubls_gpu.gpudata, 1)
                events['copy_ubls'].record()
                update_gains_cuda(gbuf_gpu, gwgt_gpu, new_gains_gpu, 
                    np.float32(gain), active_gpu, 
                    block=block, grid=ant_grid)
                #print(new_gains_gpu)
                events['update_gains'].record()
                update_ubls_cuda(ubuf_gpu, uwgt_gpu, new_ubls_gpu, 
                    np.float32(gain), active_gpu, 
                    block=block, grid=ubl_grid)
                events['update_ubls'].record()
                gen_dmdl_cuda(ggu_indices_gpu, new_gains_gpu, new_ubls_gpu,
                    dmdl_gpu, active_gpu,
                    block=block, grid=bls_grid)
                events['dmdl2'].record()
                clear_real_cuda(new_chisq_gpu, np.uint32(chunk_size),
                    real_dtype(0), block=(chunk_size,1,1), grid=(1,1))
                calc_chisq_cuda(data_gpu, dmdl_gpu, wgts_gpu,
                    new_chisq_gpu, active_gpu,
                    block=block, grid=bls_grid)
                events['chisq2'].record()
                #print(conv_sum_gpu)
                #print('CONV_WGT pre:', conv_wgt_gpu)
                #print('CONV_SUM pre:', conv_sum_gpu)
                clear_real_cuda(conv_sum_gpu, np.uint32(chunk_size),
                    real_dtype(0), block=(chunk_size,1,1), grid=(1,1))
                clear_real_cuda(conv_wgt_gpu, np.uint32(chunk_size),
                    real_dtype(0), block=(chunk_size,1,1), grid=(1,1))
                calc_conv_cuda(new_gains_gpu, gains_gpu,
                    new_ubls_gpu, ubls_gpu,
                    conv_sum_gpu, conv_wgt_gpu, active_gpu,
                    block=block, grid=(data_chunks, int(ceil(max(nants, nubls)/block[1]))))
                events['calc_conv'].record()
                #print(conv_sum_gpu)
                #print('CONV_WGT:', conv_wgt_gpu)
                #print('CONV_SUM:', conv_sum_gpu)
                update_active_cuda(new_gains_gpu, gains_gpu,
                    new_ubls_gpu, ubls_gpu,
                    conv_sum_gpu, conv_wgt_gpu,
                    conv_gpu, real_dtype(conv_crit),
                    new_chisq_gpu, chisq_gpu,
                    iters_gpu, np.uint32(i),
                    active_gpu,
                    block=block, grid=(data_chunks, int(ceil(max(nants, nubls)/block[1]))))
                events['update_active'].record()
                active_gpu.get_async(ary=active)
                events['get_active'].record()
                if not np.any(active):
                    break
                #print(conv_gpu)
                #print(active_gpu)
                if False:
                    print('Slow Iteration:', i)
                    for (e1,e2) in event_pairs[4:]:
                        print('%5.3f' % events[e2].time_since(events[e1]), e1, e2)
                    e1,e2 = 'loop_top', 'end'
                    print('%5.3f' % events[e2].time_since(events[e1]), e1, e2)
                    print()
            events['end'].record()
            if False:
                time.sleep(.01)
                for (e1,e2) in event_pairs[4:]:
                    cum_time[(e1,e2)] = cum_time.get((e1,e2), 0) + events[e2].time_since(events[e1])
        _chisq = np.empty((chunk_size,), dtype=real_dtype)
        chisq_gpu.get_async(ary=_chisq)
        chisq[px:end] = _chisq[offset:]
        _iters = np.empty((chunk_size,), dtype=np.uint32)
        iters_gpu.get_async(ary=_iters)
        iters[px:end] = _iters[offset:]
        _conv = np.empty((chunk_size,), dtype=real_dtype)
        conv_gpu.get_async(ary=_conv)
        conv[px:end] = _conv[offset:]
        _gains = np.empty(ANT_SHAPE, dtype=complex_dtype)
        gains_gpu.get_async(ary=_gains)
        #import IPython; IPython.embed()
        gains[:,px:end] = _gains.T[:,offset:]
        _ubls = np.empty(UBL_SHAPE, dtype=complex_dtype)
        ubls_gpu.get_async(ary=_ubls)
        ubls[:,px:end] = _ubls.T[:,offset:]
    t1 = time.time()
    if False:
        print('Final, nthreads=%d' % nthreads)
        for (e1,e2) in event_pairs:
            try:
                print('%6.3f' % cum_time[(e1,e2)], e1, e2)
            except(KeyError):
                pass
    print(t1 - t0)
    print()
    
    cublasDestroy(h) # teardown GPU configuration
    #print('conv:', conv)
    #print('iters:', iters)
    return {'gains':gains, 'ubls':ubls, 'chisq':chisq, 'iters':iters,
            'conv':conv}

class OmnicalSolverGPU(OmnicalSolver):
    def solve_iteratively(self, conv_crit=1e-10, maxiter=50, check_every=4, check_after=1, precision=None, verbose=False):
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
        if precision is None:
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
    def omnical_gpu(self, data, sol0, wgts={}, gain=.3, conv_crit=1e-10, maxiter=50, check_every=4, check_after=1, precision=None):
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
        meta, sol = ls.solve_iteratively(conv_crit=conv_crit, maxiter=maxiter, check_every=check_every, check_after=check_after, precision=precision)
        sol = {self.unpack_sol_key(k): sol[k] for k in sol.keys()}
        return meta, sol
