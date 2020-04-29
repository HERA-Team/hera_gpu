import pycuda.autoinit
from pycuda import compiler, gpuarray, driver
from skcuda.cublas import cublasCreate, cublasSetStream, cublasDestroy, cublasCcopy, cublasZcopy
import skcuda.linalg as linalg
import skcuda.misc
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

// Shared memory for storing baseline order reused among all cells
__shared__ uint sh_inds[{NBLS}*3];
__shared__ {CDTYPE} sh_gains[{NANTS}];
__shared__ {CDTYPE} sh_ubls[{NUBLS}];

__device__ inline {DTYPE} mag2({CDTYPE} a) {{
    return a.x * a.x + a.y * a.y;
}}

//
__global__ void gen_dmdl(uint *ggu_indices, {CDTYPE} *gains, {CDTYPE} *ubls, {CDTYPE} *dmdl)
{{
    const uint tx = threadIdx.x;
    const uint offset = tx * {CHUNK_BLS};
    int chunk_size = {NBLS} - offset;
    uint idx;
    chunk_size = (chunk_size < {CHUNK_BLS}) ? chunk_size : {CHUNK_BLS};
    
    // Copy stuff into shared memory for faster access
    if (tx == 0) {{
        // Copy (gi,gj,uij) indices
        for (uint i=0; i < {NBLS} * 3; i++) {{
            sh_inds[i] = ggu_indices[i];
        }}
    }} else if (tx == 1) {{
        // Copy gains
        for (uint i=0; i < {NANTS}; i++) {{
            sh_gains[i] = gains[i];
        }}
    }} else if (tx == 2) {{
        // Copy unique baselines
        for (uint i=0; i < {NUBLS}; i++) {{
            sh_ubls[i] = ubls[i];
        }}
    }}

    __syncthreads(); // Make sure everything is copied

    if (chunk_size <= 0) {{
        return;
    }}
    // Calculate dmdl = gi * gj.conj * ubl
    for (uint i=offset; i < offset + chunk_size; i++) {{
        idx = 3 * i;
        dmdl[i] = {CMULT}({CMULT}(sh_gains[sh_inds[idx+0]], {CONJ}(sh_gains[sh_inds[idx+1]])), sh_ubls[sh_inds[idx+2]]);
    }}

    __syncthreads(); // make sure everyone used mem before kicking out
}}


__global__ void calc_chisq({CDTYPE} *data, {CDTYPE} *dmdl, {DTYPE} *wgts, {DTYPE} *chisq) {{
    const uint tx = threadIdx.x;
    const uint offset = tx * {CHUNK_BLS};
    int chunk_size = {NBLS} - offset;
    chunk_size = (chunk_size < {CHUNK_BLS}) ? chunk_size : {CHUNK_BLS};
    {DTYPE} _chisq=0;
    {CDTYPE} diff;

    if (tx == 0) {{
        chisq[0] = 0;
    }}

    __syncthreads(); // Make sure chisq is zeroed before continuing
    
    if (chunk_size <= 0) {{
        return;
    }}
    for (uint i=offset; i < offset + chunk_size; i++) {{
        diff = {CSUB}(data[i], dmdl[i]);
        _chisq += mag2(diff) * wgts[i];
    }}
    atomicAdd(chisq, _chisq);
}}

//
__global__ void calc_dwgts({CDTYPE} *dmdl, {DTYPE} *wgts, {DTYPE} *dwgts) {{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px < {NBLS}) {{
        dwgts[px] = mag2(dmdl[px]) * wgts[px];
    }}
}}

//
__global__ void calc_gu_wgt(uint *ggu_indices, {CDTYPE} *dmdl, {DTYPE} *dwgts, {DTYPE} *gwgt, {DTYPE} *uwgt) {{
    const uint tx = threadIdx.x;
    const uint offset = tx * {CHUNK_BLS};
    int chunk_size = {NBLS} - offset;
    chunk_size = (chunk_size < {CHUNK_BLS}) ? chunk_size : {CHUNK_BLS};
    uint idx;
    {DTYPE} wf=0;

    // Copy stuff into shared memory for faster access
    if (tx == 0) {{
        // Copy (gi,gj,uij) indices
        for (uint i=0; i < {NBLS} * 3; i++) {{
            sh_inds[i] = ggu_indices[i];
        }}
    }} else if (tx == 1) {{
        // Copy gbuf
        for (uint i=0; i < {NANTS}; i++) {{
            sh_gains[i].x = 0;
        }}
    }} else if (tx == 2) {{
        // Copy ubuf
        for (uint i=0; i < {NUBLS}; i++) {{
            sh_ubls[i].x = 0;
        }}
    }}

    __syncthreads(); // make sure shared buffers are in place

    for (uint i=offset; i < offset + chunk_size; i++) {{
        idx = 3 * i;
        //wf = mag2(dmdl[i]) * wgts[i];
        wf = dwgts[i];
        atomicAdd(&sh_gains[sh_inds[idx+0]].x, wf);
        atomicAdd(&sh_gains[sh_inds[idx+1]].x, wf);
        atomicAdd(&sh_ubls[sh_inds[idx+2]].x, wf);
    }}

    //if (px < {NANTS}) {{
    //    for (uint i=0; i < {NBLS}; i++) {{
    //        idx = 3 * i;
    //        if (sh_inds[idx+0] == px) {{
    //            wf += dwgts[i];
    //        }}
    //        if (sh_inds[idx+1] == px) {{
    //            wf += dwgts[i];
    //        }}
    //    }}
    //    gwgt[px] = wf;
    //}} else if (px < {NANTS} + {NUBLS}) {{
    //    px -= {NANTS};
    //    for (uint i=0; i < {NBLS}; i++) {{
    //        idx = 3 * i;
    //        if (sh_inds[idx+2] == px) {{
    //            wf += dwgts[i];
    //        }}
    //    }}
    //    uwgt[px] = wf;
    //}}

    __syncthreads(); // make sure everyone is done before clearing bufs

    if (tx == 1) {{
        // Copy gbuf
        for (uint i=0; i < {NANTS}; i++) {{
            gwgt[i] = sh_gains[i].x;
        }}
    }} else if (tx == 2) {{
        // Copy ubuf
        for (uint i=0; i < {NUBLS}; i++) {{
            uwgt[i] = sh_ubls[i].x;
        }}
    }}
}}

__global__ void calc_dw({CDTYPE} *data, {DTYPE} *dwgts, {CDTYPE} *dmdl, {CDTYPE} *dw) {{
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    {DTYPE} w;
    {CDTYPE} d;
    if (i < {NBLS}) {{
        w = dwgts[i] / mag2(dmdl[i]);
        d = {CMULT}(data[i], {CONJ}(dmdl[i]));
        dw[i].x = d.x * w;
        dw[i].y = d.y * w;
    }}
}}

// 
__global__ void calc_gu_buf(uint *ggu_indices, {CDTYPE} *data, {DTYPE} *dwgts, {CDTYPE} *dmdl, {CDTYPE} *gbuf, {CDTYPE} *ubuf) {{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    uint idx;
    {CDTYPE} d;
    {DTYPE} w;

    // Initialize the buffers we will integrate into
    if (px < {NANTS}) {{
        gbuf[px] = make_{CDTYPE}(0, 0);
    }}
    if (px < {NUBLS}) {{
        ubuf[px] = make_{CDTYPE}(0, 0);
    }}

    __syncthreads(); // make sure buffers are cleared

    if (px < {NBLS}) {{
        idx = 3 * px;
        w = dwgts[px] / mag2(dmdl[px]);
        d = {CMULT}(data[px], {CONJ}(dmdl[px]));
        d.x = d.x * w;
        d.y = d.y * w;
        atomicAdd(&gbuf[ggu_indices[idx+0]].x,  d.x);
        atomicAdd(&gbuf[ggu_indices[idx+0]].y,  d.y);
        atomicAdd(&gbuf[ggu_indices[idx+1]].x,  d.x);
        atomicAdd(&gbuf[ggu_indices[idx+1]].y, -d.y);
        atomicAdd(&ubuf[ggu_indices[idx+2]].x,  d.x);
        atomicAdd(&ubuf[ggu_indices[idx+2]].y,  d.y);
    }}
}}


__global__ void update_gains({CDTYPE} *gbuf, {DTYPE} *gwgt, {CDTYPE} *gains, float gstep) {{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    {CDTYPE} wc;
    if (px < {NANTS}) {{
        wc.x = (1 - gstep) + gstep * gbuf[px].x / gwgt[px];
        wc.y = gstep * gbuf[px].y / gwgt[px];
        gains[px] = {CMULT}(gains[px], wc);
    }}
}}

__global__ void update_ubls({CDTYPE} *ubuf, {DTYPE} *uwgt, {CDTYPE} *ubls, float gstep) {{
    const uint px = blockIdx.x * blockDim.x + threadIdx.x;
    {CDTYPE} wc;

    if (px < {NUBLS}) {{
        wc.x = (1 - gstep) + gstep * ubuf[px].x / uwgt[px];
        wc.y = gstep * ubuf[px].y / uwgt[px];
        ubls[px] = {CMULT}(ubls[px], wc);
    }}
}}

__global__ void calc_conv_gains({CDTYPE} *new_gains, {CDTYPE} *old_gains, {DTYPE} *conv_sum, {DTYPE} *conv_wgt) {{
    const uint tx = threadIdx.x;
    const uint offset = tx * {CHUNK_ANT};
    int chunk_size = {NANTS} - offset;
    chunk_size = (chunk_size < {CHUNK_ANT}) ? chunk_size : {CHUNK_ANT};
    {DTYPE} _conv_sum=0, _conv_wgt=0;
    {CDTYPE} wc;

    if (tx == 0) {{
        conv_sum[0] = 0;
        conv_wgt[0] = 0;
    }}

    __syncthreads(); // Make sure conv buffers are cleared

    if (chunk_size <= 0) {{
        return;
    }}
    for (uint i=offset; i < offset + chunk_size; i++) {{
        wc = {CSUB}(new_gains[i], old_gains[i]);
        _conv_sum += mag2(wc);
        _conv_wgt += mag2(new_gains[i]);
    }}
    atomicAdd(conv_sum, _conv_sum);
    atomicAdd(conv_wgt, _conv_wgt);
}}

__global__ void calc_conv_ubls({CDTYPE} *new_ubls, {CDTYPE} *old_ubls, {DTYPE} *conv_sum, {DTYPE} *conv_wgt) {{
    const uint tx = threadIdx.x;
    const uint offset = tx * {CHUNK_UBL};
    int chunk_size = {NUBLS} - offset;
    chunk_size = (chunk_size < {CHUNK_UBL}) ? chunk_size : {CHUNK_UBL};
    {DTYPE} _conv_sum=0, _conv_wgt = 0;
    {CDTYPE} wc;

    // Purposely not zeroing conv buffers: have to cal calc_conv_gains first

    if (chunk_size <= 0) {{
        return;
    }}
    for (uint i=offset; i < offset + chunk_size; i++) {{
        wc = {CSUB}(new_ubls[i], old_ubls[i]);
        _conv_sum += mag2(wc);
        _conv_wgt += mag2(new_ubls[i]);
    }}
    atomicAdd(conv_sum, _conv_sum);
    atomicAdd(conv_wgt, _conv_wgt);
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
    
    #nthreads = 512 # XXX
    nthreads = 16 # XXX
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
            'CHUNK_BLS': int(ceil(nbls / nthreads)),
            'CHUNK_ANT': int(ceil(nants / nthreads)),
            'CHUNK_UBL': int(ceil(nubls / nthreads)),
    })

    gpu_module = compiler.SourceModule(gpu_code)
    gen_dmdl_cuda = gpu_module.get_function("gen_dmdl")
    calc_chisq_cuda = gpu_module.get_function("calc_chisq")
    calc_dwgts_cuda = gpu_module.get_function("calc_dwgts")
    calc_gu_wgt_cuda = gpu_module.get_function("calc_gu_wgt")
    calc_dw_cuda = gpu_module.get_function("calc_dw")
    calc_gu_buf_cuda = gpu_module.get_function("calc_gu_buf")
    update_gains_cuda = gpu_module.get_function("update_gains")
    update_ubls_cuda = gpu_module.get_function("update_ubls")
    calc_conv_gains_cuda = gpu_module.get_function("calc_conv_gains")
    calc_conv_ubls_cuda = gpu_module.get_function("calc_conv_ubls")
    h = cublasCreate() # handle for managing cublas
    linalg.init()

    # define GPU buffers and transfer initial values
    ggu_indices_gpu = gpuarray.empty(shape=ggu_indices.shape, dtype=np.uint32)
    gains_gpu = gpuarray.empty(shape=(nants,), dtype=complex_dtype)
    new_gains_gpu = gpuarray.empty(shape=(nants,), dtype=complex_dtype)
    gbuf_gpu = gpuarray.empty(shape=(nants,), dtype=complex_dtype)
    gwgt_gpu = gpuarray.empty(shape=(nants,), dtype=real_dtype)
    ubls_gpu = gpuarray.empty(shape=(nubls,), dtype=complex_dtype)
    new_ubls_gpu = gpuarray.empty(shape=(nubls,), dtype=complex_dtype)
    ubuf_gpu = gpuarray.empty(shape=(nubls,), dtype=complex_dtype)
    uwgt_gpu = gpuarray.empty(shape=(nubls,), dtype=real_dtype)
    data_gpu = gpuarray.empty(shape=(nbls,), dtype=complex_dtype)
    dw_gpu = gpuarray.empty(shape=(nbls,), dtype=complex_dtype)
    dmdl_gpu = gpuarray.empty(shape=(nbls,), dtype=complex_dtype)
    wgts_gpu = gpuarray.empty(shape=(nbls,), dtype=real_dtype)
    dwgts_gpu = gpuarray.empty(shape=(nbls,), dtype=real_dtype)
    chisq_gpu = gpuarray.empty(shape=(1,), dtype=real_dtype)
    new_chisq_gpu = gpuarray.empty(shape=(1,), dtype=real_dtype)
    conv_sum_gpu = gpuarray.empty(shape=(1,), dtype=real_dtype)
    conv_wgt_gpu = gpuarray.empty(shape=(1,), dtype=real_dtype)

    #stream = driver.Stream()
    _chisq = np.empty(shape=(1,), dtype=real_dtype)
    _new_chisq = np.empty(shape=(1,), dtype=real_dtype)
    conv_sum = np.empty(shape=(1,), dtype=real_dtype)
    conv_wgt = np.empty(shape=(1,), dtype=real_dtype)
    chisq = np.empty((ndata,), dtype=real_dtype)
    conv = np.empty((ndata,), dtype=real_dtype)
    iters = np.empty((ndata,), dtype=np.uint32)
    
    ggu_indices_gpu.set_async(ggu_indices)

    event_order = ('start', 'upload', 'dmdl', 'chisq', 'get_chisq', 'loop_top', 'calc_gu_wgt', 'calc_dw', 'calc_gu_buf', 'copy_gains', 'copy_ubls', 'update_gains', 'update_ubls', 'calc_conv_gains', 'calc_conv_ubls', 'get_conv', 'dmdl2', 'chisq2', 'get_chisq2', 'copy2', 'end')
    event_pairs = list((event_order[i],event_order[i+1]) for i in range(len(event_order[:-1])))

    import time
    cum_time = {}
    for px in range(ndata):
        events = {e:driver.Event() for e in event_order}
        t0 = time.time()
        events['start'].record()
        gains_gpu.set_async(gains[:,px])
        ubls_gpu.set_async(ubls[:,px])
        data_gpu.set_async(data[:,px])
        wgts_gpu.set_async(wgts[:,px])
        events['upload'].record()

        gen_dmdl_cuda(ggu_indices_gpu, gains_gpu, ubls_gpu, dmdl_gpu, block=(nthreads,1,1), grid=(1,1))
        events['dmdl'].record()
        calc_chisq_cuda(data_gpu, dmdl_gpu, wgts_gpu, chisq_gpu, block=(nthreads,1,1), grid=(1,1))
        events['chisq'].record()
        chisq_gpu.get_async(ary=_chisq)
        events['get_chisq'].record()
        time.sleep(.01)
        for (e1,e2) in event_pairs[:4]:
            cum_time[(e1,e2)] = cum_time.get((e1,e2), 0) + events[e2].time_since(events[e1])

        for i in range(1,maxiter+1):
            events['loop_top'].record()
            if (i % check_every) == 1:
                calc_dwgts_cuda(dmdl_gpu, wgts_gpu, dwgts_gpu, block=(nthreads,1,1), grid=(int(ceil(nbls/nthreads)),1))
                calc_gu_wgt_cuda(ggu_indices_gpu, dmdl_gpu, dwgts_gpu, gwgt_gpu, uwgt_gpu, block=(nthreads,1,1), grid=(int(ceil((nants+nubls)/nthreads)),1))
            events['calc_gu_wgt'].record()
            #calc_dw_cuda(data_gpu, dwgts_gpu, dmdl_gpu, dw_gpu, block=(nthreads,1,1), grid=(int(ceil(nbls/nthreads)),1))
            #calc_dw_cuda(data_gpu, dwgts_gpu, dmdl_gpu, dw_gpu, block=(nthreads,1,1), grid=(1,1))
            #dw_gpu = linalg.multiply(data_gpu, dwgts_gpu)
            #dw_gpu = skcuda.misc.divide(dw_gpu, dmdl_gpu)
            events['calc_dw'].record()
            #calc_gu_buf_cuda(ggu_indices_gpu, data_gpu, dwgts_gpu, dmdl_gpu, gbuf_gpu, ubuf_gpu, block=(nthreads,1,1), grid=(1,1))
            calc_gu_buf_cuda(ggu_indices_gpu, data_gpu, dwgts_gpu, dmdl_gpu, gbuf_gpu, ubuf_gpu, block=(nthreads,1,1), grid=(int(ceil(nbls/nthreads)),1))
            #calc_gu_buf_cuda(ggu_indices_gpu, dw_gpu, gbuf_gpu, ubuf_gpu, block=(nthreads,1,1), grid=(1,1))
            events['calc_gu_buf'].record()
            if (i < maxiter) and (i < check_after or (i % check_every != 0)):
                # Fast branch
                events['copy_gains'].record()
                events['copy_ubls'].record()
                update_gains_cuda(gbuf_gpu, gwgt_gpu, gains_gpu, np.float32(gain), block=(nthreads,1,1), grid=(int(ceil(nants/nthreads)),1))
                events['update_gains'].record()
                update_ubls_cuda(ubuf_gpu, uwgt_gpu, ubls_gpu, np.float32(gain), block=(nthreads,1,1), grid=(int(ceil(nubls/nthreads)),1))
                events['update_ubls'].record()
                gen_dmdl_cuda(ggu_indices_gpu, gains_gpu, ubls_gpu, dmdl_gpu, block=(nthreads,1,1), grid=(1,1))
                events['calc_conv_gains'].record()
                events['calc_conv_ubls'].record()
                events['get_conv'].record()
                events['dmdl2'].record()
                events['chisq2'].record()
                events['get_chisq2'].record()
                events['copy2'].record()
                events['end'].record()
                time.sleep(.01)
                for (e1,e2) in event_pairs[5:]:
                    cum_time[(e1,e2)] = cum_time.get((e1,e2), 0) + events[e2].time_since(events[e1])

                if False:
                    print('Fast Iteration:', i)
                    for (e1,e2) in event_pairs[5:]:
                        print('%5.3f' % events[e2].time_since(events[e1]), e1, e2)
                    e1,e2 = 'loop_top', 'end'
                    print('%5.3f' % events[e2].time_since(events[e1]), e1, e2)
                    print()
            else:
                COPY(h, nants, gains_gpu.gpudata, 1, new_gains_gpu.gpudata, 1)
                events['copy_gains'].record()
                COPY(h, nubls, ubls_gpu.gpudata, 1, new_ubls_gpu.gpudata, 1)
                events['copy_ubls'].record()
                update_gains_cuda(gbuf_gpu, gwgt_gpu, new_gains_gpu, np.float32(gain), block=(nthreads,1,1), grid=(int(ceil(nants/nthreads)),1))
                events['update_gains'].record()
                update_ubls_cuda(ubuf_gpu, uwgt_gpu, new_ubls_gpu, np.float32(gain), block=(nthreads,1,1), grid=(int(ceil(nubls/nthreads)),1))
                events['update_ubls'].record()
                calc_conv_gains_cuda(new_gains_gpu, gains_gpu, conv_sum_gpu, conv_wgt_gpu, block=(nthreads,1,1), grid=(1,1))
                events['calc_conv_gains'].record()
                calc_conv_ubls_cuda(new_ubls_gpu, ubls_gpu, conv_sum_gpu, conv_wgt_gpu, block=(nthreads,1,1), grid=(1,1))
                events['calc_conv_ubls'].record()
                conv_sum_gpu.get_async(ary=conv_sum)
                conv_wgt_gpu.get_async(ary=conv_wgt)
                _conv = np.sqrt(conv_sum[0] / conv_wgt[0])
                events['get_conv'].record()

                if _conv < conv_crit:
                    break
                
                gen_dmdl_cuda(ggu_indices_gpu, new_gains_gpu, new_ubls_gpu, dmdl_gpu, block=(nthreads,1,1), grid=(1,1))
                events['dmdl2'].record()
                calc_chisq_cuda(data_gpu, dmdl_gpu, wgts_gpu, new_chisq_gpu, block=(nthreads,1,1), grid=(1,1))
                events['chisq2'].record()

                
                new_chisq_gpu.get_async(ary=_new_chisq)
                events['get_chisq2'].record()
                if _new_chisq[0] > _chisq[0]:
                    break

                COPY(h, nants, new_gains_gpu.gpudata, 1, gains_gpu.gpudata, 1)
                COPY(h, nubls, new_ubls_gpu.gpudata, 1, ubls_gpu.gpudata, 1)
                events['copy2'].record()
                chisq_gpu = new_chisq_gpu
                events['end'].record()
                time.sleep(.01)
                for (e1,e2) in event_pairs[5:]:
                    cum_time[(e1,e2)] = cum_time.get((e1,e2), 0) + events[e2].time_since(events[e1])
                if False:
                    print('Slow Iteration:', i)
                    for (e1,e2) in event_pairs[5:]:
                        print('%5.3f' % events[e2].time_since(events[e1]), e1, e2)
                    e1,e2 = 'loop_top', 'end'
                    print('%5.3f' % events[e2].time_since(events[e1]), e1, e2)
                    print()
        chisq_gpu.get_async(ary=_chisq)
        chisq[px] = _chisq
        conv[px] = _conv
        iters[px] = i
        gains_gpu.get_async(ary=gains[:,px])
        ubls_gpu.get_async(ary=ubls[:,px])
    t1 = time.time()
    print('Final, nthreads=%d' % nthreads)
    for (e1,e2) in event_pairs:
        try:
            print('%6.3f' % cum_time[(e1,e2)], e1, e2)
        except(KeyError):
            pass
    print(t1 - t0)
    print()
    time.sleep(0.5)
    
    cublasDestroy(h) # teardown GPU configuration
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
