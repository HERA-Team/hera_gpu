import pycuda.autoinit
from pycuda import compiler, gpuarray, driver
from skcuda.cublas import cublasCreate, cublasSetStream, cublasSgemm, cublasCgemm, cublasDestroy, cublasDgemm, cublasZgemm
from astropy.constants import c as speed_of_light
import numpy as np
import time

ONE_OVER_C = 1.0 / speed_of_light.value

GPU_TEMPLATE = """
// CUDA code for interpolating antenna beams and computing "voltage" visibilities 
// [A^1/2 * I^1/2 * exp(-2*pi*i*freq*dot(a,s)/c)]
// === Template Parameters ===
// "DTYPE"  : float or double
// "CDTYPE"  : cuFloatComplex or cuDoubleComplex
// "BLOCK_PX": # of sky pixels handled by one GPU block, used to size shared memory
// "NANT"   : # of antennas to pair into visibilities
// "NPIX"   : # of sky pixels to sum over.
// "BEAM_PX": number of beam coefficients for PolyBeam

#include <cuComplex.h>
#include <pycuda-helpers.hpp>
#include <stdio.h>

// Chebval code from https://github.com/numpy/numpy/blob/v1.19.0/numpy/polynomial/chebyshev.py#L1088-L1172
__device__ %(DTYPE)s chebval(%(DTYPE)s x, %(DTYPE)s *c) {
    %(DTYPE)s x2, tmp, c0, c1;
    int i;

    x2 = 2*x;
    c0 = c[%(BEAM_PX)s-2];      // BEAM_PX is number of beam coeffs
    c1 = c[%(BEAM_PX)s-1];
    for (i=3; i<%(BEAM_PX)s+1; ++i) {
        tmp = c0;
        c0 = c[%(BEAM_PX)s-i] - c1;
        c1 = tmp + c1*x2;
    }
    return c0 + c1*x;
}

// Interpolate the beam using Chebyshev polynomials.
// top is (3, nsource), the 3 are x, y, z, so it has source information
// beam is (nant, n_coeff) or (nbeam, n_coeff) (should be the same, which is checked).
// The thread is generating the beam value for a particular ant and source (ant, pix in the
// code below), also for a particular frequency, which is inherent in the fscale values.
// fscale is (nant,)
__global__ void InterpolateBeam(%(DTYPE)s *top, %(DTYPE)s *beam_coeff, %(DTYPE)s *fscale, %(DTYPE)s *A)
{
    const uint nant = %(NANT)s;
    const uint npix = %(NPIX)s;		// actually nsource
    const uint pix = blockIdx.x * blockDim.x + threadIdx.x;  // which source this thread
    const uint ant = blockIdx.y * blockDim.y + threadIdx.y;  // which ant this thread
    const %(DTYPE)s PI =  3.141592653589793238;
    %(DTYPE)s l, m, lsqr, n, za, x;
    
    if (pix >= npix || ant >= nant) return;

    // Calculate za for this source, copying the code from lm_to_az_za in conversions
    l = top[pix]; m = top[npix+pix];     // tx, ty in the Python
    lsqr = l*l+m*m;
    if ( lsqr < 1 ) n = sqrt(1.0-lsqr); else n = 0;
    za = PI/2.0-asin(n);

    // Modify za using the fscale from the beam for this ant
    x = 2.0*sin(za/fscale[ant])-1;	// In PolyBeam interp
    
    A[ant*npix+pix] = chebval(x, beam_coeff+ant*%(BEAM_PX)s)
			/chebval(-1, beam_coeff+ant*%(BEAM_PX)s);
    if ( A[ant*npix+pix] < 0 ) A[ant*npix+pix] = 0;
}

// Shared memory for storing per-antenna results to be reused among all ants
// for "BLOCK_PX" pixels, avoiding a rush on global memory.
__shared__ %(DTYPE)s sh_buf[%(BLOCK_PX)s*5];

// Compute A*I*exp(ij*tau*freq) for all antennas, storing output in v
__global__ void MeasEq(%(DTYPE)s *A, %(DTYPE)s *I, %(DTYPE)s *tau, %(DTYPE)s freq, %(CDTYPE)s *v)
{
    const uint nant = %(NANT)s;
    const uint npix = %(NPIX)s;
    const uint tx = threadIdx.x; // switched to make first dim px
    const uint ty = threadIdx.y; // switched to make second dim ant
    const uint row = blockIdx.y * blockDim.y + threadIdx.y; // second thread dim is ant
    const uint pix = blockIdx.x * blockDim.x + threadIdx.x; // first thread dim is px
    %(DTYPE)s amp, phs;

    if (row >= nant || pix >= npix) return;
    if (ty == 0)
        sh_buf[tx] = I[pix];
    __syncthreads(); // make sure all memory is loaded before computing
    amp = A[row*npix + pix] * sh_buf[tx];
    phs = tau[row*npix + pix] * freq;
    v[row*npix + pix] = make_%(CDTYPE)s(amp * cos(phs), amp * sin(phs));
    __syncthreads(); // make sure everyone used mem before kicking out
}
"""

NTHREADS = 1024 # make 512 for smaller GPUs
MAX_MEMORY = 2**29 # floats (4B each)
MIN_CHUNK = 8

def vis_gpu(antpos, frequencies, eq2tops, crd_eq, I_skies, beam_list, vis_spec,
            nthreads=NTHREADS, max_memory=MAX_MEMORY,
            precision=1, verbose=False):
            
    assert(precision in (1,2))
    if precision == 1:
        real_dtype, complex_dtype = np.float32, np.complex64
        DTYPE, CDTYPE = 'float', 'cuFloatComplex'
        cublas_real_mm = cublasSgemm
        cublas_cmpx_mm = cublasCgemm
    else:
        real_dtype, complex_dtype = np.float64, np.complex128
        DTYPE, CDTYPE = 'double', 'cuDoubleComplex'
        cublas_real_mm = cublasDgemm
        cublas_cmpx_mm = cublasZgemm


    # ensure shapes

    nant = antpos.shape[0]
    assert(len(beam_list) == nant)
    beam_coeffs = np.empty((len(beam_list), len(beam_list[0].beam_coeffs)))
    for i in range(len(beam_list)):
        if len(beam_list[i].beam_coeffs) != len(beam_list[0].beam_coeffs):
           raise RuntimeError("Number of beam coeffs differ by beam, for GPU")
        beam_coeffs[i] = np.array(beam_list[i].beam_coeffs)

    assert(antpos.shape == (nant, 3))
    npix = crd_eq.shape[1]
    assert(I_skies.shape == (frequencies.shape[0], npix))
    assert(crd_eq.shape == (3, npix))
    ntimes = eq2tops.shape[0]
    assert(eq2tops.shape == (ntimes, 3, 3))
    # ensure data types
    antpos = antpos.astype(real_dtype)
    eq2tops = eq2tops.astype(real_dtype)
    crd_eq = crd_eq.astype(real_dtype)
    beam_coeffs = beam_coeffs.astype(real_dtype) # XXX complex?
    chunk = max(min(npix,MIN_CHUNK),2**int(np.ceil(np.log2(float(nant*npix) / max_memory / 2))))
    npixc = npix // chunk
    # blocks of threads are mapped to (pixels,ants,freqs)
    block = (max(1,nthreads//nant), min(nthreads,nant), 1)
    grid = (int(np.ceil(npixc/float(block[0]))),int(np.ceil(nant/float(block[1]))))

    # Choose to use single or double precision CUDA code
    gpu_code = GPU_TEMPLATE % {
        'NANT': nant,
        'NPIX': npixc,
        'BEAM_PX': beam_coeffs.shape[1],
        'BLOCK_PX': block[0],
        'DTYPE': DTYPE,
        'CDTYPE': CDTYPE,
    }

    gpu_module = compiler.SourceModule(gpu_code)
    bm_interp = gpu_module.get_function("InterpolateBeam")
    meas_eq = gpu_module.get_function("MeasEq")
    h = cublasCreate() # handle for managing cublas
    # define GPU buffers and transfer initial values
    antpos_gpu = gpuarray.to_gpu(antpos) # never changes, set to -2*pi*antpos/c
    Isqrt_gpu = gpuarray.empty(shape=(npixc,), dtype=real_dtype)
    A_gpu = gpuarray.empty(shape=(nant,npixc), dtype=real_dtype) # will be set on GPU by bm_interp
    beam_gpu = gpuarray.empty(shape=(nant,beam_coeffs.shape[1]), dtype=real_dtype)
    beam_gpu.set(beam_coeffs)
    fscale_gpu = gpuarray.empty(shape=(nant,), dtype=real_dtype)
    crd_eq_gpu = gpuarray.empty(shape=(3,npixc), dtype=real_dtype)
    eq2top_gpu = gpuarray.empty(shape=(3,3), dtype=real_dtype) # sent from CPU each time
    crdtop_gpu = gpuarray.empty(shape=(3,npixc), dtype=real_dtype) # will be set on GPU
    tau_gpu = gpuarray.empty(shape=(nant,npixc), dtype=real_dtype) # will be set on GPU
    v_gpu = gpuarray.empty(shape=(nant,npixc), dtype=complex_dtype) # will be set on GPU
    vis_gpus = [gpuarray.empty(shape=(nant,nant), dtype=complex_dtype) for i in range(chunk)]
    # output CPU buffers for downloading answers
    vis_cpus = [np.empty(shape=(nant,nant), dtype=complex_dtype) for i in range(chunk)]
    streams = [driver.Stream() for i in range(chunk)]
    event_order = ('start','upload','eq2top','tau','interpolate','meas_eq','vis','end')
    vis = np.empty((ntimes,nant,nant), dtype=complex_dtype)
    visfull = np.zeros(vis_spec[0], dtype=vis_spec[1])
      
 
    for index in range(len(frequencies)): 
        freq = frequencies[index]
        I_sky = I_skies[index]
        
        # apply scalars so 1j*tau*freq is the correct exponent
        freq = 2 * freq * np.pi
        Isqrt = np.sqrt(I_sky).astype(real_dtype)
 
        fscale = np.empty(len(beam_list))	# Needed to modify za
        for i in range(len(beam_list)):
            fscale[i] = ( frequencies[index] / beam_list[i].ref_freq)**beam_list[i].spectral_index
        fscale_gpu.set(fscale)
        
        for t in range(ntimes):
            eq2top_gpu.set(eq2tops[t]) # defines sky orientation for this time step
            events = [{e:driver.Event() for e in event_order} for i in range(chunk)]
            for c in range(chunk+2):
                cc = c - 1
                ccc = c - 2
                if 0 <= ccc < chunk:
                    stream = streams[ccc]
                    vis_gpus[ccc].get_async(ary=vis_cpus[ccc], stream=stream)
                    events[ccc]['end'].record(stream)
                if 0 <= cc < chunk:
                    stream = streams[cc]
                    cublasSetStream(h, stream.handle)
                    ## compute crdtop = dot(eq2top,crd_eq)
                    # cublas arrays are in Fortran order, so P=M*N is actually 
                    # peformed as P.T = N.T * M.T
                    cublas_real_mm(h, 'n', 'n', npixc, 3, 3, 1., crd_eq_gpu.gpudata, 
                    npixc, eq2top_gpu.gpudata, 3, 0., crdtop_gpu.gpudata, npixc)
                    events[cc]['eq2top'].record(stream)
                    ## compute tau = dot(antpos,crdtop) / speed_of_light
                    cublas_real_mm(h, 'n', 'n', npixc, nant, 3, ONE_OVER_C, crdtop_gpu.gpudata, 
                    npixc, antpos_gpu.gpudata, 3, 0., tau_gpu.gpudata, npixc)
                    events[cc]['tau'].record(stream)
                    ## interpolate bm_tex at specified topocentric coords, store interpolation in A
                    ## threads are parallelized across pixel axis
                    bm_interp(crdtop_gpu, beam_gpu, fscale_gpu, A_gpu, grid=grid, block=block, stream=stream)
                    events[cc]['interpolate'].record(stream)
                    # compute v = A * I * exp(1j*tau*freq)
                    meas_eq(A_gpu, Isqrt_gpu, tau_gpu, real_dtype(freq), v_gpu, 
                    grid=grid, block=block, stream=stream)
                    events[cc]['meas_eq'].record(stream)
                    # compute vis = dot(v, v.T)
                    # transpose below incurs about 20% overhead
                    cublas_cmpx_mm(h, 'c', 'n', nant, nant, npixc, 1., v_gpu.gpudata, 
                    npixc, v_gpu.gpudata, npixc, 0., vis_gpus[cc].gpudata, nant)
                    events[cc]['vis'].record(stream)
                if c < chunk:
                    stream = streams[c]
                    events[c]['start'].record(stream)
                    crd_eq_gpu.set_async(crd_eq[:,c*npixc:(c+1)*npixc], stream=stream)
                    Isqrt_gpu.set_async(Isqrt[c*npixc:(c+1)*npixc], stream=stream)
                    events[c]['upload'].record(stream)
            events[chunk-1]['end'].synchronize()
            vis[t] = np.conj(sum(vis_cpus))

            indices = np.triu_indices(vis.shape[1])
            vis_upper_tri = vis[:, indices[0], indices[1]]
            visfull[:, 0, index, 0] = vis_upper_tri.flatten()

    cublasDestroy(h)

    return visfull
