
GPU = True

import pycuda.autoinit
from pycuda import compiler, gpuarray, driver
from skcuda.cublas import cublasCreate, cublasSetStream, cublasSgemm, cublasCgemm, cublasDestroy, cublasDgemm, cublasZgemm
import numpy as np
from math import ceil
import time
from scipy.interpolate import RectBivariateSpline

NTHREADS = 1024 # make 512 for smaller GPUs
MAX_MEMORY = 2**29 # floats (4B each)

GPU_TEMPLATE_FLOAT = """
// CUDA code for interpolating antenna beams and computing "voltage" visibilities 
// [A^1/2 * I^1/2 * exp(-2*pi*i*freq*dot(a,s)/c)]
// === Template Parameters ===
// "BLOCK_PX": # of sky pixels handled by one GPU block, used to size shared memory
// "NANT"   : # of antennas to pair into visibilities
// "NPIX"   : # of sky pixels to sum over.
// "BEAM_PX": dimension of sampled square beam matrix to be interpolated.  
//            Suggest using odd number.

#include <cuComplex.h>
#include <stdio.h>

// Linearly interpolate between [v0,v1] for t=[0,1]
// v = v0 * (1-t) + v1 * t = t*v1 + (-t*v0 + v0)
// Runs on GPU only
__device__
inline float lerp(float v0, float v1, float t) {
    return fma(t, v1, fma(-t, v0, v0));
}

// 3D texture storing beam response on (x=sin th_x, y=sin th_y, nant) grid
// for fast lookup by multiple threads.  Suggest setting first 2 dims of
// bm_tex to an odd number to get pixel centered on zenith.  The pixels
// on the border are centered at [-1,1] respectively.  Note that this
// matrix appears transposed relative to the host-side matrix used to set it.
texture<float, cudaTextureType3D, cudaReadModeElementType> bm_tex;

// Shared memory for storing per-antenna results to be reused among all ants
// for "BLOCK_PX" pixels, avoiding a rush on global memory.
__shared__ float sh_buf[%(BLOCK_PX)s*5];

// Interpolate bm_tex[x,y] at top=(x,y,z) coordinates and store answer in "A"
__global__ void InterpolateBeam(float *top, float *A)
{
    //printf("FLOAT bm_tex(5,5,5) is %%f", tex3D(bm_tex, 5, 5, 5));
    
    


    const uint nant = %(NANT)s;
    const uint npix = %(NPIX)s;
    const uint tx = threadIdx.x; // switched to make first dim px
    const uint ty = threadIdx.y; // switched to make second dim ant
    const uint pix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint ant = blockIdx.y * blockDim.y + threadIdx.y;
    const uint beam_px = %(BEAM_PX)s;
    float bm_x, bm_y, px, py, pz, fx, fy, top_z;

    if (pix == 0 && ant == 0) {
        //printf("FLOAT top[3] is %%f and top[2*npix + 3] is %%f", top[3], top[2*npix + 3]);
    }

    if (pix >= npix || ant >= nant) return;
    if (ty == 0) // buffer top_z for all threads
        sh_buf[tx+%(BLOCK_PX)s * 4] = top[2*npix+pix];
    __syncthreads(); // make sure top_z exists for all threads
    top_z = sh_buf[tx+%(BLOCK_PX)s * 4];
    if (ty == 0 && top_z > 0) { // buffer x interpolation for all threads
        bm_x = (beam_px-1) * (0.5 * top[pix] + 0.5);
        //bm_x = (beam_px) * (0.5 * top[pix] + 0.5) - 0.5f;
        px = floorf(bm_x);   // integer position
        sh_buf[tx+%(BLOCK_PX)s * 0] = bm_x - px; // fx, fractional position
        sh_buf[tx+%(BLOCK_PX)s * 2] = px + 0.5f; // px, pixel index
    }
    if (ty == 1 && top_z > 0) { // buffer y interpolation for all threads
        bm_y = (beam_px-1) * (0.5 * top[npix+pix] + 0.5);
        //bm_y = (beam_px) * (0.5 * top[npix+pix] + 0.5) - 0.5f;
        py = floorf(bm_y);
        sh_buf[tx+%(BLOCK_PX)s * 1] = bm_y - py; // fy, fractional position
        sh_buf[tx+%(BLOCK_PX)s * 3] = py + 0.5f; // py, pixel index
    }
    __syncthreads(); // make sure interpolation exists for all threads
    if (top_z > 0) {
///////////
    	//printf("SHARED BUF FLOAT: %%f, %%f //n", sh_buf[tx+%(BLOCK_PX)s * 0], sh_buf[tx+%(BLOCK_PX)s * 2]);
        ///////////
	fx = sh_buf[tx+%(BLOCK_PX)s * 0];
        fy = sh_buf[tx+%(BLOCK_PX)s * 1];
        px = sh_buf[tx+%(BLOCK_PX)s * 2];
        py = sh_buf[tx+%(BLOCK_PX)s * 3];
        pz = ant + 0.5f;
        A[ant*npix+pix] = lerp(lerp(tex3D(bm_tex,px,py,pz),      tex3D(bm_tex,px+1.0f,py,pz),fx),
               lerp(tex3D(bm_tex,px,py+1.0f,pz), tex3D(bm_tex,px+1.0f,py+1.0f,pz),fx), fy);
    } else {
        A[ant*npix+pix] = 0;
    }
    __syncthreads(); // make sure everyone used mem before kicking out
}

// Compute A*I*exp(ij*tau*freq) for all antennas, storing output in v
__global__ void MeasEq(float *A, float *I, float *tau, float freq, cuFloatComplex *v)
{
    const uint nant = %(NANT)s;
    const uint npix = %(NPIX)s;
    const uint tx = threadIdx.x; // switched to make first dim px
    const uint ty = threadIdx.y; // switched to make second dim ant
    const uint row = blockIdx.y * blockDim.y + threadIdx.y; // second thread dim is ant
    const uint pix = blockIdx.x * blockDim.x + threadIdx.x; // first thread dim is px
    float amp, phs;

    if (row >= nant || pix >= npix) return;
    if (ty == 0)
        sh_buf[tx] = I[pix];
    __syncthreads(); // make sure all memory is loaded before computing
    amp = A[row*npix + pix] * sh_buf[tx];
    phs = tau[row*npix + pix] * freq;
    v[row*npix + pix] = make_cuFloatComplex(amp * cos(phs), amp * sin(phs));
    __syncthreads(); // make sure everyone used mem before kicking out
}
"""

GPU_TEMPLATE_DOUBLE = """
// CUDA code for interpolating antenna beams and computing "voltage" visibilities 
// [A^1/2 * I^1/2 * exp(-2*pi*i*freq*dot(a,s)/c)]
// === Template Parameters ===
// "BLOCK_PX": # of sky pixels handled by one GPU block, used to size shared memory
// "NANT"   : # of antennas to pair into visibilities
// "NPIX"   : # of sky pixels to sum over.
// "BEAM_PX": dimension of sampled square beam matrix to be interpolated.  
//            Suggest using odd number.

#include <cuComplex.h>
#include <pycuda-helpers.hpp>
#include <stdio.h>

// Linearly interpolate between [v0,v1] for t=[0,1]
// v = v0 * (1-t) + v1 * t = t*v1 + (-t*v0 + v0)
// Runs on GPU only
__device__
inline double lerp(double v0, double v1, double t) {
    return fma(t, v1, fma(-t, v0, v0));
}

//__device__
//inline double fetch_double(uint2 p){
//	return __hiloint2double(p.y, p.x);
//}

// 3D texture storing beam response on (x=sin th_x, y=sin th_y, nant) grid
// for fast lookup by multiple threads.  Suggest setting first 2 dims of
// bm_tex to an odd number to get pixel centered on zenith.  The pixels
// on the border are centered at [-1,1] respectively.  Note that this
// matrix appears transposed relative to the host-side matrix used to set it.



//texture<float, 3, cudaReadModeElementType> bm_tex;


texture<fp_tex_double, cudaTextureType3D, cudaReadModeElementType> bm_tex;



// Shared memory for storing per-antenna results to be reused among all ants
// for "BLOCK_PX" pixels, avoiding a rush on global memory.
__shared__ double sh_buf[%(BLOCK_PX)s*5];

// Interpolate bm_tex[x,y] at top=(x,y,z) coordinates and store answer in "A"
__global__ void InterpolateBeam(double *top, double *A)
{
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
       // printf("DOUBLE bm_tex(5,5,5) is %%f", fp_tex3D(bm_tex, 5, 5, 5));

    }
    const uint nant = %(NANT)s;
    const uint npix = %(NPIX)s;
    const uint tx = threadIdx.x; // switched to make first dim px
    const uint ty = threadIdx.y; // switched to make second dim ant
    const uint pix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint ant = blockIdx.y * blockDim.y + threadIdx.y;
    const uint beam_px = %(BEAM_PX)s;
    double bm_x, bm_y, px, py, pz, fx, fy, top_z;

    if (pix == 0 && ant == 0) {
	//printf("DOUBLE top[3] is %%f and top[2*npix + 3] is %%f", top[3], top[2*npix + 3]) ;
    }


    if (pix >= npix || ant >= nant) return;
    if (ty == 0) // buffer top_z for all threads
        sh_buf[tx+%(BLOCK_PX)s * 4] = top[2*npix+pix];
    __syncthreads(); // make sure top_z exists for all threads
    top_z = sh_buf[tx+%(BLOCK_PX)s * 4];
    if (ty == 0 && top_z > 0) { // buffer x interpolation for all threads
        bm_x = (beam_px-1) * (0.5 * top[pix] + 0.5);
        //bm_x = (beam_px) * (0.5 * top[pix] + 0.5) - 0.5f;
        px = floorf(bm_x);   // integer position
        sh_buf[tx+%(BLOCK_PX)s * 0] = bm_x - px; // fx, fractional position
        sh_buf[tx+%(BLOCK_PX)s * 2] = px + 0.5f; // px, pixel index
    }
    if (ty == 1 && top_z > 0) { // buffer y interpolation for all threads
        bm_y = (beam_px-1) * (0.5 * top[npix+pix] + 0.5);
        //bm_y = (beam_px) * (0.5 * top[npix+pix] + 0.5) - 0.5f;
        py = floorf(bm_y);
        sh_buf[tx+%(BLOCK_PX)s * 1] = bm_y - py; // fy, fractional position
        sh_buf[tx+%(BLOCK_PX)s * 3] = py + 0.5f; // py, pixel index
    }
    __syncthreads(); // make sure interpolation exists for all threads
    if (top_z > 0) {
    	//printf("SHARED BUF DOUBLE: %%f, %%f //n", sh_buf[tx+%(BLOCK_PX)s * 0], sh_buf[tx+%(BLOCK_PX)s * 2]);
        fx = sh_buf[tx+%(BLOCK_PX)s * 0];
        fy = sh_buf[tx+%(BLOCK_PX)s * 1];
        px = sh_buf[tx+%(BLOCK_PX)s * 2];
        py = sh_buf[tx+%(BLOCK_PX)s * 3];
        pz = ant + 0.5f;
        A[ant*npix+pix] = lerp(lerp(fp_tex3D(bm_tex,px,py,pz),      fp_tex3D(bm_tex,px+1.0f,py,pz),fx),
               lerp(fp_tex3D(bm_tex,px,py+1.0f,pz), fp_tex3D(bm_tex,px+1.0f,py+1.0f,pz),fx), fy);
    } else {
        A[ant*npix+pix] = 0;
    }

    __syncthreads(); // make sure everyone used mem before kicking out

    if (pix == 0 && ant == 0) {
	//printf("A[0]: %%f, A[9]: %%f, A[57]: %%f \\n", A[0], A[9], A[57]);
    }
}

// Compute A*I*exp(ij*tau*freq) for all antennas, storing output in v
__global__ void MeasEq(double *A, double *I, double *tau, double freq, cuDoubleComplex *v)
{
    const uint nant = %(NANT)s;
    const uint npix = %(NPIX)s;
    const uint tx = threadIdx.x; // switched to make first dim px
    const uint ty = threadIdx.y; // switched to make second dim ant
    const uint row = blockIdx.y * blockDim.y + threadIdx.y; // second thread dim is ant
    const uint pix = blockIdx.x * blockDim.x + threadIdx.x; // first thread dim is px
    double amp, phs;

    if (row == 0 && pix == 0) {
	//printf("A %%f --- %%f, I %%f --- %%f, TAU %%f --- %%f, FREQ %%f \\n", A[0], A[5], I[0], I[5], tau[0], tau[5], freq);
    }

    if (row >= nant || pix >= npix) return;
    if (ty == 0)
        sh_buf[tx] = I[pix];
    __syncthreads(); // make sure all memory is loaded before computing
    amp = A[row*npix + pix] * sh_buf[tx];
    phs = tau[row*npix + pix] * freq;
    v[row*npix + pix] = make_cuDoubleComplex(amp * cos(phs), amp * sin(phs));
    __syncthreads(); // make sure everyone used mem before kicking out
}
"""

def numpy3d_to_array(np_array):
    '''Copy a 3D (d,h,w) numpy array into a 3D pycuda array that can be used 
    to set a texture.  (For some reason, gpuarrays can't be used to do that 
    directly).  A transpose happens implicitly; the CUDA array has dim (w,h,d).'''
    import pycuda.autoinit
    d, h, w = np_array.shape
    descr = driver.ArrayDescriptor3D()
    descr.width = w
    descr.height = h
    descr.depth = d
    descr.format = driver.dtype_to_array_format(np_array.dtype)
    descr.num_channels = 1
    descr.flags = 0
    device_array = driver.Array(descr)
    copy = driver.Memcpy3D()
    copy.set_src_host(np_array)
    copy.set_dst_array(device_array)
    copy.width_in_bytes = copy.src_pitch = np_array.strides[1]
    copy.src_height = copy.height = h
    copy.depth = d
    copy()
    return device_array

NTHREADS = 1024 # make 512 for smaller GPUs
MAX_MEMORY = 2**29 # floats (4B each)
MIN_CHUNK = 8

def vis_gpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube,
            nthreads=NTHREADS, max_memory=MAX_MEMORY,
            real_dtype=np.float32, complex_dtype=np.complex64,
            verbose=False):

    # use double precision CUDA?
    double_precision = not (real_dtype==np.float32 and complex_dtype==np.complex64)
    # ensure shapes
    nant = antpos.shape[0]
    assert(antpos.shape == (nant, 3))
    npix = crd_eq.shape[1]
    assert(crd_eq.shape == (3, npix))
    assert(I_sky.shape == (npix,))
    beam_px = bm_cube.shape[1]
    assert(bm_cube.shape == (nant, beam_px, beam_px))
    ntimes = eq2tops.shape[0]
    assert(eq2tops.shape == (ntimes, 3, 3))
    # ensure data types
    antpos = antpos.astype(real_dtype)
    eq2tops = eq2tops.astype(real_dtype)
    crd_eq = crd_eq.astype(real_dtype)
    Isqrt = np.sqrt(I_sky).astype(real_dtype)
    bm_cube = bm_cube.astype(real_dtype) # XXX complex?
    chunk = max(min(npix,MIN_CHUNK),2**int(ceil(np.log2(float(nant*npix) / max_memory / 2))))
    npixc = npix / chunk
    # blocks of threads are mapped to (pixels,ants,freqs)
    block = (max(1,nthreads/nant), min(nthreads,nant), 1)
    grid = (int(ceil(npixc/float(block[0]))),int(ceil(nant/float(block[1]))))
    
    # Choose to use single or double precision CUDA code
    GPU_TEMPLATE = GPU_TEMPLATE_DOUBLE if double_precision else GPU_TEMPLATE_FLOAT
    
    gpu_code = GPU_TEMPLATE % {
            'NANT': nant,
            'NPIX': npixc,
            'BEAM_PX': beam_px,
            'BLOCK_PX': block[0],
    }
    gpu_module = compiler.SourceModule(gpu_code)
    bm_interp = gpu_module.get_function("InterpolateBeam")
    meas_eq = gpu_module.get_function("MeasEq")
    bm_texref = gpu_module.get_texref("bm_tex")
    h = cublasCreate() # handle for managing cublas
    # define GPU buffers and transfer initial values

    if double_precision:
    	#bm_cube=bm_cube.astype(np.float32)
        #bm_texref.set_array(numpy3d_to_array(bm_cube))
	bm_texref.set_array(driver.np_to_array(bm_cube, "C"))
    else:
        bm_texref.set_array(numpy3d_to_array(bm_cube)) # never changes, transpose happens in copy so cuda bm_tex is (BEAM_PX,BEAM_PX,NANT

    antpos_gpu = gpuarray.to_gpu(antpos) # never changes, set to -2*pi*antpos/c
    Isqrt_gpu = gpuarray.empty(shape=(npixc,), dtype=real_dtype)
    A_gpu = gpuarray.empty(shape=(nant,npixc), dtype=real_dtype) # will be set on GPU by bm_interp
    crd_eq_gpu = gpuarray.empty(shape=(3,npixc), dtype=real_dtype)
    eq2top_gpu = gpuarray.empty(shape=(3,3), dtype=real_dtype) # sent from CPU each time
    crdtop_gpu = gpuarray.empty(shape=(3,npixc), dtype=real_dtype) # will be set on GPU
    tau_gpu = gpuarray.empty(shape=(nant,npixc), dtype=real_dtype) # will be set on GPU
    v_gpu = gpuarray.empty(shape=(nant,npixc), dtype=complex_dtype) # will be set on GPU
    vis_gpus = [gpuarray.empty(shape=(nant,nant), dtype=complex_dtype) for i in xrange(chunk)]
    # output CPU buffers for downloading answers
    vis_cpus = [np.empty(shape=(nant,nant), dtype=complex_dtype) for i in xrange(chunk)]
    streams = [driver.Stream() for i in xrange(chunk)]
    event_order = ('start','upload','eq2top','tau','interpolate','meas_eq','vis','end')
    vis = np.empty((ntimes,nant,nant), dtype=complex_dtype)

    
    if double_precision:
    # FOR DOUBLE PRECISION FLOATING POINT
	for t in xrange(ntimes):
	    if verbose: print '%d/%d' % (t+1, ntimes)
	    eq2top_gpu.set(eq2tops[t]) # defines sky orientation for this time step
	    events = [{e:driver.Event() for e in event_order} for i in xrange(chunk)]
	    for c in xrange(chunk+2):
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
		    cublasDgemm(h, 'n', 'n', npixc, 3, 3, 1., crd_eq_gpu.gpudata, 
			npixc, eq2top_gpu.gpudata, 3, 0., crdtop_gpu.gpudata, npixc)
		    events[cc]['eq2top'].record(stream)
		    ## compute tau = dot(antpos,crdtop)
		    cublasDgemm(h, 'n', 'n', npixc, nant, 3, 1., crdtop_gpu.gpudata, 
			npixc, antpos_gpu.gpudata, 3, 0., tau_gpu.gpudata, npixc)
		    events[cc]['tau'].record(stream)
		    ## interpolate bm_tex at specified topocentric coords, store interpolation in A
		    ## threads are parallelized across pixel axis
		    bm_interp(crdtop_gpu, A_gpu, grid=grid, block=block, stream=stream)
		    events[cc]['interpolate'].record(stream)
		    # compute v = A * I * exp(1j*tau*freq)
		    meas_eq(A_gpu, Isqrt_gpu, tau_gpu, real_dtype(freq), v_gpu, 
			grid=grid, block=block, stream=stream)
		    events[cc]['meas_eq'].record(stream)
		    # compute vis = dot(v, v.T)
		    # transpose below incurs about 20% overhead
		    cublasZgemm(h, 'c', 'n', nant, nant, npixc, 1., v_gpu.gpudata, 
			npixc, v_gpu.gpudata, npixc, 0., vis_gpus[cc].gpudata, nant)
		    events[cc]['vis'].record(stream)
		if c < chunk:
		    stream = streams[c]
		    events[c]['start'].record(stream)
		    crd_eq_gpu.set_async(crd_eq[:,c*npixc:(c+1)*npixc], stream=stream)
		    Isqrt_gpu.set_async(Isqrt[c*npixc:(c+1)*npixc], stream=stream)
		    events[c]['upload'].record(stream)
	    events[chunk-1]['end'].synchronize()
	    vis[t] = sum(vis_cpus)
	    if verbose:
		for c in xrange(chunk):
		    print '%d:%d START->END:' % (c, chunk), events[c]['start'].time_till(events[c]['end']) * 1e-3
		    #for i,e in enumerate(event_order[:-1]):
		    #    print c, e,'->',event_order[i+1], ':', events[c][e].time_till(events[c][event_order[i+1]]) * 1e-3
		print 'TOTAL:', events[0]['start'].time_till(events[chunk-1]['end']) * 1e-3
    
    else: 	
    # FOR SINGLE PRECISION FLOATING POINT
	for t in xrange(ntimes):
	    if verbose: print '%d/%d' % (t+1, ntimes)
	    eq2top_gpu.set(eq2tops[t]) # defines sky orientation for this time step
	    events = [{e:driver.Event() for e in event_order} for i in xrange(chunk)]
	    for c in xrange(chunk+2):
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
		    cublasSgemm(h, 'n', 'n', npixc, 3, 3, 1., crd_eq_gpu.gpudata, 
			npixc, eq2top_gpu.gpudata, 3, 0., crdtop_gpu.gpudata, npixc)
		    events[cc]['eq2top'].record(stream)
		    ## compute tau = dot(antpos,crdtop)
		    cublasSgemm(h, 'n', 'n', npixc, nant, 3, 1., crdtop_gpu.gpudata, 
			npixc, antpos_gpu.gpudata, 3, 0., tau_gpu.gpudata, npixc)
		    events[cc]['tau'].record(stream)
		    ## interpolate bm_tex at specified topocentric coords, store interpolation in A
		    ## threads are parallelized across pixel axis
		    bm_interp(crdtop_gpu, A_gpu, grid=grid, block=block, stream=stream)
		    events[cc]['interpolate'].record(stream)
		    # compute v = A * I * exp(1j*tau*freq)
		    meas_eq(A_gpu, Isqrt_gpu, tau_gpu, real_dtype(freq), v_gpu, 
			grid=grid, block=block, stream=stream)
		    events[cc]['meas_eq'].record(stream)
		    # compute vis = dot(v, v.T)
		    # transpose below incurs about 20% overhead
		    cublasCgemm(h, 'c', 'n', nant, nant, npixc, 1., v_gpu.gpudata, 
			npixc, v_gpu.gpudata, npixc, 0., vis_gpus[cc].gpudata, nant)
		    events[cc]['vis'].record(stream)
		if c < chunk:
		    stream = streams[c]
		    events[c]['start'].record(stream)
		    crd_eq_gpu.set_async(crd_eq[:,c*npixc:(c+1)*npixc], stream=stream)
		    Isqrt_gpu.set_async(Isqrt[c*npixc:(c+1)*npixc], stream=stream)
		    events[c]['upload'].record(stream)
	    events[chunk-1]['end'].synchronize()
	    vis[t] = sum(vis_cpus)
	    if verbose:
		for c in xrange(chunk):
		    print '%d:%d START->END:' % (c, chunk), events[c]['start'].time_till(events[c]['end']) * 1e-3
		    #for i,e in enumerate(event_order[:-1]):
		    #    print c, e,'->',event_order[i+1], ':', events[c][e].time_till(events[c][event_order[i+1]]) * 1e-3
		print 'TOTAL:', events[0]['start'].time_till(events[chunk-1]['end']) * 1e-3



    # teardown GPU configuration
    cublasDestroy(h)
    return vis


