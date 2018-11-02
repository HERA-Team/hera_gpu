#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import scipy.signal as sps


def MedianFilter(input=None, kernel_size=3, bw=32, bh=32):

	#s = cuda.Event()
	#e = cuda.Event()

	input_list = input

	BLOCK_WIDTH = bw
	BLOCK_HEIGHT = bh

	if isinstance(kernel_size, (int, long)):
		kernel_size = [kernel_size]*2

	WS_x, WS_y = kernel_size
	padding_y = WS_x/2
	padding_x = WS_y/2

	input_list = np.asarray(input_list)

	if input_list.ndim == 3:
		_, N, M = input_list.shape
	elif input_list.ndim == 2:
		N, M = input_list.shape
		input_list = [input_list]

	expanded_N = N + (2 * padding_y)
	expanded_M = M + (2 * padding_x)

	gridx = int(np.ceil((expanded_N)/BLOCK_WIDTH))+1
	gridy = int(np.ceil((expanded_M)/BLOCK_HEIGHT))+1
	grid = (gridx,gridy, 1)
	block = (BLOCK_WIDTH, BLOCK_HEIGHT, 1)

	code = """
		#pragma comment(linker, "/HEAP:4000000")


		/* Some sample C code for the quickselect algorithm, 
		   taken from Numerical Recipes in C. */

		#define SWAP(a,b) temp=(a);(a)=(b);(b)=temp;

		__device__ float quickselect(float *arr, int n, int k) {
		  unsigned long i,ir,j,l,mid;
		  float a,temp;

		  l=0;
		  ir=n-1;
		  for(;;) {
		    if (ir <= l+1) { 
		      if (ir == l+1 && arr[ir] < arr[l]) {
			SWAP(arr[l],arr[ir]);
		      }
		      return arr[k];
		    }
		    else {
		      mid=(l+ir) >> 1; 
		      SWAP(arr[mid],arr[l+1]);
		      if (arr[l] > arr[ir]) {
			SWAP(arr[l],arr[ir]);
		      }
		      if (arr[l+1] > arr[ir]) {
			SWAP(arr[l+1],arr[ir]);
		      }
		      if (arr[l] > arr[l+1]) {
			SWAP(arr[l],arr[l+1]);
		      }
		      i=l+1; 
		      j=ir;
		      a=arr[l+1]; 
		      for (;;) { 
			do i++; while (arr[i] < a); 
			do j--; while (arr[j] > a); 
			if (j < i) break; 
			SWAP(arr[i],arr[j]);
		      } 
		      arr[l+1]=arr[j]; 
		      arr[j]=a;
		      if (j >= k) ir=j-1; 
		      if (j <= k) l=i;
		    }
		  }
		}

		/* https://softwareengineering.stackexchange.com/questions/284767/kth-selection-routine-floyd-algorithm-489
		 * Implementation from Stack Exchange user: Andy Dansby 
		 */

		__device__ float FloydWirth_kth(float arr[], const int kTHvalue) 
		{
		#define F_SWAP(a,b) { float temp=(a);(a)=(b);(b)=temp; }
		#define SIGNUM(x) ((x) < 0 ? -1 : ((x) > 0 ? 1 : (x)))

		    int left = 0;       
		    int right = %(WS^2)s - 1;     
		    int left2 = 0;
		    int right2 = %(WS^2)s - 1;

		    while (left < right) 
		    {           
		        if( arr[right2] < arr[left2] ) F_SWAP(arr[left2],arr[right2]);
		        if( arr[right2] < arr[kTHvalue] ) F_SWAP(arr[kTHvalue],arr[right2]);
		        if( arr[kTHvalue] < arr[left2] ) F_SWAP(arr[left2],arr[kTHvalue]);

		        int rightleft = right - left;

		        if (rightleft < kTHvalue)
		        {
		            int n = right - left + 1;
		            int ii = kTHvalue - left + 1;
		            int s = (n + n) / 3;
		            int sd = (n * s * (n - s) / n) * SIGNUM(ii - n / 2);
		            int left2 = max(left, kTHvalue - ii * s / n + sd);
		            int right2 = min(right, kTHvalue + (n - ii) * s / n + sd);              
		        }

		        float x=arr[kTHvalue];

		        while ((right2 > kTHvalue) && (left2 < kTHvalue))
		        {
		            do 
		            {
		                left2++;
		            }while (arr[left2] < x);

		            do
		            {
		                right2--;
		            }while (arr[right2] > x);

		            F_SWAP(arr[left2],arr[right2]);
		        }
		        left2++;
		        right2--;

		        if (right2 < kTHvalue) 
		        {
		            while (arr[left2]<x)
		            {
		                left2++;
		            }
		            left = left2;
		            right2 = right;
		        }

		        if (kTHvalue < left2) 
		        {
		            while (x < arr[right2])
		            {
		                right2--;
		            }

		            right = right2;
		            left2 = left;
		        }

		        if( arr[left] < arr[right] ) F_SWAP(arr[right],arr[left]);
		    }

		#undef F_SWAP
		#undef SIGNUM
		    return arr[kTHvalue];
		}



		texture<float, 2> tex;

		__global__ void mf(float* in, float* out, int imgDimY, int imgDimX)
		{

			float window[%(WS^2)s];

			int x_thread_offset = %(BY)s * blockIdx.x + threadIdx.x;
			int y_thread_offset = %(BX)s * blockIdx.y + threadIdx.y;
			for (int y = %(WSx/2)s + y_thread_offset; y < imgDimX - %(WSx/2)s; y += %(y_stride)s)
			{
				for (int x = %(WSy/2)s + x_thread_offset; x < imgDimY - %(WSy/2)s; x += %(x_stride)s)
				{
					int i = 0;
					for (int fx = 0; fx < %(WSy)s; ++fx)
					{
						for (int fy = 0; fy < %(WSx)s; ++fy)
						{
							//window[i] = tex2D(tex, (float) (x + fx - %(WSy/2)s), (float) (y + fy - %(WSx/2)s));
							window[i] = in[(x + fx - %(WSy/2)s) + (y + fy - %(WSx/2)s)*imgDimY];
							i += 1;
						}
					}

					// Sort to find the median
					//for (int j = 0; j < %(WS^2)s/2 + 1; j++)
					//{
					//	for (int k = j + 1; k < %(WS^2)s; k++)
					//	{
					//		if (window[j] > window[k])
					//		{
					//			float tmp = window[j];
					//			window[j] = window[k];
					//			window[k] = tmp;
					//		}
					//	}
					//}
					//out[y*imgDimY + x] = window[%(WS^2)s/2];
					out[y*imgDimY + x] = FloydWirth_kth(window, %(WS^2)s/2);
					//out[y*imgDimY + x] = quickselect(window, %(WS^2)s, %(WS^2)s/2);
				}
			}
		}


		__global__ void mf_shared(float *in, float* out, int imgDimY, int imgDimX)
		{			

			const int TSx = %(BX)s + %(WSx)s - 1;
			const int TSy = %(BY)s + %(WSy)s - 1;
            __shared__ float tile[TSx][TSy];

            float window[%(WS^2)s];
            const int x_thread_offset = %(BX)s * blockIdx.x + threadIdx.x;
            const int y_thread_offset = %(BY)s * blockIdx.y + threadIdx.y;


			const int thread_index = blockDim.y * threadIdx.x + threadIdx.y;

			int imgX = blockIdx.x * blockDim.x + thread_index;
			int imgY;

            // Load into the tile for this block
			if (thread_index < TSx && imgX < imgDimX)
			{
				for (int i = 0; i < TSy && i < imgDimY - blockIdx.y * blockDim.y; i++)
				{
					imgY = blockIdx.y * blockDim.y + i;
					tile[thread_index][i] = in[imgX * imgDimY + imgY];
					//tile[thread_index][i] = tex2D(tex, (float) imgY, (float) imgX);
				}

			}

			__syncthreads();


			int x = %(WSx/2)s + x_thread_offset;
			int y = %(WSy/2)s + y_thread_offset;

			if (x >= imgDimX - %(WSx/2)s || y >= imgDimY - %(WSy/2)s)
			{
				return;
			}

			int i = 0;
			for (int fx = 0; fx < %(WSx)s; ++fx)
			{
				for (int fy = 0; fy < %(WSy)s; ++fy)
				{
					window[i++] = tile[threadIdx.x + fx][threadIdx.y + fy];
				}
			}


			// Sort to find the median
			//for (int j = 0; j <= %(WS^2)s/2; j++)
			//{
			//	for (int k = j + 1; k < %(WS^2)s; k++)
			//	{
			//		if (window[j] > window[k])
			//		{
			//			float tmp = window[j];
			//			window[j] = window[k];
			//			window[k] = tmp;
			//		}
			//	}
			//}
			//out[x*imgDimY + y] = window[%(WS^2)s/2];

			out[x*imgDimY + y] = FloydWirth_kth(window, %(WS^2)s/2);

			//forgetfulSelection(window, %(WSx)s);
			//out[x*imgDimY + y] = window[%(WS^2)s/2];

			//out[x*imgDimY + y] = myForgetfulSelection(window);
		}

		"""

	code = code % {
			'BY' : BLOCK_WIDTH,
			'BX' : BLOCK_HEIGHT,
			'WS^2' : WS_x * WS_y,
			'x_stride' : BLOCK_WIDTH * gridx,
			'y_stride' : BLOCK_HEIGHT * gridy,
			'WSx' : WS_x,
			'WSy' : WS_y,
			'WSx/2' : WS_x/2,
			'WSy/2' : WS_y/2,
		}
	mod = SourceModule(code)
	#mf_shared = mod.get_function('mf_shared')
	mf = mod.get_function('mf')
	texref = mod.get_texref("tex")


	# NSTREAMS := NUMBER OF INPUT IMAGES
	nStreams = len(input_list)

	# Initialize the streams
	stream = [cuda.Stream()]*nStreams

	# Pad all the images with zeros
	input_list = [np.array( np.pad(img, ( (padding_y, padding_y), (padding_x, padding_x) ), 'constant', constant_values=0) , dtype=np.float32) for img in input_list]

	# Use pinned memory for all the images
	in_pin_list = [cuda.register_host_memory(img) for img in input_list]
	imgBytes = in_pin_list[0].nbytes

	# Initialize the outputs to empty images (assuming all images are of the same shape)
	outdata_list = [cuda.pagelocked_empty_like(img) for img in input_list]

	# Malloc on the GPU for each input and output image
	#in_gpu_list = [cuda.mem_alloc(pinnedImg.nbytes) for pinnedImg in in_pin_list]
	in_gpu_list = [None]*nStreams
	#out_gpu_list = [cuda.mem_alloc(pinnedImg.nbytes) for pinnedImg in in_pin_list]
	out_gpu_list = [None]*nStreams
	mf.prepare("PPii")
	for i in xrange(nStreams + 2):
		ii = i - 1
		iii = i - 2

		if 0 <= iii < nStreams:
			st = stream[iii]
			cuda.memcpy_dtoh_async(outdata_list[iii], out_gpu_list[iii], stream=st)

		if 0 <= ii < nStreams:
			st = stream[ii]
			out_gpu_list[ii] = cuda.mem_alloc(imgBytes)
			# s.record(stream=stream[0])
			# mf_shared.prepare("Pii")
			# mf_shared.prepared_async_call(grid, block, st, out_gpu_list[ii], expanded_M, expanded_N)

			#mf.prepare("PPii")
			mf.prepared_async_call(grid, block, st, in_gpu_list[ii], out_gpu_list[ii], expanded_M, expanded_N)
			# e.record(stream=stream[0])
			# e.synchronize()
			# print s.time_till(e), "ms for the kernel"

		if 0 <= i < nStreams:
			st = stream[i]
			#cuda.matrix_to_texref(in_pin_list[i], texref, order="C")
			in_gpu_list[i] = cuda.mem_alloc(imgBytes)
			cuda.memcpy_htod_async(in_gpu_list[i], in_pin_list[i], stream=st)

	if (padding_y > 0):
		outdata_list = [out[padding_y:-padding_y] for out in outdata_list]
	if (padding_x > 0):
		outdata_list = [out[:, padding_x:-padding_x] for out in outdata_list]

	return outdata_list
