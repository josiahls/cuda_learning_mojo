//===---------------------------BOILER PLATE--------------------------------====
#include <atomic>
#include <climits>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));       \
      /*  The negative exit value indicates a cuda error */                    \
      exit(-10 * error);                                                       \
    }                                                                          \
  }

void host_malloc(int **ptr, int nBytes) {  
  *ptr = (int *)malloc(nBytes); 
}

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void initialData(int *ip, int size) {
  for (int i = 0; i < size; i++) {
    ip[i] = (int)(rand() & 0xFF);
  }
}

void initialData(float *fp, int size) {
  // Generate different seed for random number
  time_t t;
  srand((unsigned int)time(&t));
  for (int i = 0; i < size; i++) {
    fp[i] = (float)(rand() & 0xFF) / 10.0f;
  }
}

void checkResult(int *h_C, int *h_GpuCheck, int nElem) {
  bool match = true;
  if (abs(h_C[0] - h_GpuCheck[0]) > 1e-5) {
    printf("Error sums do not match: %d != %d\n", h_C[0], h_GpuCheck[0]);
    match = false;
  }

  if (match) {
    printf("Test passed\n");
  } else {
    printf("Test failed\n");
  }
}

inline void printElaps(double &iStart,double &iElaps, const char * function_name, const dim3 &grid, const dim3 &block) {
  iElaps = (cpuSecond() - iStart) * 1000;
  printf("%s <<< %4d, %4d >>> elapsed %f ms \n", function_name, grid.x, block.x, iElaps);
}

//===---------------------------BOILER PLATE--------------------------------====

void hostAccumulateToIndex0(int *ip, int grid_x) {
  for (int i = 1; i < grid_x; i++) {
    ip[0] += ip[i];
  }
}

void hostReduceNeighbored(int *g_idata, int *g_odata, int n) {
  int num = 0;
  for (int idx = 0; idx < n; idx++) {
    num += g_idata[idx];
  }
  printf("Final sum: %d\n",num);
  g_odata[0] = num;
  if (INT_MAX <= num) {
    printf("At max size!\n");
  }
}

__device__ void __syncthreads(void);


__global__ void reduceNeighbored(int *g_idata, int *g_odata, int n) {
  // e.g. thread 1
  unsigned int tid = threadIdx.x;
  // e.g.: block 1 * 512(block size) + 1 -> 513
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Make a local pointer for the data in this block:
  //  - g_idata + block start index (pointer arithmetic).
  int *idata = g_idata + blockIdx.x * blockDim.x;

  if (idx >= n) {
    return;
  } 

  for (int stride=1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2 * stride)) == 0) {
      idata[tid] += idata[tid + stride];
    }

    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, int n) {
  // e.g. thread 1
  unsigned int tid = threadIdx.x;
  // e.g.: block 1 * 512(block size) + 1 -> 513
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Make a local pointer for the data in this block:
  //  - g_idata + block start index (pointer arithmetic).
  int *idata = g_idata + blockIdx.x * blockDim.x;

  if (idx >= n) {
    return;
  } 

  for (int stride=1; stride < blockDim.x; stride *= 2) {
    int index = 2 * stride * tid;
    if (index < blockDim.x) {
      idata[index] += idata[index + stride];
    }

    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = idata[0];
}



/**
Note from the book:
- This is going to yield poor performance since we are calling intra-block synchronization via __syncthreads()
so much (16,384) times.

*/
__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata, int isize) {
	unsigned int tid = threadIdx.x;
	int *idata = g_idata + blockIdx.x * blockDim.x;
	int *odata = &g_odata[blockIdx.x];

	// stop condition
	if (isize == 2 && tid == 0) {
		g_odata[blockIdx.x] = idata[0] + idata[1];
	}

	// all threads must return together to avoid __syncthreads() deadlock
	if (isize == 2) {
		return;
	}

	// nested invocation
	int istride = isize >> 1;

	if (istride > 1 && tid < istride) {
		// in place reduction
		idata[tid] += idata[tid + istride];
	}

	// sync at block level
	__syncthreads();

	if (tid == 0 ) {
		gpuRecursiveReduce <<<1, istride, 0, cudaStreamTailLaunch>>> (
			idata, odata, istride
		);

		// https://docs.nvidia.com/cuda/cuda-c-programming-guide/
		// Cuda no longer supports this devices of compute capability 9.0.
		// cudaDeviceSynchronize();
	}
	
	// // sync at block level again
	// __syncthreads();
}

/**
Remove additional synchonizations, however I think this is still going to have
implicit ones no?
*/
__global__ void gpuRecursiveReduceNoSync(int *g_idata, int *g_odata, int isize) {
	unsigned int tid = threadIdx.x;
	int *idata = g_idata + blockIdx.x * blockDim.x;
	int *odata = &g_odata[blockIdx.x];

	// stop condition
	if (isize == 2 && tid == 0) {
		g_odata[blockIdx.x] = idata[0] + idata[1];
    return;
	}

	// nested invocation
	int istride = isize >> 1;

	if (istride > 1 && tid < istride) {
		// in place reduction
		idata[tid] += idata[tid + istride];
    if (tid == 0 ) {
      gpuRecursiveReduceNoSync <<<1, istride, 0, cudaStreamTailLaunch>>> (
        idata, odata, istride
      );
    }
	}
}


__global__ void gpuRecursiveReduce2(int *g_idata, int *g_odata, int isize, int const idim) {
  const int last_block_idx = gridDim.x - 1;
	unsigned int tid = threadIdx.x;
	int *idata = g_idata + blockIdx.x * idim;

	// stop condition
	if (isize == 1 && tid == 0) {
		g_odata[blockIdx.x] = idata[0] + idata[1];
    return;
	}

  // Ok big issue:
  // Accessing element block idx 2047 * idim 512 + tid 413 + isize 256 = 1,048,733 Value 0 + 8 =
  // Accessing element block idx 2047 * idim 512 + tid 414 + isize 256 = 1,048,734 Value 0 + 124 =
  // Accessing element block idx 2047 * idim 512 + tid 415 + isize 256 = 1,048,735 Value 0 + 195 =
  // Accessing element block idx 2047 * idim 512 + tid 256 + isize 256 = 1,048,576 Value 0 + 151 =

  // The idata pointer only has len 1,048,576. '1,048,733 Value 0 + 8 =' is UB no?
  // Like... This "works" since its only ever summing real elements, but this is still... terrible?
  // I think I want the final block to be a noop no?

  // wtf given block size 512, and thread 511 + stride, what protects against this?
  // I mean I guess as long as its not the last block? Because we are probably fine
  // then. 

  // Ok I hate the last block. We are relying on out-of-memory access being zero instead
  // of weird ub

  // Reading this more, how th fuck does this work? If I'm on block
  // 2047
  // and I'm starting from element: 2047 * 512
  // and I have tid 511
  // I'm going to be accessing element 2047 * 512 + 511 + 512 = 1,049,087
  // where elements are 1,048,576.

  // where stride is 8
  // block 1, block 2, block 3
  // [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]
  // sum first batch of threads
  // in place reduction

  // A little safer, at least not accessing out of bounds, however we now
  // maybe have warp divergence.
  if (blockIdx.x != last_block_idx || tid + isize < idim) {
    idata[tid] += idata[tid + isize];
  }

  if (tid == 0 && blockIdx.x == 0) {
    gpuRecursiveReduce2 <<<gridDim.x, isize / 2, 0, cudaStreamTailLaunch>>> (
      g_idata, g_odata, isize / 2, idim
    );
  }
}


void run_reduction_function(
	void (*gpu_reducer_fn)(int*, int*, int),
	const char * function_name,
	int *h_idata,
	int *h_check_odata,
	dim3 grid,
	dim3 block,
	int nx,
	int reduction_size,
	int kernel_x_denom=1,
	int grid_x_denom=1
) {
	double iStart, iElips;
	int input_size = nx * sizeof(int);
	int output_size = (grid.x / grid_x_denom) * sizeof(int);
	int *h_odata,*d_idata,*d_odata;
	host_malloc((int**)&h_odata, output_size);
	cudaMalloc((int**)&d_idata, input_size);
	cudaMalloc((int**)&d_odata, output_size);
	cudaMemcpy(d_idata, h_idata, input_size, cudaMemcpyHostToDevice);
	iStart = cpuSecond();

	gpu_reducer_fn<<< grid.x/kernel_x_denom, block>>> (d_idata, d_odata, reduction_size);

	CHECK(cudaDeviceSynchronize());
	printElaps(iStart,iElips, function_name, grid, block);
	cudaMemcpy(h_odata, d_odata, output_size, cudaMemcpyDeviceToHost);
	hostAccumulateToIndex0(h_odata, grid.x/grid_x_denom);
	checkResult(h_check_odata, h_odata, nx);
	cudaFree(d_idata);
	cudaFree(d_odata);
	free(h_odata);
}


void run_reduction_function(
	void (*gpu_reducer_fn)(int*, int*, int, int const),
	const char * function_name,
	int *h_idata,
	int *h_check_odata,
	dim3 grid,
	dim3 block,
	int nx,
	int reduction_size,
	int kernel_x_denom=1,
	int grid_x_denom=1
) {
	double iStart, iElips;
	int input_size = nx * sizeof(int);
	int output_size = (grid.x / grid_x_denom) * sizeof(int);
	int *h_odata,*d_idata,*d_odata;
	host_malloc((int**)&h_odata, output_size);
	cudaMalloc((int**)&d_idata, input_size);
	cudaMalloc((int**)&d_odata, output_size);
	cudaMemcpy(d_idata, h_idata, input_size, cudaMemcpyHostToDevice);
	iStart = cpuSecond();

	gpu_reducer_fn<<< grid.x/kernel_x_denom, block>>> (d_idata, d_odata, reduction_size / 2, block.x);

	CHECK(cudaDeviceSynchronize());
	printElaps(iStart,iElips, function_name, grid, block);
	cudaMemcpy(h_odata, d_odata, output_size, cudaMemcpyDeviceToHost);
	hostAccumulateToIndex0(h_odata, grid.x/grid_x_denom);
	checkResult(h_check_odata, h_odata, nx);
	cudaFree(d_idata);
	cudaFree(d_odata);
	free(h_odata);
}



/**
  ncu --metric sm__warps_active.avg.pct_of_peak_sustained_active build/cuda_learning_c/chapter_3/pg_111_reduceInteger
  ncu --metric 	l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second build/cuda_learning_c/chapter_3/pg_111_reduceInteger
  ncu --metric 	l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum build/cuda_learning_c/chapter_3/pg_111_reduceInteger
  ncu --metric smsp__warp_issue_stalled_membar_per_warp_active.pct build/cuda_learning_c/chapter_3/pg_111_reduceInteger
*/
int main(int argc, char **argv) {
  const int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("%s Using device %d: %s \n", argv[0], dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  // Maximum blocks limited by GPU's concurrent nested kernel launch capacity
  // GTX 1660 Ti (compute capability 7.5) supports up to 2047 blocks with cudaStreamTailLaunch
  int nblocks = 2048;  // The default limit is 2048, so we bump the limit.
  cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 2049);

  int nthreads = 512;
  int nx = nblocks * nthreads;
  printf("Creating pointers of %d elements.\n", nx);
  size_t NBytes = nx * sizeof(int);


  if (argc > 1) nthreads = atoi(argv[1]);
  printf("Data size x: %d y: %d\n", nthreads, 1);

  dim3 block (nthreads, 1);
  dim3 grid((nx + block.x - 1) / block.x, 1);
  int output_size = grid.x * sizeof(int);

  printf("Execution Configure (block %d %d, grid %d %d)\n", block.x, block.y, grid.x, grid.y);

  int *h_idata, *h_odata, *h_check_odata;

  host_malloc((int**)&h_idata, NBytes);
  host_malloc((int**)&h_odata, output_size);
  host_malloc((int**)&h_check_odata, output_size);

  initialData(h_idata, nx);

  double iStart, iElips;
  iStart = cpuSecond();
  hostReduceNeighbored(h_idata, h_check_odata, nx);
  printElaps(iStart,iElips, "hostReduceNeighbored", grid, block);

  run_reduction_function(
  reduceNeighbored,
  "reduceNeighbored",
  h_idata,
  h_check_odata,
  grid,
  block,
  nx,
  nx
  );

  run_reduction_function(
  gpuRecursiveReduce,
  "gpuRecursiveReduce",
  h_idata,
  h_check_odata,
  grid,
  block,
  nx,
  block.x
  );

  run_reduction_function(
  gpuRecursiveReduceNoSync,
  "gpuRecursiveReduceNoSync",
  h_idata,
  h_check_odata,
  grid,
  block,
  nx,
  block.x
  );

  run_reduction_function(
  gpuRecursiveReduce2,
  "gpuRecursiveReduce2",
  h_idata,
  h_check_odata,
  grid,
  block,
  nx,
  block.x
  );
  printf("Creating pointers of %d elements.\n", nx);


  free(h_idata);
  free(h_odata);
  free(h_check_odata);
  cudaDeviceReset();

  return 0;
}