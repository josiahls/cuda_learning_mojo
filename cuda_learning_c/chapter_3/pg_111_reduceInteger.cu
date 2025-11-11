//===---------------------------BOILER PLATE--------------------------------====
#include <climits>
#include <cstdio>
#include <cuda_runtime.h>
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


__global__ void reduceNeighboredInterleaved(int *g_idata, int *g_odata, int n) {
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

  for (int stride=blockDim.x / 2; stride  > 0; stride >>= 1) {
    if (tid < stride) {
      // If stride is 128 and tid is 0
      //  - Then we sum points 0 and 128 (sampled from both halves of the block
      // If stride is 128 and tid is 127
      //  - Then we sum points 128 and 255
      idata[tid] += idata[tid + stride];
    }

    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

/*
Opted for just a diagram: 
cuda_learning_c/chapter_3/images/pg_111_reduceInteger_unrolling.png
*/
__global__ void reduceUnrolling2(int *g_idata, int *g_odata, int n) {
	unsigned int tid = threadIdx.x;
	// We are accessing 2 blocks per thread
	unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	int *idata = g_idata + blockIdx.x * blockDim.x * 2;

	// The unroll is summing consequtive data blocks
	if (idx + blockDim.x < n) {
	// Note: these are 2 blocks being unrolled. Equiv:
	//    int a1 = g_idata[idx]
	//    int a2 = g_idata[idx + blockDim.x]
	//    g_idata[idx] = a1 + a2; 

	// Example of unrolling if block size is 32:
	// where [tid,idx,sum_idx]:

	g_idata[idx] += g_idata[idx + blockDim.x];
	} 
	__syncthreads();

	// Important note, the reason why teh below loop doesn't change
	// with the number of unrolls is because the grid itself
	// is scaled smaller.
	for (int stride=blockDim.x / 2; stride  > 0; stride >>= 1) {
	if (tid < stride) {
		idata[tid] += idata[tid + stride];
	}

	__syncthreads();
	}

	if (tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceUnrolling4(int *g_idata, int *g_odata, int n) {
	unsigned int tid = threadIdx.x;
	const unsigned int unroll_factor = 4;
	// We are accessing 2 blocks per thread
	unsigned int block_start = blockIdx.x * blockDim.x * unroll_factor;
	unsigned int idx = block_start + threadIdx.x;
	int *idata = g_idata + block_start;
  
	if (idx + blockDim.x * (unroll_factor - 1) < n) {
	  // Unroll 3 data blocks (4th being handled in the stride loop)
	  int a1 = g_idata[idx];
	  int a2 = g_idata[idx + blockDim.x];
	  int a3 = g_idata[idx + blockDim.x * 2];
	  int a4 = g_idata[idx + blockDim.x * 3];

	  g_idata[idx] = a1 + a2 + a3 + a4;
	  
	} 
	__syncthreads();
  
	for (int stride=blockDim.x / 2; stride  > 0; stride >>= 1) {
	  if (tid < stride) {
		idata[tid] += idata[tid + stride];
	  }
  
	  __syncthreads();
	}
  
	if (tid == 0) g_odata[blockIdx.x] = idata[0];
  }


  __global__ void reduceUnrolling8(int *g_idata, int *g_odata, int n) {
	unsigned int tid = threadIdx.x;
	const unsigned int unroll_factor = 8;
	// We are accessing 2 blocks per thread
	unsigned int block_start = blockIdx.x * blockDim.x * unroll_factor;
	unsigned int idx = block_start + threadIdx.x;
	int *idata = g_idata + block_start;
  
	if (idx + blockDim.x * (unroll_factor - 1) < n) {
	  // Unroll 8 data blocks (4th being handled in the stride loop)
	  int a1 = g_idata[idx];
	  int a2 = g_idata[idx + blockDim.x];
	  int a3 = g_idata[idx + blockDim.x * 2];
	  int a4 = g_idata[idx + blockDim.x * 3];
	  int a5 = g_idata[idx + blockDim.x * 4];
	  int a6 = g_idata[idx + blockDim.x * 5];
	  int a7 = g_idata[idx + blockDim.x * 6];
	  int a8 = g_idata[idx + blockDim.x * 7];

	  g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
	  
	} 
	__syncthreads();

	// 512 -> 256
  
	for (int stride=blockDim.x / 2; stride  > 0; stride >>= 1) {
	  if (tid < stride) {
		idata[tid] += idata[tid + stride];
	  }
  
	  __syncthreads();
	}
  
	if (tid == 0) g_odata[blockIdx.x] = idata[0];
  }



void run_reduction_function(
	void (*gpu_reducer_fn)(int*, int*, int),
	const char * function_name,
	int NBytes,
	int *h_idata,
	int *h_check_odata,
	dim3 grid,
	dim3 block,
	int nx,
	int kernel_x_denom=1,
	int grid_x_denom=1
) {
	double iStart, iElips;
	int output_size = (grid.x / grid_x_denom) * sizeof(int);
	int *h_odata,*d_idata,*d_odata;
	host_malloc((int**)&h_odata, output_size);
	cudaMalloc((int**)&d_idata, NBytes);
	cudaMalloc((int**)&d_odata, output_size);
	cudaMemcpy(d_idata, h_idata, NBytes, cudaMemcpyHostToDevice);
	iStart = cpuSecond();
	gpu_reducer_fn<<< grid.x/kernel_x_denom, block>>> (d_idata, d_odata, nx);
	CHECK(cudaDeviceSynchronize());
	printElaps(iStart,iElips, function_name, grid, block);
	cudaMemcpy(h_odata, d_odata, output_size, cudaMemcpyDeviceToHost);
	hostAccumulateToIndex0(h_odata, grid.x/grid_x_denom);
	checkResult(h_check_odata, h_odata, nx);
	cudaFree(d_idata);
	cudaFree(d_odata);
	free(h_odata);
}



int main(int argc, char **argv) {
  const int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("%s Using device %d: %s \n", argv[0], dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  int nx = 1<<24;
  printf("Creating pointers of %d elements.\n", nx);
  size_t NBytes = nx * sizeof(int);

  int dimx = 512;
  if (argc > 1) dimx = atoi(argv[1]);
  printf("Data size x: %d y: %d\n", dimx, 1);

  dim3 block (dimx, 1);
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
	NBytes,
	h_idata,
	h_check_odata,
	grid,
	block,
	nx
  );

  run_reduction_function(
	reduceNeighboredLess,
	"reduceNeighboredLess",
	NBytes,
	h_idata,
	h_check_odata,
	grid,
	block,
	nx
  );

  run_reduction_function(
	reduceNeighboredInterleaved,
	"reduceNeighboredInterleaved",
	NBytes,
	h_idata,
	h_check_odata,
	grid,
	block,
	nx
  );
  //===-----------------------reduceUnrolling2----------------------------------
  run_reduction_function(
	reduceUnrolling2,
	"reduceUnrolling2",
	NBytes,
	h_idata,
	h_check_odata,
	grid,
	block,
	nx,
	2,
	2
  );

  run_reduction_function(
	reduceUnrolling4,
	"reduceUnrolling4",
	NBytes,
	h_idata,
	h_check_odata,
	grid,
	block,
	nx,
	4,
	4
  );

  run_reduction_function(
	reduceUnrolling8,
	"reduceUnrolling8",
	NBytes,
	h_idata,
	h_check_odata,
	grid,
	block,
	nx,
	8,
	8
  );

  free(h_idata);
  free(h_odata);
  free(h_check_odata);
  cudaDeviceReset();

  return 0;
}