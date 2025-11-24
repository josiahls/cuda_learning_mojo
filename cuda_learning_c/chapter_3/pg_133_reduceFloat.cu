//===---------------------------BOILER PLATE--------------------------------====
#include <atomic>
#include <climits>
#include <cmath>
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

void host_malloc(float **ptr, int nBytes) {  
  *ptr = (float *)malloc(nBytes); 
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
    fp[i] = (float)(rand() & 0xFF);
  }
}

void checkResult(float *h_C, float *h_GpuCheck, int nElem) {
  bool match = true;
  if (abs(h_C[0] - h_GpuCheck[0]) > 1e-5) {
    printf("Error sums do not match: %f != %f\n", h_C[0], h_GpuCheck[0]);
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

void hostAccumulateToIndex0(float *ip, int grid_x) {
  for (int i = 1; i < grid_x; i++) {
    ip[0] += ip[i];
  }
}

void hostReduceNeighbored(float *g_idata, float *g_odata, int n) {
  float num = 0;
  for (int idx = 0; idx < n; idx++) {
    num += g_idata[idx];
  }
  printf("Final sum: %f\n",num);
  g_odata[0] = num;
  if (MAXFLOAT <= num) {
    printf("At max size!\n");
  }
}

__device__ void __syncthreads(void);


__global__ void reduceNeighbored(float *g_idata, float *g_odata, int n) {
	// e.g. thread 1
	unsigned int tid = threadIdx.x;
	// e.g.: block 1 * 512(block size) + 1 -> 513
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Make a local pointer for the data in this block:
	//  - g_idata + block start index (pointer arithmetic).
	float *idata = g_idata + blockIdx.x * blockDim.x;

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


__global__ void reduceNeighboredInterleaved(float *g_idata, float *g_odata, int n) {
	// e.g. thread 1
	unsigned int tid = threadIdx.x;
	// e.g.: block 1 * 512(block size) + 1 -> 513
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
	// Make a local pointer for the data in this block:
	//  - g_idata + block start index (pointer arithmetic).
	float *idata = g_idata + blockIdx.x * blockDim.x;
  
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


  __global__ void reduceUnrolling8UnrollLastWarp(float *g_idata, float *g_odata, int n) {
	unsigned int tid = threadIdx.x;
	const unsigned int unroll_factor = 8;
	// We are accessing 2 blocks per thread
	unsigned int block_start = blockIdx.x * blockDim.x * unroll_factor;
	unsigned int idx = block_start + threadIdx.x;
	float *idata = g_idata + block_start;

	if (idx + blockDim.x * (unroll_factor - 1) < n) {
		// Unroll 8 data blocks (4th being handled in the stride loop)
		float a1 = g_idata[idx];
		float a2 = g_idata[idx + blockDim.x];
		float a3 = g_idata[idx + blockDim.x * 2];
		float a4 = g_idata[idx + blockDim.x * 3];
		float a5 = g_idata[idx + blockDim.x * 4];
		float a6 = g_idata[idx + blockDim.x * 5];
		float a7 = g_idata[idx + blockDim.x * 6];
		float a8 = g_idata[idx + blockDim.x * 7];

		g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
		
	} 
	__syncthreads();

	for (int stride=blockDim.x / 2; stride  > 32; stride >>= 1) {
		if (tid < stride) {
			idata[tid] += idata[tid + stride];
		}

		__syncthreads();
	}

	__syncthreads();

	if (tid < warpSize) {
		// Note manually unrolls the final strides of the above for loop.
		// Note: Removing volatile doesn't effect this, but likely because
		// this is otherwise nondeterministic.
		volatile float *vmem = idata;// If all values are 10
		//                            vmem[idx] == 10
		vmem[tid] += vmem[tid + 32];//vmem[idx] == 20
		vmem[tid] += vmem[tid + 16];//vmem[idx] == 30
		vmem[tid] += vmem[tid +  8];//vmem[idx] == 40
		vmem[tid] += vmem[tid +  4];//vmem[idx] == 50
		vmem[tid] += vmem[tid +  2];//vmem[idx] == 60	
		vmem[tid] += vmem[tid +  1];//vmem[idx] == 70
		printf("Unrolling 8: tid=%d idata[tid]=%f\n", tid, idata[tid]);
		
		// float *vmem = idata;// If all values are 10
		// //                            vmem[idx] == 10
		// __syncthreads();
		// vmem[tid] += vmem[tid + 32];//vmem[idx] == 20
		// __syncthreads();
		// vmem[tid] += vmem[tid + 16];//vmem[idx] == 30
		// __syncthreads();
		// vmem[tid] += vmem[tid +  8];//vmem[idx] == 40
		// __syncthreads();
		// vmem[tid] += vmem[tid +  4];//vmem[idx] == 50
		// __syncthreads();
		// vmem[tid] += vmem[tid +  2];//vmem[idx] == 60
		// __syncthreads();
		// vmem[tid] += vmem[tid +  1];//vmem[idx] == 70
		// __syncthreads();
	}

	if (tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceUnrolling16(float *g_idata, float *g_odata, int n) {
	unsigned int tid = threadIdx.x;
	const unsigned int unroll_factor = 16;
	// We are accessing 2 blocks per thread
	unsigned int block_start = blockIdx.x * blockDim.x * unroll_factor;
	unsigned int idx = block_start + threadIdx.x;
	float *idata = g_idata + block_start;

	if (idx + blockDim.x * (unroll_factor - 1) < n) {
		// Unroll 8 data blocks (4th being handled in the stride loop)
		float a1 = g_idata[idx];
		float a2 = g_idata[idx + blockDim.x];
		float a3 = g_idata[idx + blockDim.x * 2];
		float a4 = g_idata[idx + blockDim.x * 3];
		float a5 = g_idata[idx + blockDim.x * 4];
		float a6 = g_idata[idx + blockDim.x * 5];
		float a7 = g_idata[idx + blockDim.x * 6];
		float a8 = g_idata[idx + blockDim.x * 7];
		float a9 = g_idata[idx + blockDim.x * 8];
		float a10 = g_idata[idx + blockDim.x * 9];
		float a11 = g_idata[idx + blockDim.x * 10];
		float a12 = g_idata[idx + blockDim.x * 11];
		float a13 = g_idata[idx + blockDim.x * 12];
		float a14 = g_idata[idx + blockDim.x * 13];
		float a15 = g_idata[idx + blockDim.x * 14];
		float a16 = g_idata[idx + blockDim.x * 15];

		g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 + a16;
		
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
	void (*gpu_reducer_fn)(float*, float*, int),
	const char * function_name,
	float *h_idata,
	float *h_check_odata,
	dim3 grid,
	dim3 block,
	int nx,
	int kernel_x_denom=1,
	int grid_x_denom=1,
	bool use_template = false
) {
	double iStart, iElips;
	int input_size = nx * sizeof(float);
	int output_size = (grid.x / grid_x_denom) * sizeof(float);
	float *h_odata,*d_idata,*d_odata;
	host_malloc((float**)&h_odata, output_size);
	cudaMalloc((float**)&d_idata, input_size);
	cudaMalloc((float**)&d_odata, output_size);
	cudaMemcpy(d_idata, h_idata, input_size, cudaMemcpyHostToDevice);
	iStart = cpuSecond();

	gpu_reducer_fn<<< grid.x/kernel_x_denom, block>>> (d_idata, d_odata, nx);

	CHECK(cudaDeviceSynchronize());
	printElaps(iStart,iElips, function_name, grid, block);
	cudaMemcpy(h_odata, d_odata, output_size, cudaMemcpyDeviceToHost);
	hostAccumulateToIndex0(h_odata, grid.x/grid_x_denom);
	checkResult(h_odata, h_check_odata, nx);
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

  int nx = 1<<24;//24;
  printf("Creating pointers of %d elements.\n", nx);
  size_t NBytes = nx * sizeof(float);

  int dimx = 512;
  if (argc > 1) dimx = atoi(argv[1]);
  printf("Data size x: %d y: %d\n", dimx, 1);

  dim3 block (dimx, 1);
  dim3 grid((nx + block.x - 1) / block.x, 1);
  int output_size = grid.x * sizeof(float);

  printf("Execution Configure (block %d %d, grid %d %d)\n", block.x, block.y, grid.x, grid.y);

  float *h_idata, *h_odata, *h_check_odata;

  host_malloc((float**)&h_idata, NBytes);
  host_malloc((float**)&h_odata, output_size);
  host_malloc((float**)&h_check_odata, output_size);

  initialData(h_idata, nx);

  double iStart, iElips;
  iStart = cpuSecond();
  hostReduceNeighbored(h_idata, h_check_odata, nx);
  printElaps(iStart,iElips, "hostReduceNeighbored", grid, block);

//   run_reduction_function(
// 	reduceNeighbored,
// 	"reduceNeighbored",
// 	h_idata,
// 	h_check_odata,
// 	grid,
// 	block,
// 	nx
//   );

//   run_reduction_function(
// 	reduceNeighboredInterleaved,
// 	"reduceNeighboredInterleaved",
// 	h_idata,
// 	h_check_odata,
// 	grid,
// 	block,
// 	nx
//   );

  run_reduction_function(
	reduceUnrolling8UnrollLastWarp,
	"reduceUnrolling8UnrollLastWarp",
	h_idata,
	h_check_odata,
	grid,
	block,
	nx,
	8,
	8
  );

//   run_reduction_function(
// 	reduceUnrolling16,
// 	"reduceUnrolling16",
// 	h_idata,
// 	h_check_odata,
// 	grid,
// 	block,
// 	nx,
// 	16,
// 	16
//   );

  free(h_idata);
  free(h_odata);
  free(h_check_odata);
  cudaDeviceReset();

  return 0;
}