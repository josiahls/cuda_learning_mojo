#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>


void printMultiProcessorCount() {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf(
        "Number of streaming multiprocessors: %d\n",
        deviceProp.multiProcessorCount
    );
}


void determineBestGPU() {
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (numDevices > 1) {
        int maxMultiprocessors = 0, maxDevice = 0;
        for (int device=0; device<numDevices; device++) {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, device);
            if (maxMultiprocessors < props.multiProcessorCount) {
                maxMultiprocessors = props.multiProcessorCount;
                maxDevice = device;
            }

        }
        printf("Selecting gpu: %d", maxDevice);
        cudaSetDevice(maxDevice);
    }
}


int main(int argc, char **argv) {
    printf("%s, Starting...\n", argv[0]);

    printMultiProcessorCount();

    determineBestGPU();

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf(
            "cudaGetDeviceCount returned %d\n-> %s\n",
            (int)error_id, cudaGetErrorString(error_id)
        );
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev = 0, driverVersion = 0, runtimeVersion = 0;

    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d: \n%s\"\n",dev,deviceProp.name);

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf(
        "\tCUDA Driver Version / Runtime Version                         %d.%d / %d.%d\n",
        driverVersion/1000, (driverVersion%100) / 10,
        runtimeVersion/1000, (runtimeVersion%100) / 10
    );
    printf(
        "\tCUDA Capability Major/Minor version number:                   %d.%d\n",
        deviceProp.major, deviceProp.minor
    );
    // NOTE: I think the book is wrong here. was pow(..., y=3).
    printf(
        "\tTotal amount of global memory:                                %.2f MBytes (%llu bytes)\n",
        (float)deviceProp.totalGlobalMem / pow(1024.0,2),
        (unsigned long long) deviceProp.totalGlobalMem
    );
    printf(
        "\tGPU Clock rate:                                               %.0f MHz (%0.2f GHz)\n",
        deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f
    );
    printf(
        "\tMemory Clock rate:                                           %.0f Mhz\n",
        deviceProp.memoryClockRate * 1e-3f
    );
    printf(
        "\tMemory Bus Width:                                            %d-bit\n",
        deviceProp.memoryBusWidth
    );

    printf(
        "\tNumber of multiprocessors:                                   %d\n",
        deviceProp.multiProcessorCount
    );
    printf(
        "\tTotal amount of shared memory per multiprocessor:            %zu bytes\n",
        deviceProp.sharedMemPerMultiprocessor
    );
    if (deviceProp.l2CacheSize) {
        printf(
            "\tL2 Cache Size:                                            %d bytes\n",
            deviceProp.l2CacheSize
        );
    }
    printf(
        "\tMax Texture Dimension Size (x,y,z)                           1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
        deviceProp.maxTexture1D,
        deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
        deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]
    );
    printf(
        "\tMax Layered Size (dim) x layers:                             1D=(%d) x %d, 2D=(%d, %d) x %d\n",
        deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
        deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
        deviceProp.maxTexture2DLayered[2]
    );
    printf("\tTotal amount of constant memory:                          %zu bytes\n", deviceProp.totalConstMem);
    printf("\tTotal amount of shared memory per block:                  %zu bytes\n", deviceProp.sharedMemPerBlock);
    printf("\tTotal number of registers available per block:            %d (32-bit each)\n", deviceProp.regsPerBlock);
    printf("\tTotal number of registers available per multiprocessor:   %d\n", deviceProp.regsPerMultiprocessor);
    printf("\tWarp size:                                                %d \n", deviceProp.warpSize);
    printf("\tMaximum number of threads per block:                      %d \n", deviceProp.maxThreadsPerBlock);
    printf(
        "\tMaximum sizes of each dimension of a block:                  %d x %d x %d\n",
        deviceProp.maxThreadsDim[0],
        deviceProp.maxThreadsDim[1],
        deviceProp.maxThreadsDim[2]
    );
    printf(
        "\tMaximum sizes of each dimension of a grid:                   %d x %d x %d\n",
        deviceProp.maxGridSize[0],
        deviceProp.maxGridSize[1],
        deviceProp.maxGridSize[2]
    );
    printf(
        "\tMaximum memory patch:                                        %zu bytes\n",
        deviceProp.memPitch
    );
    exit(EXIT_SUCCESS);
}