from gpu.host.info import GPUInfo, _accelerator_arch
import sys



fn main() raises:
    print(sys.argv()[0], "Starting...")

    alias dev:Int = 0
    gpu_info: GPUInfo = GPUInfo.from_name[_accelerator_arch()]()


    print("Device ", dev, ":", gpu_info.name)
    print("Number of multiprocessors: ", gpu_info.sm_count)
    # Mojo does not expose this.BFloat16
    # print("Total amount of constant memory: ", gpu_info.totalConstMem / 1024.0, "KB")
    print("Total amount of shared memory per block:", gpu_info.shared_memory_per_multiprocessor / 1024.0, "KB")
    print("Total number of registers available per block: ", gpu_info.max_registers_per_block)
    print("Warp size: ", gpu_info.warp_size)
    print("Maximum number of threads per block:", gpu_info.max_thread_block_size)
    print("Maximum number of threads per multiprocessor:", gpu_info.threads_per_multiprocessor)
    print("Maximum number of warps per multiprocessor:", gpu_info.threads_per_multiprocessor / 32)