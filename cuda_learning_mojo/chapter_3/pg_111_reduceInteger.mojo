import sys
import time
from random import rand, seed, randint
from gpu import thread_idx, block_idx, block_dim, WARP_SIZE, barrier
from gpu.host import DeviceContext, Dim, HostBuffer
from gpu.host.info import GPUInfo, _accelerator_arch
from memory import memset
from gpu.random import Random


alias Int32Ptr = UnsafePointer[Int32]


struct Timer:
    var fn_name: String
    var start_sec: Float64

    fn __init__(out self, var fn_name: String):
        self.fn_name = fn_name
        self.start_sec = time.perf_counter_ns() / 1e9

    fn __enter__(ref self) -> ref [self] Self:
        return self

    fn __exit__(self):
        var end_sec = time.perf_counter_ns() / 1e9
        print(
            self.fn_name, "elapsed time: ", end_sec - self.start_sec, "seconds"
        )


fn initialize_data[n_elem: Int](ptr: Int32Ptr):
	randint(ptr, n_elem, 0, 255)


fn host_accumulate_to_index_0[n: Int](out_data: Int32Ptr):
	for i in range(1, n):
		out_data[0] += out_data[i]


fn host_reduce_neighbored[n: Int](in_data: Int32Ptr, mut out_data: Int32Ptr):
    var num: Int32 = 0
    for i in range(n):
        num += in_data[i]
    out_data[0] = num


fn reduce_neighbored[n: UInt](in_data: Int32Ptr, out_data: Int32Ptr):
	var tid: UInt = thread_idx.x
	var block_idx_start = block_idx.x * block_dim.x
	var idx: UInt = block_idx_start + thread_idx.x

	var i_data: Int32Ptr = in_data + block_idx_start

	if idx >= n:
		return

	var stride: UInt = 1
	while stride < block_dim.x:
		if tid % (UInt(2) * stride) == 0:
			i_data[tid] += i_data[tid + stride]

		barrier()
		
		stride *= 2

	if tid == 0:
		out_data[block_idx.x] = i_data[0]

fn reduce_neighbored_less[n: UInt](in_data: Int32Ptr, out_data: Int32Ptr):
	var tid: UInt = thread_idx.x
	var block_idx_start = block_idx.x * block_dim.x
	var idx: UInt = block_idx_start + thread_idx.x

	var i_data: Int32Ptr = in_data + block_idx_start

	if idx >= n:
		return

	var stride: UInt = 1
	while stride < block_dim.x:
		var index:UInt = UInt(2) * stride * tid
		if index < block_dim.x:
			i_data[tid] += i_data[index + stride]

		barrier()
		
		stride *= 2

	if tid == 0:
		out_data[block_idx.x] = i_data[0]


fn reduce_unrolling_2[n: UInt](in_data: Int32Ptr, out_data: Int32Ptr):
	var tid: UInt = thread_idx.x
	# Access 2 blocks per thread.
	var block_idx_start = block_idx.x * block_dim.x * 2
	var idx: UInt = block_idx_start + thread_idx.x

	var i_data: Int32Ptr = in_data + block_idx_start

	if idx + block_dim.x < n:
		in_data[idx] += in_data[idx + block_dim.x]
		barrier()
 
	var stride: UInt = UInt(block_dim.x / 2)
	while stride > 0:
		if tid < stride:
			i_data[tid] += i_data[tid + stride]

		barrier()
		stride >>= 2

	if tid == 0:
		out_data[block_idx.x] = i_data[0]

fn check_result(h_C: Int32Ptr, h_GpuCheck: Int32Ptr, n_elem: Int):
	if h_C[0] != h_GpuCheck[0]:
		print("Error at index", 0, ":", h_C[0], "!=", h_GpuCheck[0])
		print("Test Failed")
		return

	print("Test passed")


fn test_function_reduce[
	nx:Int, 
	function: fn(Int32Ptr, Int32Ptr) -> None
](
	mut ctx:DeviceContext, 
	function_name: String,
	h_in_data: HostBuffer[Int32.dtype], 
	h_out_data_checked: HostBuffer[Int32.dtype],
	grid: Dim, 
	block: Dim
) raises:
	print("===Begin " + function_name + "===")
	with Timer(function_name):
		h_out_data = ctx.enqueue_create_host_buffer[Int32.dtype](nx)
		d_in_data = ctx.enqueue_create_buffer[Int32.dtype](nx)
		d_out_data = ctx.enqueue_create_buffer[Int32.dtype](nx)
		ctx.synchronize()

		ctx.enqueue_copy(d_in_data, h_in_data)
		ctx.synchronize()

		ctx.enqueue_function_checked[
				reduce_neighbored[UInt(nx)], reduce_neighbored[UInt(nx)]
		](
			d_in_data, d_out_data,  grid_dim=grid, block_dim=block
		)
		# This copy isn't happening it looks like.
		ctx.enqueue_copy(h_out_data, d_out_data)

		with Timer(function_name + "_sync"):
			ctx.synchronize()
			host_accumulate_to_index_0[nx](h_out_data.unsafe_ptr())

		check_result(h_out_data.unsafe_ptr(), h_out_data_checked.unsafe_ptr(), n_elem=nx)
	print("===End " + function_name + "===")



fn main() raises:
	print(sys.argv()[0], "Starting...")
	seed(Int(time.perf_counter_ns()))

	alias dev: Int = 0
	gpu_info: GPUInfo = GPUInfo.from_name[_accelerator_arch()]()
	print("Device ", dev, ":", gpu_info)

	alias nx = 1 << 24
	alias n_bytes = nx * sys.info.size_of[Int32]()
	var dim_x = 16
	if len(sys.argv()) > 1:
		dim_x = Int(sys.argv()[1])
	print("Data size:", nx)
	print("Dimensions: x: ", dim_x, "y:", 1)

	var block: Dim = {dim_x, 1}
	var grid: Dim = {
		(nx + block.x() - 1) // block.x(),
		1,
	}
	# var grid = Dim(1)
	# var block = Dim(1)

	print(
		"Execution Configure (block:",
		block.x(),
		block.y(),
		"grid:",
		grid.x(),
		grid.y(),
		")",
	)

	with DeviceContext(dev) as ctx:
		with Timer("host_initalization"):
			h_out_data_checked = ctx.enqueue_create_host_buffer[Int32.dtype](nx)
			h_in_data = ctx.enqueue_create_host_buffer[Int32.dtype](nx)
		with Timer("host_sync"):
			ctx.synchronize()

		initialize_data[nx](h_in_data.unsafe_ptr())

		with Timer("host_reduce_neighbored"):
			host_reduce_neighbored[nx](
				h_in_data.unsafe_ptr(), h_out_data_checked._host_ptr
			)

		test_function_reduce[
			nx,
			reduce_neighbored[UInt(nx)]
		](ctx, 'reduce_neighbored' ,h_in_data,h_out_data_checked, grid=grid,block=block)

		test_function_reduce[
			nx,
			reduce_neighbored_less[UInt(nx)]
		](ctx, 'reduce_neighbored_less' ,h_in_data,h_out_data_checked, grid=grid,block=block)
		test_function_reduce[
			nx,
			reduce_unrolling_2[UInt(nx)]
		](ctx, 'reduce_unrolling_2' ,h_in_data,h_out_data_checked, grid=grid,block=block)
