"""
Note Mojo doesn't support nested kernel calls so this script will not actually compile.

Link: https://forum.modular.com/t/does-mojo-support-parent-child-grids-nested-kernels/2458

"""


from gpu import thread_idx,block_idx,block_dim
from gpu.host import Dim, DeviceContext
import sys
from gpu.host.info import GPUInfo, _accelerator_arch


fn nestedHelloWorld(i_size: Int, i_depth: Int):
	var tid:UInt = thread_idx.x
	print("Recursion=", i_depth," :Hello World from thread ",tid," block ",block_idx.x)

	if i_size == 1: return

	# Decrease the number of threads by the power of 2 (rshift)
	var nthreads:Int = i_size >> 1

	if (tid == 0 and nthreads > 0 ):
		# Dynamic parallelism - requires -rdc=true for nvcc compilation
		# clangd doesn't support CUDA dynamic parallelism checking
		var inner_i_depth = i_depth + 1
		try:
			with DeviceContext() as ctx:
				ctx.enqueue_function_checked[nestedHelloWorld,nestedHelloWorld](
					i_size,inner_i_depth,
					grid_dim=1,
					block_dim=nthreads
				)
				ctx.synchronize()
		except:
			print('failed.')
		# nestedHelloWorld<<<1, nthreads>>>(nthreads, inner_i_depth)
		print("--------> nested execution depth: ",inner_i_depth)

def main():

	alias dev: Int = 0
	gpu_info: GPUInfo = GPUInfo.from_name[_accelerator_arch()]()
	print("Device ", dev, ":", gpu_info)

	var nx:Int = 1<<4
	var dimx:Int = 512
	if len(sys.argv()) > 1:
		dim_x = Int(sys.argv()[1])
	print("Data size x: ", dimx, " y:", 1)

	var block:Dim =  {dimx, 1}
	var grid:Dim =   {Int(Int(nx + block.x() - 1) / block.x()), 1}

	with DeviceContext(dev) as ctx:
		ctx.enqueue_function_checked[nestedHelloWorld,nestedHelloWorld](
			nx,
			0,
			grid_dim=grid,
			block_dim=block
		)
		ctx.synchronize()


