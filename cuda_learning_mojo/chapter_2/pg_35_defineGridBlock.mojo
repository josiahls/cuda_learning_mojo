from gpu.host import Dim

fn main():
    # NOTE: What was the point of this exercise? This appears to just demo basic
    # struct fields + math. 
    n_elem:Int = 1024;

    var block: Dim = (n_elem)
    var grid: Dim = Int((n_elem + block.x() - 1) / block.x())
    print("grid.x ", grid.x(), " block.x ", block.x())

    block.set_x(512)
    grid.set_x(Int((n_elem + block.x() - 1) / block.x()))
    print("grid.x ", grid.x(), " block.x ", block.x())

    block.set_x(256)
    grid.set_x(Int((n_elem + block.x() - 1) / block.x()))
    print("grid.x ", grid.x(), " block.x ", block.x())

    block.set_x(128)
    grid.set_x(Int((n_elem + block.x() - 1) / block.x()))
    print("grid.x ", grid.x(), " block.x ", block.x())

