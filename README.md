# RayTracer - GPU Framework for Point Membership Classification of Geometric Models
The project objective is to embed a geometry in 3D structured grid and perform tests to check which grid points lie inside/outside the geometry. GPU framework is designed to deal with the large computation cost of this evaluation process. This algorithm has application while working in the field of immersed boundary analysis methods. 

For the algorithm used for testing the location of grid point, please refer to [Ray-Triangle Intersection](https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution).

## Dependencies
* [CUDA Toolkits 9.1](https://developer.nvidia.com/accelerated-computing-toolkit)

## User guide

* **Input:**

    * *.raw (Triangle mesh in .raw format)
    
* **Output:**

    * *.vtk (Ray tracing result in .vtk format, the file can be visualized using [Paraview 5.6.0](https://www.paraview.org/))
    
* **To compile the code** 

    In `./src` folder, run ` >> make` will compile the code to excutable file `kernel`
    
    The code has been compiled and tested on Ubuntu 16.04 with GCC 5.4.0 and CUDA 9.1. 
    
    The code was run on CMU qwe cluster with the machine specification: 
    
    * CPU: Intel(R) Core(TM) i7 CPU 960  @ 3.20GHz
    
    * GPU： Nvidia GeForce GTX 1080 Ti 
    
* **To check help information:**

   ` >> ./kernel -h` 

* **To run:**

   ` >> ./kernel -i <input_meshfile> -o <output_meshfile> --nx <ndivx> --ny <ndivy> --nz <ndivz> --pm <percentage>` 

   * `input_meshfile` is the path to the input mesh file * *.raw*, default setting: ../io/bunny_tri.raw
   
   * `output_meshfile` is the path to the output result file * *.vtk*, default setting: ../io/result.vtk
   
   * `percentage` is the percentage value of margin, default value: 0.01
   
   * `ndivx` is the number of grids in x direction, default value: 21
   
   * `ndivy` is the number of grids in y direction, default value: 21
   
   * `ndivz` is the number of grids in z direction, default value: 21
   
   * use `--cpu true` to switch on CPU code for comparison with GPU version

 * **Example:** 

   * Run `>> ./kernel` will generate the result for embedding the bunny geometry in a 21 * 21 * 21 grid using pure GPU code.
   
   * Run `>> ./kernel -i ../io/bunny_tri.raw -o ../io/result.vtk --nx 101 --ny 101 --nz 101 --pm 0.01` will generate the result for embedding the bunny geometry in a 101 * 101 * 101 grid using pure GPU code.
   
   * Run `>> ./kernel --cpu` to compare the computation time of CPU and GPU using bunny geometry.
   
  
   
   In `./io` folder, we provide the example input mesh file of bunny geometry.
