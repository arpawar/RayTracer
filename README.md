# RayTracer - GPU Framework for Point Membership Classification of Geometric Models
The project objective is to embed a geometry in 3D structured grid and perform tests to check which grid points lie inside/outside the geometry. GPU framework is designed to deal with the large computation cost of this evaluation process. This algorithm has applications while working in the field of immersed boundary analysis methods.

For the algorithm used for carrying out the point membership classification of each grid point, please refer to [Ray-Triangle Intersection](https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution).

The command line parser used in this project: [Lightweight C++ command line option parser](https://github.com/jarro2783/cxxopts)

This project includes implementation of the ray tracing algorithm for triangulated surface geometry. Users can extend the algorithm for quadrilateral meshes and CAD models. 

References:

[1] Hsu, M. C., Wang, C., Xu, F., Herrema, A. J., & Krishnamurthy, A. Direct immersogeometric fluid flow analysis using B-rep CAD models. Computer Aided Geometric Design, 43, 143-158, 2016.

[2] Towns, J., Cockerill, T., Dahan, M., Foster, I., Gaither, K., Grimshaw, A., ... & Roskies, R., XSEDE: accelerating scientific discovery. Computing in Science & Engineering, 16(5), 62-74, 2014.

[3]  Klein, F., A new approach to point membership classification in B-rep solids. IMA International Conference on Mathematics of Surfaces. Springer, Berlin, Heidelberg, 2009.

[4]  Wikipedia contributors. "Graphics processing unit." Wikipedia, The Free Encyclopedia. Wikipedia, The Free Encyclopedia, 19 Apr. 2019. Web. 7 May. 2019.

[5]  Wikipedia contributors. "Ray tracing (graphics)." Wikipedia, The Free Encyclopedia. Wikipedia, The Free Encyclopedia, 7 May. 2019. Web. 7 May. 2019.

[6] Requicha, Ari. Geometric modeling: A first course. (1996).

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

    * GPU: Nvidia GeForce GTX 1080 Ti

    We also ran the code on Pittsburgh Supercomputing Center with the machine specification as follows:

    * CPU: Intel Xeon E5-2683 v4

    * GPU: NVIDIA Tesla P100

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



   In `./io` folder, we provide example input mesh files for five different geometries, namely, bunny, eight, kitten, rod and sculpture. The corresponding result for each model is also provided, the grid resolution used here is 101 * 101 * 101.


## Website User Guide

* **Extract the code from the archive:**
     * First move to the folder where you ask the browser look at (default location). In my computer, it is in **~/Sites/**.
     * Use `>> unzip Website.zip` to extract code.
* **Recover the mysql database:**
     * Based on the file **db_credentials.php** in **private** folder, we know username of database is **webuser**, password of database is **secretpassword**.
     * Create user, the following is done in root user. Give Privilege
       * `>> mysql -u root -p`
       * `>> CREATE USER 'webuser'@'localhost' IDENTIFIED BY 'secretpassword';`
       * `>> GRANT ALL PRIVILEGES ON *.* TO ' webuser'@'localhost';`
     * Create database `>> CREATE DATABASE Ray_Tracing;` and exit;
     * Type: `>> mysql -u webuser -p Ray_Tracing < /Users/yuyuxuan/Sites/Website/Ray_Tracing.sql` in the terminal.
     * (Optional) if we had problem **Warning: mysqli_connect():** in browser, we need to change the mysql password type, in mysql: `>> ALTER USER 'webuser'@'localhost' IDENTIFIED WITH mysql_native_password BY 'secretpassword';`

* **Permissions needed for the code files:**

    * Read and write.
