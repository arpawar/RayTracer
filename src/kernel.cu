/* This code performs point membership of 3D structured grid points that
embed the geometry. For the 3D structured grid we perform test to check which
grid points lie inside/outside the geometry. For each point in the grid a ray
starts from the grid point and in the x-direction to check for intersection
with the geometry. If the points are inside, the number of intersections are odd,
else the number of intersections are even.
*/

#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <ctime>

#include "cxxopts.hpp"
#include "basic_datastructure.h"
#include "RayTracer.h"

using namespace std;

//this function calculates the normals of the surface mesh
__global__ void calculate_normal(vertex3D *vertex, tri *face, int nface)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nface)
    {
        // indices of the vertices of the face
        int v0_ind = face[i].v[0];
        int v1_ind = face[i].v[1];
        int v2_ind = face[i].v[2];

        //edge vectors initialized
        vertex3D e1, e2;
        //normal vector initialized
        vertex3D nn;

        //set the first edge vector e1 = v1 - v0
        e1.x = vertex[v1_ind].x - vertex[v0_ind].x;
        e1.y = vertex[v1_ind].y - vertex[v0_ind].y;
        e1.z = vertex[v1_ind].z - vertex[v0_ind].z;

        //set the second edge vector e2 = v2 - v0
        e2.x = vertex[v2_ind].x - vertex[v0_ind].x;
        e2.y = vertex[v2_ind].y - vertex[v0_ind].y;
        e2.z = vertex[v2_ind].z - vertex[v0_ind].z;

        //compute normal vector as cross product of e1 and e2: n = cross(e1,e2)
        nn.x = e1.y * e2.z - e1.z * e2.y;
        nn.y = e1.z * e2.x - e1.x * e2.z;
        nn.z = e1.x * e2.y - e1.y * e2.x;

        //find the magnitude of the normal vector
        float n_norm = sqrt(pow(nn.x, 2) + pow(nn.y, 2) + pow(nn.z, 2));
        //normalize the normal vector and store it in the face object
        face[i].n.x = nn.x / n_norm;
        face[i].n.y = nn.y / n_norm;
        face[i].n.z = nn.z / n_norm;
    }
}

// this function performs ray tracing and assign point membership to each grid point
__global__ void point_membership(Meshio *Mesh_dev, vertex3D *origin_dev, vertex3D *vertex, tri *face, int *bbox_flag)
{
    int ngrid = Mesh_dev->ngrid;
    int nface = Mesh_dev->nface;
    //initialize flag (inside/outside) variable
    int f1;

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < ngrid)
    {
        vertex3D origin, ray;
        origin.x = origin_dev[i].x;
        origin.y = origin_dev[i].y;
        origin.z = origin_dev[i].z;
        //set ray direction along x-axis i.e. (1,0,0)
        ray.x = 1;
        ray.y = 0;
        ray.z = 0;
        //initialize no. of intersections of the ray with faces to 0
        int count = 0;
        //loop over faces
        for (int iface = 0; iface < nface; iface++)
        {
            //indices of vertices of the face
            int v0_ind = face[iface].v[0];
            int v1_ind = face[iface].v[1];
            int v2_ind = face[iface].v[2];
            //normal vector of the face
            vertex3D nn = face[iface].n;
            
            //refer to https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
			//for more details on the formulas and variables
			//the ray vector is defines as p_vec = origin + t*ray
			//here calculating the t parameter at which the ray intersects the face
            float D = -(nn.x * vertex[v0_ind].x + nn.y * vertex[v0_ind].y + nn.z * vertex[v0_ind].z);
            float t1 = -(nn.x * origin.x + nn.y * origin.y + nn.z * origin.z + D);
            float t2 = (nn.x * ray.x + nn.y * ray.y + nn.z * ray.z);
            float t = t1 / t2;
            
            //computing p_vec
            vertex3D p_vec;
            p_vec.x = origin.x + t * ray.x;
            p_vec.y = origin.y + t * ray.y;
            p_vec.z = origin.z + t * ray.z;
            
            //calculating edge vectors e0 = v1-v0, e1 = v2-v1 and e2 = v0-v2
            vertex3D e0, e1, e2, c0, c1, c2;
            e0.x = vertex[v1_ind].x - vertex[v0_ind].x;
            e0.y = vertex[v1_ind].y - vertex[v0_ind].y;
            e0.z = vertex[v1_ind].z - vertex[v0_ind].z;
            e1.x = vertex[v2_ind].x - vertex[v1_ind].x;
            e1.y = vertex[v2_ind].y - vertex[v1_ind].y;
            e1.z = vertex[v2_ind].z - vertex[v1_ind].z;
            e2.x = vertex[v0_ind].x - vertex[v2_ind].x;
            e2.y = vertex[v0_ind].y - vertex[v2_ind].y;
            e2.z = vertex[v0_ind].z - vertex[v2_ind].z;
            
            // calculating c0 = p_vec-v0, c1 = p_vec-v1, c2 = p_vec-v2
            c0.x = p_vec.x - vertex[v0_ind].x;
            c0.y = p_vec.y - vertex[v0_ind].y;
            c0.z = p_vec.z - vertex[v0_ind].z;
            c1.x = p_vec.x - vertex[v1_ind].x;
            c1.y = p_vec.y - vertex[v1_ind].y;
            c1.z = p_vec.z - vertex[v1_ind].z;
            c2.x = p_vec.x - vertex[v2_ind].x;
            c2.y = p_vec.y - vertex[v2_ind].y;
            c2.z = p_vec.z - vertex[v2_ind].z;
            
            //calculating e0c0 = cross(e0,c0), e1c1 = cross(e1,c1), e2c2 = cross(e2,c2)
            vertex3D e0c0, e1c1, e2c2;
            e0c0.x = e0.y * c0.z - e0.z * c0.y;
            e0c0.y = e0.z * c0.x - e0.x * c0.z;
            e0c0.z = e0.x * c0.y - e0.y * c0.x;
            e1c1.x = e1.y * c1.z - e1.z * c1.y;
            e1c1.y = e1.z * c1.x - e1.x * c1.z;
            e1c1.z = e1.x * c1.y - e1.y * c1.x;
            e2c2.x = e2.y * c2.z - e2.z * c2.y;
            e2c2.y = e2.z * c2.x - e2.x * c2.z;
            e2c2.z = e2.x * c2.y - e2.y * c2.x;

            //calculating case1 = dot(n,e0c0), case2 = dot(n,e1c1), case3 = dot(n,e2c2)
            float case1, case2, case3;
            case1 = nn.x * e0c0.x + nn.y * e0c0.y + nn.z * e0c0.z;
            case2 = nn.x * e1c1.x + nn.y * e1c1.y + nn.z * e1c1.z;
            case3 = nn.x * e2c2.x + nn.y * e2c2.y + nn.z * e2c2.z;
            
            //increment counter if p_vec is inside the triangle face
            if (case1 > 0 && case2 > 0 && case3 > 0 && t > 0)
            {
                count++;
            }
        }

        //if count is even->grid point is outside the geometry
        if (count % 2 == 0)
        {
            //set flag as 0
            f1 = 0;
            bbox_flag[i] = (f1);
        }
        //if count is odd->grid point is inside the geometry
        else
        {
            //set flag as 1
            f1 = 1;
            bbox_flag[i] = (f1);
        }
    }
}

int main(int argc, char *argv[])
{

    int ndivx(21), ndivy(21), ndivz(21); // number of grids in x, y, z direction
    float pm(0.01); // percentage of margin
    char *filename; // variable to store input file name
    char *filename_out; // variable to store output file name
    bool flag_cpu, flag_gpu; // switch of cpu and gpu code

    int nvert, nface;
    vector<vertex3D> vertex_host, origin_host;
    vector<tri> face_host;

    // Parse command line, set parameters and I/O
    try
    {
        cxxopts::Options options(argv[0], " RayTracing - Command line options");
        options
            .positional_help("[optional args]")
            .show_positional_help();

        options
            .allow_unrecognised_options()
            .add_options()
            ("pm", "Percentage of margin", cxxopts::value<float>()->default_value("0.01"))
            ("nx", "Number of grids in x direction", cxxopts::value<int>()->default_value("21"))
            ("ny", "Number of grids in y direction", cxxopts::value<int>()->default_value("21"))
            ("nz", "Number of grids in z direction", cxxopts::value<int>()->default_value("21"))
            ("cpu", "Run Raytracing in CPU", cxxopts::value<bool>()->default_value("false"))
            ("gpu", "Run Raytracing in GPU", cxxopts::value<bool>()->default_value("true"))
            ("i,input", "Input file", cxxopts::value<string>()->default_value("../io/bunny_tri.raw"))
            ("o,output", "Output file", cxxopts::value<string>()->default_value("../io/result.vtk"))
            ("h,help", "Print help")
#ifdef CXXOPTS_USE_UNICODE
                ("unicode", u8"A help option with non-ascii: à. Here the size of the"
                            " string should be correct")
#endif
            ;

        options.parse_positional({"input", "output"});

        auto result = options.parse(argc, argv);
        if (result.count("help"))
        {
            std::cout << options.help({""}) << std::endl;
            exit(0);
        }

        {
            pm = result["pm"].as<float>(); // set percentage of margin
            ndivx = result["nx"].as<int>(); // set number of grids in x direction
            ndivy = result["ny"].as<int>(); // set number of grids in y direction
            ndivz = result["nz"].as<int>(); // set number of grids in z direction
            flag_cpu = result["cpu"].as<bool>(); // set switch of cpu code
            flag_gpu = result["gpu"].as<bool>(); // set switch of gpu code

            // Set input file name
            string input_str = result["i"].as<string>();
            filename = (char *)alloca(input_str.size() + 1);
            memcpy(filename, input_str.c_str(), input_str.size() + 1);

            // Set output file name
            string output_str = result["o"].as<string>();
            filename_out = (char *)alloca(output_str.size() + 1);
            memcpy(filename_out, output_str.c_str(), output_str.size() + 1);
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }
    printf("----------Ray Tracing Settings----------\n");
    cout<< "Input mesh: "<< filename <<endl;
    cout<< "Grid resolution (nx * ny * nz): "<< ndivx << " * "<< ndivy << " * "<< ndivz <<endl;

    clock_t begin, end;
    double elapsed_secs;

    printf("----------Start Ray Tracing----------\n");
    
    printf("Reading Geometry\n");
    Meshio Mesh_host;
    if(Mesh_host.flag==1)
    {
    Mesh_host.read_raw_file(filename, vertex_host, face_host); // Read input triangle mesh
    Mesh_host.set_bounding_box(ndivx, ndivy, ndivz, pm); // Set up bounding box 
    Mesh_host.calculate_grid(origin_host); // Compute the coordinates of all grid points
    
    if(flag_cpu)
    {
        printf("----------Starting CPU Code----------\n");
        begin = clock();
        Mesh_host.calculate_normal(); // Computing normal vector for each element on CPU
        printf("Ray Tracing on CPU\n");
        Mesh_host.point_membership(); // Raytracing on CPU
        end = clock();
        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        printf("CPU computing time: %f seconds\n", elapsed_secs);
    }
    
    if(flag_gpu)
    {
        printf("----------Starting GPU Code----------\n");

        cudaError_t cudaStatus = cudaSuccess;
        size_t size_class = int(sizeof(Meshio));
        int ngrid = ndivx * ndivy * ndivz;
        nvert = vertex_host.size();
        nface = face_host.size();
        int thread_per_block = 10;
    
        // Allocate memory on GPU and transfer mesh information to GPU
        printf("Sending Mesh to GPU\n");
        Meshio *Mesh_dev = NULL;
        vertex3D *vertex_dev = NULL;
        vertex3D *origin_dev = NULL;
        tri *face_dev = NULL;
        cudaStatus = cudaMalloc((void **)&Mesh_dev, size_class);
        cudaStatus = cudaMalloc((void **)&vertex_dev, nvert * sizeof(vertex3D));
        cudaStatus = cudaMalloc((void **)&origin_dev, ngrid * sizeof(vertex3D));
        cudaStatus = cudaMalloc((void **)&face_dev, nface * sizeof(tri));
    
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device mesh (error code %s)!\n", cudaGetErrorString(cudaStatus));
            exit(EXIT_FAILURE);
        }
    
        cudaStatus = cudaMemcpy(Mesh_dev, &Mesh_host, size_class, cudaMemcpyHostToDevice);
        cudaStatus = cudaMemcpy(vertex_dev, &vertex_host[0], nvert * sizeof(vertex3D), cudaMemcpyHostToDevice);
        cudaStatus = cudaMemcpy(origin_dev, &origin_host[0], ngrid * sizeof(vertex3D), cudaMemcpyHostToDevice);
        cudaStatus = cudaMemcpy(face_dev, &face_host[0], nface * sizeof(tri), cudaMemcpyHostToDevice);
    
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy host mesh to device mesh (error code %s)!\n", cudaGetErrorString(cudaStatus));
            exit(EXIT_FAILURE);
        }
    
        begin = clock();
        // Computing normal vector for each element on GPU
        calculate_normal<<<(nface + thread_per_block - 1) / thread_per_block, thread_per_block>>>(vertex_dev, face_dev, nface);
        cudaDeviceSynchronize();
    
        // Allocate memory to store result on GPU
        int *bbox_flag_dev = NULL;
        cudaStatus = cudaMalloc((void **)&bbox_flag_dev, ngrid * sizeof(int));
        cudaDeviceSynchronize();
    
        // Raytracing on GPU
        printf("Ray tracing on GPU\n");
        point_membership<<<(ngrid + thread_per_block - 1) / thread_per_block, thread_per_block>>>(Mesh_dev, origin_dev, vertex_dev, face_dev, bbox_flag_dev);
        cudaDeviceSynchronize();
    
        end = clock();
        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        printf("GPU computing time: %f seconds\n", elapsed_secs);
    
        // Transfer raytracing result from GPU to CPU
        int bbox_flag_host[ngrid]; /// Use this variable to output the result
        cudaMemcpy((void *)bbox_flag_host, (void *)bbox_flag_dev, ngrid * sizeof(int), cudaMemcpyDeviceToHost);
    
        // Generate vtk result
        Mesh_host.display_result(filename_out, bbox_flag_host);
    
        cudaFree(Mesh_dev);
        cudaFree(vertex_dev);
        cudaFree(origin_dev);
        cudaFree(face_dev);
    }
    }
    
    else
    {
        printf("Error creating Mesh object\n");
    }

    return 0;
}
