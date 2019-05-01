#include <stdio.h>
#include <typeinfo>
#include <iostream> 
#include <vector>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "basic_datastructure.h"
#include "read_raw.h"
#include <ctime>
using namespace std;

// #define ndivx 100
// #define ndivy 100
// #define ndivz 100

__global__ void calculate_normal(vertex3D* vertex, tri* face, int nface)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // for (int i = 0; i < nface; i++)
    if(i < nface)
	{
		int v0_ind = face[i].v[0];
		int v1_ind = face[i].v[1];
		int v2_ind = face[i].v[2];

		vertex3D e1, e2;
		vertex3D nn;

		e1.x = vertex[v1_ind].x - vertex[v0_ind].x;
		e1.y = vertex[v1_ind].y - vertex[v0_ind].y;
		e1.z = vertex[v1_ind].z - vertex[v0_ind].z;

		e2.x = vertex[v2_ind].x - vertex[v0_ind].x;
		e2.y = vertex[v2_ind].y - vertex[v0_ind].y;
		e2.z = vertex[v2_ind].z - vertex[v0_ind].z;

		nn.x = e1.y*e2.z - e1.z*e2.y;
		nn.y = e1.x*e2.z - e1.z*e2.x;
		nn.z = e1.x*e2.y - e1.y*e2.x;

		float n_norm = sqrt(pow(nn.x, 2) + pow(nn.y, 2) + pow(nn.z, 2));
		face[i].n.x = (e1.y*e2.z - e1.z*e2.y)/n_norm;
		face[i].n.y = (e1.x*e2.z - e1.z*e2.x)/n_norm;
		face[i].n.z = (e1.x*e2.y - e1.y*e2.x)/n_norm;
    }
    // printf("EleIdx: %d, BlockIdx: %d, ThreadIdx: %d, %f %f %f, Done!\n", i ,blockIdx.x, threadIdx.x, face[i].n.x, face[i].n.y,face[i].n.z);

}

__global__ void point_membership(Meshio *Mesh_dev, vertex3D *origin_dev, vertex3D* vertex, tri* face, int* bbox_flag)
{
    int ngrid = Mesh_dev->ngrid;
    int nface = Mesh_dev->nface;
    int f1;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // for (int i = 0; i < nface; i++)
    if(i < ngrid)
	{
		vertex3D origin, ray;
		origin.x = origin_dev[i].x;
		origin.y = origin_dev[i].y;
		origin.z = origin_dev[i].z;
		ray.x = 1;
		ray.y = 0;
		ray.z = 0;
		int count = 0;
		for (int iface = 0; iface < nface; iface++)
		{
			int v0_ind = face[iface].v[0];
			int v1_ind = face[iface].v[1];
			int v2_ind = face[iface].v[2];
			vertex3D nn = face[iface].n;
			float D = -(nn.x*vertex[v0_ind].x + nn.y*vertex[v0_ind].y + nn.z*vertex[v0_ind].z);
			float t1 = -(nn.x*origin.x + nn.y*origin.y + nn.z*origin.z + D);
			float t2 = (nn.x*ray.x + nn.y*ray.y + nn.z*ray.z);
			float t = t1 / t2;
			vertex3D p_vec;
			p_vec.x = origin.x + t*ray.x;
			p_vec.y = origin.y + t*ray.y;
			p_vec.z = origin.z + t*ray.z;
			vertex3D e0, e1, e2, c0, c1, c2;
			e0.x = vertex[v1_ind].x - vertex[v0_ind].x;  e0.y = vertex[v1_ind].y - vertex[v0_ind].y;  e0.z = vertex[v1_ind].z - vertex[v0_ind].z;
			e1.x = vertex[v2_ind].x - vertex[v1_ind].x;  e1.y = vertex[v2_ind].y - vertex[v1_ind].y;  e1.z = vertex[v2_ind].z - vertex[v1_ind].z;
			e2.x = vertex[v0_ind].x - vertex[v2_ind].x;  e2.y = vertex[v0_ind].y - vertex[v2_ind].y;  e2.z = vertex[v0_ind].z - vertex[v2_ind].z;
			c0.x = p_vec.x - vertex[v0_ind].x;  c0.y = p_vec.y - vertex[v0_ind].y;  c0.z = p_vec.z - vertex[v0_ind].z;
			c1.x = p_vec.x - vertex[v1_ind].x;  c1.y = p_vec.y - vertex[v1_ind].y;  c1.z = p_vec.z - vertex[v1_ind].z;
			c2.x = p_vec.x - vertex[v2_ind].x;  c2.y = p_vec.y - vertex[v2_ind].y;  c2.z = p_vec.z - vertex[v2_ind].z;
			vertex3D e0c0, e1c1, e2c2;
			e0c0.x = e0.y*c0.z - e0.z*c0.y; e0c0.y = e0.x*c0.z - e0.z*c0.x; e0c0.z = e0.x*c0.y - e0.y*c0.x;
			e1c1.x = e1.y*c1.z - e1.z*c1.y; e1c1.y = e1.x*c1.z - e1.z*c1.x; e1c1.z = e1.x*c1.y - e1.y*c1.x;
			e2c2.x = e2.y*c2.z - e2.z*c2.y; e2c2.y = e2.x*c2.z - e2.z*c2.x; e2c2.z = e2.x*c2.y - e2.y*c2.x;
			
			float case1, case2, case3;
			case1 = nn.x*e0c0.x + nn.y*e0c0.y + nn.z*e0c0.z;
			case2 = nn.x*e1c1.x + nn.y*e1c1.y + nn.z*e1c1.z;
			case3 = nn.x*e2c2.x + nn.y*e2c2.y + nn.z*e2c2.z;
			if (case1 > 0 && case2 > 0 && case3 > 0 && t > 0)
			{
				count++;
			}
		}
		
		if (count % 2 == 0)
		{
			f1 = 0;
			bbox_flag[i] = (f1);
		}
		else
		{
			f1 = 1;
            bbox_flag[i] = (f1);
        }
        if(bbox_flag[i]==1)
        {
            // printf("EleIdx: %d, BlockIdx: %d, ThreadIdx: %d, %d, Done!\n", i ,blockIdx.x, threadIdx.x, bbox_flag[i]);
        }
	}
}

int main()
{
	char filename[] = "../io/bunny_tri.raw";
    int nvert, nface;
    vector<vertex3D> vertex_host, origin_host;
    vector<tri> face_host;
	int ndivx, ndivy, ndivz;
	ndivx = 20;
	ndivy = 20;
    ndivz = 20;
    
    clock_t begin, end;
    double elapsed_secs;

	printf("----------Starting CPU Code----------\n");
	printf("Reading Geometry\n");
    Meshio Mesh_host;
    Mesh_host.read_raw_file(filename, vertex_host, face_host);
    Mesh_host.set_bounding_box(ndivx, ndivy, ndivz);
    Mesh_host.calculate_grid(origin_host);
    printf("minimum x, y, z = %f, %f, %f\n", Mesh_host.x_min, Mesh_host.y_min, Mesh_host.z_min);
	printf("maximum x, y, z = %f, %f, %f\n", Mesh_host.x_max, Mesh_host.y_max, Mesh_host.z_max);

    begin = clock();
    Mesh_host.calculate_normal();    
    Mesh_host.point_membership();
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    printf("CPU computing: %f\n", elapsed_secs);

    printf("----------Starting GPU Code----------\n");

    cudaError_t cudaStatus = cudaSuccess;
    size_t size_class = int(sizeof(Meshio));
    int ngrid = ndivx * ndivy * ndivz;
    nvert = vertex_host.size();
    nface = face_host.size();
    int thread_per_block = 10;

	printf("Sending Mesh to GPU\n");
    Meshio *Mesh_dev = NULL;
    vertex3D *vertex_dev = NULL;
    vertex3D *origin_dev = NULL;
    tri *face_dev = NULL;
    cudaStatus = cudaMalloc((void**)&Mesh_dev,size_class);
    cudaStatus = cudaMalloc((void**)&vertex_dev,nvert * sizeof(vertex3D));
    cudaStatus = cudaMalloc((void**)&origin_dev,ngrid * sizeof(vertex3D));
    cudaStatus = cudaMalloc((void**)&face_dev,nface* sizeof(tri));

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device mesh (error code %s)!\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
	}
	
	cudaStatus = cudaMemcpy(Mesh_dev, &Mesh_host, size_class, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(vertex_dev, &vertex_host[0], nvert * sizeof(vertex3D), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(origin_dev, &origin_host[0], ngrid * sizeof(vertex3D), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(face_dev, &face_host[0], nface* sizeof(tri), cudaMemcpyHostToDevice);
    
	if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy host mesh to device mesh (error code %s)!\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
	}



    begin = clock();
    // std::cout << typeid(*Mesh_dev).name() << '\n';
    // Calculate normals of the surface mesh
    printf("Computing normal vector for each element on GPU\n");
    calculate_normal<<<(nface + thread_per_block - 1)/thread_per_block, thread_per_block>>>(vertex_dev, face_dev, nface);
    cudaDeviceSynchronize();

    int *bbox_flag_dev = NULL;
    cudaStatus = cudaMalloc((void**)&bbox_flag_dev, ngrid * sizeof(int));
    cudaDeviceSynchronize();

    printf("Ray tracing on GPU\n");
    point_membership<<<(ngrid + thread_per_block - 1)/thread_per_block, thread_per_block>>>(Mesh_dev, origin_dev, vertex_dev, face_dev, bbox_flag_dev);
    cudaDeviceSynchronize();

    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    printf("GPU computing: %f\n", elapsed_secs);


    int bbox_flag_host[ngrid]; /// Use this variable to output the result
    cudaMemcpy((void *)bbox_flag_host, (void *)bbox_flag_dev, ngrid * sizeof(int), cudaMemcpyDeviceToHost);

    // for(int ii = 0; ii<ngrid;ii++)
    // {
    //     if(bbox_flag_host[ii]==1)
    //         printf("GridIndex:%d, %d\n",ii,bbox_flag_host[ii]);
    // }

    cudaFree(Mesh_dev);
    cudaFree(vertex_dev);
    cudaFree(origin_dev);
    cudaFree(face_dev);
    // printf("Setting bounding box on GPU\n");
    // cudaDeviceSynchronize();
	// Mesh_dev->set_bounding_box(ndivx, ndivy, ndivz);
	
	// printf("minimum x, y, z = %f, %f, %f\n", Mesh_dev->x_min, Mesh_dev->y_min, Mesh_dev->z_min);
	// printf("maximum x, y, z = %f, %f, %f\n", Mesh_dev->x_max, Mesh_dev->y_max, Mesh_dev->z_max);


    // cudaDeviceSynchronize();
	// Mesh_dev.calculate_normal();

    // //Ray-tracing
    // printf("Ray tracing on GPU\n");
    // cudaDeviceSynchronize();
    // Mesh_dev.point_membership();
    
	// getchar();
    
    return 0;
}