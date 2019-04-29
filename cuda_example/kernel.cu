
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>
#include "BasicDataStructure.h"
#include "mesh.h"

using namespace std;

__global__ void PrintMesh(TriElement* mesh_dev, int nele)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
    if (i < nele)
    {
		printf("EleIdx: %d, BlockIdx: %d, ThreadIdx: %d, %d %d %d\n", i ,blockIdx.x, threadIdx.x ,mesh_dev[i].cnct[0],mesh_dev[i].cnct[1],mesh_dev[i].cnct[2]);
    }
}

__global__ void ComputeTriArea_dev(TriElement* mesh_dev, int nele)
{
	float l[3], p;

	int e = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i < 3; i++)
	{
		l[i] = sqrt(pow(mesh_dev[e].coor[(i + 1) % 3][0] - mesh_dev[e].coor[(i) % 3][0], 2)
			+ pow(mesh_dev[e].coor[(i + 1) % 3][1] - mesh_dev[e].coor[(i) % 3][1], 2)
			+ pow(mesh_dev[e].coor[(i + 1) % 3][2] - mesh_dev[e].coor[(i) % 3][2], 2));
	}
	p = 0.5*(l[0] + l[1] + l[2]);
	mesh_dev[e].area = sqrt(p*(p - l[0])*(p - l[1])*(p - l[2]));
	printf("EleIdx: %d, BlockIdx: %d, ThreadIdx: %d, Area: %f\n", e, blockIdx.x, threadIdx.x, mesh_dev[e].area);

}


int main(void)
{
	vector<TriElement> mesh_host;
	vector<vertex3D> pts_host;
	vector<tri> tri_host;

	string fn("../io/square_tri.vtk");

	/* ----------Host Code----------*/
	// Read Mesh into CPU
	printf("----------Starting CPU Code----------\n");
	printf("Reading Geometry\n");
	ReadVtk_Tri(fn,  pts_host,  tri_host, mesh_host);
	//ComputeTriArea_host(mesh_host);

	

	/* ----------Device Code----------*/

	// Send mesh to GPU
	printf("----------Starting GPU Code----------\n");
	printf("Sending Mesh to GPU\n");
	int nele = mesh_host.size();

	cudaError_t cudaStatus = cudaSuccess;
	size_t size_class = nele * int(sizeof(TriElement));
	int thread_per_block = 4;


	TriElement *mesh_dev = NULL;
	cudaStatus = cudaMalloc((void**)&mesh_dev,size_class);

	if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device mesh (error code %s)!\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
	}
	
	cudaStatus = cudaMemcpy(mesh_dev, &mesh_host[0], sizeof(TriElement)*nele, cudaMemcpyHostToDevice);
	
	if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy host mesh to device mesh (error code %s)!\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
	}

	// Compute the Area for each element in GPU
	printf("Computing in GPU\n");
	cudaDeviceSynchronize();
	PrintMesh<<<(nele + thread_per_block - 1)/thread_per_block, thread_per_block>>>(mesh_dev, nele);
	ComputeTriArea_dev<<<(nele + thread_per_block - 1)/thread_per_block, thread_per_block>>>(mesh_dev, nele);
	
	cudaFree(mesh_dev);


	
	

	return 0;
}

// __global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
// {
//     int i = blockDim.x * blockIdx.x + threadIdx.x;

//     if (i < numElements)
//     {
//         C[i] = A[i] + B[i];
//     }
// }

// int main(void)
// {
//     // Error code to check return values for CUDA calls
//     cudaError_t err = cudaSuccess;

//     // Print the vector length to be used, and compute its size
//     int numElements = 50000;
//     size_t size = numElements * sizeof(float);
//     printf("[Vector addition of %d elements]\n", numElements);

//     // Allocate the host input vector A
//     float *h_A = (float *)malloc(size);

//     // Allocate the host input vector B
//     float *h_B = (float *)malloc(size);

//     // Allocate the host output vector C
//     float *h_C = (float *)malloc(size);

//     // Verify that allocations succeeded
//     if (h_A == NULL || h_B == NULL || h_C == NULL)
//     {
//         fprintf(stderr, "Failed to allocate host vectors!\n");
//         exit(EXIT_FAILURE);
//     }

//     // Initialize the host input vectors
//     for (int i = 0; i < numElements; ++i)
//     {
//         h_A[i] = rand()/(float)RAND_MAX;
//         h_B[i] = rand()/(float)RAND_MAX;
//     }

//     // Allocate the device input vector A
//     float *d_A = NULL;
//     err = cudaMalloc((void **)&d_A, size);

//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     // Allocate the device input vector B
//     float *d_B = NULL;
//     err = cudaMalloc((void **)&d_B, size);

//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     // Allocate the device output vector C
//     float *d_C = NULL;
//     err = cudaMalloc((void **)&d_C, size);

//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     // Copy the host input vectors A and B in host memory to the device input vectors in
//     // device memory
//     printf("Copy input data from the host memory to the CUDA device\n");
//     err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     // Launch the Vector Add CUDA Kernel
//     int threadsPerBlock = 256;
//     int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
//     printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
//     vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
//     err = cudaGetLastError();

//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     // Copy the device result vector in device memory to the host result vector
//     // in host memory.
//     printf("Copy output data from the CUDA device to the host memory\n");
//     err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     // Verify that the result vector is correct
//     for (int i = 0; i < numElements; ++i)
//     {
//         if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
//         {
//             fprintf(stderr, "Result verification failed at element %d!\n", i);
//             exit(EXIT_FAILURE);
//         }
//     }

//     printf("Test PASSED\n");

//     // Free device global memory
//     err = cudaFree(d_A);

//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     err = cudaFree(d_B);

//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     err = cudaFree(d_C);

//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     // Free host memory
//     free(h_A);
//     free(h_B);
//     free(h_C);

//     printf("Done\n");
//     return 0;
// }
