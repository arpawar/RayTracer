#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 
#ifndef BASIC_DATASTRUCTURE_H
#define BASIC_DATASTRUCTURE_H

#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <iostream>


class vertex3D
{
public:
	float x, y, z;
	CUDA_CALLABLE_MEMBER vertex3D()
	{
		x = 0.; y = 0.; z = 0.;
	};

	CUDA_CALLABLE_MEMBER ~vertex3D(){};
};

class tri
{
public:
	int v[3];
	vertex3D n;
	
	CUDA_CALLABLE_MEMBER tri()
	{
		v[0] = 0; v[1] = 0; v[2] = 0;
	};

	CUDA_CALLABLE_MEMBER ~tri(){};
};



#endif