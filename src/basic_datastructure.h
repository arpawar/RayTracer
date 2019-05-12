/* This code performs point membership of 3D structured grid points that
embed the geometry. For the 3D structured grid we perform test to check which
grid points lie inside/outside the geometry. For each point in the grid a ray
starts from the grid point and in the x-direction to check for intersection
with the geometry. If the points are inside, the number of intersections are odd,
else the number of intersections are even.
This header function defines the structure of a vertex coordinate vertex3D:
i.e. x,y,z coordinates in the 3D space.
The struct variable tri defines the index of the triangular face vertices
and the normal vector n to the face.
*/

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

// Struct variable defining vertex in 3D space
// vertex3D variable consists of x,y and z component
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
