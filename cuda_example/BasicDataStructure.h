#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 


#ifndef BASIC_DATA_STRUCTURE_H
#define BASIC_DATA_STRUCTURE_H
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <iostream>

#include "cuda_runtime.h"

using namespace std;

#define DIM 3
#define ELE_NODE 3

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

	CUDA_CALLABLE_MEMBER tri()
	{
		v[0] = 0; v[1] = 0; v[2] = 0;
	};

	CUDA_CALLABLE_MEMBER ~tri(){};
};

#endif

class TriElement
{
public:
	int cnct[3];
	//vector<vector<int>> edge;
	int edge[3];

	float coor[3][3];

	float area;

	CUDA_CALLABLE_MEMBER TriElement();

	CUDA_CALLABLE_MEMBER ~TriElement();

};

CUDA_CALLABLE_MEMBER TriElement::TriElement()
{
	for (int i = 0; i < 3; i++)
	{
		cnct[i] = 0;
		edge[i] = 0;
		for (int j = 0; j < 3; j++)
			coor[i][j] = 0.;
	}
	area = 0.;
}

CUDA_CALLABLE_MEMBER TriElement::~TriElement(){}

#endif