#ifndef MESH_H
#define MESH_H

#include "BasicDataStructure.h"
#include <iostream>
#include <array>
#include <fstream>
#include <string>
#include <utility>
#include <iomanip>
#include <sstream>
#include <cmath>

using namespace std;

typedef unsigned int uint;

/// Tri Mesh
void ReadVtk_Tri(string fn, vector<vertex3D>& pts, vector<tri>& cnct, vector<TriElement>& mesh)
{
	string stmp;
	int npts, neles, itmp;
	ifstream fin;
	fin.open(fn);
	if (fin.is_open())
	{
		for (int i = 0; i < 4; i++)
			getline(fin, stmp); //skip lines
		fin >> stmp >> npts >> stmp;
		pts.resize(npts);
		for (int i = 0; i < npts; i++)
		{
			fin >> pts[i].x >> pts[i].y >> pts[i].z;
		}
		getline(fin, stmp);
		fin >> stmp >> neles >> itmp;
		cnct.resize(neles);
		for (int i = 0; i < neles; i++)
		{
			fin >> itmp >> cnct[i].v[0] >> cnct[i].v[1] >> cnct[i].v[2];
		}
		fin.close();
	}
	else
	{
		cout << "Cannot open " << fn << "!\n";
	}
	mesh.resize(neles);
	for (int e = 0; e < neles; e++)
	{
		for (int i = 0; i < 3; i++)
		{
			mesh[e].cnct[i] = cnct[e].v[i];			
			mesh[e].coor[i][0] = pts[cnct[e].v[i]].x;
			mesh[e].coor[i][1] = pts[cnct[e].v[i]].y;
			mesh[e].coor[i][2] = pts[cnct[e].v[i]].z;
		}
	}
}

void ComputeTriArea_host(vector<TriElement>& mesh)
{
	int nele = mesh.size();
	double l[3], p;
	for (int e = 0; e < nele; e++)
	{
		for (int i = 0; i < 3; i++)
		{
			l[i] = sqrt(pow(mesh[e].coor[(i + 1) % 3][0] - mesh[e].coor[(i) % 3][0], 2)
				+ pow(mesh[e].coor[(i + 1) % 3][1] - mesh[e].coor[(i) % 3][1], 2)
				+ pow(mesh[e].coor[(i + 1) % 3][2] - mesh[e].coor[(i) % 3][2], 2));
		}
		p = 0.5*(l[0] + l[1] + l[2]);
		mesh[e].area = sqrt(p*(p - l[0])*(p - l[1])*(p - l[2]));
		printf("%f\n", mesh[e].area);
	}
}

// __global__ ComputeTriArea_dev(vector<TriElement>& mesh)
// {
	
// 	cudaco



// 	CudaClass c(1);
//     // create class storage on device and copy top level class
//     CudaClass *d_c;
//     cudaMalloc((void **)&d_c, sizeof(CudaClass));
//     cudaMemcpy(d_c, &c, sizeof(CudaClass), cudaMemcpyHostToDevice);
//     // make an allocated region on device for use by pointer in class
//     int *hostdata;
//     cudaMalloc((void **)&hostdata, sizeof(int));
//     cudaMemcpy(hostdata, c.data, sizeof(int), cudaMemcpyHostToDevice);
//     // copy pointer to allocated device storage to device class
//     cudaMemcpy(&(d_c->data), &hostdata, sizeof(int *), cudaMemcpyHostToDevice);
//     useClass<<<1,1>>>(d_c);
//     cudaDeviceSynchronize();
//     return 0;

// }

#endif
