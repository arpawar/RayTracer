/* This code performs point membership of 3D structured grid points that
embed the geometry. For the 3D structured grid we perform test to check which
grid points lie inside/outside the geometry. For each point in the grid a ray
starts from the grid point and in the x-direction to check for intersection
with the geometry. If the points are inside, the number of intersections are odd,
else the number of intersections are even.

This header file defines the variables and functions in the mesh object
*/

#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <vector>
#include <array>
#include "basic_datastructure.h"
using namespace std;

class Mesh
{
public:
	Meshio(); //constructor for the mesh object

	int flag; //  

	int nvert, nface; //nvert defines the number of vertices in the mesh and nface 				  defines the number of faces in the mesh

	float x_min, x_max, y_min, y_max, z_min, z_max;  // the minimum and maximum values of the 3D structured grid in x, y and z direction

	int ndivx, ndivy, ndivz; // number of grid points in x, y and z direction respectively

	vector<vertex3D> vertex; // vertex object

	vector<tri> face; // face object 

	vector<int> bbox_flag; // this variable denotes whether each point is inside or outside the geometry

	void read_raw_file(const char * fn); // this function reads the mesh file

	void set_bounding_box(int nx, int ny, int nz); // this function constructs the bounding box to embed the geometry

	void calculate_normal(); //this function calculates the normals of the surface mesh

	void point_membership(); // this function performs ray tracing and assign point membership to each grid point

	void display_result(); //this function writes the output to a vtk file for visualization 
};

#endif