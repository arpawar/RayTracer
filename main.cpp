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
#include "basic_datastructure.h"
#include "RayTracer.h"
using namespace std;

void main()
{
	// Enter the filename of the surface mesh file
	char filename[] = "bunny_tri.raw";

	int ndivx, ndivy, ndivz;
	ndivx = 21;  //number of grid points in x directions
	ndivy = 21;  //number of grid points in y directions
	ndivz = 21;  //number of grid points in z directions


	// Create a mesh object
	Meshio Mesh;

	// Read the mesh file
	Mesh.read_raw_file(filename);
	printf("Mesh file read\n");

	// Construct the bounding box to embed the geometry
	Mesh.set_bounding_box(ndivx, ndivy, ndivz);
	printf("3D structured grid constructed \n");

	// Calculate normals of the surface mesh
	Mesh.calculate_normal();
	printf("Calculate normals to the surface mesh\n");

	//Perform ray tracing and assign point membership to each grid point
	Mesh.point_membership();
	printf("Point membership done\n");

	//Write the output to a vtk file for visualization
	Mesh.display_result();
	printf("written result\n");
}
