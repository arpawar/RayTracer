#include <stdio.h>
#include <iostream> 
#include <vector>
#include <math.h>
#include "basic_datastructure.h"
#include "read_raw.h"
using namespace std;

void main()
{
	char filename[] = "bunny_tri.raw";
	int nvert, nface;
	int ndivx, ndivy, ndivz;
	ndivx = 100;
	ndivy = 100;
	ndivz = 100;

	Meshio Mesh;

	Mesh.read_raw_file(filename);
	Mesh.set_bounding_box(ndivx, ndivy, ndivz);
	
	printf("minimum x, y, z = %f, %f, %f\n", Mesh.x_min, Mesh.y_min, Mesh.z_min);
	printf("maximum x, y, z = %f, %f, %f\n", Mesh.x_max, Mesh.y_max, Mesh.z_max);

	// Calculate normals of the surface mesh
	Mesh.calculate_normal();

	//Ray-tracing
	Mesh.point_membership();
	getchar();
}