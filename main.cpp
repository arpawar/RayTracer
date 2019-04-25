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
	Meshio Mesh;

	Mesh.read_raw_file(filename);
	Mesh.set_bounding_box();
	
	printf("minimum x, y, z = %f, %f, %f\n", Mesh.x_min, Mesh.y_min, Mesh.z_min);
	printf("maximum x, y, z = %f, %f, %f\n", Mesh.x_max, Mesh.y_max, Mesh.z_max);


	getchar();
}