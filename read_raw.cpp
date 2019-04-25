#include "read_raw.h"
#include "basic_datastructure.h"
#pragma warning(disable:4996)

Meshio::Meshio()
{
	flag = 0;
}

void Meshio::read_raw_file(const char * fn)
{
	FILE *fp = fopen(fn, "r");
	if (fp == NULL)
	{
	printf("Error reading the file\n");
	getchar();
	}
	fscanf(fp, "%d %d", &nvert, &nface);

	vertex3D v1;

	for (int i = 0; i < nvert; i++)
	{
	fscanf(fp, "%f %f %f", &v1.x, &v1.y, &v1.z);
	vertex.push_back(v1);
	}

	tri f1;
	for (int i = 0; i < nface; i++)
	{
	fscanf(fp, "%d %d %d", &f1.v[0], &f1.v[1], &f1.v[2]);
	face.push_back(f1);
	}
}

void Meshio::set_bounding_box()
{
	float xmin, xmax, ymin, ymax, zmin, zmax;
	xmin = vertex[0].x;
	xmax = vertex[0].x;
	ymin = vertex[0].y;
	ymax = vertex[0].y;
	zmin = vertex[0].z;
	zmax = vertex[0].z;

	for (int i = 1; i < nvert; i++)
	{
		if (vertex[i].x <= xmin)
		{
			xmin = vertex[i].x;
		}

		if (vertex[i].x > xmax)
		{
			xmax = vertex[i].x;
		}

		if (vertex[i].y <= ymin)
		{
			ymin = vertex[i].y;
		}

		if (vertex[i].y > ymax)
		{
			ymax = vertex[i].y;
		}

		if (vertex[i].z <= zmin)
		{
			zmin = vertex[i].z;
		}

		if (vertex[i].z > zmax)
		{
			zmax = vertex[i].z;
		}
	}

	x_min = floor(xmin-10);
	y_min = floor(ymin-10);
	z_min = floor(zmin-10);
	x_max = floor(xmax-10);
	y_max = floor(ymax-10);
	z_max = floor(zmax-10);
}