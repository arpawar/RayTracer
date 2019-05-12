/* This code performs point membership of 3D structured grid points that
embed the geometry. For the 3D structured grid we perform test to check which
grid points lie inside/outside the geometry. For each point in the grid a ray
starts from the grid point and in the x-direction to check for intersection
with the geometry. If the points are inside, the number of intersections are odd,
else the number of intersections are even.
*/


#include "RayTracer.h"
#include "basic_datastructure.h"
#pragma warning(disable:4996)

Meshio::Meshio()
{
	// if the mesh object is created successfully set flag equal to 1
	flag = 1;
}

// this function reads the mesh file
void Meshio::read_raw_file(const char * fn)
{
	// open the mesh file
	FILE *fp = fopen(fn, "r");

	if (fp == NULL)
	{
	printf("Error reading the file\n");
	getchar();
	}
	
	// read the number of vertices and faces in mesh: nvert and nface
	fscanf(fp, "%d %d", &nvert, &nface);


	// read the vertex coordinates of the mesh in vertex object
	vertex3D v1;
	for (int i = 0; i < nvert; i++)
	{
	fscanf(fp, "%f %f %f", &v1.x, &v1.y, &v1.z);
	vertex.push_back(v1);
	}

	// store the vertex indices for each face in face object
	tri f1;
	for (int i = 0; i < nface; i++)
	{
	fscanf(fp, "%d %d %d", &f1.v[0], &f1.v[1], &f1.v[2]);
	face.push_back(f1);
	}
}

// this function constructs the bounding box to embed the geometry
void Meshio::set_bounding_box(int nx, int ny, int nz)
{
	float xmin, xmax, ymin, ymax, zmin, zmax;
	xmin = vertex[0].x;
	xmax = vertex[0].x;
	ymin = vertex[0].y;
	ymax = vertex[0].y;
	zmin = vertex[0].z;
	zmax = vertex[0].z;

	for (int i = 0; i < nvert; i++)
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
	x_max = floor(xmax+10);
	y_max = floor(ymax+10);
	z_max = floor(zmax+10);

	ndivx = nx;
	ndivy = ny;
	ndivz = nz;


}

void Meshio::calculate_normal()
{
	//char filename[] = "Sphere_100_tri_normal.txt";
	//FILE *fp = fopen(filename, "w");
	for (int i = 0; i < nface; i++)
	{
		int v0_ind = face[i].v[0];
		int v1_ind = face[i].v[1];
		int v2_ind = face[i].v[2];

		vertex3D e1, e2;
		vertex3D nn;

		e1.x = vertex[v1_ind].x - vertex[v0_ind].x;
		e1.y = vertex[v1_ind].y - vertex[v0_ind].y;
		e1.z = vertex[v1_ind].z - vertex[v0_ind].z;

		e2.x = vertex[v2_ind].x - vertex[v0_ind].x;
		e2.y = vertex[v2_ind].y - vertex[v0_ind].y;
		e2.z = vertex[v2_ind].z - vertex[v0_ind].z;

		nn.x = e1.y*e2.z - e1.z*e2.y;
		nn.y = e1.z*e2.x - e1.x*e2.z;
		nn.z = e1.x*e2.y - e1.y*e2.x;

		float n_norm = sqrt(pow(nn.x, 2) + pow(nn.y, 2) + pow(nn.z, 2));
		face[i].n.x = nn.x/n_norm;
		face[i].n.y = nn.y/n_norm;
		face[i].n.z = nn.z/n_norm;
		//fprintf(fp, "%d %f %f %f\n", i + 1, face[i].n.x, face[i].n.y, face[i].n.z);
	}
}

void Meshio::point_membership()
{
	//char filename[] = "Sphere_100_tri_o.txt";
	//FILE *fp = fopen(filename, "w");
	int ngrid = 0;
	int f1;
	for (int k = 0; k < ndivz; k++)
	{
		for (int j = 0; j < ndivy; j++)
		{
			for (int i = 0; i < ndivx; i++)
			{
				vertex3D origin, ray;
				origin.x = x_min + i*((x_max - x_min) / (ndivx - 1));
				origin.y = y_min + j*((y_max - y_min) / (ndivy - 1));
				origin.z = z_min + k*((z_max - z_min) / (ndivz - 1));
				//fprintf(fp,"%d, %f, %f, %f\n", ngrid+1, origin.x, origin.y, origin.z);

				ray.x = 1;
				ray.y = 0;
				ray.z = 0;

				int count = 0;

				for (int iface = 0; iface < nface; iface++)
				{
					int v0_ind = face[iface].v[0];
					int v1_ind = face[iface].v[1];
					int v2_ind = face[iface].v[2];

					vertex3D nn = face[iface].n;

					float D = -(nn.x*vertex[v0_ind].x + nn.y*vertex[v0_ind].y + nn.z*vertex[v0_ind].z);
					float t1 = -(nn.x*origin.x + nn.y*origin.y + nn.z*origin.z + D);
					float t2 = (nn.x*ray.x + nn.y*ray.y + nn.z*ray.z);
					float t = t1 / t2;

					vertex3D p_vec;
					p_vec.x = origin.x + t*ray.x;
					p_vec.y = origin.y + t*ray.y;
					p_vec.z = origin.z + t*ray.z;

					vertex3D e0, e1, e2, c0, c1, c2;
					e0.x = vertex[v1_ind].x - vertex[v0_ind].x;  e0.y = vertex[v1_ind].y - vertex[v0_ind].y;  e0.z = vertex[v1_ind].z - vertex[v0_ind].z;
					e1.x = vertex[v2_ind].x - vertex[v1_ind].x;  e1.y = vertex[v2_ind].y - vertex[v1_ind].y;  e1.z = vertex[v2_ind].z - vertex[v1_ind].z;
					e2.x = vertex[v0_ind].x - vertex[v2_ind].x;  e2.y = vertex[v0_ind].y - vertex[v2_ind].y;  e2.z = vertex[v0_ind].z - vertex[v2_ind].z;

					c0.x = p_vec.x - vertex[v0_ind].x;  c0.y = p_vec.y - vertex[v0_ind].y;  c0.z = p_vec.z - vertex[v0_ind].z;
					c1.x = p_vec.x - vertex[v1_ind].x;  c1.y = p_vec.y - vertex[v1_ind].y;  c1.z = p_vec.z - vertex[v1_ind].z;
					c2.x = p_vec.x - vertex[v2_ind].x;  c2.y = p_vec.y - vertex[v2_ind].y;  c2.z = p_vec.z - vertex[v2_ind].z;

					vertex3D e0c0, e1c1, e2c2;
					e0c0.x = e0.y*c0.z - e0.z*c0.y; e0c0.y = e0.z*c0.x - e0.x*c0.z; e0c0.z = e0.x*c0.y - e0.y*c0.x;
					e1c1.x = e1.y*c1.z - e1.z*c1.y; e1c1.y = e1.z*c1.x - e1.x*c1.z; e1c1.z = e1.x*c1.y - e1.y*c1.x;
					e2c2.x = e2.y*c2.z - e2.z*c2.y; e2c2.y = e2.z*c2.x - e2.x*c2.z; e2c2.z = e2.x*c2.y - e2.y*c2.x;
					
					float case1, case2, case3;
					case1 = nn.x*e0c0.x + nn.y*e0c0.y + nn.z*e0c0.z;
					case2 = nn.x*e1c1.x + nn.y*e1c1.y + nn.z*e1c1.z;
					case3 = nn.x*e2c2.x + nn.y*e2c2.y + nn.z*e2c2.z;

					if (case1 > 0 && case2 > 0 && case3 > 0 && t > 0)
					{
						count++;
					}
				}
				
				if (count % 2 == 0)
				{
					f1 = 0;
					bbox_flag.push_back(f1);
				}
				else
				{
					f1 = 1;
					bbox_flag.push_back(f1);
				}

				ngrid = ngrid++;
			}
		}
	}
}

void Meshio::display_result()
{
	char filename[] = "bunny_tri_result.txt";
	FILE *fp = fopen(filename, "w");
	float x, y, z;
	int ngrid;
	ngrid = 0;
	for (int k = 0; k < ndivz; k++)
	{
		for (int j = 0; j < ndivy; j++)
		{
			for (int i = 0; i < ndivx; i++)
			{
				
				x = x_min + i*((x_max - x_min) / (ndivx - 1));
				y = y_min + j*((y_max - y_min) / (ndivy - 1));
				z = z_min + k*((z_max - z_min) / (ndivz - 1));
				fprintf(fp, "%d %f %f %f %d\n", ngrid+1, x, y, z, bbox_flag[ngrid]);
				ngrid = ngrid + 1;
			}
		}		
	}
}