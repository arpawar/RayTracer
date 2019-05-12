/* This code performs point membership of 3D structured grid points that
embed the geometry. For the 3D structured grid we perform test to check which
grid points lie inside/outside the geometry. For each point in the grid a ray
starts from the grid point and in the x-direction to check for intersection
with the geometry. If the points are inside, the number of intersections are odd,
else the number of intersections are even.

This cpp file includes all the functions associated with ray tracing and point membership classification. 
The definition of the function are provide in RayTracer.h file in the Mesh class.
*/

#include "RayTracer.h"
#include "basic_datastructure.h"
#pragma warning(disable : 4996)

Meshio::Meshio()
{
	// if the mesh object is created successfully set flag equal to 1
	flag = 1;
}

// this function reads the mesh file
void Meshio::read_raw_file(const char *fn, vector<vertex3D> &vertex_out, vector<tri> &face_out)
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

	vertex3D v1;
	// read the vertex coordinates of the mesh in vertex object
	for (int i = 0; i < nvert; i++)
	{
		fscanf(fp, "%f %f %f", &v1.x, &v1.y, &v1.z);
		vertex.push_back(v1);
	}

	tri f1;
	// store the vertex indices for each face in face object
	for (int i = 0; i < nface; i++)
	{
		fscanf(fp, "%d %d %d", &f1.v[0], &f1.v[1], &f1.v[2]);
		face.push_back(f1);
	}
	vertex_out = vertex;
	face_out = face;
	//close mesh file
	fclose(fp);
}

// this function constructs the bounding box to embed the geometry
void Meshio::set_bounding_box(int nx, int ny, int nz, float pm)
{
	float xmin, xmax, ymin, ymax, zmin, zmax;
	xmin = vertex[0].x; //initialize minimum x of grid to be the first vertex, x component
	xmax = vertex[0].x; //initialize maximum x of grid to be the first vertex, x component
	ymin = vertex[0].y; //initialize minimum y of grid to be the first vertex, y component
	ymax = vertex[0].y; //initialize maximum y of grid to be the first vertex, y component
	zmin = vertex[0].z; //initialize minimum z of grid to be the first vertex, z component
	zmax = vertex[0].z; //initialize maximum z of grid to be the first vertex, z component

	//evaluate the minimum and maximum grid values
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

	//set the minimum and maximum grid values and add some percentage margin for the bounding box
	x_min = floor(xmin - pm*(xmax-xmin)); 
	y_min = floor(ymin - pm*(ymax-ymin));
	z_min = floor(zmin - pm*(zmax-zmin));
	x_max = ceil(xmax + pm*(xmax-xmin));
	y_max = ceil(ymax + pm*(ymax-ymin));
	z_max = ceil(zmax + pm*(zmax-zmin));

	//store the number of grid points in x, y, z direction
	ndivx = nx;
	ndivy = ny;
	ndivz = nz;
	
	//store total number of grid points
	ngrid = nx * ny * nz;
	
	bbox_flag.resize(ngrid);
}

// this function computes the coordinates of the grid points
void Meshio::calculate_grid(vector<vertex3D> &origin_out)
{
	origin_out.clear();
	origin_out.resize(ndivz * ndivy * ndivx);
#pragma omp parallel for collapse(3)
	for (int k = 0; k < ndivz; k++)
	{
		for (int j = 0; j < ndivy; j++)
		{
			for (int i = 0; i < ndivx; i++)
			{
				int e = k * ndivy * ndivx + j * ndivx + i;
				origin_out[e].x = x_min + (x_max - x_min) / (ndivx - 1) * i;
				origin_out[e].y = y_min + (y_max - y_min) / (ndivy - 1) * j;
				origin_out[e].z = z_min + (z_max - z_min) / (ndivz - 1) * k;
			}
		}
	}
}

//this function calculates the normals of the surface mesh
void Meshio::calculate_normal()
{
//loop over the faces of the mesh
#pragma omp parallel for
	for (int i = 0; i < nface; i++)
	{
		// indices of the vertices of the face
		int v0_ind = face[i].v[0];
		int v1_ind = face[i].v[1];
		int v2_ind = face[i].v[2];
		
		//edge vectors initialized
		vertex3D e1, e2;
		//normal vector initialized
		vertex3D nn;

		//set the first edge vector e1 = v1 - v0
		e1.x = vertex[v1_ind].x - vertex[v0_ind].x;
		e1.y = vertex[v1_ind].y - vertex[v0_ind].y;
		e1.z = vertex[v1_ind].z - vertex[v0_ind].z;

		//set the second edge vector e2 = v2 - v0
		e2.x = vertex[v2_ind].x - vertex[v0_ind].x;
		e2.y = vertex[v2_ind].y - vertex[v0_ind].y;
		e2.z = vertex[v2_ind].z - vertex[v0_ind].z;

		//compute normal vector as cross product of e1 and e2: n = cross(e1,e2)
		nn.x = e1.y*e2.z - e1.z*e2.y;
		nn.y = e1.z*e2.x - e1.x*e2.z;
		nn.z = e1.x*e2.y - e1.y*e2.x;

		//find the magnitude of the normal vector
		float n_norm = sqrt(pow(nn.x, 2) + pow(nn.y, 2) + pow(nn.z, 2));
		
		//normalize the normal vector and store it in the face object
		face[i].n.x = nn.x/n_norm;
		face[i].n.y = nn.y/n_norm;
		face[i].n.z = nn.z/n_norm;
	}
}

// this function performs ray tracing and assign point membership to each grid point
void Meshio::point_membership()
{
	//initialize grid point counter to zero
	int ngrid = 0;
	//initialize flag (inside/outside) variable
	int f1;
	//set ray direction along x-axis i.e. (1,0,0)
	vertex3D ray;
	ray.x = 1;
	ray.y = 0;
	ray.z = 0;
	
	//loop over grid points
#pragma omp parallel for collapse(3)
	for (int k = 0; k < ndivz; k++)
	{
		for (int j = 0; j < ndivy; j++)
		{
			for (int i = 0; i < ndivx; i++)
			{
				//compute the coordinates of grid point
				vertex3D origin;
				origin.x = x_min + (x_max - x_min) / (ndivx - 1) * i;
				origin.y = y_min + (y_max - y_min) / (ndivy - 1) * j;
				origin.z = z_min + (z_max - z_min) / (ndivz - 1) * k;

				//initialize no. of intersections of the ray with faces to 0
				int count = 0;
				
				//loop over faces
				for (int iface = 0; iface < nface; iface++)
				{
					//indices of vertices of the face
					int v0_ind = face[iface].v[0];
					int v1_ind = face[iface].v[1];
					int v2_ind = face[iface].v[2];

					//normal vector of the face
					vertex3D nn = face[iface].n;
					
					//refer to https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
					//for more details on the formulas and variables
					//the ray vector is defines as p_vec = origin + t*ray
					//here calculating the t parameter at which the ray intersects the face
					float D = -(nn.x * vertex[v0_ind].x + nn.y * vertex[v0_ind].y + nn.z * vertex[v0_ind].z);
					float t1 = -(nn.x * origin.x + nn.y * origin.y + nn.z * origin.z + D);
					float t2 = (nn.x * ray.x + nn.y * ray.y + nn.z * ray.z);
					float t = t1 / t2;

					//computing p_vec
					vertex3D p_vec;
					p_vec.x = origin.x + t * ray.x;
					p_vec.y = origin.y + t * ray.y;
					p_vec.z = origin.z + t * ray.z;

					//calculating edge vectors e0 = v1-v0, e1 = v2-v1 and e2 = v0-v2
					vertex3D e0, e1, e2, c0, c1, c2;
					e0.x = vertex[v1_ind].x - vertex[v0_ind].x;
					e0.y = vertex[v1_ind].y - vertex[v0_ind].y;
					e0.z = vertex[v1_ind].z - vertex[v0_ind].z;
					e1.x = vertex[v2_ind].x - vertex[v1_ind].x;
					e1.y = vertex[v2_ind].y - vertex[v1_ind].y;
					e1.z = vertex[v2_ind].z - vertex[v1_ind].z;
					e2.x = vertex[v0_ind].x - vertex[v2_ind].x;
					e2.y = vertex[v0_ind].y - vertex[v2_ind].y;
					e2.z = vertex[v0_ind].z - vertex[v2_ind].z;

					// calculating c0 = p_vec-v0, c1 = p_vec-v1, c2 = p_vec-v2
					c0.x = p_vec.x - vertex[v0_ind].x;
					c0.y = p_vec.y - vertex[v0_ind].y;
					c0.z = p_vec.z - vertex[v0_ind].z;
					c1.x = p_vec.x - vertex[v1_ind].x;
					c1.y = p_vec.y - vertex[v1_ind].y;
					c1.z = p_vec.z - vertex[v1_ind].z;
					c2.x = p_vec.x - vertex[v2_ind].x;
					c2.y = p_vec.y - vertex[v2_ind].y;
					c2.z = p_vec.z - vertex[v2_ind].z;

					//calculating e0c0 = cross(e0,c0), e1c1 = cross(e1,c1), e2c2 = cross(e2,c2)
					vertex3D e0c0, e1c1, e2c2;
					e0c0.x = e0.y*c0.z - e0.z*c0.y; e0c0.y = e0.z*c0.x - e0.x*c0.z; e0c0.z = e0.x*c0.y - e0.y*c0.x;
					e1c1.x = e1.y*c1.z - e1.z*c1.y; e1c1.y = e1.z*c1.x - e1.x*c1.z; e1c1.z = e1.x*c1.y - e1.y*c1.x;
					e2c2.x = e2.y*c2.z - e2.z*c2.y; e2c2.y = e2.z*c2.x - e2.x*c2.z; e2c2.z = e2.x*c2.y - e2.y*c2.x;

					//calculating case1 = dot(n,e0c0), case2 = dot(n,e1c1), case3 = dot(n,e2c2)
					float case1, case2, case3;
					case1 = nn.x * e0c0.x + nn.y * e0c0.y + nn.z * e0c0.z;
					case2 = nn.x * e1c1.x + nn.y * e1c1.y + nn.z * e1c1.z;
					case3 = nn.x * e2c2.x + nn.y * e2c2.y + nn.z * e2c2.z;

					//increment counter if p_vec is inside the triangle face
					if (case1 > 0 && case2 > 0 && case3 > 0 && t > 0)
					{
						count++;
					}
				}
				
				//if count is even-> grid point is outside the geometry
				int e = k * ndivy * ndivx + j * ndivx + i;
				if (count % 2 == 0)
				{
					//set flag as 0
					f1 = 0;
					bbox_flag[e] = (f1);
				}
				
				//grid point in inside the geometry
				else
				{
					//set flag as 1
					f1 = 1;
					bbox_flag[e] = (f1);
				}
			}
		}
	}
}

//this function writes the output to a vtk file for visualization 
void Meshio::display_result(char* filename_out, int* bbox_flag_host)
{
	//open output file
	FILE *fp = fopen(filename_out, "w");
	
	//vtk structure grid header information
	fprintf(fp, "# vtk DataFile Version 3.0\n");
	fprintf(fp, "VTK from C\n");
	fprintf(fp, "ASCII\n");
	fprintf(fp, "DATASET STRUCTURED_GRID\n");
	
	fprintf(fp, "DIMENSIONS %d %d %d\n", ndivx, ndivy, ndivz);
	int nelem = ndivx*ndivy*ndivz;
	
	//output the coordinates of the grid vertices
	fprintf(fp, "POINTS %d float\n", nelem);
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
				fprintf(fp, "%f %f %f\n", x, y, z);
				ngrid = ngrid + 1;
			}
		}		
	}
	
	//output the inside/outside flag information for the grid points
	fprintf(fp, "POINT_DATA %d\n", nelem);
	fprintf(fp, "SCALARS title float\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	ngrid = 0;
	for (int k = 0; k < ndivz; k++)
	{
		for (int j = 0; j < ndivy; j++)
		{
			for (int i = 0; i < ndivx; i++)
			{
				fprintf(fp, "%d\n", bbox_flag_host[ngrid]);
				ngrid = ngrid + 1;
			}
		}
	}	

	//close output file
	fclose(fp);
}
