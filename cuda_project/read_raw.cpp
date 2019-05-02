#include "read_raw.h"
#include "basic_datastructure.h"
#pragma warning(disable : 4996)

Meshio::Meshio()
{
	flag = 0;
}

void Meshio::read_raw_file(const char *fn, vector<vertex3D> &vertex_out, vector<tri> &face_out)
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
	vertex_out = vertex;
	face_out = face;
}

void Meshio::set_bounding_box(int nx, int ny, int nz)
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

	x_min = floor(xmin - 10);
	y_min = floor(ymin - 10);
	z_min = floor(zmin - 10);
	x_max = floor(xmax + 10);
	y_max = floor(ymax + 10);
	z_max = floor(zmax + 10);

	ndivx = nx;
	ndivy = ny;
	ndivz = nz;
	ngrid = nx * ny * nz;
	bbox_flag.resize(ngrid);
}

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
				origin_out[e].x = x_min + ((x_max - x_min) / ((ndivx - 1) * i));
				origin_out[e].y = y_min + ((y_max - y_min) / ((ndivy - 1) * j));
				origin_out[e].z = z_min + ((z_max - z_min) / ((ndivz - 1) * k));
			}
		}
	}
	// origin_out = origin;
}

void Meshio::calculate_normal()
{

#pragma omp parallel for
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

		nn.x = e1.y * e2.z - e1.z * e2.y;
		nn.y = e1.x * e2.z - e1.z * e2.x;
		nn.z = e1.x * e2.y - e1.y * e2.x;

		float n_norm = sqrt(pow(nn.x, 2) + pow(nn.y, 2) + pow(nn.z, 2));
		face[i].n.x = (e1.y * e2.z - e1.z * e2.y) / n_norm;
		face[i].n.y = (e1.x * e2.z - e1.z * e2.x) / n_norm;
		face[i].n.z = (e1.x * e2.y - e1.y * e2.x) / n_norm;
	}
}

void Meshio::point_membership()
{
	int ngrid = 0;
	int f1;
	vertex3D ray;
	ray.x = 1;
	ray.y = 0;
	ray.z = 0;
#pragma omp parallel for collapse(3)
	for (int k = 0; k < ndivz; k++)
	{
		for (int j = 0; j < ndivy; j++)
		{
			for (int i = 0; i < ndivx; i++)
			{
				vertex3D origin;
				origin.x = x_min + ((x_max - x_min) / ((ndivx - 1) * i));
				origin.y = y_min + ((y_max - y_min) / ((ndivy - 1) * j));
				origin.z = z_min + ((z_max - z_min) / ((ndivz - 1) * k));

				int count = 0;
				for (int iface = 0; iface < nface; iface++)
				{
					int v0_ind = face[iface].v[0];
					int v1_ind = face[iface].v[1];
					int v2_ind = face[iface].v[2];

					vertex3D nn = face[iface].n;
					float D = -(nn.x * vertex[v0_ind].x + nn.y * vertex[v0_ind].y + nn.z * vertex[v0_ind].z);
					float t1 = -(nn.x * origin.x + nn.y * origin.y + nn.z * origin.z + D);
					float t2 = (nn.x * ray.x + nn.y * ray.y + nn.z * ray.z);
					float t = t1 / t2;

					vertex3D p_vec;
					p_vec.x = origin.x + t * ray.x;
					p_vec.y = origin.y + t * ray.y;
					p_vec.z = origin.z + t * ray.z;

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

					c0.x = p_vec.x - vertex[v0_ind].x;
					c0.y = p_vec.y - vertex[v0_ind].y;
					c0.z = p_vec.z - vertex[v0_ind].z;
					c1.x = p_vec.x - vertex[v1_ind].x;
					c1.y = p_vec.y - vertex[v1_ind].y;
					c1.z = p_vec.z - vertex[v1_ind].z;
					c2.x = p_vec.x - vertex[v2_ind].x;
					c2.y = p_vec.y - vertex[v2_ind].y;
					c2.z = p_vec.z - vertex[v2_ind].z;

					vertex3D e0c0, e1c1, e2c2;
					e0c0.x = e0.y * c0.z - e0.z * c0.y;
					e0c0.y = e0.x * c0.z - e0.z * c0.x;
					e0c0.z = e0.x * c0.y - e0.y * c0.x;
					e1c1.x = e1.y * c1.z - e1.z * c1.y;
					e1c1.y = e1.x * c1.z - e1.z * c1.x;
					e1c1.z = e1.x * c1.y - e1.y * c1.x;
					e2c2.x = e2.y * c2.z - e2.z * c2.y;
					e2c2.y = e2.x * c2.z - e2.z * c2.x;
					e2c2.z = e2.x * c2.y - e2.y * c2.x;

					float case1, case2, case3;
					case1 = nn.x * e0c0.x + nn.y * e0c0.y + nn.z * e0c0.z;
					case2 = nn.x * e1c1.x + nn.y * e1c1.y + nn.z * e1c1.z;
					case3 = nn.x * e2c2.x + nn.y * e2c2.y + nn.z * e2c2.z;

					if (case1 > 0 && case2 > 0 && case3 > 0 && t > 0)
					{
						count++;
					}
				}
				int e = k * ndivy * ndivx + j * ndivx + i;
				if (count % 2 == 0)
				{
					f1 = 0;
					bbox_flag[e] = (f1);
				}
				else
				{
					f1 = 1;
					bbox_flag[e] = (f1);
				}
				//  if (bbox_flag[i + j*ndivx+k*ndivx*ndivy] == 1)
				//  {
				//     printf("EleIdx: %d, %d, Done!\n", i + j*ndivx+k*ndivx*ndivy, bbox_flag[i + j*ndivx+k*ndivx*ndivy]);
				//  }
			}
		}
	}
}