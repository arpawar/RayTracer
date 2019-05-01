#ifndef READ_RAW_H
#define READ_RAW_H

#include <vector>
#include <array>
#include <cmath>
#include "basic_datastructure.h"
using namespace std;

class Meshio
{
public:
	Meshio();
	int flag;
	int nvert, nface, ngrid;
	float x_min, x_max, y_min, y_max, z_min, z_max;
	int ndivx, ndivy, ndivz;
	vector<vertex3D> vertex;
	vector<tri> face;
	vector<int> bbox_flag;
	void read_raw_file(const char * fn, vector<vertex3D> &vertex_out, vector<tri> &face_out);
	void set_bounding_box(int nx, int ny, int nz);
	void calculate_grid(vector<vertex3D> &origin);
	void calculate_normal();
	void point_membership();
	
};


#endif

