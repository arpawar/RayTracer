#ifndef BASIC_DATASTRUCTURE_H
#define BASIC_DATASTRUCTURE_H

#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <iostream>


class vertex3D
{
public:
	float x, y, z;
};

class tri
{
public:
	int v[3];
	vertex3D n;
};



#endif