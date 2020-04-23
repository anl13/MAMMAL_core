#include "volume.h"

Volume::Volume()
{
	initVolume(); 
}

void Volume::initVolume()
{
	resX = 128;
	resY = 128; 
	resZ = 128;
	dx = 0.006;
	dy = 0.006;
	dz = 0.006;

	data = new float[resX * resY * resZ];
	surface = new bool[resX*resY*resZ]; 
	point_cloud.clear();
}

Volume::~Volume()
{
	delete[] data;
	delete[] surface; 
}

void Volume::get3DBox(
	std::vector<Eigen::Vector3d>& points,
	std::vector<Eigen::Vector2i>& edges) 
	// return a cube
{
	/*
	    1---2      dz
	   /   /|      |
	  4---3 6      C-->dy
	  |   |/      /
	  8---7      dx
	  5 is Origin
	*/
	std::vector<Eigen::Vector3i> dirs = {
		{-1,-1,1},{-1,1,1},{1,1,1},{1,-1,1},
	{-1,-1,-1},{-1,1,-1},{1,1,-1},{1,-1,-1}
	};
	points.resize(8);
	for (int i = 0; i < 8; i++)
	{
		points[i](0) = center(0) + dirs[i](0) * (resX / 2)*dx;
		points[i](1) = center(1) + dirs[i](1) * (resY / 2)*dy;
		points[i](2) = center(2) + dirs[i](2) * (resZ / 2)*dz;
	}
	edges = {
		{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},
	{0,4},{1,5},{2,6},{3,7}
	};
}


int Volume::xyz2index(const int& x, const int& y, const int& z)
{
	return z*(resX*resY) + y * resX + x;
}

void Volume::computeVolumeFromRoi(
	std::vector<ROIdescripter>& det
)
{
	//cv::Mat vis = vis_float_image(det[0].chamfer);
	//cv::imshow("vis chamfer", vis); 
	//cv::waitKey();
	//exit(-1);

	Eigen::Vector3f O;
	O(0) = center(0) - (resX / 2)*dx;
	O(1) = center(1) - (resY / 2)*dy;
	O(2) = center(2) - (resZ / 2)*dz;

#pragma omp parallel for
	for (int x = 0; x < resX; x++)
	{
		for (int y = 0; y < resY; y++)
		{
			for (int z = 0; z < resZ; z++)
			{
				Eigen::Vector3d P;
				P(0) = O(0) + dx * x;
				P(1) = O(1) + dy * y; 
				P(2) = O(2) + dz * z; 
				float value = 1; // innner
				for (int view = 0; view < det.size(); view++)
				{
					float f = det[view].queryChamfer(P);
					if (f < 1)
					{
						value = 0; break;
					}
				}
				int index = xyz2index(x, y, z);
				data[index] = value; 
			}
		}
	}
}

Eigen::Vector3f Volume::index2point(
	const int& x, const int& y, const int& z
)
{
	Eigen::Vector3f P;
	P(0) = center(0) - (resX / 2)*dx + dx * x;
	P(1) = center(1) - (resY / 2)*dy + dy * y;
	P(2) = center(2) - (resZ / 2)*dz + dz * z;
	return P; 
}

void Volume::getSurface()
{
	point_cloud.clear(); 
	normals.clear(); 
	// neighbour: up,down,left,right,front,back
	int nx[6] = { -1, 0, 0, 0, 0, 1 };
	int ny[6] = { 0, 1, -1, 0, 0, 0 };
	int nz[6] = { 0, 0, 0, 1, -1, 0 };

	// lambda表达式，用于判断某个点是否在Voxel的范围内
	auto outOfRange = [&](int indexX, int indexY, int indexZ) {
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= resX
			|| indexY >= resY
			|| indexZ >= resZ;
	};

	for (int indexX = 0; indexX < resX; indexX++)
		for (int indexY = 0; indexY < resY; indexY++)
			for (int indexZ = 0; indexZ < resZ; indexZ++)
			{
				int index = xyz2index(indexX, indexY, indexZ);
				if (data[index] == 0)
				{
					surface[index] = false;
					continue;
				}
				bool ans = false;
				for (int i = 0; i < 6; i++)
				{
					if (outOfRange(indexX + nx[i], indexY + ny[i], indexZ + nz[i]))
					{
						ans = false; 
						break; 
					}
					int neighbour_index = xyz2index(indexX + nx[i], indexY + ny[i], indexZ + nz[i]);
					if (data[neighbour_index] == 0)
					{
						ans = true; break;
					}
				}
				surface[index] = ans;
				if (ans)
				{
					point_cloud.emplace_back(index2point(indexX, indexY, indexZ));
					Eigen::Vector3f n = computeNormal(indexX, indexY, indexZ);
					normals.push_back(n);
				}
			}
	point_cloud_eigen.resize(3, point_cloud.size());
	for (int i = 0; i < point_cloud.size(); i++)
		point_cloud_eigen.col(i) = point_cloud[i];
}

Eigen::Vector3f Volume::computeNormal(int indX, int indY, int indZ)
{
	auto outOfRange = [&](int indexX, int indexY, int indexZ) {
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= resX
			|| indexY >= resY
			|| indexZ >= resZ;
	};
	int m_neiborSize = 2;
	std::vector<Eigen::Vector3f> neiborList;
	std::vector<Eigen::Vector3f> innerList;

	for (int dX = -m_neiborSize; dX <= m_neiborSize; dX++)
		for (int dY = -m_neiborSize; dY <= m_neiborSize; dY++)
			for (int dZ = -m_neiborSize; dZ <= m_neiborSize; dZ++)
			{
				if (!dX && !dY && !dZ)
					continue;
				int neiborX = indX + dX;
				int neiborY = indY + dY;
				int neiborZ = indZ + dZ;
				if (!outOfRange(neiborX, neiborY, neiborZ))
				{
					Eigen::Vector3f point = index2point(neiborX, neiborY, neiborZ);
					int index = xyz2index(neiborX, neiborY, neiborZ);
					if (surface[index])
						neiborList.push_back(point);
					else if (data[index] > 0)
						innerList.push_back(point);
				}
			}

	Eigen::Vector3f point = index2point(indX, indY, indZ);

	Eigen::MatrixXf matA(3, neiborList.size());
	for (int i = 0; i < neiborList.size(); i++)
		matA.col(i) = neiborList[i] - point;
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(matA * matA.transpose());
	Eigen::Vector3f eigenValues = eigenSolver.eigenvalues();
	int indexEigen = 0;
	if (abs(eigenValues[1]) < abs(eigenValues[indexEigen]))
		indexEigen = 1;
	if (abs(eigenValues[2]) < abs(eigenValues[indexEigen]))
		indexEigen = 2;
	Eigen::Vector3f normalVector = eigenSolver.eigenvectors().col(indexEigen);

	Eigen::Vector3f innerCenter = Eigen::Vector3f::Zero();
	for (auto const& vec : innerList)
		innerCenter += vec;
	innerCenter /= innerList.size();

	if (normalVector.dot(point - innerCenter) < 0)
		normalVector *= -1;
	return normalVector;
}

void Volume::saveXYZFileWithNormal(std::string filename)
{
	std::ofstream fout(filename);
	if (!fout.is_open())
	{
		std::cout << filename << " is not open!" << std::endl;
		exit(-1); 
	}
	for (int i = 0; i < point_cloud.size(); i++)
	{
		fout << point_cloud[i].transpose() << " " << normals[i].transpose() << std::endl;
	}
	fout.close();
}

void Volume::readXYZFileWithNormal(std::string filename)
{
	std::ifstream fin(filename);
	if (!fin.is_open())
	{
		std::cout << filename << " is not open!" << std::endl;
		exit(-1);
	}
	point_cloud.clear();
	while (!fin.eof())
	{
		float x, y, z;
		fin >> x;
		if (fin.eof())break;
		fin >> y >> z; 
		point_cloud.push_back(Eigen::Vector3f(x, y, z));
		fin >> x >> y >> z; 
 	}
	fin.close();

	point_cloud_eigen.resize(3, point_cloud.size());
	for (int i = 0; i < point_cloud.size(); i++)
	{
		point_cloud_eigen.col(i) = point_cloud[i];
	}
}
