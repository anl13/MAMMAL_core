#include "model.h"

void Model::CalcNormal()
{
	normals.setZero(3, vertices.cols());
	for (int fIdx = 0; fIdx < faces.cols(); fIdx++) {
		const auto face = faces.col(fIdx);
		Eigen::Vector3d normal = ((vertices.col(face.x()) - vertices.col(face.y())).cross(
			vertices.col(face.y()) - vertices.col(face.z()))).normalized();

		normals.col(face.x()) += normal;
		normals.col(face.y()) += normal;
		normals.col(face.z()) += normal;
	}
	normals.colwise().normalize();
}

void Model::Load(const std::string& filename)
{
	auto SplitString = [](const std::string& s, const std::string& c) {
		std::vector<std::string> v;
		std::string::size_type p1, p2;
		p2 = s.find(c);
		p1 = 0;
		while (std::string::npos != p2) {
			v.push_back(s.substr(p1, p2 - p1));

			p1 = p2 + c.size();
			p2 = s.find(c, p1);
		}
		if (p1 != s.length())
			v.push_back(s.substr(p1));
		return v;
	};

	std::ifstream ifs(filename);
	if (!ifs.is_open()) {
		std::cerr << "file not exist " << filename << std::endl;
		std::abort();
	}

	std::vector<Eigen::Vector3d> _vertices;
	std::vector<Eigen::Vector3u> _faces;

	char tmp[1024];
	while (ifs.getline(tmp, 1024)) {
		auto t = SplitString(std::string(tmp), " ");
		if (t.size() > 0) {
			if (t[0] == "v")
				_vertices.emplace_back(Eigen::Vector3d(std::stod(t[1]), std::stod(t[2]), std::stod(t[3])));
			else if (t[0] == "f")
				_faces.emplace_back(Eigen::Vector3u(std::stoi(SplitString(t[1], "//")[0]) - 1, std::stoi(SplitString(t[2], "//")[0]) - 1, std::stoi(SplitString(t[3], "//")[0]) - 1));
		}
	}

	if (_vertices.size()>0)
	{
		vertices.resize(3, _vertices.size());
		//vertices = Eigen::Map<Eigen::Matrix3Xd>(_vertices.begin()->data(), 3, _vertices.size());
		for (int i = 0; i < _vertices.size(); i++)
		{
			vertices.col(i) = _vertices[i];
		}
	}
	if (_faces.size() > 0)
	{
		faces.resize(3, _faces.size());
		//faces = Eigen::Map<Eigen::Matrix3Xu>(_faces.begin()->data(), 3, _faces.size());
		for (int i = 0; i < _faces.size(); i++)
		{
			faces.col(i) = _faces[i];
		}
	}
	CalcNormal();
}

void Model::Save(const std::string& filename) const
{
	std::ofstream fs(filename);
	fs << "# Vertices: " << vertices.size() << std::endl;
	fs << "# Faces: " << faces.size() << std::endl;

	for (int i = 0; i < vertices.cols(); i++)
		fs << "v " << vertices(0, i) << " " << vertices(1, i) << " " << vertices(2, i) << std::endl;

	for (int i = 0; i < faces.cols(); i++)
		fs << "f " << faces(0, i) + 1 << " " << faces(1, i) + 1 << " " << faces(2, i) + 1 << std::endl;

	fs.close();
}