#include <iostream>
#include <string>
#include <sstream>
#include <boost/algorithm/string.hpp>

#include "render_object.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "../utils/math_utils.h"

ObjData::ObjData(const std::string& objFile)
{
	LoadObj(objFile);
}

void ObjData::Deform(const Eigen::Vector3f& xyzScale)
{
	vertices.row(0) = vertices.row(0) * xyzScale(0); 
	vertices.row(1) = vertices.row(1) * xyzScale(1); 
	vertices.row(2) = vertices.row(2) * xyzScale(2); 
}

void ObjData::LoadObj(const std::string& objFile)
{
	std::fstream reader;
	reader.open(objFile.c_str(), std::ios::in);

	if (!reader.is_open())
	{
		std::cout <<"[ObjData] file not exist!" << std::endl; 
		exit(-1); 
	}

	std::vector<Eigen::Vector3f> vertexVec;
	std::vector<std::vector<Eigen::Vector3i>> faceVec;
	std::vector<Eigen::Vector2f> textureVec;
	std::vector<Eigen::Vector3f> normalVec;

	while (!reader.eof())
	{
		std::string dataType;
		reader >> dataType;

		if(reader.eof()) break;

		if (dataType == "v")
		{
			Eigen::Vector3f temp;
			reader >> temp.x() >> temp.y() >> temp.z();
			vertexVec.push_back(temp);
		}
		else if (dataType == "vn")
		{
			Eigen::Vector3f temp;
			reader >> temp.x() >> temp.y() >> temp.z();
			normalVec.push_back(temp);
		}
		else if (dataType == "vt")
		{
			Eigen::Vector2f temp;
			reader >> temp.x() >> temp.y();
			textureVec.push_back(temp);
		}
		else if (dataType == "f")
		{
			std::vector<Eigen::Vector3i> tempFace(3);
			for (int i = 0; i < 3; i++)
			{
				std::string dataStr;
				reader >> dataStr;
				std::stringstream ss(dataStr);

				for (int j = 0; j < 3; j++)
				{
					std::string temp;
					std::getline(ss, temp, '/');
					if(temp != "")
					{
						tempFace[i](j) = std::stoi(temp);
					}
					else tempFace[i](j) = 0; 
				}
			}
			faceVec.push_back(tempFace);
		}
		else
		{
			std::cout << "datatype : " << dataType << std::endl; 
			std::cout << "[ObjData] unknown type" << std::endl; 
			exit(-1); 
		}
	}

	// find relation map, use texture as key
	int allVertexSize = (int)textureVec.size();
	std::vector<int> vertexMap(allVertexSize, -1);
	std::vector<int> normalMap(allVertexSize, -1);

	for (const std::vector<Eigen::Vector3i>& face : faceVec)
	{
		for (const Eigen::Vector3i& faceParam : face)
		{
			int vertexId = faceParam.x() - 1;
			int textureId = faceParam.y() - 1;
			int normalId = faceParam.z() - 1;
			vertexMap[textureId] = vertexId;
			normalMap[textureId] = normalId;
		}
	}

	// fill vertex data with  texture-key map
	vertices.resize(3, allVertexSize);
	texcoords.resize(2, allVertexSize);
	// Attention: sort the vec as the texture sequence
	for (int textureId = 0; textureId < allVertexSize; textureId++)
	{
		texcoords.col(textureId) = textureVec[textureId];
		vertices.col(textureId) = vertexVec[vertexMap[textureId]];
	}

	// fill face data
	faces.resize(3, faceVec.size());
	for (int faceId = 0; faceId < faceVec.size(); faceId++)
	{
		for (int i = 0; i < 3; i++)
		{
			unsigned int textureId = faceVec[faceId][i].y() - 1;
			faces(i, faceId) = textureId;
		}
	}
	// std::cout << "read done. " << std::endl; 
}


// --------------------------------------------------
// RenderObject
SimpleRenderObject::SimpleRenderObject()
{
	model.setIdentity();
	materialParam = MaterialParam(0.5f, 0.6f, 0.01f, 1.0f);

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO_vertex);
	glGenBuffers(1, &EBO);
}


SimpleRenderObject::~SimpleRenderObject()
{
	glDeleteBuffers(1, &VBO_vertex);
	glDeleteBuffers(1, &EBO);
	glDeleteVertexArrays(1, &VAO);
}


void SimpleRenderObject::DrawDepth(SimpleShader& shader) const
{
	shader.SetMat4("model", model);

	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, 3 * faceNum, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}


void SimpleRenderObject::SetMaterial(const MaterialParam& _materialParam)
{
	materialParam = _materialParam;
}


void SimpleRenderObject::SetFaces(const Eigen::Matrix<unsigned int, 3, -1, Eigen::ColMajor>& faces, const bool inverse)
{
	faceNum = faces.cols();

	glBindVertexArray(VAO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	if (inverse)
	{
		Eigen::Matrix<unsigned int, -1, -1, Eigen::ColMajor> facesInverse = faces;
		facesInverse.row(0).swap(facesInverse.row(2));
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * 3 * faceNum, facesInverse.data(), GL_DYNAMIC_DRAW);
	}
	else
	{
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * 3 * faceNum, faces.data(), GL_DYNAMIC_DRAW);
	}
	glBindVertexArray(0);

}


void SimpleRenderObject::SetVertices(const Eigen::Matrix<float, 3, -1, Eigen::ColMajor>& vertices)
{
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_vertex);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3* vertices.cols(), vertices.data(), GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}


void SimpleRenderObject::SetTransform(const Eigen::Vector3f& _translation, const Eigen::Vector3f& _rotation, const float _scale)
{
	model = EigenUtil::Transform(_translation, _rotation, _scale);
}

// --------------------------------------------------
//RenderObjectColor
RenderObjectColor::RenderObjectColor()
{
	color = Eigen::Vector3f(0.5f, 0.5f, 0.5f);
}


RenderObjectColor::~RenderObjectColor()
{


}


void RenderObjectColor::DrawWhole(SimpleShader& shader) const
{
	shader.SetMat4("model", model);
	shader.SetFloat("material_ambient", materialParam.ambient);
	shader.SetFloat("material_diffuse", materialParam.diffuse);
	shader.SetFloat("material_specular", materialParam.specular);
	shader.SetFloat("material_shininess", materialParam.shininess);
	
	shader.SetVec3("object_color", color);
	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, 3 * faceNum, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}


// --------------------------------------------------
//RenderObjectTexture
RenderObjectTexture::RenderObjectTexture()
{
	glGenBuffers(1, &VBO_texcoord);
	glGenTextures(1, &textureID);

}


RenderObjectTexture::~RenderObjectTexture()
{
	glDeleteBuffers(1, &VBO_texcoord);
	glDeleteTextures(1, &textureID);
}


void RenderObjectTexture::SetTexcoords(const Eigen::Matrix<float, 2, -1, Eigen::ColMajor>& texcoords)
{
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_texcoord);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * texcoords.cols(), texcoords.data(), GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}


void RenderObjectTexture::SetTexture(const std::string& texturePath)
{
	int width, height, nrComponents;
	unsigned char *data = stbi_load(texturePath.c_str(), &width, &height, &nrComponents, 0);
	if (data)
	{
		GLenum format;
		if (nrComponents == 1)
			format = GL_RED;
		else if (nrComponents == 3)
			format = GL_RGB;
		else if (nrComponents == 4)
			format = GL_RGBA;

		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);

		glGenerateMipmap(GL_TEXTURE_2D);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		stbi_image_free(data);
	}
	else
	{
		// throw std::string("Texture failed to load at path: " + texturePath);
		std::cout << "[RenderObjectTexture] Texture failed to load at path " << texturePath << std::endl; 
		exit(-1); 
	}
}


void RenderObjectTexture::DrawWhole(SimpleShader& shader) const
{
	shader.SetMat4("model", model);
	shader.SetFloat("material_ambient", materialParam.ambient);
	shader.SetFloat("material_diffuse", materialParam.diffuse);
	shader.SetFloat("material_specular", materialParam.specular);
	shader.SetFloat("material_shininess", materialParam.shininess);
	
	glBindTexture(GL_TEXTURE_2D, textureID);

	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, 3 * faceNum, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}


// --------------------------------------------------
//RenderObjectMesh
RenderObjectMesh::RenderObjectMesh()
{
	glGenBuffers(1, &VBO_color);
}


RenderObjectMesh::~RenderObjectMesh()
{
	glDeleteBuffers(1, &VBO_color);
}


void RenderObjectMesh::SetColors(const Eigen::Matrix<float, 3, -1, Eigen::ColMajor>& colors)
{
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_color);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * colors.cols(), colors.data(), GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}


void RenderObjectMesh::DrawWhole(SimpleShader& shader) const
{
	shader.SetMat4("model", model);
	
	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, 3 * faceNum, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}


// --------------------------------------------------
//BallStickObject
//BallStickObject
BallStickObject::BallStickObject(
	const ObjData& ballObj, const ObjData& stickObj,
	const std::vector<Eigen::Vector3f>& balls, 
	const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& sticks,
	float ballSize, float StickSize, const Eigen::Vector3f& color)
{
	for (int i = 0; i < balls.size(); i++)
	{
		// std::cout << balls[i].transpose() << std::endl; 
		ObjData ballObjCopy(ballObj);
		ballObjCopy.Deform(Eigen::Vector3f(ballSize, ballSize, ballSize));

		RenderObjectColor* ballObject = new RenderObjectColor();
		ballObject->SetFaces(ballObjCopy.faces);
		ballObject->SetVertices(ballObjCopy.vertices);
		ballObject->SetTransform(balls[i], Eigen::Vector3f::Zero(), 1.0f);
		ballObject->SetColor(color);
		objectPtrs.push_back(ballObject);
	}

	for (const std::pair<Eigen::Vector3f, Eigen::Vector3f>& stick : sticks)
	{
		ObjData stickObjCopy(stickObj);
		Eigen::Vector3f direction = stick.first - stick.second;
		stickObjCopy.Deform(Eigen::Vector3f(StickSize, StickSize,  0.5f * direction.norm()));

		RenderObjectColor* stickObject = new RenderObjectColor();

		Eigen::Vector3f rotation = (Eigen::Vector3f(0.0f, 0.0f, 1.0f).cross(direction.normalized())).normalized()
			* acosf(Eigen::Vector3f(0.0f, 0.0f, 1.0f).dot(direction.normalized()));

		Eigen::Vector3f translation = (stick.first + stick.second) * 0.5f;

		stickObject->SetFaces(stickObjCopy.faces);
		stickObject->SetVertices(stickObjCopy.vertices);
		stickObject->SetTransform(translation, rotation, 1.0f);
		// stickObject->SetColor(Eigen::Vector3f::Ones() - color);
		stickObject->SetColor(color); 

		objectPtrs.push_back(stickObject);
	}
}

BallStickObject::BallStickObject( // point clouds 
	const ObjData& ballObj, 
	const std::vector<Eigen::Vector3f>& balls, 
	const std::vector<float> sizes, 
	const std::vector<Eigen::Vector3f>& colors
)
{
	for (int i = 0; i < balls.size(); i++)
	{
		// std::cout << balls[i].transpose() << std::endl; 
		float ballSize = sizes[i]; 
		Eigen::Vector3f color = colors[i]; 
		ObjData ballObjCopy(ballObj);
		ballObjCopy.Deform(Eigen::Vector3f(ballSize, ballSize, ballSize));

		RenderObjectColor* ballObject = new RenderObjectColor();
		ballObject->SetFaces(ballObjCopy.faces);
		ballObject->SetVertices(ballObjCopy.vertices);
		ballObject->SetTransform(balls[i], Eigen::Vector3f::Zero(), 1.0f);
		ballObject->SetColor(color);
		objectPtrs.push_back(ballObject);
	}
}

BallStickObject::BallStickObject(
	const ObjData& ballObj, const ObjData& stickObj,
	const std::vector<Eigen::Vector3f>& balls,
	const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& sticks,
	float ballSize, float StickSize, const std::vector<Eigen::Vector3f>& colors)
{
	for (int i = 0; i < balls.size(); i++)
	{
		// std::cout << balls[i].transpose() << std::endl; 
		ObjData ballObjCopy(ballObj);
		float current_size = ballSize; 
		Eigen::Vector3f color = colors[i];
		ballObjCopy.Deform(Eigen::Vector3f(current_size, current_size, current_size));
		RenderObjectColor* ballObject = new RenderObjectColor();
		ballObject->SetFaces(ballObjCopy.faces);
		ballObject->SetVertices(ballObjCopy.vertices);
		ballObject->SetTransform(balls[i], Eigen::Vector3f::Zero(), 1.0f);
		ballObject->SetColor(colors[i]);
		objectPtrs.push_back(ballObject);
	}

	for (const std::pair<Eigen::Vector3f, Eigen::Vector3f>& stick : sticks)
	{
		ObjData stickObjCopy(stickObj);
		Eigen::Vector3f direction = stick.first - stick.second;
		stickObjCopy.Deform(Eigen::Vector3f(StickSize, StickSize, 0.5f * direction.norm()));

		RenderObjectColor* stickObject = new RenderObjectColor();

		Eigen::Vector3f rotation = (Eigen::Vector3f(0.0f, 0.0f, 1.0f).cross(direction.normalized())).normalized()
			* acosf(Eigen::Vector3f(0.0f, 0.0f, 1.0f).dot(direction.normalized()));

		Eigen::Vector3f translation = (stick.first + stick.second) * 0.5f;

		stickObject->SetFaces(stickObjCopy.faces);
		stickObject->SetVertices(stickObjCopy.vertices);
		stickObject->SetTransform(translation, rotation, 1.0f);
		// stickObject->SetColor(Eigen::Vector3f::Ones() - color);
		stickObject->SetColor(colors[0]);

		objectPtrs.push_back(stickObject);
	}
}

BallStickObject::~BallStickObject()
{
	for (RenderObjectColor* ptr : objectPtrs)
	{
		if(ptr!=nullptr)
			delete ptr;
		ptr = nullptr;
	}
}

void BallStickObject::deleteObjects()
{
	for (RenderObjectColor* ptr : objectPtrs)
	{
		if(ptr!=nullptr)
			delete ptr;
		ptr = nullptr;
	}
	objectPtrs.clear(); 
}

void BallStickObject::Draw(SimpleShader& shader)
{
	for (RenderObjectColor* ptr : objectPtrs)
	{
		ptr->DrawWhole(shader);
	}
}