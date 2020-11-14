#include <iostream>
#include <string>
#include <sstream>
#include <boost/algorithm/string.hpp>

#include "render_object.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "../utils/math_utils.h"

// --------------------------------------------------
// RenderObject
SimpleRenderObject::SimpleRenderObject()
{
	model.setIdentity();

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO_vertex);
	glGenBuffers(1, &EBO);
	glGenBuffers(1, &VBO_normal); 

	isMultiLight = false; 
}


SimpleRenderObject::~SimpleRenderObject()
{
	glDeleteBuffers(1, &VBO_vertex);
	glDeleteBuffers(1, &VBO_normal); 
	glDeleteBuffers(1, &EBO);
	glDeleteVertexArrays(1, &VAO);
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

void SimpleRenderObject::SetFaces(const std::vector<Eigen::Vector3u>& faces)
{
	faceNum = faces.size(); 
	glBindVertexArray(VAO); 
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO); 
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * 3 * faceNum, faces.data(), GL_DYNAMIC_DRAW); 
	glBindVertexArray(0); 
}


void SimpleRenderObject::SetVertices(const Eigen::Matrix<float, 3, -1, Eigen::ColMajor>& vertices, int layout_location)
{
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_vertex);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3* vertices.cols(), vertices.data(), GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(layout_location);
	glVertexAttribPointer(layout_location, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void SimpleRenderObject::SetVertices(const std::vector<Eigen::Vector3f>& vertices, int layout_location)
{
	glBindVertexArray(VAO); 
	glBindBuffer(GL_ARRAY_BUFFER, VBO_vertex); 
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * vertices.size(), vertices.data(), GL_DYNAMIC_DRAW); 
	glEnableVertexAttribArray(layout_location); 
	glVertexAttribPointer(layout_location, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0); 
	glBindBuffer(GL_ARRAY_BUFFER, 0); 
	glBindVertexArray(0); 
}

void SimpleRenderObject::SetTransform(const Eigen::Vector3f& _translation, const Eigen::Vector3f& _rotation, const float _scale)
{
	model = Transform(_translation, _rotation, _scale);
}

void SimpleRenderObject::SetNormal(const std::vector<Eigen::Vector3f>& normals, int layout_location)
{
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_normal);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * normals.size(), normals.data(), GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(layout_location);
	glVertexAttribPointer(layout_location, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void SimpleRenderObject::SetNormal(const Eigen::Matrix<float, 3, -1, Eigen::ColMajor>& normals,int layout_location)
{
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_normal);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * normals.cols(), normals.data(), GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(layout_location);
	glVertexAttribPointer(layout_location, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
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


void RenderObjectTexture::SetTexcoords(const Eigen::Matrix<float, 2, -1, Eigen::ColMajor>& texcoords, int layout_location)
{
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_texcoord);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * texcoords.cols(), texcoords.data(), GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(layout_location);
	glVertexAttribPointer(layout_location, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void RenderObjectTexture::SetTexcoords(const std::vector<Eigen::Vector2f>& texcoords, int layout_location)
{
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_texcoord);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * texcoords.size(), texcoords.data(), GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(layout_location);
	glVertexAttribPointer(layout_location, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}


void RenderObjectTexture::SetTexture(const std::string& texturePath)
{
	int width, height, nrComponents;
	//unsigned char *data = stbi_load(texturePath.c_str(), &width, &height, &nrComponents, 0);
	cv::Mat a = cv::imread(texturePath); 
	cv::cvtColor(a, a, cv::COLOR_BGR2RGB); 
	//std::cout << "nrCompo: " << nrComponents << std::endl; 

	if (1)
	{
		GLenum format;
		//if (nrComponents == 1)
		//	format = GL_RED;
		//else if (nrComponents == 3)
		//	format = GL_RGB;
		//else if (nrComponents == 4)
		//	format = GL_RGBA;

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		format = GL_RGB; 
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D, 0, format, a.cols, a.rows, 0, format, GL_UNSIGNED_BYTE, a.data);

		glGenerateMipmap(GL_TEXTURE_2D);

		//stbi_image_free(data);
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


void RenderObjectMesh::SetColors(const Eigen::Matrix<float, 3, -1, Eigen::ColMajor>& colors, int layout_location)
{
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_color);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * colors.cols(), colors.data(), GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(layout_location);
	glVertexAttribPointer(layout_location, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}



void RenderObjectMesh::SetColors(const std::vector<Eigen::Vector3f> &colors, int layout_location)
{
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_color);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * colors.size(), colors.data(), GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(layout_location);
	glVertexAttribPointer(layout_location, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

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
	const MeshEigen& ballObj, const MeshEigen& stickObj,
	const std::vector<Eigen::Vector3f>& balls, 
	const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& sticks,
	float ballSize, float StickSize, const Eigen::Vector3f& color)
{
	for (int i = 0; i < balls.size(); i++)
	{
		// std::cout << balls[i].transpose() << std::endl; 
		//ObjData ballObjCopy(ballObj);
		//ballObjCopy.Deform(Eigen::Vector3f(ballSize, ballSize, ballSize));

		RenderObjectColor* ballObject = new RenderObjectColor();
		ballObject->SetFaces(ballObj.faces);
		ballObject->SetVertices(ballObj.vertices);
		ballObject->SetNormal(ballObj.normals); 
		ballObject->SetTransform(balls[i], Eigen::Vector3f::Zero(), ballSize);
		ballObject->SetColor(color);
		objectPtrs.push_back(ballObject);
	}

	for (const std::pair<Eigen::Vector3f, Eigen::Vector3f>& stick : sticks)
	{
		MeshEigen stickObjCopy = stickObj;
		Eigen::Vector3f direction = stick.first - stick.second;
		stickObjCopy.Deform(Eigen::Vector3f(StickSize, StickSize,  0.5f * direction.norm()));

		RenderObjectColor* stickObject = new RenderObjectColor();

		Eigen::Vector3f rotation = (Eigen::Vector3f(0.0f, 0.0f, 1.0f).cross(direction.normalized())).normalized()
			* acosf(Eigen::Vector3f(0.0f, 0.0f, 1.0f).dot(direction.normalized()));

		Eigen::Vector3f translation = (stick.first + stick.second) * 0.5f;

		stickObject->SetFaces(stickObjCopy.faces);
		stickObject->SetVertices(stickObjCopy.vertices);
		stickObject->SetNormal(stickObjCopy.normals); 
		stickObject->SetTransform(translation, rotation, 1.0f);
		stickObject->SetColor(color); 
		objectPtrs.push_back(stickObject);
	}
}

BallStickObject::BallStickObject( // point clouds 
	const MeshEigen& ballObj, 
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
		//ObjData ballObjCopy(ballObj);
		//ballObjCopy.Deform(Eigen::Vector3f(ballSize, ballSize, ballSize));

		RenderObjectColor* ballObject = new RenderObjectColor();
		ballObject->SetFaces(ballObj.faces);
		ballObject->SetVertices(ballObj.vertices);
		ballObject->SetNormal(ballObj.normals); 
		ballObject->SetTransform(balls[i], Eigen::Vector3f::Zero(), sizes[i]);
		ballObject->SetColor(color);
		objectPtrs.push_back(ballObject);
	}
}

BallStickObject::BallStickObject(
	const MeshEigen& ballObj, const MeshEigen& stickObj,
	const std::vector<Eigen::Vector3f>& balls,
	const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& sticks,
	float ballSize, float StickSize, const std::vector<Eigen::Vector3f>& colors)
{
	for (int i = 0; i < balls.size(); i++)
	{
		// std::cout << balls[i].transpose() << std::endl; 
		//ObjData ballObjCopy(ballObj);
		float current_size = ballSize; 
		Eigen::Vector3f color = colors[i];
		//ballObjCopy.Deform(Eigen::Vector3f(current_size, current_size, current_size));
		RenderObjectColor* ballObject = new RenderObjectColor();
		ballObject->SetFaces(ballObj.faces);
		ballObject->SetVertices(ballObj.vertices);
		ballObject->SetNormal(ballObj.normals); 
		ballObject->SetTransform(balls[i], Eigen::Vector3f::Zero(), ballSize);
		ballObject->SetColor(colors[i]);
		objectPtrs.push_back(ballObject);
	}

	for (const std::pair<Eigen::Vector3f, Eigen::Vector3f>& stick : sticks)
	{
		MeshEigen stickObjCopy = stickObj;
		Eigen::Vector3f direction = stick.first - stick.second;
		stickObjCopy.Deform(Eigen::Vector3f(StickSize, StickSize, 0.5f * direction.norm()));

		RenderObjectColor* stickObject = new RenderObjectColor();

		Eigen::Vector3f rotation = (Eigen::Vector3f(0.0f, 0.0f, 1.0f).cross(direction.normalized())).normalized()
			* acosf(Eigen::Vector3f(0.0f, 0.0f, 1.0f).dot(direction.normalized()));

		Eigen::Vector3f translation = (stick.first + stick.second) * 0.5f;

		stickObject->SetFaces(stickObjCopy.faces);
		stickObject->SetVertices(stickObjCopy.vertices);
		stickObject->SetNormal(stickObjCopy.normals); 
		stickObject->SetTransform(translation, rotation, 1.0f);
		stickObject->SetColor(colors[0]);

		objectPtrs.push_back(stickObject);
	}
}

BallStickObject::~BallStickObject()
{
	deleteObjects();
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