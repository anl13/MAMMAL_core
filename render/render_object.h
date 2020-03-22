#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <vector>
#include <string>
#include <Eigen/Eigen>

#include "shader.h"


enum RENDER_OBJECT_TYPE
{
	RENDER_OBJECT_COLOR,
	RENDER_OBJECT_TEXTURE,
	RENDER_OBJECT_MESH
};


class ObjData
{
public:
	ObjData() {};
	~ObjData() {};

	ObjData(const std::string& objFile);

	ObjData(
		Eigen::Matrix<float, 3, -1, Eigen::ColMajor>& _vertices,
		Eigen::Matrix<float, 2, -1, Eigen::ColMajor>& _texcoords,
		Eigen::Matrix<unsigned int, 3, -1, Eigen::ColMajor>& _faces)
		: vertices(_vertices), texcoords(_texcoords), faces(_faces)
	{}

	ObjData(
		Eigen::Matrix<float, 3, -1, Eigen::ColMajor>& _vertices,
		Eigen::Matrix<unsigned int, 3, -1, Eigen::ColMajor>& _faces)
		: vertices(_vertices), faces(_faces)
	{}

	void LoadObj(const std::string& objFile);
	void Deform(const Eigen::Vector3f& xyzScale); 

	Eigen::Matrix<float, 3, -1, Eigen::ColMajor> vertices;
	Eigen::Matrix<float, 2, -1, Eigen::ColMajor> texcoords;
	Eigen::Matrix<unsigned int, 3, -1, Eigen::ColMajor> faces;
	Eigen::Matrix<float, 3, -1, Eigen::ColMajor> colors;

private:
};


struct MaterialParam
{
	MaterialParam(float _ambient, float _diffuse, float _specular, float _shininess)
	{
		ambient = _ambient;
		diffuse = _diffuse;
		specular = _specular;
		shininess = _shininess;
	}

	MaterialParam() :MaterialParam(1.0f, 1.0f, 1.0f, 1.0f) {}

	float ambient;
	float diffuse;
	float specular;
	float shininess;
};


class RenderObject
{
public:
	RenderObject();
	RenderObject(const RenderObject& _) = delete;
	RenderObject& operator=(const RenderObject& _) = delete;
	virtual ~RenderObject();

	virtual RENDER_OBJECT_TYPE GetType() const = 0;
	virtual void DrawWhole(Shader& shader) const = 0;
	virtual void DrawDepth(Shader& shader) const;
	virtual void SetTransform(const Eigen::Vector3f& _translation, const Eigen::Vector3f& _rotation, const float _scale);
	virtual void SetMaterial(const MaterialParam& _materialParam);
	virtual void SetFaces(const Eigen::Matrix<unsigned int, 3, -1, Eigen::ColMajor>& faces, const bool inverse = false);
	virtual void SetVertices(const Eigen::Matrix<float, 3, -1, Eigen::ColMajor>& vertices);

protected:
	GLuint VAO;
	GLuint VBO_vertex;
	GLuint EBO;

	Eigen::Matrix<float,4,4,Eigen::ColMajor> model;					// transform mat, include translation, rotation, scale

	int faceNum;
	MaterialParam materialParam;
};


class RenderObjectColor : virtual public RenderObject
{
public:
	RenderObjectColor();
	RenderObjectColor(const RenderObjectColor& _) = delete;
	RenderObjectColor& operator=(const RenderObjectColor& _) = delete;
	virtual ~RenderObjectColor();

	virtual RENDER_OBJECT_TYPE GetType() const { return RENDER_OBJECT_COLOR; }
	virtual void DrawWhole(Shader& shader) const;

	void SetColor(const Eigen::Vector3f& _color) { color = _color; }
private:
	Eigen::Vector3f color;			
};


class RenderObjectTexture : virtual public RenderObject
{
public:
	RenderObjectTexture();
	RenderObjectTexture(const RenderObjectTexture& _) = delete;
	RenderObjectTexture& operator=(const RenderObjectTexture& _) = delete;
	virtual ~RenderObjectTexture();

	virtual RENDER_OBJECT_TYPE GetType() const { return RENDER_OBJECT_TEXTURE; }
	virtual void DrawWhole(Shader& shader) const;

	virtual void SetTexture(const std::string& texturePath);
	virtual void SetTexcoords(const Eigen::Matrix<float, 2, -1, Eigen::ColMajor>& texcoords);

private:
	GLuint VBO_texcoord;
	GLuint textureID;
};


class RenderObjectMesh : virtual public RenderObject
{
public:
	RenderObjectMesh();
	RenderObjectMesh(const RenderObjectMesh& _) = delete;
	RenderObjectMesh& operator=(const RenderObjectMesh& _) = delete;
	virtual ~RenderObjectMesh();

	virtual RENDER_OBJECT_TYPE GetType() const { return RENDER_OBJECT_MESH; }
	virtual void DrawWhole(Shader& shader) const;

	virtual void SetColors(const Eigen::Matrix<float, 3, -1, Eigen::ColMajor>& colors);

private:
	GLuint VBO_color;
};


class BallStickObject : virtual public RenderObjectColor
{
public:
	BallStickObject(
		const ObjData& ballObj, const ObjData& stickObj,
		const std::vector<Eigen::Vector3f>& balls, 
		const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& sticks,
		float ballSize, float StickSize, const Eigen::Vector3f& color);

	BallStickObject(
		const ObjData& ballObj, 
		const std::vector<Eigen::Vector3f>& balls, 
		const std::vector<float> sizes, 
		const std::vector<Eigen::Vector3f>& colors
	);

	BallStickObject(
		const ObjData& ballObj, const ObjData& stickObj,
		const std::vector<Eigen::Vector3f>& balls,
		const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& sticks,
		float ballSize, float StickSize, const std::vector<Eigen::Vector3f>& color);

	BallStickObject() = delete;
	BallStickObject(const BallStickObject& _) = delete;
	BallStickObject& operator=(const BallStickObject& _) = delete;
	virtual ~BallStickObject();
	void deleteObjects(); 
	virtual void Draw(Shader& shader);
private:
	std::vector<RenderObjectColor*> objectPtrs;
};