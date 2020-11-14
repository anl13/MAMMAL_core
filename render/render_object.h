#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <vector>
#include <string>
#include <Eigen/Eigen>

#include "shader.h"
#include "../utils/math_utils.h"
#include "../utils/mesh.h"

enum RENDER_OBJECT_TYPE
{
	RENDER_OBJECT_COLOR,
	RENDER_OBJECT_TEXTURE,
	RENDER_OBJECT_MESH
};

class SimpleRenderObject
{
public:
	SimpleRenderObject();
	SimpleRenderObject(const SimpleRenderObject& _) = delete;
	SimpleRenderObject& operator=(const SimpleRenderObject& _) = delete;
	virtual ~SimpleRenderObject();

	virtual RENDER_OBJECT_TYPE GetType() const = 0;
	virtual void DrawWhole(SimpleShader& shader) const = 0;
	virtual void SetTransform(const Eigen::Vector3f& _translation, const Eigen::Vector3f& _rotation, const float _scale);

	virtual void SetFaces(const Eigen::Matrix<unsigned int, 3, -1, Eigen::ColMajor>& faces, const bool inverse = false);
	virtual void SetVertices(const Eigen::Matrix<float, 3, -1, Eigen::ColMajor>& vertices, int layout_location=0);
	
	virtual void SetVertices(const std::vector<Eigen::Vector3f>& vertices, int layout_location=0); 
	virtual void SetFaces(const std::vector<Eigen::Vector3u>& faces); 

	virtual void SetNormal(const Eigen::Matrix<float, 3, -1, Eigen::ColMajor>& normals, int layout_location=1);
	virtual void SetNormal(const std::vector<Eigen::Vector3f> &_normals, int layout_location=1);

	bool isMultiLight; 
protected:
	GLuint VAO;
	GLuint EBO;

	GLuint VBO_vertex;
	GLuint VBO_normal; 

	Eigen::Matrix<float,4,4,Eigen::ColMajor> model;					// transform mat, include translation, rotation, scale

	int faceNum;
};


class RenderObjectColor : virtual public SimpleRenderObject
{
public:
	RenderObjectColor();
	RenderObjectColor(const RenderObjectColor& _) = delete;
	RenderObjectColor& operator=(const RenderObjectColor& _) = delete;
	virtual ~RenderObjectColor();

	virtual RENDER_OBJECT_TYPE GetType() const { return RENDER_OBJECT_COLOR; }
	virtual void DrawWhole(SimpleShader& shader) const;
	
	void SetColor(const Eigen::Vector3f& _color) { color = _color; }
private:
	Eigen::Vector3f color;			
};


class RenderObjectTexture : virtual public SimpleRenderObject
{
public:
	RenderObjectTexture();
	RenderObjectTexture(const RenderObjectTexture& _) = delete;
	RenderObjectTexture& operator=(const RenderObjectTexture& _) = delete;
	virtual ~RenderObjectTexture();

	virtual RENDER_OBJECT_TYPE GetType() const { return RENDER_OBJECT_TEXTURE; }
	virtual void DrawWhole(SimpleShader& shader) const;

	virtual void SetTexture(const std::string& texturePath);
	virtual void SetTexcoords(const Eigen::Matrix<float, 2, -1, Eigen::ColMajor>& texcoords, int layout_location=1);
	virtual void SetTexcoords(const std::vector<Eigen::Vector2f>& texcoords, int layout_location = 1); 

private:
	GLuint VBO_texcoord;
	GLuint textureID;
};


class RenderObjectMesh : virtual public SimpleRenderObject
{
public:
	RenderObjectMesh();
	RenderObjectMesh(const RenderObjectMesh& _) = delete;
	RenderObjectMesh& operator=(const RenderObjectMesh& _) = delete;
	virtual ~RenderObjectMesh();

	virtual RENDER_OBJECT_TYPE GetType() const { return RENDER_OBJECT_MESH; }
	virtual void DrawWhole(SimpleShader& shader) const;

	virtual void SetColors(const Eigen::Matrix<float, 3, -1, Eigen::ColMajor>& colors, int layout_location=2);

	virtual void SetColors(const std::vector<Eigen::Vector3f> &_colors, int layout_location=2); 
private:
	GLuint VBO_color;
};


class BallStickObject
{
public:
	BallStickObject(
		const MeshEigen& ballObj, const MeshEigen& stickObj,
		const std::vector<Eigen::Vector3f>& balls, 
		const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& sticks,
		float ballSize, float StickSize, const Eigen::Vector3f& color);

	// As point clouds with per-point size and color 
	BallStickObject( 
		const MeshEigen& ballObj, 
		const std::vector<Eigen::Vector3f>& balls, 
		const std::vector<float> sizes, 
		const std::vector<Eigen::Vector3f>& colors
	);

	BallStickObject(
		const MeshEigen& ballObj, const MeshEigen& stickObj,
		const std::vector<Eigen::Vector3f>& balls,
		const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& sticks,
		float ballSize, float StickSize, const std::vector<Eigen::Vector3f>& color);

	BallStickObject() = delete;
	BallStickObject(const BallStickObject& _) = delete;
	BallStickObject& operator=(const BallStickObject& _) = delete;
	virtual ~BallStickObject();
	void deleteObjects(); 
	virtual void Draw(SimpleShader& shader);
private:
	std::vector<RenderObjectColor*> objectPtrs;
};