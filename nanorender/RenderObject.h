#pragma once

// GLFW
//
#if defined(NANOGUI_GLAD)
#  if defined(NANOGUI_SHARED) && !defined(GLAD_GLAPI_EXPORT)
#    define GLAD_GLAPI_EXPORT
#  endif
#  include <glad/glad.h>
#else
#  if defined(__APPLE__)
#    define GLFW_INCLUDE_GLCOREARB
#  else
#    define GL_GLEXT_PROTOTYPES
#  endif
#endif

#include <GLFW/glfw3.h>
#include <nanogui/nanogui.h>
#include <iostream>

using namespace nanogui;


#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/point_types.h>
#include "../utils/safe_call.hpp"

#include <nanogui/shader.h>
#include <nanogui/renderpass.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <map>
#include "shader_file.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>


struct CudaSurfaceObject 
{
	cudaChannelFormatDesc channel_desc;
	cudaArray_t cuarray;
	cudaSurfaceObject_t surf;
	int width = 0, height = 0;
};

struct CudaTextureObject
{
	cudaChannelFormatDesc channel_desc;
	cudaArray_t cuarray;
	cudaTextureObject_t tex;
	int width = 0, height = 0;
};

class RenderObject : public Shader {
public:
	RenderObject(
		const std::string &name,
		const std::string &vertex_shader,
		const std::string &fragment_shader,
		BlendMode blend_mode = BlendMode::None) :
		Shader(name, vertex_shader, fragment_shader, blend_mode)
	{
		InitBuffers(1);
	}

	~RenderObject() {
		for(int i = 0; i < m_cuda_gl_resources.size(); ++i)
			cudaSafeCall(cudaGraphicsUnregisterResource(m_cuda_gl_resources[i]));
		for(int i = 0; i < m_cuda_surf_resources.size(); ++i)
			cudaSafeCall(cudaGraphicsUnregisterResource(m_cuda_surf_resources[i]));
	}

	void SetRenderPass(RenderPass* render_pass) { m_render_pass = render_pass; }

	void SetIndices(const std::vector<unsigned int>& indices)
	{
		set_buffer("indices", VariableType::UInt32, { indices.size() }, indices.data());
	}

	void SetTexCoords(const std::vector<float2>& texCoords)
	{
		set_buffer("texture_coords", VariableType::Float32, {texCoords.size(), 2}, texCoords.data());
	}

	void SetIndices(const ref<RenderObject> render_object)
	{
		const auto& buffers = render_object->m_buffers;
		const auto& it = buffers.find("indices");
		if(it == buffers.end())
			throw std::runtime_error(
				"RenderObject::SetIndices: Source Render Object " + render_object->name() + " has no indices buffer!!!");

		m_buffers.find("indices")->second = it->second;
	}


	void SetBuffer(const std::string& buf_name, const std::vector<float4>& vertex_buffer)
	{
		if (m_buffers.find(buf_name) == m_buffers.end())
		{
			printf("RenderObject::SetBuffer: Cannot find %s buffer in render object %s!!!\n", buf_name.c_str(), m_name.c_str());
		}

		set_buffer(buf_name, VariableType::Float32, { vertex_buffer.size(), 4}, vertex_buffer.data());
	}

	void SetBuffer(const std::string& buf_name, const pcl::gpu::DeviceArray<float4>& vertex_buffer);

	// share the "buf_name" buffer with another render object
	void SetBuffer(const std::string& buf_name, const ref<RenderObject> render_object)
	{
		const auto& buffers = render_object->m_buffers;
		const auto& it = buffers.find(buf_name);
		if(it == buffers.end())
			throw std::runtime_error(
				"Source Render Object " + render_object->name() + " has no " + buf_name + " buffer!!!");

		m_buffers.find(buf_name)->second = it->second;
	}

	std::map<std::string, pcl::gpu::DeviceArray<float4>> MapBuffers();
	void UnmapBuffers(const int valid_vnum = -1);

	template <typename Array> void SetUniform(const std::string& name, const Array &value) {
		set_uniform(name, value);
	}

	inline Eigen::Matrix4f _LookAt(const Eigen::Vector3f& _pos, const Eigen::Vector3f& _target, const Eigen::Vector3f& _up)
	{
		const Eigen::Vector3f direct = (_pos - _target).normalized();
		const Eigen::Vector3f right = (_up.cross(direct)).normalized();
		const Eigen::Vector3f up = (direct.cross(right)).normalized();

		Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
		Eigen::Matrix3f R = mat.block<3, 3>(0, 0);
		R.row(0) = right.transpose();
		R.row(1) = up.transpose();
		R.row(2) = direct.transpose();
		mat.block<3, 3>(0, 0) = R;
		mat.block<3, 1>(0, 3) = R * (-_pos);

		return mat;
	}

	Eigen::Matrix4f _calcRenderExt(const Eigen::Vector3f& _pos, const Eigen::Vector3f& _up, const Eigen::Vector3f& _center)
	{
		Eigen::Vector3f pos = _pos;
		Eigen::Vector3f up = _up;
		Eigen::Vector3f center = _center;

		Eigen::Vector3f front = (pos - center).normalized();
		Eigen::Vector3f right = (front.cross(up)).normalized();
		up = (right.cross(front)).normalized();

		Eigen::Matrix4f viewMat = _LookAt(pos, center, up);
		return viewMat;
	}

	Eigen::Matrix4f _calcRenderExt(const Eigen::Matrix3f& R, const Eigen::Vector3f& T)
	{
		Eigen::Vector3f front = -R.row(2).transpose();
		Eigen::Vector3f up = -R.row(1).transpose();
		Eigen::Vector3f pos = -R.transpose() * T;
		Eigen::Vector3f center = pos - 1.0f*front;
		return _calcRenderExt(pos, up, center);
	}

	nanogui::Matrix4f _eigen2nanoM4f(const Eigen::Matrix4f& mat)
	{
		nanogui::Matrix4f M;
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				M.m[j][i] = mat(i, j);
			}
		}
		return M;
	}

	void _SetViewByCameraRT(const Eigen::Matrix3d& R, const Eigen::Vector3d& T)
	{
		Eigen::Matrix4f view_eigen = _calcRenderExt(R.cast<float>(), T.cast<float>());
		nanogui::Matrix4f view_nano = _eigen2nanoM4f(view_eigen);
		SetView(view_nano);
	}

	void SetView(const Matrix4f& view)
	{
		m_view = view;
		Matrix4f mvp = m_proj * m_view * m_model;
		SetUniform("mvp", mvp);
	}

	void SetProj(const Matrix4f& proj)
	{
		m_proj = proj;
		Matrix4f mvp = m_proj * m_view * m_model;
		SetUniform("mvp", mvp);
	}

	Matrix4f GetModelRT() const { return m_model; }
	void SetModelRT(const Matrix4f& modelRT) {
		m_model = modelRT;
		Matrix4f mvp = m_proj * m_view * m_model;
		SetUniform("mvp", mvp);
	}

	bool HasBuffer(const std::string& name)
	{
		auto it = m_buffers.find(name);
		if (it == m_buffers.end())
			return false;
		else
			return true;
	}

	void UploadModelRT() {
		if(HasBuffer("model"))
			SetUniform("model", m_model);
	}

	void UploadViewRT() {
		if(HasBuffer("view"))
			SetUniform("view", m_view);
	}

	// texture operations
	void SetTexture(
		const std::string& tex_name, 
		const Texture::PixelFormat& pf, const Texture::ComponentFormat& cf, const Vector2i& size, 
		const void* data,
		const Texture::InterpolationMode min_interpolation_mode = Texture::InterpolationMode::Bilinear,
		const Texture::InterpolationMode mag_interpolation_mode = Texture::InterpolationMode::Bilinear,
		const Texture::WrapMode wrap_mode = Texture::WrapMode::ClampToEdge,
		uint8_t samples = 1,
		uint8_t flags = (uint8_t)Texture::TextureFlags::ShaderRead)
	{
		Texture* nanotex = new Texture(pf, cf, size);
		nanotex->upload((uint8_t*)data);
		set_texture(tex_name, nanotex);
	}
	void SetTexture(const std::string& tex_name, const CudaSurfaceObject& cusurf);
	std::map<std::string, CudaSurfaceObject> MapTextures();
	void UnmapTextures();

	void Draw() {

		UploadModelRT();
		UploadViewRT();

		begin();
		if (m_buffers["indices"].shape[0] == 0)
		{
			draw_array(Shader::PrimitiveType::Point, 0, m_buffers["positions"].shape[0], false);
		}
		else
		{
			draw_array(Shader::PrimitiveType::Triangle, 0, m_buffers["indices"].shape[0], true);
		}

		end();
	}

protected:
	void InitBuffers(const int& vnum);
	void InitCudaMapping();
	Matrix4f m_model = Matrix4f::identity();
	Matrix4f m_view = Matrix4f::look_at(Vector3f(0,0,0), Vector3f(0,0,-1), Vector3f(0,0,1));
	Matrix4f m_proj = Matrix4f::perspective(float(25 * M_PI / 180),0.1f,20.f);

public:
	std::vector<cudaGraphicsResource_t> m_cuda_gl_resources;
	std::vector<cudaGraphicsResource_t> m_cuda_surf_resources;
	// vertex buffers mapped from OpenGL to CUDA
	std::map<std::string, pcl::gpu::DeviceArray<float4>> m_vertex_buffers;
	std::map<std::string, CudaSurfaceObject> m_cuda_surfaces;
	bool m_mapped = false;
};


class OffscreenRenderObject : public RenderObject {
public:
	OffscreenRenderObject(
		const std::string &name,
		const std::string &vertex_shader,
		const std::string &fragment_shader,
		const int width = 200, const int height = 200, const float fx = 200, const float fy = 200,
		const float cx = 100, const float cy = 100,
		const int render_tex_num = 1,
		const bool render_float_values = false,
		const bool is_pinhole = false, 
		BlendMode blend_mode = BlendMode::None)
		: RenderObject(name, vertex_shader, fragment_shader)
	{
		Init(width, height, fx, fy, cx, cy, render_tex_num, render_float_values,is_pinhole);
	}

	~OffscreenRenderObject() {
		for(int i = 0; i < m_render_texture_num; ++i)
			cudaSafeCall(cudaGraphicsUnregisterResource(m_cuda_gl_resources_offscreen[i]));
	}

	void Init(
		const int& width, const int& height, 
		const float& fx, const float& fy, 
		const float& cx, const float& cy,
		const int& render_tex_num, const bool& render_float_values, const bool is_pinhole=false) {
		m_width = width;
		m_height = height;
		m_proj = CalcGLProjectionMatrix(width, height, fx, fy, cx, cy);
		
		m_model = Matrix4f::identity();
		// convert default GL viewport to default pinhole camera viewport (for pinhole camera only)
		if (is_pinhole)
		{
			m_view = Matrix4f::look_at(Vector3f(0, 0, 0), Vector3f(0, 0, -1), Vector3f(0, 1, 0));
			Matrix4f cam2gl = Matrix4f::rotate(Vector3f(1, 0, 0), M_PI);
			m_view = m_view * cam2gl;
		}
		else {
			// anliang:20200515
			m_view = Matrix4f::identity();
		}

		m_render_texture_num = render_tex_num;
		m_render_float_values = render_float_values;
		const int texture_samples = 1;
		const Vector2i fb_size(width, height);

		///////////////////////////////////////////////
		// init color rbo for rendering RGBA32F texture
		std::vector<Texture*> color_textures(m_render_texture_num, nullptr);
		std::vector<Object*> color_objects(m_render_texture_num, nullptr);
		for (int i = 0; i < m_render_texture_num; ++i)
		{
			color_textures[i] = new Texture(
				Texture::PixelFormat::RGBA,
				m_render_float_values ? Texture::ComponentFormat::Float32 : Texture::ComponentFormat::UInt8,
				fb_size,
				Texture::InterpolationMode::Bilinear,
				Texture::InterpolationMode::Bilinear,
				Texture::WrapMode::ClampToEdge,
				texture_samples,
				Texture::TextureFlags::RenderTarget);

			color_objects[i] = color_textures[i];
		}
		

		///////////////////////////////////
		// init depth buffer and renderpass
		ref<Texture> depth_rbo = new Texture(
			Texture::PixelFormat::Depth,
			Texture::ComponentFormat::Float32,
			fb_size,
			Texture::InterpolationMode::Bilinear,
			Texture::InterpolationMode::Bilinear,
			Texture::WrapMode::ClampToEdge,
			texture_samples,
			Texture::TextureFlags::RenderTarget);

		m_render_pass = new RenderPass(
			color_objects, depth_rbo, nullptr);
		m_render_pass->set_clear_color(0, Color(0, 0, 0, 255));

		// init cuda mapping
		m_cuda_gl_resources_offscreen.resize(m_render_texture_num);
		for (int i = 0; i < m_render_texture_num; ++i)
		{
			cudaSafeCall(cudaGraphicsGLRegisterImage(
				&m_cuda_gl_resources_offscreen[i],
				color_textures[i]->renderbuffer_handle(),
				GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
		}

		SetRenderPass(m_render_pass);
		SetView(m_view);
		SetProj(m_proj);
	}

	Matrix4f CalcGLProjectionMatrix(const int& width, const int& height, 
		const float& fx, const float& fy, 
		const float& cx, const float& cy)
	{
		// default parameters
		float n = 0.1f;
		float f = 10.f;

		Matrix4f projMat(0);
		//projMat.m[0][0] = fx / cx; 
		//projMat.m[1][1] = fy / cy; 
		int w = width ;
		int h = height ; 
		projMat.m[0][0] = 2 * fx / w;
		projMat.m[1][1] = 2 * fy / h;
		projMat.m[2][0] = -(2 * cx / w - 1);
		projMat.m[2][1] = (2 * cy / h - 1);
		projMat.m[2][2] = (-(f + n)) / (f - n);
		projMat.m[3][2] = (-2 * f * n) / (f - n);
		projMat.m[2][3] = -1.f;

		return projMat;
	}

	std::vector<CudaTextureObject> MapRenderingResults() {

		cudaSafeCall(cudaGraphicsMapResources(m_render_texture_num, &m_cuda_gl_resources_offscreen[0]));

		m_color_textures.resize(m_render_texture_num);
		for (int i = 0; i < m_render_texture_num; ++i)
		{
			cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&m_color_textures[i].cuarray, m_cuda_gl_resources_offscreen[i], 0, 0));

			// initialize cudaResourceDesc
			cudaResourceDesc res_desc;
			memset(&res_desc, 0, sizeof(cudaResourceDesc));
			res_desc.resType = cudaResourceTypeArray;

			// initialize cudaTextureDesc
			cudaTextureDesc tex_desc;
			memset(&tex_desc, 0, sizeof(cudaTextureDesc));
			tex_desc.addressMode[0] = cudaAddressModeBorder;
			tex_desc.addressMode[1] = cudaAddressModeBorder;
			tex_desc.filterMode = cudaFilterModePoint;
			tex_desc.readMode = cudaReadModeElementType;
			tex_desc.normalizedCoords = 0;

			res_desc.res.array.array = m_color_textures[i].cuarray;
			cudaSafeCall(cudaCreateTextureObject(&m_color_textures[i].tex, &res_desc, &tex_desc, NULL));
		}

		return m_color_textures;
	}

	void UnmapRenderingResults() {
		cudaSafeCall(cudaGraphicsUnmapResources(m_render_texture_num, &m_cuda_gl_resources_offscreen[0]));
	}

	void DownloadRenderingResults(std::vector<cv::Mat> &cvimgs)
	{
		if(cvimgs.size() != m_render_texture_num)
			throw std::runtime_error("OffscreenRenderObject::Download: cvimgs size is not euqal to rendered texture number !!!\n");

		MapRenderingResults();
		for (int i = 0; i < m_render_texture_num; ++i)
		{
			cv::Mat& cvimg = cvimgs[i];
			if (m_render_float_values == false && (cvimg.channels() != 4 || cvimg.elemSize() != 4 * sizeof(char))) {
				throw std::runtime_error("OffscreenRenderObject::Download: input cvimg format is not RGBA UINT8 !!!\n");
			}

			if (m_render_float_values == true && (cvimg.channels() != 4 || cvimg.elemSize() != 4 * sizeof(float))) {
				throw std::runtime_error("OffscreenRenderObject::Download: input cvimg format is not RGBA 32F !!!\n");
			}

			if (cvimg.cols != m_width || cvimg.rows != m_height)
			{
				cv::resize(cvimg, cvimg, cv::Size(m_width, m_height));
				if(m_render_float_values)
					cvimg.setTo(cv::Vec4d(0, 0, 0, 0));
				else cvimg.setTo(cv::Vec4b(0, 0, 0, 0));
			}

			cudaSafeCall(cudaMemcpy2DFromArray(
				cvimg.data, cvimg.cols * cvimg.elemSize(),
				m_color_textures[i].cuarray, 0, 0, cvimg.cols*cvimg.elemSize(), cvimg.rows, cudaMemcpyDeviceToHost));
			cv::flip(cvimg, cvimg, 0);
		}
		UnmapRenderingResults();
	}

	void DirectDownloadRenderingResults(cv::Mat& cvimg)
	{
		if (m_render_float_values == false && (cvimg.channels() != 4 || cvimg.elemSize() != 4 * sizeof(char))) {
			throw std::runtime_error("OffscreenRenderObject::Download: input cvimg format is not RGBA UINT8 !!!\n");
		}

		if (cvimg.cols != m_width || cvimg.rows != m_height)
		{
			cv::resize(cvimg, cvimg, cv::Size(m_width, m_height));
			cvimg.setTo(cv::Vec4d(0, 0, 0, 0));
		}

		m_render_pass->begin();

		glReadPixels(0, 0, cvimg.cols, cvimg.rows, GL_BGRA, GL_UNSIGNED_BYTE, cvimg.data);
		cv::flip(cvimg, cvimg, 0);

		m_render_pass->end();
	}

	void DrawOffscreen()
	{
		m_render_pass->begin();
		Draw();
		m_render_pass->end();
	}

private:
	int m_width, m_height;
	bool m_render_float_values;
	int m_render_texture_num;
	ref<RenderPass> m_render_pass;
	std::vector<cudaGraphicsResource_t> m_cuda_gl_resources_offscreen;
	std::vector<CudaTextureObject> m_color_textures;
};
