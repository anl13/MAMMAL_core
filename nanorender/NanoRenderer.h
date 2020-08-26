#pragma once

/*
	src/example3.cpp -- C++ version of an example application that shows
	how to use nanogui in an application with an already created and managed
	GLFW context.

	NanoGUI was developed by Wenzel Jakob <wenzel.jakob@epfl.ch>.
	The widget drawing code is based on the NanoVG demo application
	by Mikko Mononen.

	All rights reserved. Use of this source code is governed by a
	BSD-style license that can be found in the LICENSE.txt file.
*/

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
#include <algorithm>

using namespace nanogui;

#include <nanogui/screen.h>
#include <nanogui/layout.h>
#include <nanogui/window.h>
#include <nanogui/button.h>
#include <nanogui/canvas.h>
#include <nanogui/shader.h>
#include <nanogui/renderpass.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <math.h>
#include <map>
#include <Eigen/Eigen>
#include "shader_file.h"
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/point_types.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "../utils/safe_call.hpp"
#include "RenderObject.h"

#if defined(_WIN32)
#  if defined(APIENTRY)
#    undef APIENTRY
#  endif
#  include <windows.h>
#endif

#ifdef WIN32
#undef min
#undef max
#endif

class StaticCanvas : public Canvas {
public:
	StaticCanvas(Widget *parent, const int& width, const int& height, 
		const float& fx, const float& fy, 
		const float& cx, const float& cy,
		const bool is_pinhole = false,
		const int samples = 1, const bool has_depth = true, const bool has_stencil = false, const bool clear = true) 
		: Canvas(parent, samples, has_depth, has_stencil, clear) {
		set_size(Vector2i(width, height));
		m_proj = CalcGLProjectionMatrix(width, height, fx, fy, cx, cy);
		//// convert default GL viewport to default pinhole camera viewport (for pinhole camera only)
		if (is_pinhole)
		{
			m_view = Matrix4f::look_at(Vector3f(0, 0, 0), Vector3f(0, 0, -1), Vector3f(0, 1, 0));
			Matrix4f cam2gl = Matrix4f::rotate(Vector3f(1, 0, 0), M_PI); 
			m_view = m_view * cam2gl;
		}
		else {
			// anliang; 20200515: use default I matrix. 
			m_view = Matrix4f::identity();
		}
	}

	Matrix4f CalcGLProjectionMatrix(const int& width, const int& height, const float& fx, const float& fy, const float& cx, const float& cy)
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

	std::vector<ref<RenderObject>> GetRenderObjects() { return m_render_objects; }

	void SetRenderObjects(std::vector<ref<RenderObject>> render_objects) {
		for (int i = 0; i < render_objects.size(); ++i)
		{
			render_objects[i]->SetRenderPass(m_render_pass);
			render_objects[i]->SetProj(m_proj);
			render_objects[i]->SetView(m_view);
		}
		m_render_objects = render_objects;
	}

	void AddRenderObject(ref<RenderObject> render_object)
	{
		render_object->SetRenderPass(m_render_pass);
		render_object->SetProj(m_proj);
		render_object->SetView(m_view);
		m_render_objects.push_back(render_object);
	}

	virtual void draw_contents() override {
		for (auto& it : m_render_objects) {
			it->Draw();
		}
	}

	void clear_objects() {
		m_render_objects.clear(); 
	}
protected:
	std::vector<ref<RenderObject>> m_render_objects;
	Matrix4f m_proj, m_view;
};


class ArcballCanvas : public StaticCanvas {
public:
	ArcballCanvas(Widget *parent, const int& width, const int& height, const float& fx, const float& fy, const float& cx, const float& cy, const bool is_pinhole=false) : 
		StaticCanvas(parent, width, height, fx, fy, cx, cy,is_pinhole) {
		m_rot_center = Eigen::Vector3f(0, 0, 1);
	}

	void SetRotCenter(const Eigen::Vector3f& rot_center) { m_rot_center = rot_center; }

	void UpdateViewport(const Eigen::Matrix4f& deltaT)
	{
		m_viewRT = deltaT * m_viewRT;
		// convert to nanogui::Matrix4f, in which the memory layout is "column major"
		Matrix4f T_nano(0);
		for (int r = 0; r < 4; ++r)
		{
			for (int c = 0; c < 4; ++c)
			{
				T_nano.m[r][c] = m_viewRT(c,r);
			}
		}

		for (auto& it : m_render_objects)
		{
			Matrix4f new_view = m_view * T_nano;
			it->SetView(new_view);
		}
	}

	Eigen::Vector3f Project2Arcball(Eigen::Vector2f& p)
	{
		if (p.norm() < 1.f){
			return Eigen::Vector3f(p[0], p[1], -std::sqrtf(1 - p.squaredNorm()));
		}
		else{
			return Eigen::Vector3f(p.normalized()[0], p.normalized()[1], 0.f);
		}
	}

	Eigen::Matrix4f RotateAroundCenterAxis(const float& _theta, const Eigen::Vector3f& _w, const Eigen::Vector3f& _rot_center)
	{
		const float theta = _theta;
		const Eigen::Vector3f rot_center = _rot_center;
		Eigen::Vector3f w = _w;
		Eigen::Matrix4f T; T.setIdentity();

		if (std::fabsf(theta) < FLT_EPSILON) return T;
		if (w.isZero()) return T;

		w = w.normalized();		

		Eigen::Vector3f v = rot_center.cross(w);
		Eigen::AngleAxisf psi(theta, w);
		Eigen::Vector3f rho = theta * v;

		T.topLeftCorner(3, 3) = psi.matrix();

		Eigen::Matrix3f wwT;
		for (int r = 0; r < 3; ++r)
		{
			for (int c = 0; c < 3; ++c)
			{
				wwT(r, c) = w(r) * w(c);
			}
		}

		Eigen::Matrix3f w_hat; w_hat.setZero();
		w_hat(0, 1) = -w(2);
		w_hat(0, 2) = w(1);
		w_hat(1, 2) = -w(0);
		w_hat(1, 0) = w(2);
		w_hat(2, 0) = -w(1);
		w_hat(2, 1) = w(0);
		T.topRightCorner(3, 1) = (sinf(theta) / theta * Eigen::Matrix3f::Identity() + (1 - sinf(theta) / theta) * wwT + (1 - cosf(theta)) / theta * w_hat) * rho;

		return T;
	}

	void RotateViewport(const Vector2i &p, const Vector2i &rel)
	{
		const int canvas_width = size()[0];
		const int canvas_height = size()[1];
		//const float pixel_radius = std::min(canvas_height, canvas_width) / 2;
		const float pixel_radius = std::min(canvas_height, canvas_width) / 2;

		Vector2i pe = p + rel;
		Eigen::Vector2f pBegin((p[0] - canvas_width / 2) / pixel_radius, (p[1] - canvas_height / 2) / pixel_radius);
		Eigen::Vector2f pEnd((pe[0] - canvas_width / 2) / pixel_radius, (pe[1] - canvas_height / 2) / pixel_radius);

		Eigen::Vector3f vBegin = Project2Arcball(pBegin);
		Eigen::Vector3f vEnd = Project2Arcball(pEnd);
		
		const float theta = std::acosf(vBegin.dot(vEnd)); if (isnan(theta)) return;
		Eigen::Vector3f w = (vBegin.cross(vEnd)).normalized();
		Eigen::Matrix4f deltaT = RotateAroundCenterAxis(theta, w, m_rot_center);
		UpdateViewport(deltaT);
	}

	void TranslateViewport(const Vector2i &p, const Vector2i &rel) {
		float speed = 1e-3f;
		float x = rel[0] * speed;
		float y = rel[1] * speed;
		Eigen::Matrix4f deltaT; deltaT.setIdentity();
		deltaT.topRightCorner(3, 1) = Eigen::Vector3f(x,y,0);
		UpdateViewport(deltaT);
	}

	void ZoomViewport(const Vector2i &p, const Vector2f &rel)
	{
		const float speed = 1e-1f;
		float shift = rel[1] * speed;
		Eigen::Matrix4f deltaT; deltaT.setIdentity();
		deltaT(2,3) += shift;
		UpdateViewport(deltaT);
	}

	virtual bool mouse_drag_event(const Vector2i &p, const Vector2i &rel, int button, int modifiers) override {

		if (button == GLFW_MOUSE_BUTTON_LEFT + 1){
			Vector2i newrel = rel;
			if (m_rot_center[2] < 0) newrel[0] *= -1;
			RotateViewport(p, newrel);
		}
		if (button == GLFW_MOUSE_BUTTON_RIGHT + 1){
			Vector2i newrel = rel;
			newrel[0] *= 1; 
			newrel[1] *= -1; 
			TranslateViewport(p, newrel);
		}
		return true;
	}

	virtual bool scroll_event(const Vector2i &p, const Vector2f &rel) override {
		ZoomViewport(p, rel);
		return true;
	}
	
	void setViewRT(const Eigen::Matrix4f& view)
	{
		m_viewRT = Eigen::Matrix4f::Identity(); 
		UpdateViewport(view);
	}
private:
	Eigen::Vector3f m_rot_center;
	Eigen::Matrix4f m_viewRT = Eigen::Matrix4f::Identity();
};


enum test_enum {
	Item1 = 0,
	Item2,
	Item3
};

static Screen *screen = nullptr;
static Screen *screen2 = nullptr;

class NanoRenderer
{
public:
	NanoRenderer();
	~NanoRenderer();

	void Init(const int& window_width, const int& window_height, 
		float fx = 400.f, float fy = 400.f, float cx = -1.f, float cy = -1.f, float arcball_depth = 2.0f, bool is_pinhole=false);
	void Draw();
	bool Pause() { return m_pause; }
	void Stop();
	bool ShouldClose() {
		return glfwWindowShouldClose(window); }

	void ClearRenderObjects(); 

	ref<RenderObject> CreateRenderObject(const std::string& name, const std::string& vs, const std::string& fs) {
		ref<RenderObject> render_object = new RenderObject(name, vs, fs, Shader::BlendMode::None);
		m_render_objects.emplace(std::make_pair(name, render_object));
		m_canvas->AddRenderObject(render_object);
		return render_object;
	}
	void AddRenderObject(const std::string& name, ref<RenderObject>& render_object) {
		m_render_objects.emplace(std::make_pair(name, render_object));
		m_canvas->AddRenderObject(render_object);
	}
	std::map<std::string, ref<RenderObject>> GetRenderObjects() { return m_render_objects; }
	ref<RenderObject> GetRenderObject(const std::string& name) { 
		const auto& it = m_render_objects.find(name);
		if (it == m_render_objects.end())
			throw std::runtime_error("Renderer::GetRenderObject: Cannot find render object " + name + "!!!\n");
		return m_render_objects[name];
	}

	void CreateRenderImage(const std::string& name, const Vector2i& size, const Vector2i& pos);
	void SetRenderImage(const std::string& name, const cv::Mat& img);

	ref<OffscreenRenderObject> CreateOffscreenRenderObject(
		const std::string& name, const std::string& vs, const std::string& fs, 
		const int& width = 400, const int& height = 400, const float& fx = 400, const float& fy = 400,
		const float& cx = 400, const float& cy = 400, const int& tex_num = 1, const int& render_float_values = true, const bool is_pinhole=false) {
		ref<OffscreenRenderObject> render_object 
			= new OffscreenRenderObject(name, vs, fs, width, height, fx, fy, cx, cy, tex_num, render_float_values, is_pinhole);
		m_offscreen_render_objects.emplace(std::make_pair(name, render_object));
		return render_object;
	}
	ref<OffscreenRenderObject> GetOffscreenRenderObject(const std::string& name) {
		const auto& it = m_offscreen_render_objects.find(name);
		if (it == m_offscreen_render_objects.end())
			throw std::runtime_error("Renderer::GetRenderObject: Cannot find render object " + name + "!!!\n");
		return m_offscreen_render_objects[name];
	}

	void UpdateCanvasView(const Eigen::Matrix4f& view)
	{
		m_canvas->UpdateViewport(view);
	}

	void ApplyCanvasView()
	{
		m_canvas->UpdateViewport(Eigen::Matrix4f::Identity());
	}

	bool m_save_screen = false;
	bool m_pause = false;
	std::string m_results_folder = "./screen/";
	int ivar = 0;
	double dvar = 3.1415926;
	float fvar = (float)dvar;
	test_enum enumval = Item2;
	Color colval = Color(0.5f, 0.5f, 0.7f, 1.f);

	
private:
	GLFWwindow* window = nullptr;
	ref<ArcballCanvas> m_canvas = nullptr;
	std::map<std::string, ref<RenderObject>> m_render_objects;
	std::map<std::string, ref<OffscreenRenderObject>> m_offscreen_render_objects;
	std::map<std::string, ref<ImageView>> m_render_images;
	int m_window_width, m_window_height;
	cv::Mat m_screen_image;
	ref<Screen> m_offscreen;
};
