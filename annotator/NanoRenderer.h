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
		: Canvas(parent, samples, has_depth, has_stencil, clear) 
	{
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

		m_width = width;
		m_height = height; 
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

	virtual void SetRenderObjects(std::vector<ref<RenderObject>> render_objects) {
		for (int i = 0; i < render_objects.size(); ++i)
		{
			render_objects[i]->SetRenderPass(m_render_pass);
			render_objects[i]->SetProj(m_proj);
			render_objects[i]->SetView(m_view);
		}
		m_render_objects = render_objects;
	}

	virtual void AddRenderObject(ref<RenderObject> render_object)
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
	int m_width; 
	int m_height; 
};


class ArcballCanvas : public StaticCanvas {
public:
	ArcballCanvas(Widget *parent, const int& width, const int& height, const float& fx, const float& fy, const float& cx, const float& cy, const bool is_pinhole=false) : 
		StaticCanvas(parent, width, height, fx, fy, cx, cy,is_pinhole)
	{
		m_rot_center = Eigen::Vector3f(0, 0, 0);
		left_click_time = 0.0;
	}

	void SetRotCenter(const Eigen::Vector3f& rot_center) { m_rot_center = rot_center; }

	Eigen::Vector3f GetArcballCoord(const Eigen::Vector2f& planeCoord,
		const Eigen::Vector3f& front,
		const Eigen::Vector3f& up,
		const Eigen::Vector3f& right);

	void SetRenderObjects(std::vector<ref<RenderObject>> render_objects) override
	{
		for (int i = 0; i < render_objects.size(); ++i)
		{
			render_objects[i]->SetRenderPass(m_render_pass);
			render_objects[i]->SetProj(m_proj);
			Matrix4f T_nano(0);
			for (int r = 0; r < 4; ++r)
			{
				for (int c = 0; c < 4; ++c)
				{
					T_nano.m[r][c] = m_viewRT(c, r);
				}
			}
			render_objects[i]->SetView(T_nano);
		}
		m_render_objects = render_objects;
	}

	void UpdateViewport();

	void AddRenderObject(ref<RenderObject> render_object) override
	{
		render_object->SetRenderPass(m_render_pass);
		render_object->SetProj(m_proj);
		Matrix4f T_nano(0);
		for (int r = 0; r < 4; ++r)
		{
			for (int c = 0; c < 4; ++c)
			{
				T_nano.m[r][c] = m_viewRT(c, r);
			}
		}
		render_object->SetView(m_view * T_nano);
		m_render_objects.push_back(render_object);
	}

	Eigen::Vector3f Project2Arcball(Eigen::Vector2f& p); 

	Eigen::Matrix4f RotateAroundCenterAxis(const float& _theta, const Eigen::Vector3f& _w, const Eigen::Vector3f& _rot_center);

	void RotateViewport(const Vector2i &p, const Vector2i &rel);

	void TranslateViewport(const Vector2i &p, const Vector2i &rel);

	void ZoomViewport(const Vector2i &p, const Vector2f &rel);

	virtual bool scroll_event(const Vector2i &p, const Vector2f &rel) override {
		ZoomViewport(p, rel);
		return true;
	}

	virtual bool mouse_drag_event(const Vector2i &p, const Vector2i &rel, int button, int modifiers) override;

	virtual bool mouse_button_event(const Vector2i &p, int button, bool down, int modifiers) override
	{
		if (button == GLFW_MOUSE_BUTTON_MIDDLE && down)
		{
			canvas_state = (canvas_state + 1) % 2; 
		}
		return true; 
	}

	virtual bool keyboard_event(int key, int scancode, int action, int modifiers) override; 

	virtual bool keyboard_character_event(unsigned int codepoint);

	
	void setViewRT(const Eigen::Matrix4f& view)
	{
		m_viewRT =view; 
		UpdateViewport();
	}

	int canvas_state = 1; // set 1 to show tool panels. set 0 to hide them. Default 1. 

	void SetExtrinsic(const Eigen::Vector3f& _pos, const Eigen::Vector3f& _up, const Eigen::Vector3f& _center);
	void SetExtrinsic(const Eigen::Matrix3f& R, const Eigen::Vector3f& T);
private:
	Eigen::Vector3f m_rot_center;
	Eigen::Matrix4f m_viewRT = Eigen::Matrix4f::Identity();
	Eigen::Vector3f pos;
	Eigen::Vector3f up;
	Eigen::Vector3f front; 
	Eigen::Vector3f right; 
	Eigen::Vector3f center; 
	Eigen::Matrix3f R; 
	Eigen::Matrix3f K;
	Eigen::Vector3f T; 
	double left_click_time; 
};


enum test_enum {
	PIG_0 = 0, 
	PIG_1, 
	PIG_2, 
	PIG_3
};

static Screen *screen = nullptr;
//static Screen *screen2 = nullptr;

class NanoRenderer
{
public:
	NanoRenderer();
	~NanoRenderer();

	void Init(const int& window_width, const int& window_height, 
		float fx = 400.f, float fy = 400.f, float cx = -1.f, float cy = -1.f, float arcball_depth = 2.0f, bool is_pinhole=false, std::string result_dir="");
	void Draw();
	bool Pause() { return m_pause; }
	void Stop();
	bool ShouldClose() 
	{
		return glfwWindowShouldClose(window); 
	}

	void ClearRenderObjects(); 

	ref<RenderObject> CreateRenderObject(const std::string& name, const std::string& vs, const std::string& fs) 
	{
		ref<RenderObject> render_object = new RenderObject(name, vs, fs, Shader::BlendMode::None);
		m_render_objects.emplace(std::make_pair(name, render_object));
		m_canvas->AddRenderObject(render_object);
		return render_object;
	}
	void AddRenderObject(const std::string& name, ref<RenderObject>& render_object) 
	{
		m_render_objects.emplace(std::make_pair(name, render_object));
		m_canvas->AddRenderObject(render_object);
	}
	std::map<std::string, ref<RenderObject>> GetRenderObjects() { return m_render_objects; }
	ref<RenderObject> GetRenderObject(const std::string& name) 
	{ 
		const auto& it = m_render_objects.find(name);
		if (it == m_render_objects.end())
			throw std::runtime_error("Renderer::GetRenderObject: Cannot find render object " + name + "!!!\n");
		return m_render_objects[name];
	}

	void CreateRenderImage(const std::string& name, const Vector2i& size, const Vector2i& pos);
	void SetRenderImage(const std::string& name, const cv::Mat& img);

	ref<OffscreenRenderObject> CreateOffscreenRenderObject
	(
		const std::string& name, const std::string& vs, const std::string& fs, 
		const int& width = 400, const int& height = 400, const float& fx = 400, const float& fy = 400,
		const float& cx = 400, const float& cy = 400, const int& tex_num = 1, const int& render_float_values = true, const bool is_pinhole=false) {
		ref<OffscreenRenderObject> render_object 
			= new OffscreenRenderObject(name, vs, fs, width, height, fx, fy, cx, cy, tex_num, render_float_values, is_pinhole);
		m_offscreen_render_objects.emplace(std::make_pair(name, render_object));
		return render_object;
	}
	ref<OffscreenRenderObject> GetOffscreenRenderObject(const std::string& name) 
	{
		const auto& it = m_offscreen_render_objects.find(name);
		if (it == m_offscreen_render_objects.end())
			throw std::runtime_error("Renderer::GetRenderObject: Cannot find render object " + name + "!!!\n");
		return m_offscreen_render_objects[name];
	}

	void UpdateCanvasView(const Eigen::Matrix4f& view)
	{
		m_canvas->setViewRT(view); 
		m_canvas->UpdateViewport();
	}

	void ApplyCanvasView()
	{
		m_canvas->UpdateViewport();
	}

	void SetCanvasExtrinsic(const Eigen::Matrix3f &_R, const Eigen::Vector3f &_T); 
	void SetCanvasExtrinsic(const Eigen::Vector3f &_pos, const Eigen::Vector3f &_up, const Eigen::Vector3f &_center); 

	bool m_save_screen = false;
	bool m_pause = false;
	std::string m_results_folder = "";
	int out_frameid = 0; 
	test_enum enumval = PIG_0;
	bool m_control_panel_visible = false;

	//Color colval = Color(0.5f, 0.5f, 0.7f, 1.f);
	bool m_state_read = false; 
	bool m_state_save_obj = false;
	bool m_state_save_state = false; 
	bool m_state_load_last = false; 
	bool m_state_load_this = false; 
	bool m_state_load_next = false;
	int  m_pig_num = 4;

	/// variables and widgets for parameter maniplation 
	Eigen::Vector3f m_pig_translation; 
	float m_pig_scale; 
	std::vector<Eigen::Vector3f> m_joint_pose; // 62 

	Eigen::Vector3f m_pig_translation_init; 
	float m_pig_scale_init; 
	std::vector<Eigen::Vector3f> m_joint_pose_init; 
	float m_overlay_transparency;

	std::vector<int> joints_for_optimize; // M 
	std::vector<Slider*> m_widget_sliders;  // 3 * M + 3 + 1 
	std::vector<TextBox*> m_widget_textboxes; // 3 * M + 3 + 1
	std::vector<Button*> m_widget_reset_buttons; 

	void set_joint_pose(const std::vector<Eigen::Vector3f>& _pose); 
	void set_pig_translation(const Eigen::Vector3f& trans); 
	void set_pig_scale(const float& scale); 

	Widget* basic_widgets; 
	Widget* tools; 
	Window* nanogui_window; 
	Window* image_window;

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
