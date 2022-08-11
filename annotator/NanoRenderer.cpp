#include "NanoRenderer.h"
#include <vector_functions.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream> 

#define parameter_scale 6.0f

NanoRenderer::NanoRenderer()
{
}

NanoRenderer::~NanoRenderer()
{
}

void NanoRenderer::Init(const int& window_width, const int& window_height,
	float fx, float fy, float cx, float cy, float arcball_depth /*= 2.0f*/, bool is_pinhole, std::string result_dir)
{
	m_results_folder = result_dir;
	m_window_width = window_width;
	m_window_height = window_height;
	std::cout << "Is pinhole: " << is_pinhole << std::endl;
	if (cx < 0 || cy < 0)
	{
		cx = window_width / 2;
		cy = window_height / 2;
	}

	////////////////////////////////////
	// initialize a glfw window manually
	glfwInit();
	glfwSetTime(0);

	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	glfwWindowHint(GLFW_SAMPLES, 0);
	glfwWindowHint(GLFW_RED_BITS, 8);
	glfwWindowHint(GLFW_GREEN_BITS, 8);
	glfwWindowHint(GLFW_BLUE_BITS, 8);
	glfwWindowHint(GLFW_ALPHA_BITS, 8);
	glfwWindowHint(GLFW_STENCIL_BITS, 8);
	glfwWindowHint(GLFW_DEPTH_BITS, 24);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	glfwWindowHint(GLFW_SCALE_TO_MONITOR, GL_FALSE);

	// Create a GLFWwindow object
	window = glfwCreateWindow(m_window_width, m_window_height, "MVF", nullptr, nullptr);
	if (window == nullptr) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return;
	}

	glfwMakeContextCurrent(window);

#if defined(NANOGUI_GLAD)
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		throw std::runtime_error("Could not initialize GLAD!");
	glGetError(); // pull and ignore unhandled errors like GL_INVALID_ENUM
#endif

	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	glViewport(0,0, width, height);
	glfwSwapInterval(0);
	glfwSwapBuffers(window);

	//////////////////////////////////////////////////////////////////
	// Create a nanogui screen and pass the glfw pointer to initialize
	screen = new Screen();
	
	//return; 

	screen->initialize(window, true);
	
	m_screen_image.create(m_window_height /* * screen->pixel_ratio()*/, m_window_width/* * screen->pixel_ratio()*/, CV_8UC4);

	glfwSetCursorPosCallback(window,
		[](GLFWwindow *, double x, double y) {
			screen->cursor_pos_callback_event(x, y);
		}
	);

	glfwSetMouseButtonCallback(window,
		[](GLFWwindow *, int button, int action, int modifiers) {
			screen->mouse_button_callback_event(button, action, modifiers);
		}
	);

	glfwSetKeyCallback(window,
		[](GLFWwindow *, int key, int scancode, int action, int mods) {
			screen->key_callback_event(key, scancode, action, mods);
		}
	);

	glfwSetCharCallback(window,
		[](GLFWwindow *, unsigned int codepoint) {
			screen->char_callback_event(codepoint);
		}
	);

	glfwSetDropCallback(window,
		[](GLFWwindow *, int count, const char **filenames) {
			screen->drop_callback_event(count, filenames);
		}
	);

	glfwSetScrollCallback(window,
		[](GLFWwindow *, double x, double y) {
			screen->scroll_callback_event(x, y);
		}
	);

	glfwSetFramebufferSizeCallback(window,
		[](GLFWwindow *, int width, int height) {
			screen->resize_callback_event(width, height);
		}
	);

	//////////////////////////////////////
	// create canvas for rendering results
	//m_canvas = new ArcballCanvas(screen, window_width, window_height, fx, fy);
	m_canvas = new ArcballCanvas(screen, width/3, height/3, fx, fy, cx/3, cy/3, is_pinhole);
	m_canvas->SetRotCenter(Eigen::Vector3f(0, 0, arcball_depth));
	m_canvas->set_background_color({ 100, 100, 100, 255 });
	m_canvas->set_draw_border(true);
	m_canvas->set_cursor(Cursor::Crosshair);

	// some fixed widgets 
	//tools = new Widget(screen);
	//tools->set_layout(new BoxLayout(Orientation::Vertical, Alignment::Minimum, 5, 5));

	//Button *b0 = new Button(tools, "Random Background");
	//b0->set_callback([this]() {
	//	m_canvas->set_background_color(
	//		Vector4i(rand() % 256, rand() % 256, rand() % 256, 255));
	//	});

	//Button *b1 = new Button(tools, "Pause");
	//b1->set_callback([this]() {
	//	m_pause = !m_pause;
	//	});

	// ------!!!!! important
	// 1. init embedded parameters 
	m_joint_pose.resize(62, Eigen::Vector3f::Zero());
	m_pig_scale = 1; 
	m_pig_translation = Eigen::Vector3f::Zero(); 
	m_joint_pose_init = m_joint_pose; 
	m_pig_scale = m_pig_scale_init; 
	m_pig_translation = m_pig_translation_init; 
	const std::vector<std::string> joint_names = {
		"Hips_0",
"Spine1_1",
"Spine2_2",
"Spine3_3",
"frontHips_4",
"R_f_Hips_5",
"R_f_clavicle_6",
"R_f_leg_7",
"R_f_knee_8",
"R_f_foot_9",
"R_f_Heel_10",
"R_f_toebase_11",
"R_f_toeend_12",
"L_f_Hips_13",
"L_f_clavicle_14",
"L_f_leg_15",
"L_f_knee_16",
"L_f_foot_17",
"L_f_Heel_18",
"L_f_toebase_19",
"L_f_toeend_20",
"neck1_21",
"neck2_22",
"head_23",
"R_eye_24",
"L_eye_25",
"R_ear0_26",
"R_ear1_27",
"R_ear2_28",
"R_ear3_29",
"R_ear4_30",
"L_ear0_31",
"L_ear1_32",
"L_ear2_33",
"L_ear3_34",
"L_ear4_35",
"jaw_36",
"jawend_37",
"R_Hips_38",
"R_b_leg_39",
"R_b_knee_40",
"R_b_foot_41",
"R_b_footbase_42",
"R_b_Heel_43",
"R_b_Ball_44",
"R_b_Toe_45",
"tail1_46",
"tail2_47",
"tail3_48",
"tail4_49",
"tail5_50",
"tail6_51",
"tail7_52",
"tail8_53",
"L_Hips_54",
"L_b_leg_55",
"L_b_knee_56",
"L_b_foot_57",
"L_b_footbase_58",
"L_b_Heel_59",
"L_b_Ball_60",
"L_b_Toe_61"
};
	joints_for_optimize = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 38, 39, 40, 41, 54, 55, 56, 57, 21, 22, 23
	};
	int N = joints_for_optimize.size(); 
	m_widget_sliders.resize(N * 3 + 4);
	m_widget_textboxes.resize(N * 3 + 4); 
	m_widget_reset_buttons.resize(N + 2); 
	// new widgets, for controling pose parameters

	//basic_widgets = new Window(screen, "Parameter Tuning"); 
	basic_widgets = new Widget(screen); 
	basic_widgets->set_position(Vector2i(0, 360)); 
	basic_widgets->set_height(720);
	basic_widgets->set_width(1400);
	
	basic_widgets->set_layout(new BoxLayout(Orientation::Vertical, Alignment::Minimum, 1, 1));

	int box_height = 20;
	
	// -- scale 
	{
		Widget *acombol = new Widget(basic_widgets);
		acombol->set_layout(new BoxLayout(Orientation::Horizontal, Alignment::Minimum, 1, 1));
		Label *name_label = new Label(acombol, "scale", "sans-bold", box_height);
		name_label->set_fixed_size(Vector2i(150, box_height));
		Label *axis_name_label = new Label(acombol, "", "sans-bold", box_height);
		axis_name_label->set_fixed_size(Vector2i(50, box_height)); 

		Button* leftbutton = new Button(acombol, "-");
		leftbutton->set_fixed_size(Vector2i(box_height, box_height));

		m_widget_sliders[0] = new Slider(acombol);
		float renorm_value = m_pig_scale - 1 + 0.5;
		m_widget_sliders[0]->set_value(renorm_value);
		m_widget_sliders[0]->set_fixed_size(Vector2i(150, box_height));

		Button* rightbutton = new Button(acombol, "+");
		rightbutton->set_fixed_size(Vector2i(box_height, box_height));

		leftbutton->set_callback([this]() {
			m_pig_scale -= 0.01;
			m_widget_sliders[0]->set_value(m_pig_scale -0.5);
			m_widget_textboxes[0]->set_value(std::to_string(m_pig_scale));
			});

		rightbutton->set_callback([ this]() {
			m_pig_scale += 0.01;
			m_widget_sliders[0]->set_value(m_pig_scale - 0.5);
			m_widget_textboxes[0]->set_value(std::to_string(m_pig_scale));
			});

		m_widget_textboxes[0] = new TextBox(acombol);
		m_widget_textboxes[0]->set_fixed_size(Vector2i(150, box_height));
		m_widget_textboxes[0]->set_value(std::to_string(m_pig_scale));
		m_widget_textboxes[0]->set_units("");
		m_widget_textboxes[0]->set_font_size(box_height);

		TextBox* p_box = m_widget_textboxes[0];
		m_widget_sliders[0]->set_callback([p_box, this](float value) {
			float normalized_value = value + 0.5;
			p_box->set_value(std::to_string(normalized_value));
			m_pig_scale = normalized_value;
			});

		m_widget_reset_buttons[0] = new Button(acombol, "Reset");
		m_widget_reset_buttons[0]->set_fixed_size(Vector2i(60, box_height)); 
		m_widget_reset_buttons[0]->set_callback([this]() {
			m_pig_scale = m_pig_scale_init;
			m_widget_sliders[0]->set_value(m_pig_scale - 0.5); 
			m_widget_textboxes[0]->set_value(std::to_string(m_pig_scale)); 
			});
	}
	// -- translation 
	{
		Widget *acombol = new Widget(basic_widgets);
		acombol->set_layout(new BoxLayout(Orientation::Horizontal, Alignment::Minimum, 1, 1));
		Label *name_label = new Label(acombol, "translation", "sans-bold", box_height);
		name_label->set_fixed_size(Vector2i(150, box_height));
		std::vector<std::string> axis_names = { "    x", "    y", "    z" };
		for (int axis = 0; axis < 3; axis++)
		{
			Label *axis_name_label = new Label(acombol, axis_names[axis], "sans-bold", box_height);
			axis_name_label->set_fixed_size(Vector2i(50, box_height));

			Button* leftbutton = new Button(acombol, "-");
			leftbutton->set_fixed_size(Vector2i(box_height, box_height));

			m_widget_sliders[axis+1] = new Slider(acombol);
			float renorm_value = m_pig_translation(axis) / parameter_scale + 0.5;
			m_widget_sliders[axis+1]->set_value(renorm_value);
			m_widget_sliders[axis+1]->set_fixed_size(Vector2i(150, box_height));

			Button* rightbutton = new Button(acombol, "+");
			rightbutton->set_fixed_size(Vector2i(box_height, box_height));

			leftbutton->set_callback([axis, this]() {
				m_pig_translation(axis) -= 0.01;
				m_widget_sliders[1 + axis]->set_value((m_pig_translation(axis) / parameter_scale + 0.5));
				m_widget_textboxes[1 + axis]->set_value(std::to_string(m_pig_translation(axis)));
				});

			rightbutton->set_callback([axis, this]() {
				m_pig_translation(axis) += 0.01;
				m_widget_sliders[1 + axis]->set_value((m_pig_translation(axis) / parameter_scale + 0.5));
				m_widget_textboxes[1 + axis]->set_value(std::to_string(m_pig_translation(axis)));
				});

			m_widget_textboxes[axis+1] = new TextBox(acombol);
			m_widget_textboxes[axis + 1]->set_fixed_size(Vector2i(150, box_height));
			m_widget_textboxes[axis + 1]->set_value(std::to_string(m_pig_translation(axis)));
			m_widget_textboxes[axis + 1]->set_units("");
			m_widget_textboxes[axis + 1]->set_font_size(box_height);

			TextBox *p_box = m_widget_textboxes[axis + 1];
			m_widget_sliders[axis+1]->set_callback([p_box, axis, this](float value) {
				float normalized_value = (value - 0.5) * parameter_scale;
				p_box->set_value(std::to_string(normalized_value));
				m_pig_translation(axis) = normalized_value;
				});
		}
		m_widget_reset_buttons[1] = new Button(acombol, "Reset");
		m_widget_reset_buttons[1]->set_fixed_size(Vector2i(60, box_height));
		m_widget_reset_buttons[1]->set_callback([this]() {
			m_pig_translation = m_pig_translation_init;
			for (int axis = 0; axis < 3; axis++)
			{
				m_widget_sliders[1 + axis]->set_value(m_pig_translation(axis) / parameter_scale + 0.5);
				m_widget_textboxes[1 + axis]->set_value(std::to_string(m_pig_translation(axis)));
			}
			});
	}
	// -- rotation 
	for (int i = 0; i < joints_for_optimize.size(); i++)
	{
		int joint_id = joints_for_optimize[i];
		Widget *acombol = new Widget(basic_widgets);
		acombol->set_layout(new BoxLayout(Orientation::Horizontal, Alignment::Minimum, 1, 1));

		Label *name_label = new Label(acombol, joint_names[joint_id], "sans-bold", box_height);
		name_label->set_fixed_size(Vector2i(150, box_height));
		std::vector<std::string> axis_names = { "aa  x", "aa  y", "aa  z" };
		if (joint_id == 0)
		{
			axis_names = { "eul z", "eul y", "eul x" };
		}
		for (int axis = 0; axis < 3; axis++)
		{
			Label *axis_name_label = new Label(acombol, axis_names[axis], "sans-bold", box_height);
			axis_name_label->set_fixed_size(Vector2i(50, box_height));

			Button* leftbutton = new Button(acombol, "-"); 
			leftbutton->set_fixed_size(Vector2i(box_height, box_height));


			m_widget_sliders[4+i*3+axis] = new Slider(acombol);
			float renorm_value = m_joint_pose[joint_id](axis) / parameter_scale + 0.5; 
			m_widget_sliders[4+i * 3 + axis]->set_value(renorm_value);
			m_widget_sliders[4+i * 3 + axis]->set_fixed_size(Vector2i(150, box_height));

			Button* rightbutton = new Button(acombol, "+");
			rightbutton->set_fixed_size(Vector2i(box_height, box_height));

			leftbutton->set_callback([i, joint_id, axis, this]() {
				m_joint_pose[joint_id](axis) -= 0.01;
				m_widget_sliders[4 + 3 * i + axis]->set_value((m_joint_pose[joint_id](axis) / parameter_scale + 0.5));
				m_widget_textboxes[4 + 3 * i + axis]->set_value(std::to_string(m_joint_pose[joint_id](axis)));
				});

			rightbutton->set_callback([i, joint_id, axis, this]() {
				m_joint_pose[joint_id](axis) += 0.01;
				m_widget_sliders[4 + 3 * i + axis]->set_value((m_joint_pose[joint_id](axis) / parameter_scale + 0.5));
				m_widget_textboxes[4 + 3 * i + axis]->set_value(std::to_string(m_joint_pose[joint_id](axis)));
				});

			m_widget_textboxes[4+i*3+axis] = new TextBox(acombol);
			m_widget_textboxes[4 + i * 3 + axis]->set_fixed_size(Vector2i(150, box_height));
			m_widget_textboxes[4 + i * 3 + axis]->set_value(std::to_string(m_joint_pose[joint_id](axis)));
			m_widget_textboxes[4 + i * 3 + axis]->set_units("");
			m_widget_textboxes[4 + i * 3 + axis]->set_font_size(box_height);

			TextBox* p_box = m_widget_textboxes[4 + i * 3 + axis];
			m_widget_sliders[4 + i * 3 + axis]->set_callback([p_box,joint_id, axis,this](float value) {
				float normalized_value = (value - 0.5) * parameter_scale;
				p_box->set_value(std::to_string(normalized_value));
				m_joint_pose[joint_id](axis) = normalized_value;
				});
		}

		m_widget_reset_buttons[2+i] = new Button(acombol, "Reset");
		m_widget_reset_buttons[2+i]->set_fixed_size(Vector2i(60, box_height));
		m_widget_reset_buttons[2+i]->set_callback([i,joint_id, this]() {
			std::cout << "reset " << i << std::endl; 
			m_joint_pose[joint_id] = m_joint_pose_init[joint_id];
			for (int axis = 0; axis < 3; axis++)
			{
				m_widget_sliders[4 + 3* i + axis]->set_value(m_joint_pose[joint_id](axis) / parameter_scale + 0.5);
				m_widget_textboxes[4 + 3* i + axis]->set_value(std::to_string(m_joint_pose[joint_id](axis)));
			}
			});
	}
	
	m_overlay_transparency = 0.0f; 

	// create high level control: 
	// determine framenum, pigid, 
	// give save folder, 
	// two buttons: read
	// and write obj, write state 
	bool enabled = true;
	FormHelper *gui = new FormHelper(screen);
	nanogui_window = gui->add_window(Vector2i(10, 10), "Control Panel");
	gui->add_group("Renderer Control");
	gui->add_variable("Save Folder", m_results_folder);

	gui->add_group("Choose Target");
	std::vector<std::string> items; 
	for (int i = 0; i < m_pig_num; i++)
	{
		std::string name = "pig" + std::to_string(i);
		items.push_back(name); 
	}
	gui->add_variable("Enumeration", enumval, enabled)->set_items(items);
	//gui->add_variable("Color", colval);
	gui->add_variable("Frame ID", out_frameid);
	
	gui->add_variable("Overlay Alpha", m_overlay_transparency);

	gui->add_group("Actions");
	gui->add_button("Read Original", [this]() { m_state_read = true; })
		->set_tooltip("Load fitted result from YOUR_RESULT_FOLDER/state/.");
	
	gui->add_button("Save State", [this]() {
		m_state_save_state = true; 
		});
	gui->add_button("Save Obj", [this]() {
		m_state_save_obj = true; 
		});
	gui->add_button("Load Labeled Last", [this]() {
		m_state_load_last = true; 
		})->set_tooltip("Load annotated pose of last frame if exist");
	gui->add_button("Load Labeled This", [this]() {
		m_state_load_this = true; 
		})->set_tooltip("Load annotated pose if exist");
	gui->add_button("Load Labeled Next", [this]() {
		m_state_load_next = true;
		})->set_tooltip("Load next annotated pose if exist");
	nanogui_window->set_position(Vector2i(5,45));


	nanogui_window->set_visible(m_control_panel_visible);
	basic_widgets->set_visible(m_control_panel_visible);
	//tools->set_visible(m_control_panel_visible);
}

void NanoRenderer::set_joint_pose(const std::vector<Eigen::Vector3f>& _pose)
{
	m_joint_pose = _pose; 
	m_joint_pose_init = _pose; 
	for (int i = 0; i < joints_for_optimize.size(); i++)
	{
		int joint_id = joints_for_optimize[i];

		for (int axis = 0; axis < 3; axis++)
		{
			float renorm_value = m_joint_pose[joint_id](axis) / parameter_scale + 0.5;
			m_widget_sliders[4 + i * 3 + axis]->set_value(renorm_value);
			m_widget_textboxes[4 + i * 3 + axis]->set_value(std::to_string(m_joint_pose[joint_id](axis)));
		}
	}
}

void NanoRenderer::set_pig_translation(const Eigen::Vector3f& trans)
{
	m_pig_translation = trans; 
	m_pig_translation_init = trans; 
	for (int axis = 0; axis < 3; axis++)
	{
		float renorm_value = trans(axis) / parameter_scale + 0.5;
		m_widget_sliders[1 + axis]->set_value(renorm_value); 
		m_widget_textboxes[1 + axis]->set_value(std::to_string(trans(axis))); 
	}
}

void NanoRenderer::set_pig_scale(const float& scale)
{
	m_pig_scale = scale; 
	m_pig_scale_init = scale; 
	float renorm_value = scale - 0.5; 
	m_widget_sliders[0]->set_value(renorm_value); 
	m_widget_textboxes[0]->set_value(std::to_string(scale)); 
}


void NanoRenderer::CreateRenderImage(const std::string& name, const Vector2i& size, const Vector2i& pos)
{
	image_window = new Window(screen, name);
	image_window->set_position(pos);
	image_window->set_layout(new GroupLayout(3));

	ref<ImageView> image_view = new ImageView(image_window);
	image_view->set_size(size);
	image_view->center();
	image_window->set_visible(true);

	m_render_images.emplace(name, image_view);
}


void NanoRenderer::SetRenderImage(const std::string& name, const cv::Mat& img)
{
	if (img.channels() != img.elemSize()){
		throw std::runtime_error("Renderer::SetWindowImage: Image format mismatch, not an N (N=1/3/4) Channels UNSIGNED_BYTE image !!!\n");
	}

	const auto& it = m_render_images.find(name);
	if (it == m_render_images.end())
		throw std::runtime_error("Renderer::SetWindowImage: Cannot find image window " + name + "!!!\n");

	ref<ImageView> image_view = it->second;
	const float monitor_pixel_ratio = screen->pixel_ratio();
	const Vector2i view_size = image_view->size();
	cv::Size display_img_size(view_size[0] * monitor_pixel_ratio, view_size[1] * monitor_pixel_ratio);
	cv::Mat display_img;
	cv::resize(img, display_img, display_img_size);

	switch (img.channels())
	{
	case 1: 
		cv::cvtColor(display_img, display_img, cv::COLOR_GRAY2RGBA);
	case 3:
		cv::cvtColor(display_img, display_img, cv::COLOR_RGB2RGBA);
	default:
		break;
	}

	cv::cvtColor(display_img, display_img, cv::COLOR_BGRA2RGBA);

	Texture *tex = new Texture(
		Texture::PixelFormat::RGBA,
		Texture::ComponentFormat::UInt8,
		Vector2i(display_img.cols, display_img.rows),
		Texture::InterpolationMode::Trilinear,
		Texture::InterpolationMode::Nearest);

	tex->upload(display_img.data);
	image_view->set_image(tex);
}


void NanoRenderer::Draw()
{
	static bool init_screen_drawcall = false;
	if (!init_screen_drawcall)
	{
		init_screen_drawcall = true;
		////////////////
		// render screen
		screen->set_visible(true);
		screen->perform_layout();
		screen->clear();
		screen->draw_all();
	}

	if (m_canvas->canvas_state == 1) m_control_panel_visible = true;
	else m_control_panel_visible = false;
	nanogui_window->set_visible(m_control_panel_visible);
	basic_widgets->set_visible(m_control_panel_visible);
	//tools->set_visible(m_control_panel_visible);
	//image_window->set_visible(false); 


	if (!glfwWindowShouldClose(window))
	{
		// Check if any events have been activated (key pressed, mouse moved etc.) and call corresponding response functions
		glfwPollEvents();

		m_canvas->UpdateViewport(); 

		// Draw nanogui_
		screen->draw_setup();
		screen->clear(); // glClear
		screen->draw_contents();
		screen->draw_widgets();
		screen->draw_teardown();

		static int frameIdx = 0;
		if (m_save_screen)
		{
			glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
			glReadBuffer(GL_FRONT);
			glReadPixels(0, 0, m_screen_image.cols, m_screen_image.rows, GL_BGRA, GL_UNSIGNED_BYTE, m_screen_image.data);
			cv::flip(m_screen_image, m_screen_image, 0);

			// save screen
			const std::string img_path = m_results_folder + std::to_string(frameIdx++) + ".png";
			std::cout << "Renderer: save frame " << frameIdx << " as " << img_path << std::endl;
			cv::imwrite(img_path, m_screen_image);
		}
	}
}


void NanoRenderer::Stop()
{
	glfwTerminate();
}

void NanoRenderer::ClearRenderObjects()
{
	m_render_objects.clear(); 
	m_canvas->clear_objects();
}

void NanoRenderer::SetCanvasExtrinsic(const Eigen::Matrix3f &_R, const Eigen::Vector3f &_T)
{
	m_canvas->SetExtrinsic(_R, _T); 
	m_canvas->UpdateViewport();

}

void NanoRenderer::SetCanvasExtrinsic(const Eigen::Vector3f &_pos, const Eigen::Vector3f &_up, const Eigen::Vector3f &_center)
{
	m_canvas->SetExtrinsic(_pos, _up, _center);
	m_canvas->UpdateViewport();

}