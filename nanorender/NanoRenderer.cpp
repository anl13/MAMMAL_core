#include "NanoRenderer.h"
#include <vector_functions.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

NanoRenderer::NanoRenderer()
{
}

NanoRenderer::~NanoRenderer()
{
}

void NanoRenderer::Init(const int& window_width, const int& window_height, 
	float fx, float fy, float cx, float cy, float arcball_depth /*= 2.0f*/, bool is_pinhole)
{
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
	glViewport(0, 0, width, height);
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
	m_canvas = new ArcballCanvas(screen, width, height, fx, fy, cx, cy, is_pinhole);
	m_canvas->SetRotCenter(Eigen::Vector3f(0, 0, arcball_depth));
	m_canvas->set_background_color({ 100, 100, 100, 255 });
	m_canvas->set_draw_border(true);
	m_canvas->set_cursor(Cursor::Crosshair);

	Widget *tools = new Widget(screen);
	tools->set_layout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 5, 5));

	Button *b0 = new Button(tools, "Random Background");
	b0->set_callback([this]() {
		m_canvas->set_background_color(
			Vector4i(rand() % 256, rand() % 256, rand() % 256, 255));
	});

	Button *b1 = new Button(tools, "Pause");
	b1->set_callback([this]() {
		m_pause = !m_pause;
	});

	/////////////////////
	// Create nanogui GUI
	bool enabled = true;
	FormHelper *gui = new FormHelper(screen);
	ref<Window> nanogui_window = gui->add_window(Vector2i(10, 10), "Control Panel");
	gui->add_group("Renderer Control");
	gui->add_variable("SaveScreen", m_save_screen);
	gui->add_variable("string", m_results_folder);
	gui->add_variable("Camera", ivar)->set_spinnable(true);

	gui->add_group("Validating fields");
	gui->add_variable("float", fvar)->set_tooltip("Test.");
	gui->add_variable("double", dvar)->set_spinnable(true);

	gui->add_group("Complex types");
	gui->add_variable("Enumeration", enumval, enabled)->set_items({ "Item 1", "Item 2", "Item 3" });
	gui->add_variable("Color", colval);

	gui->add_group("Other widgets");
	gui->add_button("A button", []() { std::cout << "Button pressed." << std::endl; })
		->set_tooltip("Testing a much longer tooltip, that will wrap around to new lines multiple times.");
	nanogui_window->set_position(Vector2i(5,45));
	nanogui_window->set_visible(true);
}


void NanoRenderer::CreateRenderImage(const std::string& name, const Vector2i& size, const Vector2i& pos)
{
	auto image_window = new Window(screen, name);
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

	if (!glfwWindowShouldClose(window))
	{
		// Check if any events have been activated (key pressed, mouse moved etc.) and call corresponding response functions
		glfwPollEvents();

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
