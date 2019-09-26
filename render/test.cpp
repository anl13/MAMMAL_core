#include "render_object.h"
#include <iostream> 
#include <fstream> 
#include <sstream> 
#include <string> 

// call back function to resize viewport accoording to window size change. 
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0,0,width, height); 
}

void processInput(GLFWwindow* window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

int main()
{
    std::string filename = "/home/al17/animal/animal_calib/render/data/cube.obj"; 
    ObjData cube(filename); 
    
    glfwInit(); 
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); 
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); 
    
    GLFWwindow* window = glfwCreateWindow(800, 600, "Learn", NULL, NULL); 

    if (window == NULL)
    {
        std::cout << "fail" << std::endl; 
        glfwTerminate(); 
        return -1;
    }
    glfwMakeContextCurrent(window); 

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "fail to init glad " << std::endl; 
        return -1; 
    }

    glViewport(100, 100, 400, 300); 
    // glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); 

    while(!glfwWindowShouldClose(window))
    {
        processInput(window); 

        // glClearColor(0.2f, 0.3f, 0.3f, 1.0f); 
        glClear(GL_COLOR_BUFFER_BIT); 

        glfwSwapBuffers(window); 
        glfwPollEvents(); 
    }
    glfwTerminate(); 

    return 0; 
}