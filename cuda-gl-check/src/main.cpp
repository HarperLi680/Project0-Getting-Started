#include <cstdlib>
#define DIAG_CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(e)); std::exit(1); } } while (0)
#include <cstdio>
#include <sstream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include "main.hpp"
static cudaGraphicsResource* project0Resource = nullptr;

/**
 * C main function.
 */
int main(int argc, char* argv[]) {
    // TODO: Change this line to use your name!
    m_yourName = "Qingying Li";

    if (init(argc, argv)) {
        mainLoop();
    }

    return 0;
}

/**
 * Initialization of CUDA and GLFW.
 */
bool init(int argc, char **argv) {
    // Set window title to "Student Name: [SM 2.0] GPU Name"
    std::string deviceName;
    cudaDeviceProp deviceProp;
    int gpuDevice = 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (gpuDevice > device_count) {
        std::cout << "Error: GPU device number is greater than the number of devices!" <<
                  "Perhaps a CUDA-capable GPU is not installed?" << std::endl;
        return false;
    }
    cudaGetDeviceProperties(&deviceProp, gpuDevice);
    m_major = deviceProp.major;
    m_minor = deviceProp.minor;

    std::ostringstream ss;
    ss << m_yourName << ": [SM " << m_major << "." << m_minor << "] " << deviceProp.name;
    deviceName = ss.str();

    // Window setup stuff
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        return false;
    }
    m_width = 800;
    m_height = 800;
    m_window = glfwCreateWindow(m_width, m_height, deviceName.c_str(), NULL, NULL);
    if (!m_window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(m_window);
    glfwSetKeyCallback(m_window, keyCallback);

    glewExperimental = GL_TRUE;
    GLenum glewStatus = glewInit(); if (glewStatus != GLEW_OK && !(glewStatus == GLEW_ERROR_NO_GLX_DISPLAY && glGetString(GL_VERSION) != nullptr && GLEW_VERSION_4_0)) { fprintf(stderr, "GLEW initialization failed: %s\n", glewGetErrorString(glewStatus)); fprintf(stderr, "GLEW error code: %u\n", (unsigned)glewStatus);
        return false;
    }

    // init all of the things
    initVAO();
    initTextures();
    initCUDA();
    initPBO(&m_pbo);

    GLuint passthroughProgram;
    passthroughProgram = initShader();
    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    return true;
}

void initPBO(GLuint *pbo) {
    if (pbo) {
        // set up vertex data parameter
        int num_texels = m_width * m_height;
        int num_values = num_texels * 4;
        size_t size_tex_data = sizeof(GLubyte) * num_values;

        // Generate a buffer ID called a PBO (Pixel Buffer Object)
        glGenBuffers(1, pbo);
        // Make this the current UNPACK buffer (OpenGL is state-based)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
        // Allocate data for the buffer. 4-channel 8-bit image
        glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
        DIAG_CUDA(cudaGraphicsGLRegisterBuffer(&project0Resource, *pbo, cudaGraphicsRegisterFlagsNone));
    }
}

void initVAO() {
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texCoords[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)m_positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(m_positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texCoords), texCoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)m_texCoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(m_texCoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

void initCUDA() {
    // Default to device ID 0. If you have more than one GPU and want to test a non-default one,
    // change the device ID.
    DIAG_CUDA(cudaSetDevice(0));
}

void initTextures() {
    glGenTextures(1, &m_image);
    glBindTexture(GL_TEXTURE_2D, m_image);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_BGRA,
                 GL_UNSIGNED_BYTE, NULL);
}

GLuint initShader() {
    const char *attributeLocations[] = { "Position", "Tex" };
    GLuint program = glslUtility::createDefaultProgram(attributeLocations, 2);
    GLint location;
    glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1) {
        glUniform1i(location, 0);
    }
    return program;
}

// ====================================
// Main loop stuff
// ====================================

void runCUDA() {
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
    uchar4 *dptr = NULL;
    size_t mappedBytes = 0; DIAG_CUDA(cudaGraphicsMapResources(1, &project0Resource, 0)); DIAG_CUDA(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &mappedBytes, project0Resource));

    // Execute the kernel
    kernelVersionVis(dptr, m_width, m_height, m_major, m_minor);

    // Unmap buffer object
    DIAG_CUDA(cudaGraphicsUnmapResources(1, &project0Resource, 0));
}

void mainLoop() {
    fprintf(stderr, "Entered rendering loop\n");
    while (!glfwWindowShouldClose(m_window)) {
        static bool traceFirst = true; if (traceFirst) fprintf(stderr, "STEP 1: before events\n");
        glfwPollEvents();
        if (traceFirst) fprintf(stderr, "STEP 2: before CUDA\n");
        runCUDA();
        // PROJECT0_WINDOW_RECONNECT: restore the window context after CUDA.
        glfwMakeContextCurrent(nullptr);
        glfwMakeContextCurrent(m_window);
        if (traceFirst) fprintf(stderr, "STEP 3: CUDA finished\n");

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
        {
            static bool checkedPbo = false;
            if (!checkedPbo) {
                checkedPbo = true;
                unsigned char a[4] = {}, b[4] = {};
                glGetBufferSubData(GL_PIXEL_UNPACK_BUFFER,
                    (200 * m_width + 400) * 4, 4, a);
                glGetBufferSubData(GL_PIXEL_UNPACK_BUFFER,
                    (600 * m_width + 400) * 4, 4, b);
                fprintf(stderr, "PBO check: error=0x%x RGB1=%u,%u,%u RGB2=%u,%u,%u\n",
                    glGetError(), a[0], a[1], a[2], b[0], b[1], b[2]);
            }
        }
        glBindTexture(GL_TEXTURE_2D, m_image);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGBA,
                        GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);
        {
            static bool drawStateChecked = false; // DRAW_STATE_CHECK
            if (!drawStateChecked) {
                drawStateChecked = true;
                GLint program = 0, linked = 0, width = 0, height = 0;
                glGetIntegerv(GL_CURRENT_PROGRAM, &program);
                if (program) glGetProgramiv(program, GL_LINK_STATUS, &linked);
                glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
                glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
                fprintf(stderr, "DRAW: program=%d linked=%d texture=%dx%d\n",
                    program, linked, width, height);
                if (program) fprintf(stderr, "ATTR: Position=%d Texcoords=%d\n",
                    glGetAttribLocation(program, "Position"),
                    glGetAttribLocation(program, "Texcoords"));

                if (width == 800 && height == 800) {
                    GLint packBuffer = 0;
                    glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &packBuffer);
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
                    unsigned char* pixels = new unsigned char[800 * 800 * 4]();
                    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
                    unsigned a = (200 * 800 + 400) * 4;
                    unsigned b = (600 * 800 + 400) * 4;
                    fprintf(stderr, "TEXTURE: error=0x%x RGB1=%u,%u,%u RGB2=%u,%u,%u\n",
                        glGetError(), pixels[a], pixels[a+1], pixels[a+2],
                        pixels[b], pixels[b+1], pixels[b+2]);
                    delete[] pixels;
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, packBuffer);
                }
            }
        }

        // VAO, shader program, and texture already bound
        if (traceFirst) fprintf(stderr, "STEP 4: before draw\n");
        glViewport(0, 0, m_width, m_height);
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);
        if (traceFirst) fprintf(stderr, "STEP 5: draw returned\n");
        {
            static bool checkedFrame = false;
            if (!checkedFrame) {
                checkedFrame = true;
                GLenum drawError = glGetError();
                unsigned char a[4] = {}, b[4] = {};
                glReadBuffer(GL_BACK);
                glReadPixels(400, 200, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, a);
                glReadPixels(400, 600, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, b);
                fprintf(stderr, "Frame check: drawError=0x%x readError=0x%x RGB1=%u,%u,%u RGB2=%u,%u,%u\n",
                    drawError, glGetError(), a[0], a[1], a[2], b[0], b[1], b[2]);
            }
        }
        glfwSwapBuffers(m_window);
        if (traceFirst) fprintf(stderr, "STEP 6: frame finished\n"); traceFirst = false;
    }
    cleanupCUDA();
    glfwDestroyWindow(m_window);
    glfwTerminate();
}


void errorCallback(int error, const char *description) {
    fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

// ====================================
// Clean-up stuff
// ====================================

void cleanupCUDA() {
    if (m_pbo) {
        deletePBO(&m_pbo);
    }
    if (m_image) {
        deleteTexture(&m_image);
    }
}

void deletePBO(GLuint *pbo) {
    if (pbo) {
        // unregister this buffer object with CUDA
        DIAG_CUDA(cudaGraphicsUnregisterResource(project0Resource)); project0Resource = nullptr;

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint *tex) {
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}
