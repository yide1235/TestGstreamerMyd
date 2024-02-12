//system library
#pragma once  //make sure not duplicate reference
#include <string>
#include <vector>

//external library
#include <gst/gst.h>



// YUV to RGB convert coefficients
const float COEFF_R_V = 1.402f;
const float COEFF_G_U = -0.344136f;
const float COEFF_G_V = -0.714136f;
const float COEFF_B_U = 1.772f;
const float BIAS = 128.0f;
const int COLOR_MIN_CPU = 0;
const int COLOR_MAX_CPU = 255;
const int COLOR_CHANNELS_CPU = 3; 



class GstreamerReader {

public:


    ~GstreamerReader();

	int Open(const std::string& url);

	int Read(std::vector<unsigned char>& buffer, double& timestamp);

	std::pair<int, int> Framerate();

	void InputOriginSize(const int width, const int height);
	
	int GetHeight();

	int GetWidth();



private:
    // Helper functions
    int RecvDecodedFrame(std::vector<unsigned char>& dst, double& timestamp);
    bool yuv420_to_rgb888(const std::vector<unsigned char>& yuv, int width, int height, std::vector<unsigned char>& rgb);

    // GStreamer pipeline elements
    GstElement* pipeline_ = nullptr;
    GstElement* source_ = nullptr;
    GstElement* qtdemux_ = nullptr;
    GstElement* queue_ = nullptr;
    GstElement* h264parse_ = nullptr;
    GstElement* omxh264dec_ = nullptr;
    GstElement* sink_ = nullptr;

    // Video properties
    std::string srcFmt_;
    int paddedWidth_ = 0;
    int paddedHeight_ = 0;
    int width_ = 0;
    int height_ = 0;
    std::pair<int, int> framerate_;

    // Buffers for video processing
    std::vector<unsigned char> framebuffer;
    std::vector<unsigned char> bgrBuffer;



};
