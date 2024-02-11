//system library
#pragma once  //make sure not duplicate reference
#include <string>
#include <algorithm>
#include <map>
#include <vector>

//external library
#include <gst/gst.h>


class GstreamerReader {


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


public:


    // // Constructor
    // GstreamerReader();

    // // Destructor
    // ~GstreamerReader();

    // // Public interface
    // int Open(const std::string& url);
    // int Read(std::vector<unsigned char>& buffer, double& timestamp);
    // std::pair<int, int> Framerate() const;
    // void InputOriginSize(int width, int height);
    // int GetWidth();
    // int GetHeight();

	/**
	* @brief 
	*/
	int Open(const std::string& url);
	/**
	*
	*/
	int Read(std::vector<unsigned char>& buffer, double& timestamp);
	/**
	* @brief 
	* @param fps = framerate.first/framerate.second
	*/
	std::pair<int, int> Framerate() {
		return framerate_;
	}

	/**
	 * @brief 
	 */
	void InputOriginSize(const int width, const int height) {
		width_ = width;
		height_ = height;
	}
	~GstreamerReader();

	
	int GetHeight();

	int GetWidth();


	// int GetFrameCount() const;


	

};
