//system library
#pragma once
#include <string>
#include <algorithm>
#include <map>
#include <vector>

//external library
#include <gst/gst.h>


class GstreamerReader {


private:
	// int NextFrame(AVFrame *frame);
	int RecvDecodedFrame(std::vector<unsigned char>& dst, double& timestamp);
	GstElement* pipeline_;
	GstElement* source_;
	GstElement* qtdemux_;
	GstElement* queue_;
	GstElement* h264parse_;
	GstElement* omxh264dec_;
	GstElement* sink_;
	std::string srcFmt_;
	int paddedWidth_ = 0;
	int paddedHeight_ = 0;
	int width_ = 0;
	int height_ = 0;
	std::pair<int, int> framerate_;
	std::vector<unsigned char> framebuffer;
	std::vector<unsigned char> bgrBuffer;
	bool yuv420_to_rgb888(const std::vector<unsigned char>& yuv, int width, int height, std::vector<unsigned char>& rgb);




public:


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
