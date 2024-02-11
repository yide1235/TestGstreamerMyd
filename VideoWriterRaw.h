#pragma once


#include <string>

#include <algorithm>
#include <map>
#include <gst/gst.h>
#include <vector>

class VideoWriterRaw {
public:
    ~VideoWriterRaw();
    /**
     * @brief 
     */
    int Open(const std::string url);

    /**
     * @brief 
     * @param fps = framerate.first/framerate.second
     */
    void SetFramerate(std::pair<int, int> framerate) {
        framerate_ = framerate;
    }

    /**
     * @brief 
     */
    void SetSize(int width, int height) {
        width_ = width;
        height_ = height;
    }

    /**
     * @brief 
     * @param bitrate bit/sec
     */
    void SetBitrate(int bitrate) {
        bitrate_ = bitrate;
    }

    /**
     * @brief
     * @param timestamp
     */
    int Write(const std::vector<unsigned char>& frame, double timestamp);

private:
    int PushData2Pipeline(const  std::vector<unsigned char>& frame, double timestamp);
    GstElement* pipeline_;
    GstElement* appSrc_;
    GstElement* queue_;
    GstElement* videoConvert_;
    GstElement* encoder_;
    GstElement* capsFilter_;
    GstElement* mux_;
    GstElement* sink_;
    int width_ = 0;
    int height_ = 0;
    int bitrate_ = 0;
    std::pair<int, int> framerate_{ 30, 1 };
};

