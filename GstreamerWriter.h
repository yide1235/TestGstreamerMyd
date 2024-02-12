//system library
#pragma once
#include <string>
#include <algorithm>
#include <map>
#include <vector>

//externel library
#include <gst/gst.h>


class GstreamerWriter {


public:

    ~GstreamerWriter();
    

    int Open(const std::string url);


    void SetFramerate(std::pair<int, int> framerate) {
        framerate_ = framerate;
    }

 
    void SetSize(int width, int height) {
        width_ = width;
        height_ = height;
    }


    void SetBitrate(int bitrate) {
        bitrate_ = bitrate;
    }


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
    //default value for frame rate
    std::pair<int, int> framerate_{ 30, 1 };



};

