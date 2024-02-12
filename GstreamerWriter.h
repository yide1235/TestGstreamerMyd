//system library
#pragma once
#include <string>
#include <algorithm>
#include <map>
#include <vector>

//externel library
#include <gst/gst.h>


//constant for the writer
const guint64 SLEEP_TIME_MICROSECONDS = 4000000; 
//constant for the encoder_ bitrate
const int BITRATE = 500000;
const int REF = 4;
const int PASS = 4;
const int KEY_INT_MAX = 0;
const gboolean BYTE_STREAM = TRUE;
const guint TUNE = 0x00000004;
const int NOISE_REDUCTION = 1000;


class GstreamerWriter {


public:

    ~GstreamerWriter();
    

    int Open(const std::string url);


    void SetFramerate(std::pair<int, int> framerate);

 
    void SetSize(int width, int height);


    void SetBitrate(int bitrate);


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

