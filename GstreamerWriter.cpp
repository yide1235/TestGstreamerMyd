// system libary
#include <ostream>
#include <iostream>

// externel library
#include <gst/gst.h>

// reference header
#include "GstreamerWriter.h"

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


//definition of writer

GstreamerWriter::~GstreamerWriter()
{
    if (appSrc_)
    {
        GstFlowReturn retflow;
        g_signal_emit_by_name(appSrc_, "end-of-stream", &retflow);
        std::cout << "EOS sended. Writing last several frame..." << std::endl;

        //when handling end of stream signal, given some extra time for write the last several frames
        g_usleep(SLEEP_TIME_MICROSECONDS);
        std::cout << "Writing Done!" << std::endl;
        if (retflow != GST_FLOW_OK)
        {
            std::cerr << "We got some error when sending eos!" << std::endl;
        }
    }

    if (pipeline_)
    {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }

}


int GstreamerWriter::Open(const std::string url)
{

    appSrc_ = gst_element_factory_make("appsrc", "AppSrc");
    queue_ = gst_element_factory_make("queue", "QueueWrite");
    videoConvert_ = gst_element_factory_make("videoconvert", "Videoconvert");
    encoder_ = gst_element_factory_make("x264enc", "X264enc");
    capsFilter_ = gst_element_factory_make("capsfilter", "Capsfilter");
    mux_ = gst_element_factory_make("qtmux", "Qtmux");
    sink_ = gst_element_factory_make("filesink", "OutputFile");


    // Create the empty pipeline
    pipeline_ = gst_pipeline_new("encode-pipeline");

    if (!pipeline_ || !appSrc_ || !queue_ || !videoConvert_ || !encoder_ || !capsFilter_ || !mux_ || !sink_)
    {
        std::cerr << "Not all elements could be created" << std::endl;
        return -1;
    }

    //  src format
    std::string srcFmt = "BGR";

    // Modify element properties
    g_object_set(G_OBJECT(appSrc_), "stream-type", 0, "format", GST_FORMAT_TIME, nullptr);

    g_object_set(G_OBJECT(appSrc_), "caps", gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, srcFmt.c_str(), "width", G_TYPE_INT, width_, "height", G_TYPE_INT, height_, "framerate", GST_TYPE_FRACTION, framerate_.first, framerate_.second, nullptr), nullptr);

    g_object_set(G_OBJECT(capsFilter_), "caps", gst_caps_new_simple("video/x-h264", "stream-format", G_TYPE_STRING, "avc", "profile", G_TYPE_STRING, "main", nullptr), nullptr);

    g_object_set(G_OBJECT(sink_), "location", url.c_str(), nullptr);

    // g_object_set(G_OBJECT(encoder_), "bitrate", 500000, "ref", 4, "pass", 4 , "key-int-max", 0, "byte-stream", TRUE, "tune", 0x00000004, "noise-reduction", 1000, NULL);
    g_object_set(G_OBJECT(encoder_), "bitrate", BITRATE, "ref", REF, "pass", PASS, "key-int-max", KEY_INT_MAX, "byte-stream", BYTE_STREAM, "tune", TUNE, "noise-reduction", NOISE_REDUCTION, NULL);

    // Build the pipeline
    gst_bin_add_many(GST_BIN(pipeline_), appSrc_, queue_, videoConvert_, encoder_, capsFilter_, mux_, sink_, nullptr);

    if (gst_element_link_many(appSrc_, queue_, videoConvert_, encoder_, capsFilter_, mux_, sink_, nullptr) != TRUE)
    {
        std::cerr << "appSrc, queue, videoConvert, encoder, capsFilter, mux and sink could not be linked" << std::endl;
        return -1;
    }


    // Start playing
    auto ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE)
    {
        std::cerr << "Unable to set the pipeline to the playing state" << std::endl;
        return -1;
    }


    return 0;
}


int GstreamerWriter::PushData2Pipeline(const std::vector<unsigned char> &frame, double timestamp)
{
    GstBuffer *buffer;
    GstFlowReturn ret;
    GstMapInfo map;

    // Create a new empty buffer
    unsigned int size = frame.size();
    buffer = gst_buffer_new_and_alloc(size);

    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, frame.data(), size);

    // debug
    std::cout << "wrote size:" << size << std::endl;

    gst_buffer_unmap(buffer, &map);
    GST_BUFFER_PTS(buffer) = static_cast<uint64_t>(timestamp * GST_SECOND);
    GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND, framerate_.first / framerate_.second);

    std::cout << "send data into buffer" << std::endl;
    std::cout << "GST_BUFFER_DURATION(buffer):" << GST_BUFFER_DURATION(buffer) << std::endl;
    std::cout << "timestamp:" << static_cast<uint64_t>(timestamp * GST_SECOND) << std::endl;

    // Push the buffer into the appsrc
    g_signal_emit_by_name(appSrc_, "push-buffer", buffer, &ret);

    // Free the buffer now that we are done with it
    gst_buffer_unref(buffer);

    if (ret != GST_FLOW_OK)
    {
        // We got some error, stop sending data
        std::cout << "We got some error, stop sending data" << std::endl;
        return -1;
    }
    return 0;
}


int GstreamerWriter::Write(const std::vector<unsigned char> &frame, double timestamp)
{
    return PushData2Pipeline(frame, timestamp);
}

//end of writer

