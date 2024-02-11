#include <fstream>
#include "GstreamerReaderRAW.h"
#include <ostream>
#include <iostream>
#include "VideoWriterRaw.h"
#include <cuda_runtime.h>




__global__
void BT2020toBT709(uchar3* src, uchar3* dst, int width, int height)
{
    // int index = blockDim.x * blockIdx.x + threadIdx.x;

    // if (index >= len)
    // {
    //     return;
    // }
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int index = y * width + x;

    float R2020 = src[index].x; // Use .x for the R component
    float G2020 = src[index].y; // Use .y for the G component
    float B2020 = src[index].z; // Use .z for the B component

    // Apply the BT.2020 to BT.709 conversion formula
    float R709 = R2020 * 1.660491f + G2020 * -0.587641f + B2020 * -0.072850f;
    float G709 = R2020 * -0.124550f + G2020 * 1.132900f + B2020 * -0.008349f;
    float B709 = R2020 * -0.018151f + G2020 * -0.100579f + B2020 * 1.118730f;

    // Clamp the results and assign to the destination
    dst[index].x = fminf(fmaxf(R709, 0.0f), 255.0f);
    dst[index].y = fminf(fmaxf(G709, 0.0f), 255.0f);
    dst[index].z = fminf(fmaxf(B709, 0.0f), 255.0f);
}


class CudaFrameConverter {

private:
    uchar3* d_src = nullptr;
    uchar3* d_dst = nullptr;
    int frameWidth, frameHeight;
    size_t frameSize;
    // Define gridSize and blockSize based on frameWidth and frameHeight


public:
    CudaFrameConverter(int width, int height) : frameWidth(width), frameHeight(height) {
        frameSize = width * height * 3; // Assuming 3 channels (RGB)
        cudaMalloc(&d_src, frameSize);
        cudaMalloc(&d_dst, frameSize);
    }

    ~CudaFrameConverter() {
        cudaFree(d_src);
        cudaFree(d_dst);
    }

    void ConvertFrame(const std::vector<unsigned char>& inputFrame, std::vector<unsigned char>& outputFrame) {
        cudaMemcpy(d_src, inputFrame.data(), frameSize, cudaMemcpyHostToDevice);
        // Perform color space conversion
        // Adjust gridSize and blockSize as necessary
        dim3 blockSize(32, 32); // Example blockSize
        dim3 gridSize((frameWidth + blockSize.x - 1) / blockSize.x, (frameHeight + blockSize.y - 1) / blockSize.y);
        BT2020toBT709<<<gridSize, blockSize>>>(d_src, d_dst, frameWidth, frameHeight);
        cudaMemcpy(outputFrame.data(), d_dst, frameSize, cudaMemcpyDeviceToHost);
    }

};


void TestVideo(std::string url, std::string outUrl, int count) {
    std::cout << "video:" << url << std::endl;
    VideoReaderRaw video;

    auto ret = video.Open(url);
    if (ret < 0) {
        std::cerr << "Failed to open video: " << url << std::endl;
        return;
    }

    int videoWidth = video.GetWidth();
    int videoHeight = video.GetHeight();

    std::cout << videoWidth << " ------------" << videoHeight << std::endl;
    if (videoWidth <= 0 || videoHeight <= 0) {
        std::cerr << "Invalid video dimensions." << std::endl;
        return;
    }


    video.InputOriginSize(videoWidth, videoHeight);
    ret = video.Open(url);
    if (ret < 0) return;

    VideoWriterRaw writer;
    writer.SetSize(videoWidth, videoHeight);
    writer.SetFramerate(video.Framerate());
    ret = writer.Open(outUrl);
    if (ret < 0) return;

    CudaFrameConverter converter(videoWidth, videoHeight);

    int seq = 0;
    std::vector<unsigned char> temp, convertedTemp(videoWidth* videoHeight * 3); // Assuming frame size is 1920x1080 and 3 channels (RGB)
    double timestamp = .0;

    while (seq++ < count) {
        auto ret = video.Read(temp, timestamp);
        if (ret < 0) break;

        converter.ConvertFrame(temp, convertedTemp);

        writer.Write(convertedTemp, timestamp);
    }

    std::cout << "video read over" << std::endl;
}



int main(int argc, char* argv[]) {
	gst_init(&argc, &argv);
	std::string inputUrl(argv[1]);
	std::string outputUrl(argv[2]);
	std::cout << "read video:" << inputUrl << std::endl;
	TestVideo(inputUrl, outputUrl, 400);
	return 0;
}


// // to run: cmake .. && make 
// // && cd .. && ./TestGstreamer ./1.mp4 ./output.mp4 && cd build


// //nvcc TestGstreamer.cu GstreamerReaderRAW.cpp VideoWriterRaw.cpp -I/usr/local/include/opencv4 -I/usr/include/gstreamer-1.0 -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include -L/usr/local/lib -L/usr/lib/x86_64-linux-gnu -Xlinker -rpath -Xlinker /usr/local/lib -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0 -o TestGstreamer && ./TestGstreamer ./input.mp4 ./output.mp4