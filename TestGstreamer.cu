//for library use
#include <fstream>
#include <ostream>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

//link with other reader and writer
#include "GstreamerReader.h"
#include "GstreamerWriter.h"


// CUDA kernel for color space conversion
__global__ void BT2020toBT709(uchar3* src, uchar3* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int index = y * width + x;

    float R2020 = src[index].x;
    float G2020 = src[index].y;
    float B2020 = src[index].z;

    // Example conversion formulas
    float R709 = R2020 * 1.660491f + G2020 * -0.587641f + B2020 * -0.072850f;
    float G709 = R2020 * -0.124550f + G2020 * 1.132900f + B2020 * -0.008349f;
    float B709 = R2020 * -0.018151f + G2020 * -0.100579f + B2020 * 1.118730f;

    dst[index].x = fminf(fmaxf(R709, 0.0f), 255.0f);
    dst[index].y = fminf(fmaxf(G709, 0.0f), 255.0f);
    dst[index].z = fminf(fmaxf(B709, 0.0f), 255.0f);
}


// a class do frame color conversion
class CudaFrameConverter {

private:
    uchar3* d_src = nullptr;
    uchar3* d_dst = nullptr;
    int frameWidth, frameHeight;
    size_t frameSize;


public:
    //assigning cuda memory when this object is called, then there are no copy memory used
    CudaFrameConverter(int width, int height) : frameWidth(width), frameHeight(height) {
        frameSize = width * height * 3;
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




void TestVideo(std::string inputUrl, std::string outUrl  ) {
    std::cout << "video:" << inputUrl << std::endl;
    GstreamerReader video;

    auto ret = video.Open(inputUrl);
    if (ret < 0) {
        std::cerr << "Failed to open video: " << inputUrl << std::endl;
        return;
    }

    // //get the width and height of the video after opened the video
    int videoWidth = video.GetWidth();
    int videoHeight = video.GetHeight();

    // //code to check if the height and width are valid number
    if (videoWidth <= 0 || videoHeight <= 0) {
        std::cerr << "Invalid video dimensions." << std::endl;
        return;
    }


    video.InputOriginSize(videoWidth, videoHeight);
    ret = video.Open(inputUrl);
    if (ret < 0) return;

    GstreamerWriter writer;

    writer.SetSize(videoWidth, videoHeight);
    writer.SetFramerate(video.Framerate());
    ret = writer.Open(outUrl);
    if (ret < 0) return;

    CudaFrameConverter converter(videoWidth, videoHeight);

    // int seq = 0;

    std::vector<unsigned char> temp, convertedTemp(videoWidth* videoHeight * 3); // Assuming frame size is 1920x1080 and 3 channels (RGB)
    double timestamp = .0;

    while (true) {
        auto ret = video.Read(temp, timestamp);
        if (ret < 0) break;

        converter.ConvertFrame(temp, convertedTemp);

        writer.Write(convertedTemp, timestamp);
    }

    std::cout << "video read over" << std::endl;
}




int main(int argc, char* argv[]) {

    //check if number of parameter correct:
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <inputUrl> <outputUrl>" << std::endl;
        return -1;
    }

    //if it is correct
	gst_init(&argc, &argv);
	std::string inputUrl(argv[1]);
	std::string outputUrl(argv[2]);
	std::cout << "read video:" << inputUrl << std::endl;
	
    
    TestVideo(inputUrl, outputUrl);
	
    return 0;
}

