//for library use
#include <fstream>
#include <ostream>
#include <iostream>
#include <stdexcept>
#include <chrono>

//external library
#include <cuda_runtime.h>

//link with other reader and writer
#include "GstreamerReader.h"
#include "GstreamerWriter.h"


//settings variables for image processing part:
const int COLOR_CHANNELS=3;
const int BLOCK_SIZE=32;
//define a 3x3 matrix now is a 1d array, so the length is 3*3, and to avoid passing length from gpu, use a global
const int TRANSFORMATION_MATRIX_SIZE = 3 * 3;
//the 3x3 transmation matrix now is saved to be 1d array for better efficient in CUDA
float h_transformationMatrix_BT2020toBT709[TRANSFORMATION_MATRIX_SIZE] = {
    1.660491f, -0.587641f, -0.072850f,
    -0.124550f, 1.132900f, -0.008349f,
    -0.018151f, -0.100579f, 1.118730f
};

// Constants for color channel limits
__constant__ float COLOR_MIN = 0.0f;
__constant__ float COLOR_MAX = 255.0f;





// CUDA kernel for color space conversion
__global__ void color_transformation_3x3(uchar3* src, uchar3* dst, int width, int height, const float* transformationMatrix) {
    
    if (!transformationMatrix) return;//check if pointer of transmation matrix is null
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int index = y * width + x;

    float R_src = src[index].x;
    float G_src = src[index].y;
    float B_src = src[index].z;

    //conversion formulas
    float R_dst = R_src * transformationMatrix[0] + G_src * transformationMatrix[1] + B_src * transformationMatrix[2];
    float G_dst = R_src * transformationMatrix[3] + G_src * transformationMatrix[4] + B_src * transformationMatrix[5];
    float B_dst = R_src * transformationMatrix[6] + G_src * transformationMatrix[7] + B_src * transformationMatrix[8];

    dst[index].x = fminf(fmaxf(R_dst, COLOR_MIN), COLOR_MAX);
    dst[index].y = fminf(fmaxf(G_dst, COLOR_MIN), COLOR_MAX);
    dst[index].z = fminf(fmaxf(B_dst, COLOR_MIN), COLOR_MAX);

}





//CUDA checking code
inline void checkCudaError(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) 
                  << " at " << file << ":" << line << std::endl;
        throw std::runtime_error("CUDA Runtime Error");
    }
}

#define CHECK_CUDA_ERROR(val) checkCudaError((val), __FILE__, __LINE__)




// a class do frame color conversion
class CudaFrameConverter {

private:
    uchar3* d_src = nullptr;
    uchar3* d_dst = nullptr;
    float* d_transformationMatrix = nullptr; 
    int frameWidth, frameHeight;
    size_t frameSize;


public:
    //assigning cuda memory when this object is called, then there are no copy memory used
    CudaFrameConverter(int width, int height, const float* h_transformationMatrix)  : frameWidth(width), frameHeight(height) {
        
        frameSize = width * height * COLOR_CHANNELS;

        //this is the setting for the convert objects, make sure they are load once
        CHECK_CUDA_ERROR(cudaMalloc(&d_src, frameSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_dst, frameSize));

        //also load for the transformation matrix here
        CHECK_CUDA_ERROR(cudaMalloc(&d_transformationMatrix, TRANSFORMATION_MATRIX_SIZE * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_transformationMatrix, h_transformationMatrix, TRANSFORMATION_MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    }

    //for destructor clean the memory
    ~CudaFrameConverter() {
        cudaFree(d_src);
        cudaFree(d_dst);
        cudaFree(d_transformationMatrix); 

    }



    void ConvertFrame(const std::vector<unsigned char>& inputFrame, std::vector<unsigned char>& outputFrame) {
        
        CHECK_CUDA_ERROR(cudaMemcpy(d_src, inputFrame.data(), frameSize, cudaMemcpyHostToDevice));
        // Perform color space conversion

        // Adjust gridSize and blockSize as necessary
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); 
        dim3 gridSize((frameWidth + blockSize.x - 1) / blockSize.x, (frameHeight + blockSize.y - 1) / blockSize.y);
    
        // Ensuring the grid covers the entire image
        if (gridSize.x * blockSize.x < frameWidth || gridSize.y * blockSize.y < frameHeight) {
            std::cerr << "Error: Grid size is insufficient to cover the entire image." << std::endl;
            throw std::runtime_error("Grid size insufficient for image dimensions.");
        }

        color_transformation_3x3<<<gridSize, blockSize>>>(d_src, d_dst, frameWidth, frameHeight, d_transformationMatrix);
        cudaDeviceSynchronize(); //check excutation
        CHECK_CUDA_ERROR(cudaGetLastError()); // Check for any errors in kernel launch
        CHECK_CUDA_ERROR(cudaMemcpy(outputFrame.data(), d_dst, frameSize, cudaMemcpyDeviceToHost));
    
    }

};




void TestVideo(const std::string& inputUrl, const std::string& outUrl  ) {
    
    
    // load the reader
    GstreamerReader reader;

    // check if address is valid
    if (reader.Open(inputUrl) < 0) {
        std::cerr << "Failed to open video: " << inputUrl << std::endl;
        return;
    }

    // get the width and height of the video after opened the video
    int videoWidth = reader.GetWidth();
    int videoHeight = reader.GetHeight();


    // check if the height and width are valid
    if (videoWidth <= 0 || videoHeight <= 0) {
        std::cerr << "Invalid video dimensions." << std::endl;
        return;
    }

    //set reader size
    reader.InputOriginSize(videoWidth, videoHeight);
    // end of loading reader




    // load the writer
    GstreamerWriter writer;

    //use reader info to set reader
    writer.SetSize(videoWidth, videoHeight);
    writer.SetFramerate(reader.Framerate());

    // check of writer is valid
    if (writer.Open(outUrl) < 0){
        return;}

    //end of writer


    //call the convert object to convert color for each frame
    CudaFrameConverter converter(videoWidth, videoHeight, h_transformationMatrix_BT2020toBT709);

    //buffer size calculated
    std::vector<unsigned char> frameBuffer, convertedFrame(videoWidth* videoHeight * COLOR_CHANNELS); 
    
    //initialize the tempstamp
    double timestamp = .0;

    //to do image processing, call the memory for parameter outside of loop

    //when there is frame is the buffer, then keep running
    while (reader.Read(frameBuffer, timestamp) >= 0) {
        
        converter.ConvertFrame(frameBuffer, convertedFrame);

        writer.Write(convertedFrame, timestamp);
    }

    //std::cout << "video read and write finished." << std::endl;

}




int main(int argc, char* argv[]) {

    //check if number of parameter correct, requires the excutable, input, output
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <inputUrl> <outputUrl>" << std::endl;
        return -1;
    }

	gst_init(&argc, &argv);
	std::string inputUrl(argv[1]);
	std::string outputUrl(argv[2]);
    TestVideo(inputUrl, outputUrl);
	
    return 0;
}

