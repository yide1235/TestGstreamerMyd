# TestGstreamerMyd

//if rerun, make sure gst, cuda is installed:

    //delete content under build

    //then you can run 

    cmake ..
    
    make
    
    ./TestGstreamer ./input_video/1.mp4 ./output_video/1_output.mp4


    //if the make have some error, you can also try this command
    //nvcc TestGstreamer.cu GstreamerReaderRAW.cpp VideoWriterRaw.cpp -I/usr/local/include/opencv4 -I/usr/include/gstreamer-1.0 -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include -L/usr/local/lib -L/usr/lib/x86_64-linux-gnu -Xlinker -rpath -Xlinker /usr/local/lib -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0 -o TestGstreamer && ./TestGstreamer ./1.mp4 ./1_output.mp4



// for the code:
// gstreamer is based on cpu, cuda image processing is based on cuda, so reader(cpu) -> image processing(cpu->gpu->cpu) -> writer(cpu)

