# TestGstreamerMyd

//if rerun, make sure gst, cuda is installed:

  //delete content under build
  
  cd build
  
  cmake ..
  
  make
  
  cd ..
  
  ./TestGstreamer ./1.mp4 ./1_output.mp4
  
// for the code:
// gstreamer is based on cpu, cuda image processing is based on cuda, so reader(cpu) -> processing(cpu->gpu->cpu) -> writer(cpu)

