# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/myd/Desktop/TestGstreamerMyd

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/myd/Desktop/TestGstreamerMyd/build

# Include any dependencies generated for this target.
include CMakeFiles/TestGstreamer.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/TestGstreamer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/TestGstreamer.dir/flags.make

CMakeFiles/TestGstreamer.dir/TestGstreamer_generated_TestGstreamer.cu.o: CMakeFiles/TestGstreamer.dir/TestGstreamer_generated_TestGstreamer.cu.o.depend
CMakeFiles/TestGstreamer.dir/TestGstreamer_generated_TestGstreamer.cu.o: CMakeFiles/TestGstreamer.dir/TestGstreamer_generated_TestGstreamer.cu.o.cmake
CMakeFiles/TestGstreamer.dir/TestGstreamer_generated_TestGstreamer.cu.o: ../TestGstreamer.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/myd/Desktop/TestGstreamerMyd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/TestGstreamer.dir/TestGstreamer_generated_TestGstreamer.cu.o"
	cd /home/myd/Desktop/TestGstreamerMyd/build/CMakeFiles/TestGstreamer.dir && /usr/bin/cmake -E make_directory /home/myd/Desktop/TestGstreamerMyd/build/CMakeFiles/TestGstreamer.dir//.
	cd /home/myd/Desktop/TestGstreamerMyd/build/CMakeFiles/TestGstreamer.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/myd/Desktop/TestGstreamerMyd/build/CMakeFiles/TestGstreamer.dir//./TestGstreamer_generated_TestGstreamer.cu.o -D generated_cubin_file:STRING=/home/myd/Desktop/TestGstreamerMyd/build/CMakeFiles/TestGstreamer.dir//./TestGstreamer_generated_TestGstreamer.cu.o.cubin.txt -P /home/myd/Desktop/TestGstreamerMyd/build/CMakeFiles/TestGstreamer.dir//TestGstreamer_generated_TestGstreamer.cu.o.cmake

CMakeFiles/TestGstreamer.dir/GstreamerReaderRAW.cpp.o: CMakeFiles/TestGstreamer.dir/flags.make
CMakeFiles/TestGstreamer.dir/GstreamerReaderRAW.cpp.o: ../GstreamerReaderRAW.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/myd/Desktop/TestGstreamerMyd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/TestGstreamer.dir/GstreamerReaderRAW.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TestGstreamer.dir/GstreamerReaderRAW.cpp.o -c /home/myd/Desktop/TestGstreamerMyd/GstreamerReaderRAW.cpp

CMakeFiles/TestGstreamer.dir/GstreamerReaderRAW.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TestGstreamer.dir/GstreamerReaderRAW.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/myd/Desktop/TestGstreamerMyd/GstreamerReaderRAW.cpp > CMakeFiles/TestGstreamer.dir/GstreamerReaderRAW.cpp.i

CMakeFiles/TestGstreamer.dir/GstreamerReaderRAW.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TestGstreamer.dir/GstreamerReaderRAW.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/myd/Desktop/TestGstreamerMyd/GstreamerReaderRAW.cpp -o CMakeFiles/TestGstreamer.dir/GstreamerReaderRAW.cpp.s

CMakeFiles/TestGstreamer.dir/VideoWriterRaw.cpp.o: CMakeFiles/TestGstreamer.dir/flags.make
CMakeFiles/TestGstreamer.dir/VideoWriterRaw.cpp.o: ../VideoWriterRaw.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/myd/Desktop/TestGstreamerMyd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/TestGstreamer.dir/VideoWriterRaw.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TestGstreamer.dir/VideoWriterRaw.cpp.o -c /home/myd/Desktop/TestGstreamerMyd/VideoWriterRaw.cpp

CMakeFiles/TestGstreamer.dir/VideoWriterRaw.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TestGstreamer.dir/VideoWriterRaw.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/myd/Desktop/TestGstreamerMyd/VideoWriterRaw.cpp > CMakeFiles/TestGstreamer.dir/VideoWriterRaw.cpp.i

CMakeFiles/TestGstreamer.dir/VideoWriterRaw.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TestGstreamer.dir/VideoWriterRaw.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/myd/Desktop/TestGstreamerMyd/VideoWriterRaw.cpp -o CMakeFiles/TestGstreamer.dir/VideoWriterRaw.cpp.s

# Object files for target TestGstreamer
TestGstreamer_OBJECTS = \
"CMakeFiles/TestGstreamer.dir/GstreamerReaderRAW.cpp.o" \
"CMakeFiles/TestGstreamer.dir/VideoWriterRaw.cpp.o"

# External object files for target TestGstreamer
TestGstreamer_EXTERNAL_OBJECTS = \
"/home/myd/Desktop/TestGstreamerMyd/build/CMakeFiles/TestGstreamer.dir/TestGstreamer_generated_TestGstreamer.cu.o"

TestGstreamer: CMakeFiles/TestGstreamer.dir/GstreamerReaderRAW.cpp.o
TestGstreamer: CMakeFiles/TestGstreamer.dir/VideoWriterRaw.cpp.o
TestGstreamer: CMakeFiles/TestGstreamer.dir/TestGstreamer_generated_TestGstreamer.cu.o
TestGstreamer: CMakeFiles/TestGstreamer.dir/build.make
TestGstreamer: /usr/local/cuda-11.7/lib64/libcudart_static.a
TestGstreamer: /usr/lib/x86_64-linux-gnu/librt.so
TestGstreamer: /usr/local/cuda-11.7/lib64/libcudart_static.a
TestGstreamer: /usr/lib/x86_64-linux-gnu/librt.so
TestGstreamer: CMakeFiles/TestGstreamer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/myd/Desktop/TestGstreamerMyd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable TestGstreamer"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TestGstreamer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/TestGstreamer.dir/build: TestGstreamer

.PHONY : CMakeFiles/TestGstreamer.dir/build

CMakeFiles/TestGstreamer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/TestGstreamer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/TestGstreamer.dir/clean

CMakeFiles/TestGstreamer.dir/depend: CMakeFiles/TestGstreamer.dir/TestGstreamer_generated_TestGstreamer.cu.o
	cd /home/myd/Desktop/TestGstreamerMyd/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/myd/Desktop/TestGstreamerMyd /home/myd/Desktop/TestGstreamerMyd /home/myd/Desktop/TestGstreamerMyd/build /home/myd/Desktop/TestGstreamerMyd/build /home/myd/Desktop/TestGstreamerMyd/build/CMakeFiles/TestGstreamer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/TestGstreamer.dir/depend

