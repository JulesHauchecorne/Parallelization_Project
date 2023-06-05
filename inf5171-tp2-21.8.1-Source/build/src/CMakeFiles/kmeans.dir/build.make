# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jules/code/cpp/inf5171/tp2/inf5171-tp2-21.8.1-Source

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jules/code/cpp/inf5171/tp2/inf5171-tp2-21.8.1-Source/build

# Include any dependencies generated for this target.
include src/CMakeFiles/kmeans.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/kmeans.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/kmeans.dir/flags.make

src/CMakeFiles/kmeans.dir/main.cpp.o: src/CMakeFiles/kmeans.dir/flags.make
src/CMakeFiles/kmeans.dir/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jules/code/cpp/inf5171/tp2/inf5171-tp2-21.8.1-Source/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/kmeans.dir/main.cpp.o"
	cd /home/jules/code/cpp/inf5171/tp2/inf5171-tp2-21.8.1-Source/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kmeans.dir/main.cpp.o -c /home/jules/code/cpp/inf5171/tp2/inf5171-tp2-21.8.1-Source/src/main.cpp

src/CMakeFiles/kmeans.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kmeans.dir/main.cpp.i"
	cd /home/jules/code/cpp/inf5171/tp2/inf5171-tp2-21.8.1-Source/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jules/code/cpp/inf5171/tp2/inf5171-tp2-21.8.1-Source/src/main.cpp > CMakeFiles/kmeans.dir/main.cpp.i

src/CMakeFiles/kmeans.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kmeans.dir/main.cpp.s"
	cd /home/jules/code/cpp/inf5171/tp2/inf5171-tp2-21.8.1-Source/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jules/code/cpp/inf5171/tp2/inf5171-tp2-21.8.1-Source/src/main.cpp -o CMakeFiles/kmeans.dir/main.cpp.s

# Object files for target kmeans
kmeans_OBJECTS = \
"CMakeFiles/kmeans.dir/main.cpp.o"

# External object files for target kmeans
kmeans_EXTERNAL_OBJECTS =

bin/kmeans: src/CMakeFiles/kmeans.dir/main.cpp.o
bin/kmeans: src/CMakeFiles/kmeans.dir/build.make
bin/kmeans: lib/libkmeanslib.a
bin/kmeans: /usr/lib/x86_64-linux-gnu/libtbb.so.2
bin/kmeans: src/CMakeFiles/kmeans.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jules/code/cpp/inf5171/tp2/inf5171-tp2-21.8.1-Source/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/kmeans"
	cd /home/jules/code/cpp/inf5171/tp2/inf5171-tp2-21.8.1-Source/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kmeans.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/kmeans.dir/build: bin/kmeans

.PHONY : src/CMakeFiles/kmeans.dir/build

src/CMakeFiles/kmeans.dir/clean:
	cd /home/jules/code/cpp/inf5171/tp2/inf5171-tp2-21.8.1-Source/build/src && $(CMAKE_COMMAND) -P CMakeFiles/kmeans.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/kmeans.dir/clean

src/CMakeFiles/kmeans.dir/depend:
	cd /home/jules/code/cpp/inf5171/tp2/inf5171-tp2-21.8.1-Source/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jules/code/cpp/inf5171/tp2/inf5171-tp2-21.8.1-Source /home/jules/code/cpp/inf5171/tp2/inf5171-tp2-21.8.1-Source/src /home/jules/code/cpp/inf5171/tp2/inf5171-tp2-21.8.1-Source/build /home/jules/code/cpp/inf5171/tp2/inf5171-tp2-21.8.1-Source/build/src /home/jules/code/cpp/inf5171/tp2/inf5171-tp2-21.8.1-Source/build/src/CMakeFiles/kmeans.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/kmeans.dir/depend

