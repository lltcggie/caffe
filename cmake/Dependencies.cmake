# This list is required for static linking and exported to CaffeConfig.cmake
set(Caffe_LINKER_LIBS "")

# ---[ Boost
if(MSVC)
    add_definitions(-DBOOST_ALL_NO_LIB)    
endif()
find_package(Boost 1.46 REQUIRED COMPONENTS system thread)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${Boost_LIBRARIES})

# ---[ Threads
find_package(Threads REQUIRED)
list(APPEND Caffe_LINKER_LIBS ${CMAKE_THREAD_LIBS_INIT})

# ---[ Google-glog
include("cmake/External/glog.cmake")
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${GLOG_LIBRARIES})

# ---[ Google-gflags
include("cmake/External/gflags.cmake")
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${GFLAGS_LIBRARIES})

# ---[ Google-protobuf
include(cmake/ProtoBuf.cmake)

# ---[ HDF5
find_package(HDF5 COMPONENTS C HL static REQUIRED)
include_directories(SYSTEM ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${HDF5_C_STATIC_LIBRARY} ${HDF5_HL_STATIC_LIBRARY})

# ---[ LMDB
if(USE_LMDB)
  find_package(LMDB REQUIRED)
  include_directories(SYSTEM ${LMDB_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${LMDB_LIBRARIES})
  add_definitions(-DUSE_LMDB)
  if(ALLOW_LMDB_NOLOCK)
    add_definitions(-DALLOW_LMDB_NOLOCK)
  endif()
endif()

# ---[ LevelDB
if(USE_LEVELDB)
  find_package(LevelDB REQUIRED)
  include_directories(SYSTEM ${LevelDB_INCLUDE})
  list(APPEND Caffe_LINKER_LIBS ${LevelDB_LIBRARIES})
  add_definitions(-DUSE_LEVELDB)
endif()

# ---[ Snappy
if(USE_LEVELDB)
  find_package(Snappy REQUIRED)
  include_directories(SYSTEM ${Snappy_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${Snappy_LIBRARIES})
endif()

# ---[ CUDA
include(cmake/Cuda.cmake)
if(NOT HAVE_CUDA)
  if(CPU_ONLY)
    message(STATUS "-- CUDA is disabled. Building without it...")
  else()
    message(WARNING "-- CUDA is not detected by cmake. Building without it...")
  endif()

  # TODO: remove this not cross platform define in future. Use caffe_config.h instead.
  add_definitions(-DCPU_ONLY)
endif()

# ---[ OpenCV
if(USE_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
  if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
  endif()
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
  list(APPEND Caffe_LINKER_LIBS ${OpenCV_LIBS})
  message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
  add_definitions(-DUSE_OPENCV)
endif()

# ---[ BLAS
if(NOT APPLE)
  set(BLAS "Atlas" CACHE STRING "Selected BLAS library")
  set_property(CACHE BLAS PROPERTY STRINGS "Atlas;Open;MKL")

  if(BLAS STREQUAL "Atlas" OR BLAS STREQUAL "atlas")
    find_package(Atlas REQUIRED)
    include_directories(SYSTEM ${Atlas_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS ${Atlas_LIBRARIES})
  elseif(BLAS STREQUAL "Open" OR BLAS STREQUAL "open")
    find_package(OpenBLAS REQUIRED)
    include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS ${OpenBLAS_LIB})
  elseif(BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
    find_package(MKL REQUIRED)
    include_directories(SYSTEM ${MKL_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS ${MKL_LIBRARIES})
    add_definitions(-DUSE_MKL)
  endif()
elseif(APPLE)
  find_package(vecLib REQUIRED)
  include_directories(SYSTEM ${vecLib_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${vecLib_LINKER_LIBS})
endif()

# ---[ Python
if(BUILD_python)
  if(NOT "${python_version}" VERSION_LESS "3.0.0")
    # use python3
    find_package(PythonInterp 3.0)
    find_package(PythonLibs 3.0)
    find_package(NumPy 1.7.1)
    # Find the matching boost python implementation
    set(version ${PYTHONLIBS_VERSION_STRING})
    
    STRING( REPLACE "." "" boost_py_version ${version} )
    find_package(Boost 1.46 COMPONENTS "python-py${boost_py_version}")
    set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})
    
    while(NOT "${version}" STREQUAL "" AND NOT Boost_PYTHON_FOUND)
      STRING( REGEX REPLACE "([0-9.]+).[0-9]+" "\\1" version ${version} )
      
      STRING( REPLACE "." "" boost_py_version ${version} )
      find_package(Boost 1.46 COMPONENTS "python-py${boost_py_version}")
      set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})
      
      STRING( REGEX MATCHALL "([0-9.]+).[0-9]+" has_more_version ${version} )
      if("${has_more_version}" STREQUAL "")
        break()
      endif()
    endwhile()
    if(NOT Boost_PYTHON_FOUND)
      find_package(Boost 1.46 COMPONENTS python)
    endif()
  else()
    # disable Python 3 search
    find_package(PythonInterp 2.7)
    find_package(PythonLibs 2.7)
    find_package(NumPy 1.7.1)
    find_package(Boost 1.46 COMPONENTS python)
  endif()
  
  if(PYTHONLIBS_FOUND AND WIN32 AND NOT EXISTS ${PYTHON_DEBUG_LIBRARY})
    # replace python??_d.lib to python??.lib when not exist python??_d.lib for Windows
    function(get_library_path LIBRARIES LIBRARIE_type LIBRARIE_path)
      if(NOT LIBRARIES)
        return()
      endif()

      set(${LIBRARIE_path} "" PARENT_SCOPE)

      list(LENGTH LIBRARIES _LIBRARIES_len)

      if(_LIBRARIES_len LESS 2)
        return()
      endif()

      math(EXPR _LIBRARIES_elm_num "${_LIBRARIES_len} / 2 - 1")
    
      foreach(val RANGE ${_LIBRARIES_elm_num})
        math(EXPR val1 "${val} * 2")
        math(EXPR val2 "${val1} + 1")

        list(GET LIBRARIES ${val1} _LIBRARIES_type)
        list(GET LIBRARIES ${val2} _LIBRARIES_path)

        if(${_LIBRARIES_type} STREQUAL ${LIBRARIE_type})
          set(${LIBRARIE_path} "${_LIBRARIES_path}" PARENT_SCOPE)
        endif()
      endforeach()
    endfunction()

    function(set_library_path LIBRARIES Target_LIBRARIE_type Target_LIBRARIE_path DST_NEW_LIBRARIES)
      if(NOT LIBRARIES)
        return()
      endif()

      set(NEW_LIBRARIES)

      list(LENGTH LIBRARIES _LIBRARIES_len)

      if(_LIBRARIES_len LESS 2)
        return()
      endif()

      math(EXPR _LIBRARIES_elm_num "${_LIBRARIES_len} / 2 - 1")

      foreach(val RANGE ${_LIBRARIES_elm_num})
        math(EXPR val1 "${val} * 2")
        math(EXPR val2 "${val1} + 1")

        list(GET LIBRARIES ${val1} _LIBRARIES_type)
        list(GET LIBRARIES ${val2} _LIBRARIES_path)

        if(${_LIBRARIES_type} STREQUAL ${Target_LIBRARIE_type})
          list(APPEND NEW_LIBRARIES ${_LIBRARIES_type} ${Target_LIBRARIE_path})
        else()
          list(APPEND NEW_LIBRARIES ${_LIBRARIES_type} ${_LIBRARIES_path})
        endif()
      endforeach()

      set(${DST_NEW_LIBRARIES} "${NEW_LIBRARIES}" PARENT_SCOPE)
    endfunction()

    get_library_path("${PYTHON_LIBRARIES}" "optimized" _optimized_PYTHON_LIBRARIE)

    if(_optimized_PYTHON_LIBRARIE AND NOT "${PYTHON_DEBUG_LIBRARY}" STREQUAL "${_optimized_PYTHON_LIBRARIE}")
      set_library_path("${PYTHON_LIBRARIES}" "debug" ${_optimized_PYTHON_LIBRARIE} PYTHON_LIBRARIES)
    endif()
  endif()

  if(PYTHONLIBS_FOUND AND NUMPY_FOUND AND Boost_PYTHON_FOUND)
    set(HAVE_PYTHON TRUE)
    if(BUILD_python_layer)
      if(Boost_USE_STATIC_LIBS AND MSVC)
        add_definitions(-DBOOST_PYTHON_STATIC_LIB)
      endif()
      if(MSVC)
        add_definitions(-DMS_NO_COREDLL -DPy_ENABLE_SHARED=1)
      endif()
      add_definitions(-DWITH_PYTHON_LAYER)
      include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
      list(APPEND Caffe_LINKER_LIBS ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
    endif()
  endif()
endif()

# ---[ Matlab
if(BUILD_matlab)
  find_package(MatlabMex)
  if(MATLABMEX_FOUND)
    set(HAVE_MATLAB TRUE)
  endif()

  # sudo apt-get install liboctave-dev
  find_program(Octave_compiler NAMES mkoctfile DOC "Octave C++ compiler")

  if(HAVE_MATLAB AND Octave_compiler)
    set(Matlab_build_mex_using "Matlab" CACHE STRING "Select Matlab or Octave if both detected")
    set_property(CACHE Matlab_build_mex_using PROPERTY STRINGS "Matlab;Octave")
  endif()
endif()

# ---[ Doxygen
if(BUILD_docs)
  find_package(Doxygen)
endif()
