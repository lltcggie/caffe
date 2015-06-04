# Installation

http://caffe.berkeleyvision.org/installation.html も参考になる

# Windows
Windowsの場合は「msvc」フォルダ内に入っている「caffe.sln」でビルド出来る。
[Neil Z. SHAOさんのページ](https://initialneil.wordpress.com/2015/01/11/build-caffe-in-windows-with-visual-studio-2013-cuda-6-5-opencv-2-4-9/)を参考にして「3rdparty」フォルダ内にprotoc.exeや依存するライブラリを入れておくこと。

# Ubuntu
Ubuntu 15.04 AMD64でビルドしたときの手順は以下の通り

sudo apt-get update  
sudo apt-get upgrade  

sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev  
sudo apt-get install libatlas-base-dev  
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler  
sudo apt-get install nvidia-cuda-toolkit  
sudo apt-get install git

git clone https://github.com/BVLC/caffe.git

cd caffe  
cp Makefile.config.example Makefile.config

Makefile.configを以下のように編集

CUDA_DIR := /usr/local/cuda  
をコメントアウトし、  
\# CUDA_DIR := /usr  
のコメントアウトを外す

変更前  
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include  
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib  

変更後  
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/  
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial/  

make all
