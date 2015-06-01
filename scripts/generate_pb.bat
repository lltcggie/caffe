@echo off

cd /d "%~dp0"

set TIMESTAMP_OLD=""

if exist "../msvc/libcaffe/caffe.pb.h.timestamp.txt" (
    goto timestamp_exist;
) else (
    goto break;
)

:timestamp_exist
for /F "delims=# eol=#" %%A in (../msvc/libcaffe/caffe.pb.h.timestamp.txt) do (
    set TIMESTAMP_OLD=%%A
    goto break;
)
:break

if exist "../src/caffe/proto/caffe.pb.h" (
    goto caffe_pb_h_exist;
) else (
    goto caffe_pb_h_not_exist;
)

:caffe_pb_h_exist

set FILE1="../src/caffe/proto/caffe.pb.h"
set FILE_TIMESTAMP="../msvc/libcaffe/caffe.pb.h.timestamp.txt"
set TIMESTAMP_NEW=""

for %%a in (%FILE1%) do set TIMESTAMP_NEW=%%~ta
if "%TIMESTAMP_OLD%" NEQ "%TIMESTAMP_NEW%" (
    echo caffe.pb.h is being generated
    "../3rdparty/bin/protoc" -I="../src/caffe/proto" --cpp_out="../src/caffe/proto" "../src/caffe/proto/caffe.proto"
    echo %TIMESTAMP_NEW%> %FILE_TIMESTAMP%
) else (
    echo caffe.pb.h remains the same as before
)

:caffe_pb_h_not_exist

echo %~nx0 end
