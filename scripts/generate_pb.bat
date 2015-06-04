@echo off

cd /d "%~dp0"

set FILE1="../src/caffe/proto/caffe.proto"
set FILE_TIMESTAMP="../msvc/libcaffe/caffe.proto.timestamp.txt"
set TIMESTAMP_NEW=""
set TIMESTAMP_OLD=""

for %%a in (%FILE1%) do set TIMESTAMP_NEW=%%~ta

if exist "../msvc/libcaffe/caffe.proto.timestamp.txt" (
    goto timestamp_exist;
) else (
    goto break;
)

:timestamp_exist
for /F "delims=# eol=#" %%A in (../msvc/libcaffe/caffe.proto.timestamp.txt) do (
    set TIMESTAMP_OLD=%%A
    goto break;
)
:break

if exist "../src/caffe/proto/caffe.proto" (
    goto caffe_pb_h_exist;
) else (
    goto caffe_pb_h_generate;
)

:caffe_pb_h_exist

if "%TIMESTAMP_OLD%" NEQ "%TIMESTAMP_NEW%" (
    goto caffe_pb_h_generate;
) else (
    goto caffe_pb_h_no_generate;
)

:caffe_pb_h_generate

echo caffe.proto is being generated
"../3rdparty/bin/protoc" -I="../src/caffe/proto" --cpp_out="../src/caffe/proto" "../src/caffe/proto/caffe.proto"
echo %TIMESTAMP_NEW%> %FILE_TIMESTAMP%

goto caffe_pb_h_end;

:caffe_pb_h_no_generate

echo caffe.proto remains the same as before

goto caffe_pb_h_end;

:caffe_pb_h_end
echo %~nx0 end
