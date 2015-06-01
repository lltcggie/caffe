@echo off

cd "%~dp0"

echo copy include

if exist "../build/include" (
    rmdir /s /q "../build/include"
)

xcopy /E /I /Q "../include" "../build/include"
mkdir "..\build\include\caffe\proto\"
copy "..\src\caffe\proto\*.h" "..\build\include\caffe\proto\"
