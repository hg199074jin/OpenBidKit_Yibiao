@echo off
cd /d "%~dp0client"
set YIBIAO_OPENCODE_BIN=D:\opencode\opencode.exe
set PATH=D:\nodejs;D:\Git\cmd;%PATH%
D:\nodejs\npm.cmd run dev
pause
