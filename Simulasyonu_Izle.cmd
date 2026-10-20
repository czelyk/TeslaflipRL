@echo off
call "%~dp0run.cmd" watch.py --controller stabilize --camera follow --episodes 5
pause
