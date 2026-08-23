@echo off
cd /d C:\VSC_Project\ConvoPeq\build-icx
ninja -t targets all > C:\VSC_Project\ConvoPeq\evidence\_vi_c_targets.txt 2>&1
echo DONE
