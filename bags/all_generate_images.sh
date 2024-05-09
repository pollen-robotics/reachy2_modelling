#!/usr/bin/bash
set -xe

time -p ./analyze_bagdir.py subset --images
echo "===================================================="
time -p ./analyze_bagdir.py subset2 --images
echo "===================================================="
time -p ./analyze_bagdir.py subset3 --images
echo "===================================================="
time -p ./analyze_bagdir.py subset5 --images
echo "===================================================="
time -p ./analyze_bagdir.py subset6 --images
echo "===================================================="
time -p ./analyze_bagdir.py centrage_l_r --images
echo "===================================================="
time -p ./analyze_bagdir.py multiturns --images
echo "===================================================="
time -p ./analyze_bagdir.py unbiased --images
