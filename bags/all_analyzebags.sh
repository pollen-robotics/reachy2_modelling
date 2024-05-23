#!/usr/bin/bash
set -xe
start=`date +%s.%N`

time -p ./analyze_bagdir.py subset --force --images
time -p ./analyze_bagdir.py subset2 --force --images
time -p ./analyze_bagdir.py subset3 --force --images
time -p ./analyze_bagdir.py subset5 --force --images
time -p ./analyze_bagdir.py subset6 --force --images
time -p ./analyze_bagdir.py centrage_l_r --force --images
time -p ./analyze_bagdir.py multiturns --force --images
time -p ./analyze_bagdir.py unbiased --force --images

rm images/*
cp -v */*.png images

end=`date +%s.%N`

runtime=$( echo "$end - $start" | bc -l )
echo "Total Time: ${runtime}s"
