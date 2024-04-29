#!/usr/bin/bash
set -xe

rm -vf */*.csv

./bagtocsv.sh subset
./bagtocsv.sh subset2
./bagtocsv.sh subset3
./bagtocsv.sh subset5
./bagtocsv.sh subset6

dirpath="centrage_l_r/reachy.bag/"
if [ -d ${dirpath} ]; then
    echo "Directory exists: ${dirpath}"
    mv -v ${dirpath}/* centrage_l_r/
    rmdir ${dirpath}
fi
./bagtocsv.sh centrage_l_r/


wc -l */*.csv
ps -A | ag ros2
