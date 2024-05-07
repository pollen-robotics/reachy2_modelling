#!/usr/bin/bash
set -xe

rm -vf */*.csv

./bagdir_to_csv.py subset
./bagdir_to_csv.py subset2
./bagdir_to_csv.py subset3
./bagdir_to_csv.py subset5
./bagdir_to_csv.py subset6

dirpath="centrage_l_r/reachy.bag/"
if [ -d ${dirpath} ]; then
    echo "Directory exists: ${dirpath}"
    mv -v ${dirpath}/* centrage_l_r/
    rmdir ${dirpath}
fi
./bagdir_to_csv.py centrage_l_r
./bagdir_to_csv.py multiturns
./bagdir_to_csv.py unbiased

./verify_csvs.sh

wc -l */*.csv
ps -A | ag 'ros|python'
