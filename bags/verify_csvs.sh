#!/usr/bin/bash
set -x

./bagdir_to_csv.py subset --verify
./bagdir_to_csv.py subset2 --verify
./bagdir_to_csv.py subset3 --verify
./bagdir_to_csv.py subset5 --verify
./bagdir_to_csv.py subset6 --verify
./bagdir_to_csv.py centrage_l_r --verify
./bagdir_to_csv.py multiturns --verify
./bagdir_to_csv.py unbiased --verify
