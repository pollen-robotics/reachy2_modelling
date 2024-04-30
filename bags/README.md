
### quickstart

```bash
# extract
$ unzip subset5.zip
$ ls subset5
metadata.yaml  subset5_0.db3

# get the poses output per arm
$ ./bagdir_to_csv.py subset5
$ ls subset5
l_arm_target_pose.csv  metadata.yaml  r_arm_target_pose.csv  subset5_0.db3

# compute the configurations (q) with symbolic ik from the poses output per arm
$ posecsv_to_symik.py subset5/l_arm_target_pose.csv

# some manipulability tests when multiturn is detected
$ ./manip_analysis.py subset5/l_arm_target_pose.csv
```

#### get csv files with infos

```bash
# UNZIP all files to see directories
$ ls
subset3    centrage_l_r   subset   subset5   subset2  subset6

# generate csv files for all rosbags
$ ./all_bagtocsv.sh

# generated files
$ ls */*.csv
centrage_l_r/l_arm_target_pose.csv  subset5/l_arm_target_pose.csv
centrage_l_r/r_arm_target_pose.csv  subset5/r_arm_target_pose.csv
subset2/l_arm_target_pose.csv       subset6/l_arm_target_pose.csv
subset2/r_arm_target_pose.csv       subset6/r_arm_target_pose.csv
subset3/l_arm_target_pose.csv       subset/l_arm_target_pose.csv
subset3/r_arm_target_pose.csv       subset/r_arm_target_pose.csv
```

