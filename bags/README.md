
### howto

```
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
