#!/usr/bin/bash
set -xe

larmf="l_arm_combined.csv"
rarmf="r_arm_combined.csv"

head -n1 subset/l_arm_target_pose.csv > ${larmf}
tail -n +2 */l_arm*.csv >> ${larmf}

head -n1 subset/r_arm_target_pose.csv > ${rarmf}
tail -n +2 */r_arm*.csv >> ${rarmf}
