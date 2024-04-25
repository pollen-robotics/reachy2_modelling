#!/bin/sh

set -x

bagdir="$1"
echo "bagdir: $bagdir"


larmf="${bagdir}/l_arm_target_pose.csv"
rarmf="${bagdir}/r_arm_target_pose.csv"
echo "CSVS:  ${larmf}, ${rarmf}"
ros2 topic echo --csv /l_arm/target_pose > "${larmf}" &
ros2 topic echo --csv /r_arm/target_pose > "${rarmf}" &

ros2 bag play -r 100 $bagdir

# header:
# stamp:
# sec: 1709043931
# nanosec: 376224964
# frame_id: ''
# pose:
# position:
# x: 0.3464586138725281
# y: 0.17137935757637024
# z: -0.287570595741272
# orientation:
# x: 0.03157906025858677
# y: 0.709536301943057
# z: 0.18047085180472908
# w: -0.6804346190686003

header="stamp_sec,stamp_nanosec,stamp_frame_id,pos_x,pos_y,pos_z,or_x,or_y,or_z,or_w\n"
sed -i "1s/^/${header}/" ${larmf}
sed -i "1s/^/${header}/" ${rarmf}
