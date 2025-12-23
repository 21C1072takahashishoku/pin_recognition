ビルド
cd ~/robot_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select pin_recognition --symlink-install
source ~/robot_ws/install/setup.bash


T1（カメラ）
source /opt/ros/jazzy/setup.bash
ros2 run image_tools cam2image --ros-args -p device_id:=0 -p width:=640 -p height:=480 -p show_camera:=false



T2（追尾ノード）

source /opt/ros/jazzy/setup.bash
source ~/robot_ws/install/setup.bash
ros2 run pin_recognition pin_tracker_node --ros-args -p image_topic:=/image


T3（確認）
source /opt/ros/jazzy/setup.bash
source ~/robot_ws/install/setup.bash
ros2 topic echo /cmd_vel

T4
1) まず ON にする（必須）

source /opt/ros/jazzy/setup.bash
source ~/robot_ws/install/setup.bash
ros2 service call /tracker_mode linear_motor_msgs/srv/Mode "{mode: 'tracker_START'}"


止める：

ros2 service call /tracker_mode linear_motor_msgs/srv/Mode "{mode: 'tracker_STOP'}"

