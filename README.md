T1（カメラ）

source ~/pin_ws/setup.bash  
ros2 run image_tools cam2image --ros-args -p device_id:=0


T2（追尾ノード）

source ~/pin_ws/setup.bash  
source ~/ros2_ws/install/setup.bash  
ros2 run pin_recognition pin_tracker_node --ros-args -p image_topic:=/image


T3（確認）

source ~/pin_ws/setup.bash  
ros2 topic echo /cmd_vel
