T1（カメラ）

source /opt/ros/jazzy/setup.bash  
ros2 run image_tools cam2image --ros-args -p device_id:=0


T2（追尾ノード）

source /opt/ros/jazzy/setup.bash  
source ~/pin_ws/install/setup.bash    
ros2 run pin_recognition pin_tracker_node --ros-args -p image_topic:=/image


T3（確認）

source /opt/ros/jazzy/setup.bash  
source ~/pin_ws/install/setup.bash    
ros2 topic echo /cmd_vel


ビルド
cd ~/pin_ws  
colcon build --packages-select pin_recognition --symlink-install  
