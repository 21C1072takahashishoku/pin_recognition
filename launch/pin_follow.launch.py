from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    cam = Node(
        package="image_tools",
        executable="cam2image",
        name="cam2image",
        output="screen",
        parameters=[{"device_id": 0}],
        remappings=[
            ("image", "/image"),
            ("camera_info", "/camera_info"),
        ],
    )

    tracker = Node(
        package="pin_recognition",
        executable="pin_tracker_node",
        name="pin_tracker_node",
        output="screen",
        parameters=[{"image_topic": "/image"}],
    )

    return LaunchDescription([cam, tracker])
