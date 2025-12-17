from launch import LaunchDescription
from launch.actions import TimerAction, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    # ===== 調整ここだけ =====
    settle_sec = 30.0          # 投げて落ち着くまで待つ
    motor_pulse_sec = 1.0      # down後にどれくらいでstopするか（不要なら0）
    start_pin_after_sec = 0.5  # stop後にpin起動までの余裕

    # down（接地）
    call_down = ExecuteProcess(
        cmd=[
            "bash", "-lc",
            'ros2 service call /action_command linear_motor_msgs/srv/Act \'{action: "down"}\''
        ],
        output="screen"
    )

    # stop（任意：パルスにするなら使う）
    call_stop = ExecuteProcess(
        cmd=[
            "bash", "-lc",
            'ros2 service call /action_command linear_motor_msgs/srv/Act \'{action: "stop"}\''
        ],
        output="screen"
    )

    # 既存の pin_follow.launch.py を呼ぶ（あなたのpin_recogノード群をまとめて起動できる）
    pin_follow_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("pin_recognition"), "launch", "pin_follow.launch.py"])
        )
    )

    actions = []

    # settle後に接地
    actions.append(TimerAction(period=settle_sec, actions=[call_down]))

    # パルス運用なら stop を送る
    if motor_pulse_sec > 0.0:
        actions.append(TimerAction(period=settle_sec + motor_pulse_sec, actions=[call_stop]))
        actions.append(TimerAction(
            period=settle_sec + motor_pulse_sec + start_pin_after_sec,
            actions=[pin_follow_launch]
        ))
    else:
        # stopしない（downで接地させたまま）なら、down後少し待ってpin起動
        actions.append(TimerAction(
            period=settle_sec + start_pin_after_sec,
            actions=[pin_follow_launch]
        ))

    return LaunchDescription(actions)
