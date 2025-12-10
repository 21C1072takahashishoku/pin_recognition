#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fake_pin_camera_node.py

物理カメラなしでテストするための擬似カメラノード。
- 黒背景に「白いピン＋赤い2本の帯」を描いた画像を生成
- /camera/image_raw に一定周期でパブリッシュ
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2


class FakePinCameraNode(Node):
    def __init__(self):
        super().__init__("fake_pin_camera_node")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("freq", 5.0)  # [Hz]

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.W = self.get_parameter("width").get_parameter_value().integer_value
        self.H = self.get_parameter("height").get_parameter_value().integer_value
        self.freq = self.get_parameter("freq").get_parameter_value().double_value

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, self.image_topic, 10)
        self.timer = self.create_timer(1.0 / self.freq, self.timer_callback)

        self.get_logger().info(f"FakePinCameraNode started. Publishing to {self.image_topic}")

    def timer_callback(self):
        # 黒背景
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        # 画面中央付近に白いピン（縦長の白い矩形）を描画
        pin_w = int(self.W * 0.08)
        pin_h = int(self.H * 0.5)
        cx = self.W // 2
        cy = int(self.H * 0.55)  # 少し下寄り

        x1 = cx - pin_w // 2
        x2 = cx + pin_w // 2
        y2 = cy + pin_h // 2
        y1 = cy - pin_h // 2

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # 赤帯を2本描く（上と中ほど）
        band_h = max(4, pin_h // 12)
        # 上の赤帯
        y_band1 = y1 + pin_h // 4
        cv2.rectangle(img, (x1, y_band1), (x2, y_band1 + band_h), (0, 0, 255), -1)
        # 下の赤帯
        y_band2 = y1 + (pin_h * 2) // 3
        cv2.rectangle(img, (x1, y_band2), (x2, y_band2 + band_h), (0, 0, 255), -1)

        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakePinCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
