#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
easytracker_node.py

・/camera/image_raw を購読
・画像中の「一番大きい白い縦長物体」をピンとみなす
・ピンの左右位置に応じて /cmd_vel を出す簡易版トラッカー
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class EasyTrackerNode(Node):
    def __init__(self):
        super().__init__("easytracker_node")

        # ==== パラメータ ====
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("linear_speed", 0.2)        # [m/s]
        self.declare_parameter("angular_gain", 1.0)        # 角速度ゲイン
        self.declare_parameter("max_angular_speed", 1.0)   # [rad/s]
        self.declare_parameter("search_angular_speed", 0.4)
        self.declare_parameter("debug_view", True)

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        self.W = self.get_parameter("image_width").get_parameter_value().integer_value
        self.H = self.get_parameter("image_height").get_parameter_value().integer_value
        self.linear_speed = self.get_parameter("linear_speed").get_parameter_value().double_value
        self.angular_gain = self.get_parameter("angular_gain").get_parameter_value().double_value
        self.max_angular_speed = self.get_parameter("max_angular_speed").get_parameter_value().double_value
        self.search_angular_speed = self.get_parameter("search_angular_speed").get_parameter_value().double_value
        self.debug_view = self.get_parameter("debug_view").get_parameter_value().bool_value

        self.bridge = CvBridge()

        self.sub_image = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10
        )
        self.pub_cmd_vel = self.create_publisher(
            Twist, self.cmd_vel_topic, 10
        )

        self.get_logger().info(
            f"EasyTrackerNode started. Subscribing: {self.image_topic}, Publishing: {self.cmd_vel_topic}"
        )

    def image_callback(self, msg: Image):
        """画像を受け取ってピン検出 → /cmd_vel 出力"""
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        h, w = frame_bgr.shape[:2]
        self.W, self.H = w, h

        vis, err_x = self.detect_pin(frame_bgr)

        twist = Twist()

        if err_x is not None:
            # 画像中心からの誤差を正規化
            norm_err = err_x / (self.W / 2.0)
            ang = -self.angular_gain * norm_err

            # 角速度制限
            if ang > self.max_angular_speed:
                ang = self.max_angular_speed
            if ang < -self.max_angular_speed:
                ang = -self.max_angular_speed
            twist.angular.z = ang

            # 正面近くなら前進、それ以外は回頭優先
            if abs(norm_err) < 0.4:
                twist.linear.x = self.linear_speed
            else:
                twist.linear.x = 0.0
        else:
            # 見つからないときはグルグル回って探す
            twist.linear.x = 0.0
            twist.angular.z = self.search_angular_speed

        self.pub_cmd_vel.publish(twist)

        if self.debug_view:
            cv2.imshow("easytracker_debug", vis)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()

    # ====== 超シンプルなピン検出 ======
    def detect_pin(self, bgr: np.ndarray) -> Tuple[np.ndarray, Optional[int]]:
        """
        - グレースケール＋しきい値で明るい部分を抽出
        - 一番大きい縦長の輪郭を「ピン」とみなす
        - 画像中心からの横方向誤差 err_x を返す
        """
        vis = bgr.copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 明るい領域を抽出（環境次第で閾値150は調整）
        _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_area = 0.0
        best_box = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(h) / float(w + 1e-6)

            # 縦長のみ候補
            if ratio < 1.5:
                continue

            if area > best_area:
                best_area = area
                best_box = (x, y, w, h)

        err_x: Optional[int] = None

        if best_box is not None:
            x, y, w, h = best_box
            cx = x + w // 2

            # 描画
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.line(vis, (cx, y), (cx, y + h), (0, 255, 0), 1)
            cv2.line(vis, (self.W // 2, 0), (self.W // 2, self.H), (255, 0, 0), 1)

            err_x = cx - (self.W // 2)

            cv2.putText(vis, f"PIN area={best_area:.0f}", (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(vis, f"err_x={err_x:+d}px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(vis, "SEARCH: no bright vertical object", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        return vis, err_x


def main(args=None):
    rclpy.init(args=args)
    node = EasyTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
