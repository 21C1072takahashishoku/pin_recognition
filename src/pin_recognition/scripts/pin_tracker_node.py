#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pin_tracker_node.py

ROS2ノード:
- カメラ画像から立っているピンを検出（赤系＋縦長）
- ピンに正対するように回頭しつつ前進
- ピン高さから距離を推定し、一定距離まで近づいたら前進を止める
- 一瞬見失っても、数フレームは過去位置で追尾を継続する

依存:
  - rclpy
  - sensor_msgs.msg.Image
  - geometry_msgs.msg.Twist
  - cv_bridge
  - OpenCV (cv2)
  - numpy
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class PinTrackerNode(Node):
    def __init__(self):
        super().__init__("pin_tracker_node")

        # ==== パラメータ ====
        # トピック
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        # カメラパラメータ
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("hfov_deg", 68.0)      # カメラの水平FOV[deg]

        # ピンの実高さ [m]（実物に合わせて調整）
        self.declare_parameter("pin_height_m", 0.38)  # 例: ボウリングピン約38cm

        # 制御パラメータ
        self.declare_parameter("linear_speed", 0.2)       # [m/s]
        self.declare_parameter("angular_gain", 1.0)       # 角速度ゲイン
        self.declare_parameter("max_angular_speed", 1.0)  # [rad/s] 上限
        self.declare_parameter("stop_distance", 0.5)      # [m] この距離以内で前進停止
        self.declare_parameter("search_angular_speed", 0.4)  # 探索時の回頭速度

        # デバッグ表示
        self.declare_parameter("debug_view", True)

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        self.W = self.get_parameter("image_width").get_parameter_value().integer_value
        self.H = self.get_parameter("image_height").get_parameter_value().integer_value
        self.hfov_deg = self.get_parameter("hfov_deg").get_parameter_value().double_value
        self.pin_height_m = self.get_parameter("pin_height_m").get_parameter_value().double_value
        self.linear_speed = self.get_parameter("linear_speed").get_parameter_value().double_value
        self.angular_gain = self.get_parameter("angular_gain").get_parameter_value().double_value
        self.max_angular_speed = self.get_parameter("max_angular_speed").get_parameter_value().double_value
        self.stop_distance = self.get_parameter("stop_distance").get_parameter_value().double_value
        self.search_angular_speed = self.get_parameter("search_angular_speed").get_parameter_value().double_value
        self.debug_view = self.get_parameter("debug_view").get_parameter_value().bool_value

        # カメラの焦点距離（画素）推定
        hfov_rad = math.radians(self.hfov_deg)
        self.focal_px = (self.W / 2.0) / math.tan(hfov_rad / 2.0)

        self.bridge = CvBridge()

        # 追尾用の「過去の検出位置・高さ・距離」を保持
        self.last_cx: Optional[int] = None
        self.last_h_pix: Optional[float] = None
        self.last_dist_m: Optional[float] = None
        self.missed_frames: int = 0
        self.max_missed_frames: int = 5  # 何フレームまでは「前回の位置」で追尾し続けるか

        # サブスク / パブリッシュ
        self.sub_image = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10
        )
        self.pub_cmd_vel = self.create_publisher(
            Twist, self.cmd_vel_topic, 10
        )

        self.get_logger().info(
            f"PinTrackerNode started. Subscribing: {self.image_topic}, Publishing: {self.cmd_vel_topic}"
        )

    # ====== 画像コールバック ======
    def image_callback(self, msg: Image):
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        h, w = frame_bgr.shape[:2]
        self.W, self.H = w, h

        vis, err_x, dist_m = self.detect_pin(frame_bgr)

        twist = Twist()

        if err_x is not None:
            # ピンが見えている（or 直近の位置を利用中）の場合
            norm_err = err_x / (self.W / 2.0)  # -1〜+1程度
            ang = -self.angular_gain * norm_err  # 左が+zになるよう符号調整

            # 角速度制限
            ang = max(-self.max_angular_speed, min(self.max_angular_speed, ang))
            twist.angular.z = ang

            # 距離が推定できていて、かつ目標距離より近い → 前進停止
            if dist_m is not None and dist_m < self.stop_distance:
                twist.linear.x = 0.0
            else:
                # 常に少し前進しつつ、誤差が小さいほど速度を上げる
                base_speed = self.linear_speed
                speed_scale = max(0.0, 1.0 - abs(norm_err))  # |誤差|=1 → 0, 0 → 1
                # 下限を設けて「止まりきらない」ようにする
                min_scale = 0.2
                if speed_scale < min_scale:
                    speed_scale = min_scale
                twist.linear.x = base_speed * speed_scale
        else:
            # 完全に見失った → 探索モードでゆっくり回頭
            twist.linear.x = 0.0
            twist.angular.z = self.search_angular_speed

        self.pub_cmd_vel.publish(twist)

        if self.debug_view:
            cv2.imshow("pin_debug", vis)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()

    # ====== 色＋形状＋時間方向の追尾でピンを検出 ======
    def detect_pin(self, bgr: np.ndarray) -> Tuple[np.ndarray, Optional[int], Optional[float]]:
        """
        - ピンを赤色と仮定して HSV で色抽出
        - 画像下半分のみ探索
        - 一番大きい縦長輪郭をピンとみなす
        - 一瞬見失っても過去位置を数フレーム保持して err_x, dist を返す
        戻り値:
            vis: 描画用画像
            err_x: 画像中心からの横方向誤差 [px]（右が+）
            dist_m: 推定距離 [m]（推定できない場合 None）
        """
        vis = bgr.copy()
        h_img, w_img = bgr.shape[:2]
        self.H, self.W = h_img, w_img  # H=高さ, W=幅

        # ---- 1. BGR → HSV ----
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # ---- 2. 赤色マスク（少し広め＋S/Vそこそこ高め）----
        # 認識抜けが多いので、やや緩めの閾値にする
        lower_red1 = np.array([0,   90, 80], dtype=np.uint8)
        upper_red1 = np.array([12,  255, 255], dtype=np.uint8)
        lower_red2 = np.array([168, 90, 80], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # ---- 3. 画面の下半分だけを見る（床に立っているピン）----
        roi_start = int(h_img * 0.5)
        mask[:roi_start, :] = 0

        # ---- 4. ノイズ除去 ----
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # ---- 5. 輪郭抽出 ----
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_area = 0.0
        best_box = None

        # 調整用パラメータ：認識抜けが多いので少し緩め
        MIN_AREA = 800           # これでも抜けるなら 500 くらいまで下げる
        MIN_RATIO = 2.0          # h / w 最小（縦長度）
        MAX_RATIO = 10.0         # h / w 最大
        MIN_HEIGHT_RATIO = 0.10  # 画像高さに対する最小高さ

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(h) / float(w + 1e-6)

            if ratio < MIN_RATIO or ratio > MAX_RATIO:
                continue

            if h < h_img * MIN_HEIGHT_RATIO:
                continue

            if area > best_area:
                best_area = area
                best_box = (x, y, w, h)

        err_x: Optional[int] = None
        dist_m: Optional[float] = None

        if best_box is not None:
            # ---- 今フレームで新たに検出できた場合 ----
            x, y, w, h = best_box
            cx = x + w // 2

            # 位置・高さ・距離を記憶
            self.last_cx = cx
            self.last_h_pix = float(h)
            # 単純なピン高さからの距離推定
            if h > 0:
                dist_m = (self.pin_height_m * self.focal_px) / float(h)
            else:
                dist_m = None
            self.last_dist_m = dist_m
            self.missed_frames = 0

            # 描画
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.line(vis, (cx, y), (cx, y + h), (0, 255, 0), 1)
            cv2.line(vis, (self.W // 2, 0), (self.W // 2, self.H), (255, 0, 0), 1)

            err_x = cx - (self.W // 2)

            cv2.putText(vis, f"PIN area={best_area:.0f}", (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if dist_m is not None:
                cv2.putText(vis, f"dist={dist_m:.2f}m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # ---- 今フレームは検出できなかった場合 ----
            if self.last_cx is not None and self.missed_frames < self.max_missed_frames:
                # 直近の位置をそのまま使って追尾継続
                self.missed_frames += 1
                cx = self.last_cx
                err_x = cx - (self.W // 2)
                dist_m = self.last_dist_m

                cv2.line(vis, (cx, 0), (cx, self.H), (0, 200, 200), 1)
                cv2.putText(vis, f"TRACK (missed {self.missed_frames})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
            else:
                # 完全に見失った扱い
                self.last_cx = None
                self.last_h_pix = None
                self.last_dist_m = None
                self.missed_frames = 0
                cv2.putText(vis, "SEARCH: no red vertical object", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        return vis, err_x, dist_m


def main(args=None):
    rclpy.init(args=args)
    node = PinTrackerNode()
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
