#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from linear_motor_msgs.srv import Mode


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def tilt_deg_from_v(vx: float, vy: float) -> float:
    """
    fitLineの方向ベクトル(vx, vy)が「垂直から何度傾いているか」を返す。
    0deg=完全垂直、90deg=完全水平
    """
    ang = abs(np.degrees(np.arctan2(vy, vx)))  # 0..180
    return abs(90.0 - ang)


def rect_upright_tilt_deg(rect) -> float:
    """
    minAreaRect の angle から「垂直からの傾き」を概算（0=垂直）。
    OpenCVの角度表現は癖があるので、ここで“垂直との差”に正規化する。
    """
    (_, _), (w, h), angle = rect

    # rectの定義により w<h のとき angleが入れ替わることがあるので正規化
    if w < h:
        w, h = h, w
        angle = angle + 90.0

    a = abs(angle)
    while a > 90.0:
        a -= 180.0
        a = abs(a)
    return abs(90.0 - a)


class WhiteFollowerNode(Node):
    """
    できるだけ単純：
      - 固定しきい値で白を2値化
      - 白の最大連結成分だけ見る
      - 直立性（倒れてない）を判定
      - 検出できたら重心に向けて旋回しながら前進
    """

    def __init__(self):
        super().__init__("white_follower_node")

        # -------------------------
        # Params
        # -------------------------
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("show_windows", False)
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("publish_debug_mask", False)

        # white detect (FIXED threshold)
        self.declare_parameter("white_fixed_threshold", 40)   # ←ゆるめ固定しきい値（30〜60で調整）
        self.declare_parameter("min_white_pixels", 400)        # 小さいノイズ除外

        self.declare_parameter("morph_open_iters", 0)          # ノイズ多いなら1
        self.declare_parameter("morph_close_iters", 1)         # 穴埋め 1〜2

        # upright gate (倒れている白を弾く)
        self.declare_parameter("upright_min_points", 80)
        self.declare_parameter("upright_aspect_min_far", 2.2)  # 遠いとき（細長く見えるはず）
        self.declare_parameter("upright_aspect_min_near", 0.5) # 近いとき（画角で崩れるので緩める）
        self.declare_parameter("upright_tilt_max_far", 25.0)   # 垂直からの許容角（小さいほど厳しい）
        self.declare_parameter("upright_tilt_max_near", 35.0)

        self.declare_parameter("near_major_ratio", 0.55)       # 近距離判定：major >= 0.55*H
        self.declare_parameter("near_area_ratio", 0.08)        # 近距離判定：area >= 0.08*HW

        # control
        self.declare_parameter("angular_gain", 1.5)
        self.declare_parameter("max_angular_speed", 0.4)
        self.declare_parameter("linear_speed", 1.0)
        self.declare_parameter("search_yaw_rate", 0.3)         # 見失った時（0なら停止）
        self.calib = True

        # -------------------------
        # Read params
        # -------------------------
        self.image_topic = self.get_parameter("image_topic").value
        self.show_windows = bool(self.get_parameter("show_windows").value)
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)
        self.publish_debug_mask = bool(self.get_parameter("publish_debug_mask").value)

        self.white_fixed_threshold = int(self.get_parameter("white_fixed_threshold").value)
        self.min_white_pixels = int(self.get_parameter("min_white_pixels").value)

        self.morph_open_iters = int(self.get_parameter("morph_open_iters").value)
        self.morph_close_iters = int(self.get_parameter("morph_close_iters").value)

        self.upright_min_points = int(self.get_parameter("upright_min_points").value)
        self.upright_aspect_min_far = float(self.get_parameter("upright_aspect_min_far").value)
        self.upright_aspect_min_near = float(self.get_parameter("upright_aspect_min_near").value)
        self.upright_tilt_max_far = float(self.get_parameter("upright_tilt_max_far").value)
        self.upright_tilt_max_near = float(self.get_parameter("upright_tilt_max_near").value)

        self.near_major_ratio = float(self.get_parameter("near_major_ratio").value)
        self.near_area_ratio = float(self.get_parameter("near_area_ratio").value)

        self.angular_gain = float(self.get_parameter("angular_gain").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.search_yaw_rate = float(self.get_parameter("search_yaw_rate").value)

        self.declare_parameter("keep_straight_duration", 3.0) # 何秒間直進し続けるか
        self.keep_straight_duration = self.get_parameter("keep_straight_duration").value
        self.last_detected_time = 0.0 # 最後に認識した時刻（ROSの時刻）

        # -------------------------
        # ROS I/O
        # -------------------------
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos_profile_sensor_data
        )

        self.srv = self.create_service(Mode, "tracker_mode", self.srv_callback)
        self.mode = "OFF"

        self.bgr = None
        self.header = None

        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.annotated_pub = self.create_publisher(Image, "/white_follower/annotated", 1)
        self.detected_pub = self.create_publisher(Bool, "/white_follower/detected", 1)
        self.mask_pub = self.create_publisher(Image, "/white_follower/white_mask", 1)

        self.get_logger().info(
            f"white_follower_node started. image_topic={self.image_topic} "
            f"thr_fixed={self.white_fixed_threshold} min_white_pixels={self.min_white_pixels}"
        )

    # -------------------------
    # White mask (FIXED threshold)
    # -------------------------
    def make_white_mask(self, bgr: np.ndarray):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 固定しきい値（値を下げるほど“ゆるい”）
        t = 130
        _, bw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

        # 形態学（任意）
        k3 = np.ones((3, 3), np.uint8)
        if self.morph_open_iters > 0:
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k3, iterations=self.morph_open_iters)
        if self.morph_close_iters > 0:
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k3, iterations=self.morph_close_iters)

        return bw, t

    # -------------------------
    # Upright white detection (largest CC)
    # -------------------------
    def detect_upright_white(self, bw: np.ndarray, H: int, W: int):
        """
        白を検出するが、倒れている（水平に近い）白塊は弾く。
        戻り値: (detected:bool, cx:int|None, cy:int|None, debug:str)
        """

        n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        if n <= 1:
            return False, None, None, "no_cc"

        best_i = -1
        best_area = -1
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_i = i

        if best_i < 0 or best_area < self.min_white_pixels:
            return False, None, None, "too_small"

        comp = (labels == best_i)
        ys, xs = np.where(comp)
        if ys.size < self.upright_min_points:
            return False, None, None, "few_pts"

        pts = np.column_stack([xs, ys]).astype(np.float32)

        rect = cv2.minAreaRect(pts)
        (_, _), (rw, rh), _ = rect
        major = float(max(rw, rh))
        minor = float(max(1.0, min(rw, rh)))
        aspect = major / minor

        tilt_rect = rect_upright_tilt_deg(rect)

        is_near = (major >= self.near_major_ratio * H) or (best_area >= self.near_area_ratio * (H * W))
        aspect_min = self.upright_aspect_min_near if is_near else self.upright_aspect_min_far
        tilt_max = self.upright_tilt_max_near if is_near else self.upright_tilt_max_far

        if aspect < aspect_min:
            return False, None, None, f"aspect_ng({aspect:.2f}<{aspect_min})"
        if tilt_rect > tilt_max:
            return False, None, None, f"tilt_ng({tilt_rect:.1f}>{tilt_max})"

        vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        tilt_fit = float(tilt_deg_from_v(float(vx), float(vy)))
        if tilt_fit > tilt_max:
            return False, None, None, f"tiltfit_ng({tilt_fit:.1f}>{tilt_max})"

        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        return True, cx, cy, f"ok asp={aspect:.2f} tiltR={tilt_rect:.1f} tiltF={tilt_fit:.1f}"

    # -------------------------
    # ROS callback
    # -------------------------
    def srv_callback(self, request, response):
        if request.mode == "tracker_START":
            self.mode = "ON"
            self.get_logger().info("tracker_mode: ON")
        elif request.mode == "tracker_STOP":
            self.mode = "OFF"
            self.get_logger().info("tracker_mode: OFF")
        return response

    def image_callback(self, msg: Image):
        try:
            self.bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.header = msg.header
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")

    def timer_callback(self):
        if self.bgr is None or self.mode == "OFF":
            return

        H, W = self.bgr.shape[:2]

        bw, thr = self.make_white_mask(self.bgr)
        detected, cx, cy, dbg = self.detect_upright_white(bw, H, W)

        self.detected_pub.publish(Bool(data=bool(detected)))

        # publish mask (optional)
        if self.publish_debug_mask and self.header is not None:
            try:
                m = self.bridge.cv2_to_imgmsg(bw, encoding="mono8")
                m.header = self.header
                self.mask_pub.publish(m)
            except Exception:
                pass

        annotated = self.bgr.copy()
        cv2.line(annotated, (W // 2, 0), (W // 2, H), (255, 0, 0), 1)

        if detected and cx is not None and cy is not None:
            cv2.circle(annotated, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(
                annotated,
                f"WHITE UPRIGHT thr={thr} {dbg} cx={cx}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2
            )
        else:
            cv2.putText(
                annotated,
                f"SEARCH thr={thr} {dbg}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 180, 255), 2
            )

        # publish annotated
        if self.publish_annotated and self.header is not None:
            try:
                out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                out.header = self.header
                self.annotated_pub.publish(out)
            except Exception:
                pass

        # local view
        if self.show_windows:
            try:
                cv2.imshow("white_follower_annotated", annotated)
                cv2.imshow("white_mask", bw)
                cv2.waitKey(1)
            except Exception:
                pass

        # control
        now_sec = self.get_clock().now().nanoseconds / 1e9  # 現在時刻(秒)

        # ターゲットを認識できた場合、時刻を更新
        if detected and cx is not None:
            self.last_detected_time = now_sec

        # 「直進を維持する時間内」かどうかを判定
        is_searching = (now_sec - self.last_detected_time) > self.keep_straight_duration
        
        twist = Twist()

        if is_searching:
            # 【探索モード】5秒以上見つからないので、止まって旋回
            twist.linear.x = 0.0
            twist.angular.z = float(self.search_yaw_rate)
            self.calib = True
        else:
            # 【直進・追従モード】見つけてから5秒以内
            if self.calib and detected:
                # 最初に発見した瞬間だけのキャリブレーション動作
                twist.linear.x = 0.0
                twist.angular.z = -1.0 * float(self.search_yaw_rate)
                self.calib = False
            else:
                # ★ここがメインの直進★
                # 見えていれば cx で計算できるが、見失っていても linear.x を出す
                twist.linear.x = float(self.linear_speed)
                twist.angular.z = 0.0

#        if not detected or cx is None and not is_keep_straight:
#           twist.linear.x = 0.0
#            twist.angular.z = float(self.search_yaw_rate)
#            self.calib = True
#        else:
 #           if self.calib is True:
  #              twist.linear.x = 0.0
   #             twist.angular.z = -1.0*float(self.search_yaw_rate)
    #            self.calib = False
     #       else:
      #          err = (float(cx) - (W / 2.0)) / (W / 2.0)  # -1..1
       #         wz = -self.angular_gain * err
                #twist.angular.z = float(clamp(wz, -self.max_angular_speed, self.max_angular_speed))
        #        twist.angular.z = 0.0
         #       twist.linear.x = float(self.linear_speed)

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = WhiteFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()

