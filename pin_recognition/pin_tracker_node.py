#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def tilt_deg_from_v(vx: float, vy: float) -> float:
    """
    fitLineで得た方向ベクトル(vx,vy)が「垂直から何度傾いているか」を返す。
    0deg=完全垂直、90deg=完全水平
    """
    ang = abs(np.degrees(np.arctan2(vy, vx)))  # 0..180
    # 垂直(90deg)との差
    return abs(90.0 - ang)


class PinTrackerNode(Node):
    def __init__(self):
        super().__init__("pin_tracker_node")

        # -------------------------
        # Params
        # -------------------------
        self.declare_parameter("image_topic", "/image")

        # view/publish
        self.declare_parameter("show_windows", True)
        self.declare_parameter("publish_annotated", True)

        # control
        self.declare_parameter("angular_gain", 1.5)
        self.declare_parameter("max_angular_speed", 0.3)#1.2から変更
        self.declare_parameter("search_yaw_rate", 0.3)
        self.declare_parameter("linear_speed", 0.03)#0.15から変更
        self.declare_parameter("target_distance", 0.6)
        self.declare_parameter("distance_margin", 0.05)

        # distance model
        self.declare_parameter("pin_height_m", 0.12)  # 12cm
        self.declare_parameter("focal_px", 800.0)

        # HSV チューニング（赤帯）
        self.declare_parameter("tune_hsv", False)
        self.declare_parameter("h_low", 170)
        self.declare_parameter("h_high", 10)   # wrap対応
        self.declare_parameter("s_low", 60)
        self.declare_parameter("v_low", 40)

        # red band contour
        self.declare_parameter("min_red_area", 30)
        self.declare_parameter("red_open_iters", 1)
        self.declare_parameter("red_close_iters", 2)

        # ROI expansion around red band (白本体復元用)
        self.declare_parameter("roi_expand_x", 3.0)
        self.declare_parameter("roi_expand_up", 8.0)
        self.declare_parameter("roi_expand_dn", 6.0)

        # white extraction in ROI (黒背景想定)
        self.declare_parameter("white_min_threshold", 100)  # Otsu下限
        self.declare_parameter("white_open_iters", 2)
        self.declare_parameter("white_close_iters", 1)

        # pin shape filter (基本)
        self.declare_parameter("min_pin_aspect", 2.0)
        self.declare_parameter("min_pin_height_ratio", 0.06)  # 画像高さ比

        # --- 追加：倒れピン除外＆上ズレ対策 ---
        # 倒れピン除外：fitLineの傾き（垂直からの許容角度）
        self.declare_parameter("upright_max_tilt_deg", 22.0)  # 0=厳しい、30=緩い
        # 床接地（立ってるなら下端が画面下側に来やすい）
        self.declare_parameter("floor_min_ratio", 0.55)       # 下端がH*ratioより下
        # 上ズレ対策：赤帯がBBoxの中で「極端に下寄り/上寄り」なら弾く
        # （赤帯が上方に出る/白BBoxが上に引っ張られる等の異常を排除）
        self.declare_parameter("band_rel_min", 0.35)          # 0=上端, 1=下端
        self.declare_parameter("band_rel_max", 0.95)

        # 白BBoxの上下端を“赤帯中心ストリップ”で再推定する幅係数（重要）
        self.declare_parameter("refine_strip_half_factor", 0.8)  # band幅の何倍を半幅にするか

        # -------------------------
        # Read params
        # -------------------------
        self.image_topic = self.get_parameter("image_topic").value

        self.show_windows = bool(self.get_parameter("show_windows").value)
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)

        self.angular_gain = float(self.get_parameter("angular_gain").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.search_yaw_rate = float(self.get_parameter("search_yaw_rate").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.target_distance = float(self.get_parameter("target_distance").value)
        self.distance_margin = float(self.get_parameter("distance_margin").value)

        self.pin_height_m = float(self.get_parameter("pin_height_m").value)
        self.focal_px = float(self.get_parameter("focal_px").value)

        self.tune_hsv = bool(self.get_parameter("tune_hsv").value)
        self.h_low = int(self.get_parameter("h_low").value)
        self.h_high = int(self.get_parameter("h_high").value)
        self.s_low = int(self.get_parameter("s_low").value)
        self.v_low = int(self.get_parameter("v_low").value)

        self.min_red_area = int(self.get_parameter("min_red_area").value)
        self.red_open_iters = int(self.get_parameter("red_open_iters").value)
        self.red_close_iters = int(self.get_parameter("red_close_iters").value)

        self.roi_expand_x = float(self.get_parameter("roi_expand_x").value)
        self.roi_expand_up = float(self.get_parameter("roi_expand_up").value)
        self.roi_expand_dn = float(self.get_parameter("roi_expand_dn").value)

        self.white_min_threshold = int(self.get_parameter("white_min_threshold").value)
        self.white_open_iters = int(self.get_parameter("white_open_iters").value)
        self.white_close_iters = int(self.get_parameter("white_close_iters").value)

        self.min_pin_aspect = float(self.get_parameter("min_pin_aspect").value)
        self.min_pin_height_ratio = float(self.get_parameter("min_pin_height_ratio").value)

        self.upright_max_tilt_deg = float(self.get_parameter("upright_max_tilt_deg").value)
        self.floor_min_ratio = float(self.get_parameter("floor_min_ratio").value)
        self.band_rel_min = float(self.get_parameter("band_rel_min").value)
        self.band_rel_max = float(self.get_parameter("band_rel_max").value)
        self.refine_strip_half_factor = float(self.get_parameter("refine_strip_half_factor").value)

        # -------------------------
        # ROS I/O
        # -------------------------
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos_profile_sensor_data
        )
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.annotated_pub = self.create_publisher(Image, "/pin_tracker/annotated", 10)

        self.frame_count = 0

        # HSV trackbars
        self._trackbar_ready = False
        if self.tune_hsv and self.show_windows:
            self._init_trackbars()

        self.get_logger().info(f"pin_tracker_node started. image_topic={self.image_topic}")
        self.get_logger().info(
            f"HSV red (wrap ok): h_low={self.h_low}, h_high={self.h_high}, s_low={self.s_low}, v_low={self.v_low} "
            f"(tune_hsv={self.tune_hsv})"
        )

    # -------------------------
    # Trackbars (HSV tuning)
    # -------------------------
    def _init_trackbars(self):
        cv2.namedWindow("pin_hsv_tuner", cv2.WINDOW_NORMAL)

        def nothing(_):  # noqa
            pass

        cv2.createTrackbar("H_low",  "pin_hsv_tuner", int(clamp(self.h_low, 0, 179)), 179, nothing)
        cv2.createTrackbar("H_high", "pin_hsv_tuner", int(clamp(self.h_high, 0, 179)), 179, nothing)
        cv2.createTrackbar("S_low",  "pin_hsv_tuner", int(clamp(self.s_low, 0, 255)), 255, nothing)
        cv2.createTrackbar("V_low",  "pin_hsv_tuner", int(clamp(self.v_low, 0, 255)), 255, nothing)
        self._trackbar_ready = True

    def _read_trackbars(self):
        if not self._trackbar_ready:
            return
        self.h_low = cv2.getTrackbarPos("H_low", "pin_hsv_tuner")
        self.h_high = cv2.getTrackbarPos("H_high", "pin_hsv_tuner")
        self.s_low = cv2.getTrackbarPos("S_low", "pin_hsv_tuner")
        self.v_low = cv2.getTrackbarPos("V_low", "pin_hsv_tuner")

    # -------------------------
    # Red mask (wrap supported)
    # -------------------------
    def make_red_mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hL, hH = int(self.h_low), int(self.h_high)
        sL, vL = int(self.s_low), int(self.v_low)

        if hL <= hH:
            lower = np.array([hL, sL, vL], dtype=np.uint8)
            upper = np.array([hH, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
        else:
            # wrap: [0..hH] OR [hL..179]
            lower1 = np.array([0,  sL, vL], dtype=np.uint8)
            upper1 = np.array([hH, 255, 255], dtype=np.uint8)
            lower2 = np.array([hL, sL, vL], dtype=np.uint8)
            upper2 = np.array([179, 255, 255], dtype=np.uint8)
            mask = cv2.bitwise_or(
                cv2.inRange(hsv, lower1, upper1),
                cv2.inRange(hsv, lower2, upper2),
            )

        k3 = np.ones((3, 3), np.uint8)
        if self.red_open_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=self.red_open_iters)
        if self.red_close_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=self.red_close_iters)
        return mask

    # -------------------------
    # White in ROI (black background)
    # -------------------------
    def make_white_mask_in_roi(self, bgr_roi: np.ndarray) -> Tuple[np.ndarray, int]:
        gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
        t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = int(max(int(t), self.white_min_threshold))
        _, bw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

        k3 = np.ones((3, 3), np.uint8)
        k5 = np.ones((5, 5), np.uint8)
        if self.white_open_iters > 0:
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k3, iterations=self.white_open_iters)
        if self.white_close_iters > 0:
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k5, iterations=self.white_close_iters)
        return bw, t

    # -------------------------
    # Detect pin by "red band anchor"
    # -------------------------
    def detect_pin(self, bgr: np.ndarray) -> Tuple[np.ndarray, Optional[int], Optional[float]]:
        vis = bgr.copy()
        H, W = bgr.shape[:2]

        # update HSV from trackbars
        if self.tune_hsv and self.show_windows:
            self._read_trackbars()

        # 1) detect red band
        red = self.make_red_mask(bgr)
        contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0.0
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < self.min_red_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            # “帯”っぽさ：縦に長すぎる赤は避ける
            if h > w * 4:
                continue
            if area > best_area:
                best_area = area
                best = (x, y, w, h)

        cv2.line(vis, (W // 2, 0), (W // 2, H), (255, 0, 0), 1)

        if best is None:
            cv2.putText(
                vis,
                f"SEARCH (no red) HSV=({self.h_low}-{self.h_high},{self.s_low},{self.v_low})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )
            return vis, None, None

        x, y, bw, bh = best
        band_cx = x + bw // 2
        band_cy = y + bh // 2
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

        # 2) build ROI around band to recover white body
        ex = int(self.roi_expand_x * bw)
        eup = int(self.roi_expand_up * max(1, bh))
        edn = int(self.roi_expand_dn * max(1, bh))

        x0 = int(clamp(x - ex, 0, W - 1))
        x1 = int(clamp(x + bw + ex, 0, W - 1))
        y0 = int(clamp(y - eup, 0, H - 1))
        y1 = int(clamp(y + bh + edn, 0, H - 1))

        if x1 <= x0 + 2 or y1 <= y0 + 2:
            cv2.putText(vis, "ROI invalid", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis, None, None

        # ROI描画（通常画像の上に枠を描くだけ）
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 1)

        roi = bgr[y0:y1, x0:x1]
        white_bw, white_thr = self.make_white_mask_in_roi(roi)

        # connected components in ROI
        n, labels, stats, _ = cv2.connectedComponentsWithStats(white_bw, connectivity=8)

        # red band bbox in ROI coords
        bx0, by0 = x - x0, y - y0
        bx1, by1 = (x + bw) - x0, (y + bh) - y0

        best_id = None
        best_score = -1.0
        min_pin_h = int(self.min_pin_height_ratio * H)

        # ---- (A) 連結成分で白本体を選ぶ ----
        for i in range(1, n):
            xx, yy, ww, hh, area = stats[i]
            if hh < min_pin_h or ww <= 0:
                continue

            aspect = hh / float(max(1, ww))
            if aspect < self.min_pin_aspect:
                continue

            # overlap with red band bbox
            ox0 = max(xx, bx0); oy0 = max(yy, by0)
            ox1 = min(xx + ww, bx1); oy1 = min(yy + hh, by1)
            overlap = max(0, ox1 - ox0) * max(0, oy1 - oy0)

            # band中心に近い成分を優先（分断・誤連結に強い）
            band_cx_roi = (bx0 + bx1) * 0.5
            comp_cx = xx + ww * 0.5
            cx_dist = abs(comp_cx - band_cx_roi)

            # スコア：高さ優先 + 少し面積 + 少し重なり - 中心ズレ
            score = 2.2 * hh + 0.0015 * area + 0.01 * overlap - 0.7 * cx_dist
            if score > best_score:
                best_score = score
                best_id = i

        # ---- (B) 連結成分がダメなら、ストリップ走査で復元 ----
        use_strip_only = False
        if best_id is None:
            use_strip_only = True

        # 赤帯中心付近の縦ストリップ（ROI座標）
        strip_half = max(2, int(bw * float(self.refine_strip_half_factor)))
        sx0 = int(clamp((band_cx - strip_half) - x0, 0, white_bw.shape[1] - 1))
        sx1 = int(clamp((band_cx + strip_half) - x0, 0, white_bw.shape[1] - 1))
        if sx1 <= sx0 + 1:
            sx0 = max(0, sx0 - 2)
            sx1 = min(white_bw.shape[1] - 1, sx1 + 2)

        # ここから「pin_x,pin_y,pin_w,pin_h」を決める
        pin_x = pin_y = pin_w = pin_h = 0
        tilt_deg = None  # 倒れ判定用

        if use_strip_only:
            strip = white_bw[:, sx0:sx1]
            ys, xs = np.where(strip > 0)

            if ys.size == 0:
                cv2.putText(vis, f"FOUND RED but no WHITE (thr={white_thr})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
                return vis, None, None

            y_top_roi = int(ys.min())
            y_bot_roi = int(ys.max())
            pin_h = int(y_bot_roi - y_top_roi)
            if pin_h < min_pin_h:
                cv2.putText(vis, f"WHITE too small (h={pin_h}px thr={white_thr})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
                return vis, None, None

            pin_x = int(x0 + sx0)
            pin_y = int(y0 + y_top_roi)
            pin_w = int(max(6, (sx1 - sx0)))
            pin_h = int(max(1, pin_h))

            # 倒れ判定（ストリップ内点でfitLine）
            if ys.size >= 60:
                pts = np.column_stack([xs, ys]).astype(np.float32)  # strip座標
                vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                tilt_deg = float(tilt_deg_from_v(float(vx), float(vy)))

        else:
            # 連結成分で確定
            xx, yy, ww, hh, _ = stats[best_id]
            comp_mask = (labels == best_id)

            # ★上ズレ対策の本命：
            # 連結成分BBoxそのままだと上に引っ張られるので、
            # 「赤帯中心ストリップ ∩ comp_mask」で上下端を再推定する
            strip_mask = comp_mask[:, sx0:sx1]
            ys, xs = np.where(strip_mask)

            if ys.size >= 20:
                y_top_roi = int(ys.min())
                y_bot_roi = int(ys.max())
                pin_y = int(y0 + y_top_roi)
                pin_h = int(max(1, y_bot_roi - y_top_roi))
                # 幅は「連結成分の幅」を使う（過剰に細くしない）
                pin_x = int(x0 + xx)
                pin_w = int(max(1, ww))
            else:
                # ストリップ内に点が無いなら従来BBoxに戻す
                pin_x = int(x0 + xx)
                pin_y = int(y0 + yy)
                pin_w = int(max(1, ww))
                pin_h = int(max(1, hh))

            # 倒れ判定（連結成分の点でfitLine）
            ys2, xs2 = np.where(comp_mask)
            if ys2.size >= 120:
                pts2 = np.column_stack([xs2, ys2]).astype(np.float32)
                vx, vy, _, _ = cv2.fitLine(pts2, cv2.DIST_L2, 0, 0.01, 0.01)
                tilt_deg = float(tilt_deg_from_v(float(vx), float(vy)))

        # -------------------------
        # ここから「採用するか」判定（倒れピン＆異常BBox排除）
        # -------------------------

        # (1) 縦長チェック（倒れを弾く最初の壁）
        aspect = pin_h / float(max(1, pin_w))
        if aspect < self.min_pin_aspect:
            cv2.putText(vis, f"REJECT fallen(aspect={aspect:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis, None, None

        # (2) 床接地チェック（倒れてる/空中ノイズを排除）
        if (pin_y + pin_h) < int(H * self.floor_min_ratio):
            cv2.putText(vis, "REJECT not on floor",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis, None, None

        # (3) 赤帯の相対位置チェック（上ズレ・異常BBoxの排除）
        # 0=上端, 1=下端
        band_rel = (float(band_cy) - float(pin_y)) / float(max(1, pin_h))
        if not (self.band_rel_min <= band_rel <= self.band_rel_max):
            cv2.putText(vis, f"REJECT band_rel={band_rel:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis, None, None

        # (4) fitLineの傾きで倒れを確実に弾く（点が十分ある場合のみ）
        if tilt_deg is not None and tilt_deg > self.upright_max_tilt_deg:
            cv2.putText(vis, f"REJECT tilt={tilt_deg:.1f}deg",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis, None, None

        # -------------------------
        # 採用：出力計算
        # -------------------------
        cx_out = int(pin_x + pin_w / 2.0)
        dist_out = (self.pin_height_m * float(self.focal_px)) / float(max(1, pin_h))

        # draw pin bbox
        cv2.rectangle(vis, (pin_x, pin_y), (pin_x + pin_w, pin_y + pin_h), (0, 255, 0), 2)
        cv2.circle(vis, (cx_out, int(pin_y + pin_h / 2)), 6, (0, 255, 0), -1)

        msg1 = f"PIN d={dist_out:.2f}m  h={pin_h}px  thr={white_thr}  band_rel={band_rel:.2f}  asp={aspect:.2f}"
        if tilt_deg is not None:
            msg1 += f"  tilt={tilt_deg:.1f}deg"
        cv2.putText(vis, msg1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

        msg2 = f"HSV=({self.h_low}-{self.h_high},{self.s_low},{self.v_low}) ROI(up={self.roi_expand_up},dn={self.roi_expand_dn})"
        cv2.putText(vis, msg2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        return vis, cx_out, dist_out

    # -------------------------
    # ROS callback
    # -------------------------
    def image_callback(self, msg: Image):
        self.frame_count += 1

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        H, W = bgr.shape[:2]
        annotated, cx, dist = self.detect_pin(bgr)

        if self.publish_annotated:
            try:
                out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                out.header = msg.header
                self.annotated_pub.publish(out)
            except Exception as e:
                self.get_logger().warn(f"annotated publish error: {e}")

        if self.show_windows:
            try:
                cv2.imshow("pin_tracker_annotated", annotated)
                if self.tune_hsv:
                    cv2.imshow("pin_red_preview", cv2.cvtColor(self.make_red_mask(bgr), cv2.COLOR_GRAY2BGR))
                cv2.waitKey(1)
            except Exception:
                pass

        # control
        twist = Twist()
        if cx is None:
            twist.linear.x = 0.0
            twist.angular.z = float(self.search_yaw_rate)
        else:
            err = (float(cx) - (W / 2.0)) / (W / 2.0)
            wz = -self.angular_gain * err
            twist.angular.z = float(clamp(wz, -self.max_angular_speed, self.max_angular_speed))

            if dist is not None:
                if dist > self.target_distance + self.distance_margin:
                    twist.linear.x = float(self.linear_speed)
                elif dist < self.target_distance - self.distance_margin:
                    twist.linear.x = -float(self.linear_speed) * 0.5
                else:
                    twist.linear.x = 0.0

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = PinTrackerNode()
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
