"""ROS 2 node (Humble and Jazzy): depth from the stereo pair already published on the drone (Task C).

    python3 -m onboard.depth.stereo_depth_node --ros-args --params-file onboard/config/stereo_depth.yaml

In   left/right images (mono8 / bgr8 / rgb8, raw or rectified). On the drone: /oak/*/image_mono ONLY; subscribing to
     image_raw (or */compressed) starves cuVSLAM (see onboard/config/stereo_depth.yaml) + their CameraInfo, same stamps (the OAK
     triggers both cameras together; sync_tol_s allows a small offset).
Out  depth_topic       sensor_msgs/Image 16UC1 mm (or 32FC1 m), stamped with the LEFT image's stamp
     depth_info_topic  sensor_msgs/CameraInfo: the rectified pinhole K at the output size, D = 0
     status_topic      std_msgs/String JSON once a second: rate, processing ms, latency, valid fraction
     /tf_static        raw left optical <- rectified frame, only when rectification rotates the left camera
The consumers (refnav, map_align, live_map) take the depth topic, its CameraInfo and TF base_link <- frame_id.

Work runs on its own thread at rate_hz, always on the NEWEST pair; older pairs are dropped, never queued,
so latency stays one frame. Calibration comes from CameraInfo; when it holds no stereo rectification the
right <- left extrinsic is read once from TF (both frames must be REP-103 optical frames).
"""
from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener

from onboard.depth.stereo_depth import CameraModel, DepthParams, StereoDepthPipeline, check_baseline, to_16uc1, to_gray


def _model(m: CameraInfo) -> CameraModel:
    return CameraModel.from_lists(m.width, m.height, list(m.k), list(m.d), list(m.r), list(m.p),
                                  m.distortion_model or "plumb_bob", m.header.frame_id)


def _ns(stamp) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def _quat_to_mat(x, y, z, w) -> np.ndarray:
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class StereoDepthNode(Node):
    def __init__(self) -> None:
        super().__init__("stereo_depth")
        p = lambda n, v: self.declare_parameter(n, v).value
        self.rate_hz = float(p("rate_hz", 5.0))
        self.scale = float(p("scale", 0.5))
        self.params = DepthParams(min_m=float(p("min_m", 0.35)), max_m=float(p("max_m", 6.0)))
        self.encoding = str(p("encoding", "16UC1")).upper()
        if self.encoding not in ("16UC1", "32FC1"):
            raise ValueError("encoding must be 16UC1 or 32FC1")
        self.sync_tol_ns = int(float(p("sync_tol_s", 0.003)) * 1e9)
        self.out_frame_param = str(p("out_frame", ""))
        # refuse to publish unless the derived baseline is about this (skybeta-09 OAK: 75.3 mm; T4 sim: 75.7 mm)
        self.expected_baseline_m = float(p("expected_baseline_m", 0.075))
        self.baseline_tol_m = float(p("baseline_tol_m", 0.01))
        self.refused: str | None = None
        self.pipe: StereoDepthPipeline | None = None
        self.out_frame = ""
        self.infos: dict[str, CameraInfo] = {}
        self.latest: dict[str, Image] = {}
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)
        self.pair: tuple[Image, Image] | None = None
        self.st = dict(pairs=0, processed=0, replaced=0, unpaired=0, errors=0)
        self.proc_ms: list[float] = []
        self.lat_ms: list[float] = []
        self.valid: list[float] = []
        self.last_pub_ns: list[int] = []
        self.last_error = ""

        sd = qos_profile_sensor_data
        self.create_subscription(Image, p("left_topic", "/oak/left/image_mono"), lambda m: self._img("l", m), sd)
        self.create_subscription(Image, p("right_topic", "/oak/right/image_mono"), lambda m: self._img("r", m), sd)
        self.create_subscription(CameraInfo, p("left_info_topic", "/oak/left/camera_info"), lambda m: self._info("l", m), sd)
        self.create_subscription(CameraInfo, p("right_info_topic", "/oak/right/camera_info"), lambda m: self._info("r", m), sd)
        self.pub = self.create_publisher(Image, p("depth_topic", "/camera/depth/image_raw"), 2)
        self.pub_info = self.create_publisher(CameraInfo, p("depth_info_topic", "/camera/depth/camera_info"), 2)
        self.pub_status = self.create_publisher(String, p("status_topic", "/stereo_depth/status"), 2)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_tf = StaticTransformBroadcaster(self)
        self.create_timer(1.0, self._status)
        self.running = True
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    # --- inputs ---------------------------------------------------------------------------------------------
    def _info(self, side: str, m: CameraInfo) -> None:
        self.infos[side] = m

    def _img(self, side: str, m: Image) -> None:
        other = "r" if side == "l" else "l"
        with self.lock:
            self.latest[side] = m
            o = self.latest.get(other)
            if o is None:
                return
            if abs(_ns(m.header.stamp) - _ns(o.header.stamp)) > self.sync_tol_ns:
                if _ns(m.header.stamp) > _ns(o.header.stamp):
                    self.st["unpaired"] += 1            # the other side's frame never came: wait for its next
                return
            pair = (m, o) if side == "l" else (o, m)
            del self.latest["l"], self.latest["r"]
            self.st["pairs"] += 1
            if self.pair is not None:
                self.st["replaced"] += 1                # the worker was busy: keep only the newest pair
            self.pair = pair
            self.wake.notify()

    # --- calibration ----------------------------------------------------------------------------------------
    def _setup(self) -> bool:
        if "l" not in self.infos or "r" not in self.infos:
            return False
        L, R = _model(self.infos["l"]), _model(self.infos["r"])
        R_rl = t_rl = None
        if R.P is None or abs(R.P[0, 3]) < 1e-9:
            try:
                tf = self.tf_buffer.lookup_transform(R.frame_id, L.frame_id, Time())     # X_right = R X_left + t
            except Exception as e:  # noqa: BLE001 (tf2 raises several types)
                self._error(f"no stereo rectification in CameraInfo and no TF {R.frame_id} <- {L.frame_id}: {type(e).__name__}")
                return False
            q, t = tf.transform.rotation, tf.transform.translation
            R_rl, t_rl = _quat_to_mat(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])
            rv = np.degrees(np.array([math.atan2(R_rl[2, 1], R_rl[2, 2]), -math.asin(max(-1.0, min(1.0, R_rl[2, 0]))),
                                      math.atan2(R_rl[1, 0], R_rl[0, 0])]))
            self.get_logger().info(f"extrinsic from TF {R.frame_id} <- {L.frame_id}: t = ({t.x:.4f}, {t.y:.4f}, {t.z:.4f}) m, "
                                   f"rpy = ({rv[0]:.3f}, {rv[1]:.3f}, {rv[2]:.3f}) deg")
        pipe = StereoDepthPipeline(L, R, self.scale, self.params, R_rl, t_rl)
        why = check_baseline(pipe.rect.baseline_m, self.expected_baseline_m, self.baseline_tol_m)
        if why:
            self.refused = why
            self.get_logger().error(why)
            return False
        self.pipe = pipe
        r = self.pipe.rect
        self.out_frame = L.frame_id if r.identity else (self.out_frame_param or f"{L.frame_id}_rect")
        if not r.identity:
            ts = TransformStamped()
            ts.header.stamp = self.get_clock().now().to_msg()
            ts.header.frame_id, ts.child_frame_id = L.frame_id, self.out_frame
            x, y, z, w = r.raw_from_rect_quat_xyzw()
            ts.transform.rotation.x, ts.transform.rotation.y, ts.transform.rotation.z, ts.transform.rotation.w = x, y, z, w
            self.static_tf.sendTransform(ts)
        self.get_logger().info(
            f"stereo depth: calibration from {r.source}, f {r.f:.1f} px, baseline {r.baseline_m:.4f} m, "
            f"{r.size[0]}x{r.size[1]} at {self.rate_hz:g} Hz, {self.pipe.matcher.num_disp} disparities "
            f"({self.params.min_m:g}-{self.params.max_m:g} m), frame {self.out_frame}")
        return True

    def _error(self, msg: str) -> None:
        self.st["errors"] += 1
        if msg != self.last_error:
            self.get_logger().warn(msg)
            self.last_error = msg

    # --- work -----------------------------------------------------------------------------------------------
    def _loop(self) -> None:
        period = 1.0 / self.rate_hz if self.rate_hz > 0 else 0.0
        next_t = 0.0
        while self.running and rclpy.ok():
            with self.lock:
                while self.pair is None and self.running:
                    self.wake.wait(0.2)
                if not self.running:
                    return
                if time.monotonic() < next_t:
                    pair = None
                else:
                    pair, self.pair = self.pair, None
            if pair is None:
                time.sleep(max(0.0, next_t - time.monotonic()))
                continue
            next_t = time.monotonic() + period
            try:
                self._process(*pair)
            except Exception as e:  # noqa: BLE001 (keep the node alive; report it)
                self._error(f"stereo depth failed: {type(e).__name__}: {e}")

    def _process(self, left: Image, right: Image) -> None:
        if self.refused:
            return                                     # wrong calibration: never publish (logged once, in status)
        if self.pipe is None and not self._setup():
            return
        t0 = time.perf_counter()
        gl = to_gray(left.encoding, left.height, left.width, left.step, bytes(left.data))
        gr = to_gray(right.encoding, right.height, right.width, right.step, bytes(right.data))
        z = self.pipe(gl, gr)
        w, h = self.pipe.size
        out = Image()
        out.header.stamp, out.header.frame_id = left.header.stamp, self.out_frame
        out.height, out.width, out.is_bigendian = h, w, 0
        if self.encoding == "16UC1":
            a = to_16uc1(z); out.encoding, out.step = "16UC1", w * 2
        else:
            a = np.nan_to_num(z, nan=0.0).astype(np.float32); out.encoding, out.step = "32FC1", w * 4
        out.data = a.tobytes()
        info = CameraInfo()
        info.header = out.header
        info.height, info.width, info.distortion_model = h, w, "plumb_bob"
        K = self.pipe.K
        info.k = [float(v) for v in K.ravel()]
        info.d = [0.0] * 5
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [float(K[0, 0]), 0.0, float(K[0, 2]), 0.0, 0.0, float(K[1, 1]), float(K[1, 2]), 0.0, 0.0, 0.0, 1.0, 0.0]
        self.pub_info.publish(info)
        self.pub.publish(out)
        now = self.get_clock().now().nanoseconds
        with self.lock:
            self.st["processed"] += 1
            self.proc_ms.append((time.perf_counter() - t0) * 1e3)
            self.lat_ms.append((now - _ns(left.header.stamp)) * 1e-6)
            band = z[int(0.1 * h):int(0.9 * h), int(0.15 * w):int(0.95 * w)]
            self.valid.append(float(np.isfinite(band).mean()))
            self.last_pub_ns.append(time.monotonic_ns())

    def _status(self) -> None:
        with self.lock:
            pm, lm, vf, pubs = self.proc_ms, self.lat_ms, self.valid, self.last_pub_ns
            self.proc_ms, self.lat_ms, self.valid, self.last_pub_ns = [], [], [], []
            st = dict(self.st)
        rate = (len(pubs) - 1) / ((pubs[-1] - pubs[0]) * 1e-9) if len(pubs) > 1 else float(len(pubs))
        pct = lambda a, q: round(float(np.percentile(a, q)), 1) if a else None
        s = dict(ok=self.pipe is not None and bool(pubs), rate_hz=round(rate, 2), proc_ms_median=pct(pm, 50),
                 proc_ms_p95=pct(pm, 95), latency_ms_median=pct(lm, 50), latency_ms_p95=pct(lm, 95),
                 valid_frac=round(float(np.mean(vf)), 3) if vf else None, frame=self.out_frame, **st,
                 last_error=self.last_error or None, refused=self.refused,
                 baseline_m=round(self.pipe.rect.baseline_m, 5) if self.pipe else None)
        self.pub_status.publish(String(data=json.dumps(s)))

    def destroy_node(self) -> None:
        self.running = False
        with self.lock:
            self.wake.notify_all()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StereoDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
