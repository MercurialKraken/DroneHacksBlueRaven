#!/usr/bin/env python3
"""ROS vision odometry -> native PX4 external vision, on hardware or ROS /clock.

The capture timestamp stays the measurement timestamp. Middleware owns clock
transport; this node owns frame conversion, validity, source resets and health.
"""
from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry
from std_msgs.msg import Bool

from .frames import WORLDS,initial_reset_counter,odometry_to_px4,quat_xyzw_to_wxyz
from .ownership import PublisherGuard
from .ros_common import PX4_QOS,SENSOR_QOS,declare,fill_vehicle_odometry,now_us,stamp_of
from .touchdown_hold import TouchdownHold
from .validation import CaptureGate,sample_problem
from .vio_health import VioHealth

NAV_AUTO_LAND = 18


class EvBridge(Node):
    def __init__(self) -> None:
        super().__init__('px4_vision_bridge')
        self.world: str = str(declare(self,'world','flu'))
        if self.world not in WORLDS:raise ValueError('world must be enu or flu')
        heading=float(declare(self,'heading_enu_rad',math.nan))
        reference=str(declare(self,'heading_reference',''))
        self.heading: float | None = None if math.isnan(heading) else heading
        if self.heading is not None and (not math.isfinite(heading) or not reference.strip() or self.world!='flu'):
            raise ValueError('heading alignment needs a finite surveyed FLU-to-ENU reference and its description')
        self.required: set[str] = set(declare(self,'required_fields',['position','orientation']))
        if not self.required or not self.required<={'position','orientation','velocity'}:
            raise ValueError('required_fields must name the position/orientation/velocity fields consumed by PX4')
        self.in_topic=str(declare(self,'odom_topic','/visual_slam/tracking/odometry'))
        self.out_topic=str(declare(self,'out_topic','/fmu/in/vehicle_visual_odometry'))
        self.velocity_in_body=bool(declare(self,'velocity_in_body',True))
        self.quality=int(declare(self,'quality',0))
        self.jump_m=float(declare(self,'jump_reset_m',1.))
        self.health_stale_s=float(declare(self,'health_stale_s',0.5))
        self.max_speed=float(declare(self,'max_speed_mps',5.))
        self.quiet_s=float(declare(self,'health_quiet_s',5.))
        if not all(math.isfinite(v) and v>0 for v in (self.jump_m,self.health_stale_s,self.max_speed,self.quiet_s)):
            raise ValueError('health limits must be positive finite numbers')
        self.gate: CaptureGate = CaptureGate(float(declare(self,'max_sample_age_s',.3)),float(declare(self,'future_tol_s',.005)))
        self.health: VioHealth = self._health()
        self.floors: dict[str, list[float]] = {name:[float(v) for v in declare(self,name,[0.,0.,0.])]
                     for name in ('position_floor','orientation_floor','velocity_floor')}
        if any(len(v)!=3 or not all(math.isfinite(x) and x>=0 for x in v) for v in self.floors.values()):
            raise ValueError('variance floors need three finite nonnegative values')
        self.pub=self.create_publisher(VehicleOdometry,self.out_topic,PX4_QOS)
        self.ok_pub=self.create_publisher(Bool,str(declare(self,'health_topic','/estimator/healthy')),10)
        self.guard: PublisherGuard | None = PublisherGuard(self,self.out_topic) if self.out_topic.startswith('/fmu/in/') else None
        self.sub=self.create_subscription(Odometry,self.in_topic,self._on_odom,SENSOR_QOS)
        # Touchdown hold (touchdown_hold.py): during PX4 AUTO_LAND, once the downward range is below engage_m,
        # publish the last pose seen above ref_m instead of the camera's (it slides near the floor).
        self.hold: TouchdownHold | None = None
        if bool(declare(self,'touchdown_hold',True)):
            from px4_msgs.msg import DistanceSensor, VehicleStatus
            self.hold=TouchdownHold(engage_m=float(declare(self,'hold_engage_m',.7)),ref_m=float(declare(self,'hold_ref_m',.8)),
                                    release_m=float(declare(self,'hold_release_m',.9)),hold_var=float(declare(self,'hold_var_m2',.01)))
            self.create_subscription(DistanceSensor,str(declare(self,'range_topic','/fmu/out/distance_sensor')),self._on_range,PX4_QOS)
            self.create_subscription(VehicleStatus,str(declare(self,'status_topic','/fmu/out/vehicle_status_v1')),self._on_status,PX4_QOS)
        self.reset_counter: int = initial_reset_counter()
        self.last_pos: tuple[float, float, float] | None = None
        self.frame_id: str | None = None
        self.last_now: int | None = None      # node clock, us
        self.last_good: int | None = None     # node clock of the last published sample, us
        self.ok: bool = False
        self.reason: str = 'no samples'
        self.counts: dict[str, int] = {'input':0,'published':0,'invalid':0,'withheld':0,'unowned':0,'clock_resets':0}
        self.create_timer(.2,self._publish_health)
        self.create_timer(5.,self._log)
        self.get_logger().info(f'{self.in_topic} -> {self.out_topic}; frame={self.world}; heading reference={reference or "unreferenced (FRD)"}')

    def _on_range(self, msg) -> None:
        if self.hold is not None:
            was=self.hold.active
            self.hold.on_range(now_us(self)*1e-6,float(msg.current_distance))
            if self.hold.active!=was:
                self.get_logger().info(f"touchdown hold {'ENGAGED' if self.hold.active else 'released'} at range {msg.current_distance:.2f} m")

    def _on_status(self, msg) -> None:
        if self.hold is not None:
            self.hold.landing=int(msg.nav_state)==NAV_AUTO_LAND

    def _health(self) -> VioHealth:
        return VioHealth(max_speed_mps=self.max_speed,jump_m=self.jump_m,quiet_s=self.quiet_s)

    def _observe_clock(self, now: int) -> None:
        if self.last_now is not None and now<self.last_now:
            self.gate.reset();self.health=self._health();self.last_pos=None;self.frame_id=None
            self.reset_counter=(self.reset_counter+1)&255;self.last_good=None;self.ok=False
            self.reason='ROS clock restarted';self.counts['clock_resets']+=1
        self.last_now=now

    def _healthy_now(self) -> bool:
        now=now_us(self);self._observe_clock(now)
        return bool(self.ok and self.last_good is not None and 0<=now-self.last_good<self.health_stale_s*1e6)

    def _publish_health(self) -> None:
        self.ok_pub.publish(Bool(data=self._healthy_now()))

    def _on_odom(self, msg: Odometry) -> None:
        self.counts['input']+=1
        now=now_us(self);self._observe_clock(now);capture=stamp_of(msg.header)
        accepted,reason,_=self.gate.check(capture,now)
        if not accepted:
            if reason not in ('duplicate','reordered'):self.ok=False
            self.reason=reason;return
        p,q=msg.pose.pose.position,msg.pose.pose.orientation
        v,w=msg.twist.twist.linear,msg.twist.twist.angular
        pos=(p.x,p.y,p.z);quat=(q.x,q.y,q.z,q.w);vel=(v.x,v.y,v.z)
        pose_cov=list(msg.pose.covariance);twist_cov=list(msg.twist.covariance);ang=(w.x,w.y,w.z)
        held=self.hold is not None and self.hold.active
        if held:
            # touchdown hold: the camera's pose is not used; re-publish the reference pose, stamped now, with
            # inflated position variance and unknown velocity
            if self.guard is not None and not self.guard.may_publish():
                self.ok=False;self.reason='proving sole vision ownership';self.counts['unowned']+=1;return
            (pos,quat,pose_cov),_=self.hold.filter(capture*1e-6,None)
            pose_cov=list(pose_cov);vel=None;ang=None;twist_cov=None
            for i in (0,7,14):pose_cov[i]=max(pose_cov[i],0.)+self.hold.hold_var
            self.ok,self.reason=True,'touchdown hold'
            self.counts['held']=self.counts.get('held',0)+1
        else:
            problem=sample_problem(self.required,pos,quat,vel)
            if problem:
                self.ok=False;self.reason=problem;self.counts['invalid']+=1;return
            sane,reason=self.health.update(capture*1e-6,pos)
            if self.guard is not None and not self.guard.may_publish():
                self.ok=False;self.reason='proving sole vision ownership';self.counts['unowned']+=1;return
            self.ok,self.reason=sane,reason
            if not sane:self.counts['withheld']+=1;return
            if self.hold is not None:
                self.hold.filter(capture*1e-6,(pos,quat,pose_cov))      # remembers the reference pose
        changed=(self.frame_id is not None and self.frame_id!=msg.header.frame_id)
        jumped=self.last_pos is not None and math.dist(self.last_pos,pos)>self.jump_m
        if changed or jumped:self.reset_counter=(self.reset_counter+1)&255
        self.frame_id=msg.header.frame_id;self.last_pos=pos
        norm=math.sqrt(sum(x*x for x in quat));quat=tuple(x/norm for x in quat) if norm>0 else quat
        value=odometry_to_px4(stamp_sample_us=capture,now_us=now,position=pos,
            q_wxyz=quat_xyzw_to_wxyz(*quat),world=self.world,heading_enu_rad=self.heading,
            velocity=vel,velocity_in_body=self.velocity_in_body,angular_velocity=ang,
            pose_cov=pose_cov,twist_cov=twist_cov,
            reset_counter=self.reset_counter,quality=self.quality,**self.floors)
        self.pub.publish(fill_vehicle_odometry(VehicleOdometry(),value))
        self.last_good=now;self.counts['published']+=1

    def _log(self) -> None:
        self.get_logger().info(f'counts={self.counts} captures={self.gate.counts} healthy={self._healthy_now()} reason={self.reason}')

    def destroy_node(self) -> None:
        guard=getattr(self,'guard',None)
        if guard is not None:guard.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args);node=None
    try:
        node=EvBridge();rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        if node is not None:node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':main()
