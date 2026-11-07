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
from .anchor import Anchor
from .land_freeze import LandFreeze
from .validation import CaptureGate,sample_problem
from .vio_health import VioHealth


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
        self.max_var=float(declare(self,'max_position_var_m2',.05))
        self.anchor: Anchor = Anchor()
        self.last_out: tuple | None = None    # (pos, q_xyzw) last published, for re-anchoring
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
        # Land freeze (land_freeze.py, hinson-claude's 7b0344a, landed 2/2 on the T4): once PX4 is armed in
        # AUTO_LAND/DESCEND with a fresh downward range <= land_freeze_m, publish the last PUBLISHED sample at 15 Hz
        # until disarm, so PX4 never chases cuVSLAM near the pad. Replaces darion's touchdown_hold, whose reference
        # pose was taken only while range >= 0.8 m: at the 0.70-0.77 m cruise it was 6 s old and 1.47 m off on
        # T4 darion-wholestack run 2, and PX4 reset onto it. One guard added here: a sample older than
        # freeze_max_age_s is never latched (so a freeze can't hold a pose from before a vision outage).
        from px4_msgs.msg import DistanceSensor, VehicleStatus
        self._DistanceSensor,self._VehicleStatus=DistanceSensor,VehicleStatus
        self.freeze=LandFreeze(float(declare(self,'land_freeze_m',.25)))
        self.freeze_max_age_s=float(declare(self,'freeze_max_age_s',.3))
        self.nav: int | None = None
        self.armed: bool = False
        self.range: tuple[int, float | None] | None = None   # (node clock us, valid downward range or None)
        self.sample: dict | None = None                        # odometry_to_px4 kwargs of the last published sample
        self.sample_us: int | None = None                      # node clock when it was published
        self.create_subscription(DistanceSensor,str(declare(self,'range_topic','/fmu/out/distance_sensor')),self._on_range,PX4_QOS)
        self.create_subscription(VehicleStatus,str(declare(self,'status_topic','/fmu/out/vehicle_status_v1')),self._on_status,PX4_QOS)
        self.create_timer(1/15,self._publish_frozen)
        self.reset_counter: int = initial_reset_counter()
        self.last_pos: tuple[float, float, float] | None = None
        self.frame_id: str | None = None
        self.last_now: int | None = None      # node clock, us
        self.last_good: int | None = None     # node clock of the last published sample, us
        self.ok: bool = False
        self.reason: str = 'no samples'
        self.counts: dict[str, int] = {'input':0,'published':0,'invalid':0,'withheld':0,'unowned':0,'clock_resets':0,'frozen':0}
        self.create_timer(.2,self._publish_health)
        self.create_timer(5.,self._log)
        self.get_logger().info(f'{self.in_topic} -> {self.out_topic}; frame={self.world}; heading reference={reference or "unreferenced (FRD)"}')

    def _on_range(self, m) -> None:
        if m.orientation!=self._DistanceSensor.ROTATION_DOWNWARD_FACING:return
        d=float(m.current_distance)
        self.range=(now_us(self),d if m.signal_quality!=0 and m.min_distance<=d<=m.max_distance else None)
        self._update_freeze()

    def _on_status(self, m) -> None:
        self.armed=m.arming_state==self._VehicleStatus.ARMING_STATE_ARMED;self.nav=int(m.nav_state)
        self._update_freeze()

    def _update_freeze(self) -> None:
        was=self.freeze.frozen is not None
        now=now_us(self)
        age=None if self.range is None else (now-self.range[0])*1e-6
        fresh=self.sample is not None and self.sample_us is not None and 0<=(now-self.sample_us)*1e-6<=self.freeze_max_age_s
        frozen=self.freeze.update(self.nav,self.armed,None if self.range is None else self.range[1],age,
                                  self.sample if fresh else None) is not None
        if frozen!=was:
            self.get_logger().info(f'vision pose {"frozen: PX4 landing at the floor" if frozen else "released"} (nav {self.nav}, armed {self.armed}, range {self.range})')

    def _publish_frozen(self) -> None:
        frozen=self.freeze.frozen
        if frozen is None or (self.guard is not None and not self.guard.may_publish()):return
        now=now_us(self);self._observe_clock(now)
        self.pub.publish(fill_vehicle_odometry(VehicleOdometry(),odometry_to_px4(
            stamp_sample_us=now,now_us=now,reset_counter=self.reset_counter,**frozen)))
        self.last_good=now;self.ok=True;self.reason='frozen: PX4 landing at the floor';self.counts['frozen']+=1

    def _health(self) -> VioHealth:
        return VioHealth(max_speed_mps=self.max_speed,jump_m=self.jump_m,quiet_s=self.quiet_s,max_var_m2=self.max_var)

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
        if self.freeze.frozen is not None:return              # landing near the floor: the camera is ignored
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
        problem=sample_problem(self.required,pos,quat,vel)
        if problem:
            self.ok=False;self.reason=problem;self.counts['invalid']+=1;return
        var=max(pose_cov[0],pose_cov[7],pose_cov[14]) if len(pose_cov)>=15 else None
        sane,reason=self.health.update(capture*1e-6,pos,var)
        if self.guard is not None and not self.guard.may_publish():
            self.ok=False;self.reason='proving sole vision ownership';self.counts['unowned']+=1;return
        self.ok,self.reason=sane,reason
        if not sane:self.counts['withheld']+=1;return
        if self.health.rebased and self.last_out is not None:
            # a persistent jump / re-init: keep the published track continuous instead of passing the jump
            self.anchor.rebase(self.last_out[0],self.last_out[1],pos,quat)
            self.counts['reanchored']=self.counts.get('reanchored',0)+1
            self.get_logger().warning(f'VIO re-anchored ({reason}); offset yaw {math.degrees(self.anchor.dyaw):.1f} deg, '
                                      f't=({self.anchor.tx:.2f},{self.anchor.ty:.2f},{self.anchor.tz:.2f})')
        pos,quat=self.anchor.apply(pos,quat)
        self.last_out=(pos,quat)
        changed=(self.frame_id is not None and self.frame_id!=msg.header.frame_id)
        jumped=self.last_pos is not None and math.dist(self.last_pos,pos)>self.jump_m
        if changed or jumped:self.reset_counter=(self.reset_counter+1)&255
        self.frame_id=msg.header.frame_id;self.last_pos=pos
        norm=math.sqrt(sum(x*x for x in quat));quat=tuple(x/norm for x in quat) if norm>0 else quat
        kw=dict(position=pos,q_wxyz=quat_xyzw_to_wxyz(*quat),world=self.world,heading_enu_rad=self.heading,
            velocity=vel,velocity_in_body=self.velocity_in_body,angular_velocity=ang,
            pose_cov=pose_cov,twist_cov=twist_cov,quality=self.quality,**self.floors)
        value=odometry_to_px4(stamp_sample_us=capture,now_us=now,reset_counter=self.reset_counter,**kw)
        self.pub.publish(fill_vehicle_odometry(VehicleOdometry(),value))
        self.sample=dict(kw,velocity=(0.,0.,0.),angular_velocity=(0.,0.,0.));self.sample_us=now   # what a freeze would hold
        if var is not None and var>self.max_var:self.counts['high_var_passed']=self.counts.get('high_var_passed',0)+1
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
