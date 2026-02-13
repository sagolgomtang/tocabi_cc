#!/usr/bin/env python3

import os
import rospy
import csv

from sensor_msgs.msg import JointState
from tocabi_msgs.msg import positionCommand


JOINT_NAMES = [
    "L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint",
    "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
    "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint",
    "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint",
    "Waist1_Joint", "Waist2_Joint", "Upperbody_Joint",
    "L_Shoulder1_Joint", "L_Shoulder2_Joint", "L_Shoulder3_Joint", "L_Armlink_Joint",
    "L_Elbow_Joint", "L_Forearm_Joint", "L_Wrist1_Joint", "L_Wrist2_Joint",
    "Neck_Joint", "Head_Joint",
    "R_Shoulder1_Joint", "R_Shoulder2_Joint", "R_Shoulder3_Joint", "R_Armlink_Joint",
    "R_Elbow_Joint", "R_Forearm_Joint", "R_Wrist1_Joint", "R_Wrist2_Joint",
]


class ChirpCollector:
    def __init__(self):
        self.duration = float(rospy.get_param("~duration", 20.0))
        self.output_dir = rospy.get_param(
            "~output_dir",
            "/home/user/other_ws/src/pace-sim2real/data/tocabi_mujoco",
        )
        self.output_name = rospy.get_param("~output_name", "chirp_data")
        self.output_format = rospy.get_param("~output_format", "txt").lower()
        self.started = False
        self.start_time = None
        self.times = []
        self.dof_pos = []
        self.des_dof_pos = []
        self.latest_q = None
        self.latest_stamp = None

        self.name_to_idx = {name: i for i, name in enumerate(JOINT_NAMES)}
        self.joint_state_sub = rospy.Subscriber("/tocabi/jointstates", JointState, self.on_joint_state)
        self.pos_cmd_sub = rospy.Subscriber("/tocabi/positioncommand", positionCommand, self.on_position_command)
        rospy.loginfo("[CHIRP] Waiting for positioncommand to start recording.")

    def on_joint_state(self, msg: JointState):
        q = [0.0] * len(JOINT_NAMES)
        for name, pos in zip(msg.name, msg.position):
            idx = self.name_to_idx.get(name)
            if idx is not None:
                q[idx] = pos
        self.latest_q = q
        self.latest_stamp = msg.header.stamp if msg.header.stamp is not None else rospy.Time.now()

    def on_position_command(self, msg: positionCommand):
        if self.latest_q is None:
            return
        now = rospy.Time.now().to_sec()
        if not self.started:
            self.started = True
            self.start_time = now
            rospy.loginfo("[CHIRP] Recording started.")
        t = now - self.start_time
        if t > self.duration:
            self.save_and_exit()
            return
        self.times.append(t)
        self.dof_pos.append(self.latest_q)
        self.des_dof_pos.append(list(msg.position))

    def save_and_exit(self):
        if not self.times:
            rospy.logwarn("[CHIRP] No samples recorded. Nothing to save.")
            rospy.signal_shutdown("no data")
            return
        os.makedirs(self.output_dir, exist_ok=True)
        if self.output_format == "txt":
            time_path = os.path.join(self.output_dir, f"{self.output_name}_time.txt")
            dof_pos_path = os.path.join(self.output_dir, f"{self.output_name}_dof_pos.txt")
            des_dof_pos_path = os.path.join(self.output_dir, f"{self.output_name}_des_dof_pos.txt")
            with open(time_path, "w", newline="") as f_time:
                writer = csv.writer(f_time, delimiter=" ")
                for t in self.times:
                    writer.writerow([f"{t:.6f}"])
            with open(dof_pos_path, "w", newline="") as f_pos:
                writer = csv.writer(f_pos, delimiter=" ")
                for row in self.dof_pos:
                    writer.writerow([f"{v:.6f}" for v in row])
            with open(des_dof_pos_path, "w", newline="") as f_des:
                writer = csv.writer(f_des, delimiter=" ")
                for row in self.des_dof_pos:
                    writer.writerow([f"{v:.6f}" for v in row])
            rospy.loginfo(
                "[CHIRP] Saved %d samples to %s (time/dof_pos/des_dof_pos).",
                len(self.times),
                self.output_dir,
            )
        else:
            raise RuntimeError(f"Unsupported output_format: {self.output_format}")
        rospy.signal_shutdown("done")


def main():
    rospy.init_node("tocabi_chirp_collector")
    ChirpCollector()
    rospy.spin()


if __name__ == "__main__":
    main()
