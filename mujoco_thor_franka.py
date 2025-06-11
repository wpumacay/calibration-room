from typing import Dict, Any, Mapping, Set
import numpy as np
from scipy.spatial.transform import Rotation as R

import mujoco
from mujoco import MjModel, MjData

from mujoco_thor_abstract import Agent


class NamespaceDictWrapper:
    def __init__(self, namespace: str, d: Dict[str, Any]):
        assert isinstance(d, Mapping)
        self.namespace = namespace
        self.d = d

    def __getitem__(self, item: str):
        return self.d[self.namespace + item]

    def __setitem__(self, key: str, value: Any):
        self.d[self.namespace + key] = value

    def __contains__(self, item: str):
        return self.namespace + item in self.d

    def __len__(self):
        return len(self.d)

    def keys(self):
        return [k[len(self.namespace) :] for k in self.d]


class FrankaFR3Agent(Agent):
    INIT_JOINT_POS = np.array([0, -0.7853, 0, -2.35619, 0, 1.57079, 0.7853])

    GRIPPER_LEFT_NAME = "fingertip_pad_collision_left"  # the geom used to check success
    GRIPPER_RIGHT_NAME = "fingertip_pad_collision_right"

    @staticmethod
    def init_joint_pos():
        return FrankaFR3Agent.INIT_JOINT_POS.copy()

    @staticmethod
    def init_gripper_pos():
        return 0.08

    def __init__(self, model: MjModel, namespace: str = "robot_0/"):
        super().__init__(model)
        self.namespace = namespace
        self._grasp_site_id = model.site(f"{self.namespace}grasp_site").id
        self._base_id = model.body(f"{self.namespace}base_link").id
        self._root_id = model.body_rootid[self._base_id].item()

        self._set_joint_names(model)
        self._set_arm_joint_limits(model)
        self._set_actuator_to_joints(model)

        self._get_gripper_geom_names()
        self.reset()

    @classmethod
    def from_model(cls, model: MjModel, namespace: str = "robot_0/"):
        return cls(model, namespace=namespace)

    def __call__(self, model: MjModel, data: MjData) -> "FrankaFR3Agent":
        assert self.model is model, "[FrankaFR3Agent.__call__] Model mismatch"
        self.update(data)
        return self

    def reset(self):
        self._joint_pos = None
        self._joint_vel = None
        self._gripper_pos = None
        self._gripper_vel = None
        self._position = None
        self._quaternion = None
        self._base_pose = None
        self._ee_pose = None

    def _body_pose(self, body_id: int, data: MjData):
        pose = np.eye(4)
        pose[:3, :3] = data.xmat[body_id].reshape(3, 3)
        pose[:3, 3] = data.xpos[body_id]
        return pose

    def _site_pose(self, site_id: int, data: MjData):
        pose = np.eye(4)
        pose[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
        pose[:3, 3] = data.site_xpos[site_id]
        return pose

    @property
    def root_id(self) -> int:
        return self._root_id

    @property
    def finger_body_ids(self):
        geom_left_id = self.model.geom(self.left_gripper_geom_name).id
        geom_right_id = self.model.geom(self.right_gripper_geom_name).id
        body_left_id = self.model.geom(self.left_gripper_geom_name).bodyid.item()
        body_right_id = self.model.geom(self.right_gripper_geom_name).bodyid.item()
        finger_tip_id = [body_left_id, body_right_id]
        return finger_tip_id

    @property
    def name(self):
        return "franka"

    @property
    def joint_pos(self):
        return self._joint_pos

    @property
    def arm_joint_pos(self):
        return self._joint_pos

    @property
    def all_joint_pos(self):
        return self._joint_pos

    @property
    def gripper_pos(self):
        return self._gripper_pos

    @property
    def joint_vel(self):
        return self._joint_vel

    @property
    def gripper_vel(self):
        return self._gripper_vel

    @property
    def position(self):
        return self._position

    @property
    def quaternion(self):
        return self._quaternion

    @property
    def base_pose(self):
        return self._base_pose

    @property
    def grasp_center_from_base(self):
        xyz = self._grasp_center_position
        rpy = R.from_quat(self._grasp_center_quaternion, scalar_first=True).as_euler(
            "xyz", degrees=False
        )

        def normalize_angle(angle):
            return np.mod(angle + np.pi, 2 * np.pi) - np.pi

        rpy = np.array([normalize_angle(angle) for angle in rpy])

        return np.concatenate([xyz, rpy])

    @property
    def ee_pose_from_base(self):
        trf = np.eye(4)
        trf[:3, :3] = R.from_quat(self._grasp_center_quaternion, scalar_first=True).as_matrix()
        trf[:3, 3] = self._grasp_center_position
        return trf

    @property
    def joint_limits(self):
        return self._joint_limits.T

    # ---- Set Functions ----
    def _set_joint_names(self, model: MjModel):
        self.joint_names = [
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
        ]
        self.gripper_actuator_name = "panda_hand"

    def _set_arm_joint_limits(self, model: MjModel):
        self._joint_limits = np.array(
            [
                [-2.7437, 2.7437],  # joint1
                [-1.7837, 1.7837],  # joint2
                [-2.9007, 2.9007],  # joint3
                [-3.0421, -0.1518],  # joint4
                [-2.8065, 2.8065],  # joint5
                [0.5445, 4.5169],  # joint6
                [-3.0159, 3.0159],  # joint7
            ]
        )

    def _set_actuator_to_joints(self, model: MjModel):
        _actuator_to_joints = {}
        start_idx = len(self.namespace)
        actuators_trn_ids = model.actuator_trnid
        actuators_trn_types = model.actuator_trntype
        # debug:
        # print("actuators_trn_ids", actuators_trn_ids)
        # print("actuators_trn_types", actuators_trn_types)
        for i, act_tr in enumerate(actuators_trn_types):
            actuator_name = model.actuator(i).name[start_idx:]
            _actuator_to_joints[actuator_name] = []
            trn_id = actuators_trn_ids[i][0]  # (id, -1)
            if act_tr == 0:  # joint
                trn_name = model.jnt(trn_id).name[start_idx:]
                _actuator_to_joints[actuator_name].append(trn_name)
            if act_tr == 3:  # tendon
                trn_name = model.tendon(trn_id).name[start_idx:]
                _actuator_to_joints[actuator_name].append(trn_name)
            if act_tr == 4:  # site
                trn_name = model.site(trn_id).name[start_idx:]
                _actuator_to_joints[actuator_name].append(trn_name)
        self._actuator_to_joints = NamespaceDictWrapper(self.namespace, _actuator_to_joints)

    def _set_joint_pos(self, data: MjData, joint_pos: np.ndarray):
        data.qpos[self._get_joint_qposadr()] = joint_pos

    def _set_joint_ctrl(self, data: MjData, ctrl: np.ndarray):
        data.ctrl[self._get_joint_actuator_ids()] = ctrl

    def set_joint_qpos(self, data: MjData, qpos: np.ndarray):
        self._set_joint_pos(data, qpos)
        self._set_joint_ctrl(data, qpos)

    def set_gripper_pos(self, data: MjData, pos: float):
        data.qpos[self.finger_qpos_adrs] = pos / len(self.finger_qpos_adrs)

    def set_gripper_ctrl(self, data: MjData, ctrl: float):
        data.ctrl[self._get_gripper_actuator_ids()] = ctrl

    # ---- Update Functions ----
    def _read_from_sensor(self, sensor_name: str, data: MjData):
        s_adr = self.model.sensor(self.namespace + sensor_name).adr.item()
        s_dim = self.model.sensor(self.namespace + sensor_name).dim.item()
        return data.sensordata[s_adr : s_adr + s_dim].copy()

    def update(self, data: MjData):
        self._position = self._read_from_sensor("base_position", data)
        self._quaternion = self._read_from_sensor("base_quaternion", data)
        self._grasp_center_position = self._read_from_sensor("grasp_center_pos_from_base", data)
        self._grasp_center_quaternion = self._read_from_sensor("grasp_center_quat_from_base", data)
        self.update_joints_pos(data)
        self.update_actuator_ctrl_inputs(data)
        self.update_base_pose()

    def update_base_pose(self):
        #  update_base_transform(self, data: MjData):
        r = R.from_quat(self.quaternion, scalar_first=True).as_matrix()
        t = self.position
        transform = np.eye(4)
        transform[0:3, 0:3] = r
        transform[0:3, 3] = t
        self._base_pose = transform

    def update_joints_pos(self, data: MjData):
        _joints = np.zeros(len(self.joint_names))
        for i, joint_name in enumerate(self.joint_names):
            _joints[i] = self._read_from_sensor(joint_name, data).item()
        self._joint_pos = _joints

    def update_actuator_ctrl_inputs(self, data):
        n = self.model.nu
        ctrl_inputs = np.zeros(n)
        for i in range(n):
            if self.model.actuator(i).name.startswith(self.namespace):
                ctrl_inputs[i] = data.actuator(self.model.actuator(i).name).ctrl[0]
        self._actuator_ctrl_inputs = ctrl_inputs

    # ---- get functions ----
    def _get_joint_qposadr(self):
        """Get joint ids for the given group name."""
        return [
            self.model.jnt_qposadr[self.model.joint(self.namespace + name).id]
            for name in self.joint_names
        ]

    def _get_joint_dofadr(self):
        """Get joint ids for the given group name."""
        return [self.model.joint(self.namespace + name).dofadr[0] for name in self.joint_names]

    def _get_joint_actuator_ids(self):
        return [self.model.actuator(self.namespace + name).id for name in self.joint_names]

    def _get_gripper_actuator_ids(self):
        return self.model.actuator(self.namespace + self.gripper_actuator_name).id

    def _get_gripper_geom_names(self):
        self.left_gripper_geom_name = self.namespace + self.GRIPPER_LEFT_NAME
        self.right_gripper_geom_name = self.namespace + self.GRIPPER_RIGHT_NAME

    def get_grasped_objs(self, model: MjModel, data: MjData) -> Set[int]:
        """Return a set of grasped object body IDs."""
        root_id = model.body_rootid[next(iter(self.finger_body_ids))]
        grasped_objs = set()
        for c in data.contact:
            if c.exclude != 0:
                continue
            b1 = model.geom_bodyid[c.geom1]
            b2 = model.geom_bodyid[c.geom2]

            # ignore all collisions not involving only one finger
            if not ((b1 in self.finger_body_ids) ^ (b2 in self.finger_body_ids)):
                continue
            # ignore self-collisions
            if model.body_rootid[b1] == root_id and model.body_rootid[b2] == root_id:
                continue

            if b1 in self.finger_body_ids:
                grasped_objs.add(b2)
            else:
                grasped_objs.add(b1)
        grasped_objs = {model.body_rootid[obj_id] for obj_id in grasped_objs}

        return grasped_objs

    def check_collision(self, model: MjModel, data: MjData, grasped_objs: Set[int]):
        assert model is self.model
        for c in data.contact:
            if c.exclude != 0:
                continue
            b1 = model.geom_bodyid[c.geom1]
            b2 = model.geom_bodyid[c.geom2]

            if {b1, b2} <= self.finger_body_ids:
                continue
            if len({b1, b2} & self.finger_body_ids) == 1 and len({b1, b2} & grasped_objs) == 1:
                continue

            rb1 = model.body_rootid[b1]
            rb2 = model.body_rootid[b2]
            if rb1 == self.root_id or rb2 == self.root_id:
                return True
            if len(grasped_objs & {b1, b2}) == 1:
                return True
        return False

    def ee_jacobian(self, model: MjModel, data: MjData):
        assert self.model is model
        J = np.zeros((6, self.model.nv))
        mujoco.mj_jacSite(model, data, J[:3], J[-3:], self._grasp_site_id)
        return J[:, self._get_joint_dofadr()]
