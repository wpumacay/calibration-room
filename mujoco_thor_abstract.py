import abc
from typing import NamedTuple, Dict, Optional, Set
import numpy as np

from mujoco import MjData, MjModel
from mujoco.viewer import Handle


class ControllerPhysicsParams(NamedTuple):
    delta_time: float
    nsteps: int


class Controller(abc.ABC):
    def __init__(self, max_time_ms: float, cpp: ControllerPhysicsParams):
        self._max_time_ms = max_time_ms
        self.cpp = cpp

    @abc.abstractmethod
    def __call__(self, model: MjModel, data: MjData, viewer: Optional[Handle] = None):
        raise NotImplementedError

    @abc.abstractmethod
    def set_goal(self, goal, model: MjModel, data: MjData):
        raise NotImplementedError

    @property
    def max_time_ms(self):
        return self._max_time_ms

    @max_time_ms.setter
    def max_time_ms(self, max_time_ms: float):
        self._max_time_ms = max_time_ms

    @property
    def max_n_steps(self):
        return int((self.max_time_ms / (self.cpp.delta_time * 1000)) / self.cpp.nsteps)


class TrajFollower(abc.ABC):
    def __init__(self, controller: Controller):
        self.controller = controller

    @abc.abstractmethod
    def __call__(self, model: MjModel, data: MjData):
        raise NotImplementedError


class DataView(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class Agent(DataView):
    def __init__(self, model: MjModel):
        self.model = model

    @classmethod
    @abc.abstractmethod
    def from_model(
        cls,
        model: MjModel,
        sensor_name2id: Dict[str, int],
        sensordata_ptr: Optional[str] = None,
        namespace: str = "robot_0/",
    ) -> "Agent":
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def init_joint_pos() -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def init_gripper_pos() -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def check_collision(self, model: MjModel, data: MjData, grasped_objs: Set[int]) -> bool:
        """
        Check if the given state is in collision.
        Args:
            model (MjModel): The MuJoCo model containing the simulation data.
            data (MjData): The MuJoCo data object containing the current state of the simulation.
            grasped_objs (Set[int]): A set of object IDs that are currently grasped, can be empty.
        Returns:
            bool: True if there is a collision, False otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, model: MjModel, data: MjData) -> "Agent":
        raise NotImplementedError

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
        return grasped_objs

    def set_jointpos(self, data: MjData, joint_pos: np.ndarray):
        raise NotImplementedError

    def set_joint_ctrl(self, data: MjData, ctrl: np.ndarray):
        raise NotImplementedError

    def set_gripper_pos(self, data: MjData, pos: float):
        raise NotImplementedError

    def set_gripper_ctrl(self, data: MjData, ctrl: float):
        raise NotImplementedError

    def set_gripper_ctrl_open(self, data: MjData, open: bool):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def gripper_vel(self) -> float:
        """Gripper velocity"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def joint_pos(self) -> np.ndarray:
        """Array of joint positions"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def all_joint_pos(self) -> np.ndarray:
        """All joint positions (except base), which may include joints like the gripper or movable cameras."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def ee_pose_from_base(self) -> np.ndarray:
        """
        End-effector pose in base frame as 4x4 matrix.

        Axes convention for gripper:
            Origin between the fingers
            +x: towards the "side" of the gripper (axis of motion for fingers)
            +y: normal to grasp plane
            +z: along the gripper (towards the object)
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def base_pose(self) -> np.ndarray:
        """Base pose in global frame as 4x4 matrix"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def joint_limits(self):
        """Joint limits for all actuatable joints used for planning and IK"""
        raise NotImplementedError

    @abc.abstractmethod
    def ee_jacobian(self, model: MjModel, data: MjData) -> np.ndarray:
        """6xNDOF Jacobian of end-effector in base frame"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def root_id(self) -> int:
        """Body ID of the root body"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def finger_body_ids(self) -> Set[int]:
        """Set of body IDs corresponding to the fingers."""
        raise NotImplementedError
