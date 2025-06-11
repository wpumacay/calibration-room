import warnings
from typing import Union, List, Type
import numpy as np

import mujoco
from mujoco import MjModel, MjData, mjtObj
from mujoco.viewer import Handle

from mujoco_thor_abstract import Agent, Controller, ControllerPhysicsParams
from mujoco_thor_utils import extract_mj_names


class JointPosController(Controller):
    def __init__(
        self,
        model: MjModel,
        data: MjData,
        actuator_names: Union[str, List[str]],
        cpp: ControllerPhysicsParams,
        agent_cls: Type[Agent],
        threshold: Union[List[float], float],
        max_time_ms: float = 5000,  # ms
        namespace: str = "robot_0/",
        simulate_for_fixed_time: bool = False,
        verbose: bool = False,
    ):
        super().__init__(max_time_ms, cpp)
        self.namespace = namespace
        if isinstance(threshold, float):
            threshold = [threshold] * len(actuator_names)
        assert len(threshold) == len(actuator_names), (len(threshold), len(actuator_names))
        self.model = model
        self.data = data
        self.threshold = np.array(threshold)
        self.goal_updated = False
        self._action_dim = np.arange(len(actuator_names))
        self._dir = None
        self.namespace = namespace
        self._prev_joints = None
        self.simulate_for_fixed_time = simulate_for_fixed_time
        self._arm_err = np.zeros(len(actuator_names))

        self.verbose = verbose

        # actuators
        self.actuator_names = actuator_names
        (
            _,
            actuator_name2id,
            actuator_id2name,
        ) = extract_mj_names(
            model=self.model,
            num_obj=self.model.nu,
            obj_type=mjtObj.mjOBJ_ACTUATOR,
        )
        self.actuator_ids = [
            actuator_name2id[namespace + actuator_name] for actuator_name in self.actuator_names
        ]
        self.ctrl_index = [
            actuator_name2id[namespace + actuator_name] for actuator_name in self.actuator_names
        ]
        self.actuator_name2id = actuator_name2id
        self.actuator_id2name = actuator_id2name

        # agent
        self.agent_view = agent_cls.from_model(self.model, namespace=namespace)
        self.ctrl_min_range = np.array(
            [
                self.model.actuator(self.namespace + name).ctrlrange[0]
                for name in self.actuator_names
            ]
        )
        self.ctrl_max_range = np.array(
            [
                self.model.actuator(self.namespace + name).ctrlrange[1]
                for name in self.actuator_names
            ]
        )

    def __call__(
        self, model: MjModel, data: MjData, viewer: Handle = None
    ):  # =None as default. and use self.model, etc
        return self._feedback_loop(model, data, viewer)

    @property
    def curr_joints(self):
        return self._prev_joints

    def compute_ctrl_inputs(self, data=None):
        if not self.goal_updated:
            return np.array(
                [data.actuator(self.namespace + name).ctrl[0] for name in self.actuator_names]
            )

        goal_joint_positions = np.clip(
            self.goal_joint_positions, self.ctrl_min_range, self.ctrl_max_range
        )
        # TODO: safety measure (allow up to 50% max range for example)
        return goal_joint_positions

        # DEBUG: experiment with stepwise goal
        if curr is None:
            curr = self.curr_joints
        _sub_goal_joint_positions = curr + self._step_goal_dir
        _sub_goal_joint_positions = np.clip(
            _sub_goal_joint_positions, _sub_goal_joint_positions, self.goal_joint_positions
        )

        return _sub_goal_joint_positions

    def _feedback_loop(self, model, data, viewer=None):
        if self.goal_updated == False:
            raise ValueError("Goal pose not set. Call set_goal_pose() before calling controller.")

        if model.opt.timestep != self.cpp.delta_time:
            model.opt.timestep = self.cpp.delta_time

        goal_reached = False
        time = 0.0
        for _ in range(self.max_n_steps):
            if goal_reached and not self.simulate_for_fixed_time:
                if self.verbose:
                    print(f"[{self.__class__.__name__}] Goal Reached")
                break

            # compute control inputs
            ctrl_inputs = self.compute_ctrl_inputs()
            if ctrl_inputs is None:
                # use prev ctrl inputs
                ctrl_inputs = np.array(
                    [data.actuator(self.namespace + name).ctrl[0] for name in self.actuator_names]
                )

            # apply control inputs
            data.ctrl[self.ctrl_index] = ctrl_inputs
            mujoco.mj_step(m=model, d=data, nstep=self.cpp.nsteps)
            if viewer is not None:
                viewer.sync()

            # check if goal reached
            goal_reached = self._check_goal_reached(model, data)

            # check if timeout
            time += self.cpp.nsteps * self.cpp.delta_time * 1000  # ms

        if time >= self.max_time_ms:
            warnings.warn(f"[{self.__class__.__name__}] Timeout")

        # Unset goal
        self.reset()
        return {"goal_is_reached": np.all(goal_reached)}

    def set_delta_goal(
        self, delta_goal_joint_positions, model: MjModel, data: MjData, goal_joint_names=None
    ):
        self._dir = np.sign(delta_goal_joint_positions)
        # prev ctrl inputs
        # curr_joint_positions = np.array(
        #    [data.actuator(self.namespace + name).ctrl[0] for name in self.actuator_names]
        # )
        curr_joint_positions = self.agent_view(model, data).actuator_ctrl_inputs[self.ctrl_index]

        # actual joint position
        # curr_joint_positions = self.agent_view(model, data).get_actuator_joints(self.actuator_names)

        goal_joint_positions = curr_joint_positions + delta_goal_joint_positions
        return self._set_goal(goal_joint_positions, model, data, goal_joint_names)

    def set_goal(
        self, goal_joint_positions, model: MjModel, data: MjData, goal_joint_names=None, **kwargs
    ):
        # curr_joint_positions = self.agent_view(model, data).actuator_ctrl_inputs[self.ctrl_index]
        curr_joint_positions = self.agent_view(model, data).arm_joint_pos  # ignore base joints
        self._dir = np.sign(curr_joint_positions - goal_joint_positions)

        return self._set_goal(goal_joint_positions, model, data, goal_joint_names)

    def _set_goal(
        self, goal_joint_positions, model: MjModel, data: MjData, goal_joint_names=None, **kwargs
    ):
        if goal_joint_names is not None:
            # recompute index based on names
            self.ctrl_index = [
                self.actuator_name2id[self.namespace + actuator_name]
                for actuator_name in self.actuator_names
            ]
        self.goal_joint_positions = np.clip(
            goal_joint_positions, self.ctrl_min_range, self.ctrl_max_range
        )
        self.goal_updated = True

    def reset(self):
        self.goal_updated = False
        self._dir = None
        self._arm_err = None

    def _check_goal_reached(self, model: MjModel, data: MjData):
        # check if goal reached
        # curr_joint_positions = self.agent_view(model, data).joints
        # curr_joint_positions = np.array(
        #    [data.actuator(self.namespace + name).ctrl[0] for name in self.actuator_names]
        # )
        curr_joint_positions = self.agent_view(model, data).arm_joint_pos  # ignore base joints

        delta_arm = curr_joint_positions - self.goal_joint_positions
        delta_arm_err = np.abs(delta_arm)
        self._arm_err = delta_arm_err

        # check if curr has passed goal state
        _dir = np.sign(delta_arm)
        if self._dir is not None:
            if not (_dir[self._action_dim] == self._dir[self._action_dim]).any():
                if self.verbose:
                    print(f"Direction change detected {delta_arm_err[self._action_dim]}")
                return True
        self._dir = _dir

        ret = np.all(delta_arm_err[self._action_dim] < self.threshold[self._action_dim])
        # if ret:
        #    print("Error: ", delta_arm_err[self._action_dim])
        self._prev_joints = curr_joint_positions
        return ret
