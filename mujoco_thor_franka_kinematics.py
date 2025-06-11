import warnings
from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation as R

import mujoco
from mujoco import MjModel, MjData

from mujoco_thor_franka import FrankaFR3Agent
from mujoco_thor_kinematics import MujocoKinematics
from mujoco_thor_utils import inverse_homogeneous_matrix, transform_to_twist


class FrankaKinematics(MujocoKinematics):
    agent: FrankaFR3Agent

    def __init__(self, model: MjModel, data: Optional[MjData], namespace: str = "robot_0/"):
        if data is None:
            data = MjData(model)
        agent = FrankaFR3Agent(model, namespace)
        lo, hi = agent.joint_limits
        super().__init__(model, data, agent, lo, hi)
        mujoco.mj_forward(self.model, self.data)

    def _set_qpos(self, qpos: np.ndarray):
        self.agent.set_joint_qpos(self.data, qpos)

    def _enforce_joint_limits(self, q: np.ndarray):
        target = (self._state_space_low + self._state_space_high) / 2
        wrapped_q = target + np.arctan2(np.sin(q - target), np.cos(q - target))
        return wrapped_q

    def ik_pose(
        self,
        pose: np.ndarray,
        q0: Optional[np.ndarray] = None,
        sample_q0: bool = False,
        eps=1e-4,
        max_iter=1000,
        damping=1e-12,
        dt=1.0,
    ):
        """
        Perform inverse kinematics to compute the joint angles for the given position and orientation.
        Args:
            pose (np.ndarray): The 4x4 target pose matrix of the end effector.
            q0 (np.ndarray, optional): Initial guess for the joint angles. If None, a random sample will be used if sample_q0 is True.
            sample_q0 (bool, optional): Whether to sample a random initial guess for the joint angles. Implementation-defined if q0 is specified.
            eps (float, optional): Convergence threshold for the error norm. Default is 1e-4.
            max_iter (int, optional): Maximum number of iterations for the IK solver. Default is 1000.
            damping (float, optional): Damping factor for the weighted least squares solution. Default is 1e-12.
            dt (float, optional): Time step for integrating joint velocities. Default is 1.0.
        Returns:
            np.ndarray or None: The computed joint angles if the IK solver converges, otherwise None.
        """
        assert pose.shape == (4, 4)
        assert q0 is not None or sample_q0, "Must provide an initial guess or allow sampling!"

        if q0 is None or sample_q0:
            q0 = (
                np.random.rand(self.state_space_dim)
                * (self._state_space_high - self._state_space_low)
                + self._state_space_low
            )

        q = q0.copy()
        succ = False
        err = None
        i = 0
        for i in range(max_iter):
            # compute forward kinematics
            self.fk(q)
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_tendon(
                self.model, self.data
            )  # compute tendon length - need for stretch model
            mujoco.mj_sensorPos(
                self.model, self.data
            )  # compute sensor data - need for stretch data view
            self.agent(self.model, self.data)
            ee_pose = self.agent.ee_pose_from_base

            # compute error
            err_trf = inverse_homogeneous_matrix(ee_pose) @ pose
            twist_lin, twist_ang = transform_to_twist(err_trf)

            err = np.concatenate([ee_pose[:3, :3] @ twist_lin, ee_pose[:3, :3] @ twist_ang])
            if np.linalg.norm(err) < eps:
                succ = True
                break
            elif i == max_iter - 1:
                succ = False
                break

            # compute J
            J: np.ndarray = self.agent.ee_jacobian(self.model, self.data)
            if (JJT_det := np.linalg.det(J @ J.T)) < 1e-20:
                warnings.warn(f"IK Jacobian is rank deficient! det(JJ^T)={JJT_det:.0e}")

            # Weighted damped least squares
            W = self.W if self.W is not None else np.eye(J.shape[1])
            H = J @ W @ J.T + damping * np.eye(J.shape[0])
            q_dot = J.T @ np.linalg.solve(H, err)  # .ravel())

            # integrate the joint velocities in-place
            q += q_dot * dt

            # joint limit
            q = self._enforce_joint_limits(q)

        if succ:
            goal_joint_state = self.agent.joint_pos.copy()
            return goal_joint_state
        else:
            assert err is not None
            warnings.warn(
                f"Failed to solve IK, completed {i} iterations with remaining error {np.linalg.norm(err)}, q={q}"
            )
            return q0
