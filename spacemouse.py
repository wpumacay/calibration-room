"""Driver class for SpaceMouse controller. Modified based on the robosuite code.

This class provides a driver support to SpaceMouse on Mac OS X.
In particular, we assume you are using a SpaceMouse Wireless by default.

To set up a new SpaceMouse controller:
    1. Download and install driver from https://www.3dconnexion.com/service/drivers.html
    2. Install hidapi library through pip
       (make sure you run uninstall hid first if it is installed).
    3. Make sure SpaceMouse is connected before running the script
    4. (Optional) Based on the model of SpaceMouse, you might need to change the
       vendor id and product id that correspond to the device.

For Linux support, you can find open-source Linux drivers and SDKs online.
    See http://spacenav.sourceforge.net/

"""

import threading
import time
from collections import namedtuple

import numpy as np
import os
import sys

try:
    import hid
except ModuleNotFoundError as exc:
    raise ImportError(
        "Unable to load module hid, required to interface with SpaceMouse. "
        "Only Mac OS X is officially supported. Install the additional "
        "requirements with `pip install -r requirements-ik.txt`"
    ) from exc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scipy.spatial.transform import Rotation as R

AxisSpec = namedtuple("AxisSpec", ["channel", "byte1", "byte2", "scale"])

SPACE_MOUSE_SPEC = {
    "x": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
    "y": AxisSpec(channel=1, byte1=1, byte2=2, scale=-1),
    "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
    "roll": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
    "pitch": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
    "yaw": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
}

SPACE_MOUSE_WIRELESS_SPEC = {
    "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
    "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
    "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
    "roll": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
    "pitch": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
    "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
}


def to_int16(y1, y2):
    """
    Convert two 8 bit bytes to a signed 16 bit integer.

    Args:
        y1 (int): 8-bit byte
        y2 (int): 8-bit byte

    Returns:
        int: 16-bit integer
    """
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """
    Normalize raw HID readings to target range.

    Args:
        x (int): Raw reading from HID
        axis_scale (float): (Inverted) scaling factor for mapping raw input value
        min_v (float): Minimum limit after scaling
        max_v (float): Maximum limit after scaling

    Returns:
        float: Clipped, scaled input from HID
    """
    x = x / axis_scale
    x = min(max(x, min_v), max_v)
    return x


def convert(b1, b2):
    """
    Converts SpaceMouse message to commands.

    Args:
        b1 (int): 8-bit byte
        b2 (int): 8-bit byte

    Returns:
        float: Scaled value from Spacemouse message
    """
    return scale_to_control(to_int16(b1, b2))


def nms_max_axis(control: np.ndarray, threshold=0.6):
    """
    Suppress all but the axis with the maximum |value|.
    The max axis is set to -1 or 1 based on sign, others are zeroed.

    Args:
        control (np.ndarray): 6D input vector, assumed scaled in [-1, 1]
        threshold (float): minimum |value| to count as valid input

    Returns:
        np.ndarray: filtered control vector with only max direction
    """
    if np.all(np.abs(control) < threshold):
        return np.zeros_like(control)

    max_idx = np.argmax(np.abs(control))
    out = np.zeros_like(control)
    out[max_idx] = np.sign(control[max_idx])
    return np.array(out)


class SpaceMouse:
    """
    A minimalistic driver class for SpaceMouse with HID library.

    Note: Use hid.enumerate() to view all USB human interface devices (HID).
    Make sure SpaceMouse is detected before running the script.
    You can look up its vendor/product id from this method.

    Args:
        env (RobotEnv): The environment which contains the robot(s) to control
                        using this device.
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(
        self,
        vendor_id=9583,
        product_id=50746,  # 50746 for wireless, 50741 for wire, 50770 for usb receiver
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
    ):

        print("Opening SpaceMouse device")
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.device = hid.device()
        try:
            self.device.open(self.vendor_id, self.product_id)  # SpaceMouse
        except OSError as e:
            print("Failed to open SpaceMouse device cause: ", e)
            pass

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        print("Manufacturer: %s" % self.device.get_manufacturer_string())
        print("Product: %s" % self.device.get_product_string())

        # 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self.gripper = 0.0
        self.gripper_state = False  # Track gripper open/close state
        self.last_button_state = 0  # Track last button state

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self._enabled = False

        # launch a new listener thread to listen to SpaceMouse
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """

        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        # Reset 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        # Reset control
        self._control = np.zeros(6)
        # Reset grasp
        self.gripper = 0.0
        self.gripper_state = False  # Track gripper open/close state
        self.last_button_state = 0  # Track last button state

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    from scipy.spatial.transform import Rotation as R

    def rotation_matrix(self, angle, direction, point=None):
        direction = np.array(direction) / np.linalg.norm(direction)
        return R.from_rotvec(angle * direction).as_matrix()

    def get_controller_state(self):
        """
        Grabs the current state of the 3D mouse.

        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        dpos = self.control[:3] * 0.005 * self.pos_sensitivity
        roll, pitch, yaw = self.control[3:] * 0.005 * self.rot_sensitivity

        # convert RPY to an absolute orientation
        drot1 = self.rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = self.rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = self.rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.gripper,
            reset=self._reset_state,
        )

    def run(self):
        """Listener method that keeps pulling new messages."""

        t_last_click = -1

        while True:
            d = self.device.read(13)
            if d is not None and self._enabled:

                if self.product_id == 50741:
                    ## logic for older spacemouse model

                    if d[0] == 1:  ## readings from 6-DoF sensor
                        self.y = convert(d[1], d[2])
                        self.x = convert(d[3], d[4])
                        self.z = convert(d[5], d[6]) * -1.0

                    elif d[0] == 2:

                        self.roll = convert(d[1], d[2])
                        self.pitch = convert(d[3], d[4])
                        self.yaw = convert(d[5], d[6])

                        self._control = [
                            self.x,
                            self.y,
                            self.z,
                            self.roll,
                            self.pitch,
                            self.yaw,
                        ]
                else:
                    ## default logic for all other spacemouse models

                    if d[0] == 1:  ## readings from 6-DoF sensor
                        self.y = convert(d[1], d[2]) * -1.0
                        self.x = convert(d[3], d[4]) * -1.0
                        self.z = convert(d[5], d[6]) * -1.0

                        self.roll = convert(d[7], d[8]) * -1.0
                        self.pitch = convert(d[9], d[10]) * -1.0
                        self.yaw = convert(d[11], d[12]) * -1.0

                        self._control = [
                            self.x,
                            self.y,
                            self.z,
                            self.roll,
                            self.pitch,
                            self.yaw,
                        ]

                if d[0] == 3:  ## readings from the side buttons

                    # press left button
                    if d[1] == 1:
                        if not self.last_button_state:  # Button just pressed
                            self.gripper_state = not self.gripper_state  # Toggle gripper state
                            self.gripper = 255.0 if self.gripper_state else 0.0

                    # Update last button state
                    self.last_button_state = d[1]

                    # right button is for reset
                    # if d[1] == 2:
                    #     self._reset_state = 1
                    #     self._enabled = False
                    #     self._reset_internal_state()

    @property
    def control(self):
        """
        Grabs current pose of Spacemouse

        Returns:
            np.array: 6-DoF control value
        """
        return np.array(self._control)

    def _postprocess_device_outputs(self, dpos, drotation):
        drotation = drotation * 50
        dpos = dpos * 125

        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)

        return dpos, drotation


if __name__ == "__main__":

    space_mouse = SpaceMouse()
    space_mouse.start_control()
    while True:
        print(space_mouse.control)
        time.sleep(0.02)
