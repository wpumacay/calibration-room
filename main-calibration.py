import argparse
import glob
import json
import re
import multiprocessing
from enum import Enum
from multiprocessing import Process, Value, Lock
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Lock as LockType
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import glfw
import imgui
import numpy as np
import open3d as o3d
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
from scipy.spatial.transform import Rotation as R

import mujoco as mj
import mujoco.viewer as mjviewer

SPACEMOUSE_WORKING = True
try:
    from spacemouse import SpaceMouse
except ImportError:
    print(f"Spacemouse device is not available, using keyboard as backup")
    SPACEMOUSE_WORKING = False

from mujoco_thor_abstract import Agent, Controller, ControllerPhysicsParams
from mujoco_thor_kinematics import MujocoKinematics
from mujoco_thor_franka import FrankaFR3Agent
from mujoco_thor_franka_kinematics import FrankaKinematics
from mujoco_thor_joint_pos_controller import JointPosController


CURRENT_DIR = Path(__file__).parent
ASSETS_DIR = CURRENT_DIR / "assets"

SHOW_LEFT_UI = True
SHOW_RIGHT_UI = True

STR_DESC_STIFFNESS = """Joint stiffness. It represents how strong is the force
of a spring added at the equilibrium position.
"""

STR_DESC_FRICTIONLOSS = """Friction loss at the joint due to dry friction."""

STR_DESC_DAMPING = """Damping applied at the joint, which is a force
linear in velocity."""

STR_DESC_ARMATURE = """Extra inertia added to the joint not due to the body mass."""


# These are the defaults, but these are overwritten by the path from the config file (if given) ------------
EMPTY_SCENE_PATH = ASSETS_DIR / "empty_scene.xml"

ROBOT_ID_TO_PATH = {
    "stretch": ASSETS_DIR / "stretch_robots" / "stretch3.xml",
    "xarm7_ranger": ASSETS_DIR / "xarm7_ranger" / "xarm7_ranger.xml",
    "franka": ASSETS_DIR / "franka_fr3" / "fr3.xml",
}
# ----------------------------------------------------------------------------------------------------------

CATEGORIES_NAMES = [
    "Microwave",
    "Dresser",
    "Light_Switch",
    "Toilet",
    "Book",
    "Shelving_Unit",
    "Side_Table",
]

CATEGORY_ID_TO_POSITION = {
    # "Fridge": [0, -0.95, 0.75],
    "Microwave": [0.95, 0.0, 0.75],
    "Dresser": [0, -0.95, 0.75],
    "Light_Switch": [0, -0.95, 0.75],
    "Toilet": [0, -0.95, 0.75],
    "Book": [0, -0.95, 0.75],
    "Shelving_Unit": [0, -0.95, 0.75],
    "Side_Table": [0, -0.95, 0.75],
}

CATEGORY_ID_TO_EULER = {
    # "Fridge": [1.57, 0, 3.14],
    "Microwave": [1.57, 0.0, -1.57],
    "Dresser": [1.57, 0, 3.14],
    "Light_Switch": [1.57, 0, 3.14],
    "Toilet": [1.57, 0, 3.14],
    "Book": [1.57, 0, 3.14],
    "Shelving_Unit": [1.57, 0, 3.14],
    "Side_Table": [1.57, 0, 3.14],
}

# Default home keyframe for franka-fr3 arm
ROBOT_HOME = {
    "qpos": [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04],
    "qvel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "ctrl": [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.0],
}

class InputDevice(Enum):
    KEYBOARD = 0
    GAMEPAD = 1
    SPACEMOUSE = 2

class Context:
    def __init__(self):
        self.robot_id: str = "stretch"
        self.category_id: str = "Fridge"
        self.instances_per_category: List[Path] = []
        self.index_in_category: int = 0
        self.num_items_in_category: int = 1

        self.model: Optional[mj.MjModel] = None
        self.data: Optional[mj.MjData] = None

        self.dirty_next_model: bool = False  # Whether or not to load next model in category
        self.dirty_reload: bool = False  # Whether or not to reload the current model
        self.dirty_robot_home: bool = False
        
        self.use_table: bool = False
        self.use_item: bool = True

        self.robot_position_xyz = [0.0, 0.0, 0.0]
        self.robot_rotation_rpy = [0.0, 0.0, 0.0]

        self.ee_step_size_x = 0.01
        self.ee_step_size_y = 0.01
        self.ee_step_size_z = 0.01

        self.timestep = 0.002

        self.assets_dir = ASSETS_DIR

        self.target_pos_x = 0.5
        self.target_pos_y = 0.0
        self.target_pos_z = 0.5

        self.target_ee_roll = 0#2*np.pi
        self.target_ee_pitch = np.pi/2
        self.target_ee_yaw = 0.0

        self.target_pose = np.eye(4)

        self.gripper_state = 1

        self.input_device = InputDevice.KEYBOARD

        self.spacemouse_translation_sensitivity = 0.001
        self.spacemouse_rotation_sensitivity = 0.001


g_context: Context = Context()


def callback(keycode) -> None:
    global g_context

    # print(f"keycode: {keycode}, chr: {chr(keycode)}")

    if chr(keycode) == " ":
        g_context.index_in_category = (
            g_context.index_in_category + 1
        ) % g_context.num_items_in_category
        print(
            f"Loading next model: {g_context.instances_per_category[g_context.index_in_category]}"
        )
        g_context.dirty_next_model = True
    elif chr(keycode) == "E":
        print(
            f"Reloading current model: {g_context.instances_per_category[g_context.index_in_category]}"
        )
        g_context.dirty_reload = True
    elif chr(keycode) == "R":
        print(f"Reset robot to home pose")
        g_context.dirty_robot_home = True
    elif chr(keycode) == "ć":
        print("Moving end effector to the left")
        g_context.target_pos_x += g_context.ee_step_size_x
    elif chr(keycode) == "Ć":
        print("Moving end effector to the right")
        g_context.target_pos_x -= g_context.ee_step_size_x
    elif chr(keycode) == "ĉ":
        print("Moving end effector forward")
        g_context.target_pos_y -= g_context.ee_step_size_y
    elif chr(keycode) == "Ĉ":
        print("Moving end effector backward")
        g_context.target_pos_y += g_context.ee_step_size_y
    elif chr(keycode) == ";":
        print("Moving end effector up")
        g_context.target_pos_z += g_context.ee_step_size_z
    elif chr(keycode) == "'":
        print("Moving end effector down")
        g_context.target_pos_z -= g_context.ee_step_size_z
    elif chr(keycode) == "G":
        action = "Closing" if g_context.gripper_state == 1 else "Opening"
        print(f"{action} gripper")
        g_context.gripper_state = 1 - g_context.gripper_state
    elif chr(keycode) == "V":
        print("(+) Roll rotation for the gripper")
        g_context.target_ee_roll += 0.05
    elif chr(keycode) == ",":
        print("(-) Roll rotation for the gripper")
        g_context.target_ee_roll -= 0.05
    elif chr(keycode) == "B":
        print("(+) Pitch rotation for the gripper")
        g_context.target_ee_pitch += 0.05
    elif chr(keycode) == ".":
        print("(-) Pitch rotation for the gripper")
        g_context.target_ee_pitch -= 0.05
    elif chr(keycode) == "N":
        print("(+) Yaw rotation for the gripper")
        g_context.target_ee_yaw += 0.05
    elif chr(keycode) == "/":
        print("(-) Yaw rotation for the gripper")
        g_context.target_ee_yaw -= 0.05



def get_instances_per_category(category: str) -> List[Path]:
    global g_context
    base_path = str((g_context.assets_dir / "ThorAssets").resolve())
    path_candidates = [Path(path) for path in glob.glob(f"{base_path}/**/*.xml", recursive=True)]
    pattern = re.compile(category + r"_\d+")
    instances = []
    for path_candidate in path_candidates:
        if not path_candidate.is_file():
            continue
        if not pattern.match(path_candidate.stem):
            continue
        if 'old' in path_candidate.stem:
            continue
        if 'copy' in path_candidate.stem:
            continue
        instances.append(path_candidate)

    def extract_key(path: Path) -> int:
        numbers = re.findall(r"(\d+)", path.stem)
        if numbers:
            last_number = numbers[-1]
            return int(last_number)
        else:
            return 0

    return sorted(instances, key=extract_key)


def compute_aabb_from_model(model_xml_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    model = mj.MjModel.from_xml_path(str(model_xml_path.resolve()))
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    p_min = np.array([np.inf, np.inf, np.inf])
    p_max = np.array([-np.inf, -np.inf, -np.inf])
    for geom_idx in range(model.ngeom):
        pos = data.geom_xpos[geom_idx]
        rot = data.geom_xmat[geom_idx].reshape(3, 3)

        center = model.geom_aabb[geom_idx][:3]
        extents = model.geom_aabb[geom_idx][3:]

        x_min, x_max = center[0] - extents[0], center[0] + extents[0]
        y_min, y_max = center[1] - extents[1], center[1] + extents[1]
        z_min, z_max = center[2] - extents[2], center[2] + extents[2]

        aabb_vertices = np.array(
            [
                [x_min, y_min, z_min],
                [x_min, y_min, z_max],
                [x_min, y_max, z_min],
                [x_min, y_max, z_max],
                [x_max, y_min, z_min],
                [x_max, y_min, z_max],
                [x_max, y_max, z_min],
                [x_max, y_max, z_max],
            ],
            dtype=np.float64,
        )

        aabb_vertices = np.dot(aabb_vertices, rot.T) + pos

        min_point = np.min(aabb_vertices, axis=0)
        max_point = np.max(aabb_vertices, axis=0)

        p_min[0] = min(p_min[0], min_point[0])
        p_min[1] = min(p_min[1], min_point[1])
        p_min[2] = min(p_min[2], min_point[2])

        p_max[0] = max(p_max[0], max_point[0])
        p_max[1] = max(p_max[1], max_point[1])
        p_max[2] = max(p_max[2], max_point[2])
    
    return p_min, p_max

def reset_robot_to_home(model: mj.MjModel, data: mj.MjData, home: Dict[str, List[float]]) -> None:
    data.qpos[:9] = home["qpos"]
    data.qvel[:9] = home["qvel"]
    data.ctrl[:8] = home["ctrl"]

    mj.mj_forward(model, data)

def load_category_item(category_id: str, category_index: int) -> mj.MjSpec:
    obj_path = g_context.instances_per_category[category_index]
    obj_spec = mj.MjSpec.from_file(obj_path.as_posix())
    obj_root_body = obj_spec.worldbody.first_body()
    obj_root_body.pos = np.array(
        CATEGORY_ID_TO_POSITION.get(category_id, [0, -0.95, 0.75])
    )
    obj_root_body.quat = R.from_euler(
        "xyz", CATEGORY_ID_TO_EULER.get(category_id, [1.57, 0, 3.14])
    ).as_quat(scalar_first=True)

    # --------------------------------------------------------------------------
    # Remove all the primitive colliders from the original model and replace
    # these colliders with the mesh colliders that are available in the folder

    def random_rgba():
        return np.array([np.random.random(), np.random.random(), np.random.random(), 1.0], dtype=np.float32)

    meshes_names = [mesh.name for mesh in obj_spec.meshes]
    for body in obj_spec.bodies:
        mesh_to_add: List[str] = []
        primitive_geoms: List[mj.MjsGeom] = []
        for geom in body.geoms:
            if geom.classname.name == "__VISUAL_MJT__":
                mesh_to_add.append(geom.meshname)
            elif geom.classname.name == "__DYNAMIC_MJT__":
                primitive_geoms.append(geom)

        for mesh_name in mesh_to_add:
            mesh_idx = meshes_names.index(mesh_name)
            if mesh_idx == -1:
                # Mesh with given name not found (weird, but just in case check)
                continue
            mesh_spec = obj_spec.meshes[mesh_idx]
            mesh_scale = mesh_spec.scale
            mesh_filepath = Path(mesh_spec.file)
            base_dir = obj_path.parent
            collider_workdir = base_dir / mesh_filepath.parent / mesh_filepath.stem
            all_obj_files = collider_workdir.glob("**/*_collision_*.obj")

            for collider_idx, collider_obj_file in enumerate(all_obj_files):
                mesh = o3d.io.read_triangle_mesh(collider_obj_file.as_posix())
                shellinertia = True
                if mesh.is_watertight():
                    volume = mesh.get_volume()
                    if volume > 0:
                        shellinertia = False

                asset_mesh_name = collider_obj_file.stem
                relative_path = collider_obj_file.relative_to(obj_path.parent)
                obj_spec.add_mesh(
                    name=asset_mesh_name,
                    file=relative_path.as_posix(),
                    scale=mesh_scale,
                    inertia=mj.mjtMeshInertia.mjMESH_INERTIA_SHELL if shellinertia else mj.mjtMeshInertia.mjMESH_INERTIA_LEGACY
                )
                body.add_geom(
                    type=mj.mjtGeom.mjGEOM_MESH,
                    name=f"{category_id}_{mesh_name}_MeshCollider_{collider_idx}",
                    meshname=asset_mesh_name,
                    # classname="__DYNAMIC_MJT__",
                    contype=1,
                    conaffinity=15,
                    group=4,
                    friction=np.array([0.90000000000000002, 0.90000000000000002, 0.001]),
                    solref=np.array([0.025000000000000001, 1]),
                    # TODO(wilbert): solimp expects a length 5 array
                    # solimp=np.array([0.998, 0.998, 0.001]),
                    rgba=random_rgba(),
                )

        for prim_geom in primitive_geoms:
            prim_geom.delete()
    # --------------------------------------------------------------------------

    return obj_spec

def load_scene(
    robot_id: str,
    category_id: str,
    category_index: int = 0,
    use_item: bool = True,
) -> Tuple[mj.MjModel, mj.MjData]:
    global g_context
    root_spec = mj.MjSpec.from_file(str(EMPTY_SCENE_PATH.resolve()))

    # Set simulation properties similar to stretch scene
    root_spec.option.timestep = g_context.timestep
    root_spec.option.impratio = 100
    root_spec.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST.value
    root_spec.option.cone = mj.mjtCone.mjCONE_ELLIPTIC.value
    root_spec.option.jacobian = mj.mjtJacobian.mjJAC_SPARSE.value
    root_spec.option.solver = mj.mjtSolver.mjSOL_NEWTON.value
    root_spec.option.noslip_iterations = 3

    root_spec_frame = root_spec.worldbody.add_frame()

    robot_spec = mj.MjSpec.from_file(str(ROBOT_ID_TO_PATH[robot_id].resolve()))
    robot_root_body = robot_spec.worldbody.first_body()
    root_spec_frame.attach_body(robot_root_body, "robot-", "")

    if use_item:
        category_item_spec = load_category_item(category_id, category_index)
        category_item_root_body = category_item_spec.worldbody.first_body()

        # root_spec_frame.attach_body(category_item_root_body, f"{category_id}-", "")
        root_spec_frame.attach_body(category_item_root_body, "", "")

        if g_context.use_table:
            robot_pmin, robot_pmax = compute_aabb_from_model(ROBOT_ID_TO_PATH[robot_id])
            asset_pmin, asset_pmax = compute_aabb_from_model(g_context.instances_per_category[category_index])

            # print(f"robot pmin: {robot_pmin} / pmax: {robot_pmax}")
            # print(f"asset pmin: {asset_pmin} / pmax: {asset_pmax}")

            robot_height = robot_pmax[2] - robot_pmin[2]
            asset_height = asset_pmax[1] - asset_pmin[1] # Asset is rotated

            if robot_height > asset_height:
                diff_height = robot_height - asset_height
                # print(f"robot-height: {robot_height}")
                # print(f"asset-height: {asset_height}")
                # print(f"diff-height: {diff_height}")
                table_body = root_spec.worldbody.add_body(
                    name="table", pos=[1.0, 0.0, diff_height / 4]
                )
                table_body.add_geom(
                    type=mj.mjtGeom.mjGEOM_BOX,
                    size=[0.6, 0.5, diff_height / 4],
                    mass=1,
                    material="wood",
                )

    model = root_spec.compile()
    data = mj.MjData(model)

    reset_robot_to_home(model, data, ROBOT_HOME)

    with open("model.xml", "w") as fhandle:
        fhandle.write(root_spec.to_xml())

    return model, data

class JointInfo:
    def __init__(self):
        self.jnt_id: int = 0
        self.jnt_type: mj.mjtJoint = mj.mjtJoint.mjJNT_HINGE
        self.jnt_name: str = ''
        self.jnt_range: Tuple[float, float] = (-np.inf, np.inf)

        self.jnt_mpvar_qpos: Optional[Synchronized[float]] = None
        self.jnt_mpvar_stiffness: Optional[Synchronized[float]] = None
        self.jnt_mpvar_armature: Optional[Synchronized[float]] = None
        self.jnt_mpvar_damping: Optional[Synchronized[float]] = None
        self.jnt_mpvar_frictionloss: Optional[Synchronized[float]] = None

def run_imgui_interface(
        var_jnt_id: Synchronized,
        var_jnt_qpos_change: Synchronized,
        joints_info: Dict[int, JointInfo],
        lock: LockType,
    ):
    if not glfw.init():
        print("Couldn't initialize GLFW")
        return
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    window = glfw.create_window(500, 400, "Properties tuning application", None, None)

    if window is None:
        print("GLFW Window creation failed")
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    imgui.create_context()
    imgui_renderer = GlfwRenderer(window)

    jnt_selected = 0

    jnt_ids = list(joints_info.keys())
    jnt_names = [joints_info[key].jnt_name for key in joints_info]

    while not glfw.window_should_close(window):
        glfw.poll_events()
        imgui_renderer.process_inputs()

        imgui.new_frame()
        # GUI creation goes here -----------------------------------------------

        io = imgui.get_io()
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(io.display_size.x, io.display_size.y)
        # Combo-box used to select which joint to configure
        with imgui.begin("Simulation properties"):
            if len(joints_info) > 0:
                with imgui.begin_combo("Joint", jnt_names[jnt_selected]) as combo:
                    if combo.opened:
                        for i, jnt_name in enumerate(jnt_names):
                            is_selected = (i == jnt_selected)
                            if imgui.selectable(jnt_name, is_selected)[0]:
                                jnt_selected = i
                                with lock:
                                    var_jnt_id.value = jnt_ids[jnt_selected]

                            if is_selected:
                                imgui.set_item_default_focus()

                if var_jnt_id.value in joints_info:
                    # changed, qpos_value = imgui.slider_float(
                    #     "Value",
                    #     joints_info[var_jnt_id.value].jnt_mpvar_qpos.value,
                    #     min_value=joints_info[var_jnt_id.value].jnt_range[0],
                    #     max_value=joints_info[var_jnt_id.value].jnt_range[1],
                    # )
                    # if changed:
                    #     with lock:
                    #         joints_info[var_jnt_id.value].jnt_mpvar_qpos.value = qpos_value
                    #         var_jnt_qpos_change.value = 1

                    changed, stiffness_value = imgui.input_float("Stiffness", joints_info[var_jnt_id.value].jnt_mpvar_stiffness.value)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip(STR_DESC_STIFFNESS)
                    if changed:
                        with lock:
                            joints_info[var_jnt_id.value].jnt_mpvar_stiffness.value = stiffness_value

                    changed, frictionloss_value = imgui.input_float("FrictionLoss", joints_info[var_jnt_id.value].jnt_mpvar_frictionloss.value)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip(STR_DESC_FRICTIONLOSS)
                    if changed:
                        with lock:
                            joints_info[var_jnt_id.value].jnt_mpvar_frictionloss.value = frictionloss_value

                    changed, damping_value = imgui.input_float("Damping", joints_info[var_jnt_id.value].jnt_mpvar_damping.value)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip(STR_DESC_DAMPING)
                    if changed:
                        with lock:
                            joints_info[var_jnt_id.value].jnt_mpvar_damping.value = damping_value

                    changed, armature_value = imgui.slider_float(
                        "Armature",
                        joints_info[var_jnt_id.value].jnt_mpvar_armature.value,
                        min_value=0.0,
                        max_value=1.0,
                    )
                    if imgui.is_item_hovered():
                        imgui.set_tooltip(STR_DESC_ARMATURE)
                    if changed:
                        with lock:
                            joints_info[var_jnt_id.value].jnt_mpvar_armature.value = armature_value

                    save = imgui.button("Save Properties")
                    if save:
                        with lock:
                            json_data = dict(joints={})
                            for jnt_id, jnt_name in zip(jnt_ids, jnt_names):
                                json_data["joints"][jnt_name] = {}
                                json_data["joints"][jnt_name]["stiffness"] = joints_info[jnt_id].jnt_mpvar_stiffness.value
                                json_data["joints"][jnt_name]["armature"] = joints_info[jnt_id].jnt_mpvar_armature.value
                                json_data["joints"][jnt_name]["damping"] = joints_info[jnt_id].jnt_mpvar_damping.value
                                json_data["joints"][jnt_name]["frictionloss"] = joints_info[jnt_id].jnt_mpvar_frictionloss.value

                            with open("parameters.json", "w") as fhandle:
                                json.dump(json_data, fhandle, indent=4)

            else:
                imgui.text("No joints for this object")

        # ----------------------------------------------------------------------

        imgui.render()

        gl.glClearColor(0.0, 0.0, 0.2, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui_renderer.render(imgui.get_draw_data())

        glfw.swap_buffers(window)

    imgui_renderer.shutdown()
    glfw.terminate()

def run_launcher_interface(
        categories: Dict[str, List[Path]],
        var_category_id: Synchronized,
        var_item_id: Synchronized,
        var_model_change: Synchronized,
        var_robot_home: Synchronized,
        var_spacemouse_trans_sens: Synchronized,
        var_spacemouse_rot_sens: Synchronized,
        var_spacemouse_sens_change: Synchronized,
        lock: LockType
    ):
    if not glfw.init():
        print("Couldn't initialize GLFW")
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    window = glfw.create_window(500, 400, "Model selector", None, None)

    if window is None:
        print("GLFW Window creation failed")
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    imgui.create_context()
    imgui_renderer = GlfwRenderer(window)

    categories_names = list(categories.keys())
    category_selected = 0

    items_paths = categories[categories_names[category_selected]]
    items_names = [item_path.stem for item_path in items_paths]
    item_selected = 0

    while not glfw.window_should_close(window):
        glfw.poll_events()
        imgui_renderer.process_inputs()

        imgui.new_frame()
        # GUI creation goes here -----------------------------------------------

        # Combo-box used to select which joint to configure
        io = imgui.get_io()
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(io.display_size.x, io.display_size.y)
        with imgui.begin("Launcher options"):
            imgui.text("Model options")
            imgui.spacing()
            if len(categories_names) > 0:
                with imgui.begin_combo("Category", categories_names[category_selected]) as combo:
                    if combo.opened:
                        for i, category_name in enumerate(categories_names):
                            is_selected = (i == category_selected)
                            if imgui.selectable(category_name, is_selected)[0]:
                                category_selected = i
                                with lock:
                                    var_category_id.value = category_selected

                            if is_selected:
                                imgui.set_item_default_focus()
                items_paths = categories[categories_names[category_selected]]
                items_names = [item_path.stem for item_path in items_paths]
                with imgui.begin_combo("Item", items_names[item_selected]) as combo:
                    if combo.opened:
                        for i, item_name in enumerate(items_names):
                            is_selected = (i == item_selected)
                            if imgui.selectable(item_name, is_selected)[0]:
                                item_selected = i
                                with lock:
                                    var_item_id.value = item_selected
                                    
                if imgui.button("Load model"):
                    with lock:
                        var_model_change.value = 1
            else:
                imgui.text("No models found, maybe setup was wrong")

            imgui.spacing()
            imgui.text("Simulation options")
            if imgui.button("Robot Home"):
                with lock:
                    var_robot_home.value = 1

            imgui.spacing()
            imgui.text("Controls options")
            changed, trans_sens_value = imgui.slider_float(
                "Translation Sensitivity",
                var_spacemouse_trans_sens.value,
                0.001,
                0.01,
            )
            if changed:
                with lock:
                    var_spacemouse_sens_change.value = 1
                    var_spacemouse_trans_sens.value = trans_sens_value
            
            changed, rot_sens_value = imgui.slider_float(
                "Rotation Sensitivity",
                var_spacemouse_rot_sens.value,
                0.001,
                0.01,
            )
            if changed:
                with lock:
                    var_spacemouse_sens_change.value = 1
                    var_spacemouse_rot_sens.value = rot_sens_value

        # ----------------------------------------------------------------------

        imgui.render()

        gl.glClearColor(0.0, 0.0, 0.2, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui_renderer.render(imgui.get_draw_data())

        glfw.swap_buffers(window)

    imgui_renderer.shutdown()
    glfw.terminate()
    pass


def load_joints(model: mj.MjModel, data: mj.MjData) -> Dict[int, JointInfo]:
    joints_info: Dict[int, JointInfo] = {}
    for idx in range(model.njnt):
        jnt_info = JointInfo()
        jnt_info.jnt_id = idx
        jnt_info.jnt_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT.value, idx)
        jnt_info.jnt_range = (model.jnt_range[idx][0], model.jnt_range[idx][1])
        if jnt_info.jnt_name is None:
            # Skip joints that don't have a valid name associated
            continue
        jnt_info.jnt_type = model.jnt_type[idx]
        if jnt_info.jnt_type == mj.mjtJoint.mjJNT_FREE:
            # Skip joints that are of the free type
            continue
        if "robot" in jnt_info.jnt_name:
            # Don't include joints from the robot
            continue
        jnt_qpos_adr = model.jnt_qposadr[idx]
        jnt_dof_adr = model.jnt_dofadr[idx]
        jnt_info.jnt_mpvar_qpos = Value('d', data.qpos[jnt_qpos_adr])
        jnt_info.jnt_mpvar_stiffness = Value('d', model.jnt_stiffness[idx])
        jnt_info.jnt_mpvar_armature = Value('d', model.dof_armature[jnt_dof_adr])
        jnt_info.jnt_mpvar_damping = Value('d', model.dof_damping[jnt_dof_adr])
        jnt_info.jnt_mpvar_frictionloss = Value('d', model.dof_frictionloss[jnt_dof_adr])

        joints_info[jnt_info.jnt_id] = jnt_info
    return joints_info

def load_categories() -> Dict[str, List[Path]]:
    global g_context
    global CATEGORIES_NAMES
    valid_categories = {category: [] for category in CATEGORIES_NAMES}

    for category in valid_categories.keys():
        valid_categories[category] = get_instances_per_category(category)

    return valid_categories

def wrap_ee_to_joint(
    agent: Agent,
    kinematics: MujocoKinematics,
    joint_control: np.ndarray,
    gripper_control: float,
):
    global g_context
    assert g_context.model is not None
    assert g_context.data is not None

    current_joints = agent(g_context.model, g_context.data).all_joint_pos.copy()

    if np.all(np.abs(joint_control) < 0.5):
        return current_joints, gripper_control, False

    dy, dx, dz = joint_control[:3] * g_context.spacemouse_translation_sensitivity
    droll, dpitch, dyaw = joint_control[3:] * g_context.spacemouse_rotation_sensitivity

    drot = R.from_euler("xyz", [droll, dpitch, dyaw]).as_matrix()

    delta_transform_pos = np.eye(4)
    delta_transform_pos[:3, 3] = [dx, -dy, dz]

    delta_transform_rot = np.eye(4)
    delta_transform_rot[:3, :3] = drot

    new_target_pose = (delta_transform_pos @ g_context.target_pose) @ delta_transform_rot

    target_joints = kinematics.ik_pose(pose=new_target_pose, q0=current_joints)
    if np.allclose(target_joints, current_joints, atol=1e-4):
        return current_joints, gripper_control, False

    g_context.target_pos_x = new_target_pose[0, 3]
    g_context.target_pos_y = new_target_pose[1, 3]
    g_context.target_pos_z = new_target_pose[2, 3]

    ee_angles = R.from_matrix(new_target_pose[:3, :3]).as_euler("xyz")
    g_context.target_ee_roll = ee_angles[0]
    g_context.target_ee_pitch = ee_angles[1]
    g_context.target_ee_yaw = ee_angles[2]

    g_context.target_pose = new_target_pose
    return target_joints, gripper_control, True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        type=str,
        default="franka",
        help="The relative path to the XML file for the robot you want to use",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="Microwave",
        help="The category of assets to be load for the calibration test",
    )
    parser.add_argument(
        "--nogui",
        action="store_true",
        help="Whether or not to launch the GUI for parameter tunning",
    )
    parser.add_argument(
        "--nolauncher",
        action="store_true",
        help="Whether or not to launch the model selector GUI",
    )
    parser.add_argument(
        "--notable",
        action="store_true",
        help="Whether or not to spawn a table to compensate for the height difference"
    )

    args = parser.parse_args()

    global g_context
    global EMPTY_SCENE_PATH
    global ROBOT_ID_TO_PATH
    global SPACEMOUSE_WORKING
    g_context.robot_id = args.robot
    g_context.category_id = args.category

    try:
        config_filepath = str((CURRENT_DIR / "main-calibration.json").resolve())
        with open(config_filepath, "r") as fhandle:
            json_data = json.load(fhandle)
            config_data = json_data.get("config", {})
            g_context.use_item = config_data.get("use_item", True)
            g_context.ee_step_size_x = config_data.get("ee_step_size_x", 0.01)
            g_context.ee_step_size_y = config_data.get("ee_step_size_y", 0.01)
            g_context.ee_step_size_z = config_data.get("ee_step_size_z", 0.01)
            g_context.timestep = config_data.get("timestep", 0.002)
            config_assets_dir = config_data.get("assets_dir", "")
            if config_assets_dir != "":
                g_context.assets_dir = Path(config_assets_dir)
                print(f"Using assets_dir: {str(g_context.assets_dir.resolve())}")

            EMPTY_SCENE_PATH = g_context.assets_dir / "empty_scene.xml"

            ROBOT_ID_TO_PATH = {
                "stretch": g_context.assets_dir / "stretch_robots" / "stretch3.xml",
                "xarm7_ranger": g_context.assets_dir / "xarm7_ranger" / "xarm7_ranger.xml",
                "franka": g_context.assets_dir / "franka_fr3" / "fr3.xml",
            }

            print("Successfully loaded config data from config file")
    except FileNotFoundError:
        print("No config file provided, using defaults instead")
        pass
    except json.JSONDecodeError as e:
        print(f"Error parsing the JSON config file: {e}. Using defaults instead")
        pass

    g_context.instances_per_category = get_instances_per_category(args.category)
    g_context.num_items_in_category = len(g_context.instances_per_category)
    g_context.use_table = not args.notable

    # Check if we have access to a spacemouse ----------------------------------
    try:
        spacemouse = SpaceMouse() if SPACEMOUSE_WORKING else None
    except ValueError:
        SPACEMOUSE_WORKING = False
        spacemouse = None
    if spacemouse:
        g_context.input_device = InputDevice.SPACEMOUSE
        spacemouse.start_control()
    # --------------------------------------------------------------------------

    print(f"Loading model category: {g_context.category_id}")
    print(f"Index in category: {g_context.index_in_category}")
    print(f"Instances per category: {g_context.instances_per_category}")
    print(f"Loading model: {g_context.instances_per_category[g_context.index_in_category]}")

    # TODO(wilbert): might need to set simulation parameters to get a stable stretch robot simulation
    model, data = load_scene(args.robot, args.category, 0, g_context.use_item)
    g_context.model = model
    g_context.data = data

    print(f"model.nq : {model.nq}")
    print(f"model.nv : {model.nv}")

    var_jnt_id = Value('i', -1)
    var_jnt_qpos_change = Value('i', 0)
    var_category_id = Value('i', 0)
    var_item_id = Value('i', 0)
    var_model_change = Value('i', 0)
    var_robot_home = Value('i', 0)
    var_spacemouse_trans_sens = Value('d', g_context.spacemouse_translation_sensitivity)
    var_spacemouse_rot_sens = Value('d', g_context.spacemouse_rotation_sensitivity)
    var_spacemouse_sens_change = Value('i', 0)
    lock = Lock()

    joints_info: Dict[int, JointInfo] = load_joints(model, data)
    gui_process = None
    if not args.nogui:
        gui_process = Process(target=run_imgui_interface, args=(var_jnt_id, var_jnt_qpos_change, joints_info, lock))
        gui_process.start()

    categories_info: Dict[str, List[Path]] = load_categories()
    launcher_process = None
    if not args.nolauncher:
        launcher_process = Process(
            target=run_launcher_interface,
            args=(
                categories_info,
                var_category_id,
                var_item_id,
                var_model_change,
                var_robot_home,
                var_spacemouse_trans_sens,
                var_spacemouse_rot_sens,
                var_spacemouse_sens_change,
                lock,
            )
        )
        launcher_process.start()

    agent = FrankaFR3Agent(model=model, namespace="robot-")
    kinematics = FrankaKinematics(model=model, data=data, namespace="robot-")
    controller = JointPosController(
        model=model,
        data=data,
        actuator_names=[
                "fr3_joint1",
                "fr3_joint2",
                "fr3_joint3",
                "fr3_joint4",
                "fr3_joint5",
                "fr3_joint6",
                "fr3_joint7",
        ],
        cpp=ControllerPhysicsParams(0.002, 10),
        agent_cls=FrankaFR3Agent,
        threshold=0.035,
        max_time_ms=100,
        namespace="robot-",
        simulate_for_fixed_time=False,
    )

    controller.set_goal(agent.INIT_JOINT_POS, model, data)
    g_context.target_pose = agent(model, data).ee_pose_from_base.copy()

    mj.mj_forward(model, data)

    with mjviewer.launch_passive(model, data, key_callback=callback, show_left_ui=SHOW_LEFT_UI, show_right_ui=SHOW_RIGHT_UI) as viewer:
        viewer.cam.lookat = [0.04764209, -0.02826854, 0.41072837]
        viewer.cam.azimuth = -94.582
        viewer.cam.elevation = -25.376
        viewer.cam.distance = 1.311
        while viewer.is_running():
            if g_context.dirty_robot_home:
                g_context.dirty_robot_home = False

                controller.set_goal(agent.INIT_JOINT_POS, model, data)
                controller(model, data)

                data.qpos[:6] = agent.INIT_JOINT_POS[:6]
                data.qvel[:6] = np.zeros(6)

                mj.mj_step(model, data)

                g_context.target_pose = agent(model, data).ee_pose_from_base.copy()
                g_context.target_pos_x = g_context.target_pose[0, 3].item()
                g_context.target_pos_y = g_context.target_pose[1, 3].item()
                g_context.target_pos_z = g_context.target_pose[2, 3].item()

                roll, pitch, yaw = R.from_matrix(g_context.target_pose[:3, :3]).as_euler("xyz")
                g_context.target_ee_roll = roll
                g_context.target_ee_pitch = pitch
                g_context.target_ee_yaw = yaw

                data.mocap_pos[0] = [g_context.target_pos_x, g_context.target_pos_y, g_context.target_pos_z]
                data.mocap_quat[0] = R.from_euler("xyz", [g_context.target_ee_roll, g_context.target_ee_pitch, g_context.target_ee_yaw]).as_quat(scalar_first=True)

            if g_context.dirty_next_model or g_context.dirty_reload:
                # Load the next model into the simulation ----------------------
                g_context.dirty_next_model = False
                g_context.dirty_reload = False
                model, data = load_scene(
                    g_context.robot_id, g_context.category_id, g_context.index_in_category, g_context.use_item
                )
                sim = viewer._get_sim()
                if sim is not None:
                    sim.load(model, data, "Categories Visualizer")
                    viewer._user_scn = mj.MjvScene(model, sim.MAX_GEOM)

                viewer.cam.lookat = [0.04764209, -0.02826854, 0.41072837]
                viewer.cam.azimuth = -94.582
                viewer.cam.elevation = -25.376
                viewer.cam.distance = 1.311

                g_context.model = model
                g_context.data = data

                agent = FrankaFR3Agent(model=model, namespace="robot-")
                kinematics = FrankaKinematics(model=model, data=data, namespace="robot-")
                controller = JointPosController(
                    model=model,
                    data=data,
                    actuator_names=[
                            "fr3_joint1",
                            "fr3_joint2",
                            "fr3_joint3",
                            "fr3_joint4",
                            "fr3_joint5",
                            "fr3_joint6",
                            "fr3_joint7",
                    ],
                    cpp=ControllerPhysicsParams(0.002, 10),
                    agent_cls=FrankaFR3Agent,
                    threshold=0.035,
                    max_time_ms=100,
                    namespace="robot-",
                    simulate_for_fixed_time=False,
                )

                controller.set_goal(agent.INIT_JOINT_POS, model, data)
                g_context.target_pose = agent(model, data).ee_pose_from_base.copy()

                if not args.nogui:
                    if gui_process is not None and gui_process.is_alive():
                        gui_process.kill()

                    with lock:
                        joints_info = load_joints(model, data)

                    var_jnt_id = Value('i', -1)
                    lock = Lock()
                    gui_process = Process(target=run_imgui_interface, args=(var_jnt_id, var_jnt_qpos_change, joints_info, lock))
                    gui_process.start()

                mj.mj_forward(model, data)
                viewer.sync()
                # --------------------------------------------------------------

            # camera = viewer.cam
            # print(f"pos: {camera.lookat}")
            # print(f"distance: {camera.distance}")
            # print(f"azimuth: {camera.azimuth}")
            # print(f"elevation: {camera.elevation}")

            # Handle IK based on input device ----------------------------------
            if g_context.input_device == InputDevice.SPACEMOUSE:
                assert spacemouse is not None
                target_joints, gripper_control, update_controller = wrap_ee_to_joint(agent, kinematics, spacemouse.control, spacemouse.gripper)
            elif g_context.input_device ==InputDevice.KEYBOARD:
                gripper_control = 255.0 * g_context.gripper_state
                update_controller = True

                new_target_pose = np.eye(4)
                new_target_pose[:3, 3] = [g_context.target_pos_x, g_context.target_pos_y, g_context.target_pos_z]
                new_target_pose[:3, :3] = R.from_euler("xyz", [g_context.target_ee_roll, g_context.target_ee_pitch, g_context.target_ee_yaw]).as_matrix()

                current_joints = agent(model, data).all_joint_pos.copy()
                target_joints = kinematics.ik_pose(pose=new_target_pose, q0=current_joints)
                if np.allclose(target_joints, current_joints, atol=1e-4):
                    update_controller = False
                
                g_context.target_pose = new_target_pose
            elif g_context.input_device == InputDevice.GAMEPAD:
                # TODO(wilbert): handle gamepad-based updates here
                update_controller = False
                gripper_control = 255.0 * g_context.gripper_state
            else:
                update_controller = False
                gripper_control = 255.0 * g_context.gripper_state

            if update_controller:
                controller.set_goal(target_joints, model, data)
                controller(model, data)
            agent.set_gripper_ctrl(data, gripper_control)

            data.mocap_pos[0] = [g_context.target_pos_x, g_context.target_pos_y, g_context.target_pos_z]
            data.mocap_quat[0] = R.from_euler("xyz", [g_context.target_ee_roll, g_context.target_ee_pitch, g_context.target_ee_yaw]).as_quat(scalar_first=True)
            # ------------------------------------------------------------------

            # Update the model parameters from the GUI -------------------------
            if not args.nogui:
                if var_jnt_id.value != -1:
                    jnt_id = var_jnt_id.value
                    with lock:
                        jnt_dof_adr = model.jnt_dofadr[jnt_id]
                        jnt_qpos_adr = model.jnt_qposadr[jnt_id]
                        model.jnt_stiffness[jnt_id] = joints_info[jnt_id].jnt_mpvar_stiffness.value
                        model.dof_armature[jnt_dof_adr] = joints_info[jnt_id].jnt_mpvar_armature.value
                        model.dof_damping[jnt_dof_adr] = joints_info[jnt_id].jnt_mpvar_damping.value
                        model.dof_frictionloss[jnt_dof_adr] = joints_info[jnt_id].jnt_mpvar_frictionloss.value
                        if var_jnt_qpos_change.value == 1:
                            # Change only when UI slider changed
                            var_jnt_qpos_change.value = 0
                            data.qpos[jnt_qpos_adr] = joints_info[jnt_id].jnt_mpvar_qpos.value
                            data.qvel[jnt_dof_adr] = 0.0
                        else:
                            joints_info[jnt_id].jnt_mpvar_qpos.value = data.qpos[jnt_qpos_adr]

            # ------------------------------------------------------------------

            # Check for the launcher GUI logic ---------------------------------
            if not args.nolauncher:
                if var_model_change.value == 1:
                    with lock:
                        var_model_change.value = 0
                        if var_category_id.value != -1 and var_item_id.value != -1:
                            categories_names = list(categories_info.keys())
                            category_name = categories_names[var_category_id.value]

                            g_context.dirty_next_model = True
                            g_context.category_id = category_name
                            g_context.index_in_category = var_item_id.value
                            g_context.instances_per_category = categories_info[category_name]
                if var_robot_home.value == 1:
                    with lock:
                        var_robot_home.value = 0
                        g_context.dirty_robot_home = True
                if var_spacemouse_sens_change.value == 1:
                    with lock:
                        var_spacemouse_sens_change.value = 0
                        g_context.spacemouse_translation_sensitivity = var_spacemouse_trans_sens.value
                        g_context.spacemouse_rotation_sensitivity = var_spacemouse_rot_sens.value               
            # ------------------------------------------------------------------

            mj.mj_step(model, data)
            viewer.sync()

    if gui_process is not None and gui_process.is_alive():
        gui_process.join()
    if launcher_process is not None and launcher_process.is_alive():
        launcher_process.join()
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
