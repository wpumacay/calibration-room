import argparse
import glob
import re
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

CURRENT_DIR = Path(__file__).parent
ASSETS_DIR = CURRENT_DIR / "assets"
EMPTY_SCENE_PATH = ASSETS_DIR / "empty_scene.xml"

ROBOT_ID_TO_PATH = {
    "stretch": ASSETS_DIR / "stretch_robots" / "stretch3.xml",
    "xarm7_ranger": ASSETS_DIR / "xarm7_ranger" / "xarm7_ranger.xml",
    "franka": ASSETS_DIR / "franka_fr3" / "fr3.xml",
}

CATEGORY_ID_TO_POSITION = {
    "Fridge": [0, -0.95, 0.75],
    "Microwave": [0, -0.95, 0.75],
    "Dresser": [0, -0.95, 0.75],
    "Light_Switch": [0, -0.95, 0.75],
    "Toilet": [0, -0.95, 0.75],
    "Book": [0, -0.95, 0.75],
    "Shelving_Unit": [0, -0.95, 0.75],
    "Side_Table": [0, -0.95, 0.75],
}

CATEGORY_ID_TO_EULER = {
    "Fridge": [1.57, 0, 3.14],
    "Microwave": [1.57, 0, 3.14],
    "Dresser": [1.57, 0, 3.14],
}



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
        
        self.use_table: bool = False


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


def get_instances_per_category(category: str) -> List[Path]:
    base_path = str((ASSETS_DIR / "ThorAssets").resolve())
    print(f"Looking for models in this folder: {base_path}")
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
    robot_id: str, category_id: str, category_index: int = 0
) -> Tuple[mj.MjModel, mj.MjData]:
    global g_context
    root_spec = mj.MjSpec.from_file(str(EMPTY_SCENE_PATH.resolve()))

    # Set simulation properties similar to stretch scene
    root_spec.option.timestep = 0.002
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
                name="table", pos=[0, -1.0, diff_height / 4]
            )
            table_body.add_geom(
                type=mj.mjtGeom.mjGEOM_BOX,
                size=[0.6, 0.5, diff_height / 4],
                mass=1,
                material="wood",
            )

    model = root_spec.compile()
    data = mj.MjData(model)

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
                    if changed:
                        with lock:
                            joints_info[var_jnt_id.value].jnt_mpvar_stiffness.value = stiffness_value

                    # changed, armature_value = imgui.input_float("Armature", joints_info[var_jnt_id.value].jnt_mpvar_armature.value)
                    changed, armature_value = imgui.slider_float(
                        "Armature",
                        joints_info[var_jnt_id.value].jnt_mpvar_armature.value,
                        min_value=0.0,
                        max_value=1.0,
                    )
                    if changed:
                        with lock:
                            joints_info[var_jnt_id.value].jnt_mpvar_armature.value = armature_value

                    changed, damping_value = imgui.input_float("Damping", joints_info[var_jnt_id.value].jnt_mpvar_damping.value)
                    if changed:
                        with lock:
                            joints_info[var_jnt_id.value].jnt_mpvar_damping.value = damping_value

                    changed, frictionloss_value = imgui.input_float("FrictionLoss", joints_info[var_jnt_id.value].jnt_mpvar_frictionloss.value)
                    if changed:
                        with lock:
                            joints_info[var_jnt_id.value].jnt_mpvar_frictionloss.value = frictionloss_value
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

def run_launcher_interface(lock: LockType):
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

    while not glfw.window_should_close(window):
        glfw.poll_events()
        imgui_renderer.process_inputs()

        imgui.new_frame()
        # GUI creation goes here -----------------------------------------------

        # Combo-box used to select which joint to configure
        with imgui.begin("Model Selection"):
            pass

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

def load_valid_categories(assets_path: Path) -> List[str]:
    valid_categories = []

    # breakpoint()

    return valid_categories

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
        "--launcher",
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
    g_context.robot_id = args.robot
    g_context.category_id = args.category
    g_context.instances_per_category = get_instances_per_category(args.category)
    g_context.num_items_in_category = len(g_context.instances_per_category)
    g_context.use_table = not args.notable

    print(f"Loading model category: {g_context.category_id}")
    print(f"Index in category: {g_context.index_in_category}")
    print(f"Instances per category: {g_context.instances_per_category}")
    print(f"Loading model: {g_context.instances_per_category[g_context.index_in_category]}")

    # TODO(wilbert): might need to set simulation parameters to get a stable stretch robot simulation
    model, data = load_scene(args.robot, args.category)
    g_context.model = model
    g_context.data = data

    var_jnt_id = Value('i', -1)
    var_jnt_qpos_change = Value('i', 0)
    lock = Lock()

    joints_info: Dict[int, JointInfo] = load_joints(model, data)
    gui_process = None
    if not args.nogui:
        gui_process = Process(target=run_imgui_interface, args=(var_jnt_id, var_jnt_qpos_change, joints_info, lock))
        gui_process.start()

    categories_info: List[str] = load_valid_categories(ASSETS_DIR)
    launcher_process = None
    if args.launcher:
        launcher_process = Process(target=run_launcher_interface, args=(lock,))
        launcher_process.start()

    with mjviewer.launch_passive(model, data, key_callback=callback) as viewer:
        while viewer.is_running():
            if g_context.dirty_next_model or g_context.dirty_reload:
                # Load the next model into the simulation -----------------------------------
                g_context.dirty_next_model = False
                g_context.dirty_reload = False
                model, data = load_scene(
                    g_context.robot_id, g_context.category_id, g_context.index_in_category
                )
                sim = viewer._get_sim()
                if sim is not None:
                    sim.load(model, data, "Categories Visualizer")
                    viewer._user_scn = mj.MjvScene(model, sim.MAX_GEOM)

                g_context.model = model
                g_context.data = data

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
                # ---------------------------------------------------------------------------
            
            # Update the model parameters from the GUI -------------------------------------------------------
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
            # ------------------------------------------------------------------------------------------------

            mj.mj_step(model, data)
            viewer.sync()

    if gui_process is not None and gui_process.is_alive():
        gui_process.join()
    if launcher_process is not None and launcher_process.is_alive():
        launcher_process.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
