from pathlib import Path
from multiprocessing import Process, Value
from time import sleep

import glfw
import imgui
import mujoco as mj
import mujoco.viewer as mjviewer
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer

CURRENT_DIR = Path(__file__).parent
ASSETS_DIR = CURRENT_DIR.parent / "assets"
EMPTY_SCENE_PATH = ASSETS_DIR / "empty_scene.xml"


def main() -> int:
    var_gravity = Value('d', -9.81)

    gui_process = Process(target=run_imgui, args=(var_gravity,))
    gui_process.start()

    run_mujoco(var_gravity)

    gui_process.join()

    return 0


def run_imgui(var_gravity) -> None:
    if not glfw.init():
        print("Couldn't initialize GLFW")
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    window = glfw.create_window(1200, 900, "MuJoCo + ImGui Integration", None, None)

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
        imgui.begin("Simulation properties", True)

        changed, value = imgui.slider_float(
            "gravity", var_gravity.value, min_value=-20.0, max_value=20.0
        )
        if changed:
            var_gravity.value = value

        imgui.end()
        # ----------------------------------------------------------------------

        imgui.render()

        gl.glClearColor(0.0, 0.0, 0.2, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui_renderer.render(imgui.get_draw_data())

        glfw.swap_buffers(window)

    imgui_renderer.shutdown()
    glfw.terminate()


def run_mujoco(var_gravity) -> None:
    model = mj.MjModel.from_xml_path(str(EMPTY_SCENE_PATH.resolve()))
    data = mj.MjData(model)

    with mjviewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Update parameters from the GUI            
            model.opt.gravity = [0, 0, var_gravity.value]

            step_start = data.time
            while (data.time - step_start) < 1. / 60.:
                mj.mj_step(model, data)
                sleep(0.001)
            viewer.sync()

if __name__ == "__main__":
    raise SystemExit(main())



