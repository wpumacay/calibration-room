from pathlib import Path

import glfw
import mujoco as mj

CURRENT_DIR = Path(__file__).parent
ASSETS_DIR = CURRENT_DIR.parent / "assets"
EMPTY_SCENE_PATH = ASSETS_DIR / "empty_scene.xml"

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900
WINDOW_NAME = "Sample App"


def main() -> int:
    if not glfw.init():
        print("Failed GLFW initialization")
        return -1

    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_NAME, None, None)
    if not window:
        print("Failed creating a GLFW window")
        glfw.terminate()
        return -1

    glfw.make_context_current(window)

    # MuJoCo stuff
    model = mj.MjModel.from_xml_path(str(EMPTY_SCENE_PATH.resolve()))
    data = mj.MjData(model)

    viewport = mj.MjrRect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
    scene = mj.MjvScene(model, 1000)
    camera = mj.MjvCamera()
    vopt = mj.MjvOption()
    pert = mj.MjvPerturb()

    ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150) # type: ignore

    mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW.value, ctx)

    while not glfw.window_should_close(window):
        mj.mj_step(model, data)

        # Do some rendering stuff

        viewport.width, viewport.height = glfw.get_framebuffer_size(window)

        mj.mjv_updateScene(
            model,
            data,
            vopt,
            mj.MjvPerturb(),
            camera,
            mj.mjtCatBit.mjCAT_ALL.value,
            scene
        )

        mj.mjr_render(viewport, scene, ctx)

        glfw.swap_buffers(window)

        glfw.poll_events()

    glfw.destroy_window(window)
    glfw.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



