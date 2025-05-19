from pathlib import Path
from multiprocessing import Process, Value
from time import sleep

import cv2
import numpy as np
import mujoco as mj
import mujoco.viewer as mjviewer

MAIN_WINDOW_NAME = "MuJoCo GUI controls"

CURRENT_DIR = Path(__file__).parent
ASSETS_DIR = CURRENT_DIR.parent / "assets"
EMPTY_SCENE_PATH = ASSETS_DIR / "empty_scene.xml"


def main() -> int:
    mp_gravity = Value('d', -9.81)
    # Create a separate UI thread to run the OpenCV GUI
    ui_process = Process(target=run_opencv_gui, args=(mp_gravity,))
    ui_process.start()

    # Run the MuJoCo simulation in the main thread / process
    run_mujoco(mp_gravity)

    ui_process.join()

    return 0

# This should run in the main thread / process
def run_mujoco(gravity) -> None:
    model = mj.MjModel.from_xml_path(str(EMPTY_SCENE_PATH.resolve()))
    data = mj.MjData(model)

    with mjviewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Update parameters from the GUI            
            # ...
            model.opt.gravity = [0, 0, gravity.value]

            mj.mj_step(model, data)
            sleep(0.001)
            viewer.sync()



# This should run in a separate UI thread
def run_opencv_gui(gravity) -> None:
    create_opencv_gui(gravity)

    img = np.zeros((100, 500, 3), dtype=np.uint8)

    while True:
        img.fill(0)

        cv2.imshow(MAIN_WINDOW_NAME, img)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def update_gravity(gravity, value: float) -> None:
    gravity.value = value

def create_opencv_gui(gravity) -> None:
    cv2.namedWindow(MAIN_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(MAIN_WINDOW_NAME, 500, 300)

    cv2.createTrackbar('Gravity', MAIN_WINDOW_NAME, 981, 2000, lambda x: update_gravity(gravity, -x / 100))





if __name__ == "__main__":
    raise SystemExit(main())



