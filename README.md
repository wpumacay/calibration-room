# calibration-room
A set of tools to handle tuning of joint properties in MuJoCo

## Generating executable on Windows

You need to have a Windows machine, and have Python setup. Then you should just
install `pyinstaller` from PyPI.

```bash
pip install pyinstaller
```

Once you have `pyinstaller`, just run the following from the root of the repo:

```bash
pyinstaller --hidden-import numpy.core.multiarray main-calibration.py
```

This will generate a `dist` folder, which contains a folder with the name of
the application (i.e. `main-calibration`) and inside an executable with the
same name (i.e. `main-calibration.exe`).
