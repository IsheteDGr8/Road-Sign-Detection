# Road Sign Detection

Detects US-style road signs from still images and dashcam video using color masks and simple shape rules. Built for CSS 487 with **OpenCV 4 only** — no extra libraries, no bundled executables, no network calls.

**Authors:** Ishaan, Manish

## What you need

- Visual Studio 2022 (Desktop development with C++)
- OpenCV 4 (same install used for other course projects)
- Test images and videos are already in `data/`

The C++ program does not require Python. Python scripts under `scripts/` are optional helpers for batch-testing and saving result images.

## One-time setup

1. Open `build/SignDetector.sln` in Visual Studio (generate with CMake if `build/` is missing — see below).
2. If CMake cannot find OpenCV, edit `OpenCV_DIR` in `CMakeLists.txt` to match your machine (e.g. `C:/opencv/build`).
3. Build **Debug | x64** once (F7).

### Generate the solution with CMake (if needed)

```bat
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
```

## How to run (Visual Studio — recommended)

1. Open `src/main.cpp`.
2. Set working directory to the **project root** (folder that contains `data/`):
   - Project → Properties → Debugging → Working Directory → `$(ProjectDir)..\..`
   - Or in `launch.vs.json` / debug settings, cwd = project root.
3. **F7** to build, **Shift+F5** (or Ctrl+F5) to run.

The program looks for `data/` first, then `build/data/` if the root folder is missing.

## How to run (batch file)

From the project root:

```bat
scripts\run_demo.bat
```

Builds are expected at `build\Debug\SignDetector.exe`. The batch file runs from the project root so paths resolve correctly.

## What `main` does

Runs in order; press any key to advance through still images, **ESC** to stop video playback early.

| Step | Folder / file | Detects |
|------|----------------|---------|
| 1 | `data/construction signs/` | Orange diamonds |
| 2 | `data/guide signs/` | Green rectangles |
| 3 | `data/service signs/` | Blue rectangles |
| 4 | `data/warning signs/` | Yellow diamonds |
| 5 | `data/regulatory signs/` | Stop, do not enter, speed limit (subset of images) |
| 6 | `data/dashcam.mp4` | Live detection on video |

## Optional: Python batch tests

Saves annotated images to `build/test_out/` (mirrors `data/` layout). Requires Python 3 and `opencv-python`:

```bat
scripts\run_tests.bat
```

Individual tests: `python scripts\test_construction.py`, `test_warning.py`, `test_service.py`, `test_regulatory.py`, etc.

## Project layout

```
include/          ColorSegmenter.h, ShapeAnalyzer.h
src/              main.cpp, ColorSegmenter.cpp, ShapeAnalyzer.cpp
data/             Test images and dashcam .mp4 files (see data/SOURCES.txt)
scripts/          Optional Python tests and run_demo.bat / run_tests.bat
build/            Visual Studio solution and build output (not required in zip if rebuilt)
```

## Image sources

See `data/SOURCES.txt` for Wikimedia Commons URLs and licensing notes.

## Notes for grading

- **Dependencies:** OpenCV 4 only for the main executable.
- **No external processes** (no chess engines, REST clients, or other third-party runtimes).
- Dashcam quality affects results; static test images are the primary evaluation set.
- Value added beyond raw OpenCV: tuned HSV masks, shape/heuristic classifiers per MUTCD sign category, speed-limit bbox refinement, and simple digit OCR (template match + hole counting).
