import time
import mujoco
import mujoco.viewer
import numpy as np
import cv2
import argparse
import sys
import glfw

XML = r"""
<mujoco model="gui_test">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.08" rgba="0.2 0.4 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""

def print_report():
  print("--- MuJoCo Installation Check ---")
  print(f"MuJoCo version: {mujoco.__version__}")
  print(f"NumPy version: {np.__version__}")
  print(f"OpenCV version: {cv2.__version__}")

  # Test GLFW
  if glfw.init():
    print("GLFW: Available (GUI support enabled)")
    glfw.terminate()
  else:
    print("GLFW: Not available (Headless mode only)")

  print()

def main(headless=False):
  model = mujoco.MjModel.from_xml_string(XML)
  data = mujoco.MjData(model)

  if headless:
    # Use offscreen renderer
    renderer = mujoco.Renderer(model, width=640, height=480)

    # Set camera
    camera = mujoco.MjvCamera()
    camera.distance = 3.0
    camera.azimuth = 90
    camera.elevation = -20

    # Step simulation a bit
    for _ in range(100):
      mujoco.mj_step(model, data)

    # Render
    renderer.update_scene(data, camera=camera)
    img = renderer.render()

    # Save image
    cv2.imwrite('mujoco_render.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"✓ Offscreen Rendering: Success (640x480)")
    print(f"✓ Image saved: mujoco_render.png")
    print(f"✓ Simulation steps: 100 (Ball position: {data.qpos[2]:.4f})")

    renderer.close()
  else:
    # Interactive GUI
    print("✓ Launching interactive viewer (Close window to exit)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
      viewer.cam.distance = 3.0
      viewer.cam.azimuth = 90
      viewer.cam.elevation = -20

      step_count = 0
      while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
        step_count += 1

      print(f"✓ Interactive GUI: Success ({step_count} simulation steps)")

if __name__ == "__main__":
  print(f"Python Interpreter: {sys.executable}\n")
  print_report()

  parser = argparse.ArgumentParser(description="MuJoCo GUI Test")
  parser.add_argument("--headless", action="store_true", help="Run in headless mode and save image")
  args = parser.parse_args()
  main(headless=args.headless)

  print("\nMuJoCo installation test complete.")
