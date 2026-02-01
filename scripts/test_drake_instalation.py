import sys
import numpy as np

def print_report():
    print("--- Drake Installation Check ---")

    try:
        import pydrake
        # Drake doesn't have __version__, use getDrakePath or package info
        try:
            from importlib.metadata import version
            drake_version = version("drake")
            print(f"Drake version: {drake_version}")
        except Exception:
            print("Drake: Imported successfully (version info not available)")
    except ImportError as e:
        print(f"✗ Drake import failed: {e}")
        return False

    print(f"NumPy version: {np.__version__}")

    try:
        import scipy
        print(f"SciPy version: {scipy.__version__}")
    except ImportError:
        print("SciPy: Not available")

    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        print("OpenCV: Not available")

    try:
        import meshcat
        print("Meshcat: Available (web-based 3D visualization)")
    except ImportError:
        print("Meshcat: Not available")

    print()
    return True


def test_drake_simulation():
    """Test basic Drake simulation capabilities."""
    from pydrake.systems.framework import DiagramBuilder
    from pydrake.systems.analysis import Simulator
    from pydrake.systems.primitives import ConstantVectorSource, LogVectorOutput

    print("--- Drake Simulation Test ---")

    # Create a simple system: constant source -> logger
    builder = DiagramBuilder()

    # Add a constant source that outputs [1.0, 2.0, 3.0]
    source = builder.AddSystem(ConstantVectorSource([1.0, 2.0, 3.0]))
    source.set_name("source")

    # Add a logger to record the output
    logger = LogVectorOutput(source.get_output_port(), builder)

    # Build the diagram
    diagram = builder.Build()

    # Create simulator
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(0)

    # Run simulation for 0.1 seconds
    simulator.AdvanceTo(0.1)

    # Get logged data
    log = logger.FindLog(simulator.get_context())
    final_value = log.data()[:, -1]

    print(f"✓ DiagramBuilder: Success")
    print(f"✓ Simulator: Success")
    print(f"✓ Simulation output: {final_value}")

    return True


def test_drake_math():
    """Test Drake math utilities."""
    from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw

    print("\n--- Drake Math Test ---")

    # Test rotation matrix
    rpy = RollPitchYaw(0.1, 0.2, 0.3)
    R = RotationMatrix(rpy)
    print(f"✓ RollPitchYaw: {rpy.vector()}")
    print(f"✓ RotationMatrix created successfully")

    # Test rigid transform
    X = RigidTransform(R, [1.0, 2.0, 3.0])
    print(f"✓ RigidTransform translation: {X.translation()}")

    return True


def test_drake_multibody():
    """Test Drake multibody plant basics."""
    from pydrake.multibody.plant import MultibodyPlant
    from pydrake.multibody.parsing import Parser

    print("\n--- Drake MultibodyPlant Test ---")

    # Create a simple plant
    plant = MultibodyPlant(time_step=0.001)

    # Add a simple box
    from pydrake.geometry import Box
    from pydrake.multibody.tree import SpatialInertia, UnitInertia

    # Create spatial inertia for a 1kg box
    M = SpatialInertia(
        mass=1.0,
        p_PScm_E=[0, 0, 0],
        G_SP_E=UnitInertia.SolidBox(0.1, 0.1, 0.1)
    )

    # Add the body
    box_body = plant.AddRigidBody("box", M)

    plant.Finalize()

    print(f"✓ MultibodyPlant created")
    print(f"✓ Number of bodies: {plant.num_bodies()}")
    print(f"✓ Time step: {plant.time_step()}s")

    return True


def test_meshcat_visualization():
    """Test Meshcat visualization availability."""
    print("\n--- Meshcat Visualization Test ---")

    try:
        from pydrake.geometry import StartMeshcat

        # Start meshcat (this creates a server)
        meshcat = StartMeshcat()
        print(f"✓ Meshcat server started")
        print(f"  Open visualization at: {meshcat.web_url()}")

        # Add a simple shape
        from pydrake.geometry import Sphere, Rgba
        meshcat.SetObject("test_sphere", Sphere(0.1), Rgba(0.2, 0.4, 0.9, 1.0))
        print(f"✓ Added test sphere to visualization")

        return True
    except Exception as e:
        print(f"! Meshcat visualization: {e}")
        return False


if __name__ == "__main__":
    print(f"Python Interpreter: {sys.executable}\n")

    if not print_report():
        sys.exit(1)

    try:
        test_drake_simulation()
        test_drake_math()
        test_drake_multibody()
        test_meshcat_visualization()
        print("\n✓ All Drake tests passed!")
    except Exception as e:
        print(f"\n✗ Drake test failed: {e}")
        sys.exit(1)

    print("\nDrake installation test complete.")
