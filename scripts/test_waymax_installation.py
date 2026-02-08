#!/usr/bin/env python3
"""
Waymax Installation Verification Script

Tests the core components needed for SQP-based Rulebook MPC development:
1. JAX with GPU support
2. Waymax simulator
3. Waymo Open Dataset tools
4. Optimization libraries (CasADi, CVXPY, OSQP)
"""

import sys
from typing import Tuple


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_result(name: str, success: bool, details: str = "") -> None:
    """Print a test result with status indicator."""
    status = "[PASS]" if success else "[FAIL]"
    print(f"  {status} {name}")
    if details:
        print(f"         {details}")


def test_jax() -> Tuple[bool, str]:
    """Test JAX installation and GPU availability."""
    try:
        import jax
        import jax.numpy as jnp

        # Check version
        version = jax.__version__

        # Check available devices
        devices = jax.devices()
        device_types = [str(d.platform) for d in devices]

        # Test basic computation
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x**2)
        assert float(y) == 14.0

        gpu_available = "gpu" in device_types or "cuda" in device_types
        device_info = f"v{version}, devices: {device_types}"

        if gpu_available:
            return True, f"{device_info} (GPU enabled)"
        else:
            return True, f"{device_info} (CPU only)"

    except Exception as e:
        return False, str(e)


def test_waymax() -> Tuple[bool, str]:
    """Test Waymax installation."""
    try:
        import waymax
        from waymax import config as waymax_config

        version = getattr(waymax, "__version__", "unknown")

        # Test that we can access config
        _ = waymax_config.WaymaxConfig

        return True, f"v{version}"

    except Exception as e:
        return False, str(e)


def test_tensorflow() -> Tuple[bool, str]:
    """Test TensorFlow (required for Waymo Open Dataset)."""
    try:
        import tensorflow as tf

        version = tf.__version__

        # Check GPU availability
        gpus = tf.config.list_physical_devices("GPU")
        gpu_info = f"{len(gpus)} GPU(s)" if gpus else "CPU only"

        return True, f"v{version}, {gpu_info}"

    except Exception as e:
        return False, str(e)


def test_casadi() -> Tuple[bool, str]:
    """Test CasADi for symbolic optimization."""
    try:
        import casadi as ca

        version = ca.__version__

        # Test basic symbolic computation
        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        f = x**2 + y**2
        grad_f = ca.gradient(f, ca.vertcat(x, y))

        # Test QP solver availability
        qp = {"x": ca.vertcat(x, y), "f": f}
        solver = ca.qpsol("qp", "osqp", qp)

        return True, f"v{version}, QP solvers available"

    except Exception as e:
        return False, str(e)


def test_cvxpy() -> Tuple[bool, str]:
    """Test CVXPY for convex optimization."""
    try:
        import cvxpy as cp

        version = cp.__version__

        # Test basic QP
        x = cp.Variable(2)
        objective = cp.Minimize(cp.sum_squares(x))
        constraints = [x >= -1, x <= 1]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        # Check available solvers
        solvers = cp.installed_solvers()
        solver_info = f"{len(solvers)} solvers: {solvers[:5]}..."

        return True, f"v{version}, {solver_info}"

    except Exception as e:
        return False, str(e)


def test_osqp() -> Tuple[bool, str]:
    """Test OSQP solver."""
    try:
        import numpy as np
        import osqp
        from scipy import sparse

        version = osqp.__version__

        # Test basic QP: min 0.5 x'Px + q'x s.t. Ax <= u
        P = sparse.csc_matrix([[4.0, 1.0], [1.0, 2.0]])
        q = np.array([1.0, 1.0])
        A = sparse.csc_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        l = np.array([1.0, 0.0, 0.0])
        u = np.array([1.0, 0.7, 0.7])

        solver = osqp.OSQP()
        solver.setup(P, q, A, l, u, verbose=False)
        results = solver.solve()

        if results.info.status == "solved":
            return True, f"v{version}, QP solved successfully"
        else:
            return False, f"v{version}, QP status: {results.info.status}"

    except Exception as e:
        return False, str(e)


def test_scipy_optimize() -> Tuple[bool, str]:
    """Test SciPy optimization (for SQP)."""
    try:
        import numpy as np
        from scipy import __version__ as scipy_version
        from scipy.optimize import minimize

        # Test SQP-style optimization
        def objective(x):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2

        def constraint(x):
            return x[0] - 2 * x[1] + 2

        x0 = [0, 0]
        cons = {"type": "ineq", "fun": constraint}
        result = minimize(objective, x0, method="SLSQP", constraints=cons)

        if result.success:
            return True, f"v{scipy_version}, SLSQP working"
        else:
            return False, f"v{scipy_version}, optimization failed"

    except Exception as e:
        return False, str(e)


def test_control_library() -> Tuple[bool, str]:
    """Test python-control library."""
    try:
        import control

        version = control.__version__

        # Test basic state-space system
        A = [[0, 1], [-2, -3]]
        B = [[0], [1]]
        C = [[1, 0]]
        D = [[0]]
        sys = control.ss(A, B, C, D)

        # Test LQR
        Q = [[1, 0], [0, 1]]
        R = [[1]]
        K, S, E = control.lqr(sys, Q, R)

        return True, f"v{version}, LQR working"

    except Exception as e:
        return False, str(e)


def test_dompc() -> Tuple[bool, str]:
    """Test do-mpc library."""
    try:
        import do_mpc

        version = do_mpc.__version__

        return True, f"v{version}"

    except Exception as e:
        return False, str(e)


def test_visualization() -> Tuple[bool, str]:
    """Test visualization libraries."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np

        # Test basic plot creation
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        ax.plot(x, np.sin(x))
        plt.close(fig)

        import plotly

        return True, f"matplotlib + plotly available"

    except Exception as e:
        return False, str(e)


def main() -> int:
    """Run all installation tests."""
    print_header("Waymax-SQP-MPC Installation Verification")

    all_passed = True
    results = []

    # Core dependencies
    print_header("Core Dependencies")

    success, details = test_jax()
    print_result("JAX", success, details)
    results.append(("JAX", success))
    all_passed &= success

    success, details = test_tensorflow()
    print_result("TensorFlow", success, details)
    results.append(("TensorFlow", success))

    success, details = test_waymax()
    print_result("Waymax", success, details)
    results.append(("Waymax", success))
    all_passed &= success

    # Optimization libraries
    print_header("Optimization Libraries")

    success, details = test_casadi()
    print_result("CasADi", success, details)
    results.append(("CasADi", success))
    all_passed &= success

    success, details = test_cvxpy()
    print_result("CVXPY", success, details)
    results.append(("CVXPY", success))
    all_passed &= success

    success, details = test_osqp()
    print_result("OSQP", success, details)
    results.append(("OSQP", success))
    all_passed &= success

    success, details = test_scipy_optimize()
    print_result("SciPy SLSQP", success, details)
    results.append(("SciPy SLSQP", success))
    all_passed &= success

    # Control libraries
    print_header("Control Libraries")

    success, details = test_control_library()
    print_result("python-control", success, details)
    results.append(("python-control", success))

    success, details = test_dompc()
    print_result("do-mpc", success, details)
    results.append(("do-mpc", success))

    # Visualization
    print_header("Visualization")

    success, details = test_visualization()
    print_result("Visualization", success, details)
    results.append(("Visualization", success))

    # Summary
    print_header("Summary")
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\n  Tests passed: {passed}/{total}")

    if all_passed:
        print("\n  All critical components installed successfully!")
        print("  Ready for SQP-based Rulebook MPC development with Waymax.")
        return 0
    else:
        print("\n  Some critical components failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
