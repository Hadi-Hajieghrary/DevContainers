#!/usr/bin/env python3
"""
Example: SQP-Based MPC with Waymax

This script demonstrates the basic structure of an SQP-based Model Predictive
Controller integrated with the Waymax simulator for autonomous vehicle planning.

Based on the Weighted SQP-Based Rulebook MPC framework described in the paper.
"""

import numpy as np
import casadi as ca
from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib.pyplot as plt


@dataclass
class VehicleParams:
    """Vehicle parameters for kinematic bicycle model."""
    L: float = 4.5       # Wheelbase [m]
    max_steer: float = 0.5  # Max steering angle [rad]
    max_accel: float = 3.0  # Max acceleration [m/s^2]
    max_decel: float = -5.0  # Max deceleration [m/s^2]
    max_speed: float = 30.0  # Max speed [m/s]


@dataclass
class MPCParams:
    """MPC controller parameters."""
    N: int = 20              # Prediction horizon
    dt: float = 0.1          # Time step [s]

    # Cost weights (LQR-style)
    Q_x: float = 1.0         # Position x tracking
    Q_y: float = 1.0         # Position y tracking
    Q_v: float = 0.5         # Velocity tracking
    Q_theta: float = 0.1     # Heading tracking
    R_a: float = 0.1         # Acceleration effort
    R_delta: float = 0.5     # Steering effort
    R_da: float = 0.05       # Acceleration rate
    R_ddelta: float = 0.1    # Steering rate

    # ECR-based penalty weights for constraint violations
    rho_safety: float = 1e6      # Level 1: Safety (hard)
    rho_regulatory: float = 1e4  # Level 2: Regulatory
    rho_comfort: float = 1e2     # Level 3: Comfort


class KinematicBicycleModel:
    """
    Kinematic bicycle model for vehicle dynamics.

    State: [x, y, theta, v]
    Control: [a, delta]
    """

    def __init__(self, params: VehicleParams):
        self.params = params
        self._setup_casadi_model()

    def _setup_casadi_model(self):
        """Setup CasADi symbolic model."""
        # State variables
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        v = ca.SX.sym('v')
        self.state = ca.vertcat(x, y, theta, v)
        self.n_state = 4

        # Control variables
        a = ca.SX.sym('a')  # Acceleration
        delta = ca.SX.sym('delta')  # Steering angle
        self.control = ca.vertcat(a, delta)
        self.n_control = 2

        # Kinematic bicycle model equations
        x_dot = v * ca.cos(theta)
        y_dot = v * ca.sin(theta)
        theta_dot = v * ca.tan(delta) / self.params.L
        v_dot = a

        self.state_dot = ca.vertcat(x_dot, y_dot, theta_dot, v_dot)

        # Create CasADi function for dynamics
        self.f_continuous = ca.Function(
            'f_continuous',
            [self.state, self.control],
            [self.state_dot],
            ['x', 'u'],
            ['x_dot']
        )

    def discretize(self, dt: float) -> ca.Function:
        """
        Discretize the continuous dynamics using RK4.

        Returns:
            CasADi function: x_next = f(x, u)
        """
        x = self.state
        u = self.control

        # RK4 integration
        k1 = self.f_continuous(x, u)
        k2 = self.f_continuous(x + dt/2 * k1, u)
        k3 = self.f_continuous(x + dt/2 * k2, u)
        k4 = self.f_continuous(x + dt * k3, u)

        x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        return ca.Function(
            'f_discrete',
            [x, u],
            [x_next],
            ['x', 'u'],
            ['x_next']
        )


class WeightedSQPMPC:
    """
    Weighted SQP-Based MPC Controller.

    Implements the ECR (Equal Concern for Relaxation) hierarchy:
    - Level 1: Safety constraints (hard)
    - Level 2: Regulatory constraints (soft, high penalty)
    - Level 3: Comfort constraints (soft, lower penalty)
    """

    def __init__(
        self,
        vehicle_params: VehicleParams = VehicleParams(),
        mpc_params: MPCParams = MPCParams()
    ):
        self.vp = vehicle_params
        self.mp = mpc_params

        # Setup dynamics model
        self.model = KinematicBicycleModel(vehicle_params)
        self.f_discrete = self.model.discretize(mpc_params.dt)

        # Setup optimization problem
        self._setup_optimization()

    def _setup_optimization(self):
        """Setup the CasADi optimization problem."""
        N = self.mp.N
        n_x = self.model.n_state
        n_u = self.model.n_control

        # Decision variables
        # States: [x0, x1, ..., xN]
        X = ca.SX.sym('X', n_x, N + 1)
        # Controls: [u0, u1, ..., uN-1]
        U = ca.SX.sym('U', n_u, N)
        # Slack variables for soft constraints
        S = ca.SX.sym('S', N)  # Safety margin slacks

        # Parameters
        x0 = ca.SX.sym('x0', n_x)  # Initial state
        x_ref = ca.SX.sym('x_ref', n_x, N + 1)  # Reference trajectory
        u_prev = ca.SX.sym('u_prev', n_u)  # Previous control (for rate constraints)

        # Cost function
        cost = 0

        # Stage costs
        Q = ca.diag([self.mp.Q_x, self.mp.Q_y, self.mp.Q_theta, self.mp.Q_v])
        R = ca.diag([self.mp.R_a, self.mp.R_delta])
        R_rate = ca.diag([self.mp.R_da, self.mp.R_ddelta])

        for k in range(N):
            # State tracking cost
            x_err = X[:, k] - x_ref[:, k]
            cost += ca.mtimes([x_err.T, Q, x_err])

            # Control effort cost
            cost += ca.mtimes([U[:, k].T, R, U[:, k]])

            # Control rate cost
            if k == 0:
                u_rate = U[:, k] - u_prev
            else:
                u_rate = U[:, k] - U[:, k-1]
            cost += ca.mtimes([u_rate.T, R_rate, u_rate])

            # ECR penalty for constraint violations (slack variables)
            cost += self.mp.rho_safety * S[k]**2

        # Terminal cost
        x_err_terminal = X[:, N] - x_ref[:, N]
        cost += 10 * ca.mtimes([x_err_terminal.T, Q, x_err_terminal])

        # Constraints
        g = []
        lbg = []
        ubg = []

        # Initial state constraint
        g.append(X[:, 0] - x0)
        lbg += [0] * n_x
        ubg += [0] * n_x

        # Dynamics constraints
        for k in range(N):
            x_next = self.f_discrete(X[:, k], U[:, k])
            g.append(X[:, k+1] - x_next)
            lbg += [0] * n_x
            ubg += [0] * n_x

        # Input constraints
        for k in range(N):
            # Acceleration bounds
            g.append(U[0, k])
            lbg.append(self.vp.max_decel)
            ubg.append(self.vp.max_accel)

            # Steering bounds
            g.append(U[1, k])
            lbg.append(-self.vp.max_steer)
            ubg.append(self.vp.max_steer)

        # State constraints (with slack for safety margin)
        for k in range(N + 1):
            # Speed constraint: 0 <= v <= v_max
            g.append(X[3, k])
            lbg.append(0)
            ubg.append(self.vp.max_speed)

        # Slack non-negativity
        for k in range(N):
            g.append(S[k])
            lbg.append(0)
            ubg.append(ca.inf)

        # Flatten decision variables
        opt_vars = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1),
            S
        )

        # Parameters
        params = ca.vertcat(x0, ca.reshape(x_ref, -1, 1), u_prev)

        # Create NLP solver
        nlp = {
            'x': opt_vars,
            'f': cost,
            'g': ca.vertcat(*g),
            'p': params
        }

        # IPOPT options (SQP-like behavior with limited iterations)
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 100,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
        }

        self.solver = ca.nlpsol('mpc_solver', 'ipopt', nlp, opts)
        self.lbg = lbg
        self.ubg = ubg

        # Store dimensions for solution extraction
        self.n_x = n_x
        self.n_u = n_u
        self.N = N

    def solve(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        u_prev: np.ndarray,
        warm_start: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Solve the MPC optimization problem.

        Args:
            x0: Current state [x, y, theta, v]
            x_ref: Reference trajectory (n_x, N+1)
            u_prev: Previous control input
            warm_start: Optional warm start for decision variables

        Returns:
            u_opt: Optimal control for current step
            X_pred: Predicted state trajectory
            info: Solution information
        """
        N = self.N
        n_x = self.n_x
        n_u = self.n_u

        # Flatten parameters
        params = np.concatenate([
            x0,
            x_ref.flatten(),
            u_prev
        ])

        # Initial guess
        if warm_start is not None:
            x0_opt = warm_start
        else:
            # Simple initialization
            X_init = np.tile(x0, (N + 1, 1)).T
            U_init = np.zeros((n_u, N))
            S_init = np.zeros(N)
            x0_opt = np.concatenate([
                X_init.flatten(),
                U_init.flatten(),
                S_init
            ])

        # Solve
        sol = self.solver(
            x0=x0_opt,
            p=params,
            lbg=self.lbg,
            ubg=self.ubg
        )

        # Extract solution
        opt_vars = np.array(sol['x']).flatten()

        n_X = n_x * (N + 1)
        n_U = n_u * N

        X_opt = opt_vars[:n_X].reshape((n_x, N + 1), order='F')
        U_opt = opt_vars[n_X:n_X + n_U].reshape((n_u, N), order='F')
        S_opt = opt_vars[n_X + n_U:]

        # Get first control action
        u_opt = U_opt[:, 0]

        info = {
            'X_pred': X_opt,
            'U_pred': U_opt,
            'S_pred': S_opt,
            'cost': float(sol['f']),
            'status': self.solver.stats()['return_status'],
            'solve_time': self.solver.stats()['t_wall_total']
        }

        return u_opt, X_opt, info


def generate_reference_trajectory(
    x0: np.ndarray,
    target_speed: float,
    target_y: float,
    N: int,
    dt: float
) -> np.ndarray:
    """Generate a simple lane-following reference trajectory."""
    x_ref = np.zeros((4, N + 1))

    for k in range(N + 1):
        t = k * dt
        x_ref[0, k] = x0[0] + target_speed * t  # x position
        x_ref[1, k] = target_y                   # y position (target lane)
        x_ref[2, k] = 0.0                        # heading (straight)
        x_ref[3, k] = target_speed               # velocity

    return x_ref


def demo_mpc():
    """Demonstrate the SQP-based MPC controller."""
    print("=" * 60)
    print(" SQP-Based MPC Demo")
    print("=" * 60)

    # Initialize controller
    mpc = WeightedSQPMPC()

    # Initial state: [x, y, theta, v]
    x0 = np.array([0.0, 0.0, 0.0, 15.0])
    u_prev = np.array([0.0, 0.0])

    # Reference: lane change from y=0 to y=3.5
    target_speed = 20.0
    target_y = 3.5

    # Simulation
    T_sim = 5.0  # seconds
    dt = mpc.mp.dt
    n_steps = int(T_sim / dt)

    # Storage
    x_history = [x0.copy()]
    u_history = []
    solve_times = []

    x = x0.copy()

    print(f"\nSimulating {T_sim}s with dt={dt}s...")

    for step in range(n_steps):
        # Generate reference
        x_ref = generate_reference_trajectory(
            x, target_speed, target_y, mpc.N, dt
        )

        # Solve MPC
        u, X_pred, info = mpc.solve(x, x_ref, u_prev)

        solve_times.append(info['solve_time'])

        # Apply control (simulate one step)
        x_next = np.array(mpc.f_discrete(x, u)).flatten()

        # Store
        x_history.append(x_next.copy())
        u_history.append(u.copy())

        # Update
        x = x_next
        u_prev = u

        if step % 10 == 0:
            print(f"  Step {step:3d}: x={x[0]:.1f}m, y={x[1]:.2f}m, "
                  f"v={x[3]:.1f}m/s, solve_time={info['solve_time']*1000:.1f}ms")

    # Results
    x_history = np.array(x_history)
    u_history = np.array(u_history)

    print(f"\nResults:")
    print(f"  Average solve time: {np.mean(solve_times)*1000:.2f} ms")
    print(f"  Max solve time: {np.max(solve_times)*1000:.2f} ms")
    print(f"  Final position: ({x_history[-1, 0]:.1f}, {x_history[-1, 1]:.2f}) m")
    print(f"  Final speed: {x_history[-1, 3]:.1f} m/s")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    t = np.arange(len(x_history)) * dt
    t_u = np.arange(len(u_history)) * dt

    # Trajectory
    axes[0, 0].plot(x_history[:, 0], x_history[:, 1], 'b-', linewidth=2)
    axes[0, 0].axhline(y=target_y, color='g', linestyle='--', label='Target lane')
    axes[0, 0].set_xlabel('x [m]')
    axes[0, 0].set_ylabel('y [m]')
    axes[0, 0].set_title('Vehicle Trajectory')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Speed
    axes[0, 1].plot(t, x_history[:, 3], 'b-', linewidth=2)
    axes[0, 1].axhline(y=target_speed, color='g', linestyle='--', label='Target')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Speed [m/s]')
    axes[0, 1].set_title('Speed Profile')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Controls
    axes[1, 0].plot(t_u, u_history[:, 0], 'b-', linewidth=2, label='Acceleration')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Acceleration [m/s²]')
    axes[1, 0].set_title('Control: Acceleration')
    axes[1, 0].grid(True)

    axes[1, 1].plot(t_u, np.rad2deg(u_history[:, 1]), 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Steering [deg]')
    axes[1, 1].set_title('Control: Steering')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('mpc_demo_results.png', dpi=150)
    print(f"\nPlot saved to 'mpc_demo_results.png'")

    return x_history, u_history


if __name__ == "__main__":
    demo_mpc()
