import numpy as np
import cvxpy as cp

class MPCController:
    def __init__(self, horizon=20, dt=0.02):  # Note: Default dt=0.02 (50Hz)
        self.N = horizon
        self.dt = dt

        # --- Robot Constraints ---
        self.v_max = 0.26
        self.v_min = 0 # Allow reversing
        self.w_max = 1.82
        
        # Acceleration limits (Now these are the INPUT limits)
        self.acc_v_max = 0.5   # m/s^2
        self.acc_w_max = 3.0   # rad/s^2

        # --- Tuning Weights (Delta Formulation) ---
        
        # Q: State Penalty [x, y, theta, v, w]
        # We now track velocity errors directly in the state cost
        self.Q = np.diag([150.0, 150.0, 1.0, 5.0, 8.5])

        # R: Input Penalty [delta_v, delta_w]
        # This penalizes CHANGE in velocity (smoothness)
        self.R = np.diag([1, 0.01])
        
        # Terminal weight
        self.terminal_weight = 100.0

    def get_augmented_model(self, x_lin, u_lin):
        """
        Returns linearized model for state [x, y, theta, v, w]
        and input [delta_v, delta_w]
        """
        theta = x_lin[2]
        v = x_lin[3] # Velocity is now part of the state
        
        # Singularity Fix: Ghost velocity
        if abs(v) < 1e-3:
            v = 1e-3

        # Jacobian w.r.t State (5x5)
        # x_next = x + v*cos(th)*dt
        # v_next = v + delta_v (so v depends on v with factor 1)
        A = np.eye(5)
        
        # Partial derivatives for Position w.r.t Theta
        A[0, 2] = -v * np.sin(theta) * self.dt
        A[1, 2] =  v * np.cos(theta) * self.dt
        
        # Partial derivatives for Position w.r.t Velocity
        A[0, 3] = np.cos(theta) * self.dt
        A[1, 3] = np.sin(theta) * self.dt
        
        # Theta depends on w
        A[2, 4] = self.dt 

        # Jacobian w.r.t Input (Deltas) (5x2)
        # v_next = v + delta_v  => 1.0 effect
        # w_next = w + delta_w  => 1.0 effect
        B = np.zeros((5, 2))
        B[3, 0] = 1.0  # Effect of delta_v on v
        B[4, 1] = 1.0  # Effect of delta_w on w

        return A, B
    
    def solve(self, current_pose, ref_trajectory, last_u):
        """
        Args:
            current_pose: [x, y, theta]
            ref_trajectory: Nx5 array [x, y, theta, v, w]
            last_u: [v_current, w_current] (The starting velocity)
        """

        ref_traj_adjusted = ref_trajectory.copy()
        current_theta = current_pose[2]

        diff = ref_traj_adjusted[0, 2] - current_theta
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        ref_traj_adjusted[0, 2] = current_theta + diff

        for i in range(1, len(ref_traj_adjusted)):
            diff = ref_traj_adjusted[i, 2] - ref_traj_adjusted[i-1, 2]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            ref_traj_adjusted[i, 2] = ref_traj_adjusted[i-1, 2] + diff

        # 1. Setup Augmented State
        # Current state includes the velocity we are moving at RIGHT NOW
        if last_u is None:
            last_u = np.array([0.0, 0.0])
            
        x0 = np.concatenate([current_pose, last_u]) # Size 5
        
        X = cp.Variable((5, self.N + 1))  # State: [x, y, th, v, w]
        U = cp.Variable((2, self.N))      # Input: [dv, dw] (Acceleration * dt)

        cost = 0
        constraints = []

        # Initial Condition
        constraints.append(X[:, 0] == x0)
        
        # Linearization point (updated in loop)
        x_pred = x0.copy()

        for k in range(self.N):
            # Reference
            ref_idx = min(k, len(ref_traj_adjusted) - 1)
            # Ref state is full [x, y, th, v, w]
            state_ref = ref_traj_adjusted[ref_idx] 
            
            # Linearize
            # Note: We pass 0 for u_lin because B matrix is constant 
            # for this formulation, but we need x_pred for A matrix.
            A_k, B_k = self.get_augmented_model(x_pred, None)
            
            # Dynamics
            constraints.append(X[:, k + 1] == A_k @ X[:, k] + B_k @ U[:, k])
            
            # Update prediction (simple Euler for the loop)
            # x_pred is updated assuming zero acceleration for linearization
            x_pred = A_k @ x_pred 

            # --- Cost ---
            state_error = X[:, k] - state_ref
            cost += cp.quad_form(state_error, self.Q)

            # Penalize high acceleration (smoothness)
            cost += cp.quad_form(U[:, k], self.R)

            # --- Constraints ---
            
            # 1. Hard limits on Velocity (State Constraints)
            constraints.append(X[3, k+1] <= self.v_max)
            constraints.append(X[3, k+1] >= self.v_min)
            constraints.append(cp.abs(X[4, k+1]) <= self.w_max)
            
            # 2. Hard limits on Acceleration (Input Constraints)
            # Note: U is change per step, so limit is acc_max * dt
            constraints.append(cp.abs(U[0, k]) <= self.acc_v_max * self.dt)
            constraints.append(cp.abs(U[1, k]) <= self.acc_w_max * self.dt)

        # Terminal Cost
        term_ref = ref_trajectory[min(self.N, len(ref_trajectory)-1)]
        cost += cp.quad_form(X[:, self.N] - term_ref, self.Q * self.terminal_weight)

        # Solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-3, eps_rel=1e-3)

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"MPC Warning: {prob.status}")
            return np.array([0.0, 0.0]), None

        # Return the NEW velocity command (v_next = v_curr + delta)
        # This is found in the second column of the State matrix (index 1)
        # X[:, 1] is the state at k=1
        cmd_v = X[3, 1].value
        cmd_w = X[4, 1].value
        
        # Return command and full trajectory (x,y part only) for viz
        return np.array([cmd_v, cmd_w]), X[:3, :].value