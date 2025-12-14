import numpy as np
import cvxpy as cp

class MPCController:
    def __init__(self, horizon=20, dt=0.02):
        self.N = horizon
        self.dt = dt

        # --- Robot Constraints ---
        self.v_max = 0.5   # Increased slightly for testing
        self.v_min = -0.2
        self.w_max = 1.82
        
        # Input Limits (Acceleration)
        self.acc_v_max = 0.5 
        self.acc_w_max = 3.0 

        # --- Tuning Weights ---
        # Note: Tuning often needs re-adjustment when adding C matrix
        # because the model is now "stiffer" and more accurate.
        self.Q = np.diag([200.0, 200.0, 100.0, 50.0, 200.0]) 
        self.R = np.diag([100.0, 100.0])
        self.terminal_weight = 50.0

    def get_nonlinear_next_state(self, x, u_delta=np.array([0,0])):
        """
        Predicts the next state using the exact nonlinear Unicycle Model.
        x: [x, y, theta, v, w]
        u_delta: [delta_v, delta_w] (Acceleration * dt)
        """
        x_next = np.zeros(5)
        
        # Current states
        theta = x[2]
        v = x[3]
        w = x[4]
        
        # Apply Inputs (Change in velocity)
        # Note: We apply input first, then move. (Or move then input, depends on convention. 
        # Standard Euler often uses v_k to compute pos_k+1, then updates v_k to v_k+1)
        v_next = v + u_delta[0]
        w_next = w + u_delta[1]
        
        # Update Position (using current v, theta)
        x_next[0] = x[0] + v * np.cos(theta) * self.dt
        x_next[1] = x[1] + v * np.sin(theta) * self.dt
        x_next[2] = x[2] + w * self.dt
        x_next[3] = v_next
        x_next[4] = w_next
        
        return x_next

    def get_linearized_matrices(self, x_bar):
        """
        Computes A, B, C matrices linearized around state x_bar.
        We assume u_bar (delta inputs) are 0 for the linearization reference 
        (constant acceleration assumption).
        """
        theta = x_bar[2]
        v = x_bar[3]
        
        # 1. Jacobian w.r.t State (A)
        A = np.eye(5)
        A[0, 2] = -v * np.sin(theta) * self.dt
        A[1, 2] =  v * np.cos(theta) * self.dt
        A[0, 3] = np.cos(theta) * self.dt
        A[1, 3] = np.sin(theta) * self.dt
        A[2, 4] = self.dt 

        # 2. Jacobian w.r.t Input (B)
        # Stays constant because v_next = v + delta_v is linear
        B = np.zeros((5, 2))
        B[3, 0] = 1.0 
        B[4, 1] = 1.0

        # 3. Affine Constant (C)
        # C = f(x_bar, 0) - (A @ x_bar + B @ 0)
        # This captures the "drift" of the linearization point
        x_next_nonlinear = self.get_nonlinear_next_state(x_bar, u_delta=[0,0])
        C = x_next_nonlinear - (A @ x_bar)

        return A, B, C
    
    def solve(self, current_pose, ref_trajectory, last_u):
        """
        current_pose: [x, y, theta]
        ref_trajectory: [N, 5] -> (x, y, theta, v, w)
        last_u: [v, w] current robot velocity
        """

        # --- Angle Normalization (Crucial for loop closure) ---
        ref_traj_adjusted = ref_trajectory.copy()
        current_theta = current_pose[2]
        
        # Normalize first point relative to robot
        diff = ref_traj_adjusted[0, 2] - current_theta
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        ref_traj_adjusted[0, 2] = current_theta + diff

        # Unwind the rest of the trajectory
        for i in range(1, len(ref_traj_adjusted)):
            diff = ref_traj_adjusted[i, 2] - ref_traj_adjusted[i-1, 2]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            ref_traj_adjusted[i, 2] = ref_traj_adjusted[i-1, 2] + diff

        # Setup Initial State
        if last_u is None: last_u = np.array([0.0, 0.0])
        x0 = np.concatenate([current_pose, last_u])
        
        # Optimization Variables
        X = cp.Variable((5, self.N + 1))
        U = cp.Variable((2, self.N)) # Deltas

        cost = 0
        constraints = [X[:, 0] == x0]

        # --- MPC Horizon Loop ---
        for k in range(self.N):
            # 1. Select Linearization Point (Reference Trajectory)
            ref_idx = min(k, len(ref_traj_adjusted) - 1)
            x_bar = ref_traj_adjusted[ref_idx]
            
            # 2. Compute LTV Matrices (A, B, C)
            A_k, B_k, C_k = self.get_linearized_matrices(x_bar)

            # 3. Dynamics Constraint (with C matrix!)
            constraints.append(X[:, k + 1] == A_k @ X[:, k] + B_k @ U[:, k] + C_k)
            
            # 4. Cost Function
            # We compare predicted state X against the Reference State
            state_error = X[:, k] - x_bar
            cost += cp.quad_form(state_error, self.Q)
            cost += cp.quad_form(U[:, k], self.R)

            # 5. Constraints
            constraints.append(X[3, k+1] <= self.v_max)
            constraints.append(X[3, k+1] >= self.v_min)
            constraints.append(cp.abs(X[4, k+1]) <= self.w_max)
            constraints.append(cp.abs(U[0, k]) <= self.acc_v_max * self.dt)
            constraints.append(cp.abs(U[1, k]) <= self.acc_w_max * self.dt)

        # Terminal Cost
        term_ref = ref_traj_adjusted[min(self.N, len(ref_traj_adjusted)-1)]
        cost += cp.quad_form(X[:, self.N] - term_ref, self.Q * self.terminal_weight)

        # Solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-3, eps_rel=1e-3)

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return np.array([0.0, 0.0]), None
        
        cmd_v = X[3, 1].value
        cmd_w = X[4, 1].value
        
        return np.array([cmd_v, cmd_w]), X[:3, :].value