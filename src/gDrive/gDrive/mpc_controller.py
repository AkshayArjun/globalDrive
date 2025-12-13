import numpy as np
import cvxpy as cp

class MPCController:
    def __init__(self, horizon=20, dt=0.1):
        self.N = horizon
        self.dt = dt

        # acc to turtle bot - waffle pi documentation  :
        #max speed = 0.26m/s
        #max angular speed = 1.82rad/s

        # --- 1. User Defined Constraints ---
        self.v_max = 0.26  # m/s
        self.v_min = -0.26 # m/s
        self.w_max = 1.82  # rad/s

        # --- 2. MPC Weights (The Tuning Knobs) ---
        # Q: State Error Penalty [x, y, theta]
        self.Q = np.diag([100.0, 100.0, 10.0])

        # R: Input Penalty [v, w]
        self.R = np.diag([1.0, 10.0])

        # Rd: Input/slew Rate Penalty [dv, dw]
        self.Rd = np.diag([5.0, 50.0])


    def get_linear_model(self, x_ref, u_ref):
        """
        Get the linearized discrete-time model matrices A, B around the reference state and input.
        x_ref: [x, y, theta]
        u_ref: [v, w]
        """
        theta = x_ref[2]
        v = u_ref[0]

        A_c = np.array([
            [0, 0, -v * np.sin(theta)],
            [0, 0,  v * np.cos(theta)],
            [0, 0, 0]
        ])

        B_c = np.array([
            [np.cos(theta), 0],
            [np.sin(theta), 0],
            [0, 1]  
        ])

        #now we have x[t + 1] = (A_c*dt + I)*x[t] + B_c*dt*u[t]

        a_d = np.eye(3) + A_c * self.dt
        b_d = B_c * self.dt

        return a_d, b_d
    
    def solve(self, current_state, ref_trajectory):
        X = cp.Variable((3, self.N + 1))  # States over horizon
        U = cp.Variable((2, self.N))      # Inputs over horizon

        cost = 0
        constraints = []

        constraints.append(X[:, 0] == current_state)

        for k in range(self.N):
            ref_idx = min(k, len(ref_trajectory) - 1)
            x_r = ref_trajectory[ref_idx, :3]
            u_r = ref_trajectory[ref_idx, 3:]

            Ak, Bk = self.get_linear_model(x_r, u_r)

            constraints.append(X[:, k + 1] == Ak @ X[:, k] + Bk @ U[:, k])

            state_error = X[:, k] - x_r
            cost += cp.quad_form(state_error, self.Q)

            input_error = U[:, k] - u_r
            cost += cp.quad_form(input_error, self.R)

            if k > 0:
                delta_u = U[:, k] - U[:, k - 1]
                cost += cp.quad_form(delta_u, self.Rd)

            constraints.append(U[0, k] <= self.v_max)
            constraints.append(U[0, k] >= self.v_min)
            constraints.append(cp.abs(U[1, k]) <= self.w_max)
        
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP)

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print("MPC problem not solved optimally.")
            return np.array([0.0, 0.0])  # Return zero commands on failure
        
        return U[:, 0].value
    


