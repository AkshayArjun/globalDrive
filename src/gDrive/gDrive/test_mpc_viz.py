import numpy as np
import plotly.graph_objects as go
import cvxpy as cp
from mpc_controller import MPCController

# We need to inherit or slightly modify the class to extract the "Predicted Horizon" (X)
# for visualization purposes. The normal .solve() only returns inputs U.
class DebugMPC(MPCController):
    def solve_debug(self, current_state, ref_traj):
        # ... (Copying key parts for visualization extraction) ...
        X = cp.Variable((3, self.N + 1)) 
        U = cp.Variable((2, self.N))     
        cost = 0
        constraints = [X[:, 0] == current_state]
        
        for k in range(self.N):
            ref_idx = min(k, len(ref_traj)-1)
            x_r = ref_traj[ref_idx, :3]
            u_r = ref_traj[ref_idx, 3:]
            Ak, Bk = self.get_linear_model(x_r, u_r)
            constraints.append(X[:, k+1] == Ak @ X[:, k] + Bk @ U[:, k])
            
            # Simplified Cost for Viz
            state_error = X[:, k] - x_r
            cost += cp.quad_form(state_error, self.Q)
            if k > 0: cost += cp.quad_form(U[:, k] - U[:, k-1], self.Rd)
            
            constraints.append(U[0, k] <= self.v_max)
            constraints.append(U[0, k] >= self.v_min)
            constraints.append(cp.abs(U[1, k]) <= self.w_max)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)
        
        # RETURN THE FULL PREDICTED PATH (X)
        return X.value

def main():
    mpc = DebugMPC(horizon=20, dt=0.1)
    
    # 1. Create a Curved Reference Path (Green)
    # A simple turn: Moving at 0.2 m/s, Turning at 0.3 rad/s
    ref_traj = []
    x, y, theta = 0, 0, 0
    for _ in range(30):
        ref_traj.append([x, y, theta, 0.2, 0.3])
        x += 0.2 * 0.1 * np.cos(theta)
        y += 0.2 * 0.1 * np.sin(theta)
        theta += 0.3 * 0.1
    ref_traj = np.array(ref_traj)

    # 2. Perturb the Robot (Place it OFF the path)
    # Robot starts at (0, -0.2), pointing wrong way (-0.2 rad)
    current_state = np.array([0.0, -0.2, -0.2])
    
    # 3. Solve
    predicted_path = mpc.solve_debug(current_state, ref_traj[:20])
    
    # 4. Plot
    fig = go.Figure()

    # Trace 1: The Reference Path (Green)
    fig.add_trace(go.Scatter(
        x=ref_traj[:, 0], y=ref_traj[:, 1],
        mode='lines+markers', name='Reference Path',
        line=dict(color='green', width=3)
    ))

    # Trace 2: The Robot Current Position (Blue X)
    fig.add_trace(go.Scatter(
        x=[current_state[0]], y=[current_state[1]],
        mode='markers', name='Robot',
        marker=dict(size=15, color='blue', symbol='x')
    ))

    # Trace 3: The MPC Prediction (Red Dots)
    # This shows how the robot PLANS to merge back
    fig.add_trace(go.Scatter(
        x=predicted_path[0, :], y=predicted_path[1, :],
        mode='lines+markers', name='MPC Predicted Horizon',
        line=dict(color='red', dash='dot')
    ))

    fig.update_layout(
        title="MPC Horizon Visualization (Merging Check)",
        xaxis_title="X (m)", yaxis_title="Y (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white"
    )
    
    fig.show()

if __name__ == "__main__":
    main()