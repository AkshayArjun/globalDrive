import numpy as np
import plotly.graph_objects as go
from mpc_controller import MPCController

def simulate_robot_motion(state, u, dt):
    """
    Simple Unicycle Physics: Updates x, y, theta based on v, w
    state: [x, y, theta]
    u: [v, w]
    """
    x, y, theta = state
    v, w = u
    
    # Update State (Forward Euler)
    new_x = x + v * np.cos(theta) * dt
    new_y = y + v * np.sin(theta) * dt
    new_theta = theta + w * dt
    
    return np.array([new_x, new_y, new_theta])

def main():
    # 1. Setup
    dt = 0.1
    mpc = MPCController(horizon=20, dt=dt)
    
    # Check if we are using the 'Winning' Weights
    print(f"Testing Weights: Q={np.diag(mpc.Q)[:2]}...")

    # 2. Create Reference Path (Straight Line for clarity)
    # Target: Drive along Y=1.0 line at 0.2 m/s
    ref_traj = []
    for i in range(100): # 10 seconds of path
        ref_traj.append([i*0.2*dt, 1.0, 0.0, 0.2, 0.0])
    ref_traj = np.array(ref_traj)

    # 3. Initial State (Start at 0,0 - creates 1.0m Position Error)
    robot_state = np.array([0.0, 0.0, 0.0])
    
    # History for plotting
    history_x = [robot_state[0]]
    history_y = [robot_state[1]]

    # 4. Run Control Loop (Simulation)
    print("Simulating 50 steps...")
    for i in range(50):
        # Create a rolling reference horizon
        horizon_ref = []
        for k in range(mpc.N):
            idx = min(i + k, len(ref_traj) - 1)
            horizon_ref.append(ref_traj[idx])
        horizon_ref = np.array(horizon_ref)
        
        # A. Solve MPC
        u_opt = mpc.solve(robot_state, horizon_ref)
        
        # B. Move Robot (Simulate Physics)
        robot_state = simulate_robot_motion(robot_state, u_opt, dt)
        
        # Save history
        history_x.append(robot_state[0])
        history_y.append(robot_state[1])

    # 5. Visualize
    fig = go.Figure()

    # Trace 1: The Goal Line (Green)
    fig.add_trace(go.Scatter(
        x=ref_traj[:, 0], y=ref_traj[:, 1],
        mode='lines', name='Reference (Target)',
        line=dict(color='green', width=2, dash='dash')
    ))

    # Trace 2: The Actual Path Driven (Blue)
    fig.add_trace(go.Scatter(
        x=history_x, y=history_y,
        mode='lines+markers', name='Actual Robot Path',
        marker=dict(size=5, color='blue'),
        line=dict(width=3)
    ))

    fig.update_layout(
        title="Closed-Loop Simulation (Does it Overshoot?)",
        xaxis_title="X Position (m)",
        yaxis_title="Y Position (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white"
    )
    fig.show()

if __name__ == "__main__":
    main()