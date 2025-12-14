import numpy as np
import plotly.graph_objects as go
import sys
import os

# --- SYS PATH HACK ---
# This ensures we can import gDrive even if running this script directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from gDrive.mpc_controller import MPCController

def main():
    # 1. Initialize the REAL Controller
    # We don't need a Debug class because .solve() already returns the predicted path!
    dt = 0.1
    horizon = 20
    mpc = MPCController(horizon=horizon, dt=dt)
    
    # 2. Create a Reference Path (Green)
    # Let's make a gentle turn
    ref_traj = []
    x, y, theta = 0.0, 0.0, 0.0
    v_target = 0.2
    w_target = 0.3
    
    # Generate enough points for the horizon
    for _ in range(horizon + 5):
        # Format: [x, y, theta, v, w]
        ref_traj.append([x, y, theta, v_target, w_target])
        
        # Simple Euler integration to generate the path
        x += v_target * dt * np.cos(theta)
        y += v_target * dt * np.sin(theta)
        theta += w_target * dt
        
    ref_traj = np.array(ref_traj)

    # 3. Define Robot State (Blue X)
    # Perturb the robot: Start at (0, -0.2) with a heading error (-0.2 rad)
    current_pose = np.array([0.0, -0.2, -0.2]) # [x, y, theta]
    current_vel  = np.array([0.0, 0.0])        # [v, w] (Starting from stop)
    
    # 4. Solve
    # Note: solve(current_pose, ref_trajectory, current_vel)
    # It returns: optimal_u (controls), pred_traj (the planned path)
    try:
        _, predicted_path = mpc.solve(
            current_pose, 
            ref_traj[:horizon], 
            current_vel
        )
    except Exception as e:
        print(f"Solver failed: {e}")
        return

    if predicted_path is None:
        print("MPC returned None (Infeasible). Check constraints/weights.")
        return

    # 5. Plot
    fig = go.Figure()

    # Trace 1: The Reference Path (Green)
    fig.add_trace(go.Scatter(
        x=ref_traj[:horizon, 0], y=ref_traj[:horizon, 1],
        mode='lines+markers', name='Reference Path',
        line=dict(color='green', width=3, dash='dash')
    ))

    # Trace 2: The Robot Current Position (Blue X)
    fig.add_trace(go.Scatter(
        x=[current_pose[0]], y=[current_pose[1]],
        mode='markers', name='Robot Start',
        marker=dict(size=15, color='blue', symbol='x')
    ))

    # Trace 3: The MPC Prediction (Red Dots)
    # The solver returns shape (3, N+1), so we access rows 0 (x) and 1 (y)
    fig.add_trace(go.Scatter(
        x=predicted_path[0, :], y=predicted_path[1, :],
        mode='lines+markers', name='MPC Predicted Horizon',
        line=dict(color='red', width=2)
    ))

    # Add arrow for initial heading
    fig.add_annotation(
        x=current_pose[0] + 0.1 * np.cos(current_pose[2]),
        y=current_pose[1] + 0.1 * np.sin(current_pose[2]),
        ax=current_pose[0], ay=current_pose[1],
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='blue'
    )

    fig.update_layout(
        title=f"MPC Horizon Check<br><sub>Robot (Blue) merging to Reference (Green)</sub>",
        xaxis_title="X (m)", yaxis_title="Y (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white",
        width=800, height=600
    )
    
    print("Displaying plot...")
    fig.show()

if __name__ == "__main__":
    main()