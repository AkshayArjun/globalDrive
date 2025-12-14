import numpy as np

class FastSplineGenerator:
    def __init__(self): 
        self.M = np.array([
            [ 1.0,  0.0,  0.0,  0.0,  0.0,  0.0], # c0 = p0
            [ 0.0,  1.0,  0.0,  0.0,  0.0,  0.0], # c1 = v0
            [ 0.0,  0.0,  0.5,  0.0,  0.0,  0.0], # c2 = 0.5*a0
            [-10.0, -6.0, -1.5, 10.0, -4.0,  0.5], # c3
            [ 15.0,  8.0,  1.5, -15.0,  7.0, -1.0], # c4
            [-6.0, -3.0, -0.5,  6.0, -3.0,  0.5]  # c5
        ])

    def generate(self, waypoints, avg_speed=0.15, ds=0.05, corner_sharpness=0.2):
        """
        Generate smooth spline trajectory through waypoints.
        
        Args:
            waypoints: List of [x, y] waypoint coordinates
            avg_speed: Average speed along trajectory (m/s)
            ds: Distance between discretized points (m)
            corner_sharpness: 0.0-1.0, lower = sharper corners, higher = smoother
        """
        waypoints = np.array(waypoints)
        if len(waypoints) < 2:
            return None
        
        n = len(waypoints)
        tangents = np.zeros_like(waypoints)

        # Calculate tangent vectors at each waypoint
        for i in range(n):
            if i == 0:
                # First waypoint: direction to next point
                tangents[i] = (waypoints[1] - waypoints[0])
            elif i == n - 1:
                # Last waypoint: direction from previous point
                tangents[i] = (waypoints[-1] - waypoints[-2])
            else:
                # Middle waypoints: weighted average based on corner angle
                vec_in = waypoints[i] - waypoints[i-1]
                vec_out = waypoints[i+1] - waypoints[i]
                
                # Normalize the vectors
                norm_in = np.linalg.norm(vec_in)
                norm_out = np.linalg.norm(vec_out)
                
                if norm_in > 1e-6:
                    vec_in = vec_in / norm_in
                if norm_out > 1e-6:
                    vec_out = vec_out / norm_out
                
                # Calculate the angle between segments
                dot_product = np.dot(vec_in, vec_out)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle = np.arccos(dot_product)
                
                # Sharper corners get smaller tangent magnitudes
                # angle = 0 (straight) -> factor = 1.0
                # angle = π (U-turn) -> factor = corner_sharpness
                corner_factor = 1.0 - (1.0 - corner_sharpness) * (angle / np.pi)
                
                # Average the directions
                tangent_dir = (vec_in + vec_out) / 2.0
                norm = np.linalg.norm(tangent_dir)
                
                if norm > 1e-6:
                    tangents[i] = (tangent_dir / norm) * corner_factor
                else:
                    # If vectors cancel out (180° turn), use perpendicular
                    tangents[i] = np.array([-vec_in[1], vec_in[0]]) * corner_sharpness
        
        # Scale tangents by desired speed
        for i in range(n):
            norm = np.linalg.norm(tangents[i])
            if norm > 1e-6:
                tangents[i] = (tangents[i] / norm) * avg_speed
            else:
                tangents[i] = np.array([avg_speed, 0.0])

        # Build segments between consecutive waypoints
        p0 = waypoints[:-1]
        p1 = waypoints[1:]
        v0 = tangents[:-1]
        v1 = tangents[1:]
        a0 = np.zeros_like(p0)
        a1 = np.zeros_like(p0)

        # Segment durations based on distance and speed
        seg_dists = np.linalg.norm(p1 - p0, axis=1)
        T = np.maximum(seg_dists / avg_speed, 1e-3)

        # Normalize velocities and accelerations by time
        v0_n = v0 * T[:, None] 
        v1_n = v1 * T[:, None]
        a0_n = a0 * (T**2)[:, None]
        a1_n = a1 * (T**2)[:, None]

        # Stack boundary conditions for each segment
        G = np.stack([p0, v0_n, a0_n, p1, v1_n, a1_n], axis=1)

        # Compute polynomial coefficients
        coeffs = np.einsum('ij,sjk->sik', self.M, G)

        full_traj = []
        current_global_time = 0.0

        for i in range(len(T)):
            # Discretize based on desired spacing
            num_steps = int(seg_dists[i] / ds)
            if num_steps < 2: 
                num_steps = 2
            
            u = np.linspace(0, 1, num_steps)
            seg_time = current_global_time + (u * T[i])
            
            # Position: polynomial evaluation
            U_pow = np.stack([np.ones_like(u), u, u**2, u**3, u**4, u**5])
            seg_pos = (coeffs[i].T @ U_pow).T
            
            # Velocity: derivative of position polynomial
            U_vel = np.stack([np.zeros_like(u), np.ones_like(u), 2*u, 3*u**2, 4*u**3, 5*u**4])
            seg_vel = (coeffs[i].T @ U_vel).T / T[i]
            
            # Compute heading and speed
            theta = np.arctan2(seg_vel[:, 1], seg_vel[:, 0])
            v_mag = np.linalg.norm(seg_vel, axis=1)
            
            # Compute angular velocity: w = (vx * ay - vy * ax) / v^2
            U_acc = np.stack([np.zeros_like(u), np.zeros_like(u), np.full_like(u, 2), 6*u, 12*u**2, 20*u**3])
            seg_acc = (coeffs[i].T @ U_acc).T / (T[i]**2)
            
            cross_prod = seg_vel[:,0] * seg_acc[:,1] - seg_vel[:,1] * seg_acc[:,0]
            w_ref = np.divide(cross_prod, v_mag**2, out=np.zeros_like(cross_prod), where=v_mag > 0.01)

            # Stack: [x, y, theta, v, w]
            seg_data = np.column_stack([seg_pos, theta, v_mag, w_ref , seg_time,])
            
            # Avoid duplicate points between segments
            if i > 0:
                seg_data = seg_data[1:]
                
            full_traj.append(seg_data)

            current_global_time += T[i]

        return np.vstack(full_traj)