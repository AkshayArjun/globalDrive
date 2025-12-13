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

    def generate(self, waypoints, avg_speed = 0.5, ds = 0.05):
        waypoints = np.array(waypoints)
        if len(waypoints) < 2:
            return None # Need at least two waypoints to create a spline
        
        tangents = np.zeros_like(waypoints)

        tangents[0] = [0.0, 0.0]
        tangents[-1] = [0.0, 0.0]

        vecs = waypoints[2:] - waypoints[:-2]
        dist = np.linalg.norm(vecs, axis=1, keepdims=True)
        norm_vecs = np.divide(vecs, dist, out=np.zeros_like(vecs), where=dist!=0)
        tangents[1:-1] = norm_vecs * avg_speed

        p0 = waypoints[:-1]
        p1 = waypoints[1:]
        v0 = tangents[:-1]
        v1 = tangents[1:]
        a0 = np.zeros_like(p0)
        a1 = np.zeros_like(p0)

        seg_dists = np.linalg.norm(p1 - p0, axis=1)
        T = np.maximum(seg_dists / avg_speed, 1e-3)

        v0_n = v0 * T[:, None] 
        v1_n = v1 * T[:, None]
        a0_n = a0 * (T**2)[:, None]
        a1_n = a1 * (T**2)[:, None]

        G = np.stack([p0, v0_n, a0_n, p1, v1_n, a1_n], axis=1)

        coeffs = np.einsum('ij,sjk->sik',  self.M, G)

        full_traj = []

        for i in range(len(T)):
            # Create time steps
            num_steps = int(seg_dists[i] / ds)
            if num_steps < 2: num_steps = 2
            
            u = np.linspace(0, 1, num_steps)
            
            # Powers of u matrix: [1, u, u^2, u^3, u^4, u^5]
            U_pow = np.stack([np.ones_like(u), u, u**2, u**3, u**4, u**5])
            
            # Position = C^T * U
            seg_pos = (coeffs[i].T @ U_pow).T
            
            # Velocity = Derivative of C^T * U
            # Derivative powers: [0, 1, 2u, 3u^2, 4u^3, 5u^4]
            U_vel = np.stack([np.zeros_like(u), np.ones_like(u), 2*u, 3*u**2, 4*u**3, 5*u**4])
            # Scale back to real time (divide by T)
            seg_vel = (coeffs[i].T @ U_vel).T / T[i]
            
            # Compute Heading (theta) and Speed (v)
            theta = np.arctan2(seg_vel[:, 1], seg_vel[:, 0])
            v_mag = np.linalg.norm(seg_vel, axis=1)
            
            # Compute Angular Velocity (w)
            # w = (vx * ay - vy * ax) / v^2
            # We need acceleration for this.
            U_acc = np.stack([np.zeros_like(u), np.zeros_like(u), np.full_like(u, 2), 6*u, 12*u**2, 20*u**3])
            seg_acc = (coeffs[i].T @ U_acc).T / (T[i]**2)
            
            cross_prod = seg_vel[:,0] * seg_acc[:,1] - seg_vel[:,1] * seg_acc[:,0]
            w_ref = np.divide(cross_prod, v_mag**2, out=np.zeros_like(cross_prod), where=v_mag > 0.01)

            # Stack it all: [x, y, theta, v, w]
            seg_data = np.column_stack([seg_pos, theta, v_mag, w_ref])
            
            # Avoid duplicating points between segments
            if i > 0:
                seg_data = seg_data[1:]
                
            full_traj.append(seg_data)

        return np.vstack(full_traj)