import numpy as np
import rclpy
import cvxpy as cp

class LidarTracker:
    def __init__(self):
        self.prev_center = None
        self.prev_time = None
        # NEW: Smoothing buffer for velocity
        self.v_smoothed = np.array([0.0, 0.0])

    def process_scan(self, msg):
        ranges = np.array(msg.ranges)
        # 1. Filter Invalid Data
        valid_indices = np.where((ranges > 0.1) & (ranges < 3.0) & np.isfinite(ranges))[0]
        
        if len(valid_indices) == 0:
            return None

        # 2. "Simple Logic" Clustering
        clusters = []
        current_cluster = []
        
        # FOV Limit: +/- 135 degrees
        fov_limit = 135.0 * (np.pi / 180.0) 
        
        for i in range(len(valid_indices)):
            idx = valid_indices[i]
            r = ranges[idx]
            # Normalize angle
            raw_angle = msg.angle_min + idx * msg.angle_increment
            angle = np.arctan2(np.sin(raw_angle), np.cos(raw_angle))
            
            if angle < -fov_limit or angle > fov_limit:
                continue

            point = np.array([r * np.cos(angle), r * np.sin(angle)])
            
            if len(current_cluster) == 0:
                current_cluster.append(point)
                continue
                
            prev_point = current_cluster[-1]
            dist = np.linalg.norm(point - prev_point)
            
            if dist < 0.3:
                current_cluster.append(point)
            else:
                if len(current_cluster) > 3:
                    clusters.append(np.array(current_cluster))
                current_cluster = [point]
        
        if len(current_cluster) > 3:
            clusters.append(np.array(current_cluster))

        if not clusters:
            return None

        # 3. Find Most Dangerous Cluster
        closest_dist = float('inf')
        target_cluster = None

        for cluster in clusters:
            centroid = np.mean(cluster, axis=0)
            dist = np.linalg.norm(centroid)
            
            if dist < closest_dist:
                closest_dist = dist
                target_cluster = cluster
        
        if target_cluster is None:
            return None

        # 4. Fit Virtual Circle (OPTION 2: Relaxed Padding)
        cx, cy = np.mean(target_cluster, axis=0)
        dists_to_center = np.linalg.norm(target_cluster - np.array([cx, cy]), axis=1)
        # Reduced padding from 0.15 -> 0.10 to prevent getting stuck in tight gaps
        radius = np.max(dists_to_center) + 0.25

        # 5. Velocity Estimation (Smoothed)
        vx, vy = 0.0, 0.0
        now = rclpy.clock.Clock().now().nanoseconds / 1e9
        
        if self.prev_center is not None and self.prev_time is not None:
            dt = now - self.prev_time
            if dt > 0.001 and dt < 1.0:
                # Raw finite difference
                vx_raw = (cx - self.prev_center[0]) / dt
                vy_raw = (cy - self.prev_center[1]) / dt
                
                # Low Pass Filter (Alpha = 0.2 means 80% history, 20% new)
                # This kills the "jitter" that makes the robot jerky
                alpha = 0.2
                self.v_smoothed[0] = alpha * vx_raw + (1 - alpha) * self.v_smoothed[0]
                self.v_smoothed[1] = alpha * vy_raw + (1 - alpha) * self.v_smoothed[1]
                
                # Clip to reasonable limits
                vx = max(min(self.v_smoothed[0], 2.0), -2.0)
                vy = max(min(self.v_smoothed[1], 2.0), -2.0)
        
        self.prev_center = np.array([cx, cy])
        self.prev_time = now

        return np.array([cx, cy, radius, vx, vy])

class C3BFSolver:
    """
    [cite_start]Implements the Collision Cone Control Barrier Function QP[cite: 1].
    """
    def __init__(self, lookahead_l=0.2):
        self.l = lookahead_l
        self.u = cp.Variable(2)
        self.Lgh = cp.Parameter(2)
        self.Lfh = cp.Parameter()
        self.gamma_h = cp.Parameter()
        self.u_des = cp.Parameter(2)

        cost = cp.sum_squares(self.u - self.u_des)
        constraints = [
            self.Lgh @ self.u >= -self.Lfh - self.gamma_h,
            self.u[0] <= 1.0, self.u[0] >= -1.0,
            self.u[1] <= 3.0, self.u[1] >= -3.0
        ]
        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, robot_vel, mpc_vel, obstacle_state):
        v, w = robot_vel
        cx, cy, r, cvx, cvy = obstacle_state
        
        # 1. Lookahead Geometry
        p_rel = np.array([cx - self.l, cy]) 
        dist_sq = np.dot(p_rel, p_rel)
        dist = np.sqrt(dist_sq)

        # OPTION 2: Relaxed Emergency Stop
        # Allows robot to be slightly inside conservative radius (r - 0.05)
        # before triggering a hard stop.
        if dist < r - 0.3: 
            return 0.0, 0.0 

        # 2. Relative Velocity
        v_robot_L = np.array([v, self.l * w]) 
        v_rel = np.array([cvx, cvy]) - v_robot_L
        norm_v_rel = np.linalg.norm(v_rel) + 1e-6

        # [cite_start]3. Barrier Function h [cite: 214]
        cos_phi = np.sqrt(max(0, dist_sq - r**2)) / dist
        h = np.dot(p_rel, v_rel) + dist * norm_v_rel * cos_phi

        # 4. Lie Derivatives
        term_A = p_rel + v_rel * (np.sqrt(max(0, dist_sq - r**2)) / norm_v_rel)
        J_u = np.array([[-1.0, 0.0], [0.0, -self.l]])
        Lgh_val = term_A @ J_u
        
        # 5. Solve QP
        dt = 0.1 
        a_ref = (mpc_vel[0] - v) / dt
        alpha_ref = (mpc_vel[1] - w) / dt

        self.u_des.value = np.array([a_ref, alpha_ref])
        self.Lgh.value = Lgh_val
        self.Lfh.value = 0.0
        self.gamma_h.value = 1.0 * h 

        try:
            self.prob.solve(solver=cp.OSQP, verbose=False)
            if self.prob.status == 'optimal':
                safe_v = v + self.u.value[0] * dt
                safe_w = w + self.u.value[1] * dt
                return safe_v, safe_w
            else:
                return 0.0, 0.0
        except Exception:
            return 0.0, 0.0