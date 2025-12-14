import rclpy 
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory

import numpy as np
import plotly.graph_objects as go
import os
import math

from gDrive.mpc_controller import MPCController
from gDrive.spline_generator import FastSplineGenerator
from gDrive.waypoint_loader import WaypointLoader
from gDrive.c3bf_filter import LidarTracker, C3BFSolver

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        # Control settings
        self.control_rate = 10.0  # Hz
        self.dt = 1.0 / self.control_rate
        self.goal_tolerance = 0.2  # Tighter tolerance

        try:
            pkg_share = get_package_share_directory('gDrive')
            csv_path = os.path.join(pkg_share, 'resource', 'mission.csv')
            self.get_logger().info(f"Loading waypoints from: {csv_path}")
            loader = WaypointLoader()
            self.waypoints = loader.load_from_csv(csv_path)
        
        except Exception as e:
            self.get_logger().error(f"Failed to load waypoints: {e}")
            self.waypoints = []
        
        if not self.waypoints:
            self.get_logger().info("Using default square waypoints.")
            self.waypoints = [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
                [0.0, 0.0]
            ]

        # Initialize controller with shorter horizon for faster computation
        self.controller = MPCController(horizon=50, dt=self.dt)
        self.planner = FastSplineGenerator()

        self.lidar_tracker = LidarTracker()
        self.safety_solver = C3BFSolver(lookahead_l=0.2)
        self.obstacle_state = None

        # Generate reference path with tighter corners
        self.global_path = self.planner.generate(
            self.waypoints, 
            avg_speed=0.15, 
            ds=0.02, 
            corner_sharpness=0.2 
        )
        self.last_closest_idx = 0
        
        if self.global_path is None:
            self.get_logger().error("Failed to generate global path.")
            return
        self.get_logger().info(f"Generated path with {len(self.global_path)} points.")

        

        # ROS publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'global_path', 10)
        self.mpc_path_pub = self.create_publisher(Path, 'mpc_predicted', 1)   
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, qos_profile_sensor_data
        )
        self.lidar_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data
        )
        
        # Control loop timer
        self.create_timer(1.0 / self.control_rate, self.control_loop)

        # State variables
        self.robot_state = None
        self.current_path_index = 0
        self.complete = False
        self.last_u = np.array([0.0, 0.0])
        
        # Performance tracking
        self.tracking_errors = []
        # Tracking history
        self.history_x = []
        self.history_y = []
        self.history_time = []
        self.start_time = None

    def odom_callback(self, msg):
        if self.start_time is None:
            self.start_time = self.get_clock().now()
            
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = self.euler_from_quaternion([q.x, q.y, q.z, q.w])
       

        v_linear = msg.twist.twist.linear.x
        w_angular = msg.twist.twist.angular.z
        self.robot_state = np.array([x, y, yaw, v_linear, w_angular])
    
    def scan_callback(self, msg):
        self.obstacle_state = self.lidar_tracker.process_scan(msg)

    def find_closest_index(self, state):
        """Find closest point on path with local search."""
        # Search locally around current index
        search_range = 150
        start_i = max(0, self.current_path_index - 20)
        end_i = min(start_i + search_range, len(self.global_path))
        
        # Compute distances
        dists = np.linalg.norm(
            self.global_path[start_i:end_i, :2] - state[:2], 
            axis=1
        )
        
        best_idx = start_i + np.argmin(dists)
        return best_idx

    def control_loop(self):
    
        if self.robot_state is None:
            self.get_logger().info("Waiting for odometry...", throttle_duration_sec=2.0)
            return
        
        curr_pose = self.robot_state[:3]
        current_vel = self.robot_state[3:]
        
        if self.complete:
            return
        
        curr_closest = self.find_closest_index(self.robot_state)

        if curr_closest < self.last_closest_idx:
            closest_idx = self.last_closest_idx
        else:
            closest_idx = curr_closest
            self.last_closest_idx = closest_idx
        
        # Record trajectory
        self.history_x.append(self.robot_state[0])
        self.history_y.append(self.robot_state[1])
        current_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        self.history_time.append(current_time)


        # CRITICAL: Use adaptive lookahead based on current speed
        # Faster speeds need more lookahead
        if len(self.history_x) > 1:
            prev_pos = np.array([self.history_x[-2], self.history_y[-2]])
            current_speed = np.linalg.norm(self.robot_state[:2] - prev_pos) / self.dt
        else:
            current_speed = 0.0
        
        # Lookahead: 2-6 points depending on speed (reduced for tighter tracking)
        lookahead = int(np.clip(2 + current_speed * 20, 2, 6))
        self.current_path_index = min(
            closest_idx + lookahead, 
            len(self.global_path) - 1
        )

        # Check if goal reached
        dist_to_goal = np.linalg.norm(
            self.robot_state[:2] - self.global_path[-1, :2]
        )
        near_end = closest_idx >= len(self.global_path) - 20
        
        if dist_to_goal < self.goal_tolerance and near_end:
            self.get_logger().info("=== Goal Reached! ===")
            self.complete = True
            self.stop_robot()
            self.generate_report()
            return
        
        # Build MPC horizon reference
        horizon_ref = []
        for k in range(self.controller.N):
            idx = min(self.current_path_index + k, len(self.global_path) - 1)
            horizon_ref.append(self.global_path[idx, :5])
        horizon_ref = np.array(horizon_ref)
        
        # Solve MPC
        optimal_u, pred_traj = self.controller.solve(
            curr_pose, 
            horizon_ref,
            current_vel
        )

        safe_u = optimal_u

        if self.obstacle_state is not None:
            safe_v, safe_w = self.safety_solver.solve(
                current_vel,
                optimal_u,
                self.obstacle_state
            )

            if optimal_u[0] > 0.1 and safe_v < 0.05:
                self.get_logger().warn("Blocked by obstacle! Initiating Evasive Maneuver.")
                
                # Force a rotation to find a clear path
                safe_v = 0.0
                safe_w = 0.4  # Rotate left in place
                
                # (Advanced: Check obstacle_state.y to decide left/right turn)
                # If obstacle is to our left (cy > 0), turn right (negative w)
                cx, cy, _, _, _ = self.obstacle_state
                if cy > 0: 
                    safe_w = -0.4
            
            # Update the command
            safe_u = np.array([safe_v, safe_w])

        
        # Publish control command
        cmd = Twist()
        cmd.linear.x = float(safe_u[0])
        cmd.angular.z = float(safe_u[1])
        self.cmd_pub.publish(cmd)

        self.last_u = safe_u
        # Track prediction for visualization
        if pred_traj is not None and not np.isnan(pred_traj).any():
            self.publish_mpc_prediction(pred_traj)
        else:
            self.get_logger().warn("MPC Prediction contains NaNs - skipping viz")
        
        # Compute and log tracking error
        ref_point = self.global_path[closest_idx, :2]
        error = np.linalg.norm(self.robot_state[:2] - ref_point)
        self.tracking_errors.append(error)
        
        if len(self.tracking_errors) % 50 == 0:
            avg_error = np.mean(self.tracking_errors[-50:])
            self.get_logger().info(
                f"Progress: {100*closest_idx/len(self.global_path):.1f}% | "
                f"Error: {error:.3f}m | Avg: {avg_error:.3f}m"
            )


        # Publish path visualization periodically
        if len(self.history_x) % 10 == 0:
            self.publish_viz_path()
    
    def generate_report(self):
        self.get_logger().info("Generating trajectory report...")
        
        # Calculate statistics
        errors = np.array(self.tracking_errors)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        std_error = np.std(errors)
        
        fig = go.Figure()

        # Reference path
        fig.add_trace(go.Scatter(
            x=self.global_path[:, 0], 
            y=self.global_path[:, 1],
            mode='lines', 
            name='Reference Path',
            line=dict(color='green', width=2, dash='dash')
        ))

        # Actual robot path
        fig.add_trace(go.Scatter(
            x=self.history_x, 
            y=self.history_y,
            mode='lines', 
            name='Robot Path',
            line=dict(color='blue', width=3)
        ))

        # Waypoints
        wp_array = np.array(self.waypoints)
        fig.add_trace(go.Scatter(
            x=wp_array[:, 0], 
            y=wp_array[:, 1],
            mode='markers', 
            name='Waypoints',
            marker=dict(size=12, color='red', symbol='star')
        ))

        fig.update_layout(
            title=f"MPC Tracking Report<br>" +
                  f"<sub>Mean Error: {mean_error:.3f}m | " +
                  f"Max Error: {max_error:.3f}m | " +
                  f"Std Dev: {std_error:.3f}m</sub>",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            template="plotly_white",
            width=900,
            height=700
        )

        save_path = os.path.expanduser("~/mission_result.html")
        fig.write_html(save_path)
        self.get_logger().info(f"Report saved: {save_path}")
        
        # Print summary
        self.get_logger().info(
            f"\n=== Performance Summary ===\n"
            f"Mean tracking error: {mean_error:.3f} m\n"
            f"Max tracking error:  {max_error:.3f} m\n"
            f"Std deviation:       {std_error:.3f} m\n"
            f"Mission time:        {self.history_time[-1]:.1f} s\n"
            f"=========================="
        )
    
    def stop_robot(self):
        """Send zero velocity command."""
        cmd = Twist()
        for _ in range(5):  # Send multiple times to ensure it's received
            self.cmd_pub.publish(cmd)
    
    def publish_viz_path(self):
        """Publish reference path for RViz visualization."""
        path_msg = Path()
        path_msg.header.frame_id = "odom"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for point in self.global_path[::5]:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(point[0]) 
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
    
    def publish_mpc_prediction(self, pred_traj):
        """Publish MPC predicted trajectory."""
        path_msg = Path()
        path_msg.header.frame_id = "odom"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for i in range(pred_traj.shape[1]):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(pred_traj[0, i])
            pose.pose.position.y = float(pred_traj[1, i])
            pose.pose.position.z = 0.3
            path_msg.poses.append(pose)
        
        self.mpc_path_pub.publish(path_msg)

    def euler_from_quaternion(self, quat):
        """Convert quaternion to Euler angles."""
        x, y, z, w = quat
        
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = math.asin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
        if not node.complete:
            node.generate_report()
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()