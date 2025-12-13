import rclpy 
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path, Odometry

import numpy as np
import plotly.graph_objects as go
import os
import math

from gDrive.mpc_controller import MPCController
from gDrive.spline_generator import FastSplineGenerator

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        #node settings:
        self.control_rate = 10
        self.dt = 1.0 / self.control_rate
        self.goal_tolerance = 0.15

        #way points: 
         # tbd by reading from a file or parameter
        self.waypoints = [
            [0.0, 0.0],   # Start
            
            # Corner 1 (Bottom Right)
            [1.5, 0.0],   # Start turning here
            [2.0, 0.5],   # Finish turning here
            
            # Corner 2 (Top Right)
            [2.0, 1.5],   # Start turning
            [1.5, 2.0],   # Finish turning
            
            # Corner 3 (Top Left)
            [0.5, 2.0],   # Start turning
            [0.0, 1.5],   # Finish turning
            
            # Corner 4 (Back Home)
            [0.0, 0.5],   # Start turning
            [0.0, 0.0]    # Finish at home
        ]

        #initialise the classes:
        self.controller = MPCController(horizon=20, dt=self.dt)
        self.planner = FastSplineGenerator()

        #generate the global path:
        self.global_path = self.planner.generate(self.waypoints, avg_speed=0.2, ds=0.05)
        if self.global_path is None:
            self.get_logger().error("Failed to generate global path from waypoints.")
            return
        
        self.get_logger().info(f"Generated global path with {len(self.global_path)} points.")

        self.history_x = []
        self.history_y = []

        #pubs and subs:
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'global_path', 10)
        self.mpc_path_pub = self.create_publisher(Path, 'mpc_predicted', 10)   

        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)

        self.create_timer(1.0 / self.control_rate, self.control_loop)

        self.robot_state = None
        self.current_path_index = 0
        self.complete = False

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        _, _, yaw = self.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.robot_state = np.array([x, y, yaw])

    def find_closest_index(self, state):
        search_window = 100 
        start_i = self.current_path_index
        end_i = min(start_i + search_window, len(self.global_path))
        
        # Calculate distances to all points in window
        dists = np.linalg.norm(self.global_path[start_i:end_i, :2] - state[:2], axis=1)
        
        # Return global index of the closest point
        best_local_idx = np.argmin(dists)
        return start_i + best_local_idx

    def control_loop(self):
        if self.robot_state is None:
            self.get_logger().info("Waiting for odometry...", throttle_duration_sec=2.0)
            return
        
        if self.complete:
            self.stop_robot()
            return
        
        self.history_x.append(self.robot_state[0])
        self.history_y.append(self.robot_state[1])

        closest_idx = self.find_closest_index(self.robot_state)
        self.current_path_index = closest_idx

        dist_to_end = np.linalg.norm(self.robot_state[:2] - self.global_path[-1, :2])
        near_end_of_array = (self.current_path_index >= len(self.global_path) - 5)
        if dist_to_end < self.goal_tolerance and near_end_of_array:
            self.get_logger().info("Goal Reached!")
            self.complete = True
            self.stop_robot()
            return
        
        horizon_ref = []
        for k in range(self.controller.N):
            idx = min(self.current_path_index + k, len(self.global_path) - 1)
            horizon_ref.append(self.global_path[idx])
        
        horizon_ref = np.array(horizon_ref)
        optimal_u, pred_traj = self.controller.solve(self.robot_state, horizon_ref)
        if pred_traj is not None:
            self.publish_mpc_prediction(pred_traj)

        cmd = Twist()
        cmd.linear.x = float(optimal_u[0])
        cmd.angular.z = float(optimal_u[1])
        self.cmd_pub.publish(cmd)

        self.publish_viz_path()
    
    def generate_report(self):
        self.get_logger().info("Plotting Trajectory...")
        
        fig = go.Figure()

        # Trace 1: The Reference Path (Green)
        fig.add_trace(go.Scatter(
            x=self.global_path[:, 0], y=self.global_path[:, 1],
            mode='lines', name='Reference Path',
            line=dict(color='green', width=3, dash='dash')
        ))

        # Trace 2: The Actual Robot Path (Blue)
        fig.add_trace(go.Scatter(
            x=self.history_x, y=self.history_y,
            mode='lines', name='Actual Robot Path',
            line=dict(color='blue', width=3)
        ))

        # Trace 3: Start/End Markers
        fig.add_trace(go.Scatter(
            x=[self.waypoints[0][0], self.waypoints[-1][0]],
            y=[self.waypoints[0][1], self.waypoints[-1][1]],
            mode='markers', name='Start/End',
            marker=dict(size=12, color='red', symbol='star')
        ))

        fig.update_layout(
            title="MPC Mission Report: Reference vs Actual",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            template="plotly_white"
        )

        # Save to file
        save_path = os.path.expanduser("~/mission_result.html")
        fig.write_html(save_path)
        self.get_logger().info(f"REPORT SAVED TO: {save_path}")
    
    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
    
    def publish_viz_path(self):
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
        path_msg = Path()
        path_msg.header.frame_id = "odom"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        num_points = pred_traj.shape[1]
        for i in range(num_points):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(pred_traj[0, i])
            pose.pose.position.y = float(pred_traj[1, i])
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)
        
        self.mpc_path_pub.publish(path_msg)

    def euler_from_quaternion(self, quat):
        """
        Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw)
        """
        x, y, z, w = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z

def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (Ctrl+C) Detected!')
        node.generate_report()
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()