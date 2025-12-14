import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')

    gazebo_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_turtlebot3_gazebo, 'launch', 'empty_world.launch.py')
        )
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    gdrive_node = Node(
        package='gDrive',
        executable='navigation_node',
        name='mpc_controller',
        output='screen',
        parameters=[{'use_sim_time': True}] 
    )

    # --- 5. Assemble the Launch Description ---
    return LaunchDescription([
        gazebo_sim,
        rviz_node,
        TimerAction(
            period=5.0,
            actions=[gdrive_node]
        )
    ])