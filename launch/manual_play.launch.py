import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('car_sim'), 'config', 'sim_params.yaml')

    road_seed_arg = DeclareLaunchArgument(
        'road_seed', default_value='42',
        description='Seed for the procedural road generator')

    sim_node = Node(
        package='car_sim',
        executable='sim_node',
        name='sim_node',
        output='screen',
        parameters=[config, {'road_seed': LaunchConfiguration('road_seed')}],
    )

    manual_control_node = Node(
        package='car_sim',
        executable='manual_control_node',
        name='manual_control_node',
        output='screen',
        parameters=[{
            'window_width': 1000,
            'window_height': 1000,
            'publish_rate': 20.0,
        }],
    )

    return LaunchDescription([road_seed_arg, sim_node, manual_control_node])
