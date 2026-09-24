"""Launch file for UAV Aegis fault inference."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_arg = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to trained model (.pth)'
    )
    
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='live',
        description='Operating mode: live or replay'
    )
    
    input_arg = DeclareLaunchArgument(
        'input_file',
        default_value='',
        description='Input CSV file for replay mode'
    )
    
    topic_arg = DeclareLaunchArgument(
        'imu_topic',
        default_value='/imu/data',
        help='IMU topic to subscribe to'
    )
    
    output_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='/fault_detection',
        help='Output topic for fault detection'
    )
    
    return LaunchDescription([
        model_arg,
        mode_arg,
        input_arg,
        topic_arg,
        output_arg,
        Node(
            package='uav_aegis',
            executable='fault_inference_node',
            name='fault_inference_node',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'mode': LaunchConfiguration('mode'),
                'input_file': LaunchConfiguration('input_file'),
                'imu_topic': LaunchConfiguration('imu_topic'),
                'output_topic': LaunchConfiguration('output_topic'),
            }],
        ),
    ])