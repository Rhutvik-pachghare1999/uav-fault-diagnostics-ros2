"""Launch file for UAV Aegis fault inference.

The node is argparse-driven (it also runs outside ROS2 for CSV replay), so we
launch the installed console script directly with command-line arguments
instead of ROS parameters, which the node would silently ignore.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import FindExecutable, LaunchConfiguration


def generate_launch_description():
    model_arg = DeclareLaunchArgument(
        "model_path", default_value="", description="Path to trained model (.pth), REQUIRED"
    )

    mode_arg = DeclareLaunchArgument("mode", default_value="live", description="Operating mode: live or replay")

    input_arg = DeclareLaunchArgument("input_file", default_value="", description="Input CSV file for replay mode")

    topic_arg = DeclareLaunchArgument("imu_topic", default_value="/imu/data", description="IMU topic to subscribe to")

    rpm_topic_arg = DeclareLaunchArgument(
        "rpm_topic",
        default_value="/rotor_rpms",
        description="rotor RPM telemetry topic (Float32MultiArray, 4 values); "
        "required when the model consumes rpm channels",
    )

    output_arg = DeclareLaunchArgument(
        "output_topic", default_value="/fault_detection", description="Output topic for fault detection"
    )

    return LaunchDescription(
        [
            model_arg,
            mode_arg,
            input_arg,
            topic_arg,
            rpm_topic_arg,
            output_arg,
            ExecuteProcess(
                cmd=[
                    FindExecutable(name="fault_inference_node"),
                    "--model",
                    LaunchConfiguration("model_path"),
                    "--mode",
                    LaunchConfiguration("mode"),
                    "--input",
                    LaunchConfiguration("input_file"),
                    "--topic",
                    LaunchConfiguration("imu_topic"),
                    "--rpm-topic",
                    LaunchConfiguration("rpm_topic"),
                    "--publish_topic",
                    LaunchConfiguration("output_topic"),
                ],
                output="screen",
            ),
        ]
    )
