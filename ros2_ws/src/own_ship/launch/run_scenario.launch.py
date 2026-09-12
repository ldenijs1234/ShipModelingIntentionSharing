from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    scenario_arg = DeclareLaunchArgument('scenario', default_value='case01')
    tau_arg = DeclareLaunchArgument('tau', default_value='0.5')
    share_intent_arg = DeclareLaunchArgument('share_intent', default_value='True')
    t_advance_arg = DeclareLaunchArgument('t_advance', default_value='15.0')
    speed_factor_arg = DeclareLaunchArgument('speed_factor', default_value='1.0')

    return LaunchDescription([
        scenario_arg,
        tau_arg,
        share_intent_arg,
        t_advance_arg,
        speed_factor_arg,
        # 1. Target Ship Node
        Node(
            package='target_ship',
            executable='target_ship_node',
            name='target_ship_node',
            output='screen',
            parameters=[{
                'scenario': LaunchConfiguration('scenario'),
                'share_intent': LaunchConfiguration('share_intent'),
                't_advance': LaunchConfiguration('t_advance'),
                'speed_factor': LaunchConfiguration('speed_factor'),
            }]
        ),

        # 2. Latency Bridge Node
        Node(
            package='communication_layer',
            executable='latency_bridge_node',
            name='latency_bridge_node',
            output='screen',
            parameters=[{
                'latency': LaunchConfiguration('tau'),
            }]
        ),

        # 3. Own Ship GNC Node
        Node(
            package='own_ship',
            executable='own_ship_node',
            name='own_ship_node',
            output='screen',
            parameters=[{
                'scenario': LaunchConfiguration('scenario'),
                'speed_factor': LaunchConfiguration('speed_factor'),
            }]
        ),

        # 4. Live Plotter Node
        Node(
            package='own_ship',
            executable='live_plotter_node',
            name='live_plotter_node',
            output='screen',
            parameters=[{
                'scenario': LaunchConfiguration('scenario'),
            }]
        )

    ])