from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import EmitEvent

def generate_launch_description():
    scenario_arg = DeclareLaunchArgument('scenario', default_value='case01')
    tau_arg = DeclareLaunchArgument('tau', default_value='0.5')
    share_intent_arg = DeclareLaunchArgument('share_intent', default_value='True')
    route_interval_arg = DeclareLaunchArgument('route_interval', default_value='3.0')
    speed_factor_arg = DeclareLaunchArgument('speed_factor', default_value='1.0')
    auto_close_arg = DeclareLaunchArgument('auto_close', default_value='False')

    return LaunchDescription([
        scenario_arg,
        tau_arg,
        share_intent_arg,
        route_interval_arg,
        speed_factor_arg,
        auto_close_arg,
        
        # 1. Target Ship Node
        Node(
            package='target_ship',
            executable='target_ship_node',
            name='target_ship_node',
            output='screen',
            parameters=[{
                'scenario': LaunchConfiguration('scenario'),
                'share_intent': LaunchConfiguration('share_intent'),
                'route_interval': LaunchConfiguration('route_interval'),
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
                'auto_close': LaunchConfiguration('auto_close')
            }],
            on_exit=[EmitEvent(event=Shutdown())]
        ),

        # 4. Live Plotter Node
        Node(
            package='own_ship',
            executable='live_plotter_node',
            name='live_plotter_node',
            output='screen',
            parameters=[{
                'scenario': LaunchConfiguration('scenario'),
                'mode': LaunchConfiguration('share_intent'),
                'latency': LaunchConfiguration('tau'),
                'interval': LaunchConfiguration('route_interval'), # Mapped to plotter's 'interval' parameter
                'auto_close': LaunchConfiguration('auto_close')
            }],
            on_exit=[EmitEvent(event=Shutdown())]
        )
    ])