from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent
from launch.conditions import UnlessCondition
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scenario_arg = DeclareLaunchArgument('scenario', default_value='case01')
    tau_arg = DeclareLaunchArgument('tau', default_value='0.5')
    share_intent_arg = DeclareLaunchArgument('share_intent', default_value='True')
    route_interval_arg = DeclareLaunchArgument('route_interval', default_value='3.0')
    speed_factor_arg = DeclareLaunchArgument('speed_factor', default_value='1.0')
    auto_close_arg = DeclareLaunchArgument('auto_close', default_value='False')
    min_intent_range_arg = DeclareLaunchArgument('min_intent_range', default_value='10.0')
    headless_arg = DeclareLaunchArgument('headless', default_value='False', description='Disable live GUI plotting')

    # Global sim time parameter for all nodes reading /clock
    sim_time_dict = {'use_sim_time': True}

    # 0. Master Simulation Clock Node (Controls pacing across all nodes)
    sim_clock_node = Node(
        package='communication_layer',
        executable='sim_clock_node',
        name='sim_clock_node',
        output='screen',
        parameters=[{
            'speed_factor': LaunchConfiguration('speed_factor'),
            'step_size': 0.01  # 100 Hz clock resolution
        }]
    )

    # 1. Target Ship Node
    target_ship_node = Node(
        package='target_ship',
        executable='target_ship_node',
        name='target_ship_node',
        output='screen',
        parameters=[
            sim_time_dict,
            {
                'scenario': LaunchConfiguration('scenario'),
                'share_intent': LaunchConfiguration('share_intent'),
                'route_interval': LaunchConfiguration('route_interval'),
            }
        ]
    )

    # 2. Latency Bridge Node
    latency_bridge_node = Node(
        package='communication_layer',
        executable='latency_bridge_node',
        name='latency_bridge_node',
        output='screen',
        parameters=[
            sim_time_dict,
            {
                'latency': LaunchConfiguration('tau'),
            }
        ]
    )

    # 3. Own Ship GNC Node (Directly triggers global launch shutdown on exit)
    own_ship_node = Node(
        package='own_ship',
        executable='own_ship_node',
        name='own_ship_node',
        output='screen',
        parameters=[
            sim_time_dict,
            {
                'scenario': LaunchConfiguration('scenario'),
                'mode': LaunchConfiguration('share_intent'),
                'latency': LaunchConfiguration('tau'),
                'interval': LaunchConfiguration('route_interval'),
                'auto_close': LaunchConfiguration('auto_close'),
                'min_intent_range': LaunchConfiguration('min_intent_range'),
            }
        ],
        on_exit=[EmitEvent(event=Shutdown())]
    )

    # 4. Live Plotter Node (only starts if headless is False)
    live_plotter_node = Node(
        package='own_ship',
        executable='live_plotter_node',
        name='live_plotter_node',
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('headless')),
        parameters=[
            sim_time_dict,
            {
                'scenario': LaunchConfiguration('scenario'),
                'mode': LaunchConfiguration('share_intent'),
                'latency': LaunchConfiguration('tau'),
                'interval': LaunchConfiguration('route_interval'),
                'auto_close': LaunchConfiguration('auto_close')
            }
        ],
    )

    return LaunchDescription([
        scenario_arg,
        tau_arg,
        share_intent_arg,
        route_interval_arg,
        speed_factor_arg,
        auto_close_arg,
        min_intent_range_arg,
        headless_arg,
        sim_clock_node,
        target_ship_node,
        latency_bridge_node,
        own_ship_node,
        live_plotter_node,
    ])