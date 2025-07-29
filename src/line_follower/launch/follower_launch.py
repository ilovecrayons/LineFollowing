from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    dc_node = Node(
        package='camera_ros',
        executable='camera_node',
        name='downward_camera',
        namespace='downwardCamera',
        parameters=[
            {'camera': '/base/axi/pcie@120000/rp1/i2c@88000/imx708@1a'},
            {'frame_id': 'downward_camera_frame'},
            {'fps': 10.0},
            {'width': 1280},
            {'height': 720}
        ],
        output='screen'
    )

    fc_node = Node(
        package='camera_ros',
        executable='camera_node',
        name='front_camera',
        namespace='frontCamera',
        parameters=[
            {'camera': '/base/axi/pcie@120000/rp1/i2c@80000/imx708@1a'},
            {'frame_id': 'front_camera_frame'},
            {'fps': 10.0},
            {'width': 1280},
            {'height': 720}
        ],
        output='screen'
    )
    
    
    follower_node = Node(
        package='line_follower',
        executable='tracker',
        output='screen'
    )

    detector_node = Node(
        package='line_follower',
        executable='detector',
        output='screen'
    )

    return LaunchDescription([dc_node, fc_node, follower_node, detector_node])
