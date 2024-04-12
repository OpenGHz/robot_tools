import roslaunch


class Launcher:
    """https://blog.csdn.net/weixin_44362628/article/details/124097524"""
    def __init__(self, launch_file_path):
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)
        self.launch = roslaunch.parent.ROSLaunchParent(self.uuid, [launch_file_path])

    def start(self):
        self.launch.start()

    def shutdown(self):
        self.launch.shutdown()

# def start_launch_file():
#     # 创建一个roslaunch配置对象
#     launch = roslaunch.scriptapi.ROSLaunch()
    
#     # 指定ROS Master的URI
#     roslaunch.configure_logging('/tmp/roslaunch.log')
#     launch.start()
    
#     # 加载指定的launch文件
#     package = 'your_package_name'
#     launch_file = 'your_launch_file.launch'
#     launch_file_path = roslaunch.rlutil.resolve_launch_arguments([package, launch_file])
#     launch_file_args = []
#     node = roslaunch.core.Node(package, 'your_node_name', args=launch_file_args)
    
#     # 启动launch文件中的节点
#     launch.launch(node)