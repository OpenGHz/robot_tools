import roslaunch
import time


class Launcher:
    """https://blog.csdn.net/weixin_44362628/article/details/124097524"""
    def __init__(self, launch_file_path):
        """Initialize the Launcher object with the path to the launch file."""
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)
        self.launch = roslaunch.parent.ROSLaunchParent(self.uuid, [launch_file_path])

    def start(self, sleep=0):
        """Start the launch file. This is a non-blocking call."""
        self.launch.start()
        if sleep > 0:
            time.sleep(sleep)

    def shutdown(self):
        """This is a blocking call that will not return until the launch file has been shutdown."""
        self.launch.shutdown()


class Noder(object):
    """https://blog.csdn.net/weixin_44362628/article/details/124097524"""
    def __init__(self, package, node_type, node_name, node_args):
        """Initialize the Noder object with the package, node type, node name, and node arguments."""
        self.package = package
        self.node_type = node_type
        self.node_name = node_name
        self.node_args = node_args
        self.node = roslaunch.core.Node(package, node_type, node_name, node_args)
        self.launch = roslaunch.scriptapi.ROSLaunch()

    def start(self, sleep=0):
        """Start the node. This is a non-blocking call."""
        self.launch.start()
        self.launch.launch(self.node)
        if sleep > 0:
            time.sleep(sleep)

    def shutdown(self):
        """This is a blocking call that will not return until the node has been shutdown."""
        self.launch.stop()


if __name__ == "__main__":
    pass
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