from subprocess import Popen
from typing import List
import os


def get_shell(only_name=False) -> str:
    if only_name:
        return os.environ['SHELL'].split('/')[-1]
    else:
        return os.environ['SHELL']

def check_conda():
    if 'CONDA_PREFIX' in os.environ:
        print("Running in Conda environment:", os.environ['CONDA_PREFIX'])
    else:
        print("Not running in a Conda environment")

def run_command(command, wait=False, executable=None) -> Popen:
    if executable is None:
        executable = get_shell()
    process = Popen(command, shell=True, executable=executable, env=os.environ.copy())
    if wait:
        process.wait()
    return process

def run_commands(commands, wait, executable=None) -> List[Popen]:
    processes:List[Popen] = []
    for command in commands:
        processes.append(run_command(command, wait, executable))
    if wait:
        for process in processes:
            process.wait()
    return processes

def shutdown_processes(processes:List[Popen]) -> None:
    for process in processes:
        process.kill()

import subprocess
import atexit
import signal


class SubPython(object):
    child_processes:List[Popen] = []

    @classmethod
    def start_process(cls, script_path, python_path='python3'):
        process = subprocess.Popen([python_path, script_path])
        cls.child_processes.append(process)
    
    @classmethod
    def cleanup_child_processes(cls):
        for process in cls.child_processes:
            process.send_signal(signal.SIGTERM)
        for process in cls.child_processes:
            process.wait()
    
    @classmethod
    def register_exit_clean(cls):
        atexit.register(cls.cleanup_child_processes)


# # 存储子进程的列表
# _child_processes_:List[Popen] = []

# def start_process(script_path, python_path='python3'):
#     # 启动子进程
#     process = subprocess.Popen([python_path, script_path])
#     # 将子进程添加到列表中
#     _child_processes_.append(process)

# # 注册函数，在主进程退出时关闭子进程
# def cleanup_child_processes():
#     for process in _child_processes_:
#         # 发送 SIGTERM 信号关闭子进程
#         process.send_signal(signal.SIGTERM)
#     # 等待所有子进程退出
#     for process in _child_processes_:
#         process.wait()


if __name__ == '__main__':
    commands = [
        'python3 -m robot_tools.multi_tp',
        'python3 -m robot_tools.multi_tp',
        'python3 -m robot_tools.multi_tp',
    ]
    processes = run_commands(commands, wait=False)
    shutdown_processes(processes)