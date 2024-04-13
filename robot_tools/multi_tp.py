from subprocess import Popen
import os
import atexit
import signal
import time

from typing import List, Union, Optional


class CLItools(object):

    @staticmethod
    def get_shell(only_name=False) -> str:
        if only_name:
            return os.environ["SHELL"].split("/")[-1]
        else:
            return os.environ["SHELL"]

    @staticmethod
    def check_conda(show=False) -> bool:
        if "CONDA_PREFIX" in os.environ:
            use_conda = True
            info = f"Running in Conda environment: {os.environ['CONDA_PREFIX']}"
        else:
            use_conda = False
            info = "Not running in a Conda environment"
        if show:
            print(info)
        return use_conda

    @classmethod
    def run_command(cls, command, wait=False, executable=None, sleep=0) -> Popen:
        if executable is None:
            executable = cls.get_shell()
        process = Popen(
            command, shell=True, executable=executable, env=os.environ.copy()
        )
        if wait:
            process.wait()
        if sleep:
            time.sleep(sleep)
        return process

    @classmethod
    def run_commands(cls, commands, wait, executable=None, sleeps=None) -> List[Popen]:
        processes: List[Popen] = []
        for index, cmd in enumerate(commands):
            processes.append(cls.run_command(cmd, wait, executable))
            if sleeps is not None:
                time.sleep(sleeps[index])
        if wait:
            for process in processes:
                process.wait()
        return processes

    @staticmethod
    def shutdown_processes(processes: List[Popen]) -> None:
        for process in processes:
            process.kill()
        for process in processes:
            process.wait()


class SubCLIer(CLItools):
    child_processes: List[Popen] = []

    @classmethod
    def run(
        cls,
        cmds: Union[str, List[str]],
        wait=False,
        executable=None,
        sleeps:Union[float, List[float]]=None,
    ) -> Union[Popen, List[Popen]]:
        if isinstance(cmds, str):
            if isinstance(sleeps, list):
                sleeps = sleeps[0]
            p = cls.run_command(cmds, wait, executable, sleeps)
            cls.child_processes.append(p)
            return p
        else:
            if isinstance(sleeps, float):
                sleeps = [sleeps] * len(cmds)
            ps = cls.run_commands(cmds, wait, executable, sleeps)
            cls.child_processes.extend(ps)
            return ps

    @classmethod
    def kill(cls, processes: Optional[List[Popen]] = None) -> None:
        if processes is None:
            processes = cls.child_processes

        cls.shutdown_processes(processes)

    @classmethod
    def kill_atexit(cls):
        atexit.register(cls.kill)

    @classmethod
    def get_pids(cls) -> List[int]:
        return [p.pid for p in cls.child_processes]


class SubPython(object):
    child_processes: List[Popen] = []

    @classmethod
    def start_process(cls, script_path, python_path="python3"):
        process = Popen([python_path, script_path])
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
