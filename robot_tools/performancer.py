import os
import psutil


class Memorier(object):
    pid = os.getpid()
    process = psutil.Process(pid)

    @classmethod
    def get_current_memory_mb(cls) -> float:
        # 获取当前进程内存占用
        info = cls.process.memory_full_info()
        return info.uss / 1024.0 / 1024.0

    @classmethod
    def get_sys_memory_max_gb(cls, show=False) -> float:
        # 获取系统总内存
        cls.mem = psutil.virtual_memory()
        zj = float(cls.mem.total) / 1024 / 1024 / 1024
        if show:
            print("系统总内存:%d.3GB" % zj)
        return zj

    @classmethod
    def get_sys_memory_free_gb(cls, show=False) -> float:
        # 获取系统空闲内存
        cls.mem = psutil.virtual_memory()
        kx = float(cls.mem.free) / 1024 / 1024 / 1024
        if show:
            print("系统空闲内存:%d.3GB" % kx)
        return kx

    @classmethod
    def get_sys_memory_used_gb(cls, show=False) -> float:
        # 获取系统已使用内存
        cls.mem = psutil.virtual_memory()
        ysy = float(cls.mem.used) / 1024 / 1024 / 1024
        if show:
            print("系统已使用内存:%d.3GB" % ysy)
        return ysy

    @classmethod
    def get_sys_memory_info_gb(cls, show=False) -> tuple:
        # 获取系统内存占用信息
        max = cls.get_sys_memory_max_gb()
        free = cls.get_sys_memory_free_gb()
        used = cls.get_sys_memory_used_gb()
        if show:
            print("系统内存占用信息:总内存%.3fGB,空闲%.3fGB,已使用%.3fGB" % (max, free, used))
        return max, used, free

    @classmethod
    def memory_monitor(cls, show=False):
        # 内存监控 TODO: in thread and show in tdqm
        cls.mem = psutil.virtual_memory()
        cls.get_sys_memory_info_gb(show)
        cls.get_current_memory_mb()
        if show:
            print("当前进程内存占用:%.3fMB" % cls.get_current_memory_mb())
        return cls.get_current_memory_mb()