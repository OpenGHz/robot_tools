import numpy as np
from typing import List, Tuple, Dict, Union
from . import recorder


class TrajInfo(object):
    def __init__(
        self,
        trajs_num: int,
        each_points_num: np.ndarray,
        max_points_num: int,
        features_num: int,
    ):
        self.trajs_num = trajs_num
        self.each_points_num = each_points_num
        self.max_points_num = max_points_num
        self.features_num = features_num


class TrajTools(object):
    @staticmethod
    def construct_trajs(
        traj_times: dict, key, num_key=None, series_type=None, mixed_type=None
    ) -> Tuple[Union[tuple, np.ndarray], TrajInfo]:
        """
        按时间(行坐标)组合多组轨迹数据（列坐标）；且有额外两种拼接功能：
        轨迹x的键值为"x"，每个轨迹中根据key选择轨迹点类型；
        每个轨迹点类型对应一条特定类型的轨迹，格式为列表，每个元素为一个时间点的特征数据，格式为列表或者数值；
        每个轨迹的轨迹点数可以不同，但是每个轨迹点类型的特征数必须相同；
        num_key为每个轨迹的轨迹点数量对应的键名，如果为None，则根据key的长度确定点数，否则根据num_key确定点数；
        返回值为按时间组合的轨迹数据，轨迹数，每个轨迹的点数，最大点数；
        轨迹数据格式为numpy数组，轨迹数为列数，每个轨迹点类型的特征数为深度，轨迹点数为行数，轨迹点数不足的用nan填充；
        若特征为数值，则返回的轨迹数据为二维数组，否则为三维数组；
        series_type和mixed_type任何一个不为None时，返回值的第一个元素是一个tuple，包含三个元素，分别为：
            0：按时间组合的轨迹数据；
            1：按轨迹串行拼接的轨迹数据，始终二维；
            2：按时间拼接的轨迹数据，始终二维；
        """
        assert (
            traj_times.get("0") is not None
        ), "Each trajectory must have a continous str(int) key"
        trajs_num = len(traj_times)  # 轨迹数
        trajs_index_max = trajs_num - 1
        end_key = list(traj_times.keys())[-1]
        assert (
            str(trajs_index_max) == end_key
        ), f"Trajectory number must be continous: {trajs_num}:{end_key}"
        # 每个轨迹的点数以及最大点数
        each_points_num = np.zeros(trajs_num)
        if num_key is not None:
            for i in range(trajs_num):
                each_points_num[i] = traj_times[str(i)][num_key]
            max_points_num = np.max(each_points_num)
        else:
            for i in range(trajs_num):
                each_points_num[i] = len(traj_times[str(i)][key])
            max_points_num = np.max(each_points_num)
        try:
            features_num = len(traj_times["0"][key][0])  # 特征数
        except TypeError:
            features_num = 1
        # 按时间组合（时间以最大点数轨迹为准）
        if features_num == 1:
            time_trajs = np.full((int(max_points_num), trajs_num), np.nan)
        else:
            time_trajs = np.full((int(max_points_num), trajs_num, features_num), np.nan)
        if series_type is not None:
            all_points_num = int(sum(each_points_num))
            # default grow vertically
            trajs_series = np.zeros((all_points_num, features_num))
        else:
            trajs_series = None
        if mixed_type is not None:
            trajs_mixed = np.full(
                (int(max_points_num * trajs_num), features_num), np.nan
            )
        else:
            trajs_mixed = None
        current_points_num = 0
        for i in range(trajs_num):
            for j in range(int(each_points_num[i])):
                time_trajs[j, i] = traj_times[str(i)][key][j]
                if mixed_type is not None:
                    trajs_mixed[int(i + j * trajs_num), :] = time_trajs[j, i]
            if series_type is not None:
                points_num = int(each_points_num[i])
                trajs_series[
                    current_points_num : current_points_num + points_num, :
                ] = time_trajs[:, i][:points_num, :]
                current_points_num += points_num
        info = TrajInfo(trajs_num, each_points_num, max_points_num, features_num)
        if series_type is not None or mixed_type is not None:
            if series_type == "h":
                trajs_series = trajs_series.T
            if mixed_type == "h":
                trajs_mixed = trajs_mixed.T
            return (
                (time_trajs, trajs_series, trajs_mixed),
                info,
            )
        else:
            return time_trajs, info

    @staticmethod
    def traj_times_from_time_trajs(time_trajs: np.ndarray) -> np.ndarray:
        """将按时间组合的轨迹数据还原为多组轨迹数据"""
        return time_trajs.T

    @staticmethod
    def delete_nan_from_time_trajs(time_trajs: np.ndarray) -> np.ndarray:
        """删除按时间组合的轨迹数据中的nan"""
        return time_trajs[~np.isnan(time_trajs).all(axis=1)]

    @staticmethod
    def delete_element_by_traj(
        trajs_series: np.ndarray, each_points_num: np.ndarray, index: int, grow_type="h"
    ) -> np.ndarray:
        """删除按轨迹串行拼接的轨迹数据中的某个位置的轨迹（通常是第一个和最后一个）"""
        base = 0
        axis = 0 if grow_type == "v" else 1
        for num in each_points_num:
            if index < 0:
                index_new = int(num + index + base)
            else:
                index_new = int(index + base)
            trajs_series = np.delete(trajs_series, index_new, axis=axis)
            base += num - 1
        return trajs_series

    @staticmethod
    def delete_mixed_by_time(
        trajs_mixed: np.ndarray,
        index: int,
        trajs_num: int,
        max_points_num: int = None,
        grow_type: str = "h",
    ) -> np.ndarray:
        """删除按时间拼接的轨迹数据中的某个位置的轨迹（通常是第一个和最后一个）"""
        axis = 0 if grow_type == "v" else 1
        if index < 0:
            assert (
                max_points_num is not None
            ), "max_points_num must be given when index < 0"
            index = max_points_num + index
        start = int(index * trajs_num)
        end = int((index + 1) * trajs_num)
        return np.delete(
            trajs_mixed,
            slice(start, end, 1),
            axis=axis,
        )

    @staticmethod
    def get_grow_type(trajs: np.ndarray):
        """检查轨迹数据是按v还是h拼接的（要求特征数<轨迹点数）"""
        shape = trajs.shape
        if shape[0] > shape[1]:
            return "v"
        elif shape[0] == shape[1]:
            print("Warning: Traj shape is square")
        else:
            return "h"


class Trajer(object):
    traj_tool = TrajTools

    def __init__(self) -> None:
        pass
