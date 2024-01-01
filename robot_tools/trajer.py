import numpy as np
from typing import List, Tuple, Dict, Union
from . import recorder
from matplotlib import pyplot as plt
from copy import deepcopy


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
    def mixed_trajs_from_time_trajs(
        time_trajs: np.ndarray,
        trajs_num: int,
        max_points_num: int,
        grow_type: str = "h",
    ) -> np.ndarray:
        """将按时间组合的轨迹数据还原为按时间拼接的轨迹数据"""
        if grow_type == "h":
            return time_trajs.reshape((int(max_points_num * trajs_num), -1)).T
        else:
            return time_trajs.reshape((int(max_points_num * trajs_num), -1))

    @staticmethod
    def mixed_trajs_from_series_trajs(
        trajs_series: np.ndarray,
        each_points_num: np.ndarray,
        trajs_num: int,
        max_points_num: int,
        features_num: int = None,
        grow_type: str = "h",
    ) -> np.ndarray:
        """将按轨迹串行拼接的轨迹数据还原为按时间拼接的轨迹数据"""
        if grow_type != "h":
            trajs_series = trajs_series.T
        if features_num is None:
            features_num = trajs_series.shape[0]
        trajs_mixed = np.full((features_num, (int(max_points_num * trajs_num))), np.nan)
        base = 0
        for i in range(trajs_num):
            points_num = int(each_points_num[i])
            arr = trajs_series[:, int(base) : int(base + points_num)]
            arr_pad = np.pad(
                arr,
                ((0, 0), (0, int(max_points_num - points_num))),
                constant_values=np.nan,
            )
            trajs_mixed[:, i::trajs_num] = arr_pad
            base += points_num
        if grow_type == "h":
            return trajs_mixed
        else:
            return trajs_mixed.T

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

    @staticmethod
    def get_traj_from_mixed_trajs(
        trajs_mixed: np.ndarray, traj_index: int, trajs_num: int, grow_type: str = "h"
    ) -> np.ndarray:
        """从按时间拼接的轨迹数据中获取某个轨迹"""
        if grow_type == "h":
            return trajs_mixed[:, traj_index::trajs_num]
        else:
            return trajs_mixed[traj_index::trajs_num, :]

    @staticmethod
    def get_traj_from_series_trajs(
        trajs_series: np.ndarray,
        each_points_num: np.ndarray,
        traj_index: int,
        trajs_num: int = None,
        grow_type: str = "h",
    ) -> np.ndarray:
        """从按轨迹串行拼接的轨迹数据中获取某个轨迹"""
        if traj_index < 0:
            if trajs_num is None:
                trajs_num = len(each_points_num)
            traj_index = trajs_num + traj_index
        sum_each_points_num = np.sum(each_points_num[:traj_index])
        start = int(sum_each_points_num)
        end = int(sum_each_points_num + each_points_num[traj_index])

        if grow_type == "h":
            return trajs_series[:, start:end]
        else:
            return trajs_series[start:end, :]

    @staticmethod
    def delete_nan(trajs: np.ndarray, axis: int = 0) -> np.ndarray:
        """按行（axis=1）/列（axis=0）删除数组中的nan"""
        if axis == 0:
            return trajs[:, ~np.any(np.isnan(trajs), axis=0)]
        elif axis == 1:
            return trajs[np.all(~np.isnan(trajs), axis=1)]

    @staticmethod
    def concatenate_trajs(*args, grow_type="h"):
        """将轨迹根据grow_type进行拼接"""
        type2axis = {"h": 0, "v": 1}
        trajs, infos = args[0]
        infos = deepcopy(infos)
        for traj, info in args[1:]:
            trajs = np.concatenate((trajs, traj), type2axis[grow_type])
            # infos: TrajInfo
            # info: TrajInfo
            infos.features_num += info.features_num
        return trajs, infos


class Trajer(object):
    traj_tool = TrajTools

    def __init__(self) -> None:
        pass


class TrajsPainter(object):
    traj_tool = TrajTools

    def __init__(
        self, trajs: np.ndarray, info: TrajInfo, type: str = "mixed_h"
    ) -> None:
        """给定轨迹及其对应的类型（目前支持series_v、series_h、mixed_v、mixed_h、time_trajs、traj_times）"""
        self.update_trajs(trajs, info, type)
        # 特征参数配置，一般用于绘制feature-time图
        self.features_axis_labels = tuple(
            [rf"$x_{i}$" for i in range(info.features_num)]
        )
        self.features_lines = ("k",) * info.features_num
        self.features_scatters = (None,) * info.features_num
        self.features_sharex = True
        self.features_sharetitle = "Features Trajectories"
        self.features_titles = ("Features Trajs",) * info.features_num
        self.features_self_labels = (None,) * info.features_num
        self.time_label = r"$t$"
        # 轨迹参数配置，一般用于绘制2Dfeatures轨迹图
        self.trajs_lines = "-ok"
        self.trajs_labels = r"$trajectories_1$"
        self.trajs_markersize = 5
        # 通用绘图参数配置
        self.figure_size = (12, 4)
        self.save_path = None
        self.plt_pause = 0

    def get_trajs_and_info(self) -> Tuple[np.ndarray, TrajInfo]:
        return self._trajs, self._trajs_info

    def update_trajs(self, trajs: np.ndarray, info: TrajInfo, type: str = "mixed_h"):
        if type in ["series_v", "traj_times", "mixed_v"]:
            # 统一转换为水平增长的轨迹
            trajs = trajs.T
        # 统一转换成mixed类型轨迹
        if "series" in type:
            trajs = TrajTools.mixed_trajs_from_series_trajs(
                trajs,
                info.each_points_num,
                info.trajs_num,
                info.max_points_num,
                info.features_num,
            )
        elif "time" in type:
            trajs = TrajTools.mixed_trajs_from_time_trajs(
                trajs, info.trajs_num, info.max_points_num
            )
        self._trajs_info = info
        self._trajs = trajs

    def config_2D(self, labels=(None, None), title=None, save_path=None):
        self.features_axis_labels[0] = (
            labels[0] if labels[0] is not None else self.features_axis_labels[0]
        )
        self.features_axis_labels[1] = (
            labels[1] if labels[1] is not None else self.features_axis_labels[1]
        )
        self.features_titles = title if title is not None else self.features_titles
        self.save_path = save_path if save_path is not None else self.save_path

    def set_pause(self, time: float):
        self.plt_pause = time

    def show(self, pause: float = 0):
        block = False if pause > 0 else True
        plt.show(block=block)
        if not block:
            plt.pause(pause)
            plt.close()

    def plot_features_with_t(
        self,
        points: tuple,
        trajs: tuple,
        indexes: tuple,
        dT: float = 1,
        row_col: tuple = None,
        given_axs=None,
        return_axs=None,
    ):
        """points是连贯的点，而trajs和indexes是指定的序号，可以不连贯"""
        start_point = points[0]
        end_point = points[1]
        assert end_point <= self._trajs_info.max_points_num, "end_point is too large"
        # Time vector
        t = np.arange(0, (end_point - start_point) * dT, dT)
        # Visualize start->end steps of the training data
        if given_axs is None:
            if row_col is None:
                row_col = (len(indexes), 1)
            fig, axs = plt.subplots(
                *row_col,
                sharex=self.features_sharex,
                tight_layout=True,
                figsize=self.figure_size,
            )
        else:
            axs = given_axs
        for traj_idx in trajs:
            x = self._trajs[
                :, traj_idx :: self._trajs_info.trajs_num
            ]  # 从mixed中采样一个轨迹的点
            # 画出所有按index指定的features
            end_index = indexes[-1]
            for index in indexes:
                if self.features_scatters[0] is not None:
                    axs[index].scatter(
                        t[start_point:end_point],
                        x[index, start_point:end_point],
                        marker=self.features_scatters[index][0],
                        color=self.features_scatters[index][1],
                        alpha=0.3,
                    )
                else:
                    axs[index].plot(
                        t[start_point:end_point],
                        x[index, start_point:end_point],
                        self.features_lines[index],
                        alpha=0.3,
                        label=self.features_self_labels[index],
                    )
                    axs[index].legend(loc="best")

                if index != end_index:
                    axs[index].set(ylabel=self.features_axis_labels[index])
                else:
                    axs[index].set(
                        ylabel=self.features_axis_labels[index], xlabel=self.time_label
                    )
                if self.features_sharetitle is None:
                    axs[index].set(title=self.features_titles[index])
        if self.features_sharetitle is not None:
            axs[0].set(title=self.features_sharetitle)

        if self.save_path is not None:
            plt.savefig(self.save_path)
        if return_axs:
            return axs
        else:
            self.show(self.plt_pause)

    def plot_2D_features(
        self,
        points: tuple,
        trajs: tuple,
        indexes: tuple,
        fmt: str = "-ok",
        title: str = None,
        given_axs=None,
        return_axs=None,
    ):
        trajs_num = self._trajs_info.trajs_num
        start = points[0]
        end = points[1]
        start_bias = int(start * trajs_num)
        end_index = int((end + 1) * trajs_num)
        # Visualize first 100 steps of the training data
        if given_axs:
            axs = given_axs
        else:
            fig, axs = plt.subplots(1, 1, tight_layout=True, figsize=(4, 4))
        for traj_idx in trajs:
            start_index = int(traj_idx + start_bias)
            axs.plot(
                self._trajs[indexes[0], start_index:end_index:trajs_num],
                self._trajs[indexes[1], start_index:end_index:trajs_num],
                self._trajs_lines[traj_idx],
                markersize=self._trajs_markersize[traj_idx],
                label=self._trajs_labels[traj_idx],
            )
        # 仅在不给定axs时创建轴和标题
        if not given_axs:
            axs.set(
                ylabel=self.features_axis_labels[indexes[1]],
                xlabel=self.features_axis_labels[indexes[0]],
            )
            if title is None:
                title = "training data. num traj = {}, max time steps = {}".format(
                    len(trajs), points[1] - points[0]
                )
            axs.set_title(title)
        if return_axs:
            return axs
        else:
            axs.legend(loc="best")
            self.show(self.plt_pause)

    @staticmethod
    def draw_trajs(
        trajs: np.ndarray,
        each_points_num: np.ndarray,
        max_points_num: int,
        features_num: int,
        title: str = None,
        save_path: str = None,
        show: bool = True,
    ):
        """画轨迹图"""
        import matplotlib.pyplot as plt

        trajs_num = len(each_points_num)
        if features_num == 1:
            time_trajs = np.full((int(max_points_num), trajs_num), np.nan)
        else:
            time_trajs = np.full((int(max_points_num), trajs_num, features_num), np.nan)
        base = 0
        for i in range(trajs_num):
            points_num = int(each_points_num[i])
            time_trajs[:points_num, i] = trajs[base : base + points_num, i]
            base += points_num
        if features_num == 1:
            plt.plot(time_trajs)
        else:
            for i in range(features_num):
                plt.plot(time_trajs[:, :, i])
        if title is not None:
            plt.title(title)
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def draw_traj(
        traj: np.ndarray,
        title: str = None,
        save_path: str = None,
        show: bool = True,
    ):
        """画轨迹图"""
        import matplotlib.pyplot as plt

        if len(traj.shape) == 1:
            plt.plot(traj)
        else:
            for i in range(traj.shape[1]):
                plt.plot(traj[:, i])
        if title is not None:
            plt.title(title)
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def draw_trajs_from_time_trajs(
        time_trajs: np.ndarray,
        title: str = None,
        save_path: str = None,
        show: bool = True,
    ):
        """画轨迹图"""
        import matplotlib.pyplot as plt

        trajs = TrajTools.traj_times_from_time_trajs

    @property
    def trajs_labels(self):
        return self._trajs_labels

    @trajs_labels.setter
    def trajs_labels(self, labels: Union[str, tuple]):
        if isinstance(labels, str):
            self._trajs_labels = tuple(
                [labels] + [None] * (self._trajs_info.trajs_num - 1)
            )
        else:
            self._trajs_labels = labels

    @property
    def trajs_lines(self):
        return self._trajs_lines

    @trajs_lines.setter
    def trajs_lines(self, lines: Union[str, tuple]):
        if isinstance(lines, str):
            self._trajs_lines = tuple([lines] * self._trajs_info.trajs_num)
        else:
            self._trajs_lines = lines

    @property
    def trajs_markersize(self):
        return self._trajs_markersize

    @trajs_markersize.setter
    def trajs_markersize(self, markersize: Union[int, tuple]):
        if isinstance(markersize, int):
            self._trajs_markersize = tuple([markersize] * self._trajs_info.trajs_num)
        else:
            self._trajs_markersize = markersize
