import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Any, Set, Iterable
from . import recorder
from matplotlib import pyplot as plt
from copy import deepcopy
from datetime import datetime
import atexit


class TrajsRecorder(object):
    def __init__(self, features: List[str], path: str = None) -> None:
        """
        用于记录轨迹数据的类：
            features: 轨迹的特征种类名（每个轨迹中的所有featurs长度相同，除非被设置为非count特征）;
            path: 轨迹数据存储路径，若为None则自动根据当前时间生成;
        注：该类中的轨迹数据是按时间点组织的，即每个特征对应的轨迹应为“垂直生长（或叫向下生长）”。
        """
        self._traj = {feature: [] for feature in features}
        self._trajs = {0: deepcopy(self._traj)}
        self._features = features  # list类型（set类型可能打乱默认顺序）
        self._features_num = len(self._features)
        self._recorded = False
        self._path = path
        self._each_points_num = None
        self._not_count_features = set()
        self.feature_add_cnt = 0
        self.features_add_cnt = 0

    def add_new_features(self, features: List[str]) -> None:
        """添加新的特征种类名(需在记录数据前完成配置，否则轨迹会被刷新而丢失)；"""
        assert set(features).isdisjoint(self._features), "Features name conflict"
        self._features.extend(features)
        self._features_num = len(self._features)
        self._traj = {feature: [] for feature in features}
        self._trajs = {0: deepcopy(self._traj)}

    def set_not_count_features(self, features: Set[str]) -> None:
        """
        设置不计数的特征种类名（需在执行add前完成配置）；
        """
        features = set(features)
        if not features.issubset(self._features):
            raise ValueError("Features not in all (init) features")
        self._not_count_features |= features

    def feature_add(
        self, traj_id: int, feature: str, value: Any, all: bool = False
    ) -> None:
        """
        添加一个特征值value到指定traj_id轨迹中的feature里（自动增添轨迹ID）；
            若value为可迭代对象，则将其转换为json支持的list格式；
            若value为数值类型，则将其转换为float类型；
        all: 若为True，则将value直接赋值给feature而不是向其中添加（不自动进行类型转换）；
        每个特征对应的轨迹应为“垂直生长（或叫向下生长）”，如通过list.append()添加新的时间点的值，
        然后经np.array后会变成列数为该特征维数，行数为轨迹点数的一个二维轨迹数组；
        """
        if self._trajs.get(traj_id) is None:
            self._trajs[traj_id] = deepcopy(self._traj)
        if not all:
            if not isinstance(value, str):
                if isinstance(value, Iterable):
                    value = list(value)  # 转换为json支持的list格式
                # 排除numpy的数值类型
                elif not isinstance(value, (int, float)):
                    value = float(value)
            self._trajs[traj_id][feature].append(value)
            self.feature_add_cnt += 1
        else:
            self._trajs[traj_id][feature] = value
            self.feature_add_cnt += len(value)

    def features_add(
        self, traj_id: int, features_val: list, features_name: Optional[list] = None
    ) -> None:
        """
        添加多个特征值到指定轨迹中（自动增添轨迹ID）;
        features_val: 与features对应的特征值列表, 顺序与features一致的子集;
        始终从第一个特征开始添加，若特征值列表长度小于特征数，则多余的特征值将被忽略；
        """
        if features_name is None:
            features_name = self._features
        else:
            assert len(features_name) == len(features_val), "Features name length error"
        for i, val in enumerate(features_val):
            self.feature_add(traj_id, features_name[i], val)
        self.features_add_cnt += 1

    def check(self, trajs=None, not_counted=None) -> bool:
        """检查轨迹数据是否完整（每个轨迹中的所有计数特征是否有相同的长度；各个轨迹是否有相同的特征种类）"""
        if trajs is None:
            trajs = self._trajs
        each_points_num = np.zeros(self.trajs_num, dtype=np.int64)
        if not_counted is not None:
            self.set_not_count_features(not_counted)
        counted = set(self._features) - self._not_count_features
        first_counted = list(counted)[0]
        for i, traj in trajs.items():
            each_points_num[i] = len(traj[first_counted])
            for feature in counted:
                if traj.get(feature) is None:
                    print(f"Error: Traj {i} does not have {feature}")
                    return False
                elif len(traj[feature]) != each_points_num[i]:
                    print(f"Error: Traj {i} has different length of {feature}")
                    return False
        if len(set(each_points_num)) != 1:
            print("Note: Different trajs have different points num")
        self._each_points_num = each_points_num
        return True

    def save(
        self,
        path: str = None,
        trajs: Optional[Dict[int, Dict[str, List[Any]]]] = None,
        check: bool = False,
    ) -> bool:
        """存储轨迹数据为json文件"""
        if path is None:
            if self._path is None:
                # 获取当前系统时间
                current_time = datetime.now()
                # 格式化时间为指定的格式，精确到毫秒
                formatted_time = current_time.strftime("%Y-%m-%d-%H%M%S%f")[:-3]
                path = f"trajs_{formatted_time}.json"
            else:
                path = self._path
        if trajs is None:
            trajs = self._trajs
            # 保存非内部轨迹数据不修改内部记录状态
            self._recorded = True
        if check:
            if not self.check(trajs):
                return False
        recorder.json_process(path, write=trajs)
        return True

    def auto_save(self):
        """
        若未手动存储，则在程序退出时尝试自动存储轨迹数据
        （非法退出时也无法保证可以自动记录）
        """
        atexit.register(lambda: self.save() if not self._recorded else None)

    @property
    def trajs(self):
        return self._trajs

    @property
    def trajs_num(self):
        return len(self._trajs)

    @property
    def features(self):
        return self._features

    @property
    def features_num(self):
        """所有特征种类数"""
        return self._features_num

    @property
    def each_points_num(self):
        """每个轨迹的点数（根据第一个计数特征算）"""
        if self._each_points_num is None:
            self._each_points_num = np.zeros(self.trajs_num, dtype=np.int64)
        counted = set(self._features) - self._not_count_features
        first_counted = list(counted)[0]
        for i, traj in self._trajs.items():
            self._each_points_num[i] = len(traj[first_counted])
        return self._each_points_num

    @property
    def each_features_dim(self):
        """每个特征的维度（仅考虑计数特征）"""
        counted = set(self._features) - self._not_count_features
        return [len(self._trajs[0][feature]) for feature in counted]

    def __getitem__(self, index):
        return self._trajs[index]


class TrajInfo(object):
    def __init__(
        self,
        trajs_num: int,
        each_points_num: Union[int, np.ndarray],
        max_points_num: Optional[int],
        points_dim: int,
    ):
        """
        each_points_num: 每个轨迹的点数；
            为int时，认为轨迹点数相同自动计算；
            为np.ndarray时，每个轨迹点数可以不同；
            为None时，认为轨迹点数相同，自动根据max_points_num计算（需给定）；
        max_points_num: 每个轨迹的最大点数；为None时，自动根据each_points_num计算，当轨迹数很多时可能会有较大的计算开销，因此这里允许直接指定；
        make_trajs: series_v, series_h, mixed_v, mixed_h, time_trajs, traj_times；
            若不为None，则根据trajs_num, each_points_num, max_points_num, features_num构造nan轨迹数据；
            构造的轨迹数据可以通过get_trajs()获取。若初始化未构造或构造的类型和目标类型不一致，则该函数将先完成（重新）构造；
        """
        self.trajs_num = trajs_num
        if isinstance(each_points_num, int):
            each_points_num = np.full(trajs_num, each_points_num)
        if each_points_num is None:
            assert max_points_num is not None, "max_points_num must be given"
            each_points_num = np.full(trajs_num, max_points_num)
        self.each_points_num = each_points_num
        if max_points_num is None:
            max_points_num = np.max(each_points_num)
        self.max_points_num = int(max_points_num)
        self.points_dim = points_dim
        self._trajs = None
        self._type = None

    def __eq__(self, other):
        if isinstance(other, TrajInfo):
            return (
                self.trajs_num == other.trajs_num
                and np.allclose(self.each_points_num, other.each_points_num)
                and self.max_points_num == other.max_points_num
                and self.points_dim == other.points_dim
            )
        return False

    @classmethod
    def consruct(
        cls,
        trajs: np.ndarray,
        type: str,
        trajs_num: int = None,
        each_points_num: Union[str, np.ndarray] = None,
        max_points_num: int = None,
        points_dim: int = None,
        log: bool = False,
    ):
        """
        根据轨迹数据构造TrajInfo;
        type: series_v, series_h, mixed_v, mixed_h, time_trajs, traj_times；
        each_points_num: None, "equal", np.ndarray；
        """
        if "time" in type:
            if type == "traj_times":
                trajs = trajs.T
            trajs_num = trajs.shape[1] if trajs_num is None else trajs_num
            max_points_num = (
                trajs.shape[0] if max_points_num is None else max_points_num
            )
            points_dim = trajs.shape[2] if points_dim is None else points_dim
            if each_points_num == "equal":
                each_points_num = np.full(trajs_num, max_points_num)
            elif each_points_num is not None:
                each_points_num = np.array(each_points_num)
            else:
                each_points_num = np.zeros(trajs_num)
                for i in range(trajs_num):
                    each_points_num[i] = len(
                        np.delete(trajs[:, i, 0], np.where(np.isnan(trajs[:, i, 0])))
                    )
        else:
            if "v" in type:
                trajs = trajs.T
            shape = trajs.shape
            points_dim = shape[0] if points_dim is None else points_dim
            trajs_lenth = shape[1]
            if isinstance(each_points_num, np.ndarray):
                trajs_num = len(each_points_num) if trajs_num is None else trajs_num
                max_points_num = (
                    np.max(each_points_num)
                    if max_points_num is None
                    else max_points_num
                )
            else:
                if max_points_num is None:
                    assert trajs_num is not None, "trajs_num must be given"
                    max_points_num = trajs_lenth // trajs_num
                elif trajs_num is None:
                    assert max_points_num is not None, "max_points_num must be given"
                    trajs_num = trajs_lenth // max_points_num
                if each_points_num == "equal" or trajs_num == 1:
                    each_points_num = np.full(trajs_num, max_points_num)
                else:
                    each_points_num = np.zeros(trajs_num)
                    for i in range(trajs_num):
                        traj = TrajTools.get_traj_from_mixed_trajs(trajs, i, trajs_num)
                        each_points_num[i] = int(
                            max_points_num
                            - len(np.where(np.any(np.isnan(traj), axis=0))[0])
                        )
        if log:
            print(
                f"trajs_num: {trajs_num}, each_points_num: {each_points_num}, max_points_num: {max_points_num}, points_dim: {points_dim}"
            )
        return cls(trajs_num, each_points_num, max_points_num, points_dim)

    def _make_trajs(self, type: str = "mixed_h"):
        """
        根据TrajInfo构造nan轨迹数据；
        type: series_v, series_h, mixed_v, mixed_h, time_trajs, traj_times；
        """
        self._type = type
        if "time" in type:
            trajs = np.full(
                (int(self.max_points_num), self.trajs_num, self.points_dim), np.nan
            )
        elif "mixed" in type:
            trajs = np.full(
                (self.points_dim, int(self.max_points_num * self.trajs_num)), np.nan
            )
        elif "series" in type:
            trajs = np.full(
                (self.points_dim, int(np.sum(self.each_points_num))), np.nan
            )

        if "v" in type or type == "traj_times":
            trajs = trajs.T

        self._trajs = trajs

    def get_trajs(self, type: str = "mixed_h") -> np.ndarray:
        """
        根据TrajInfo构造nan轨迹数据并返回一个拷贝；
        type: series_v, series_h, mixed_v, mixed_h, time_trajs, traj_times；
        若已经构造的类型和给定目标类型不一致，则该函数将重新构造轨迹；
        """
        if self._trajs is None or self._type != type:
            self._make_trajs(type)
        return deepcopy(self._trajs)


class TrajTools(object):
    @staticmethod
    def construct_trajs(
        trajs: dict,
        feature: str,
        series_type=None,
        mixed_type=None,
        show_info=False,
    ) -> Tuple[Union[tuple, np.ndarray], TrajInfo]:
        """
        按时间(行坐标)组合多组轨迹数据（列坐标；一般是由TrajsRecorder记录的）；且有额外两种拼接功能：
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
        # 基本轨迹信息检查
        assert (
            trajs.get("0") is not None
        ), "Each trajectory must have a continous str(int) key and the minimum key is '0'"
        trajs_num = len(trajs)  # 轨迹数
        trajs_index_max = trajs_num - 1
        end_key = list(trajs.keys())[-1]
        assert (
            str(trajs_index_max) == end_key
        ), f"Trajectory number must be continous: 轨迹数为{trajs_num}，但是end_key={end_key}"
        # 每个轨迹的点数以及最大点数
        each_points_num = np.zeros(trajs_num)
        for i in range(trajs_num):
            each_points_num[i] = len(trajs[str(i)][feature])
        max_points_num = np.max(each_points_num)
        try:
            points_dim = len(trajs["0"][feature][0])  # 特征维度
        except TypeError:
            points_dim = 1
        # 按时间组合（时间以最大点数轨迹为准）
        if points_dim == 1:
            time_trajs = np.full((int(max_points_num), trajs_num), np.nan)
        else:
            time_trajs = np.full((int(max_points_num), trajs_num, points_dim), np.nan)
        if series_type is not None:
            all_points_num = int(sum(each_points_num))
            # default grow vertically
            trajs_series = np.zeros((all_points_num, points_dim))
        else:
            trajs_series = None
        if mixed_type is not None:
            trajs_mixed = np.full((int(max_points_num * trajs_num), points_dim), np.nan)
        else:
            trajs_mixed = None
        current_points_num = 0
        for i in range(trajs_num):
            for j in range(int(each_points_num[i])):
                time_trajs[j, i] = trajs[str(i)][feature][j]
                if mixed_type is not None:
                    trajs_mixed[int(i + j * trajs_num), :] = time_trajs[j, i]
            if series_type is not None:
                points_num = int(each_points_num[i])
                trajs_series[
                    current_points_num : current_points_num + points_num, :
                ] = time_trajs[:, i][:points_num, :]
                current_points_num += points_num
        info = TrajInfo(trajs_num, each_points_num, max_points_num, points_dim)
        if show_info:
            print(f"{feature} trajectories number: ", trajs_num)
            print(f"{feature} each points number: ", each_points_num)
            print(f"{feature} points dim: ", points_dim)
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
        points_dim: int = None,
        grow_type: str = "h",
    ) -> np.ndarray:
        """将按轨迹串行拼接的轨迹数据还原为按时间拼接的轨迹数据"""
        if grow_type != "h":
            trajs_series = trajs_series.T
        if points_dim is None:
            points_dim = trajs_series.shape[0]
        trajs_mixed = np.full((points_dim, (int(max_points_num * trajs_num))), np.nan)
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

    @classmethod
    def to_mixed_trajs(
        cls,
        trajs: np.ndarray,
        info: TrajInfo,
        in_type: str,
        out_type: str = "h",
    ) -> np.ndarray:
        """
        将轨迹数据从in_type转换为mixed_out_type；
        in_type: series_v, series_h, mixed_v, mixed_h, time_trajs, traj_times；
        out_type: v, h；
        """
        if "v" in in_type or in_type == "traj_times":
            # 方向统一
            trajs = trajs.T
        # 统一转换成mixed类型轨迹
        if "series" in in_type:
            trajs = TrajTools.mixed_trajs_from_series_trajs(
                trajs,
                info.each_points_num,
                info.trajs_num,
                info.max_points_num,
                info.points_dim,
            )
        elif "time" in in_type:
            trajs = TrajTools.mixed_trajs_from_time_trajs(
                trajs, info.trajs_num, info.max_points_num
            )
        if out_type == "h":
            return trajs
        else:
            return trajs.T

    @staticmethod
    def delete_nan_from_time_trajs(time_trajs: np.ndarray) -> np.ndarray:
        """删除按时间组合的轨迹数据中的nan"""
        return time_trajs[~np.isnan(time_trajs).all(axis=1)]

    @staticmethod
    def delete_element_by_traj(
        trajs_series: np.ndarray,
        index: int,
        trajs_info: TrajInfo,
        grow_type="h",
    ) -> Tuple[np.ndarray, TrajInfo]:
        """删除按轨迹串行拼接的轨迹数据中的某个时间位置的轨迹点（通常是第一个和最后一个）"""
        if index < -trajs_info.max_points_num or index >= trajs_info.max_points_num:
            return trajs_series, trajs_info
        else:
            base = 0
            axis = 0 if grow_type == "v" else 1
            trajs_info_new = deepcopy(trajs_info)
            each_points_num = trajs_info.each_points_num
            for i, num in enumerate(each_points_num):
                index_posi = index if index >= 0 else num + index
                # 超出范围的不删除
                if abs(index_posi) >= num:
                    base += num - 1
                    continue
                trajs_info_new.each_points_num[i] -= 1
                if trajs_info_new.each_points_num[i] == 0:
                    trajs_info_new.trajs_num -= 1
                index_new = int(index_posi + base)
                trajs_series = np.delete(trajs_series, index_new, axis=axis)
                base += num - 1
            trajs_info_new.max_points_num -= 1
            trajs_info_new.each_points_num = np.delete(
                trajs_info_new.each_points_num, np.where(each_points_num == 0)
            )
            return trajs_series, trajs_info_new

    @staticmethod
    def delete_mixed_at_time(
        trajs_mixed: np.ndarray,
        index: int,
        trajs_info: TrajInfo,
        grow_type: str = "h",
    ) -> Tuple[np.ndarray, TrajInfo]:
        """
        删除按时间拼接的轨迹数据中的某个位置的轨迹（通常是第一个和最后一个）;
        index为负数时，表示从后往前数的第几个轨迹，此时必须给定max_points_num；
        删除后的轨迹的最大轨迹点数max_points_num减少1；
        每个轨迹的点数each_points_num若本小于index则不变；
        若each_points_num中某个位置为0，则删除该位置对应的轨迹，导致轨迹数减少（移除全部单点轨迹）；
        """
        axis = 0 if grow_type == "v" else 1
        if index < 0:
            index = trajs_info.max_points_num + index
        start = int(index * trajs_info.trajs_num)
        end = int((index + 1) * trajs_info.trajs_num)
        new_trajs = np.delete(
            trajs_mixed,
            slice(start, end, 1),
            axis=axis,
        )
        # 更新trajs_info
        trajs_info_new = deepcopy(trajs_info)
        trajs_info_new.max_points_num -= 1
        each_points_num = trajs_info_new.each_points_num.copy()
        each_points_num[each_points_num > index] -= 1
        trajs_slices = np.where(each_points_num <= 0)[0]
        len_slices = len(trajs_slices)
        if len_slices > 0:
            each_points_num = np.delete(each_points_num, trajs_slices)
            trajs_len = int(trajs_info_new.trajs_num * trajs_info_new.max_points_num)
            trajs_num = trajs_info_new.trajs_num
            for i in trajs_slices:
                new_trajs = np.delete(
                    new_trajs, slice(i, trajs_len, trajs_num), axis=axis
                )
                trajs_len -= 1
            trajs_info_new.each_points_num = each_points_num
            trajs_info_new.trajs_num -= len_slices

        return new_trajs, trajs_info_new

    @staticmethod
    def delete_mixed_at_traj(
        trajs_mixed: np.ndarray,
        index: int,
        trajs_info: TrajInfo,
        grow_type: str = "h",
    ) -> Tuple[np.ndarray, TrajInfo]:
        """
        从按时间拼接的轨迹数据中删除某个轨迹；
        index为负数时，表示从后往前数的第几个轨迹；
        删除后轨迹数减少1；若删除的轨迹点数等于唯一的最大轨迹点数，则最大点数max_points_num也减少至第二大；
        """
        trajs_num = trajs_info.trajs_num
        trajs_len = int(trajs_num * trajs_info.max_points_num)
        axis = 0 if grow_type == "v" else 1
        if index < 0:
            index = trajs_num + index
        trajs_new = np.delete(
            trajs_mixed, slice(index, trajs_len, trajs_num), axis=axis
        )
        trajs_info_new = deepcopy(trajs_info)
        trajs_info_new.each_points_num = np.delete(
            trajs_info_new.each_points_num, index
        )
        trajs_info_new.trajs_num -= 1
        # 若删除的轨迹点数等于唯一的最大轨迹点数，需要更新max_points_num
        points_num = trajs_info.each_points_num[index]
        max_points_num = trajs_info.max_points_num
        if points_num == max_points_num:
            # 确保最大点唯一性
            if len(np.where(trajs_info.each_points_num == max_points_num)) == 1:
                trajs_num = trajs_info_new.trajs_num
                trajs_len = int(trajs_num * trajs_info_new.max_points_num)
                # 还需删除轨迹的最后一组点
                trajs_info_new.max_points_num = np.max(trajs_info_new.each_points_num)
                delta_max = max_points_num - trajs_info_new.max_points_num
                trajs_new = np.delete(
                    trajs_new,
                    slice(int(trajs_len - delta_max * trajs_num), trajs_len),
                    axis=axis,
                )
        return trajs_new, trajs_info_new

    @staticmethod
    def get_sub_mixed_trajs(
        trajs_mixed: np.ndarray,
        trajs_info: TrajInfo,
        points: tuple,
        trajs: tuple,
        grow_type: str = "h",
    ):
        """
        从按时间拼接的轨迹数据中获取某个子集;
        points: (start_point, end_point)；不包括end_point；
        trajs: indexes，最后生成的矩阵的轨迹顺序将按此中顺序；
        """
        start_point = points[0]
        end_point = points[1]
        each_points_num = trajs_info.each_points_num[list(trajs)]
        max_points_num = np.max(each_points_num)
        if end_point > max_points_num:
            print(
                f"end_point {end_point} is larger than max_points_num {max_points_num} of all the selected trajectories {trajs}, so it will be set to {max_points_num}"
            )
            end_point = max_points_num

        # 新信息
        each_points_num[each_points_num > end_point] = end_point
        each_points_num -= start_point
        max_points_num = int(end_point - start_point)
        points_dim = trajs_info.points_dim
        # 删除点数为0的轨迹
        slices = each_points_num > 0
        trajs = np.array(trajs)[slices].tolist()
        each_points_num = each_points_num[slices]
        trajs_num = len(trajs)

        if grow_type != "h":
            trajs_mixed = trajs_mixed.T
        # 初始子矩阵
        sub_trajs = np.zeros((points_dim, int(trajs_num * max_points_num)))
        start_bias = start_point * trajs_info.trajs_num
        end_bias = end_point * trajs_info.trajs_num
        for i, index in enumerate(trajs):
            base = int(index + start_bias)
            end = index + end_bias
            sub_trajs[:, i::trajs_num] = trajs_mixed[
                :, base : end : trajs_info.trajs_num
            ]
        return sub_trajs, TrajInfo(
            trajs_num, each_points_num, max_points_num, points_dim
        )

    @classmethod
    def get_sub_series_trajs(
        cls,
        trajs_series: np.ndarray,
        trajs_info: TrajInfo,
        points: tuple,
        trajs: tuple,
        grow_type: str = "h",
    ):
        """
        从按轨迹串行拼接的轨迹数据中获取某个子集;
        points: (start_point, end_point)；不包括end_point；也可以嵌套，如((0, 1), (0, 1))，为相应轨迹指定不同的时间段；
        trajs: indexes，最后生成的矩阵的轨迹顺序将按此中顺序；
        """
        new_each_points_num = trajs_info.each_points_num[list(trajs)]
        new_each_points_num[new_each_points_num > points[1]] = points[1]
        new_each_points_num -= points[0]
        trajs = np.array(trajs)[new_each_points_num > 0].tolist()
        new_each_points_num = new_each_points_num[new_each_points_num > 0]
        new_series_trajs = np.zeros(
            (trajs_info.points_dim, int(np.sum(new_each_points_num)))
        )
        # 获取全部轨迹
        base = 0
        each = trajs_info.each_points_num
        cnt = 0
        points_flag = True if isinstance(points[0], int) else False
        for index in trajs:
            if points_flag:
                points_ = points
            else:
                points_ = points[cnt]
            base = int(np.sum(each[:index]))
            end = int(base + each[index])
            new_series_trajs[:, base:end] = TrajTools.get_traj_from_series_trajs(
                trajs_series,
                each,
                index,
                trajs_info.trajs_num,
                grow_type=grow_type,
            )[:, points_[0] : points_[1]]
            cnt += 1
        return new_series_trajs, TrajInfo(
            len(trajs),
            new_each_points_num,
            np.max(new_each_points_num),
            trajs_info.points_dim,
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
        """从按时间（轨迹点顺序）拼接的轨迹数据中获取某个轨迹"""
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
        """从按轨迹依次串行拼接的轨迹数据中获取某个轨迹"""
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
    def has_nan(trajs: np.ndarray) -> bool:
        """检查数组中是否有nan"""
        return np.any(np.isnan(trajs))

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

    @staticmethod
    def normalize_trajs(
        trajs: np.ndarray, grow_type: str = "h", has_nan: bool = False
    ) -> np.ndarray:
        """对数组进行归一化（）"""
        if has_nan:
            trajs = deepcopy(trajs)
            nan_mask = np.isnan(trajs)
            trajs[nan_mask] = 0
        type2axis = {"h": 1, "v": 0}
        trajs_max = np.max(np.abs(trajs), axis=type2axis[grow_type], keepdims=True)
        trajs_max[trajs_max == 0] = 1
        trajs = trajs / trajs_max
        if has_nan:
            trajs[nan_mask] = np.nan
        return trajs

    @staticmethod
    def append_point(
        trajs: np.ndarray,
        point: np.ndarray,
        grow_type: str = "h",
        del_fist: bool = False,
    ) -> np.ndarray:
        """
        将点数据(N,)添加到轨迹数据中；
        del_fist: 是否删除第一个点；
        """
        if grow_type == "h":
            grow_type = 1
        else:
            grow_type = 0
        trajs = np.append(trajs, [point], axis=grow_type)
        if del_fist:
            if grow_type == 0:
                trajs = trajs[:-1]
            else:
                trajs = trajs[:, :-1]
        return trajs

    @staticmethod
    def insert_point(
        trajs: np.ndarray,
        index: int,
        point: np.ndarray,
        grow_type: str = "h",
        del_last: bool = False,
    ) -> np.ndarray:
        """
        将点数据(N,)插入到轨迹数据中的指定位置；
        index: 插入位置；
        del_last: 是否删除最后一个点；
        """
        if grow_type == "h":
            grow_type = 1
        else:
            grow_type = 0
        trajs = np.insert(trajs, index, [point], axis=grow_type)
        if del_last:
            if grow_type == 0:
                trajs = trajs[:-1]
            else:
                trajs = trajs[:, :-1]
        return trajs

    @staticmethod
    def append_traj(
        trajs: np.ndarray, traj: np.ndarray, grow_type: str = "h"
    ) -> np.ndarray:
        """将轨迹数据(N, M)添加到轨迹数据中"""
        if grow_type == "h":
            return np.append(trajs, traj, axis=1)
        else:
            return np.append(trajs, traj, axis=0)

    @staticmethod
    def insert_traj(
        trajs: np.ndarray, index: int, traj: np.ndarray, grow_type: str = "h"
    ) -> np.ndarray:
        """将轨迹数据(N, M)插入到轨迹数据中"""
        if grow_type == "h":
            return np.insert(trajs, index, traj, axis=1)
        else:
            return np.insert(trajs, index, traj, axis=0)

    @staticmethod
    def repeat_points(
        trajs: np.ndarray, times: int, grow_type: str = "h"
    ) -> np.ndarray:
        """将轨迹数据中的每个轨迹点重复times次"""
        if grow_type == "h":
            return np.repeat(trajs, times, axis=1)
        else:
            return np.repeat(trajs, times, axis=0)

    @staticmethod
    def repeat_trajs(trajs: np.ndarray, times: int, grow_type: str = "h") -> np.ndarray:
        """将轨迹数据重复times次"""
        if grow_type == "h":
            return np.tile(trajs, (1, times))
        else:
            return np.tile(trajs, (times, 1))


class Trajer(object):
    traj_tool = TrajTools

    # TODO
    def __init__(self, type: str = "s") -> None:
        """
        集成了TrajTools的更方便操作的轨迹类.
        通过type指定具体轨迹的类型，后续所有操作均只针对该类型。

        :param type: 轨迹的类型
        :returns: None
        :raises: None
        """
        self._type = type

    # 获得子轨迹
    def sub():
        pass


class TrajsPainter(object):
    traj_tool = TrajTools

    def __init__(
        self, trajs: np.ndarray = None, info: TrajInfo = None, type: str = "mixed_h"
    ) -> None:
        """给定轨迹及其对应的类型（目前支持series_v、series_h、mixed_v、mixed_h、time_trajs、traj_times）"""
        self._inited = False
        if trajs is not None and info is not None:
            self.update_trajs(trajs, info, type)
            # 特征参数配置，一般用于绘制feature-time图
            self.set_default()

    def set_default(self):
        """设置默认的绘图参数"""
        # 在绘图时认为points的每个维度都是一个特征
        features_num = self._trajs_info.points_dim
        self.features_axis_labels = tuple([rf"$x_{i}$" for i in range(features_num)])
        self._features_lines = ("k",) * features_num
        self.features_scatters = (None,) * features_num
        self.features_sharex = True
        self.features_sharetitle = "Features Trajectories"
        self.features_titles = ("Features Trajs",) * features_num
        self._features_self_labels = (None,) * features_num
        self.time_label = r"$t$"
        # 轨迹参数配置，一般用于绘制2Dfeatures轨迹图
        self.trajs_lines = "-ok"
        self.trajs_labels = r"$trajectories_1$"
        self.trajs_markersize = 5
        # 通用绘图参数配置
        self.figure_size_t = (12, 4)
        self.figure_size_2D = (6, 6)
        self.save_path = None
        self.plt_pause = 0
        # 初始化完成
        self._inited = True

    def get_trajs_and_info(self) -> Tuple[np.ndarray, TrajInfo]:
        return self._trajs, self._trajs_info

    def update_trajs(
        self,
        trajs: np.ndarray,
        info: TrajInfo,
        type: str = "mixed_h",
        reset: bool = False,
    ):
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
                info.points_dim,
            )
        elif "time" in type:
            trajs = TrajTools.mixed_trajs_from_time_trajs(
                trajs, info.trajs_num, info.max_points_num
            )
        self._trajs_info = info
        self._trajs = trajs
        if not self._inited or reset:
            self.set_default()

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
        given_axis=None,
        return_axis=False,
    ):
        """points是连贯的点，而trajs和indexes是指定的序号，可以不连贯"""
        start_point = points[0]
        end_point = points[1]
        assert (
            end_point <= self._trajs_info.max_points_num
        ), f"end_point {end_point} is larger than max_points_num {self._trajs_info.max_points_num - 1}"
        # Time vector
        t = np.arange(0, (end_point - start_point) * dT, dT)
        # Visualize start->end steps of the training data
        if given_axis is None:
            if row_col is None:
                row_col = (len(indexes), 1)
            fig, axis = plt.subplots(
                *row_col,
                sharex=self.features_sharex,
                tight_layout=True,
                figsize=self.figure_size_t,
            )
            if row_col[0] == 1:
                axis = (axis,)
        else:
            axis = given_axis
        for traj_idx in trajs:
            x = self._trajs[
                :, traj_idx :: self._trajs_info.trajs_num
            ]  # 从mixed中采样一个轨迹的点
            # 画出所有按index指定的features
            end_index = indexes[-1]
            for index in indexes:
                if self.features_scatters[0] is not None:
                    axis[index].scatter(
                        t[start_point:end_point],
                        x[index, start_point:end_point],
                        marker=self.features_scatters[index][0],
                        color=self.features_scatters[index][1],
                        alpha=0.3,
                    )
                else:
                    axis[index].plot(
                        t[start_point:end_point],
                        x[index, start_point:end_point],
                        self._features_lines[index],
                        alpha=0.3,
                        label=self._features_self_labels[index],
                    )
                    if self._features_self_labels[index] is not None:
                        axis[index].legend(loc="best")

                if index != end_index:
                    axis[index].set(ylabel=self.features_axis_labels[index])
                else:
                    axis[index].set(
                        ylabel=self.features_axis_labels[index], xlabel=self.time_label
                    )
                if self.features_sharetitle is None:
                    axis[index].set(title=self.features_titles[index])
        if self.features_sharetitle is not None:
            axis[0].set(title=self.features_sharetitle)

        if self.save_path is not None:
            plt.savefig(self.save_path)
        if return_axis:
            return axis
        else:
            self.show(self.plt_pause)

    def plot_2D_features(
        self,
        points: tuple,
        trajs: tuple,
        indexes: tuple,
        title: str = None,
        given_axis=None,
        return_axis=False,
    ):
        trajs_num = self._trajs_info.trajs_num
        start = points[0]
        end = points[1]
        start_bias = int(start * trajs_num)
        end_index = int(end * trajs_num)
        if given_axis:
            axis = given_axis
        else:
            fig, axis = plt.subplots(
                1, 1, tight_layout=True, figsize=self.figure_size_2D
            )
        for traj_idx in trajs:
            start_index = int(traj_idx + start_bias)
            axis.plot(
                self._trajs[indexes[0], start_index:end_index:trajs_num],
                self._trajs[indexes[1], start_index:end_index:trajs_num],
                self._trajs_lines[traj_idx],
                markersize=self._trajs_markersize[traj_idx],
                label=self._trajs_labels[traj_idx],
            )
        # 仅在不给定axs时创建轴和标题
        if not given_axis:
            axis.set(
                ylabel=self.features_axis_labels[indexes[1]],
                xlabel=self.features_axis_labels[indexes[0]],
            )
            if title is None:
                title = "trajs_num = {}, points_num = {}".format(
                    len(trajs), points[1] - points[0]
                )
            axis.set_title(title)
        if return_axis:
            return axis
        else:
            axis.legend(loc="best")
            self.show(self.plt_pause)

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

    @property
    def features_lines(self):
        return self._features_lines

    @features_lines.setter
    def features_lines(self, lines: Union[str, tuple]):
        if isinstance(lines, str):
            self._features_lines = tuple([lines] * self._trajs_info.points_dim)
        else:
            self._features_lines = lines

    @property
    def features_self_labels(self):
        return self._features_self_labels

    @features_self_labels.setter
    def features_self_labels(self, labels: Union[str, tuple]):
        if isinstance(labels, str):
            self._features_self_labels = tuple([labels] * self._trajs_info.points_dim)
        else:
            self._features_self_labels = labels
