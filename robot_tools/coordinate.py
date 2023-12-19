from . import transformations
import numpy as np
from typing import Union


class CoordinateTools(object):
    """坐标系转换工具类:
    position: 位置，三维向量
    orientation: 姿态，四元数(xyzw)或欧拉角(rpy, sxyz)
    pose: 位姿，位置和姿态的组合;tuple/list, (position, orientation)
    error: target - current；姿态各轴角度误差为向量夹角，即范围为[0,pi]
    """

    @staticmethod
    def transform_as_matrix(
        position: np.ndarray, orientation: np.ndarray
    ) -> np.ndarray:
        """将相对位姿转换为变换矩阵"""
        if len(orientation) == 3:
            trans_q = transformations.euler_matrix(*orientation)
        elif len(orientation) == 4:
            trans_q = transformations.quaternion_matrix(orientation)
        trans_t = transformations.translation_matrix(position)
        trans_TF = np.matmul(trans_t, trans_q)
        return trans_TF

    @classmethod
    def to_robot_coordinate(
        cls, target_in_world: tuple, robot_in_world: tuple
    ) -> tuple:
        """目标在世界坐标系下的位姿转换为在机器人坐标系下的位姿"""
        target_in_world = cls.transform_as_matrix(*target_in_world)
        robot_in_world = cls.transform_as_matrix(*robot_in_world)
        target_in_robot = np.matmul(np.linalg.inv(robot_in_world), target_in_world)
        t_scale, t_shear, t_angles, t_trans, t_persp = transformations.decompose_matrix(
            target_in_robot
        )
        return np.array(t_trans, dtype=np.float64), np.array(t_angles, dtype=np.float64)

    @classmethod
    def to_world_coordinate(
        cls, target_in_robot: tuple, robot_in_world: tuple
    ) -> tuple:
        """目标在机器人坐标系下的位姿转换为在世界坐标系下的位姿"""
        target_in_robot = cls.transform_as_matrix(*target_in_robot)
        robot_in_world = cls.transform_as_matrix(*robot_in_world)
        target_in_world = np.matmul(robot_in_world, target_in_robot)
        t_scale, t_shear, t_angles, t_trans, t_persp = transformations.decompose_matrix(
            target_in_world
        )
        return np.array(t_trans, dtype=np.float64), np.array(t_angles, dtype=np.float64)

    @staticmethod
    def get_radial_distance(position: np.ndarray) -> float:
        """获取目标位置在标准球坐标系中的径向距离"""
        return np.linalg.norm(position)

    @staticmethod
    def get_axis_error(target_vector: np.ndarray, base_vector: np.ndarray):
        """计算两个向量在各分方向上的误差"""
        return target_vector - base_vector

    @classmethod
    def get_spherical(cls, position: np.ndarray) -> tuple:
        """获取目标位置在标准球坐标系中的径向距离、极角和方位角"""
        radial_distance = cls.get_radial_distance(
            position
        )  # 径向距离（radial distance），范围[0,inf)
        thita = np.arccos(
            position[2] / radial_distance
        )  # 极角（polar/inclination/zenith angle），与z轴的夹角，范围[0,pi]
        fai = np.arctan2(
            position[1], position[0]
        )  # 方位角（azimuth angle），与x轴的夹角，范围[-pi,pi]
        return radial_distance, thita, fai

    @staticmethod
    def get_position_distance(
        target_position: np.ndarray, current_position: np.ndarray
    ) -> float:
        """获取两个位置点之间的距离(欧式距离/向量差的二范数)"""
        return np.linalg.norm(target_position - current_position)

    @classmethod
    def get_orientation_distance(
        cls, target_orientation: np.ndarray, current_orientation: np.ndarray
    ) -> float:
        """获取两个姿态点之间的距离(各轴所代表的角度向量的夹角构成向量的二范数)"""
        target_orientation = cls.to_euler(target_orientation)
        current_orientation = cls.to_euler(current_orientation)
        raw_error = cls.get_axis_error(target_orientation, current_orientation)
        good_error = cls.change_to_pi_scope(raw_error)
        return np.linalg.norm(good_error)

    @classmethod
    def get_pose_distance(
        cls, target_pose: np.ndarray, current_pose: np.ndarray
    ) -> np.ndarray:
        """得到机器人当前位姿与目标位姿的误差（位置计算欧氏距离；姿态每个轴分别计算向量夹角再得到总体偏差）"""
        position_dis = cls.get_position_distance(target_pose[0], current_pose[0])
        orientation_dis = cls.get_orientation_distance(target_pose[1], current_pose[1])
        return np.array([position_dis, orientation_dis], dtype=np.float64)

    @classmethod
    def get_pose_error_in_axis(
        cls, target_pose: np.ndarray, current_pose: np.ndarray
    ) -> tuple:
        """得到机器人当前位姿与目标位姿在各个对应轴上的分误差(姿态角度误差为向量夹角，即范围为[0,pi])"""
        position_error = cls.get_axis_error(target_pose[0], current_pose[0])
        orientation_error = cls.get_axis_error(target_pose[1], current_pose[1])
        orientation_error = cls.change_to_pi_scope(orientation_error)
        return position_error, orientation_error

    @staticmethod
    def norm(vector: np.ndarray) -> float:
        """计算向量的模（二范数）"""
        return np.linalg.norm(vector)

    @staticmethod
    def change_to_half_pi_scope(
        direction: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """将方向角从[-pi,pi]转换为[-pi/2,pi/2]（一般用于使轴线重合而不要求同向）"""
        if isinstance(direction, np.ndarray):
            direction[direction > np.pi / 2] -= np.pi
            direction[direction < -np.pi / 2] += np.pi
        else:
            if direction > np.pi / 2:
                direction -= np.pi
            elif direction < -np.pi / 2:
                direction += np.pi
        return direction

    @staticmethod
    def change_to_pi_scope(
        direction: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """将角度从[-2pi,2pi]转换为[-pi,pi]（一般用于通过优弧对齐姿态）"""
        if isinstance(direction, np.ndarray):
            direction[direction > np.pi] -= 2 * np.pi
            direction[direction < -np.pi] += 2 * np.pi
        else:
            if direction > np.pi:
                direction -= 2 * np.pi
            elif direction < -np.pi:
                direction += 2 * np.pi
        return direction

    @staticmethod
    def to_euler(orientation: np.ndarray) -> np.ndarray:
        """如果输入为四元数则转换为欧拉角，若为欧拉角则直接返回"""
        if len(orientation) == 4:
            orientation = transformations.euler_from_quaternion(orientation)
        return orientation

    @classmethod
    def ensure_euler(cls, euler: np.ndarray) -> bool:
        """保证欧拉角在合理的范围内(该函数直接通过引用方式修改传入参数)"""
        if len(euler) != 3:
            print("The length of euler angle must be 3!")
            return False
        elif abs(euler[1]) > np.pi / 2:
            print(f"The pitch angle {euler[1]} is out of range [-pi/2,pi/2]!")
            return False
        euler[0] = cls.change_to_pi_scope(euler[0])
        euler[1] = cls.change_to_half_pi_scope(euler[1])
        euler[2] = cls.change_to_pi_scope(euler[2])
        return True
