import numpy as np
from typing import Tuple
import os
import glob


def remove_target(arr: np.ndarray, target) -> Tuple[np.ndarray, np.ndarray]:
    """删除一维数组中等于0的元素并返回原数组和被删除元素的索引"""
    return arr[arr != 0], np.where(arr == target)[0]


def send_to_trash(pattern: str):
    # 定义要匹配的文件名模式
    # pattern = "trajs_*.json"
    # 获取当前工作目录
    current_directory = os.getcwd()
    # 构造匹配模式的文件路径
    files_to_delete = glob.glob(os.path.join(current_directory, pattern))
    # 遍历匹配到的文件并删除到回收站
    for file_path in files_to_delete:
        try:
            from send2trash import send2trash

            send2trash(file_path)
            print(f"Sent file to trash: {file_path}")
        except Exception as e:
            print(f"Error sending file to trash {file_path}: {e}")


def least_squares(X: np.ndarray, Y: np.ndarray, with_bias: bool = False) -> np.ndarray:
    """
    多维输入输出的最小二乘拟合：Y = X * K^T + C，返回拟合参数K

    参数：
    X: 输入数据矩阵，每行是一个数据点，每列是一个特征
    Y: 输出数据矩阵，每行是一个数据点的目标值的向量
    with_bias: 是否包含偏置项，默认为False，若为True，则自动在X中添加一列全为1的列，
    也可以手动添加：X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))，
    此时with_bias务必设置为False

    返回拟合参数
    """
    if with_bias:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
    # 计算拟合参数
    K_T = np.linalg.lstsq(X, Y, rcond=None)[0]
    return K_T.T


if __name__ == "__main__":
    # # 删除数组中的目标元素测试
    # test_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # test_array, index = remove_target(test_array, 3)
    # print(test_array), print(index)

    # 最小二乘拟合测试
    """
    每个x=[x1, x2, x3]^T对应y=[y1, y2, y3]^T -> y = Kx + c, 其中常数偏置项c=[c1, c2, c3]^T
    为了能更好地用numpy矩阵表示，即将x和y变成行的形式：y = Kx + c -> y^T = x^T K^T + c^T
    从而多组数据可以表示为矩阵乘法 Y = X * K^T + C，其中：
        Y是输出数据矩阵，每行是一个数据点的目标值的向量；
        X是输入数据矩阵，每行是一个数据点，每列是一个特征；
        K是拟合参数，每行是一个输出特征的权重；
        C是偏置项，每行是一个输出特征的偏置项（各行完全相等）。
    对于偏置项C，可以在X中添加一列全为1的列：X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))，
    此时K的最后一行对应偏置项：K_with_bias.T = np.vstack((K.T, c)) 或者 K_with_bias = np.hstack((K, c[:, np.newaxis])
    """
    # 生成随机的多维输入和多维输出数据
    np.random.seed(42)
    X = np.random.rand(1000, 3)  # N个数据点，每个点有3个特征
    X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))
    c = np.random.rand(3)  # 偏置项
    K = np.array(
        [[2, -1, 3], [1, 0, -2], [5, 2, -1]]
    )  # 真实参数，每行是一个输出特征的权重，没有偏置项
    K_with_bias = np.hstack((K, c[:, np.newaxis]))  # 添加偏置项
    Y = np.dot(X, K.T) + 0.1 * np.random.randn(1000, 3)  # 添加噪声
    Y_with_bias = np.dot(X_with_bias, K_with_bias.T) + 0.1 * np.random.randn(
        1000, 3
    )  # 添加噪声
    # 使用最小二乘进行拟合
    K_fit = least_squares(X, Y)
    print("真实参数:\n", K)
    print("拟合参数:\n", K_fit)
    K_fit_with_bias = least_squares(X, Y_with_bias, with_bias=True)
    print("真实参数:\n", K_with_bias)
    print("拟合参数:\n", K_fit_with_bias)
