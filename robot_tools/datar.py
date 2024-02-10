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


if __name__ == "__main__":
    test_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    test_array, index = remove_target(test_array, 3)
    print(test_array), print(index)
