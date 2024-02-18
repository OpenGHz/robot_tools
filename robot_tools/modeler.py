import numpy as np


class ModelerTemplate(object):
    """模型基类"""

    def __init__(self):
        self._model = None

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass

    def simulate(self, X):
        pass


class LeastSquares(object):
    def __init__(self, type:str = "basic") -> None:
        """最小二乘法拟合，type为拟合类型，目前只有basic一种类型，后续考虑加入正则化项等"""
        self._type = type
        self._model = None

    def fit(self, X: np.ndarray, Y: np.ndarray, with_bias: bool = False) -> np.ndarray:
        if self._type == "basic":
            self._model = self._basic_fit(X, Y, with_bias)
            return self._model
        else:
            raise ValueError("Unknown type of least squares fitting.")

    def _basic_fit(
        X: np.ndarray, Y: np.ndarray, with_bias: bool = False
    ) -> np.ndarray:
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
    pass
"""
https://blog.csdn.net/cengjing12/article/details/106178518
"""