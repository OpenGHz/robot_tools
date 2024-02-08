from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 需要import来支持3D，尽管不直接使用


class Painter2D(object):
    @staticmethod
    def plot_xy(x: np.ndarray, y: np.ndarray, title=None) -> None:
        fig, ax = plt.subplots()
        plt.plot(x, y)
        ax.set_aspect("equal")
        if title is not None:
            plt.title(title)
        plt.show()
        fig.clear()

    @staticmethod
    def plot_points(points, title=None) -> None:
        points = np.array(points)
        fig, ax = plt.subplots()
        plt.plot(points[:, 0], points[:, 1])
        ax.set_aspect("equal")
        if title is not None:
            plt.title(title)
        plt.show()
        fig.clear()

    @staticmethod
    def get_circle_points(center, radius, num_points, plot=False) -> np.ndarray:
        points = []
        for i in range(num_points):
            theta = i * 2 * np.pi / num_points
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            points.append((x, y))
        # plot these points
        points = np.array(points)
        if plot:
            fig, ax = plt.subplots()
            plt.plot(points[:, 0], points[:, 1])
            ax.set_aspect("equal")
            plt.show()
            fig.clear()
        return points

    @staticmethod
    def get_ellipse_points(center, a, b, num_points, plot=False) -> np.ndarray:
        points = []
        for i in range(num_points):
            theta = i * 2 * np.pi / num_points
            x = center[0] + a * np.cos(theta)
            y = center[1] + b * np.sin(theta)
            points.append((x, y))
        # plot these points
        points = np.array(points)
        if plot:
            fig, ax = plt.subplots()
            plt.plot(points[:, 0], points[:, 1])
            ax.set_aspect("equal")
            plt.show()
            fig.clear()
        return points

    @staticmethod
    def get_rectangle_points(center, width, height, plot=False) -> np.ndarray:
        points = []
        points.append((center[0] - width / 2, center[1] - height / 2))
        points.append((center[0] + width / 2, center[1] - height / 2))
        points.append((center[0] + width / 2, center[1] + height / 2))
        points.append((center[0] - width / 2, center[1] + height / 2))
        points.append((center[0] - width / 2, center[1] - height / 2))
        # plot these points
        points = np.array(points)
        if plot:
            fig, ax = plt.subplots()
            plt.plot(points[:, 0], points[:, 1])
            ax.set_aspect("equal")
            plt.show()
            fig.clear()
        return points

    @classmethod
    def get_square_points(cls, center, width, plot=False) -> np.ndarray:
        return cls.get_rectangle_points(center, width, width, plot)

    @staticmethod
    def get_polygon_points(center, radius, num_points, plot=False):
        points = []
        for i in range(num_points):
            theta = i * 2 * np.pi / num_points
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            points.append((x, y))
        # plot these points
        points = np.array(points)
        if plot:
            fig, ax = plt.subplots()
            plt.plot(points[:, 0], points[:, 1])
            ax.set_aspect("equal")
            plt.show()
            fig.clear()
        return points

    @staticmethod
    def get_heart_points(center, radius, num_points, plot=False) -> np.ndarray:
        points = []
        for i in range(num_points):
            theta = i * 2 * np.pi / num_points
            x = center[0] + radius * (16 * np.sin(theta) ** 3) / 3
            y = center[1] + radius * (
                13 * np.cos(theta)
                - 5 * np.cos(2 * theta)
                - 2 * np.cos(3 * theta)
                - np.cos(4 * theta)
            )
            points.append((x, y))
        # plot these points
        points = np.array(points)
        if plot:
            fig, ax = plt.subplots()
            plt.plot(points[:, 0], points[:, 1])
            ax.set_aspect("equal")
            plt.show()
            fig.clear()
        return points

    @staticmethod
    def get_star_points(center, radius, num_points, plot=False) -> np.ndarray:
        points = []
        for i in range(num_points):
            theta = i * 2 * np.pi / num_points
            if i % 2 == 0:
                x = center[0] + radius * np.cos(theta)
                y = center[1] + radius * np.sin(theta)
            else:
                x = center[0] + radius / 2 * np.cos(theta)
                y = center[1] + radius / 2 * np.sin(theta)
            points.append((x, y))
        # plot these points
        points = np.array(points)
        if plot:
            fig, ax = plt.subplots()
            plt.plot(points[:, 0], points[:, 1])
            ax.set_aspect("equal")
            plt.show()
            fig.clear()
        return points

    @staticmethod
    def get_cross_points(center, radius, num_points, plot=False) -> np.ndarray:
        points = []
        for i in range(num_points):
            theta = i * 2 * np.pi / num_points
            if i % 2 == 0:
                x = center[0] + radius * np.cos(theta)
                y = center[1] + radius * np.sin(theta)
            else:
                x = center[0] + radius / 2 * np.cos(theta)
                y = center[1] + radius / 2 * np.sin(theta)
            points.append((x, y))
        # plot these points
        points = np.array(points)
        if plot:
            fig, ax = plt.subplots()
            plt.plot(points[:, 0], points[:, 1])
            ax.set_aspect("equal")
            plt.show()
            fig.clear()
        return points

    @staticmethod
    def get_pentagram_points(center, radius, num_points, plot=False) -> np.ndarray:
        points = []
        for i in range(num_points):
            theta = i * 2 * np.pi / num_points
            if i % 2 == 0:
                x = center[0] + radius * np.cos(theta)
                y = center[1] + radius * np.sin(theta)
            else:
                x = center[0] + radius / 2 * np.cos(theta)
                y = center[1] + radius / 2 * np.sin(theta)
            points.append((x, y))
        # plot these points
        points = np.array(points)
        if plot:
            fig, ax = plt.subplots()
            plt.plot(points[:, 0], points[:, 1])
            ax.set_aspect("equal")
            plt.show()
            fig.clear()
        return points

    def get_spiral_points(
        a, b, num_points=1000, turns=2, plot=True, start_point=(0, 0)
    ):
        """
        生成螺旋线轨迹的函数（由内向外展开）

        参数：
        - a: 螺旋线的扭曲参数
        - b: 螺旋线的展开参数(b=0时为圆形螺旋线, b越大螺旋线越松散)
        - num_points: 生成轨迹的点数
        - turns: 螺旋线的圈数
        - plot: 是否绘制螺旋线图形，默认为True
        - start_point: 螺旋线的起始点，默认为原点(0, 0)

        返回：
        - x: 生成的螺旋线 x 坐标数组
        - y: 生成的螺旋线 y 坐标数组
        """
        x_bias = -start_point[0] + a
        y_bias = start_point[1]
        t = np.linspace(0, 2 * np.pi * turns, num_points)
        x = (a + b * t) * np.cos(t) - x_bias
        y = (a + b * t) * np.sin(t) + y_bias

        if plot:
            plt.plot(x, y, label=f"Spiral (a={a}, b={b})")
            plt.title("Spiral Trajectory")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.legend()
            plt.grid(True)
            plt.show()

        return x, y


class Painter3D(object):
    def plot_points(points, plot=True, title=None, connect=False) -> None:
        points = np.array(points)
        if plot:
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            if connect:
                ax.plot(points[:, 0], points[:, 1], points[:, 2])
            else:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2])
            # ax.set_aspect("equal")  # It is not currently possible to manually set the aspect on 3D axes
            if title is not None:
                plt.title(title)
            plt.show()
            fig.clear()


if __name__ == "__main__":
    TEST = "2D"
    if TEST == "2D":
        # test for 2D painter
        # Painter2D.get_circle_points((0, 0), 1, 100, plot=True)
        # Painter2D.get_ellipse_points((0, 0), 1, 2, 100, plot=True)
        # Painter2D.get_rectangle_points((0, 0), 1, 2, plot=True)
        # Painter2D.get_square_points((0, 0), 2, plot=True)
        # Painter2D.get_polygon_points((0, 0), 1, 5, plot=True)
        # Painter2D.get_heart_points((0, 0), 1, 100, plot=True)
        # Painter2D.get_star_points((0, 0), 1, 100, plot=True)
        # Painter2D.get_cross_points((0, 0), 1, 100, plot=True)
        # Painter2D.get_pentagram_points((0, 0), 1, 100, plot=True)
        Painter2D.get_spiral_points(0.1, 0.1, num_points=1000, turns=2, plot=True)
    else:
        # test for 3D painter
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        Painter3D.plot_points(points, plot=True)

# https://pythondict.com/python-qa/%E5%A6%82%E4%BD%95%E5%9C%A8matplotlib%E4%B8%AD%E8%AE%BE%E7%BD%AE%E7%BA%B5%E6%A8%AA%E6%AF%94%EF%BC%9F/
