from robot_tools.datar import TrajTools
import numpy as np


if __name__ == "__main__":
    # 二维测试
    test_data = {
        "0": {"observation": [1, 2, 3], "action": [4, 5, 6], "num": 3},
        "1": {"observation": [7, 8], "action": [10, 11], "num": 2},
        "2": {"observation": [13, 14, 15, 16], "action": [17, 18, 19, 20], "num": 4},
    }
    out_put1 = TrajTools.construct_trajs(test_data, key="observation", num_key="num")
    trajs1, info1 = out_put1
    out_put2 = TrajTools.construct_trajs(test_data, key="observation")
    trajs2, info2 = out_put2

    assert (  # 有nan的情况下无法判断是否相等
        np.nan_to_num(trajs1[0]) == np.nan_to_num(trajs1[0])
    ).all(), f"\n{trajs1[0]}\n{trajs1[0]}"
    assert info1.trajs_num == info2.trajs_num
    assert (info1.each_points_num == info2.each_points_num).all()
    assert info1.max_points_num == info2.max_points_num
    assert info1.features_num == info2.features_num

    # 三维测试
    test_data = {
        "0": {
            "observation": [[1, 2, 3], [3, 4, 5]],
            "action": [[4, 5, 6], [6, 7, 8]],
            "num": 2,
        },
        "1": {
            "observation": [[5, 6, 0], [6, 7, 0], [7, 8, 0]],
            "action": [[7, 8, 0], [8, 9, 0], [9, 1, 0]],
            "num": 3,
        },
    }
    out_put3 = TrajTools.construct_trajs(test_data, key="observation", num_key="num")
    trajs3, info3 = out_put3
    print("三维测试")
    print(trajs3[0])
    print(trajs3[0][0, 1])

    test_traj_times = TrajTools.traj_times_from_time_trajs(trajs3[0])
    # print("test_traj_times\n")
    # print(test_time_trajs)
    # print(test_traj_times)

    # grow and mixed test
    out_put4 = TrajTools.construct_trajs(
        test_data, key="observation", num_key="num", series_type="h", mixed_type="h"
    )
    trajs4, info4 = out_put4
    print("\ngrow and mixed test")
    print(trajs4[0])

    # delete test
    X = TrajTools.delete_element_by_traj(trajs4[1], info4.each_points_num, 0)
    Y = TrajTools.delete_element_by_traj(trajs4[1], info4.each_points_num, -1)
    print("\ndelete_element_by_traj test")
    print(trajs4[1])
    print(X)
    print(Y)
    Z = TrajTools.delete_mixed_by_time(trajs4[2], 0, info4.trajs_num)
    W = TrajTools.delete_mixed_by_time(
        trajs4[2], -1, info4.trajs_num, info4.max_points_num
    )
    print("\ndelete_mixed_by_time test")
    print(trajs4[2])
    print(Z)
    print(W)
