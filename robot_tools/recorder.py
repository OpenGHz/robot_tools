""" 数据记录（加载）相关函数 """


import json


def json_process(file_path, write=None, log=False):
    """读取/写入json文件"""

    if write is not None:
        with open(file_path, "w") as f_obj:
            json.dump(write, f_obj)
        if log:
            print("写入数据为：", write)
    else:
        with open(file_path) as f_obj:
            write = json.load(f_obj)
        if log:
            print("加载数据为：", write)
    return write
