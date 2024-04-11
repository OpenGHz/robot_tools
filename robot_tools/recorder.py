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


def json_append(file_path, write, log=False):
    """向json文件中追加数据"""

    with open(file_path, "r+") as f_obj:
        data = json.load(f_obj)
        data.update(write)
        f_obj.seek(0)
        json.dump(data, f_obj)
    if log:
        print("追加数据为：", write)


def json_clear(file_path, log=False):
    """清空json文件"""

    with open(file_path, "w") as f_obj:
        json.dump({}, f_obj)
    if log:
        print("清空文件：", file_path)


def json_delete(file_path, key, log=False):
    """删除json文件中的某个键值对"""

    with open(file_path, "r+") as f_obj:
        data = json.load(f_obj)
        data.pop(key)
        f_obj.seek(0)
        json.dump(data, f_obj)
    if log:
        print("删除键值对：", key)


def json_update(file_path, key, value, log=False):
    """更新json文件中的某个键值对"""

    with open(file_path, "r+") as f_obj:
        data = json.load(f_obj)
        data[key] = value
        f_obj.seek(0)
        json.dump(data, f_obj)
    if log:
        print("更新键值对：", key, value)


import time
from collections import deque
import numpy as np
from threading import Thread
from typing import Union

try:
    import cv2
except ImportError as e:
    CV2_ERROR = e
else:
    CV2_ERROR = None


class ImageRecorderRos:
    def __init__(
        self,
        camera_names,
        is_debug=False,
        topic_names=None,
        show_images=True,
        image_shape=None,
    ):
        print("Starting ROS image recorder...")
        from collections import deque
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        self.is_debug = is_debug
        self.show_images = show_images
        self.target_image_shape = image_shape
        self.bridge = CvBridge()
        self.camera_names = camera_names
        self.image_info = {"raw_image": {}, "secs": {}, "nsecs": {}, "timestamps": {}}
        assert rospy.get_name() != "/unnamed", "Please init the ROS node first."
        for c, cam_name in enumerate(camera_names):
            topic_name = (
                f"/usb_cam_{cam_name}/image_raw"
                if topic_names is None
                else topic_names[c]
            )
            image = rospy.wait_for_message(topic_name, Image, timeout=3)
            self.image_callback(image, cam_name)
            rospy.Subscriber(topic_name, Image, self.image_callback, cam_name)
            if self.is_debug:
                setattr(self, f"{cam_name}_timestamps", deque(maxlen=50))
        time.sleep(0.5)
        if self.show_images:
            Thread(target=self._show_images, daemon=True).start()
        print("ROS Image recorder started.")

    def _show_images(self):
        import rospy

        rate = rospy.Rate(30)
        # try:
        while not rospy.is_shutdown():
            images = list(self.image_info["raw_image"].values())
            combined_frame = np.hstack(images)
            cv2.imshow(f"{self.camera_names}", combined_frame)
            cv2.waitKey(1)
            rate.sleep()
        # except rospy.exceptions.ROSInterruptException as e:
        #     print("ROS Image recorder: ", e)

    def image_callback(self, data, cam_name):
        image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        if self.target_image_shape is not None:
            image = cv2.resize(image, (self.target_image_shape[1], self.target_image_shape[0]))
        self.image_info["raw_image"][cam_name] = image
        self.image_info["secs"][cam_name] = data.header.stamp.secs
        self.image_info["nsecs"][cam_name] = data.header.stamp.nsecs
        if self.is_debug:
            self.image_info["timestamps"][cam_name].append(
                data.header.stamp.secs + data.header.stamp.secs * 1e-9
            )

    def get_images(self):
        return self.image_info["raw_image"]

    def get_images_shape(self):
        shape = [image.shape for image in self.image_info["raw_image"].values()]
        return shape

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        for cam_name in self.camera_names:
            image_freq = 1 / dt_helper(self.image_info["timestamps"][cam_name])
            print(f"{cam_name} {image_freq=:.2f}")
        print()


class ImageRecorderVideo:
    def __init__(
        self,
        cameras: Union[dict, list],
        is_debug=False,
        image_shape=(480, 640, 3),
        show_images=True,
        fps=30,
    ):
        print("Starting image recorder...")
        self.is_debug = is_debug
        self.show_images = show_images
        self.fps = fps
        if isinstance(cameras, dict):
            camera_names = list(cameras.keys())
            camera_indices = list(cameras.values())
            self.name2index = cameras
            self.index2name = {index: name for name, index in cameras.items()}
        else:
            camera_names = [f"camera_{index}" for index in cameras]
            camera_indices = cameras
            self.name2index = {
                name: index for name, index in zip(camera_names, camera_indices)
            }
            self.index2name = {
                index: name for name, index in zip(camera_names, camera_indices)
            }
        self.camera_names = camera_names
        self.camera_indices = camera_indices
        self.image_info = {"raw_image": {}, "secs": {}, "nsecs": {}}
        self.cap = {index: cv2.VideoCapture(int(index)) for index in camera_indices}
        # check
        for index in camera_indices:
            if not self.cap[index].isOpened():
                raise Exception(f"Failed to open camera {index}")
            else:
                ret, frame = self.cap[index].read()
                if not ret:
                    raise Exception(f"Failed to read from camera {index}")
                else:
                    assert tuple(frame.shape) == image_shape
        if self.is_debug:
            self.timestamps = {index: deque(maxlen=50) for index in cameras}
        Thread(target=self._image_reading, daemon=True).start()
        print("Image recorder started.")

    def _read_image_once(self):
        images = {}
        for cam_index in self.camera_indices:
            ret, frame = self.cap[cam_index].read()
            if ret:
                images[self.index2name[cam_index]] = frame
            else:
                raise Exception(f"Failed to read from camera {cam_index}")
        return images

    def _read_one_image(self, cam_name):
        ret, frame = self.cap[self.name2index[cam_name]].read()
        if ret:
            return frame
        else:
            raise Exception(f"Failed to read from camera {cam_name}")

    def _image_reading(self):
        duration = 1 / self.fps
        if self.show_images:
            # 将图像现实在一个窗口中，在图像上显示图像对应的摄像头编号
            window_name = f"Camera {self.camera_indices}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        start_time = time.time()
        while True:
            fps_time = start_time
            image_info = {"raw_image": {}, "secs": {}, "nsecs": {}}
            for cam_name in self.camera_names:
                image_info["raw_image"][cam_name] = self._read_one_image(cam_name)
                time_sec = int(time.time())
                image_info["secs"][cam_name] = time_sec
                time_ns = time.time_ns()
                image_info["nsecs"][cam_name] = time_ns - time_sec
                if self.is_debug:
                    self.timestamps[cam_name].append(time_ns * 1e-9)
            # 同步更新
            self.image_info = image_info
            # 将图像现实在一个窗口中，在图像上显示图像对应的摄像头编号
            frames = list(self.image_info["raw_image"].values())
            combined_frame = np.hstack(frames)
            time.sleep(max(0, duration - (time.time() - start_time)))
            start_time = time.time()
            if self.show_images:
                fps = 1 / (time.time() - fps_time)
                cv2.putText(
                    combined_frame,
                    f"FPS: {fps:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(window_name, combined_frame)
                cv2.waitKey(1)
            start_time = time.time()

    def get_images(self):
        return self.image_info["raw_image"]

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        for index in self.camera_indices:
            if index in self.timestamps:
                image_freq = 1 / dt_helper(self.timestamps[index])
                print(f"Camera {index} image frequency: {image_freq:.2f}")
            else:
                print(f"No timestamps available for camera {index}")
        print()


class ImageRecorderFake(object):
    def __init__(self, camera_names, is_debug=False, show_images=True):
        print("Starting fake image recorder...")
        self.is_debug = is_debug
        self.show_images = show_images
        self.camera_names = camera_names
        self.image_info = {"raw_image": {}, "secs": {}, "nsecs": {}}
        self.fake_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        if self.is_debug:
            self.timestamps = {cam_name: deque(maxlen=50) for cam_name in camera_names}
        Thread(target=self._image_reading, daemon=True).start()
        print("Fake image recorder started.")

    def _image_reading(self):
        while True:
            time_sec = int(time.time())
            time_ns = time.time_ns()
            for cam_name in self.camera_names:
                self.image_info["raw_image"][cam_name] = self.fake_image
                self.image_info["secs"][cam_name] = time_sec
                self.image_info["nsecs"][cam_name] = time_ns - time_sec
                if self.is_debug:
                    self.timestamps[cam_name].append(time_ns * 1e-9)
            time.sleep(1 / 30)

    def get_images(self):
        return self.image_info["raw_image"]

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        for cam_name in self.camera_names:
            if cam_name in self.timestamps:
                image_freq = 1 / dt_helper(self.timestamps[cam_name])
                print(f"{cam_name} image frequency: {image_freq:.2f}")
            else:
                print(f"No timestamps available for camera {cam_name}")
        print()


if __name__ == "__main__":
    json_process("data.json", write={"a": 1, "b": 2}, log=True)
    json_process("data.json", log=True)
