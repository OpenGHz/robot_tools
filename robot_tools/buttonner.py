import logging
import time

# 配置日志格式和输出文件
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='button_history.log',
    filemode='a'  # 使用'w'模式会覆盖文件，使用'a'模式会追加到文件末尾
)

class ButtonnerConfig(object):
    def __init__(self):
        self.name = f"{self.name}"
        self.pressed_value = 1
        self.released_value = 0
        self.debounce_time = 50 * 1e6
        self.long_press_time = 1500 * 1e6
        self.short_release_time = 250 * 1e6
        self.init_state = 0


class Buttonner(object):

    def __init__(self, config:ButtonnerConfig):
        self.debounce_time = config.debounce_time
        self.long_press_time = config.long_press_time
        self.short_release_time = config.short_release_time
        self.pressed_value = config.pressed_value
        self.released_value = config.released_value
        self.name = config.name

        self.button_history = {
            "state": config.init_state,
            "timestamp": {
                "on_press": 0,
                "on_release": 0
            },
            "duration": {
                "press_first": 0,
                "release_first": 0,
                "press_second": 0,
                "release_second":0
            }
        }

        self.logger = logging.getLogger(self.name)

    def set_logger(self, logger:logging.Logger):
        self.logger = logger
    
    def get_logger(self) -> logging.Logger:
        return self.logger

    def set_current_state(self, value):
        if value != self.button_history["state"]:
            if value == self.pressed_value:
                self.get_logger().info(f"{self.name} pressed")
                self._process_last(self.button_history, "on_press")
            else:
                self.get_logger().info(f"{self.name} released")
                self._process_last(self.button_history, "on_release")
            self.button_history["state"] = value
        else:
            self._process_last(self.button_history, None)

    def _process_last(self, button_last:dict, trigger, current=None):
            # trigger发生时，记录对应的时间戳，排除抖动后记录有效的按键持续时间
            if trigger == "on_press":
                button_last["timestamp"]["on_press"] = time.time_ns()
                if button_last["timestamp"]["on_release"] > 0:
                    duration = time.time_ns() - button_last["timestamp"]["on_release"]
                    if duration <= self.debounce_time:
                        return
                    else:
                        if button_last["duration"]["release_first"] <= self.debounce_time:
                            button_last["duration"]["release_first"] = duration
                        else:
                            button_last["duration"]["release_second"] = duration
            elif trigger == "on_release":
                button_last["timestamp"]["on_release"] = time.time_ns()
                if button_last["timestamp"]["on_press"] > 0:
                    duration = time.time_ns() - button_last["timestamp"]["on_press"]
                    if duration <= self.debounce_time:
                        return
                    elif duration > self.long_press_time:
                        # long press在释放前就完成判断，但在释放后才清空dict状态
                        self._reset_history()
                        return
                    else:
                        if button_last["duration"]["press_first"] <= self.debounce_time:
                            button_last["duration"]["press_first"] = duration
                        else:
                            button_last["duration"]["press_second"] = duration
                            self.get_logger().info("Double Press")
                            self._reset_history()
            # 无trigger时
            elif trigger is None:
                assert current is not None, "current state is required"
                if current == self.pressed_value:
                    duration = time.time_ns() - button_last["timestamp"]["on_press"]
                    if duration > self.long_press_time:
                        if button_last["duration"]["press_first"] > self.debounce_time:
                            # short&long press
                            self.get_logger().info("Short&Long press")
                        else:
                            # long press
                            self.get_logger().info("Long press")
                elif current == self.released_value:
                    if button_last["duration"]["press_first"] > self.debounce_time:
                        duration = time.time_ns() - button_last["timestamp"]["on_release"]
                        if duration > self.short_release_time:
                            # short press
                            self.get_logger().info("Short Press")
                            # reset the timestamp and duration
                            self._reset_history()
    
    def _reset_history(self):
        for value in self.button_history.values():
            # do not clear the state
            if isinstance(value, dict):
                for key in value.keys():
                    value[key] = 0
        self.get_logger().info(f"Cleared last button dict {self.button_history}")