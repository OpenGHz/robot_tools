import unittest
from unittest.mock import patch
from robot_tools.trajer import TrajsRecorder
import numpy as np


class TestTrajsRecorder(unittest.TestCase):
    def test_init(self):
        features = ["feature1", "feature2"]
        path = "/path/to/trajs.json"
        recorder = TrajsRecorder(features, path)
        self.assertEqual(recorder._features, features)
        self.assertEqual(recorder._traj, {"feature1": [], "feature2": []})
        self.assertEqual(recorder._trajs, {0: {"feature1": [], "feature2": []}})
        self.assertFalse(recorder._recorded)
        self.assertEqual(recorder._path, path)

    def test_feature_add_scalar(self):
        recorder = TrajsRecorder(["feature1", "feature2"])
        recorder.feature_add(0, "feature1", 10)
        recorder.feature_add(0, "feature2", "value1")
        recorder.feature_add(0, "feature1", 20)
        recorder.feature_add(0, "feature2", "value2")
        self.assertEqual(
            recorder.trajs, {0: {"feature1": [10, 20], "feature2": ["value1", "value2"]}}
        )
        feature = 'feature1'
        self.assertEqual(recorder.trajs[0][feature], [10, 20])

    def test_feature_add_iterable(self):
        recorder = TrajsRecorder(["feature1", "feature2"])
        recorder.feature_add(0, "feature1", [1,2,3])
        recorder.feature_add(0, "feature2", np.array([4,5,6]))
        recorder.feature_add(0, "feature1", (7,8,9))
        recorder.feature_add(0, "feature2", {10,11,12})
        self.assertEqual(
            recorder.trajs, {0: {"feature1": [[1, 2, 3], [7, 8, 9]], "feature2": [[4, 5, 6], [10, 11, 12]]}}
        )
        feature = 'feature1'
        self.assertEqual(recorder.trajs[0][feature], [[1, 2, 3], [7, 8, 9]])

    def test_feature_add_all(self):
        recorder = TrajsRecorder(["feature1"])
        recorder.feature_add(0, "feature1", [1,2,3,4], all=True)
        self.assertEqual(
            recorder.trajs, {0: {"feature1": [1,2,3,4]}}
        )

    def test_features_add(self):
        recorder = TrajsRecorder(["feature1", "feature2"])
        recorder.features_add(0, [10, "value"])
        recorder.features_add(0, [20, "value2"])
        self.assertEqual(
            recorder.trajs, {0: {"feature1": [10, 20], "feature2": ["value", "value2"]}}
        )

    @patch("robot_tools.trajer.recorder.json_process")
    def test_record_with_check(self, mock_json_process):
        recorder = TrajsRecorder(["feature1", "feature2", "not_count"], "trajs.json")
        recorder.set_not_count_features({"not_count"})
        recorder.feature_add(0, "feature1", 10)
        recorder.feature_add(0, "feature2", "value")
        recorder.feature_add(0, "not_count", "none1")
        recorder.feature_add(0, "not_count", "none2")
        recorder.save(check=True)
        mock_json_process.assert_called_once_with(
            "trajs.json", write={0: {"feature1": [10], "feature2": ["value"], "not_count": ["none1", "none2"]}}
        )
        self.assertEqual(recorder.each_points_num[0], 1)

    @patch("robot_tools.trajer.recorder.json_process")
    def test_record_without_check(self, mock_json_process):
        recorder = TrajsRecorder(["feature1", "feature2", "not_count"], "trajs.json")
        recorder.set_not_count_features({"not_count"})
        recorder.feature_add(0, "feature1", 10)
        recorder.feature_add(0, "feature2", "value")
        recorder.feature_add(0, "not_count", "none1")
        recorder.feature_add(0, "not_count", "none2")
        recorder.save()
        mock_json_process.assert_called_once_with(
            "trajs.json", write={0: {"feature1": [10], "feature2": ["value"], "not_count": ["none1", "none2"]}}
        )
        self.assertEqual(recorder.each_points_num[0], 1)

    @patch("robot_tools.trajer.recorder.json_process")
    def test_record_with_custom_path_and_trajs(self, mock_json_process):
        recorder = TrajsRecorder(["feature1", "feature2"])
        recorder.feature_add(0, "feature1", 10)
        recorder.feature_add(0, "feature2", "value")
        recorder.save(
            path="/path/to/custom.json",
            trajs={0: {"feature1": [20], "feature2": ["custom"]}},
        )
        mock_json_process.assert_called_once_with(
            "/path/to/custom.json",
            write={0: {"feature1": [20], "feature2": ["custom"]}},
        )

    def test_auto_record(self):
        recorder = TrajsRecorder(["feature1", "feature2"])
        recorder.feature_add(0, "feature1", 10)
        recorder.feature_add(0, "feature2", "value")
        with patch("robot_tools.trajer.recorder.json_process") as mock_json_process:
            recorder.auto_save()
        mock_json_process.assert_not_called()

    def test_trajs(self):
        recorder = TrajsRecorder(["feature1", "feature2"])
        recorder.feature_add(0, "feature1", 10)
        recorder.feature_add(0, "feature2", "value")
        self.assertEqual(recorder.trajs, {0: {"feature1": [10], "feature2": ["value"]}})

    def test_getitem(self):
        recorder = TrajsRecorder(["feature1", "feature2"])
        recorder.feature_add(0, "feature1", 10)
        recorder.feature_add(0, "feature2", "value")
        self.assertEqual(recorder[0], {"feature1": [10], "feature2": ["value"]})

    def test_check_complete_data(self):
        recorder = TrajsRecorder(["feature1", "feature2", 'feature3'])
        recorder.feature_add(0, "feature1", [1, 2, 3])
        recorder.feature_add(0, "feature2", [4, 5, 6])
        recorder.feature_add(0, "feature3", [7, 8, 9])
        recorder.feature_add(1, "feature1", [10, 11, 12])
        recorder.feature_add(1, "feature2", [13, 14, 15])
        recorder.feature_add(1, "feature3", [16, 17, 18])
        self.assertTrue(recorder.check())

    def test_check_missing_feature(self):
        recorder = TrajsRecorder(["feature1", "feature2", 'feature3'])
        recorder.feature_add(0, "feature1", [1, 2, 3])
        recorder.feature_add(0, "feature2", [4, 5, 6])
        recorder.feature_add(1, "feature1", [10, 11, 12])
        recorder.feature_add(1, "feature2", [13, 14, 15])
        recorder.feature_add(1, "feature3", [16, 17, 18])
        self.assertFalse(recorder.check())

    def test_check_different_length(self):
        recorder = TrajsRecorder(["feature1", "feature2", 'feature3'])
        recorder.feature_add(0, "feature1", [1, 2, 3])
        recorder.feature_add(0, "feature2", [4, 5, 6])
        recorder.feature_add(1, "feature1", [10, 11, 12])
        recorder.feature_add(1, "feature2", [13, 14, 15])
        recorder.feature_add(1, "feature3", [16, 17])
        self.assertFalse(recorder.check())


    def remove(self):
        pass

if __name__ == "__main__":
    unittest.main()
