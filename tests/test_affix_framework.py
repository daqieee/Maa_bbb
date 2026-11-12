"""Exercise nested MaaFramework tasks without connecting to a game or device."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

import jsonc
import numpy as np
from maa.controller import CustomController
from maa.custom_action import CustomAction
from maa.custom_recognition import CustomRecognition
from maa.resource import Resource
from maa.tasker import LoggingLevelEnum, Tasker

from agent.custom.action.AffixRerollV2 import AffixRerollV2


PIPELINE = Path(__file__).resolve().parents[1] / "assets/resource/base/pipeline/小工具/洗词缀V2.json"
BOX = (100, 80, 50, 20)


class RecordingController(CustomController):
    def __init__(self):
        super().__init__()
        self.clicks = []

    def connect(self):
        return True

    def request_uuid(self):
        return "affix-offline-test"

    def get_features(self):
        return 0  # Use the click callback, with no real mouse/touch events.

    def screencap(self):
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    def click(self, x, y):
        self.clicks.append((x, y))
        return True

    # None of these operations is used by a Click node. Fail if one is attempted.
    def start_app(self, intent):
        return False

    def stop_app(self, intent):
        return False

    def swipe(self, x1, y1, x2, y2, duration):
        return False

    def touch_down(self, contact, x, y, pressure):
        return False

    def touch_move(self, contact, x, y, pressure):
        return False

    def touch_up(self, contact):
        return False

    def click_key(self, keycode):
        return False

    def input_text(self, text):
        return False

    def key_down(self, keycode):
        return False

    def key_up(self, keycode):
        return False


class ButtonRecognition(CustomRecognition):
    def __init__(self):
        super().__init__()
        self.hit = True
        self.calls = []
        self.started = threading.Event()

    def analyze(self, context, argv):
        self.calls.append(argv.node_name)
        self.started.set()
        return BOX if self.hit else None


class NestedButtonAction(CustomAction):
    def __init__(self):
        super().__init__()
        self.affix = AffixRerollV2()
        self.results = []

    def run(self, context, argv):
        node = json.loads(argv.custom_action_param)
        result = self.affix._find_and_click(context, node, Mock(), node)
        self.results.append(result)
        return result


class AffixFrameworkTests(unittest.TestCase):
    def setUp(self):
        Tasker.set_stdout_level(LoggingLevelEnum.Off)
        pipeline = jsonc.loads(PIPELINE.read_text(encoding="utf-8"))
        self.buttons = [name for name, node in pipeline.items() if node.get("action") == "Click"]
        self.assertEqual(len(self.buttons), 5)
        # Keep each production Click definition; replace only image recognition
        # and timing so no OCR model, device, or game is needed.
        test_pipeline = {}
        for name in self.buttons:
            test_pipeline[name] = dict(pipeline[name])
            test_pipeline[name].update(
                recognition={"type": "Custom", "param": {"custom_recognition": "Button"}},
                timeout=100,
                rate_limit=10,
                pre_delay=0,
                post_delay=0,
            )
        test_pipeline["Entry"] = {
            "action": "Custom",
            "custom_action": "NestedButton",
            "pre_delay": 0,
            "post_delay": 0,
        }
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        directory = Path(self.directory.name) / "pipeline"
        directory.mkdir()
        (directory / "buttons.json").write_text(json.dumps(test_pipeline, ensure_ascii=False), encoding="utf-8")
        self.resource = Resource()
        self.recognition = ButtonRecognition()
        self.action = NestedButtonAction()
        self.resource.register_custom_recognition("Button", self.recognition)
        self.resource.register_custom_action("NestedButton", self.action)
        self.assertTrue(self.resource.post_bundle(self.directory.name).wait().succeeded)
        self.controller = RecordingController()
        self.assertTrue(self.controller.post_connection().wait().succeeded)
        self.tasker = Tasker()
        self.tasker.bind(self.resource, self.controller)
        self.assertTrue(self.tasker.inited)
        self.addCleanup(lambda: self.tasker.post_stop().wait())

    def start(self, node, overrides=None):
        override = {"Entry": {"custom_action_param": node}}
        override.update(overrides or {})
        return self.tasker.post_task("Entry", override)

    def test_all_buttons_recognize_entry_and_click_recognition_box(self):
        for node in self.buttons:
            with self.subTest(node=node):
                self.assertTrue(self.start(node).wait().succeeded)
                self.assertEqual(self.recognition.calls[-1], node)
                self.assertTrue(self.action.results[-1])
                x, y = self.controller.clicks[-1]
                self.assertTrue(BOX[0] <= x < BOX[0] + BOX[2])
                self.assertTrue(BOX[1] <= y < BOX[1] + BOX[3])
        self.assertEqual(len(self.controller.clicks), len(self.buttons))

    def test_unrecognized_buttons_fail_without_clicking(self):
        self.recognition.hit = False
        for node in self.buttons:
            with self.subTest(node=node):
                self.assertTrue(self.start(node).wait().failed)
                self.assertIn(node, self.recognition.calls)
                self.assertFalse(self.action.results[-1])
        self.assertEqual(self.controller.clicks, [])

    def test_stop_during_recognition_does_not_click_or_report_button_success(self):
        self.recognition.hit = False
        node = self.buttons[0]
        job = self.start(node, {node: {"timeout": 5000}})
        self.assertTrue(self.recognition.started.wait(timeout=2))
        self.tasker.post_stop().wait()
        job.wait()
        self.assertEqual(self.action.results, [False])
        self.assertEqual(self.controller.clicks, [])


if __name__ == "__main__":
    unittest.main()
