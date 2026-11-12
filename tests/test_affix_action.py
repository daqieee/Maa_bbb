"""Offline regressions for affix reroll decisions and MaaFramework integration."""

import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import jsonc
from maa.resource import Resource
from maa.tasker import LoggingLevelEnum, Tasker

from agent.custom.action.AffixRerollV2 import AffixRerollV2
from agent.custom.utils.AffixRules import evaluate_side


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "assets/resource/base/pipeline/小工具/洗词缀V2.json"
MODULE = "agent.custom.action.AffixRerollV2"
ENTRY = "洗词缀V2"
KEEP = "洗词缀V2-点击-保留旧词缀"
WRITE = "洗词缀V2-点击-写入新词缀"
REROLL = "洗词缀V2-点击-再次重构"
PLAIN_ROLL = "洗词缀V2-普通-点击-词缀重构"
LOCK_ROLL = "洗词缀V2-锁-点击-词缀重构"


class AffixConfigurationTests(unittest.TestCase):
    def test_ui_selections_merge_without_losing_other_values_or_nodes(self):
        Tasker.set_stdout_level(LoggingLevelEnum.Off)
        pipeline = jsonc.loads(PIPELINE.read_text(encoding="utf-8"))
        options = jsonc.loads(
            (ROOT / "assets/tasks/工具/圣痕洗词缀.json").read_text(encoding="utf-8")
        )["option"]
        resource = Resource()
        self.assertTrue(resource.override_pipeline(pipeline))
        for name, selected in (
            ("洗词缀模式", "词缀锁洗词缀"),
            ("确认使用词缀锁", "Yes"),
            ("单条高攻", "No"),
            ("角色属性", "虚数"),
            ("武器类别", "大剑、拳套"),
        ):
            case = next(case for case in options[name]["cases"] if case["name"] == selected)
            self.assertTrue(resource.override_pipeline(case["pipeline_override"]))
        for name, value in (("单条达标值", "23.1"), ("目标总攻击力", "43.5"), ("重构次数上限", "7")):
            option = options[name]
            override = json.dumps(option["pipeline_override"], ensure_ascii=False)
            override = override.replace("{" + option["inputs"][0]["name"] + "}", value)
            self.assertTrue(resource.override_pipeline(json.loads(override)))
        params = resource.get_node_data(ENTRY)["attach"]
        self.assertEqual(params["模式"], "词缀锁洗词缀")
        self.assertEqual(params["确认词缀锁"], "Yes")
        self.assertEqual(params["单条高攻"], "No")
        self.assertEqual(params["属性"], "虚数")
        self.assertEqual(params["武器"], "大剑、拳套")
        self.assertEqual(float(params["单条达标值"]), 23.1)
        self.assertEqual(float(params["目标攻击力"]), 43.5)
        self.assertEqual(int(params["最大重构次数"]), 7)
        self.assertEqual(params["OCR重试次数"], pipeline[ENTRY]["attach"]["OCR重试次数"])
        self.assertEqual(params["nodes"], pipeline[ENTRY]["attach"]["nodes"])


class AffixActionTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = jsonc.loads(PIPELINE.read_text(encoding="utf-8"))
        self.params = copy.deepcopy(self.pipeline[ENTRY]["attach"])
        self.params.update(
            {
                "属性": "虚数",
                "武器": "大剑、拳套",
                "目标攻击力": 42,
                "最大重构次数": 2,
                "单条高攻": "No",
                "OCR重试次数": 1,
            }
        )
        self.context = SimpleNamespace(
            get_node_data=Mock(return_value={"attach": self.params}),
            run_task=Mock(return_value=SimpleNamespace(status=SimpleNamespace(succeeded=True))),
            tasker=SimpleNamespace(stopping=False),
        )
        self.argv = SimpleNamespace(node_name=ENTRY, custom_action_param="{}")
        self.action = AffixRerollV2()
        self.logger_patch = patch(MODULE + ".Logger")
        self.logger_patch.start()
        self.addCleanup(self.logger_patch.stop)

    def side(self, first, second):
        return evaluate_side([first, second], self.params["属性"], self.params["武器"])

    def run_sides(self, *sides):
        pairs = list(zip(sides[::2], sides[1::2]))
        remaining = iter(pairs)
        read = lambda *args: next(remaining, pairs[-1])
        with patch.object(self.action, "_read_sides", side_effect=read), patch.object(
            self.action, "_missing_markers", return_value=[]
        ), patch(MODULE + ".time.sleep"):
            return self.action.run(self.context, self.argv)

    def clicked_nodes(self):
        return [call.args[0] for call in self.context.run_task.call_args_list]

    def assert_no_reroll(self):
        self.assertTrue(
            {REROLL, PLAIN_ROLL, LOCK_ROLL}.isdisjoint(self.clicked_nodes()),
            self.clicked_nodes(),
        )

    def test_different_condition_with_same_kind_and_value_is_not_a_lock(self):
        self.params.update({"模式": "词缀锁洗词缀", "确认词缀锁": "Yes"})
        old = self.side("虚数角色攻击提升13.0", "攻击提升9.0")
        new = self.side("机械角色攻击提升13.0", "攻击提升10.0")
        result = self.run_sides(old, new, old, new)
        self.assertFalse(result.success)
        self.context.run_task.assert_not_called()

    def test_two_unchanged_rows_are_ambiguous_and_do_not_consume_locks(self):
        self.params.update({"模式": "词缀锁洗词缀", "确认词缀锁": "Yes"})
        side = self.side("虚数角色攻击提升13.0", "攻击提升9.0")
        result = self.run_sides(side, side, side, side)
        self.assertFalse(result.success)
        self.context.run_task.assert_not_called()

    def test_unreadable_locked_panel_stops_before_clicking(self):
        self.params.update({"模式": "词缀锁洗词缀", "确认词缀锁": "Yes"})
        old = self.side("核心", "攻击提升9.0")
        new = self.side("虚数角色攻击提升13.0", "攻击提升10.0")
        result = self.run_sides(old, new, old, new)
        self.assertFalse(result.success)
        self.context.run_task.assert_not_called()

    def test_target_candidate_is_written_without_spending_another_reroll(self):
        old = self.side("虚数角色攻击提升20.0", "大剑、拳套角色攻击提升20.0")
        new = self.side("虚数角色攻击提升21.0", "大剑、拳套角色攻击提升22.0")
        result = self.run_sides(old, new)
        self.assertTrue(result.success)
        self.assertEqual(self.clicked_nodes(), [WRITE])

    def test_final_allowed_round_can_complete(self):
        self.params["最大重构次数"] = 1
        old = self.side("虚数角色攻击提升20.0", "大剑、拳套角色攻击提升20.0")
        worse = self.side("虚数角色攻击提升19.0", "攻击提升8.0")
        new = self.side("虚数角色攻击提升21.0", "大剑、拳套角色攻击提升22.0")
        result = self.run_sides(old, worse, old, new)
        self.assertTrue(result.success)
        self.assertEqual(self.clicked_nodes(), [REROLL, WRITE])

    def test_round_limit_does_not_generate_an_unread_candidate(self):
        self.params["最大重构次数"] = 1
        old = self.side("虚数角色攻击提升20.0", "攻击提升9.0")
        new = self.side("虚数角色攻击提升19.0", "攻击提升8.0")
        result = self.run_sides(old, new)
        self.assertFalse(result.success)
        self.assertEqual(self.clicked_nodes(), [REROLL])

    def test_last_round_still_keeps_a_better_result(self):
        self.params["最大重构次数"] = 1
        old = self.side("虚数角色攻击提升19.0", "攻击提升9.0")
        worse = self.side("虚数角色攻击提升18.0", "攻击提升9.0")
        new = self.side("虚数角色攻击提升20.0", "攻击提升9.0")
        result = self.run_sides(old, worse, old, new)
        self.assertFalse(result.success)
        self.assertEqual(self.clicked_nodes(), [REROLL, WRITE])

    def test_existing_target_is_kept(self):
        old = self.side("虚数角色攻击提升21.0", "大剑、拳套角色攻击提升22.0")
        new = self.side("虚数角色攻击提升19.0", "攻击提升8.0")
        result = self.run_sides(old, new)
        self.assertTrue(result.success)
        self.assertEqual(self.clicked_nodes(), [KEEP])

    def test_existing_target_can_be_kept_when_new_ocr_is_unreadable(self):
        old = self.side("虚数角色攻击提升21.0", "大剑、拳套角色攻击提升22.0")
        new = self.side("核心", "")
        result = self.run_sides(old, new)
        self.assertTrue(result.success)
        self.assertEqual(self.clicked_nodes(), [KEEP])

    def test_single_target_uses_the_best_value_across_rows(self):
        self.params["单条高攻"] = "Yes"
        old = self.side("攻击提升9.0", "虚数角色攻击提升22.0")
        new = self.side("虚数角色攻击提升23.0", "生命上限提升235.4")
        result = self.run_sides(old, new)
        self.assertTrue(result.success)
        self.assertEqual(self.clicked_nodes(), [WRITE])

    def test_single_target_never_replaces_an_already_better_other_row(self):
        self.params["单条高攻"] = "Yes"
        old = self.side("虚数角色攻击提升23.0", "攻击提升9.0")
        new = self.side("攻击提升8.0", "虚数角色攻击提升22.9")
        result = self.run_sides(old, new)
        self.assertTrue(result.success)
        self.assertEqual(self.clicked_nodes(), [KEEP])

    def test_failed_or_missing_framework_task_is_not_success(self):
        for task in (None, SimpleNamespace(status=SimpleNamespace(succeeded=False))):
            with self.subTest(task=task):
                self.context.run_task.return_value = task
                old = self.side("虚数角色攻击提升20.0", "大剑、拳套角色攻击提升20.0")
                new = self.side("虚数角色攻击提升21.0", "大剑、拳套角色攻击提升22.0")
                result = self.run_sides(old, new)
                self.assertFalse(result.success)
                self.assert_no_reroll()

    def test_stop_request_does_not_click(self):
        self.context.tasker.stopping = True
        result = self.run_sides()
        self.assertFalse(result.success)
        self.context.run_task.assert_not_called()

    def test_stop_requested_during_ocr_does_not_click(self):
        old = self.side("虚数角色攻击提升20.0", "大剑、拳套角色攻击提升20.0")
        new = self.side("虚数角色攻击提升21.0", "大剑、拳套角色攻击提升22.0")

        def read(*args):
            self.context.tasker.stopping = True
            return old, new

        with patch.object(self.action, "_read_sides", side_effect=read), patch.object(
            self.action, "_missing_markers", return_value=[]
        ):
            result = self.action.run(self.context, self.argv)
        self.assertFalse(result.success)
        self.context.run_task.assert_not_called()

    def test_lock_position_must_remain_the_same_between_rounds(self):
        self.params.update({"模式": "词缀锁洗词缀", "确认词缀锁": "Yes", "OCR重试次数": 3})
        old = self.side("虚数角色攻击提升13.0", "攻击提升9.0")
        first = self.side("虚数角色攻击提升13.0", "攻击提升8.0")
        drifted = self.side("虚数角色攻击提升14.0", "攻击提升9.0")
        result = self.run_sides(old, first, old, drifted)
        self.assertFalse(result.success)
        self.assertEqual(self.clicked_nodes(), [KEEP, LOCK_ROLL])

    def test_same_total_cannot_hide_a_failed_write(self):
        self.params["OCR重试次数"] = 3
        old = self.side("虚数角色攻击提升13.0", "攻击提升9.0")
        expected = self.side("虚数角色攻击提升14.0", "攻击提升9.0")
        actual = self.side("虚数角色攻击提升13.0", "攻击提升10.0")
        next_new = self.side("虚数角色攻击提升15.0", "攻击提升10.0")
        result = self.run_sides(old, expected, actual, next_new)
        self.assertFalse(result.success)
        self.assertEqual(self.clicked_nodes(), [WRITE, PLAIN_ROLL])

    def test_ocr_retry_recovers_without_spending_materials(self):
        self.params["OCR重试次数"] = 3
        unreadable = self.side("核心", "攻击提升9.0")
        old = self.side("虚数角色攻击提升20.0", "大剑、拳套角色攻击提升20.0")
        new = self.side("虚数角色攻击提升21.0", "大剑、拳套角色攻击提升22.0")
        result = self.run_sides(unreadable, new, old, new)
        self.assertTrue(result.success)
        self.assertEqual(self.clicked_nodes(), [WRITE])

    def test_invalid_numeric_parameters_fail_before_reading_or_clicking(self):
        for key, value in (
            ("目标攻击力", "nan"),
            ("目标攻击力", "inf"),
            ("目标攻击力", 0),
            ("单条达标值", "-inf"),
            ("最大重构次数", "2.5"),
            ("最大重构次数", "inf"),
            ("最大重构次数", -1),
            ("OCR重试次数", "1.5"),
            ("OCR重试次数", 0),
        ):
            with self.subTest(key=key, value=value), patch.dict(self.params, {key: value}):
                result = self.run_sides()
                self.assertFalse(result.success)
                self.context.run_task.assert_not_called()

    def test_fragmented_ocr_keeps_the_condition_with_the_attack_value(self):
        fragments = [
            SimpleNamespace(text="13.0", box=[865, 169, 40, 20]),
            SimpleNamespace(text="攻击提升", box=[750, 170, 100, 20]),
            SimpleNamespace(text="机械角色", box=[640, 168, 100, 20]),
        ]
        self.context.run_recognition = Mock(
            return_value=SimpleNamespace(
                hit=True,
                all_results=fragments,
                filtered_results=fragments,
                best_result=fragments[1],
            )
        )
        text = self.action._ocr_text(self.context, "词缀", object())
        self.assertEqual(text, "机械角色攻击提升13.0")
        self.assertEqual(self.side(text, "攻击提升9.0").total, 9)

    def test_wrapped_ocr_is_read_top_to_bottom_then_left_to_right(self):
        fragments = [
            SimpleNamespace(text="攻击提升13.0", box=[640, 195, 150, 20]),
            SimpleNamespace(text="角色", box=[950, 168, 40, 20]),
            SimpleNamespace(text="重炮、镰刀、火箭锤", box=[640, 169, 300, 20]),
        ]
        self.context.run_recognition = Mock(
            return_value=SimpleNamespace(hit=True, all_results=fragments, filtered_results=fragments)
        )
        text = self.action._ocr_text(self.context, "词缀", object())
        self.assertEqual(text, "重炮、镰刀、火箭锤角色攻击提升13.0")

    def test_filtered_out_condition_cannot_become_unconditional_attack(self):
        condition = SimpleNamespace(text="机械角色", box=[640, 168, 100, 20])
        attack = SimpleNamespace(text="攻击提升13.0", box=[750, 168, 150, 20])
        self.context.run_recognition = Mock(
            return_value=SimpleNamespace(
                hit=True,
                all_results=[condition, attack],
                filtered_results=[attack],
                best_result=attack,
            )
        )
        text = self.action._ocr_text(self.context, "词缀", object())
        self.assertEqual(text, "")
        self.assertFalse(self.side(text, "攻击提升9.0").reliable)


if __name__ == "__main__":
    unittest.main()
