"""Run with python -m unittest discover -s tests -p 'test_affix_rules.py'."""

import unittest

from agent.custom.utils.AffixRules import (
    KIND_AMBIGUOUS,
    KIND_ATTRIBUTE,
    KIND_INVALID,
    KIND_UNCONDITIONAL,
    KIND_UNREADABLE,
    KIND_WEAPON,
    classify_line,
    evaluate_side,
    extract_attack_value,
    same_affix,
    validate_selection,
)


class AffixRulesTests(unittest.TestCase):
    def line(self, text, attribute="虚数", weapon="大剑、拳套"):
        return classify_line(text, attribute, weapon)

    def test_attack_values_keep_supported_ocr_corrections(self):
        for text, expected in (
            ("攻击提升9.0", 9.0),
            ("虚数角色攻击提升13,0", 13.0),
            ("攻 击 提 升 ９．０", 9.0),
            ("机械角色攻击提升1O.O", 10.0),
            ("重炮角色攻击提升227", 22.7),
            ("虚数角色攻击提升130", 13.0),
            ("攻击提升888", None),
            ("攻击提升35.0", None),
            ("攻击提升0", None),
            ("生命上限提升235.4", None),
        ):
            with self.subTest(text=text):
                self.assertEqual(extract_attack_value(text), expected)

    def test_known_attack_conditions(self):
        for text, attribute, weapon, kind, value in (
            ("攻击提升9.0", "机械", "大剑、拳套", KIND_UNCONDITIONAL, 9.0),
            ("虚数角色攻击提升13.0", "虚数", "大剑、拳套", KIND_ATTRIBUTE, 13.0),
            ("虚数角色攻击提升13.0", "机械", "大剑、拳套", KIND_ATTRIBUTE, 0),
            ("重孑角色攻击提升13.0", "量子", "大剑、拳套", KIND_ATTRIBUTE, 13.0),
            ("星辰角色攻击提升18.6", "星尘", "大剑、拳套", KIND_ATTRIBUTE, 18.6),
            ("拳套、大剑角色攻击提升21.0", "机械", "大剑、拳套", KIND_WEAPON, 21.0),
            ("旧词缀技能大剑、拳套角色攻击提升21.0", "机械", "大剑、拳套", KIND_WEAPON, 21.0),
            ("重炮、镰刀、火箭缍角色攻击提升22.7", "虚数", "重炮、镰刀、火箭锤", KIND_WEAPON, 22.7),
            ("重炮、镰刀角色攻击提升22.7", "机械", "大剑、拳套", KIND_INVALID, 0),
        ):
            with self.subTest(text=text, attribute=attribute):
                line = self.line(text, attribute, weapon)
                self.assertEqual((line.kind, line.value), (kind, value))
                self.assertTrue(line.trustworthy)

    def test_known_non_attack_affixes_are_reliable_zero(self):
        for text in ("生命上限提升235.4", "暴击伤害提升4.95%", "会心提升6.3"):
            with self.subTest(text=text):
                line = self.line(text)
                self.assertEqual(line.kind, KIND_INVALID)
                self.assertEqual(line.value, 0)
                self.assertTrue(line.trustworthy)

    def test_unknown_or_truncated_text_cannot_lower_the_old_score(self):
        for text, kind in (
            ("核心", KIND_UNREADABLE),
            ("", KIND_UNREADABLE),
            ("虚数角色功去提丹13.0", KIND_AMBIGUOUS),
            ("未知类型13.0", KIND_AMBIGUOUS),
            ("未知角色攻击提升13.0", KIND_AMBIGUOUS),
            ("大剑、重炮角色攻击提升13.0", KIND_AMBIGUOUS),
        ):
            with self.subTest(text=text):
                side = evaluate_side(["攻击提升9.0", text], "虚数", "大剑、拳套")
                self.assertEqual(side.lines[1].kind, kind)
                self.assertFalse(side.reliable)

    def test_empty_panel_is_not_reliable(self):
        side = evaluate_side([], "虚数", "大剑、拳套")
        self.assertFalse(side.readable)
        self.assertFalse(side.reliable)

    def test_side_total_and_best_single(self):
        side = evaluate_side(
            ["攻击提升9.0", "重炮、镰刀、火箭锤角色攻击提升22.7"],
            "虚数",
            "重炮、镰刀、火箭锤",
        )
        self.assertTrue(side.reliable)
        self.assertEqual(side.total, 31.7)
        self.assertEqual(side.best_single, 22.7)

    def test_same_affix_accepts_only_full_normalized_text(self):
        for old, new in (
            ("虚数角色攻击提升13.0", "虚 数 角 色 攻 击 提 升 １３．０"),
            ("生命上限提升235.4", "生命上限提升235.4"),
            ("攻击提升9.0", "攻击提升9.0"),
        ):
            with self.subTest(old=old, new=new):
                self.assertTrue(same_affix(self.line(old), self.line(new)))

    def test_same_value_and_kind_do_not_identify_the_condition(self):
        for old, new in (
            ("机械角色攻击提升13.0", "量子角色攻击提升13.0"),
            ("重炮、镰刀角色攻击提升13.0", "弓箭、环刃角色攻击提升13.0"),
            ("大剑、拳套角色攻击提升13.0", "拳套角色攻击提升13.0"),
        ):
            with self.subTest(old=old, new=new):
                a, b = self.line(old), self.line(new)
                self.assertEqual(a.kind, b.kind)
                self.assertEqual(a.shown, b.shown)
                self.assertFalse(same_affix(a, b))

    def test_same_affix_does_not_guess_through_ocr_differences(self):
        for old, new in (
            ("虚数角色攻击提升13.0", "虚敉角色攻击提升13.0"),
            ("虚数角色攻击提升13.0", "虚数角色攻击提升13,0"),
            ("虚数角色攻击提升13.0", "虚数角色攻击提升14.0"),
            ("未知角色攻击提升13.0", "未知角色攻击提升13.0"),
            ("核心", "核心"),
            ("", ""),
        ):
            with self.subTest(old=old, new=new):
                self.assertFalse(same_affix(self.line(old), self.line(new)))

    def test_menu_selection_uses_the_same_attribute_normalization(self):
        self.assertIsNone(validate_selection(" 星辰 ", "大剑、拳套"))
        self.assertIsNotNone(validate_selection("火焰", "大剑、拳套"))
        self.assertIsNotNone(validate_selection("虚数", "未知武器"))


if __name__ == "__main__":
    unittest.main()
