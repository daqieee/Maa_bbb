"""智能洗词缀：pipeline 负责界面操作，Python 负责解析和择优。

从重构结果页开始。词缀锁模式只接受新旧面板中唯一、全文相同的锁定行；
OCR 重试在当前页面完成，仍无法确认时停止，不通过继续重构来重试识别。
"""

from __future__ import annotations

import math
import time
from typing import Optional

from maa.context import Context
from maa.custom_action import CustomAction

from ..utils.AffixRules import SideResult, evaluate_side, same_affix, validate_selection
from ..utils.Logger import Logger

MODE_PLAIN = "直接洗词缀"
MODE_LOCK = "词缀锁洗词缀"


class AffixRerollV2(CustomAction):
    """按有效总攻击力择优；普通模式还可在单条高攻达标时结束。"""

    def _load_params(self, context: Context, argv: CustomAction.RunArg) -> dict:
        # UI 的各选项按 key 合并到 attach，默认值只在 pipeline 中维护。
        data = context.get_node_data(argv.node_name)
        if data is None:
            raise ValueError(f"找不到入口节点 {argv.node_name}")
        params = dict(data["attach"])
        if params["模式"] not in (MODE_PLAIN, MODE_LOCK):
            raise ValueError(f"未知的洗词缀模式：{params['模式']}")
        problem = validate_selection(params["属性"], params["武器"])
        if problem:
            raise ValueError(problem)
        for key in ("目标攻击力", "单条达标值"):
            value = float(params[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key}必须是大于 0 的有限数值")
            params[key] = value
        for key in ("最大重构次数", "OCR重试次数"):
            value = float(params[key])
            if not math.isfinite(value) or not value.is_integer() or value <= 0:
                raise ValueError(f"{key}必须是正整数")
            params[key] = int(value)
        for key in ("确认词缀锁", "单条高攻"):
            if params[key] not in ("Yes", "No"):
                raise ValueError(f"{key}必须为 Yes 或 No")
        return params

    def _screencap(self, context: Context):
        return context.tasker.controller.post_screencap().wait().get()

    def _ocr_text(self, context: Context, node: str, image) -> str:
        if context.tasker.stopping:
            return ""
        detail = context.run_recognition(node, image)
        if detail is None or not detail.hit:
            return ""
        # 词缀节点不设 expected。若置信度过滤丢掉了条件块，剩余数值
        # 即使可读也不能当成完整词缀，留在本页重试。
        if len(detail.filtered_results) != len(detail.all_results):
            return ""
        # 一条长词缀可能被 OCR 分成多个文本块。按行、从左到右拼接，
        # 避免只取 best_result 时把条件丢掉，误认成无条件攻击。
        rows = []
        for item in sorted(detail.filtered_results, key=lambda item: item.box[1]):
            center = item.box[1] + item.box[3] / 2
            if rows and abs(center - rows[-1][0]) < min(item.box[3], rows[-1][1]) / 2:
                rows[-1][2].append(item)
            else:
                rows.append((center, item.box[3], [item]))
        return "".join(
            item.text
            for _, _, row in rows
            for item in sorted(row, key=lambda item: item.box[0])
        )

    def _find_and_click(self, context: Context, node: str, logger: Logger, label: str) -> bool:
        """由节点的 OCR、timeout、Click 和 post_delay 完成识别到点击。"""
        if context.tasker.stopping:
            return False
        result = context.run_task(node)
        success = result is not None and result.status.succeeded and not context.tasker.stopping
        if not success:
            logger.error(f"未能执行「{label}」，已停止")
        return success

    def _missing_markers(self, context: Context, markers: list) -> list[str]:
        image = self._screencap(context)
        missing = []
        for node, label in markers:
            detail = context.run_recognition(node, image)
            if detail is None or not detail.hit:
                missing.append(label)
        return missing

    def _read_sides(self, context: Context, nodes: dict, attribute: str, weapon: str):
        """用同一帧读取新旧面板；不在识别过程中操作界面。"""
        image = self._screencap(context)
        return tuple(
            evaluate_side([self._ocr_text(context, node, image) for node in nodes[side]], attribute, weapon)
            for side in ("old", "new")
        )

    @staticmethod
    def _detect_locked_row(old_side: SideResult, new_side: SideResult) -> Optional[int]:
        if not old_side.reliable or not new_side.reliable:
            return None
        same = [
            i for i, (old, new) in enumerate(zip(old_side.lines, new_side.lines))
            if same_affix(old, new)
        ]
        return same[0] if len(same) == 1 else None

    @staticmethod
    def _reached(side: SideResult, target: float, single_on: bool, single_value: float) -> tuple[bool, str]:
        if side.total >= target:
            return True, f"有效总攻击力 {side.total} ≥ 目标 {target}"
        if single_on and side.best_single >= single_value:
            return True, f"单条有效攻击力 {side.best_single} ≥ 达标值 {single_value}"
        return False, ""

    @staticmethod
    def _decide_write(
        old: SideResult, new: SideResult, single_on: bool, single_value: float
    ) -> tuple[bool, str]:
        if not old.reliable or not new.reliable:
            return False, "有一侧词缀未可靠识别"
        if new.total > old.total:
            return True, f"有效总攻击力 {new.total} > {old.total}"
        if single_on and new.best_single >= single_value and new.best_single > old.best_single:
            return True, f"单条有效攻击力 {new.best_single} ≥ {single_value}"
        return False, f"新 {new.total} 不比旧 {old.total} 好"

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        logger = Logger("洗词缀V2", context)
        try:
            params = self._load_params(context, argv)
        except (KeyError, TypeError, ValueError) as exc:
            logger.ui(f"❌ 洗词缀参数不合法：{exc}", "red")
            return CustomAction.RunResult(success=False)
        if params["模式"] == MODE_LOCK and params["确认词缀锁"] != "Yes":
            logger.ui("❌ 请先在任务设置中确认使用词缀锁", "red")
            return CustomAction.RunResult(success=False)
        return CustomAction.RunResult(success=self._run_result_loop(context, logger, params))

    def _run_result_loop(self, context: Context, logger: Logger, params: dict) -> bool:
        mode = params["模式"]
        nodes = params["nodes"]
        home = nodes["homes"][mode]
        single_on = mode == MODE_PLAIN and params["单条高攻"] == "Yes"
        target, single_value = params["目标攻击力"], params["单条达标值"]
        limit = params["最大重构次数"]
        # 结果页两个按钮也是页面标记；角色/词缀配置均来自 pipeline。
        keep, write_node = (marker[0] for marker in nodes["result_markers"])
        if context.tasker.stopping:
            return False
        missing = self._missing_markers(context, nodes["result_markers"])
        if missing:
            logger.ui("❌ 请停在重构结果页后开始任务，缺少：" + "、".join(missing), "red")
            return False

        logger.ui(f"开始智能洗词缀【{mode}】：{params['属性']} / {params['武器']} / 目标 {target}")
        locked_index = None
        pending_side = None
        rerolls = 0
        while not context.tasker.stopping:
            # 对 OCR、锁定行及写入结果的复核都留在本轮，不消耗材料。
            for attempt in range(params["OCR重试次数"]):
                if context.tasker.stopping:
                    return False
                old_side, new_side = self._read_sides(context, nodes, params["属性"], params["武器"])
                problem = ""
                if not old_side.reliable:
                    problem = f"旧词缀未可靠识别：{old_side.detail()}"
                elif pending_side is not None and not all(
                    same_affix(old, expected) for old, expected in zip(old_side.lines, pending_side.lines)
                ):
                    problem = "当前旧词缀与上一轮提交的完整词缀不一致"
                elif mode == MODE_PLAIN and self._reached(old_side, target, single_on, single_value)[0]:
                    # 保留已经达标的旧词缀，不依赖新候选能否读清。
                    break
                elif not new_side.reliable:
                    problem = f"新词缀未可靠识别：{new_side.detail()}"
                elif mode == MODE_LOCK:
                    index = self._detect_locked_row(old_side, new_side)
                    if index is None:
                        problem = "无法确认唯一的锁定行（两条都变化、都没变化，或条件文字不一致）"
                    elif locked_index is not None and index != locked_index:
                        problem = "锁定行的位置与上一轮不一致"
                    else:
                        locked_index = index
                if not problem:
                    break
                logger.warning(f"第 {attempt + 1} 次复核：{problem}")
                if attempt + 1 < params["OCR重试次数"]:
                    time.sleep(0.2)
            if context.tasker.stopping:
                return False
            if problem:
                logger.ui(f"⚠️ {problem}，已停止，请检查当前词缀和锁定状态", "orange")
                return False
            pending_side = None

            logger.info(
                f"[已重构 {rerolls} 次] 旧 {old_side.total}（{old_side.detail()}） | "
                f"新 {new_side.total}（{new_side.detail()}）"
            )
            done, why = self._reached(old_side, target, single_on, single_value)
            if done:
                if not self._find_and_click(context, keep, logger, "保留旧词缀"):
                    return False
                logger.ui(f"✅ 完成：{why}", "green")
                return True

            write, why = self._decide_write(old_side, new_side, single_on, single_value)
            if write:
                logger.info(f"写入新词缀：{why}")
                if not self._find_and_click(context, write_node, logger, "写入新词缀"):
                    return False
                pending_side = new_side
            else:
                if rerolls >= limit:
                    break
                if home["fast_reroll"] is not None:
                    if not self._find_and_click(context, home["fast_reroll"], logger, "再次重构"):
                        return False
                    rerolls += 1
                    continue
                if not self._find_and_click(context, keep, logger, "保留旧词缀"):
                    return False
                pending_side = old_side

            if context.tasker.stopping:
                return False
            missing = self._missing_markers(context, home["markers"])
            if missing:
                logger.ui(f"⚠️ 提交后未回到{home['label']}，已停止：" + "、".join(missing), "orange")
                return False
            done, why = self._reached(pending_side, target, single_on, single_value)
            if done:
                logger.ui(f"✅ 已写入并返回{home['label']}：{why}", "green")
                return True
            if rerolls >= limit:
                break
            if not self._find_and_click(context, home["roll"], logger, "词缀重构"):
                return False
            rerolls += 1
            missing = self._missing_markers(context, nodes["result_markers"])
            if missing:
                logger.ui("⚠️ 未回到重构结果页（可能材料不足），已停止", "orange")
                return False

        if not context.tasker.stopping:
            logger.ui(f"⚠️ 已达到自动重构次数上限 {limit}，仍未达标，已停止", "orange")
        return False
