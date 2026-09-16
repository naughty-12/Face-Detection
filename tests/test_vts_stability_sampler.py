"""线 A 验收第 5 条（连续 ≥5 分钟稳定）的采样判据 —— 纯函数部分。

为什么需要这些测试
------------------
2026-09-16 那次 7 分钟采样的脚本**把"桥接启动前"的窗口也算成了掉线**，于是打印出
"未达 5 分钟或出现掉线，需重测"，人工核对数据后才判定为通过。

根因是判据写错了：`connectedPlugins` 是 **VTS 当前 API 会话数**（含采样器自己），
实测（2026-09-16，桥接在线时用 1/2 个探针连接验证）：

    2 个探针 + 1 个桥接 -> connectedPlugins = 3
    1 个探针 + 1 个桥接 -> connectedPlugins = 2
    只有采样器自己     -> connectedPlugins = 1     <- 桥接还没启动，不是掉线

所以正确判据是：**只统计桥接连上（`connectedPlugins >= 1 + 其它插件数`）之后的样本**，
且掉线只能记在"连上之后"的连上→断开跳变上。这些测试把这条判据钉死。
"""
import csv
import os
import shutil
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "apps", "vtube_bridge"))

import sample_vts_params as sampler  # noqa: E402


def sample(t, values, plugins=2, uptime=1000, framerate=59, allowed=1, error=None):
    """一行采样记录。plugins 是 VTS 报的会话数（含采样器自己）。"""
    return sampler.build_sample(t, values, plugins, allowed, uptime, framerate, error=error)


def run_samples(step, total, attach_at, value_of):
    """0..total 秒，attach_at 之前插件数为 1（桥接未连）且参数不动，之后为 2 且按 value_of 变化。"""
    rows = []
    t = 0.0
    while t <= total + 1e-9:
        attached = t >= attach_at
        rows.append(sample(round(t, 3),
                           value_of(t) if attached else frozen(t),
                           plugins=2 if attached else 1))
        t += step
    return rows


def moving(t):
    """两个参数在动（幅度远大于 epsilon）。"""
    return {"MouthOpen": 0.5 + 0.3 * (t % 2.0), "FaceAngleX": -10.0 + 5.0 * (t % 3.0)}


def frozen(t):
    return {"MouthOpen": 0.5, "FaceAngleX": -10.0}


class TestPreAttachSamplesAreNotDisconnects(unittest.TestCase):
    """这条就是当初打印"需重测"的那个 bug。"""

    def test_the_bridge_not_started_yet_is_not_a_disconnect(self):
        rows = run_samples(0.5, total=420.0, attach_at=90.0, value_of=moving)

        result = sampler.evaluate_stability(rows, window_seconds=30.0,
                                            required_stable_seconds=300.0)

        self.assertEqual(result["disconnects"], [],
                         "plugins == 1 表示桥接还没连上，不是掉线")
        self.assertTrue(result["attached"])
        self.assertAlmostEqual(result["attached_at"], 90.0, places=3)

    def test_windows_are_anchored_at_the_moment_the_bridge_attaches(self):
        rows = run_samples(0.5, total=330.0, attach_at=30.0, value_of=moving)

        result = sampler.evaluate_stability(rows, window_seconds=30.0)

        self.assertEqual(result["windows_total"], 10)
        self.assertTrue(result["windows"][0]["ok"],
                        "第一个窗口必须从连上的那一刻算起，而不是从脚本启动算起")
        self.assertAlmostEqual(result["windows"][0]["start"], 30.0, places=3)


class TestStabilityVerdict(unittest.TestCase):
    def test_ten_clean_windows_after_attach_pass(self):
        rows = run_samples(0.2, total=390.0, attach_at=90.0, value_of=moving)

        result = sampler.evaluate_stability(rows, window_seconds=30.0,
                                            required_stable_seconds=300.0)

        self.assertEqual((result["windows_ok"], result["windows_total"]), (10, 10))
        self.assertAlmostEqual(result["stable_seconds"], 300.0, delta=0.5)
        self.assertTrue(result["passed"])
        self.assertEqual(result["reasons"], [])

    def test_a_disconnect_after_attach_is_recorded_at_its_time(self):
        rows = [sample(t / 2.0, moving(t / 2.0), plugins=(2 if t / 2.0 < 120.0 else 1))
                for t in range(0, 361)]

        result = sampler.evaluate_stability(rows, window_seconds=30.0,
                                            required_stable_seconds=300.0)

        self.assertEqual(len(result["disconnects"]), 1)
        self.assertAlmostEqual(result["disconnects"][0], 120.0, places=3)
        self.assertAlmostEqual(result["stable_seconds"], 120.0, delta=0.5)
        self.assertFalse(result["passed"],
                         "只连了 2 分钟不能算通过 5 分钟稳定性")

    def test_a_frozen_window_fails_and_names_itself(self):
        rows = []
        t = 0.0
        while t <= 120.0:
            values = frozen(t) if 60.0 <= t < 90.0 else moving(t)
            rows.append(sample(round(t, 3), values))
            t += 0.5

        result = sampler.evaluate_stability(rows, window_seconds=30.0,
                                            required_stable_seconds=60.0)

        self.assertEqual(result["windows_total"], 4)
        self.assertEqual(result["windows_ok"], 3)
        self.assertFalse(result["passed"])
        self.assertFalse(result["windows"][2]["ok"], "人脸离开画面的那 30 秒参数不会变")
        self.assertEqual(result["windows"][2]["changed"], [])
        self.assertTrue(any("window 2" in reason for reason in result["reasons"]),
                        f"理由里要点名是哪个窗口，实际 reasons={result['reasons']}")

    def test_min_changed_params_is_respected(self):
        one_moving = lambda t: {"MouthOpen": 0.5 + 0.3 * (t % 2.0), "FaceAngleX": -10.0}
        two_moving = lambda t: {"MouthOpen": 0.5 + 0.3 * (t % 2.0),
                               "FaceAngleX": -10.0 + 5.0 * (t % 3.0)}
        rows = [sample(t / 2.0, one_moving(t / 2.0)) for t in range(0, 61)]

        strict = sampler.evaluate_stability(rows, window_seconds=30.0, min_changed_params=2)
        rows2 = [sample(t / 2.0, two_moving(t / 2.0)) for t in range(0, 61)]
        loose = sampler.evaluate_stability(rows2, window_seconds=30.0, min_changed_params=2)

        self.assertEqual(strict["windows_ok"], 0)
        self.assertEqual(loose["windows_ok"], 1)

    def test_span_exactly_at_epsilon_does_not_count_as_change(self):
        jitter = lambda t: {"MouthOpen": 0.5 + (0.02 if t >= 10.0 else 0.0)}
        rows = [sample(t / 2.0, jitter(t / 2.0)) for t in range(0, 61)]

        result = sampler.evaluate_stability(rows, window_seconds=30.0, epsilon=0.02)

        self.assertAlmostEqual(result["windows"][0]["spans"]["MouthOpen"], 0.02, places=6)
        self.assertEqual(result["windows"][0]["changed"], [],
                         "跨度恰好等于 epsilon 视为没变（epsilon 是严格大于的门槛）")

    def test_vts_restart_is_detected_from_a_dropping_uptime(self):
        rows = [sample(t / 2.0, moving(t / 2.0), uptime=(100000 if t / 2.0 < 60.0 else 500))
                for t in range(0, 121)]

        result = sampler.evaluate_stability(rows, window_seconds=30.0)

        self.assertEqual(len(result["vts_restarts"]), 1)
        self.assertAlmostEqual(result["vts_restarts"][0], 60.0, places=3)

    def test_stable_seconds_is_the_longest_contiguous_attached_run(self):
        rows = []
        for i in range(0, 1321):                       # 0..660 s，步长 0.5
            t = i / 2.0
            plugins = 1 if 200.0 <= t < 260.0 else 2
            rows.append(sample(t, moving(t), plugins=plugins))

        result = sampler.evaluate_stability(rows, window_seconds=30.0,
                                            required_stable_seconds=300.0)

        self.assertAlmostEqual(result["stable_seconds"], 400.0, delta=0.5,
                               msg="连续时长取最长的一段，不是所有连上时间之和")
        self.assertAlmostEqual(result["total_attached_seconds"], 600.0, delta=0.5)
        self.assertTrue(result["passed"])
        self.assertEqual(result["windows_total"], 13, "窗口只判最长的那一段")

    def test_trailing_partial_window_is_not_judged(self):
        rows = [sample(t / 2.0, moving(t / 2.0)) for t in range(0, 621)]   # 0..310 s

        result = sampler.evaluate_stability(rows, window_seconds=30.0)

        self.assertEqual(result["windows_total"], 10)
        self.assertAlmostEqual(result["partial_seconds"], 10.0, delta=0.5)

    def test_short_run_fails_with_a_reason(self):
        rows = [sample(t / 2.0, moving(t / 2.0)) for t in range(0, 241)]   # 0..120 s

        result = sampler.evaluate_stability(rows, window_seconds=30.0,
                                            required_stable_seconds=300.0)

        self.assertFalse(result["passed"])
        self.assertTrue(any("300" in reason for reason in result["reasons"]),
                        f"理由里要说清要求了多久，实际 reasons={result['reasons']}")

    def test_no_bridge_at_all_is_reported_as_not_attached(self):
        rows = [sample(t / 2.0, frozen(t / 2.0), plugins=1) for t in range(0, 121)]

        result = sampler.evaluate_stability(rows, window_seconds=30.0)

        self.assertFalse(result["attached"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["windows_total"], 0)
        self.assertTrue(any("never attached" in reason for reason in result["reasons"]))

    def test_a_sampling_gap_breaks_continuity_even_if_the_bridge_looked_connected(self):
        rows = [sample(t / 2.0, moving(t / 2.0)) for t in range(0, 241)]              # 0..120 s
        rows += [sample(400.0 + t / 2.0, moving(t / 2.0)) for t in range(0, 241)]     # 400..520 s

        result = sampler.evaluate_stability(rows, window_seconds=30.0, max_gap_seconds=2.0,
                                            required_stable_seconds=100.0)

        self.assertEqual(len(result["sampling_gaps"]), 1)
        self.assertAlmostEqual(result["stable_seconds"], 120.0, delta=0.5,
                               msg="中间没采到样的那 280 秒不能算进连续时长")
        self.assertEqual(result["windows_total"], 4)
        self.assertTrue(result["passed"])

    def test_a_request_error_is_not_reported_as_a_bridge_disconnect(self):
        rows = [sample(t / 2.0, moving(t / 2.0)) for t in range(0, 61)]
        rows[30] = sample(15.0, {}, plugins=None, error="connection reset by peer")

        result = sampler.evaluate_stability(rows, window_seconds=30.0)

        self.assertEqual(result["disconnects"], [],
                         "请求失败说明不了桥接掉线，只能记成'这一行没采到'")
        self.assertEqual(len(result["sample_errors"]), 1)
        self.assertAlmostEqual(result["sample_errors"][0], 15.0, places=3)


class TestEvidenceFiles(unittest.TestCase):
    """证据文件要真能落盘 —— 用固定的工作目录，不用 mkdtemp（本机沙箱下那个目录写不进去）。"""

    TEST_DIR = os.path.join(PROJECT_ROOT, "artifacts", "diag", "_sampler_test")

    def setUp(self):
        os.makedirs(self.TEST_DIR, exist_ok=True)
        self.addCleanup(shutil.rmtree, self.TEST_DIR, True)

    def path(self, name):
        return os.path.join(self.TEST_DIR, name)

    def test_csv_round_trip_keeps_every_parameter(self):
        parameters = ["MouthOpen", "FaceAngleX"]
        rows = [sample(0.0, {"MouthOpen": 0.25, "FaceAngleX": -3.5}),
                sample(0.2, {"MouthOpen": 0.75, "FaceAngleX": 7.25})]

        sampler.write_samples_csv(self.path("samples.csv"), rows, parameters)

        with open(self.path("samples.csv"), "r", encoding="utf-8", newline="") as handle:
            table = list(csv.DictReader(handle))
        self.assertEqual(len(table), 2)
        self.assertEqual(float(table[1]["MouthOpen"]), 0.75)
        self.assertEqual(float(table[1]["FaceAngleX"]), 7.25)
        self.assertEqual(int(table[1]["connectedPlugins"]), 2)

    def test_csv_records_a_failed_sample_as_unknown_rather_than_zero(self):
        parameters = ["MouthOpen"]
        rows = [sample(0.0, {"MouthOpen": 0.5}, plugins=None, error="boom")]

        sampler.write_samples_csv(self.path("failed.csv"), rows, parameters)

        with open(self.path("failed.csv"), "r", encoding="utf-8", newline="") as handle:
            table = list(csv.DictReader(handle))
        self.assertEqual(table[0]["connectedPlugins"], "")
        self.assertEqual(table[0]["error"], "boom")


class FakeClient:
    """只实现采样器用到的那两个请求，用来在真实时间之外驱动采集循环。"""

    def __init__(self, fail_at=None, vts_parameters=None):
        self.fail_at = fail_at or []
        self.calls = 0
        self.vts_parameters = vts_parameters or ["MouthOpen", "FaceAngleX"]

    def request(self, message_type, data=None):
        self.calls += 1
        if self.calls in self.fail_at:
            raise RuntimeError("connection reset by peer")
        if message_type == "StatisticsRequest":
            return {"connectedPlugins": 2, "allowedPlugins": 1, "uptime": 123456,
                    "framerate": 59, "vTubeStudioVersion": "1.35.10"}
        if message_type == "InputParameterListRequest":
            return {"modelLoaded": True, "modelName": "hiyori",
                    "defaultParameters": [{"name": name} for name in self.vts_parameters]}
        if message_type == "ParameterValueRequest":
            return {"name": data["name"], "value": 0.5, "min": 0.0, "max": 1.0}
        raise AssertionError(f"unexpected request: {message_type}")


def fake_clock(step):
    """返回 (clock, sleep)：sleep 推进假时间，于是采集循环可以离线跑完。"""
    state = {"now": 0.0}

    def clock():
        return state["now"]

    def sleep(seconds):
        state["now"] += max(seconds, step)

    return clock, sleep


class TestCollectSamples(unittest.TestCase):
    def test_collects_until_the_duration_is_reached(self):
        client = FakeClient()
        clock, sleep = fake_clock(0.2)

        rows = sampler.collect_samples(client, ["MouthOpen"], duration=1.0, interval=0.2,
                                       stats_interval=0.4, clock=clock, sleep=sleep,
                                       log=lambda message: None)

        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[-1]["values"]["MouthOpen"], 0.5)
        self.assertEqual(rows[-1]["uptime_ms"], 123456)

    def test_a_transient_request_error_does_not_end_the_run(self):
        client = FakeClient(fail_at=[2])          # 第 2 个请求 = 本轮读参数，状态那一拍是成功的
        clock, sleep = fake_clock(0.2)
        logged = []

        rows = sampler.collect_samples(client, ["MouthOpen"], duration=1.0, interval=0.2,
                                       stats_interval=0.4, clock=clock, sleep=sleep,
                                       log=logged.append)

        self.assertEqual(len(rows), 5)
        self.assertTrue(logged, "出错要留下日志，不能静默")
        self.assertIsNotNone(rows[0]["error"])
        self.assertEqual(rows[0]["values"], {}, "这一拍没采到值，不能拿上一拍的值糊过去")
        self.assertEqual(rows[0]["connected_plugins"], 2,
                         "这一拍的状态是读到了的：读参数失败不等于连接状态未知")
        self.assertIsNone(rows[-1]["error"])

    def test_a_failed_state_request_leaves_the_connection_state_unknown(self):
        client = FakeClient(fail_at=[1])          # 第 1 个请求就是 StatisticsRequest
        clock, sleep = fake_clock(0.2)

        rows = sampler.collect_samples(client, ["MouthOpen"], duration=1.0, interval=0.2,
                                       stats_interval=0.4, clock=clock, sleep=sleep,
                                       log=lambda message: None)

        self.assertEqual(len(rows), 5)
        self.assertIsNone(rows[0]["connected_plugins"],
                          "连状态都没读到就必须写未知，不能当成'还连着'")
        self.assertFalse(sampler.bridge_attached(rows[0]))
        self.assertIsNone(rows[-1]["error"], "后面几拍要恢复正常，不能一直躺下")

    def test_unknown_parameter_names_are_reported(self):
        client = FakeClient(vts_parameters=["MouthOpen"])
        clock, sleep = fake_clock(0.2)

        missing = sampler.check_parameter_names(client, ["MouthOpen", "NotAParameter"])

        self.assertEqual(missing, ["NotAParameter"])


class TestIncrementalEvidence(unittest.TestCase):
    """原始数据必须**边采边落盘**：跑到一半崩了也不能整批丢掉（失误 8 的教训）。"""

    TEST_DIR = os.path.join(PROJECT_ROOT, "artifacts", "diag", "_sampler_test")

    def setUp(self):
        os.makedirs(self.TEST_DIR, exist_ok=True)
        self.addCleanup(shutil.rmtree, self.TEST_DIR, True)

    def read_rows(self, path):
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_rows_reach_the_disk_before_the_run_ends(self):
        path = os.path.join(self.TEST_DIR, "incremental.csv")
        writer = sampler.IncrementalCsvWriter(path, ["MouthOpen"])

        writer(sample(0.0, {"MouthOpen": 0.1}))
        writer(sample(0.2, {"MouthOpen": 0.2}))

        rows = self.read_rows(path)
        self.assertEqual(len(rows), 2, "没 close 之前就已经写下去两行了")

        writer(sample(0.4, {"MouthOpen": 0.3}))
        writer.close()
        self.assertEqual(len(self.read_rows(path)), 3)

    def test_every_sample_is_handed_over_as_soon_as_it_is_taken(self):
        delivered = []
        client = FakeClient()
        clock, sleep = fake_clock(0.2)

        def sink(row):
            delivered.append(row)
            if len(delivered) == 3:
                raise RuntimeError("simulated crash")

        with self.assertRaises(RuntimeError):
            sampler.collect_samples(client, ["MouthOpen"], duration=5.0, interval=0.2,
                                   stats_interval=1.0, clock=clock, sleep=sleep,
                                   log=lambda message: None, on_sample=sink)

        self.assertEqual([row["t"] for row in delivered], [0.0, 0.2, 0.4],
                         "采到一行就交一行，不能攒到最后一起给")
        self.assertEqual([row["values"]["MouthOpen"] for row in delivered], [0.5, 0.5, 0.5])


class TestConsoleEncoding(unittest.TestCase):
    """本机实测：控制台代码页是 65001(UTF-8)，而 Python 的 stdout 却是 gbk → 中文乱码。"""

    def test_utf8_console_with_gbk_stdout_needs_fixing(self):
        self.assertTrue(sampler.needs_utf8_console(65001, "gbk"))

    def test_matching_encodings_need_no_fix(self):
        self.assertFalse(sampler.needs_utf8_console(65001, "utf-8"))
        self.assertFalse(sampler.needs_utf8_console(65001, "UTF-8"))

    def test_a_gbk_console_is_left_alone(self):
        self.assertFalse(sampler.needs_utf8_console(936, "utf-8"),
                         "控制台是 GBK 时改成 UTF-8 只是把乱码换到另一边")
        self.assertFalse(sampler.needs_utf8_console(936, "gbk"))


class TestArguments(unittest.TestCase):
    def test_defaults_leave_margin_over_the_five_minute_line(self):
        args = sampler.parse_args([])

        self.assertEqual(args.duration, 600.0)
        self.assertEqual(args.window, 30.0)
        self.assertEqual(args.required_stable, 300.0)
        self.assertGreater(args.duration - args.required_stable, 60.0,
                           "采样时长要明显长于要求，否则又只能'正好卡线'")

    def test_parameter_list_accepts_commas_and_spaces(self):
        self.assertEqual(sampler.parse_parameter_list("MouthOpen, FaceAngleX ,EyeLeftX"),
                         ["MouthOpen", "FaceAngleX", "EyeLeftX"])
        args = sampler.parse_args(["--parameters", "MouthOpen FaceAngleX"])
        self.assertEqual(args.parameters, ["MouthOpen", "FaceAngleX"])

    def test_default_parameters_are_ones_the_bridge_actually_drives(self):
        from vtube_studio_bridge.vtube_studio_bridge import TRACKING_PARAMETER_NAMES

        for name in sampler.DEFAULT_PARAMETERS:
            self.assertIn(name, TRACKING_PARAMETER_NAMES)


if __name__ == "__main__":
    unittest.main()
