"""线 A 验收第 5 条（连续 ≥5 分钟稳定）的采样器：只读观察 VTS 参数与连接状态并出证据文件。

为什么需要它
------------
2026-09-16 的 7 分钟采样实测通过（10/10 个 30 s 窗口参数在变化、无掉线、连续 ≈5.0 分钟），
但那次采样**是临时脚本跑的**：脚本没进仓库、原始数据没落盘，而它打印的结论行还是错的 ——
它把"桥接启动前"的窗口也算成掉线，于是输出"未达 5 分钟或出现掉线，需重测"，靠人工核对才改判。
本文件把那次的判据固化成可复现的工具：脚本入库、原始 CSV 落盘、结论行只按正确判据算。

判据（`connectedPlugins` 的语义是实测出来的，不是猜的）
------------------------------------------------------
`connectedPlugins` 来自 `StatisticsRequest`，含义是 **VTS 当前活着的 API 会话数（含采样器自己）**。
2026-09-16 在本机（桥接在线、VTS 1.35.10）用 1/2 个探针连接实测：

    2 个探针 + 1 个桥接 -> connectedPlugins = 3, allowedPlugins = 1
    1 个探针 + 1 个桥接 -> connectedPlugins = 2, allowedPlugins = 1
    只有采样器自己     -> connectedPlugins = 1

所以 `connectedPlugins >= 1 + 其它插件数`（默认其它插件数 = 1，即桥接）才是"桥接连上了"。
**启动后 1 分钟里 plugins == 1 不是掉线**，那只是桥接还没起来。

顺带记下的三个字段（同为 `StatisticsRequest` 返回，实测确认）：
`uptime`（毫秒，重启会归零/回退 → 可判定 VTS 重启）、`framerate`（VTS 自身渲染帧率）、
`vTubeStudioVersion`。

读参数用的是 `ParameterValueRequest`，键名是 **`name`**（实测：传 `id` 会得到
`APIError 500 "You have to provide a parameter name."`），而且**一次只能读一个参数**。

用法（必须在**使用者自己的终端**里跑；采样器自己不开摄像头）
----------------------------------------------------------
    # 1) 先启动采样器（它会等你把桥接起来）
    python apps/vtube_bridge/sample_vts_params.py
    # 2) 另开一个终端启动桥接
    python apps/vtube_bridge/main.py --input 0 --sink vts --no-gui

默认采样 **600 秒**（要求 300 秒，留 5 分钟余量，免得又"正好卡线"），0.2 秒一个参数样本。
采样器用桥接自己的插件身份与已缓存 token 连 VTS，所以**不会再弹一次授权窗**；
首次运行（没有 token）时才需要点一次"允许"。

输出（`artifacts/diag/`，文件名带时间戳，一次运行一套）
* `vts_stability_<时间>.csv`：**原始数据** —— 每分钟 300 行，含连接状态、uptime、framerate 与各参数值；
* `vts_stability_<时间>.json`：判据参数 + 逐窗口结论 + 总判定（机器可读）；
* `vts_stability_<时间>.txt`：控制台报告的副本（给人看）。

CSV 是**边采边写、每行 flush** 的：中途 Ctrl+C 或崩掉，已经采到的数据仍在盘上
（"跑完才落盘"正是 `.dsh-dev/mistakes.md` 失误 8 丢掉证据的形态）。

退出码：0 = 通过，2 = 未通过（数据仍然落盘），1 = 运行失败（连不上 VTS 等）。
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

BRIDGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BRIDGE_ROOT.parents[1]
DIAG_DIR = PROJECT_ROOT / "artifacts" / "diag"

# 默认追踪的参数：都是桥接真的在驱动、且实测会变化的量（人脸离开画面时它们会冻住，
# 这正是"稳定性"要抓的情况）。参数越多，每个采样周期的请求越多，故默认只取 5 个。
DEFAULT_PARAMETERS = ["MouthOpen", "FaceAngleX", "FaceAngleY", "EyeOpenLeft", "EyeLeftX"]
DEFAULT_DURATION = 600.0
DEFAULT_INTERVAL = 0.2
DEFAULT_STATS_INTERVAL = 1.0
DEFAULT_WINDOW = 30.0
DEFAULT_EPSILON = 0.02
DEFAULT_REQUIRED_STABLE = 300.0
DEFAULT_OTHER_PLUGINS = 1
FLOAT_GUARD = 1e-9          # 抵消二进制浮点误差，让 epsilon 是"严格大于"的门槛

CSV_COLUMNS = ["t", "connectedPlugins", "allowedPlugins", "uptimeMs", "framerate"]


def parse_parameter_list(text):
    """接受 "A,B" 或 "A B" 两种写法，去掉空项。"""
    if not text:
        return []
    names = []
    for chunk in text.replace(",", " ").split():
        name = chunk.strip()
        if name:
            names.append(name)
    return names


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Sample VTube Studio parameters over time and judge the >=5 minute stability criterion.")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help=f"总共采样多少秒（默认 {DEFAULT_DURATION:.0f}）")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help=f"每个采样周期的间隔秒数（默认 {DEFAULT_INTERVAL}）")
    parser.add_argument("--stats-interval", type=float, default=DEFAULT_STATS_INTERVAL,
                        help=f"多久读一次 StatisticsRequest（默认 {DEFAULT_STATS_INTERVAL}）")
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW,
                        help=f"判据里的窗口长度秒数（默认 {DEFAULT_WINDOW:.0f}）")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                        help=f"窗口内参数跨度大于该值才算「有变化」（默认 {DEFAULT_EPSILON}）")
    parser.add_argument("--min-changed-params", type=int, default=1,
                        help="每个窗口至少要有几个参数在变化（默认 1）")
    parser.add_argument("--required-stable", type=float, default=DEFAULT_REQUIRED_STABLE,
                        help=f"要求连续稳定多少秒（默认 {DEFAULT_REQUIRED_STABLE:.0f}）")
    parser.add_argument("--other-plugins", type=int, default=DEFAULT_OTHER_PLUGINS,
                        help=f"除采样器外应连接的插件数，用于判定'桥接连上了'（默认 {DEFAULT_OTHER_PLUGINS}）")
    parser.add_argument("--parameters", default=",".join(DEFAULT_PARAMETERS),
                        help=f"要采样的参数，逗号或空格分隔（默认 {','.join(DEFAULT_PARAMETERS)}）")
    parser.add_argument("--out-dir", default=str(DIAG_DIR), help=f"证据文件输出目录（默认 {DIAG_DIR}）")
    parser.add_argument("--host", default="127.0.0.1", help="VTS 主机（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8001, help="VTS API 端口（默认 8001）")
    args = parser.parse_args(argv)
    args.parameters = parse_parameter_list(args.parameters)
    if not args.parameters:
        parser.error("--parameters 不能为空")
    return args


def build_sample(t, values, connected_plugins, allowed_plugins, uptime_ms, framerate, error=None):
    """一行采样记录。连接状态未知时一律写 None（**不要写 0 或上一轮的值**：未知就是未知）。"""
    return {
        "t": t,
        "values": dict(values or {}),
        "connected_plugins": connected_plugins,
        "allowed_plugins": allowed_plugins,
        "uptime_ms": uptime_ms,
        "framerate": framerate,
        "error": error,
    }


def bridge_attached(sample, other_plugins=DEFAULT_OTHER_PLUGINS):
    """连接状态已知且会话数达标才算"桥接连上了"。"""
    plugins = sample.get("connected_plugins")
    if plugins is None:
        return False
    return plugins >= 1 + other_plugins


def read_statistics(client):
    """StatisticsRequest -> 连接状态与运行时长（字段名均为实测所得）。"""
    data = client.request("StatisticsRequest")
    return {
        "connectedPlugins": data.get("connectedPlugins"),
        "allowedPlugins": data.get("allowedPlugins"),
        "uptime": data.get("uptime"),
        "framerate": data.get("framerate"),
        "vTubeStudioVersion": data.get("vTubeStudioVersion"),
    }


def read_parameter_values(client, parameters):
    """逐个读回参数当前值。注意键名是 name（传 id 会被 VTS 拒绝），且一次只返回一个参数。"""
    values = {}
    for name in parameters:
        data = client.request("ParameterValueRequest", {"name": name})
        value = data.get("value")
        if value is not None:
            values[name] = float(value)
    return values


def check_parameter_names(client, parameters):
    """返回 VTS 当前模型里不存在的参数名（模型换了就可能在半路失效）。"""
    data = client.request("InputParameterListRequest")
    available = set()
    for section in ("defaultParameters", "customParameters"):
        for parameter in data.get(section, []):
            name = parameter.get("name")
            if name:
                available.add(name)
    return [name for name in parameters if name not in available]


def collect_samples(client, parameters, duration, interval=DEFAULT_INTERVAL,
                    stats_interval=DEFAULT_STATS_INTERVAL, clock=None, sleep=None, log=None,
                    on_sample=None):
    """按固定间隔采样，直到累计时长到达 duration。

    单个请求失败**不终止整轮**：那一行记成 error（连接状态留空），下一拍继续 ——
    掉线本来就该被记录下来并打断"连续"判定，而不是让脚本崩掉、什么都不留下。

    `on_sample` 每采到一行就调用一次（主流程用它把原始数据立刻写盘）。
    """
    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    log = log or print
    rows = []
    started = clock()
    next_stats = started
    stats = {}
    while True:
        now = clock()
        elapsed = now - started
        if elapsed >= duration:
            break

        error = None
        if now >= next_stats - FLOAT_GUARD:
            next_stats = now + stats_interval
            try:
                stats = read_statistics(client)
            except Exception as exc:                      # noqa: BLE001 - 任何网络异常都只影响这一行
                error = f"StatisticsRequest failed: {exc}"
                stats = {}

        values = {}
        if error is None:
            try:
                values = read_parameter_values(client, parameters)
            except Exception as exc:                      # noqa: BLE001
                error = f"ParameterValueRequest failed: {exc}"

        if error:
            log(f"[sample] {elapsed:7.1f} s  {error}")
        row = build_sample(round(elapsed, 4), values,
                           stats.get("connectedPlugins"), stats.get("allowedPlugins"),
                           stats.get("uptime"), stats.get("framerate"), error=error)
        rows.append(row)
        if on_sample is not None:
            on_sample(row)
        sleep(interval)
    return rows


def _attached_runs(samples, other_plugins, max_gap_seconds):
    """把样本切成若干"连续连上"的区间（用下标表示），返回 (runs, gaps)。

    区间在三种情况下断开：连接状态未知（该行采样失败）、会话数不达标、
    或者两行样本之间的时间间隔超过 max_gap_seconds（脚本自己卡住了，不能算"连续"）。
    """
    runs = []
    gaps = []
    start = None
    for index, item in enumerate(samples):
        if bridge_attached(item, other_plugins):
            if start is None:
                start = index
            elif max_gap_seconds is not None:
                delta = item["t"] - samples[index - 1]["t"]
                if delta > max_gap_seconds:
                    runs.append((start, index - 1))
                    gaps.append({"from": samples[index - 1]["t"], "to": item["t"]})
                    start = index
        elif start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(samples) - 1))
    return runs, gaps


def _window_spans(rows):
    """窗口内每个参数的最大值减最小值（只统计窗口里出现过的参数）。"""
    spans = {}
    for row in rows:
        for name, value in row["values"].items():
            low, high = spans.get(name, (value, value))
            spans[name] = (min(low, value), max(high, value))
    return {name: high - low for name, (low, high) in spans.items()}


def _judge_windows(samples, run, window_seconds, epsilon, min_changed_params):
    """只判"最长连续区间"里的完整窗口；末尾不足一个窗口的余量单独报告。"""
    first, last = run
    start = samples[first]["t"]
    end = samples[last]["t"]
    windows = []
    index = 0
    while start + (index + 1) * window_seconds <= end + FLOAT_GUARD:
        window_start = start + index * window_seconds
        window_end = window_start + window_seconds
        rows = [item for item in samples[first:last + 1]
                if window_start - FLOAT_GUARD <= item["t"] < window_end - FLOAT_GUARD]
        spans = _window_spans(rows)
        changed = sorted(name for name, span in spans.items() if span > epsilon + FLOAT_GUARD)
        windows.append({
            "index": index,
            "start": round(window_start, 4),
            "end": round(window_end, 4),
            "spans": {name: round(span, 4) for name, span in sorted(spans.items())},
            "changed": changed,
            "ok": len(changed) >= min_changed_params,
        })
        index += 1
    return windows


def evaluate_stability(samples, window_seconds=DEFAULT_WINDOW, epsilon=DEFAULT_EPSILON,
                       min_changed_params=1, required_stable_seconds=DEFAULT_REQUIRED_STABLE,
                       other_plugins=DEFAULT_OTHER_PLUGINS, max_gap_seconds=None):
    """按第 5 条验收判据给结论。

    判据：只统计**桥接连上之后**的样本；连续时长取**最长的一段**（断线/采样中断要重新计）；
    该段里的每个完整 30 s 窗口都必须至少有一个参数在动。
    """
    reasons = []
    runs, gaps = _attached_runs(samples, other_plugins, max_gap_seconds)

    disconnects = []
    sample_errors = []
    vts_restarts = []
    was_attached = False
    previous_uptime = None
    for item in samples:
        if item.get("error"):
            sample_errors.append(item["t"])
        uptime = item.get("uptime_ms")
        if uptime is not None:
            if previous_uptime is not None and uptime < previous_uptime:
                vts_restarts.append(item["t"])
            previous_uptime = uptime
        if bridge_attached(item, other_plugins):
            was_attached = True
        elif was_attached:
            if item.get("connected_plugins") is not None:
                disconnects.append(item["t"])
            was_attached = False

    if not runs:
        reasons.append(f"the bridge never attached: connectedPlugins never reached "
                       f"{1 + other_plugins} (sampler alone = 1)")
        return {
            "attached": False, "attached_at": None, "last_attached_at": None,
            "stable_seconds": 0.0, "total_attached_seconds": 0.0,
            "required_seconds": required_stable_seconds,
            "disconnects": disconnects, "sample_errors": sample_errors,
            "vts_restarts": vts_restarts, "sampling_gaps": gaps,
            "windows": [], "windows_total": 0, "windows_ok": 0, "partial_seconds": 0.0,
            "passed": False, "reasons": reasons,
        }

    certified = max(runs, key=lambda run: samples[run[1]]["t"] - samples[run[0]]["t"])
    attached_at = samples[certified[0]]["t"]
    last_attached_at = samples[certified[1]]["t"]
    stable_seconds = last_attached_at - attached_at
    total_attached = sum(samples[run[1]]["t"] - samples[run[0]]["t"] for run in runs)
    windows = _judge_windows(samples, certified, window_seconds, epsilon, min_changed_params)
    windows_ok = sum(1 for window in windows if window["ok"])
    partial_seconds = stable_seconds - len(windows) * window_seconds

    if stable_seconds < required_stable_seconds:
        reasons.append(f"longest continuous stretch is {stable_seconds:.1f} s, "
                       f"below the required {required_stable_seconds:.0f} s")
    for window in windows:
        if not window["ok"]:
            reasons.append(f"window {window['index']} ({window['start']:.1f}-{window['end']:.1f} s) "
                           f"showed no change in any of the tracked parameters "
                           f"(need >= {min_changed_params} changed)")
    if not windows:
        reasons.append(f"no complete {window_seconds:.0f} s window inside the longest stretch")

    passed = (stable_seconds >= required_stable_seconds and bool(windows)
              and windows_ok == len(windows))
    return {
        "attached": True,
        "attached_at": attached_at,
        "last_attached_at": last_attached_at,
        "stable_seconds": round(stable_seconds, 4),
        "total_attached_seconds": round(total_attached, 4),
        "required_seconds": required_stable_seconds,
        "disconnects": disconnects,
        "sample_errors": sample_errors,
        "vts_restarts": vts_restarts,
        "sampling_gaps": gaps,
        "windows": windows,
        "windows_total": len(windows),
        "windows_ok": windows_ok,
        "partial_seconds": round(partial_seconds, 4),
        "passed": passed,
        "reasons": reasons,
    }


def csv_header(parameters):
    return CSV_COLUMNS + list(parameters) + ["error"]


def _csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def csv_row(item, parameters):
    row = [f"{item['t']:.4f}", _csv_cell(item["connected_plugins"]), _csv_cell(item["allowed_plugins"]),
           _csv_cell(item["uptime_ms"]), _csv_cell(item["framerate"])]
    row += [_csv_cell(item["values"].get(name)) for name in parameters]
    row.append(item.get("error") or "")
    return row


class IncrementalCsvWriter:
    """边采边写：每写一行就 flush。

    为什么要这样：原始数据是这个验收**唯一的硬证据**，"跑完再一次性落盘"意味着中途崩掉
    （或被 Ctrl+C）就什么证据都不剩 —— 那正是 `.dsh-dev/mistakes.md` 失误 8 的形态。
    """

    def __init__(self, path, parameters):
        self.path = path
        self.parameters = list(parameters)
        self.handle = None
        self.writer = None

    def _open(self):
        if self.handle is None:
            self.handle = open(self.path, "w", encoding="utf-8", newline="")
            self.writer = csv.writer(self.handle)
            self.writer.writerow(csv_header(self.parameters))

    def __call__(self, item):
        self._open()
        self.writer.writerow(csv_row(item, self.parameters))
        self.handle.flush()

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
            self.writer = None


def write_samples_csv(path, samples, parameters):
    """一次性把已有样本写盘（`IncrementalCsvWriter` 的批量版本，两者共用同一套行格式）。"""
    writer = IncrementalCsvWriter(path, parameters)
    try:
        for item in samples:
            writer(item)
    finally:
        writer.close()


def format_report(result, samples, parameters, settings):
    """给人看的报告：先说结论，再列窗口，最后给复现方式。"""
    lines = []
    lines.append("[vts-stability] 判据：connectedPlugins >= %d 才算桥接连上（采样器自己占 1）"
                 % (1 + settings["other_plugins"]))
    lines.append(f"[vts-stability] 参数={', '.join(parameters)}  "
                 f"采样 {settings['duration']:.0f} s / {len(samples)} 行  "
                 f"窗口 {settings['window']:.0f} s / epsilon {settings['epsilon']}")
    lines.append("")
    if not result["attached"]:
        lines.append("[verdict] NOT ATTACHED - 采样期间没看到桥接连上")
    else:
        lines.append(f"[verdict] {'PASS' if result['passed'] else 'FAIL'}  "
                     f"连续 {result['stable_seconds']:.1f} s（要求 {result['required_seconds']:.0f} s）"
                     f"  窗口 {result['windows_ok']}/{result['windows_total']} 通过")
        lines.append(f"          桥接连上于 {result['attached_at']:.1f} s，"
                     f"最长一段结束于 {result['last_attached_at']:.1f} s，"
                     f"末尾余量 {result['partial_seconds']:.1f} s")
        lines.append(f"          所有连上时间合计 {result['total_attached_seconds']:.1f} s  "
                     f"掉线 {len(result['disconnects'])} 次  "
                     f"采样失败 {len(result['sample_errors'])} 行  "
                     f"VTS 重启 {len(result['vts_restarts'])} 次  "
                     f"采样中断 {len(result['sampling_gaps'])} 次")
    for reason in result["reasons"]:
        lines.append(f"[reason] {reason}")
    if result["windows"]:
        lines.append("")
        for window in result["windows"]:
            spans = " ".join(f"{name}={span:g}" for name, span in window["spans"].items())
            lines.append(f"  window {window['index']:>2}  {window['start']:7.1f}-{window['end']:7.1f} s  "
                         f"{'ok ' if window['ok'] else 'BAD'}  {spans}")
    lines.append("")
    lines.append("[how-to-read]")
    lines.append("  * 启动后 plugins == 1 只是桥接还没起来，不是掉线（判据只从连上那一刻算起）")
    lines.append("  * 某窗口 span 全为 0 → 那 30 秒参数没动（人离开画面 / 桥接停了 / 模型被换掉）")
    lines.append("  * uptime 回退 → VTS 自己重启过，连续性要重新计")
    lines.append("  * 通过与否都落盘原始 CSV，别人可以自己重算，不必相信这一行结论")
    return lines


def needs_utf8_console(console_cp, stdout_encoding):
    """控制台代码页是 UTF-8，而 Python 仍按本地代码页（如 gbk）输出 → 中文必然乱码。

    本机实测（2026-09-16）：`GetConsoleOutputCP()` = **65001**，而 `sys.stdout.encoding` = **gbk**，
    于是脚本打印的中文在控制台里变成方块 —— 证据文件本身是 UTF-8 没受影响，但报告没法读。
    """
    if console_cp != 65001:
        return False
    return (stdout_encoding or "").lower().replace("_", "-") not in ("utf-8", "utf8")


def ensure_utf8_console(stream=None, console_cp=None):
    """需要时把 stdout 切到 UTF-8；任何异常都不影响主流程（控制台只是给人看的）。"""
    stream = stream if stream is not None else sys.stdout
    if console_cp is None:
        try:
            import ctypes

            console_cp = ctypes.windll.kernel32.GetConsoleOutputCP()
        except Exception:                                  # noqa: BLE001 - 非 Windows / 无控制台
            return False
    if not needs_utf8_console(console_cp, getattr(stream, "encoding", None)):
        return False
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
        return True
    except Exception:                                      # noqa: BLE001 - 不是 TextIOWrapper 就算了
        return False


def _connect(args, log):
    """连 VTS。插件身份与桥接相同 → 复用已缓存 token，不会再弹授权窗。"""
    from vtube_studio_bridge.vtube_studio_bridge import VTubeStudioClient

    client = VTubeStudioClient(host=args.host, port=args.port)
    client.connect()
    log(f"[vts-stability] connected to VTS at ws://{args.host}:{args.port}")
    client.authenticate()
    log("[vts-stability] authenticated (reusing the bridge's cached token)")
    return client


def main(argv=None):
    ensure_utf8_console()
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = out_dir / f"vts_stability_{stamp}"

    lines = []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    log(f"[vts-stability] duration={args.duration:.0f}s interval={args.interval}s "
        f"window={args.window:.0f}s required={args.required_stable:.0f}s "
        f"parameters={','.join(args.parameters)}")
    log("[vts-stability] 现在（或稍后）在另一个终端启动桥接："
        "python apps/vtube_bridge/main.py --input 0 --sink vts --no-gui")

    try:
        client = _connect(args, log)
    except Exception as exc:                              # noqa: BLE001
        log(f"[vts-stability] cannot talk to VTube Studio: {exc}")
        log("[vts-stability] 检查：VTS 是否在运行、插件 API 是否打开（Config_StartAPI）、端口 8001 是否在监听")
        (base.with_suffix(".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 1

    csv_path = base.with_suffix(".csv")
    json_path = base.with_suffix(".json")
    txt_path = base.with_suffix(".txt")
    writer = IncrementalCsvWriter(csv_path, args.parameters)
    try:
        missing = check_parameter_names(client, args.parameters)
        if missing:
            log(f"[vts-stability] 警告：当前模型没有这些参数，会被跳过 -> {', '.join(missing)}")
        stats = read_statistics(client)
        log(f"[vts-stability] VTS {stats.get('vTubeStudioVersion')}  "
            f"connectedPlugins={stats.get('connectedPlugins')} (1 = 只有采样器自己，"
            f"桥接起来后应变成 {1 + args.other_plugins})  framerate={stats.get('framerate')}")
        log(f"[vts-stability] 原始数据边采边写：{csv_path}")
        samples = collect_samples(client, args.parameters, args.duration, args.interval,
                                  args.stats_interval, log=log, on_sample=writer)
    finally:
        writer.close()
        try:
            client.close()
        except Exception:                                  # noqa: BLE001
            pass

    max_gap = max(2.0, args.interval * 10.0)
    result = evaluate_stability(samples, window_seconds=args.window, epsilon=args.epsilon,
                                min_changed_params=args.min_changed_params,
                                required_stable_seconds=args.required_stable,
                                other_plugins=args.other_plugins, max_gap_seconds=max_gap)

    settings = {
        "duration": args.duration, "interval": args.interval, "stats_interval": args.stats_interval,
        "window": args.window, "epsilon": args.epsilon,
        "min_changed_params": args.min_changed_params,
        "required_stable": args.required_stable, "other_plugins": args.other_plugins,
        "parameters": args.parameters, "host": args.host, "port": args.port,
        "max_gap_seconds": max_gap,
    }
    for line in format_report(result, samples, args.parameters, settings):
        log(line)

    csv_path = base.with_suffix(".csv")
    json_path = base.with_suffix(".json")
    txt_path = base.with_suffix(".txt")
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "criterion": {
            "connectedPlugins_at_least": 1 + args.other_plugins,
            "note": "connectedPlugins counts live VTS API sessions including this sampler; "
                    "1 = sampler alone (bridge not attached yet, NOT a disconnect)",
        },
        "settings": settings,
        "samples": len(samples),
        "raw_csv": str(csv_path),
        "verdict": result,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("")
    log(f"[vts-stability] 原始数据 {csv_path}")
    log(f"[vts-stability] 结论     {json_path}")
    log(f"[vts-stability] 报告     {txt_path}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
