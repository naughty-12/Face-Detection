// FaceParamReceiver.cs —— 接收 Python 捕捉管线通过 UDP 发来的表情/姿态参数。
//
// 设计要点（与 Python 侧同一套原则）：
//   1. 收包在自己的后台线程，渲染主循环只做一件事：取走"最新"的一包；
//   2. 用 seq 丢弃乱序/重发的旧包，避免模型被旧数据拽着来回跳；
//   3. 端口必须与桥接的 --unity-port 一致（默认 39540）。
//
// 用法：把本文件放到 Unity 工程的 Assets/Scripts/ 下，在场景里挂到一个空物体上。
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace FaceLink
{
    [Serializable]
    public class ParamEntry
    {
        public string id;
        public float value;
        public float weight;
    }

    [Serializable]
    public class FaceParamPacket
    {
        // 字段名必须与 Python 侧 UnityUdpSink 输出的 JSON 完全一致（JsonUtility 按名匹配）。
        public int seq;
        public double t;
        public bool face_found;
        public ParamEntry[] parameter_values;
    }

    public class FaceParamReceiver : MonoBehaviour
    {
        [Tooltip("必须与桥接的 --unity-port 一致")]
        [SerializeField] private int port = 39540;
        [SerializeField] private bool autoStart = true;

        private readonly ConcurrentQueue<FaceParamPacket> pending = new ConcurrentQueue<FaceParamPacket>();
        private readonly Dictionary<string, float> values = new Dictionary<string, float>();

        private Thread thread;
        private UdpClient client;
        private volatile bool running;
        private float rateWindowStart;
        private int packetsInWindow;

        /// <summary>最近一包中是否检测到人脸。</summary>
        public bool FaceFound { get; private set; }

        /// <summary>最近一包的时间戳（接收端本地时钟）。</summary>
        public float LastPacketTime { get; private set; }

        public int ReceivedPackets { get; private set; }
        public int StalePackets { get; private set; }
        public float ReceiveRate { get; private set; }
        public int LastSeq { get; private set; }

        /// <summary>参数名 → 值。缺失的参数不会出现在这里，请用 Get() 取。</summary>
        public IReadOnlyDictionary<string, float> Values => values;

        public float Get(string id) => values.TryGetValue(id, out float v) ? v : 0f;

        private void OnEnable()
        {
            if (autoStart) StartReceiver();
        }

        private void OnDisable()
        {
            StopReceiver();
        }

        public void StartReceiver()
        {
            if (running) return;

            try
            {
                client = new UdpClient(new IPEndPoint(IPAddress.Any, port));
            }
            catch (Exception e)
            {
                Debug.LogError($"[FaceLink] 无法绑定 UDP 端口 {port}：{e.Message}（端口被占用？）");
                return;
            }

            running = true;
            thread = new Thread(ReceiveLoop) { IsBackground = true, Name = "FaceParamReceiver" };
            thread.Start();
            Debug.Log($"[FaceLink] 正在监听 UDP {port} 上的表情参数。");
        }

        public void StopReceiver()
        {
            running = false;
            try { client?.Close(); } catch { /* 关闭时抛错是正常的 */ }
            client = null;
            if (thread != null && thread.IsAlive) thread.Join(300);
            thread = null;
        }

        private void ReceiveLoop()
        {
            IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);
            while (running)
            {
                try
                {
                    byte[] data = client.Receive(ref remote);
                    FaceParamPacket packet = JsonUtility.FromJson<FaceParamPacket>(Encoding.UTF8.GetString(data));
                    if (packet != null) pending.Enqueue(packet);
                }
                catch (SocketException)
                {
                    // StopReceiver() 关掉 socket 时会走到这里，静默退出。
                    if (!running) return;
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[FaceLink] 丢弃一个无法解析的包：{e.Message}");
                }
            }
        }

        private void Update()
        {
            int consumed = 0;
            while (pending.TryDequeue(out FaceParamPacket packet))
            {
                // seq 只增不减：seq 更小说明是乱序或重复的旧包，丢掉。
                if (packet.seq <= LastSeq) { StalePackets++; continue; }
                LastSeq = packet.seq;
                ReceivedPackets++;
                consumed++;
                FaceFound = packet.face_found;
                LastPacketTime = Time.realtimeSinceStartup;

                if (packet.parameter_values == null) continue;
                foreach (ParamEntry entry in packet.parameter_values)
                {
                    if (entry == null || string.IsNullOrEmpty(entry.id)) continue;
                    // weight 目前恒为 1.0，保留字段是为将来做多源混合。
                    values[entry.id] = entry.weight == 0f ? entry.value : entry.value * entry.weight;
                }
            }

            // 收包速率用指数平滑，避免 HUD 上的数字乱跳。
            packetsInWindow += consumed;
            float elapsed = Time.realtimeSinceStartup - rateWindowStart;
            if (elapsed >= 0.5f)
            {
                float instant = packetsInWindow / elapsed;
                ReceiveRate = ReceiveRate <= 0f ? instant : Mathf.Lerp(ReceiveRate, instant, 0.4f);
                packetsInWindow = 0;
                rateWindowStart = Time.realtimeSinceStartup;
            }
        }
    }
}
