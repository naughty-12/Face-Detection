// FaceParamHud.cs —— 运行时诊断面板：收包速率、丢包、帧率、关键参数。
//
// 为什么要有它：P0 阶段最常见的三个问题是「完全没动」「方向反了」「过一会儿卡住」，
// 而这三者的排查入口都是"包到底有没有到、到得多快"。没有面板就只能靠猜。
using UnityEngine;

namespace FaceLink
{
    public class FaceParamHud : MonoBehaviour
    {
        [SerializeField] private FaceParamReceiver receiver;
        [SerializeField] private bool visible = true;

        private static readonly string[] Preview =
        {
            "FaceAngleX", "FaceAngleY", "FaceAngleZ",
            "MouthOpen", "EyeOpenLeft", "EyeOpenRight",
        };

        private float fpsSmoothed;

        private void Reset()
        {
            receiver = FindObjectOfType<FaceParamReceiver>();
        }

        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.F1)) visible = !visible;
            float dt = Time.unscaledDeltaTime;
            if (dt > 0f)
            {
                float instant = 1f / dt;
                fpsSmoothed = fpsSmoothed <= 0f ? instant : Mathf.Lerp(fpsSmoothed, instant, 0.1f);
            }
        }

        private void OnGUI()
        {
            if (!visible || receiver == null) return;

            float silence = receiver.ReceivedPackets == 0
                ? -1f
                : Time.realtimeSinceStartup - receiver.LastPacketTime;

            bool stalled = silence < 0f || silence > 1.0f;

            GUILayout.BeginArea(new Rect(10, 10, 340, 250), GUI.skin.box);
            GUILayout.Label("<b>FaceLink 诊断</b>  (F1 隐藏)", RichLabel());
            GUILayout.Label($"渲染帧率: {fpsSmoothed,6:0.0} FPS");
            GUILayout.Label($"收包速率: {receiver.ReceiveRate,6:0.0} pkt/s   (期望 ≈ 30)");
            GUILayout.Label($"累计收包: {receiver.ReceivedPackets}   乱序丢弃: {receiver.StalePackets}");
            GUILayout.Label($"最新 seq: {receiver.LastSeq}");

            if (silence < 0f)
            {
                GUILayout.Label("<color=#ff5555>尚未收到任何包</color>", RichLabel());
                GUILayout.Label("检查：桥接是否用 --sink unity 启动？端口是否一致？");
            }
            else if (stalled)
            {
                GUILayout.Label($"<color=#ffaa00>已静默 {silence:0.0} 秒</color>", RichLabel());
                GUILayout.Label("桥接进程还在吗？摄像头被别的程序占用了？");
            }
            else
            {
                GUILayout.Label($"<color=#55ff55>数据流正常</color>（{silence * 1000f:0} ms 前收到）", RichLabel());
            }

            GUILayout.Label($"face_found: {(receiver.FaceFound ? "true" : "false")}");
            GUILayout.Space(4);
            foreach (string id in Preview)
            {
                float value = receiver.Get(id);
                GUILayout.Label($"{id,-14} {value,7:+0.000;-0.000; 0.000}");
            }
            GUILayout.EndArea();
        }

        private static GUIStyle RichLabel()
        {
            return new GUIStyle(GUI.skin.label) { richText = true };
        }
    }
}
