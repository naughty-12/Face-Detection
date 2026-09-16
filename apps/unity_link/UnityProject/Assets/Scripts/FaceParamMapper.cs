// FaceParamMapper.cs —— 把参数映射到场景物体上。
//
// P0 阶段（还没有皮套）：挂到一个方块/胶囊上，用旋转与缩放验证"参数真的通了"。
// P1 阶段（接入皮套）：保留下面的平滑与坐标处理，把最后一段替换为 VRM / Live2D 的写入。
using UnityEngine;

namespace FaceLink
{
    public class FaceParamMapper : MonoBehaviour
    {
        [Header("数据源")]
        [SerializeField] private FaceParamReceiver receiver;

        [Header("P0 目标物体（没有皮套时用占位物体）")]
        [SerializeField] private Transform head;   // 跟随头部三轴旋转
        [SerializeField] private Transform mouth;  // 跟随张嘴做纵向缩放
        [SerializeField] private Transform eyes;   // 跟随眼球视线平移

        [Header("轴向符号 —— 若动作左右/上下反了，只改这里的正负")]
        [Tooltip("FaceAngleX（左右转头）：图像坐标 +x 朝向模型的左侧")]
        [SerializeField] private float yawSign = -1f;
        [Tooltip("FaceAngleY（抬头/低头）：图像 y 向下，Unity y 向上")]
        [SerializeField] private float pitchSign = -1f;
        [SerializeField] private float rollSign = 1f;

        [Header("增益")]
        [SerializeField] private float angleGain = 1f;
        [SerializeField] private float mouthScaleGain = 0.6f;
        [SerializeField] private float gazeOffset = 0.15f;

        [Header("平滑（沿用 Python 侧的策略：表情快、姿态稳）")]
        [Range(0f, 1f)][SerializeField] private float poseAlpha = 0.35f;
        [Range(0f, 1f)][SerializeField] private float expressionAlpha = 0.45f;

        private Vector3 poseSmoothed;
        private float mouthSmoothed;
        private Vector2 gazeSmoothed;

        private void Reset()
        {
            receiver = FindObjectOfType<FaceParamReceiver>();
        }

        private void Update()
        {
            if (receiver == null) return;

            float yaw = receiver.Get("FaceAngleX") * angleGain;
            float pitch = receiver.Get("FaceAngleY") * angleGain;
            float roll = receiver.Get("FaceAngleZ") * angleGain;

            // 逐通道 EMA。注意：这是按帧插值，因此平滑强度会随帧率变化；
            // 若要严格一致，应改为按时间常数插值（1 - exp(-dt/tau)）。
            poseSmoothed = Vector3.Lerp(
                poseSmoothed,
                new Vector3(pitch * pitchSign, yaw * yawSign, roll * rollSign),
                poseAlpha);

            mouthSmoothed = Mathf.Lerp(mouthSmoothed, receiver.Get("MouthOpen"), expressionAlpha);

            float gazeX = (receiver.Get("EyeLeftX") + receiver.Get("EyeRightX")) * 0.5f;
            float gazeY = (receiver.Get("EyeLeftY") + receiver.Get("EyeRightY")) * 0.5f;
            gazeSmoothed = Vector2.Lerp(gazeSmoothed, new Vector2(gazeX, gazeY), expressionAlpha);

            if (head != null) head.localRotation = Quaternion.Euler(poseSmoothed);

            if (mouth != null)
            {
                Vector3 scale = mouth.localScale;
                scale.y = 1f + mouthSmoothed * mouthScaleGain;
                mouth.localScale = scale;
            }

            if (eyes != null)
            {
                eyes.localPosition = new Vector3(
                    gazeSmoothed.x * gazeOffset,
                    gazeSmoothed.y * gazeOffset,
                    0f);
            }

            // 人脸丢失时，管线上游会持续发送中性值（face_found=false），
            // 所以这里无需额外处理；若将来要做"保持上一姿态"，用 receiver.FaceFound 做闸门。
        }

        // =====================================================================
        // P1：接入真实皮套时，用下面任意一段替换上面的 head/mouth/eyes 部分。
        // 先删掉上面三处 localRotation / localScale / localPosition 的写法。
        // =====================================================================
        //
        // ── 3D：VRM（UniVRM，推荐先走这条）────────────────────────────────
        // 头部：直接转骨骼，或交给 VRM LookAt
        //   headBone.localRotation = Quaternion.Euler(pitch * pitchSign, yaw * yawSign, roll * rollSign);
        // 眨眼（注意 blendshape 是"闭眼"权重，所以要取反）
        //   float blinkL = 1f - receiver.Get("EyeOpenLeft");
        //   float blinkR = 1f - receiver.Get("EyeOpenRight");
        // 口型：VRM 1.0 用 viseme（aa/ih/ou/ee/oh），MediaPipe 的 jawOpen 映射到 "aa"
        //   SetExpression("aa", mouthSmoothed);
        // 视线：VRM 1.0 的 LookAt，或直接转眼球骨骼
        //
        // ── 2D：Live2D Cubism SDK for Unity ────────────────────────────────
        // 参数名因模型而异，务必先读模型参数清单再写映射（常见命名如下）
        //   CubismModel model = avatar.GetComponent<CubismModel>();
        //   model.Parameters.FindById("ParamAngleX").Value = yaw;
        //   model.Parameters.FindById("ParamAngleY").Value = pitch;
        //   model.Parameters.FindById("ParamAngleZ").Value = roll;
        //   model.Parameters.FindById("ParamEyeLOpen").Value = receiver.Get("EyeOpenLeft");
        //   model.Parameters.FindById("ParamEyeROpen").Value = receiver.Get("EyeOpenRight");
        //   model.Parameters.FindById("ParamMouthOpenY").Value = mouthSmoothed;
        //   model.Parameters.FindById("ParamEyeBallX").Value = gazeSmoothed.x;
        //   model.Parameters.FindById("ParamEyeBallY").Value = gazeSmoothed.y;
    }
}
