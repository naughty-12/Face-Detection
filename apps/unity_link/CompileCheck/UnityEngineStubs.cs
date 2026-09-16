// UnityEngineStubs.cs —— 只为"语法与类型用法"静态检查服务的 Unity API 桩。
//
// 目的：本机没有 Unity，无法真正编译 C# 脚本。用这套最小桩 + `dotnet build`，
// 可以在交付前抓住**语法错误、拼错的成员名、类型不匹配、C# 版本过新**等问题。
//
// ⚠️ 边界必须说清楚：桩是手写的，**它不等于 Unity 的真实 API**。
//    "能过这套桩"只能证明脚本自身语法与类型用法自洽，**不能**证明在 Unity 里能编译通过。
//    真正的验收仍然是 Unity 里打开工程看 Console。
using System;

namespace UnityEngine
{
    public class Object
    {
        public static T FindObjectOfType<T>() where T : Object => null;
        public static void Destroy(Object target) { }
    }

    public class Component : Object
    {
        public Transform transform { get; set; }
        public GameObject gameObject { get; set; }
        public T GetComponent<T>() => default;
    }

    public class Behaviour : Component
    {
        public bool enabled { get; set; }
    }

    public class MonoBehaviour : Behaviour { }

    public class GameObject : Object { }

    public class Transform : Component
    {
        public Vector3 localPosition { get; set; }
        public Vector3 localScale { get; set; }
        public Quaternion localRotation { get; set; }
    }

    public struct Vector2
    {
        public float x, y;
        public Vector2(float x, float y) { this.x = x; this.y = y; }
        public static Vector2 Lerp(Vector2 a, Vector2 b, float t) => a;
    }

    public struct Vector3
    {
        public float x, y, z;
        public Vector3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
        public static Vector3 Lerp(Vector3 a, Vector3 b, float t) => a;
    }

    public struct Quaternion
    {
        public float x, y, z, w;
        public static Quaternion Euler(float x, float y, float z) => default;
        public static Quaternion Euler(Vector3 euler) => default;
    }

    public struct Rect
    {
        public Rect(float x, float y, float width, float height) { }
    }

    public static class Mathf
    {
        public static float Lerp(float a, float b, float t) => a;
        public static float Clamp01(float value) => value;
        public static float Clamp(float value, float min, float max) => value;
    }

    public static class Time
    {
        public static float deltaTime => 0f;
        public static float unscaledDeltaTime => 0f;
        public static float realtimeSinceStartup => 0f;
    }

    public enum KeyCode { None = 0, F1 = 282, Space = 32, Escape = 27 }

    public static class Input
    {
        public static bool GetKeyDown(KeyCode key) => false;
    }

    public static class Debug
    {
        public static void Log(object message) { }
        public static void LogWarning(object message) { }
        public static void LogError(object message) { }
    }

    public static class JsonUtility
    {
        public static T FromJson<T>(string json) => default;
        public static string ToJson(object obj) => string.Empty;
    }

    public class GUIStyle
    {
        public bool richText { get; set; }
        public GUIStyle() { }
        public GUIStyle(GUIStyle other) { }
    }

    public class GUISkin
    {
        public GUIStyle box { get; } = new GUIStyle();
        public GUIStyle label { get; } = new GUIStyle();
    }

    public static class GUI
    {
        public static GUISkin skin { get; } = new GUISkin();
    }

    public static class GUILayout
    {
        public static void BeginArea(Rect screenRect) { }
        public static void BeginArea(Rect screenRect, GUIStyle style) { }
        public static void EndArea() { }
        public static void Label(string text) { }
        public static void Label(string text, GUIStyle style) { }
        public static void Space(float pixels) { }
    }

    [AttributeUsage(AttributeTargets.Field)]
    public class SerializeField : Attribute { }

    [AttributeUsage(AttributeTargets.Field)]
    public class HeaderAttribute : Attribute
    {
        public HeaderAttribute(string header) { }
    }

    [AttributeUsage(AttributeTargets.Field)]
    public class TooltipAttribute : Attribute
    {
        public TooltipAttribute(string tooltip) { }
    }

    [AttributeUsage(AttributeTargets.Field)]
    public class RangeAttribute : Attribute
    {
        public RangeAttribute(float min, float max) { }
    }
}
