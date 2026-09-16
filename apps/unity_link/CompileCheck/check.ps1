# 用 Visual Studio 自带的 Roslyn 编译器 + 手写 Unity 桩，对 Unity 脚本做静态检查。
#
# 用法（在项目根目录或本目录均可）：
#     pwsh apps/unity_link/CompileCheck/check.ps1
#
# ⚠️ 它不是 Unity 编译：桩是手写的，与真实 Unity API 有差异。
#    "通过"只能证明脚本自身语法 / 成员名 / 类型用法自洽、且未超过 C# 9。
#    真正的验收仍然是在 Unity 里打开工程看 Console。

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$scripts = Join-Path $here "..\UnityProject\Assets\Scripts"
$outDir = Join-Path $here "bin"
$outDll = Join-Path $outDir "FaceLinkCheck.dll"

# 1) 找编译器：优先 dotnet SDK，其次 VS 自带的 Roslyn
$csc = $null
$dotnetSdks = & dotnet --list-sdks 2>$null
if ($LASTEXITCODE -eq 0 -and $dotnetSdks) {
    Write-Host "[信息] 检测到 .NET SDK，改用 dotnet build（更贴近真实编译环境）"
    & dotnet build (Join-Path $here "CompileCheck.csproj") --nologo -v quiet
    exit $LASTEXITCODE
}

$vsRoots = @(
    "C:\Program Files\Microsoft Visual Studio",
    "C:\Program Files (x86)\Microsoft Visual Studio"
)
foreach ($root in $vsRoots) {
    if (-not (Test-Path $root)) { continue }
    $found = Get-ChildItem $root -Recurse -Filter csc.exe -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match "Roslyn" } | Select-Object -First 1
    if ($found) { $csc = $found.FullName; break }
}

if (-not $csc) {
    Write-Host "[跳过] 本机既没有 .NET SDK，也没有 Visual Studio 的 Roslyn 编译器。"
    Write-Host "       安装任一个后即可运行本检查；在此之前，C# 脚本仍属未编译验证。"
    exit 2
}
Write-Host "[信息] 编译器: $csc"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# 2) 编译：脚本 + 桩，语言版本对齐 Unity 2022.3 的 C# 9
#    /nowarn:0649 —— [SerializeField] 字段本来就由 Inspector 赋值，
#    在桩编译里必然报"从未赋值"，属噪声，会淹没真正的问题。
& $csc /nologo /target:library /langversion:9 /nowarn:0649 /out:$outDll `
    (Join-Path $here "UnityEngineStubs.cs") `
    (Join-Path $scripts "FaceParamReceiver.cs") `
    (Join-Path $scripts "FaceParamMapper.cs") `
    (Join-Path $scripts "FaceParamHud.cs")

$code = $LASTEXITCODE
if ($code -eq 0) {
    Write-Host "[PASS] C# 静态检查通过（语法 / 成员名 / 类型用法 / C# 9 兼容）。"
    Write-Host "       注意：这只是桩编译，不等于 Unity 编译 —— 仍需在 Unity 里过一遍 Console。"
} else {
    Write-Host "[FAIL] 编译失败，退出码 $code。上面带 error 的行就是需要修的地方。"
}
exit $code
