<#
.SYNOPSIS
    构建 WavetonePro 独立 exe（PyInstaller onedir，自带 CUDA 运行库）。

.DESCRIPTION
    一键完成四件事：
      1. 建一个干净的构建 venv（.venv-build），只装 requirements-build.txt 里的库
      2. 从系统 CUDA Toolkit 里挑出真正用到的 4 个 DLL 复制到 .cuda_runtime\bin
      3. 用 wavetonepro.spec 打包到 dist\WavetonePro
      4. 打印体积明细

    产物不依赖目标机器的 Python，也不依赖目标机器装没装 CUDA Toolkit
    （只需要有 NVIDIA 显卡驱动）。

.PARAMETER CpuOnly
    不打包 CUDA 运行库，产出 dist\WavetonePro-CPU（小约 384MB，无 N 卡机器用）。

.PARAMETER IncludeScipy
    把 scipy 也打进去（+约 130MB）。默认不打：soundfile + 内置 wave 已覆盖音频解码。

.PARAMETER NoConsole
    不显示控制台窗口（默认显示，方便看 [gpu]/[midi] 日志）。

.PARAMETER Clean
    构建前清空 build\ 和 dist\，并让 PyInstaller 丢弃缓存。

.PARAMETER RecreateVenv
    删掉 .venv-build 重新创建（依赖装乱了/换 Python 版本时用）。

.PARAMETER CudaBin
    手动指定 CUDA Toolkit 的 bin 目录（默认自动探测 v11.x）。

.PARAMETER PythonExe
    指定用来创建 venv 的 Python（默认自动探测 3.12）。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File build_exe.ps1
    powershell -ExecutionPolicy Bypass -File build_exe.ps1 -CpuOnly
#>
[CmdletBinding()]
param(
    [switch]$CpuOnly,
    [switch]$IncludeScipy,
    [switch]$NoConsole,
    [switch]$Clean,
    [switch]$RecreateVenv,
    [string]$CudaBin = '',
    [string]$PythonExe = ''
)

$ErrorActionPreference = 'Stop'

# 本脚本住在 <项目>\build\ 里；$BuildDir 放所有打包相关的东西（工具 + 产物），
# $Root 是项目根（wavetonepro.py 所在目录）。
$BuildDir = $PSScriptRoot
$Root = Split-Path $BuildDir -Parent

$VenvDir = Join-Path $BuildDir 'venv'
$VenvPy = Join-Path $VenvDir 'Scripts\python.exe'
$StageDir = Join-Path $BuildDir 'cuda'
$StageBin = Join-Path $StageDir 'bin'
# GPU 版和 CPU 版各用一个 workpath：PyInstaller 的 Analysis 缓存按 spec 文件名存放，
# 两个变体共用会复用上一次的模块图，可能出现 CPU 版里混进 cupy、或 GPU 版漏 cupy。
$WorkDir = Join-Path $BuildDir ('work\' + $(if ($CpuOnly) { 'cpu' } else { 'gpu' }))
$DistDir = Join-Path $BuildDir 'dist'
$TmpDir = Join-Path $BuildDir 'tmp'
$LogDir = Join-Path $BuildDir 'logs'
$Requirements = Join-Path $BuildDir 'requirements-build.txt'
$SpecFile = Join-Path $BuildDir 'wavetonepro.spec'

$CudaDlls = @(
    'cudart64_110.dll',
    'cufft64_10.dll',
    'nvrtc64_112_0.dll',
    'nvrtc-builtins64_116.dll'
)

function Write-Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Write-Ok($text) { Write-Host "  [OK] $text" -ForegroundColor Green }
function Write-Warn2($text) { Write-Host "  [!] $text" -ForegroundColor Yellow }

# --------------------------------------------------------------------------
# 0. 运行期把临时目录 / 缓存都放在 build\ 内，避免受系统目录权限影响
# --------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $TmpDir, $LogDir | Out-Null
$env:PIP_CACHE_DIR = Join-Path $BuildDir 'pip-cache'
$env:PYINSTALLER_CONFIG_DIR = Join-Path $BuildDir 'pyinstaller-cache'
$SavedTemp = $env:TEMP
$SavedTmp = $env:TMP
$env:TEMP = $TmpDir
$env:TMP = $TmpDir
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR, $env:PYINSTALLER_CONFIG_DIR | Out-Null

try {

Write-Host 'WavetonePro 打包脚本' -ForegroundColor White
Write-Host "项目根目录: $Root"
Write-Host "打包工作区: $BuildDir"

# --------------------------------------------------------------------------
# 1. 干净的构建 venv
# --------------------------------------------------------------------------
Write-Step '1/5 准备构建虚拟环境'

function Resolve-BasePython {
    param([string]$Explicit)
    $cands = @()
    if ($Explicit) { $cands += [pscustomobject]@{ Exe = $Explicit; Pre = @() } }
    if ($env:WAVETONEPRO_PYTHON) { $cands += [pscustomobject]@{ Exe = $env:WAVETONEPRO_PYTHON; Pre = @() } }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) { $cands += [pscustomobject]@{ Exe = $py.Source; Pre = @('-3.12') } }
    $py2 = Get-Command python -ErrorAction SilentlyContinue
    if ($py2) { $cands += [pscustomobject]@{ Exe = $py2.Source; Pre = @() } }
    $known = 'C:\Users\Arcueid Brunestud\AppData\Local\Programs\Python\Python312\python.exe'
    if (Test-Path $known) { $cands += [pscustomobject]@{ Exe = $known; Pre = @() } }

    # 注意：探测代码里绝对不能出现双引号 —— Windows PowerShell 5.1 调用原生
    # exe 时会吞掉 -c 参数里的双引号，脚本会变成语法错误。
    $probeCode = 'import sys; print(sys.version_info.major, sys.version_info.minor)'

    foreach ($c in $cands) {
        $argList = @() + $c.Pre + @('-c', $probeCode)
        try {
            $out = (& $c.Exe @argList 2>$null | Out-String).Trim()
            if ($LASTEXITCODE -eq 0 -and $out -eq '3 12') { return $c }
        } catch { }
    }
    throw '找不到 Python 3.12。用 -PythonExe 指定，例如 -PythonExe C:\Python312\python.exe'
}

if ($RecreateVenv -and (Test-Path $VenvDir)) {
    Write-Warn2 "删除旧的 venv: $VenvDir"
    Remove-Item -Recurse -Force $VenvDir
}

$Base = Resolve-BasePython -Explicit $PythonExe
Write-Ok "基础解释器: $($Base.Exe) $($Base.Pre -join ' ')"

if (-not (Test-Path $VenvPy)) {
    $argList = @() + $Base.Pre + @('-m', 'venv', $VenvDir)
    & $Base.Exe @argList
    if ($LASTEXITCODE -ne 0) { throw '创建 venv 失败' }
    Write-Ok "已创建 venv: $VenvDir"
} else {
    Write-Ok "复用 venv: $VenvDir"
}

# 依赖是否齐全
$probe = 'import PyInstaller, cupy, PyQt5, soundfile, mido, rtmidi, numpy'
& $VenvPy -c $probe 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host '  正在安装依赖（首次约几分钟）...'
    & $VenvPy -m pip install --no-input --progress-bar off -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw '依赖安装失败' }
    & $VenvPy -c $probe
    if ($LASTEXITCODE -ne 0) { throw '依赖安装后仍无法导入，请检查要求清单' }
    Write-Ok '依赖已安装'
} else {
    Write-Ok '依赖已齐全'
}

# --------------------------------------------------------------------------
# 2. 收集 CUDA 运行库
# --------------------------------------------------------------------------
Write-Step '2/5 准备 CUDA 运行库'

function Find-CudaDll {
    param([string]$Name, [string]$ExplicitBin)

    $search = @()
    if ($ExplicitBin) { $search += $ExplicitBin }
    if ($env:CUDA_PATH) { $search += (Join-Path $env:CUDA_PATH 'bin') }

    # 系统安装的 CUDA Toolkit（优先 11.x 高版本）
    $tk = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path $tk) {
        Get-ChildItem $tk -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^v11\.' } |
            Sort-Object { [double]($_.Name -replace '^v', '') } -Descending |
            ForEach-Object { $search += (Join-Path $_.FullName 'bin') }
    }

    # pip 安装的 nvidia-*-cu11 wheel 布局
    $nv = Join-Path $VenvDir 'Lib\site-packages\nvidia'
    if (Test-Path $nv) {
        $search += (Get-ChildItem $nv -Recurse -Directory -Filter 'bin' -ErrorAction SilentlyContinue |
            ForEach-Object { $_.FullName })
    }

    foreach ($dir in $search) {
        if (-not $dir) { continue }
        $p = Join-Path $dir $Name
        if (Test-Path $p) { return $p }
    }
    return $null
}

if ($CpuOnly) {
    Write-Ok '已选择纯 CPU 版，跳过 CUDA'
} else {
    New-Item -ItemType Directory -Force -Path $StageBin | Out-Null
    $missing = @()
    foreach ($dll in $CudaDlls) {
        $src = Find-CudaDll -Name $dll -ExplicitBin $CudaBin
        if (-not $src) { $missing += $dll; continue }
        Copy-Item $src (Join-Path $StageBin $dll) -Force
        $mb = [math]::Round((Get-Item (Join-Path $StageBin $dll)).Length / 1MB, 1)
        Write-Ok ("{0,-26} {1,7} MB   <- {2}" -f $dll, $mb, $src)
    }
    if ($missing.Count) {
        throw @"
缺少这些 CUDA 运行库: $($missing -join ', ')

它们来自 CUDA 11.x Toolkit（cupy-cuda11x 需要的版本）。
请安装 CUDA Toolkit 11.8（勾选 Runtime + cuFFT + NVRTC），或用 -CudaBin 指定
已有的 ...\CUDA\v11.x\bin 目录。

如果确实不需要 GPU，可以直接构建 CPU 版：
    powershell -ExecutionPolicy Bypass -File build_exe.ps1 -CpuOnly
"@
    }
}

# --------------------------------------------------------------------------
# 3. 清理
# --------------------------------------------------------------------------
if ($Clean) {
    Write-Step '3/5 清理旧产物'
    # 只删当前变体的产物和 workpath，不动另一个变体
    $AppName = if ($CpuOnly) { 'WavetonePro-CPU' } else { 'WavetonePro' }
    foreach ($d in @($WorkDir, (Join-Path $DistDir $AppName))) {
        if (Test-Path $d) {
            $full = (Resolve-Path $d).Path
            if ($full.StartsWith($Root, [StringComparison]::OrdinalIgnoreCase)) {
                Remove-Item -Recurse -Force $full
                Write-Ok "已删除 $full"
            } else {
                Write-Warn2 "跳过（不在项目目录内）: $full"
            }
        }
    }
} else {
    Write-Host "`n=== 3/5 跳过清理（需要时加 -Clean）===" -ForegroundColor Cyan
}

# --------------------------------------------------------------------------
# 4. 打包
# --------------------------------------------------------------------------
Write-Step '4/5 运行 PyInstaller'

$env:WAVETONEPRO_BUNDLE_CUDA = if ($CpuOnly) { '0' } else { '1' }
$env:WAVETONEPRO_INCLUDE_SCIPY = if ($IncludeScipy) { '1' } else { '0' }
$env:WAVETONEPRO_CONSOLE = if ($NoConsole) { '0' } else { '1' }

$piArgs = @(
    '-m', 'PyInstaller',
    '--noconfirm',
    '--distpath', $DistDir,
    '--workpath', $WorkDir,
    $SpecFile
)
if ($Clean) { $piArgs += '--clean' }

# PyInstaller 把 INFO 日志写到 stderr，Windows PowerShell 5.1 会把原生程序的
# stderr 当成终止性错误（结果构建成功了也会报 exit 1）。所以整份日志落盘，
# 只回显关键行。
$logFile = Join-Path $LogDir 'pyinstaller.log'
$savedEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $VenvPy @piArgs *> $logFile
$piExit = $LASTEXITCODE
$ErrorActionPreference = $savedEap

Get-Content $logFile -ErrorAction SilentlyContinue |
    Where-Object { $_ -cmatch '^\[spec\]|^\d+ ERROR|^ERROR|^\d+ WARNING' } |
    ForEach-Object { Write-Host "  $_" }
Write-Host "  （完整构建日志: $logFile）"

if ($piExit -ne 0) { throw "PyInstaller 打包失败，请查看日志: $logFile" }

# --------------------------------------------------------------------------
# 5. 报告
# --------------------------------------------------------------------------
Write-Step '5/5 完成'

$AppName = if ($CpuOnly) { 'WavetonePro-CPU' } else { 'WavetonePro' }
$OutDir = Join-Path $DistDir $AppName
$ExePath = Join-Path $OutDir "$AppName.exe"

if (-not (Test-Path $ExePath)) { throw "没找到产物: $ExePath" }

# 产物里放一份给"接收者"看的说明：未签名 exe 第一次运行会被 SmartScreen 拦，
# 接收者照着点就行，省得每次都来问你。
$readme = @"
$AppName 首次运行说明
========================================

如果双击 $AppName.exe 时弹出
「Microsoft Defender SmartScreen 阻止了无法识别的应用启动」
这是 Windows 对"没有数字签名的自制程序"的常规提示，不是程序有问题。

三种解决办法，任选一种：

  1) 弹窗里点【更多信息】，再点右下角出现的【仍要运行】。

  2) 右键 $AppName.exe ->【属性】-> 最下面勾上【解除锁定】-> 确定，再双击。

  3) 打开 PowerShell 执行（路径换成你解压出来的目录）：
       Get-ChildItem "D:\$AppName" -Recurse | Unblock-File


其他说明
----------------------------------------
* 必须整个文件夹一起用，不能只把 $AppName.exe 单独拷走
  —— 真正的依赖在 _internal 文件夹里。
* 想用 GPU 加速，需要 NVIDIA 显卡并已安装显卡驱动。
* 想确认这台机器的环境是否齐全，在命令行里跑：
      $AppName.exe --selftest
  会检查 GPU / 音频解码 / 界面 / MIDI 四项，并列出实际加载的 CUDA 库。
"@
Set-Content -Path (Join-Path $OutDir '首次运行说明.txt') -Value $readme -Encoding UTF8

$files = Get-ChildItem $OutDir -Recurse -File
$totalMb = [math]::Round(($files | Measure-Object Length -Sum).Sum / 1MB, 0)

Write-Host ("  产物: {0}" -f $ExePath) -ForegroundColor Green
Write-Host ("  总大小: {0} MB   （{1} 个文件）" -f $totalMb, $files.Count)

Write-Host "`n  体积最大的 8 个文件：" -ForegroundColor White
$files | Sort-Object Length -Descending | Select-Object -First 8 | ForEach-Object {
    Write-Host ("    {0,8:N1} MB  {1}" -f ($_.Length / 1MB), $_.FullName.Substring($OutDir.Length + 1))
}

# 直接跑一次冻结后程序的自检，确认 GPU 通路真的通
# （不要等用户双击了才发现 import cupy 缺模块）
if (-not $CpuOnly) {
    Write-Host "`n  运行 GPU 自检 ..." -ForegroundColor White
    $selfLog = Join-Path $LogDir 'selftest.log'
    # 自检时把 CuPy 的 JIT 缓存指到项目内：默认位置（%LOCALAPPDATA%）在受限
    # 环境里可能不可写，而不可写时 CuPy 编译内核会**空转卡死**而不是报错。
    $selfCache = Join-Path $TmpDir 'selftest_cache'
    New-Item -ItemType Directory -Force -Path $selfCache | Out-Null
    $savedCache = $env:CUPY_CACHE_DIR
    $savedGpuCache = $env:WAVETONEPRO_GPU_CACHE
    $env:CUPY_CACHE_DIR = $selfCache
    $env:WAVETONEPRO_GPU_CACHE = Join-Path $selfCache 'gpu.json'
    $savedEap2 = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    # 顺手拿项目里的音频文件测一下解码（有的话）
    $selfArgs = @('--selftest')
    $sample = Join-Path $Root '1.20.flac'
    if (Test-Path $sample) { $selfArgs += $sample }
    & $ExePath @selfArgs *> $selfLog
    $selfExit = $LASTEXITCODE
    $ErrorActionPreference = $savedEap2
    $env:CUPY_CACHE_DIR = $savedCache
    $env:WAVETONEPRO_GPU_CACHE = $savedGpuCache
    Get-Content $selfLog -ErrorAction SilentlyContinue | Select-Object -Last 16 |
        ForEach-Object { Write-Host "    $_" }
    if ($selfExit -eq 0) {
        Write-Ok '冻结后的程序可以用 GPU'
    } else {
        Write-Warn2 '自检没通过。目标机器没有 N 卡可忽略；否则照上面的报错查。'
    }
}

Write-Host "`n  自检（确认 GPU 是否真的启用）：" -ForegroundColor White
Write-Host "    & '$ExePath' --selftest"
Write-Host "`n  分发：整个 $AppName 文件夹一起拷给对方，双击 $AppName.exe 即可。" -ForegroundColor White

}
finally {
    $env:TEMP = $SavedTemp
    $env:TMP = $SavedTmp
}

exit 0
