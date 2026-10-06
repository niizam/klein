# Builds klein with Ninja + MSVC + CUDA on Windows.
# Usage: scripts\build.ps1 [-BuildDir build] [-Jobs 4] [-Targets "klein klein-tokenize"] [-Config Release]
#                          [-CudaArch "86-real;89-real;120-real"]   (default 86-real: RTX 30 series)
# Runs at BelowNormal priority so the desktop stays responsive while nvcc compiles.
param(
    [string]$BuildDir = "build",
    [int]$Jobs = 4,
    [string]$Targets = "",
    [string]$Config = "Release",
    [string]$CudaArch = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath
if (-not $vs) { throw "Visual Studio 2022 with C++ tools not found" }
Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64" | Out-Null
try { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' } catch {}

$bd = Join-Path $root $BuildDir
if (-not (Test-Path (Join-Path $bd "build.ninja"))) {
    $arch = @()
    if ($CudaArch) { $arch = @("-DCMAKE_CUDA_ARCHITECTURES=$CudaArch") }
    cmake -S $root -B $bd -G Ninja "-DCMAKE_BUILD_TYPE=$Config" @arch
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
$t = @()
if ($Targets) { $t = @("--target") + ($Targets -split ' ') }
cmake --build $bd -j $Jobs @t
exit $LASTEXITCODE
