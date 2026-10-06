# Packages a release build into dist\:
#   klein-<version>-win-cuda13-x64.zip   klein.exe, tools, README, docs, licenses
#   cuda13-runtime-win-x64.zip            NVIDIA cuBLAS DLLs klein.exe loads (redistributable under the CUDA EULA)
# Usage: scripts\package.ps1 -Version v0.1.0 [-BuildDir build-release]
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$BuildDir = "build-release"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$bd = Join-Path $root $BuildDir
$dist = Join-Path $root "dist"
$stage = Join-Path $dist "stage"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force (Join-Path $stage "klein"), (Join-Path $stage "cuda") | Out-Null

# klein
$k = Join-Path $stage "klein"
foreach ($exe in "klein.exe", "klein-tokenize.exe", "klein-cpubench.exe") {
    Copy-Item (Join-Path $bd $exe) $k
}
Copy-Item (Join-Path $root "README.md"), (Join-Path $root "LICENSE") $k
Copy-Item -Recurse (Join-Path $root "docs") (Join-Path $k "docs")
$lic = Join-Path $k "third-party-licenses"
New-Item -ItemType Directory -Force $lic | Out-Null
Copy-Item (Join-Path $root "third_party\ggml\LICENSE") (Join-Path $lic "ggml-llama.cpp-MIT.txt")
Copy-Item (Join-Path $root "third_party\httplib\LICENSE") (Join-Path $lic "cpp-httplib-MIT.txt")
Copy-Item (Join-Path $root "third_party\json\LICENSE.MIT") (Join-Path $lic "nlohmann-json-MIT.txt")
Set-Content (Join-Path $lic "stb_image.txt") "stb_image (https://github.com/nothings/stb): public domain (Unlicense) or MIT, see the end of third_party/stb/stb_image.h in the source."

# CUDA runtime libraries
$cudaBin = Join-Path $env:CUDA_PATH "bin\x64"
if (-not (Test-Path (Join-Path $cudaBin "cublas64_13.dll"))) { $cudaBin = Join-Path $env:CUDA_PATH "bin" }
$c = Join-Path $stage "cuda"
foreach ($dll in "cublas64_13.dll", "cublasLt64_13.dll") { Copy-Item (Join-Path $cudaBin $dll) $c }
Set-Content (Join-Path $c "NVIDIA-NOTICE.txt") @"
cublas64_13.dll and cublasLt64_13.dll are part of the NVIDIA CUDA Toolkit 13.4 and are redistributed under the
NVIDIA CUDA Toolkit End User License Agreement (https://docs.nvidia.com/cuda/eula/), which lists them as
redistributable files. Unpack them next to klein.exe.
"@

$z1 = Join-Path $dist "klein-$Version-win-cuda13-x64.zip"
$z2 = Join-Path $dist "cuda13-runtime-win-x64.zip"
Remove-Item $z1, $z2 -ErrorAction SilentlyContinue
Compress-Archive -Path (Join-Path $k "*") -DestinationPath $z1
Compress-Archive -Path (Join-Path $c "*") -DestinationPath $z2
Remove-Item -Recurse -Force $stage
Get-Item $z1, $z2 | Select-Object Name, @{ n = 'MiB'; e = { [math]::Round($_.Length / 1MB, 1) } }
