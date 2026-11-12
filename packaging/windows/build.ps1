param(
    [Parameter(Mandatory = $true)]
    [string] $PythonEmbedZip,
    [string] $Python = 'python',
    [string] $Wheelhouse,
    [string] $OutputZip = (Join-Path $PSScriptRoot '../../dist/YOLO-Studio-Windows-x64.zip')
)

$ErrorActionPreference = 'Stop'

function Invoke-Tool {
    param([string] $Program, [string[]] $Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE." }
}

function Confirm-Hash {
    param([string] $Path, [string] $Expected)
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $Expected) {
        throw "Unexpected SHA-256 for $Path."
    }
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$embedZip = (Resolve-Path -LiteralPath $PythonEmbedZip).Path
Confirm-Hash $embedZip '4ACBED6DD1C744B0376E3B1CF57CE906F9DC9E95E68824584C8099A63025A3C3'
$models = @{
    'yolo11n.pt' = '0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1'
    'sam2.1_t.pt' = '3C1E81CA9B037DD39D70A014DDB9A813D6C4C4E12555420DB7EAFF31689BD4E3'
}
foreach ($model in $models.Keys) {
    Confirm-Hash (Join-Path $repo $model) $models[$model]
}

if ((& gcc -dumpmachine) -ne 'x86_64-w64-mingw32') {
    throw 'The Windows x64 MinGW-w64 gcc compiler is required.'
}
Invoke-Tool $Python @('-c', 'import platform, sys; assert sys.version_info[:2] == (3, 12) and platform.machine() == ''AMD64'', ''Use Python 3.12 x64 to build''')
$output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputZip)
if ([IO.Path]::GetExtension($output) -ne '.zip') { throw 'OutputZip must end with .zip.' }
$outputDir = [IO.Path]::GetDirectoryName($output)
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$tempRoot = (Resolve-Path -LiteralPath ([IO.Path]::GetTempPath())).Path
$stage = Join-Path $tempRoot ('yolo-studio-build-' + [guid]::NewGuid().ToString('N'))
$release = Join-Path $stage 'YOLO Studio'
$runtime = Join-Path $release 'runtime'
$app = Join-Path $release 'app/yolo_studio'
$sitePackages = Join-Path $runtime 'Lib/site-packages'
$pendingZip = Join-Path $outputDir ('.yolo-studio-' + [guid]::NewGuid().ToString('N') + '.zip')

try {
    New-Item -ItemType Directory -Path $runtime, $app, $sitePackages -Force | Out-Null
    Expand-Archive -LiteralPath $embedZip -DestinationPath $runtime -Force
    Set-Content -LiteralPath (Join-Path $runtime 'python312._pth') -Encoding ASCII -Value @(
        'python312.zip', '.', 'Lib\site-packages', '..\app', 'import site'
    )

    $pipArgs = @('-m', 'pip', '--isolated', 'install', '--disable-pip-version-check', '--no-compile', '--only-binary=:all:',
        '--target', $sitePackages, '-r', (Join-Path $PSScriptRoot 'requirements.txt'))
    if ($Wheelhouse) {
        $pipArgs += @('--no-index', '--find-links', (Resolve-Path -LiteralPath $Wheelhouse).Path)
    }
    Invoke-Tool $Python $pipArgs

    # These wrappers contain paths to the build interpreter. The app imports the packages directly.
    $wrappers = Join-Path $sitePackages 'bin'
    if (Test-Path -LiteralPath $wrappers) {
        $resolvedWrappers = (Resolve-Path -LiteralPath $wrappers).Path
        if ($resolvedWrappers -ne (Join-Path $sitePackages 'bin')) { throw 'Unexpected pip wrapper directory.' }
        Remove-Item -LiteralPath $resolvedWrappers -Recurse -Force
    }

    # Qt supplies the app-local Microsoft C++ redistributable DLLs used by native dependencies.
    $crt = Get-ChildItem -LiteralPath (Join-Path $sitePackages 'PySide6') -File -Filter '*.dll' |
        Where-Object Name -Match '^(concrt|msvcp|vcamp|vccorlib|vcomp|vcruntime)140.*\.dll$'
    if ('msvcp140.dll' -notin $crt.Name) { throw 'Qt did not supply the expected C++ runtime DLLs.' }
    $crt | Copy-Item -Destination $runtime -Force

    $pkgSource = (Resolve-Path -LiteralPath (Join-Path $repo 'yolo_studio')).Path
    Get-ChildItem -LiteralPath $pkgSource -Recurse -File -Filter '*.py' | ForEach-Object {
        $rel = $_.FullName.Substring($pkgSource.Length).TrimStart('\', '/')
        $dest = Join-Path $app $rel
        New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($dest)) -Force | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $dest -Force
    }
    Copy-Item -LiteralPath (Join-Path $repo 'yolo_studio/assets') -Destination $app -Recurse
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'launch.py') -Destination $runtime
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'start-here.txt') -Destination (Join-Path $release 'START HERE.txt')
    foreach ($model in $models.Keys) {
        Copy-Item -LiteralPath (Join-Path $repo $model) -Destination $release
    }

    $icon = (Join-Path $app 'assets/app.ico').Replace('\', '/')
    $resource = Join-Path $stage 'launcher.rc'
    $resourceObject = Join-Path $stage 'launcher.o'
    Set-Content -LiteralPath $resource -Encoding ASCII -Value ('1 ICON "' + $icon + '"')
    Invoke-Tool 'windres' @($resource, $resourceObject)
    Invoke-Tool 'gcc' @('-O2', '-municode', '-mwindows', '-static-libgcc', '-finput-charset=UTF-8',
        (Join-Path $PSScriptRoot 'launcher.c'), $resourceObject, '-o', (Join-Path $release 'YOLO Studio.exe'))

    Invoke-Tool (Join-Path $runtime 'python.exe') @('-I', '-B', '-c', 'import sys; from importlib.metadata import version; print(sys.version); print(version(''torch''), version(''PySide6-Essentials''))')
    Invoke-Tool 'tar.exe' @('-a', '-cf', $pendingZip, '-C', $stage, 'YOLO Studio')
    Move-Item -LiteralPath $pendingZip -Destination $output -Force
    Get-Item -LiteralPath $output
} finally {
    if (Test-Path -LiteralPath $pendingZip) { Remove-Item -LiteralPath $pendingZip -Force }
    if (Test-Path -LiteralPath $stage) {
        $resolvedStage = (Resolve-Path -LiteralPath $stage).Path
        if ($resolvedStage -eq $stage -and [IO.Path]::GetDirectoryName($resolvedStage) -eq $tempRoot) {
            Remove-Item -LiteralPath $resolvedStage -Recurse -Force
        }
    }
}
