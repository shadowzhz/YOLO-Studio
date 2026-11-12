#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_TAR=""
WHEELHOUSE=""
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
OUTPUT_TAR="$REPO_ROOT/dist/YOLO-Studio-Linux-x86_64.tar.gz"
SKIP_SYSLIBS=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --python-tar)
            PYTHON_TAR="${2:?Missing path for --python-tar}"
            shift 2
            ;;
        --wheelhouse)
            WHEELHOUSE="${2:?Missing path for --wheelhouse}"
            shift 2
            ;;
        --torch-index)
            TORCH_INDEX="${2:-}"
            shift 2
            ;;
        --output)
            OUTPUT_TAR="${2:?Missing path for --output}"
            shift 2
            ;;
        --skip-syslibs)
            SKIP_SYSLIBS=1
            shift
            ;;
        -h|--help)
            cat <<'EOF'
Usage: packaging/linux/build.sh [options]
  --python-tar <path>   Local python-build-standalone 3.12 x86_64 tar.gz
  --wheelhouse <dir>    Offline directory of wheels (--no-index --find-links)
  --torch-index <url>   PyTorch wheel index (default: https://download.pytorch.org/whl/cu128)
  --output <path>       Output .tar.gz path (default: dist/YOLO-Studio-Linux-x86_64.tar.gz)
  --skip-syslibs        Do not copy host Qt6 XCB helper libraries into runtime/syslibs
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

ARCH="$(uname -m)"
if [ "$ARCH" != "x86_64" ]; then
    echo "This Linux portable build script targets x86_64 (detected: $ARCH)." >&2
    exit 1
fi

case "$OUTPUT_TAR" in
    *.tar.gz) ;;
    *)
        echo "--output must end with .tar.gz" >&2
        exit 1
        ;;
esac

verify_sha256() {
    local file="$1"
    local expected="$2"
    local actual=""
    if command -v sha256sum >/dev/null 2>&1; then
        actual="$(sha256sum "$file" | awk '{print toupper($1)}')"
    elif command -v shasum >/dev/null 2>&1; then
        actual="$(shasum -a 256 "$file" | awk '{print toupper($1)}')"
    else
        echo "sha256sum or shasum is required for integrity verification." >&2
        exit 1
    fi
    if [ "$actual" != "$(printf '%s' "$expected" | tr '[:lower:]' '[:upper:]')" ]; then
        echo "Unexpected SHA-256 for $file: $actual != $expected" >&2
        exit 1
    fi
}

download_verified() {
    local target="$1"
    local url="$2"
    local expected="$3"
    if [ ! -f "$target" ]; then
        mkdir -p "$(dirname "$target")"
        echo "Downloading $url -> $target"
        if command -v curl >/dev/null 2>&1; then
            curl -fL --retry 3 -o "$target" "$url"
        elif command -v wget >/dev/null 2>&1; then
            wget -O "$target" "$url"
        else
            echo "curl or wget is required to download missing assets." >&2
            exit 1
        fi
    fi
    verify_sha256 "$target" "$expected"
}

resolve_model_file() {
    local name="$1"
    if [ -f "$REPO_ROOT/models/$name" ]; then
        printf '%s\n' "$REPO_ROOT/models/$name"
    elif [ -f "$REPO_ROOT/$name" ]; then
        printf '%s\n' "$REPO_ROOT/$name"
    else
        printf '%s\n' "$REPO_ROOT/models/$name"
    fi
}

PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261110/cpython-3.12.16%2B20261110-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
PBS_SHA="6A9E62DAD033790AEF722DBD370E0CE76CE17B73627BA3AE679F537049350B39"

YOLO_MODEL_PATH="$(resolve_model_file yolo11n.pt)"
SAM_MODEL_PATH="$(resolve_model_file sam2.1_t.pt)"
download_verified "$YOLO_MODEL_PATH" \
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt" \
    "0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1"
download_verified "$SAM_MODEL_PATH" \
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_t.pt" \
    "3C1E81CA9B037DD39D70A014DDB9A813D6C4C4E12555420DB7EAFF31689BD4E3"

mkdir -p "$(dirname "$OUTPUT_TAR")"
OUTPUT_DIR="$(cd "$(dirname "$OUTPUT_TAR")" && pwd)"
OUTPUT_ABS="$OUTPUT_DIR/$(basename "$OUTPUT_TAR")"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/yolo-studio-linux-XXXXXX")"
PENDING_TAR="$OUTPUT_DIR/.yolo-studio-$(basename "$OUTPUT_ABS").$$.tmp"

cleanup() {
    rm -f "$PENDING_TAR"
    if [ -n "${STAGE:-}" ] && [ -d "$STAGE" ]; then
        rm -rf "$STAGE"
    fi
}
trap cleanup EXIT

RELEASE="$STAGE/YOLO-Studio"
RUNTIME="$RELEASE/runtime"
APP="$RELEASE/app/yolo_studio"
mkdir -p "$RUNTIME" "$APP" "$RELEASE/models"

if [ -z "$PYTHON_TAR" ]; then
    PYTHON_TAR="$STAGE/python-standalone.tar.gz"
    download_verified "$PYTHON_TAR" "$PBS_URL" "$PBS_SHA"
else
    verify_sha256 "$PYTHON_TAR" "$PBS_SHA"
fi

echo "Extracting standalone Python 3.12 runtime..."
tar -xzf "$PYTHON_TAR" --strip-components=1 -C "$RUNTIME"

PIP_ARGS=(
    -m pip --isolated install
    --disable-pip-version-check
    --no-cache-dir
    --no-compile
    --only-binary=:all:
    -r "$REPO_ROOT/packaging/requirements.txt"
)
if [ -n "$WHEELHOUSE" ]; then
    PIP_ARGS+=(--no-index --find-links "$WHEELHOUSE")
elif [ -n "$TORCH_INDEX" ]; then
    PIP_ARGS+=(--extra-index-url "$TORCH_INDEX")
fi

"$RUNTIME/bin/python3" -I "${PIP_ARGS[@]}"

# Strip pip wrapper scripts in runtime/bin that embed build-time shebang paths.
for wrapper in "$RUNTIME/bin/"*; do
    [ -f "$wrapper" ] || continue
    case "$(basename "$wrapper")" in
        python*) continue ;;
    esac
    magic=""
    IFS= read -r -n 2 magic <"$wrapper" 2>/dev/null || true
    if [ "$magic" = "#!" ]; then
        rm -f "$wrapper"
    fi
done

if [ "$SKIP_SYSLIBS" -eq 0 ]; then
    mkdir -p "$RUNTIME/syslibs"
    XCB_LIBS=(
        libxcb-cursor.so.0
        libxcb-icccm.so.4
        libxcb-image.so.0
        libxcb-keysyms.so.1
        libxcb-randr.so.2
        libxcb-render-util.so.0
        libxcb-shape.so.0
        libxcb-xfixes.so.3
        libxcb-xinerama.so.0
        libxcb-xinput.so.0
        libxcb-util.so.1
        libxkbcommon-x11.so.0
        libxkbcommon.so.0
    )
    missing_syslibs=()
    ld_cache=""
    if command -v ldconfig >/dev/null 2>&1; then
        ld_cache="$(ldconfig -p 2>/dev/null || true)"
    fi
    for lib in "${XCB_LIBS[@]}"; do
        src=""
        if [ -n "$ld_cache" ]; then
            src="$(printf '%s\n' "$ld_cache" | awk -v target="$lib" '$1 == target {print $NF; exit}')"
        fi
        if [ -z "$src" ] || [ ! -e "$src" ]; then
            for dir in /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib64 /lib64 /usr/lib; do
                if [ -e "$dir/$lib" ]; then
                    src="$dir/$lib"
                    break
                fi
            done
        fi
        if [ -n "$src" ] && [ -e "$src" ]; then
            cp -L "$src" "$RUNTIME/syslibs/$lib"
        else
            missing_syslibs+=("$lib")
        fi
    done
    if [ "${#missing_syslibs[@]}" -gt 0 ]; then
        echo "Warning: the following Qt6 XCB libraries were not found on the build host and were not bundled:" >&2
        printf '  - %s\n' "${missing_syslibs[@]}" >&2
        echo "Build inside Ubuntu 22.04+ with libxcb-cursor0, libxcb-icccm4, libxcb-image0, libxcb-keysyms1, libxcb-randr0, libxcb-render-util0, libxcb-shape0, libxcb-xfixes0, libxcb-xinerama0, libxcb-xinput0, libxkbcommon-x11-0 installed for maximum compatibility." >&2
    fi
fi

# Bundle a CJK font when present on the build host so English-only Linux desktops render Chinese UI cleanly.
mkdir -p "$RUNTIME/fonts"
for font_candidate in \
    /usr/share/fonts/truetype/wqy/wqy-microhei.ttc \
    /usr/share/fonts/wqy-microhei/wqy-microhei.ttc \
    /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc \
    /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc \
    /usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc
do
    if [ -f "$font_candidate" ]; then
        cp -L "$font_candidate" "$RUNTIME/fonts/"
        break
    fi
done
if [ -z "$(ls -A "$RUNTIME/fonts" 2>/dev/null)" ]; then
    rmdir "$RUNTIME/fonts"
    echo "Warning: no CJK font found on build host (e.g. fonts-wqy-microhei). Minimal English Linux desktops may need a system CJK font." >&2
fi

(
    cd "$REPO_ROOT/yolo_studio"
    find . -type f -name '*.py' | while IFS= read -r rel; do
        mkdir -p "$APP/$(dirname "$rel")"
        cp "$rel" "$APP/$rel"
    done
)
cp -R "$REPO_ROOT/yolo_studio/assets" "$APP/"
cp "$REPO_ROOT/packaging/launch.py" "$RUNTIME/launch.py"
cp "$SCRIPT_DIR/launcher.sh" "$RELEASE/YOLO-Studio"
cp "$SCRIPT_DIR/install-desktop.sh" "$RELEASE/install-desktop.sh"
cp "$SCRIPT_DIR/start-here.txt" "$RELEASE/START HERE.txt"
sed -i 's/\r$//' "$RELEASE/YOLO-Studio" "$RELEASE/install-desktop.sh"
chmod +x "$RELEASE/YOLO-Studio" "$RELEASE/install-desktop.sh"

cp "$YOLO_MODEL_PATH" "$RELEASE/models/yolo11n.pt"
cp "$SAM_MODEL_PATH" "$RELEASE/models/sam2.1_t.pt"

if [ -d "$REPO_ROOT/examples" ]; then
    mkdir -p "$RELEASE/examples"
    find "$REPO_ROOT/examples" -maxdepth 1 -type f -exec cp {} "$RELEASE/examples/" \;
fi

echo "Running isolated runtime import smoke test..."
YOLO_STUDIO_HOME="$RELEASE" "$RUNTIME/bin/python3" -I -B -c "
import sys
sys.path.insert(0, '$RELEASE/app')
from importlib.metadata import version
import torch, torchvision, ultralytics, PySide6.QtWidgets, onnx, onnxruntime, onnxsim, onnxslim, yaml, yolo_studio
print(sys.version)
print(version('torch'), version('PySide6-Essentials'), 'cuda_built=', torch.version.cuda)
"
"$RUNTIME/bin/python3" -I -B -X utf8 "$RUNTIME/launch.py" --help >/dev/null

echo "Creating archive $OUTPUT_ABS ..."
tar -czf "$PENDING_TAR" -C "$STAGE" "YOLO-Studio"
mv -f "$PENDING_TAR" "$OUTPUT_ABS"
ls -lh "$OUTPUT_ABS"
