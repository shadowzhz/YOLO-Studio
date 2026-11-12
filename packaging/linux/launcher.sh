#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="$HERE/runtime"

unset PYTHONPATH PYTHONHOME QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH

if [ -d "$RUNTIME/syslibs" ]; then
    need_syslibs=0
    if command -v ldconfig >/dev/null 2>&1; then
        ld_cache="$(ldconfig -p 2>/dev/null || true)"
        for lib in \
            libxcb-cursor.so.0 \
            libxcb-icccm.so.4 \
            libxcb-image.so.0 \
            libxcb-keysyms.so.1 \
            libxcb-randr.so.2 \
            libxcb-render-util.so.0 \
            libxcb-shape.so.0 \
            libxcb-xfixes.so.3 \
            libxcb-xinerama.so.0 \
            libxcb-xinput.so.0 \
            libxcb-util.so.1 \
            libxkbcommon-x11.so.0 \
            libxkbcommon.so.0
        do
            if [ -f "$RUNTIME/syslibs/$lib" ] && ! printf '%s\n' "$ld_cache" | grep -Fq "$lib"; then
                need_syslibs=1
                break
            fi
        done
    else
        need_syslibs=1
    fi
    if [ "$need_syslibs" -eq 1 ]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$RUNTIME/syslibs"
    fi
fi

if [ "$#" -eq 0 ]; then
    if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
        echo "[错误] 未检测到图形显示环境 (DISPLAY / WAYLAND_DISPLAY)。" >&2
        echo "       请在桌面环境运行，或使用命令行模式：\"$0\" --help" >&2
        exit 1
    fi
    if [ -z "${QT_QPA_PLATFORM:-}" ]; then
        if [ -n "${DISPLAY:-}" ]; then
            export QT_QPA_PLATFORM="xcb;wayland"
        else
            export QT_QPA_PLATFORM="wayland;xcb"
        fi
    fi
fi

exec "$RUNTIME/bin/python3" -I -B -X utf8 "$RUNTIME/launch.py" "$@"
