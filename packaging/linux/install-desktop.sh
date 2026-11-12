#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESKTOP_FILE="$APP_DIR/yolo-studio.desktop"

mkdir -p "$APP_DIR"
cat >"$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=YOLO Studio
Comment=YOLO 目标检测、分割标注与端侧部署工作台
Exec="$HERE/YOLO-Studio"
Icon=$HERE/app/yolo_studio/assets/app.svg
Path=$HERE
Terminal=false
Categories=Development;Science;Graphics;
StartupWMClass=YOLO Studio
EOF
chmod +x "$DESKTOP_FILE"

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APP_DIR" >/dev/null 2>&1 || true
fi

echo "已创建桌面/应用菜单快捷方式：$DESKTOP_FILE"
echo "如需卸载快捷方式，直接删除该文件即可。"
