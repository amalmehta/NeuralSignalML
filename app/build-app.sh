#!/bin/bash
# Build "Neural Signal ML.app" into app/build/. Usage: app/build-app.sh [--open]
set -euo pipefail
cd "$(dirname "$0")"
PROJECT="$(cd .. && pwd)"
APP="build/Neural Signal ML.app"

swift build -c release
BIN="$(swift build -c release --show-bin-path)/NeuralSignalML"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/Neural Signal ML"
[ -f Resources/AppIcon.icns ] && cp Resources/AppIcon.icns "$APP/Contents/Resources/"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Neural Signal ML</string>
  <key>CFBundleDisplayName</key><string>Neural Signal ML</string>
  <key>CFBundleExecutable</key><string>Neural Signal ML</string>
  <key>CFBundleIdentifier</key><string>com.amalmehta.neuralsignalml</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMLProjectFolder</key><string>${PROJECT}</string>
</dict>
</plist>
PLIST

codesign --force --sign - "$APP" >/dev/null 2>&1 || true
echo "Built $(pwd)/$APP"
[ "${1:-}" = "--open" ] && open "$APP"
exit 0
