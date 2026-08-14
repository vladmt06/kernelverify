#!/bin/zsh
# Start a detached binding run. Terminal can quit the moment this prints "armed".
#
# Installs a one-shot launchd job (gui domain) that runs bench/detached_run.py
# under caffeinate -i, so the machine will not idle-sleep while it waits for a
# strong-idle window and measures. The plist is generated into bench/.cache/
# rather than ~/Library/LaunchAgents, so it never runs again at login.
# Re-running this script replaces any previous instance.
#
# The full operator card is bench/OPERATOR-CARD.md.

set -euo pipefail

ROOT="${0:A:h:h}"
LABEL="com.kernelverify.binding-run"
PLIST="$ROOT/bench/.cache/$LABEL.plist"
LOG="$ROOT/bench/.baselines/detached.log"

mkdir -p "$ROOT/bench/.cache" "$ROOT/bench/.baselines"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/caffeinate</string>
    <string>-i</string>
    <string>$ROOT/.venv/bin/python</string>
    <string>-u</string>
    <string>$ROOT/bench/detached_run.py</string>
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
  <key>RunAtLoad</key><true/>
  <key>ProcessType</key><string>Interactive</string>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

echo "armed: detached run is waiting for a strong-idle window"
echo "now plug in AC power, quit every app (Terminal included), and walk away"
echo "check later with:  tail -3 $LOG"
