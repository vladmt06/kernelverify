#!/bin/zsh
# Start a detached long run. Terminal can quit when this prints "armed".
#
# Installs a one-shot launchd job in the gui domain. The job runs the generic
# detached runner under caffeinate -i, so the machine will not idle-sleep while
# it waits for a strong-idle window and measures. The plist lives in
# bench/.cache/ rather than ~/Library/LaunchAgents, so it never runs at login.
# Re-running this script replaces the same harness's previous instance.
#
# The full operator card is bench/OPERATOR-CARD.md.

set -euo pipefail

ROOT="${0:A:h:h}"
PRINT_PLIST=0
if [[ "${1:-}" == "--print-plist" ]]; then
  PRINT_PLIST=1
  shift
fi

HARNESS_ARG="${1:-bench/measure_baselines.py}"
if (( $# > 0 )); then
  shift
fi

# A leading dash is a flag this script does not know, not a harness. Without
# this check, `--print-plsit` (one transposed letter) was accepted as a
# harness path, armed a real launchd job, printed "armed", and the machine
# then waited up to 12 hours to run a file that does not exist.
if [[ "$HARNESS_ARG" == -* ]]; then
  print -u2 -- "unknown flag: $HARNESS_ARG (the only flag is --print-plist, first)"
  exit 2
fi

HARNESS_ARGS=()
if (( $# > 0 )); then
  if [[ "$1" != "--" ]]; then
    print -u2 -- "harness arguments must follow --"
    exit 2
  fi
  shift
  HARNESS_ARGS=("$@")
fi

if [[ "$HARNESS_ARG" == /* ]]; then
  HARNESS="$HARNESS_ARG"
else
  HARNESS="$ROOT/$HARNESS_ARG"
fi

# Validated before EITHER path - printing a plist for a harness that does not
# exist is as wrong as arming one. A typo'd path otherwise armed a job whose
# every attempt exits 2 ("can't open file"), which the runner rightly calls
# crashed, but only after the operator has already walked away for the night.
if [[ ! -f "$HARNESS" ]]; then
  print -u2 -- "no such harness: $HARNESS"
  exit 2
fi

HARNESS_NAME="${HARNESS:t}"
HARNESS_STEM="${HARNESS_NAME:r}"
LABEL="com.kernelverify.detached-$HARNESS_STEM"
PLIST="$ROOT/bench/.cache/$LABEL.plist"
LOG="$ROOT/bench/.baselines/detached-$HARNESS_STEM.log"

RUNNER_ARGS=("$ROOT/bench/detached_run.py")
if [[ "$HARNESS_NAME" != "measure_baselines.py" ]]; then
  RUNNER_ARGS+=("--harness" "$HARNESS" "--protocol" "exit-code")
fi
if (( ${#HARNESS_ARGS} > 0 )); then
  RUNNER_ARGS+=("--" "${HARNESS_ARGS[@]}")
fi

xml_escape() {
  local value="$1"
  value="${value//&/&amp;}"
  value="${value//</&lt;}"
  value="${value//>/&gt;}"
  value="${value//\"/&quot;}"
  value="${value//\'/&apos;}"
  print -r -- "$value"
}

emit_string() {
  print -r -- "    <string>$(xml_escape "$1")</string>"
}

emit_plist() {
  print -r -- '<?xml version="1.0" encoding="UTF-8"?>'
  print -r -- '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
  print -r -- '<plist version="1.0">'
  print -r -- '<dict>'
  print -r -- "  <key>Label</key><string>$(xml_escape "$LABEL")</string>"
  print -r -- '  <key>ProgramArguments</key>'
  print -r -- '  <array>'
  emit_string '/usr/bin/caffeinate'
  emit_string '-i'
  emit_string "$ROOT/.venv/bin/python"
  emit_string '-u'
  for argument in "${RUNNER_ARGS[@]}"; do
    emit_string "$argument"
  done
  print -r -- '  </array>'
  print -r -- "  <key>WorkingDirectory</key><string>$(xml_escape "$ROOT")</string>"
  print -r -- "  <key>StandardOutPath</key><string>$(xml_escape "$LOG")</string>"
  print -r -- "  <key>StandardErrorPath</key><string>$(xml_escape "$LOG")</string>"
  print -r -- '  <key>RunAtLoad</key><true/>'
  print -r -- '  <key>ProcessType</key><string>Interactive</string>'
  print -r -- '</dict>'
  print -r -- '</plist>'
}

if (( PRINT_PLIST )); then
  emit_plist
  exit 0
fi

mkdir -p "$ROOT/bench/.cache" "$ROOT/bench/.baselines"
emit_plist > "$PLIST"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

echo "armed: $HARNESS_STEM is waiting for a strong-idle window"
echo "now plug in AC power, quit every app (Terminal included), and walk away"
echo "check later with:  tail -3 $LOG"
