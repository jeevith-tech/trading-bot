#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-$HOME/trading-bot}"
SERVICE_NAME="${SERVICE_NAME:-trading-bot-paper}"

echo "Timer:"
systemctl status "$SERVICE_NAME.timer" --no-pager || true
echo
echo "Last service logs:"
journalctl -u "$SERVICE_NAME.service" -n 80 --no-pager || true
echo
echo "Status file:"
if [ -f "$INSTALL_DIR/reports/live_paper/status.md" ]; then
  cat "$INSTALL_DIR/reports/live_paper/status.md"
else
  echo "No status file yet."
fi
