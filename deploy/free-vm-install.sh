#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/jeevith-tech/trading-bot.git}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/trading-bot}"
SERVICE_NAME="${SERVICE_NAME:-trading-bot-paper}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

echo "Installing system packages..."
sudo apt-get update
sudo apt-get install -y ca-certificates git python3

if [ -d "$INSTALL_DIR/.git" ]; then
  echo "Updating existing repo at $INSTALL_DIR..."
  git -C "$INSTALL_DIR" pull --ff-only
else
  echo "Cloning $REPO_URL into $INSTALL_DIR..."
  git clone "$REPO_URL" "$INSTALL_DIR"
fi

mkdir -p "$INSTALL_DIR/reports/live_paper"

echo "Creating systemd service..."
sudo tee "/etc/systemd/system/$SERVICE_NAME.service" >/dev/null <<SERVICE
[Unit]
Description=Trading Bot v17 Live Paper Tick
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$PYTHON_BIN scripts/live_paper_tick.py
SERVICE

echo "Creating systemd timer..."
sudo tee "/etc/systemd/system/$SERVICE_NAME.timer" >/dev/null <<TIMER
[Unit]
Description=Run Trading Bot v17 Live Paper Tick Every 15 Minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=15min
AccuracySec=1min
Persistent=true
Unit=$SERVICE_NAME.service

[Install]
WantedBy=timers.target
TIMER

echo "Enabling timer..."
sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME.timer"

echo "Running one immediate paper tick..."
sudo systemctl start "$SERVICE_NAME.service" || true

echo
echo "Free VM paper runner installed."
echo "Timer status:"
systemctl list-timers "$SERVICE_NAME.timer" --no-pager
echo
echo "Latest bot status file:"
echo "$INSTALL_DIR/reports/live_paper/status.md"
echo
echo "Useful commands:"
echo "  systemctl status $SERVICE_NAME.timer"
echo "  journalctl -u $SERVICE_NAME.service -n 80 --no-pager"
echo "  cat $INSTALL_DIR/reports/live_paper/status.md"
