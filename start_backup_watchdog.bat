@echo off
start /B pythonw scripts\17_scheduled_backup.py --interval 6
echo Backup watchdog started (every 6 hours). Logs: logs\backup_watchdog.log
