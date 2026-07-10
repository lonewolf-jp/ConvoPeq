' run-mempalace-mine-hidden.vbs
' Runs mine-copilot-sessions.ps1 with hidden window (no console popup)
' Called from Windows Task Scheduler
CreateObject("WScript.Shell").Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -File ""C:\VSC_Project\ConvoPeq\tools\mine-copilot-sessions.ps1""", 0, True
