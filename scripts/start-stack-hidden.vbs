' Window-less shim for the "llama-stack" Scheduled Task.
'
' A .bat action under a logon trigger always pops a console window.
' WScript.Shell with intWindowStyle=0 starts it fully hidden. Nothing is lost:
' start-stack.bat already redirects all output to logs\stack.log.
'
' bWaitOnReturn=True keeps this script alive for as long as the stack runs, so
' the task stays "Running" and its restart-on-failure settings still apply.

Option Explicit
Dim shell, fso, here, bat, rc
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)
bat = fso.BuildPath(here, "start-stack.bat")
If Not fso.FileExists(bat) Then WScript.Quit 1
shell.CurrentDirectory = here
rc = shell.Run("""" & bat & """", 0, True)
WScript.Quit rc
