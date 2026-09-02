$ErrorActionPreference = 'SilentlyContinue'
$ev = Get-WinEvent -FilterHashtable @{LogName='Application'; Id=1000} -MaxEvents 20 | Where-Object { $_.Message -match 'ConvoPeq' } | Select-Object -First 5
foreach ($e in $ev) {
  $m = $e.Message
  $app = ($m -split "`r?`n" | Where-Object { $_ -match '障害が発生しているアプリケーション名|フォールト オフセット|Faulting' }) -join ' | '
  Write-Output ($e.TimeCreated.ToString('MM-dd HH:mm:ss') + ' :: ' + $app)
}
if (-not $ev) { Write-Output 'none' }
