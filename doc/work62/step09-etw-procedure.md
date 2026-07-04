# Step 9: ETW 取得手順

## 事前準備
wpr.exe / wpa.exe は利用可能（確認済み）

## 取得手順（管理者PowerShell）

```powershell
# 1. トレース開始（推奨: カスタムプロファイル）
wpr -start tools\convopeq-xrun-etl.wprp -start ConvoPeq

# 2. ConvoPeq を起動し音飛びを再現（1〜2分）

# 3. トレース停止
wpr -stop ConvoPeqTrace.etl

# 4. WPA で解析
wpa ConvoPeqTrace.etl
```

## 評価指標（優先順位順）

| 指標 | 閾値 | 意味 |
|------|------|------|
| **ReadyThread P99** | >1ms | Schedulerによるディスパッチ遅延を強く示唆 |
| ReadyThread P95 | >500μs | テール遅延の兆候 |
| ReadyThread 最大 | >3ms | 単発の大きな遅延 |
| XRUN時刻との一致 | — | Ready待ちとXRUN callbackが一致→因果関係 |
| CSwitch頻度 | — | コンテキストスイッチ過多 |
| DPC/ISR時間 | — | ドライバ（Voicemeeter）起因の可能性 |

## 分岐ロジック

```
ReadyThread P99 > 1ms?
  ├── Yes → Schedulerによるディスパッチ遅延を強く示唆
  │         → Step 11a (MMCSS) + Step 11b (2048) で緩和策へ
  │
  └── No → ConvoPeq/JUCE 側の精査へ
            → Step 10a〜10e (CallbackArrival 解析)
```
