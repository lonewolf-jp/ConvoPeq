# MANIFEST — 切替負荷での shutdown 停止計測

計測日時: 2026-11-13 09:10〜10:50 JST
実行バイナリ: `build/ConvoPeq_artefacts/Release/ConvoPeq.exe`（MSVC 14.52.37025 / Ninja Multi-Config / Release）
計測は `CONVOPEQ_STOP_PROBE=1` のときだけ有効。

| ファイル | 行数 | バイト | SHA-256（先頭16桁） | 内容 |
| --- | ---: | ---: | --- | --- |
| `m6_c5n.jsonl` | 449 | 42731 | `fd970f97575006c4` | 通常負荷 C5（30 試行・29 有効） |
| `m6_c8n.jsonl` | 636 | 63981 | `714fc07396d5be7f` | 通常負荷 C8（30 試行・27 有効） |
| `m13_c5eq.jsonl` | 390 | 36150 | `58d56ee1d17464ed` | C5 EQ 切替（30 試行・有効 30） |
| `m13_c5bp.jsonl` | 360 | 31914 | `b1ae18d9554a0dd1` | C5 bypass 切替（30 試行・有効 0） |
| `m13_c8eq.jsonl` | 442 | 43697 | `1e6b503482966fc8` | C8 EQ 切替 0.5s IR（30 試行・有効 2） |
| `m13_c8bp.jsonl` | 532 | 53469 | `376e69708ff6123c` | C8 bypass 切替（30 試行・有効 0） |
| `m14_c8eq.jsonl` | 759 | 76523 | `ac9ac2f0464449ef` | C8 EQ 切替 平坦 5.2s IR（30 試行・有効 30） |
| `p2_c5bp.jsonl` | 36 | 3190 | `ae2ab1cd2ac57d1d` | C5 bypass 平坦 IR 追試（3 試行） |
| `p2_c8bp.jsonl` | 30 | 2808 | `59d0950315943f9d` | C8 bypass 平坦 IR 追試（3 試行） |

## 有効条件

- V-1: `run_start` が切替開始より前
- V-2: `run_exit` が shutdown 到着（`stopLearning_terminal`）より後
- V-3: C8 は `upgrade_start` の `appliedFft = 512`
- V-4: `stopThread` スパンが存在する
