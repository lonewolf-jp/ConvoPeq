# MANIFEST — 停止レイテンシ実測（W2 / W5 / W6 / stopLearning）

計測日時: 2026-11-13 05:20〜06:10 JST
実行バイナリ: `build/ConvoPeq_artefacts/Release/ConvoPeq.exe`（MSVC 14.52.37025 / Ninja Multi-Config / Release）
計測は `CONVOPEQ_STOP_PROBE=1` のときだけ有効。未設定なら `convo::probe::enabled()` が false を返し、記録も出力もしない。

| ファイル | 行数 | バイト | SHA-256（先頭16桁） | 内容 |
| --- | ---: | ---: | --- | --- |
| `m2_w2.jsonl` | 36 | 3415 | `a85a04c3ac6feeba` | W2 coord_stop + stopLearning terminal（12 試行） |
| `m2_learn.jsonl` | 12 | 1145 | `1a1e54155a957996` | 学習 ON の stopLearning terminal（4 試行） |
| `m2_w5.jsonl` | 380 | 35510 | `74c18d566bd23aa8` | W5 LoaderThread 走行中停止（5 試行 / 35 標本） |
| `m1_E4.jsonl` | 85 | 8206 | `661c52e04f760d41` | W6 探索（1 試行 / 5 標本） |
| `m3_w6.jsonl` | 442 | 42379 | `19f073946285bb09` | W6 ProgressiveUpgradeThread 走行中停止（6 試行 / 25 標本）ほか |

## 記録形式

```json
{"ts":<steady_clock マイクロ秒>,"ev":"run_start|first_check|request|run_exit|span|mark", ...}
```

- `span`: `op` / `us` / `samples` / `fft` / `phase` / `cancelled`
- `mark`: `op` / `a` / `b`（`coord_stop_result` と `*_stopThread` の `a` は協調終了の真偽）
- `request` / `run_exit`: `req_to_exit_us` / `req_to_first_check_us`
