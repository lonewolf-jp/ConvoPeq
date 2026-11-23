# MANIFEST — 負荷条件別の shutdown 停止計測

計測日時: 2026-11-13 08:10〜08:55 JST
実行バイナリ: `build/ConvoPeq_artefacts/Release/ConvoPeq.exe`（MSVC 14.52.37025 / Ninja Multi-Config / Release）
計測は `CONVOPEQ_STOP_PROBE=1` のときだけ有効。

| ファイル | 行数 | バイト | SHA-256（先頭16桁） | 内容 |
| --- | ---: | ---: | --- | --- |
| `m6_c5n.jsonl` | 449 | 42731 | `fd970f97575006c4` | C5 通常負荷（30 試行） |
| `m6_c5s.jsonl` | 420 | 38666 | `45b9382435992796` | C5 EQ 切替（30 試行） |
| `m6_c8n.jsonl` | 636 | 63981 | `714fc07396d5be7f` | C8 通常負荷（30 試行） |
| `m6_c8s.jsonl` | 492 | 47518 | `0659aec23eaf7b09` | C8 EQ 切替（30 試行） |
| `m7_c5s2.jsonl` | 390 | 35337 | `9ca670a9e7071c76` | C5 bypass 切替 2s（30 試行） |
| `m7_c8s2.jsonl` | 300 | 28564 | `a9a2e8f46f6ff843` | C8 bypass 切替 2s（30 試行） |
| `m8_c5s3.jsonl` | 330 | 28678 | `af6c9cb419ceac13` | C5 bypass 切替 260ms + 6 秒 IR（無効・上限超過） |
| `m8_c8s3.jsonl` | 180 | 16017 | `762fdf4c82b9e1d6` | C8 bypass 切替 260ms（upgrade 起動せず・無効） |
