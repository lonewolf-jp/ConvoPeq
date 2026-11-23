# MANIFEST — 停止レイテンシ再集計

計測日時: 2026-11-13 05:20〜07:10 JST
実行バイナリ: `build/ConvoPeq_artefacts/Release/ConvoPeq.exe`（MSVC 14.52.37025 / Ninja Multi-Config / Release）
計測は `CONVOPEQ_STOP_PROBE=1` のときだけ有効。

| ファイル | 行数 | バイト | SHA-256（先頭16桁） | 内容 |
| --- | ---: | ---: | --- | --- |
| `m2_w2.jsonl` | 36 | 3415 | `a85a04c3ac6feeba` | C1/C2 基礎（12 試行・追加負荷なし） |
| `m2_learn.jsonl` | 12 | 1145 | `1a1e54155a957996` | C3（学習 ON 4 試行） |
| `m2_w5.jsonl` | 380 | 35510 | `74c18d566bd23aa8` | C4 W5 置換キャンセル（35 標本） |
| `m3_w6.jsonl` | 442 | 42379 | `19f073946285bb09` | C6 W6 置換キャンセル（25 標本） |
| `m1_E4.jsonl` | 85 | 8206 | `661c52e04f760d41` | C7 W6 探索（参考・別集計） |
| `m4_w5shutdown.jsonl` | 82 | 7685 | `a95b8618ef70d620` | C5 W5 shutdown 時 stopThread 待機（6 標本） |
| `m4_w6shutdown.jsonl` | 36 | 3202 | `2c338ba606818e8f` | upgrade 不成立（キャッシュ温）。C1/C2 のみ寄与 |
| `m5_w6shutdown.jsonl` | 144 | 14602 | `2ce1b4772b0aed82` | C8 W6 shutdown 時 stopThread 待機（6 標本） |
