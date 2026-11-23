# MANIFEST — 停止レイテンシ計測の生データ

計測日時: 2026-11-13 00:07〜00:52 JST
実行バイナリ: `build/ConvoPeq_artefacts/Release/ConvoPeq.exe`（MSVC 14.52.37025 / Ninja Multi-Config / Release）
計測は `CONVOPEQ_STOP_PROBE=1` のときだけ有効。未設定時は `convo::probe::enabled()` が false を返し、記録も出力もしない。

| ファイル | 行数 | バイト | SHA-256（先頭16桁） | 内容 |
| --- | ---: | ---: | --- | --- |
| `p2_A.jsonl` | 5 | 495 | `75d142e8749b89f0` | 同期リビルド（既定 IR 192,000 samples）、終了予定 800 ms |
| `p2_B.jsonl` | 5 | 495 | `8f4146455418988d` | 同期リビルド（既定 IR）、終了予定 1400 ms |
| `p2_C.jsonl` | 5 | 495 | `e4f301f6f376d0a6` | 同期リビルド（既定 IR）、終了予定 1500 ms |
| `p2_D.jsonl` | 5 | 495 | `a49db7aaf1c20f2e` | 同期リビルド（既定 IR）、終了予定 4000 ms |
| `p2_E.jsonl` | 4 | 383 | `11680196488b1551` | 同期リビルド（既定 IR・Minimum）、終了予定 2500 ms |
| `p2_F.jsonl` | 5 | 495 | `92b2c53f732b60a3` | 同期リビルド（既定 IR）、終了予定 5000 ms |
| `p3_I.jsonl` | 6 | 569 | `c7bc9a62c2b0388f` | 実 IR 初回ロード（キャッシュ冷）、Mixed（旧スコープ） |
| `p3_J.jsonl` | 4 | 351 | `ac8a3be2f5fcb875` | 実 IR 2 回目（キャッシュ温）、Mixed（旧スコープ） |
| `p3_K.jsonl` | 28 | 2457 | `b1c531e6aa8cdb85` | 実 IR 再ロード 7 回、Mixed（旧スコープ） |
| `p3_L.jsonl` | 6 | 569 | `0c8b7401091380a1` | 実 IR 初回ロード（キャッシュ冷）、Minimum（旧スコープ） |
| `p3_M.jsonl` | 22 | 1973 | `dd8c9cf0ff0a3bc5` | 実 IR 再ロード 5 回、Minimum（旧スコープ） |
| `p4_N.jsonl` | 9 | 846 | `0886aba1287b2b06` | 終了予定 200 ms + 同期リビルド（旧スコープ） |
| `p5_P.jsonl` | 4 | 351 | `672f0c3705a5795d` | 実 IR 温キャッシュ、Mixed（タイトスコープ） |
| `p5_Q.jsonl` | 8 | 788 | `b41c512781250e20` | 実 IR 冷キャッシュ、Mixed（タイトスコープ） |
| `p5_R.jsonl` | 8 | 787 | `891aab7cf07a88c2` | 実 IR 冷キャッシュ、Minimum（タイトスコープ） |
| `p5_S.jsonl` | 9 | 846 | `cffeca1ca5f1a3f6` | 同期リビルド、終了予定 5000 ms（タイトスコープ） |
| `p5_T.jsonl` | 9 | 846 | `e01e1c2550eeabf4` | 同期リビルド、終了予定 150 ms（タイトスコープ） |

## 記録形式

```json
{"ts":<steady_clock マイクロ秒>,"ev":"run_start|first_check|request|run_exit|span|mark", ...}
```

- `span`: `op` / `us`（所要）/ `samples` / `fft` / `phase` / `cancelled`
- `request` / `run_exit`: `req_to_exit_us`（要求から終了まで。要求が無ければ `null`）、`req_to_first_check_us`
- `mark`: `op` / `a` / `b`（`perform_load` の `a` は `thread != nullptr` の真偽。`0` は同期実行）

## スコープについて

`p2` / `p3` / `p4` は `span` の終端が外側ブロックまで伸びていた（入れ子になる）版。総時間の算出には影響しない。
`p5` は変換・位相変換の `span` をラムダで囲み、操作の直前から直後までに締めた版。
