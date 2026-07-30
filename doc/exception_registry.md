# MMCSS 例外登録簿

MMCSS（Multimedia Class Scheduler Service）API の呼び出しを一元管理する例外登録簿。
全 MMCSS 関数呼び出しは本登録簿に記載必須。

## 登録ルール

1. 新規 MMCSS 呼び出し追加時は本登録簿に追記する
2. CI の ast-grep ルールで未登録の呼び出しを検出する
3. コードレビュー時に本登録簿を参照する

## 登録一覧

| # | 関数名 | ファイル | 行 | 契約 | 理由 | 承認日 | 承認者 |
|---|--------|---------|----|------|------|-------|-------|
| EX-1 | `AvRevertMmThreadCharacteristics` | `AudioEngine.Mmcss.cpp` | 204 | MMCSS-EX-1 | ASIO 停止時のMMCSS特性解除（Audio Thread で実行、RT-safe） | 2026-07-29 | TBD |
| EX-2 | `AvSetMmThreadCharacteristicsW` | `AudioEngine.Mmcss.cpp` | 43 | MMCSS-EX-2 | オーディオデバイス起動時のMMCSS特性設定（NonRT, Message Thread） | 2026-07-29 | TBD |
| EX-3 | `AvSetMmThreadPriority` | `AudioEngine.Mmcss.cpp` | 124 | MMCSS-EX-3 | ASIO スレッド優先度設定（Audio Thread） | 2026-07-29 | TBD |
| EX-4 | `AvSetMmThreadPriority` | `AudioEngine.Mmcss.cpp` | 169 | MMCSS-EX-3 | セカンダリスレッド優先度設定 | 2026-07-29 | TBD |

## 契約

| ID | 契約 |
|----|------|
| MMCSS-EX-1 | `AvRevertMmThreadCharacteristics` は RT パス（Audio Thread）からのみ呼び出す。MSDN 要件: same-thread。 |
| MMCSS-EX-2 | `AvSetMmThreadCharacteristicsW` は NonRT（Message Thread）からのみ呼び出す。 |
| MMCSS-EX-3 | `AvSetMmThreadPriority` は Audio Thread からのみ呼び出す。 |
| MMCSS-EX-4 | 全 MMCSS 関数呼び出しは本登録簿に登録する。 |
| MMCSS-EX-5 | 新規 MMCSS 呼び出し追加時は必ず本登録簿に追記する（CI gate）。 |
