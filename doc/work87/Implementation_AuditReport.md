# 実装監査レポート

> 作成日: 2026-07-25
> 監査対象: `doc/work87/RemediationPlan.md` の実装
> 監査方法: 全変更ファイルのgrep網羅確認 + VS Code diagnostics + semble

---

## 監査結果サマリー

| 項目 | 判定 | 備考 |
|------|------|------|
| P0-1 全アクセス網羅 | ✅ | 30/30箇所 .load/.store(relaxed) 確認 |
| P0-1 取りこぼし | ✅ なし | テストファイル・ドキュメントに参照なし |
| P1-0 | ✅ | copyLatest acquire順序修正確認 |
| P1-1 | ✅ | #define 40 確認 |
| P1-2 | ✅ | kLoadStride2Offset/avxMinConvIdx 確認 |
| P2-1 | ✅ | 3/3箇所 static_cast<size_t> 確認 |
| P2-5 | ✅ | 2/2箇所 {false} 初期化確認 |
| P3-1 | ✅ | 2/2クラス LEAK_DETECTOR追加確認 |
| リグレッション | ✅ なし | 全変更ファイル diagnostic error 0 |
| ビルド確認 | ✅ | Debug Build 成功 |

## P0-1 詳細監査

全30アクセス（宣言4 + Worker 8 + RT 18）を確認:

### 宣言（EQProcessor.h）

| 行 | 変数 | 型 | 初期値 |
|----|------|-----|--------|
| 694 | `rtAgcCurrentGainShadow` | `std::atomic<double>` | `1.0` |
| 695 | `rtAgcEnvInputShadow` | `std::atomic<double>` | `0.0` |
| 696 | `rtAgcEnvOutputShadow` | `std::atomic<double>` | `0.0` |
| 697 | `rtBypassedShadow` | `std::atomic<bool>` | `false` |

### Worker書込（EQProcessor.Core.cpp）

| 行 | 関数 | 操作 |
|----|------|------|
| 592 | `syncStateFrom()` | `rtBypassedShadow.store(syncedBypassed, relaxed)` |
| 594 | | `rtAgcCurrentGainShadow.store(syncedAgcCurrentGain, relaxed)` |
| 595 | | `rtAgcEnvInputShadow.store(syncedAgcEnvInput, relaxed)` |
| 596 | | `rtAgcEnvOutputShadow.store(syncedAgcEnvOutput, relaxed)` |
| 655 | `syncGlobalStateFrom()` | `rtBypassedShadow.store(syncedBypassed, relaxed)` |
| 657 | | `rtAgcCurrentGainShadow.store(syncedAgcCurrentGain, relaxed)` |
| 658 | | `rtAgcEnvInputShadow.store(syncedAgcEnvInput, relaxed)` |
| 659 | | `rtAgcEnvOutputShadow.store(syncedAgcEnvOutput, relaxed)` |

### RT読込/書込（EQProcessor.Processing.cpp）

読込.load(relaxed): 415, 416, 417, 497, 509, 1020 = 6箇所
書込.store(relaxed): 434, 435, 436, 506, 517, 586, 587, 588, 1009, 1011, 1071, 1072, 1073 = 13箇所

### スコープからの除外確認

`rtActiveStructureShadow`, `rtDeferredBandResetMask`, `rtSeenBandResetSerial`, `rtSeenAgcResetSerial` は本P0のスコープ外であり、変更していない。これらは別途将来検討。

`m_rtBypassShadow`（RT単一スレッドのみ書込）は non-atomic のまま。これは正しい。

### リグレッション評価

| 観点 | 評価 | 理由 |
|------|------|------|
| 未初期化読み取り | なし | 全変数に初期化子あり |
| 誤ったメモリオーダー | なし | 全アクセス relaxed 統一 |
| スレッド安全性 | 改善 | UBだったデータ競合がatomic化で解消 |
| パフォーマンス | 影響軽微 | relaxed はx64でlock-free, 追加バリアなし |
| 既存の relaxed 使用との整合 | 良好 | DSPCoreDouble/Floatでも relaxed 使用 |
