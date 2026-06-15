# ConvoPeq 稼働ログ安定性検証レポート

**日付**: 2026-06-15
**対象ログ**: `ConvoPeq.log` (15 Jun 2026 10:53:05pm 開始)
**分析者**: GitHub Copilot (DeepSeek V4 Flash)

---

## 📊 総評: 概ね安定稼働中

クリティカルなクラッシュ、アサーション失敗、メモリ破壊の兆候は**見られません**。
ただし、**パフォーマンス上の懸念が1件**あります。

---

## 🔴 重大懸念: RECOVERY ポーリングループ

```
[RECOVERY] execute action=0
```

この行がログ中に **数百回** 出力されています。常に `action=0`（無処理）にもかかわらず、非常に高頻度で呼び出され続けています。

| 項目 | 内容 |
|---|---|
| ログ出現パターン | 大量連続、他のログ出力の合間にも頻出 |
| 想定される原因 | 高頻度タイマー（~10ms程度）またはビジーループ |
| 影響 | CPU使用率の無駄遣い、バッテリー消費、Audio Threadへの間接的負荷 |
| 推奨対応 | ポーリング周期の延長（例: 100ms→500ms以上）、またはイベント駆動への変更を検討 |

---

## 🟡 軽度の懸念事項

### 1. duplicate releaseResources 呼び出し

```
[DIAG] releaseResources: enter
[DIAG] releaseResources: duplicate release ignored (already Unprepared)
```

- **発生回数**: 2回
- **状態**: ガードは機能している（"duplicate release ignored"）
- **リスク**: 低（ガード済み）
- **推奨**: 二重呼び出しが発生するパスを特定し、根本原因の除去を推奨

### 2. REBUILD_MERGED の多発

Snapshot系の rebuild が高頻度でキューされ、デバウンスによりマージされている。

| Intent ID | 理由 | レイテンシ |
|---|---|---|
| 7-10, 16, 19 | `same_as_pending_would_merge` | 400ms |

```
[REBUILD_TELEMETRY] event=REBUILD_REQUESTED intentId=8 reason=enqueue_snapshot_command class=Snapshot
[REBUILD_TELEMETRY] event=REBUILD_MERGED intentId=8 reason=same_as_pending_would_merge
```

- **状態**: 400ms のデバウンスが機能しており、機能的には問題なし
- **リスク**: 低（設計範囲内）
- **推奨**: トリガー元（UIパラメータ変更かタイマー起因か）の確認。多数の merge が発生した後の generation=11 でも正常動作確認済み

### 3. convolverParamsChanged → DEFERRED パス

```
[REBUILD_TELEMETRY] event=REBUILD_DEFERRED intentId=22 reason=prepared_ir_apply_window
```

- **状態**: IR適用直後のウィンドウ中に Structural rebuild が deferred され、後で timerCallback 経由でリリースされた。正常動作
- **リスク**: 低
- **注意点**: この deferred 解除が timer polling に依存している場合、RECOVERY ポーリングと合わせて見直しを検討

---

## 🟢 正常動作確認項目

### 初期化チェーン (全成功)

| フェーズ | 結果 |
|---|---|
| `ippInit()` succeeded | ✅ |
| `EQ_CTOR` → `resetToDefaults` | ✅ (複数回) |
| `DSPCORE_PREPARE` 全フェーズ | ✅ |
| `EQ_PREPARE` scratch/dry/parallel/xfade/agc 全確保 | ✅ |

### サンプルレート遷移 (正常)

```
48000 → 192000 (prepareToPlay)
spb: 65536 → 1024
processingRate: 384000 / 768000 / 384000
```

すべての再初期化でバッファ再確保が成功。

### トランザクションカウンタ (健全)

| メトリクス | 最終値 | 評価 |
|---|---|---|
| `lifecycle(pub/ret/reclaim)` | 9/0/0 | 全publish成功、retire発生なし ✅ |
| `rebuild(req/queued/blockP/blockR/queueFull/drain/match/fallback)` | 11/11/0/0/0/0/0/0 | リクエスト=キュー、ブロッキング/溢れ/フォールバックなし ✅ |
| `eqCacheMiss(create/lookup)` | 0/0 | EQキャッシュヒット率良好 ✅ |
| `convDebounce(req/defer/sched/trigger)` | 0/0/0/0 | コンボルバデバウンス正常 ✅ |
| `shutdownPhase` | RUNNING | 正常動作中 ✅ |

### NoiseShaperLearner (正常動作)

```
worker: loop iter=0..80+
accepted=136792 bufferedSamples=3840000
bankIndex=107, sampleRateHz=192000
```

ワーカースレッドが正常にサンプルを収集・処理中。

### applyComputedIR

```
applyComputedIR: applied scaleFactor=0.132589 to timeDomainIR and partitionData
```

IR適用が正常に完了。

### その他正常動作

| 項目 | 結果 |
|---|---|
| `setNoiseShaperType: newType=2 wasAdaptive=1` | ✅ |
| `MixedPhaseUI` Close/Resume | ✅ |
| `AudioEngine.processLearningCommands` DSPReady / Start dispatch | ✅ |
| `MainWindow::changeListenerCallback` enter/leave | ✅ (複数回) |

---

## 📋 推奨アクション一覧

| 優先度 | 項目 | 内容 | 該当箇所 |
|---|---|---|---|
| **🔴 高** | RECOVERY ポーリング | タイマー周期の延長、またはイベント駆動化を検討 | `[RECOVERY] execute action=0` |
| **🟡 中** | releaseResources 二重呼び出し | 呼び出しパスを確認し、予防ガードに加えて原因除去を推奨 | `AudioEngine.Processing.ReleaseResources.cpp` |
| **🟢 低** | REBUILD_MERGED 多発のトリガー元確認 | UI変更のバーストが疑われる場合、変更送信側のデバウンス追加を検討 | Snapshot rebuild 発行元 |
| **ℹ️ 情報** | 全体的な安定性 | クリティカルな問題はなし、上記対応でより堅牢に | - |

---

## 付録: ログ統計サマリ

| 指標 | 値 |
|---|---|
| 総 rebuild intent | 33 (うち merged 6, deferred 1) |
| 総 publish | 9 |
| rebuild generation | 11 |
| NoiseShaperLearner accepted samples | 136,792 |
| NoiseShaperLearner buffered samples | 3,840,000 |
| サンプルレート変化 | 48000 → 192000 Hz |
| オーバーサンプリング後レート | 384000 / 768000 Hz |
| 内部ブロックサイズ (max) | 524288 samples |
