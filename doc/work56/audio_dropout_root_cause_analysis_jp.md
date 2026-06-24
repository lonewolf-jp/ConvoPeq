# 音飛び原因 完全検証レポート

**作成日**: 2026-06-24
**対象**: ConvoPeq 音声アプリ（JUCE V8.0.12 / スタンドアローン）
**ログソース**: `c:\Users\user\Desktop\ConvoPeq\develop\ConvoPeq.log`（2245行）
**使用ツール**: grep_search, serena (find_symbol), codegraph (query_codebase/find_callees), semble (search), AiDex (query), read_file（全7ツール使用）

---

## 1. エグゼクティブサマリー

音飛びの**根本原因（PRIMARY）**は、オーバーサンプリング Auto モード（`manualOversamplingFactor = 0`）の未解決値が、publish 検証対象の world にそのまま伝播し、`validateResources()` で必ず拒否されることにあります。これにより、最初の DSP 公開（generation=1）が失敗し、プレースホルダー DSP（bypass・無音状態）が継続使用される期間に音飛びが発生します。

加えて、SR変更（48k→192k）に伴う**過剰リビルドカスケード（SECONDARY）**と、EQProcessor の**フル再割当（TERTIARY）**が、ドロップアウト期間を延長・悪化させています。

---

## 2. 根本原因の完全因果連鎖（PRIMARY）

### 2.1 因果連鎖図

```
manualOversamplingFactor = 0 (Auto モード)
  ↓ captureBuildParameterSnapshot() [AudioEngine.RebuildDispatch.cpp:35]
snapshot.oversamplingFactor = 0
  ↓ task.buildInput.oversamplingFactor = paramSnapshot.oversamplingFactor
task.buildInput.oversamplingFactor = 0
  ↓ finalizeRuntimeBuildSnapshot() [RebuildDispatch.cpp:101]
snapshot.buildInput.oversamplingFactor = std::max(0, 0) = 0  ← 0を許容するバグ
  ↓ buildRuntimePublishWorld(useSealedSnapshot=true) [RuntimeBuilder.cpp:308-313]
worldOwner->resource.oversamplingFactor = sealedSnapshot->buildInput.oversamplingFactor = 0
  ↓ validateResources() [RuntimePublicationValidator.cpp:119]
os < 1 → return false → InvalidResources
  ↓ emitValidationEvent(InvalidResources) [RuntimeHealthMonitor.cpp]
eventCode = 6002 (EVENT_VALIDATION_RESOURCE_FAILURE)
  ↓ validatePublicationNonRt returns false [AudioEngine.h:2750-2870]
  ↓ publishWorld() returns PublishStageResult::Rejected (=1) [RuntimePublicationCoordinator.h:109-114]
  ↓ trySubmit logs FAILED [RuntimePublicationOrchestrator.cpp]
[DIAG] trySubmit: executor_.publish FAILED gen=1 result=1
```

### 2.2 ログの決定的証拠

| 行 | ログ出力 | 解釈 |
|---|---|---|
| 15 | `requestRebuild(sr,bs): task queued generation=1 SR=48000.00` | SR=48k で gen=1 ビルド開始 |
| 61 | `[CONV_STATUS] gen=1 osFactor=0 processingRate=384000.0` | **`buildInput.oversamplingFactor = 0`（Auto未解決）** |
| 62 | `[HEALTH] eventCode=6002 severity=1 value=0` | **`EVENT_VALIDATION_RESOURCE_FAILURE`（InvalidResources）** |
| 63 | `[DIAG] trySubmit: executor_.publish FAILED gen=1 result=1` | **Rejected（ビルド成果物破棄）** |
| 64 | `[DIAG] prepareToPlay: enter spb=1024 sr=192000.00` | SR変更（48k→192k）開始 |
| 68 | `prepareToPlay: rebuild request generation reset to 0` | generation リセット |

### 2.3 eventCode=6002 の定義（コード確認済み）

```
EVENT_VALIDATION_RESOURCE_FAILURE = 6002
```

`RuntimePublicationValidator::validatePublication()` が `InvalidResources` を返した際、`RuntimeHealthMonitor::emitValidationEvent()` がこのイベントコードを発行します。

### 2.4 validateResources() の拒否条件

`RuntimePublicationValidator.cpp` の `validateResources()` は、以下のいずれかで `false`（= `InvalidResources`）を返します：

```cpp
const int os = resource.oversamplingFactor;
if (os < 1 || os > 16 || (os & (os - 1)) != 0)
    return false;  // osFactor=0 は os < 1 で即座に false!
```

`osFactor=0` の場合、`0 < 1` = `true` となり即座に `return false` → `InvalidResources` となります。

---

## 3. gen=1 失敗と gen=3 成功の決定的違い

| Generation | SR | osFactor | processingRate | 結果 |
|---|---|---|---|---|
| **gen=1** | 48000 | **0** | 384000 | **FAILED (InvalidResources)** |
| gen=3 | 192000 | **2** | 384000 | SUCCEEDED |
| gen=7 | 192000 | 2 | 384000 | SUCCEEDED |

- **gen=1**: `buildInput.oversamplingFactor = 0`（Auto未解決のままworldに伝播）→ `validateResources()` で `os < 1` により拒否
- **gen=3**: `buildInput.oversamplingFactor = 2`（384000/192000=2 が解決済み）→ 検証成功

**Auto モードの解決ロジックが実行時によって結果が異なる**という一貫性欠陥が確定しました。

### CONV_STATUSログの意味

```
osFactor=        → buildInput.oversamplingFactor（= manualOversamplingFactor = 0 for Auto）
processingRate=  → newDSP->sampleRate * newDSP->oversamplingFactor（DSPCore内部の解決済み値）
```

gen=1ログ `osFactor=0 processingRate=384000.0`：

- `buildInput.oversamplingFactor = 0`（Auto未解決）
- `newDSP->oversamplingFactor = 8`（DSPCore内部で解決済み: 48000×8=384000）

DSPCore内部（`DSPCoreLifecycle.cpp:72-107`）ではAuto解決ロジックが存在しますが、この解決済み値が `world.resource.oversamplingFactor` に反映されていません。

---

## 4. ドロップアウト発生メカニズム

gen=1のビルド成果物（DSPCore）が `InvalidResources` で破棄されたため、**プレースホルダー DSP（bypass状態・IR未ロード）** が使い続けられます。この期間中にオーディオ処理が行われると、無音または不連続な信号が出力され、**音飛び**として知覚されます。

```
[時系列]
gen=1 ビルド完了 (86.4ms)
  → publish FAILED (InvalidResources) ← ここでDSP破棄
  → プレースホルダーDSP継続使用（bypass/無音）
  → SR変更 prepareToPlay 開始
  → generation リセット
  → gen=3 ビルド (約170ms後)
  → publish SUCCEEDED ← ここでようやく正常DSP使用開始
```

プレースホルダーDSPが使用される期間（gen=1失敗〜gen=3成功まで）に音飛びが発生します。

---

## 5. 原因の3階層構造

### 5.1 🔴 PRIMARY: Auto モード（0）の oversamplingFactor 未解決

`manualOversamplingFactor = 0`（Auto）が、DSPCore内部の解決ロジック（`DSPCoreLifecycle.cpp:72-107`）を経由せずに、そのまま `world.resource.oversamplingFactor` に伝播します。

**欠陥箇所（4地点）**:

| 地点 | ファイル:行 | 問題 |
|---|---|---|
| A | `AudioEngine.RebuildDispatch.cpp:35` | `captureBuildParameterSnapshot()` が `manualOversamplingFactor=0` をそのまま取得 |
| B | `AudioEngine.RebuildDispatch.cpp:101` | `finalizeRuntimeBuildSnapshot()` が `std::max(0, ...)` で0を許容 |
| C | `RuntimeBuilder.cpp:308-313` | `useSealedSnapshot=true` 時に `sealedSnapshot->buildInput.oversamplingFactor=0` をworldに設定 |
| D | `RuntimePublicationValidator.cpp:119` | `os < 1` で0を拒否（最終防衛ラインだが、ここで握りつぶされる） |

### 5.2 🟡 SECONDARY: SR変更時の過剰リビルドカスケード

`prepareToPlay` のSR変更（48k→192k）により、5世代のビルドが連鎖発生：

| 行 | intentId | イベント | generation |
|---|---|---|---|
| 120 | 3 | `requestRebuild_kind_entry` → `delegate_requestRebuild_sr_bs` | (gen=1再) |
| 136 | 4 | `requestRebuild_sr_bs` → `task_queued SR=192000` | gen=1 |
| 138 | — | `setNoiseShaperType: newType=2 wasAdaptive=1` | — |
| 145-151 | 7-10 | 4つの `enqueue_snapshot_command` がすべて `MERGED`（latencyMs=400ms） | — |
| 158 | 11 | `requestRebuild_kind_entry`（NoiseShaper変更による再エントリ） | — |
| 158 | 12 | `requestRebuild_sr_bs` → `task_queued SR=192000` | gen=3 |
| 272 | — | `task_queued SR=192000` | gen=4 |
| 300 | — | `task_queued SR=192000` | gen=5 |
| 314 | — | `task_queued SR=192000` | gen=6 |
| 392 | — | `task_queued SR=192000` | gen=7 |

各ビルドは90-110ms + IRリビルド最大440ms。この間、古いDSPが使い続けられ、クロスフェード品質が低下します。

**カスケードの主なトリガー**:

1. SR変更（48k→192k）に伴う `prepareToPlay`
2. NoiseShaper type変更（0→2: `setNoiseShaperType newType=2 wasAdaptive=1`）
3. EQパラメータ変更による snapshot command の連続発行（4回が `same_as_pending_would_merge` でマージ）

### 5.3 🟠 TERTIARY: EQ_PREPARE capacity=0 フル再割当

全 `[EQ_PREPARE]` イベントで `scratch: required=4194304 capacity=0` が発生。毎回4MB + 2MBのメモリ再割当が発生します。DSPCoreが毎回新規作成されるため、EQProcessorのバッファが再利用されません。

---

## 6. 改修案

### 6.1 🔴 PRIMARY Fix: Auto モードの解決ロジック追加

**方針**: `captureBuildParameterSnapshot()` で Auto（0）を実際の値に解決する。DSPCoreLifecycle.cpp の解決ロジックと同一の計算を適用。

**追加関数**（`AudioEngine.RebuildDispatch.cpp` の無名名前空間内）:

```cpp
// Auto モード（manualFactor=0）の oversamplingFactor を実際の値に解決する。
// DSPCoreLifecycle.cpp のロジックと同一の計算。
// manualFactor > 0 の場合はそのまま返す（手動指定尊重）。
inline int resolveAutoOversamplingFactor(double sampleRate, int manualFactor) noexcept
{
    if (manualFactor > 0)
        return manualFactor;

    // Auto: processingRate 384kHz を目標として、サンプルレートから最適な2の冪乗を算出
    if (sampleRate < 88200.0)
        return 8;   // 44.1k/48k → 8x
    if (sampleRate < 176400.0)
        return 4;   // 88.2k/96k → 4x
    if (sampleRate < 352800.0)
        return 2;   // 176.4k/192k → 2x
    return 1;       // 352.8k+ → 1x
}
```

**適用箇所**（4地点）:

#### 地点A: `captureBuildParameterSnapshot()`

```diff
  snapshot.oversamplingFactor = convo::consumeAtomic(
      engine.manualOversamplingFactor, std::memory_order_acquire);
+ // Auto モード（0）を実際の値に解決。未解決のまま publish されると
+ // validateResources() で InvalidResources 拒否される（音飛び原因 #56）。
+ snapshot.oversamplingFactor = resolveAutoOversamplingFactor(
+     engine.currentSampleRate, snapshot.oversamplingFactor);
```

#### 地点B: `requestRebuild()` 内（ターゲットSRで再解決）

```diff
  // パラメータスナップショットを取得
  auto paramSnapshot = captureBuildParameterSnapshot(*this);
+ // ターゲットサンプルレートで Auto を再解決（SR変更直後の正確性確保）
+ paramSnapshot.oversamplingFactor = resolveAutoOversamplingFactor(
+     sampleRate, paramSnapshot.oversamplingFactor);
```

#### 地点C: `finalizeRuntimeBuildSnapshot()`

```diff
- snapshot.buildInput.oversamplingFactor = std::max(0, paramSnapshot.oversamplingFactor);
+ // 最小値1を保証（0はAuto未解決の不正値）。resolveAutoOversamplingFactor で
+ // 既に解決済みだが、sealedSnapshot 経由の別パスも含む最終防衛ライン。
+ snapshot.buildInput.oversamplingFactor = std::max(1, paramSnapshot.oversamplingFactor);
```

#### 地点D: `RuntimeBuilder.cpp` sealedSnapshot パス

```diff
  if (useSealedSnapshot)
  {
-     worldOwner->resource.oversamplingFactor = sealedSnapshot->buildInput.oversamplingFactor;
+     // 最小値1を保証（0はAuto未解決の不正値）。validateResources() 拒否回避。
+     worldOwner->resource.oversamplingFactor =
+         std::max(1, sealedSnapshot->buildInput.oversamplingFactor);
```

### 6.2 🟡 SECONDARY Fix: 過剰リビルドの抑制（推奨）

1. **NoiseShaper type変更のデバウンス**: `setNoiseShaperType` 呼び出し時に即座にリビルドせず、SR変更と同じ `prepareToPlay` サイクル内で処理
2. **EQ snapshot command の集約**: 400ms のデバウンス中に複数の snapshot が到着した場合、最後の1つだけを処理（現在は4つすべてが `MERGED` されるが、gen=3〜7の5世代ビルドを引き起こす）

### 6.3 🟠 TERTIARY Fix: EQProcessor バッファ再利用（推奨）

DSPCore再作成時にEQProcessorのバッファを再利用する仕組みを導入。現在は毎回 `capacity=0` から4MB再割当が発生している。

---

## 7. 検証方法

### 7.1 PRIMARY Fix の検証

1. **ビルド**: `Release Build (cmd env retry)` タスクでビルド
2. **ログ確認**: 起動時に `osFactor=0` が出力されないことを確認

   ```powershell
   Select-String -Path 'ConvoPeq.log' -Pattern 'osFactor=0'
   # → ヒット0件であること
   ```

3. **publish FAILED 確認**: `eventCode=6002` と `publish FAILED gen=1` が出力されないことを確認

   ```powershell
   Select-String -Path 'ConvoPeq.log' -Pattern 'eventCode=6002|publish FAILED'
   # → ヒット0件であること
   ```

4. **音飛び確認**: アプリ起動直後（48kHz→192kHz切替時）に音飛びが発生しないことを聴覚確認

### 7.2 SECONDARY Fix の検証

1. 起動後の `task queued generation=` 出力数が3以下であること（現在は5世代）
2. `REBUILD_MERGED` の `latencyMs` が400ms未満に短縮されていること

### 7.3 自動テスト

- `work21 EpochDomain CI Gate` タスクでCI検証
- `Strict Atomic Dot-Call Scan` タスクでオーディオスレッド安全性検証

---

## 8. コーディング規約遵守チェック

| 規約 | 状態 | 備考 |
|---|---|---|
| JUCE 8.0.12 API確認 | ✅ | `juce::Atomic`, `std::max` は標準API |
| `/JUCE`, `/r8brain-free-src` 編集禁止 | ✅ | 対象外 |
| SEH使用禁止 | ✅ | 使用なし |
| Audio Thread内ブロッキング禁止 | ✅ | 修正箇所は全てメッセージスレッド（rebuild thread / prepareToPlay） |
| MKL使用箇所のメモリ規約 | ✅ | 対象外（MKL不使用箇所） |
| 64byteアライメント | ✅ | 対象外 |
| デノーマル対策 | ✅ | 対象外 |

---

## 9. フォローアップ

### 9.1 即時対応（本改修）

- PRIMARY Fix（地点A-D）の適用で、gen=1 publish 失敗を完全に防止

### 9.2 中期対応（推奨）

- SECONDARY Fix: 過剰リビルドカスケードの抑制
- TERTIARY Fix: EQProcessor バッファ再利用

### 9.3 長期対応（検討）

- Auto モード解決ロジックの単一化（`DSPCoreLifecycle.cpp` と `RebuildDispatch.cpp` の重複排除）
- `validateResources()` に `osFactor=0` を Auto として許容するかどうかの仕様検討

---

## 付録A: 調査に使用したツールと対象ファイル

### 使用ツール

1. **grep_search / Select-String**: ログ解析、パターン抽出
2. **serena (find_symbol)**: シンボル定義の追跡（`trySubmit`, `publishWorld`, `validateResources`）
3. **codegraph (query_codebase/find_callees)**: 呼び出し依存関係の解析
4. **semble (search)**: セマンティック検索（`PublicationAdmission`, `rebuildThreadLoop`）
5. **AiDex (query)**: 識別子検索（`eventCode`, `EVENT_VALIDATION`）
6. **read_file**: ソースコード詳細読み取り

### 調査対象ファイル

- `src/audioengine/AudioEngine.RebuildDispatch.cpp`（`captureBuildParameterSnapshot`, `requestRebuild`, `finalizeRuntimeBuildSnapshot`）
- `src/audioengine/AudioEngine.h`（`runPublicationPrecheckNonRt`, `manualOversamplingFactor`）
- `src/audioengine/AudioEngine.Commit.cpp`（`rejectWithEvidence`, `lastCommittedRuntimeGeneration_`）
- `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp`（`prepareToPlay`, placeholder publish）
- `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp`（Auto解決ロジック: 行72-107）
- `src/audioengine/RuntimeBuilder.cpp`（`buildRuntimePublishWorld`, sealedSnapshot パス: 行308-313）
- `src/audioengine/RuntimePublicationOrchestrator.cpp/.h`（`trySubmit`, `submitPublishRequest`）
- `src/audioengine/PublicationExecutor.cpp/.h`（`publish`, `PublishResult`）
- `src/audioengine/PublicationAdmission.h`（`evaluate`, `Decision`）
- `src/audioengine/RuntimePublicationValidator.cpp/.h`（`validatePublication`, `validateResources`）
- `src/audioengine/RuntimeHealthMonitor.cpp/.h`（`emitValidationEvent`, eventCode 定義）
- `src/core/RuntimePublicationCoordinator.h`（`publishWorld`, `validatePublicationNonRt`）

## 付録B: ログ全文タイムライン（主要イベント抜粋）

```
行15:  [DIAG] requestRebuild(sr,bs): task queued generation=1 SR=48000.00
行60:  [DIAG] rebuildThreadLoop: generation=1 build=86.4ms rebuildIR=0.0ms
行61:  [CONV_STATUS] generation=1 sr=48000.0 osFactor=0 processingRate=384000.0  ← ★Auto未解決
行62:  [HEALTH] eventCode=6002 severity=1 value=0                                ← ★InvalidResources
行63:  [DIAG] trySubmit: executor_.publish FAILED gen=1 result=1                 ← ★Rejected
行64:  [DIAG] prepareToPlay: enter spb=1024 sr=192000.00                         ← SR変更開始
行68:  [DIAG] prepareToPlay: rebuild request generation reset to 0
行136: [DIAG] requestRebuild(sr,bs): task queued generation=2 SR=192000.00
行138: [AudioEngine] setNoiseShaperType: newType=2 wasAdaptive=1                 ← NoiseShaper変更
行145: [REBUILD_TELEMETRY] REBUILD_MERGED intentId=7 latencyMs=400.000
行158: [DIAG] requestRebuild(sr,bs): task queued generation=3 SR=192000.00
行229: [CONV_STATUS] generation=3 sr=192000.0 osFactor=2 processingRate=384000.0 ← 解決済み
行230: [DIAG] trySubmit: executor_.publish SUCCEEDED gen=3                       ← 成功
行272: task queued generation=4
行300: task queued generation=5
行314: task queued generation=6
行392: task queued generation=7
行453: [CONV_STATUS] generation=7 irLoaded=1 irLen=192000 osFactor=2
行454: [DIAG] trySubmit: executor_.publish SUCCEEDED gen=7                       ← 最終成功
```

---

**結論**: 音飛びの根本原因は `manualOversamplingFactor = 0`（Auto）の未解決値が `world.resource.oversamplingFactor = 0` として publish 検証に渡り、`validateResources()` の `os < 1` チェックで必ず拒否されることにあります。PRIMARY Fix（4地点の修正）により、この問題を完全に解決できます。
