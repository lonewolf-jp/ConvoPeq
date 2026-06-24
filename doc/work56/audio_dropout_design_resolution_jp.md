# 音飛び原因分析 設計確定書

**ステータス**: 確定（v7.0: Phase2 deferred理由コメント追加、全ProjectionフィールドのIntent/Deferred明確化）
**作成日**: 2026-06-24
**対象**: ConvoPeq 音声アプリ（JUCE V8.0.12 / スタンドアローン）
**調査ログ**: `doc/work56/ConvoPeq.log`
**分析書**: `doc/work56/audio_dropout_root_cause_analysis_jp.md`
**調査ツール**: read_file, Select-String, Serena MCP, CodeGraph MCP, AiDex MCP, semble, ccc

---

## 目次

1. [調査サマリー](#1-調査サマリー)
2. [検証済み因果連鎖](#2-検証済み因果連鎖)
3. [確定した事実と否定された仮説](#3-確定した事実と否定された仮説)
4. [アーキテクチャ上の設計判断](#4-アーキテクチャ上の設計判断)
5. [buildRuntimePublishWorld 実引数検証](#5-buildruntimepublishworld-実引数検証)
6. [確定修正（3Phase計画）](#6-確定修正3phase計画)
7. [検証方法](#7-検証方法)
8. [フォローアップ項目](#8-フォローアップ項目)

---

## 1. 調査サマリー

### 1.1 背景

ConvoPeq 起動直後（48kHz→192kHz切り替え時）に音飛びが発生するという報告を受け、`doc/work56/audio_dropout_root_cause_analysis_jp.md` が作成された。本設計書は、同分析書の主張を7種類のツールを用いてコードベースから検証し、確定した事実と修正方針を記録する。

### 1.2 分析書の評価

| 主張 | 検証結果 | 確定度 |
| --- | --- | --- |
| `osFactor=0` → `validateResources()` が拒否 | ✅ **正しい** | 100% |
| sealedSnapshot 経路で未解決の 0 が伝播 | ✅ **正しい** | 100% |
| gen=1 publish 失敗の因果連鎖 | ✅ **正しい** | 100% |
| eventCode=6002 との対応 | ✅ **正しい** | 100% |
| 音飛びの唯一の根本原因と断定 | ⚠️ **過剰断定** | — |
| Fix A/B (capture/request での Auto 解決) | ❌ **設計違反** | — |
| Fix C/D (std::max(1, ...)) | ⚠️ **対症療法** | — |
| SECONDARY/TERTIARY の実害 | ⚠️ **実害は限定的** | — |

### 1.3 未確定事項の調査結果（全7項目確定）

| # | 未確定事項 | 結論 | エビデンス |
| --- | --- | --- | --- |
| 1 | placeholder DSP の実挙動 | **無音ではない。dry信号（EQ処理＋conv無効）を出力する** | `ConvolverProcessor::processBypassWithLatencyCompensation` が入力を遅延コピー |
| 2 | audio callback の DSP 解決パス | `runtimeWorld->engine.current` から取得。publish 失敗時は bootstrap world 維持 | `AudioBlock.cpp:143`, `AudioEngine.h:2578-2584` |
| 3 | publish 失敗時の runtime 継続 | new DSP は retire、RuntimeStore 不変、旧 world 継続 | `RuntimePublicationOrchestrator.cpp:150-155` |
| 4 | `captureRuntimeBuildSnapshot` 設計 | PR-2 コメントと実装が乖離。DSP 解決値が入っていない | `RebuildDispatch.cpp:93` vs `RuntimeBuildTypes.h:45-49` |
| 5 | CrossfadeAuthority 投影値 | sealedSnapshot 経路で 0 が投影される。fallback 経路は正しい | `CrossfadeAuthority.cpp:23`, `RuntimeBuilder.cpp:231/239` |
| 6 | リビルドカスケード実害 | 全 rebuild は非 RT スレッド。Audio Thread に直接影響せず | ログ gen=3-7, rebuildThreadLoop |
| 7 | EQ_PREPARE capacity=0 実害 | 毎回割当だが rebuild thread 内の1回限り。Audio Thread 非影響 | `EQProcessor.Core.cpp:670` |

---

## 2. 検証済み因果連鎖

### 2.1 PRIMARY: oversamplingFactor=0 の伝播と拒否

```text
manualOversamplingFactor = 0 (Auto)
  |
  +---> captureBuildParameterSnapshot()  [RebuildDispatch.cpp:35]
  |     snapshot.oversamplingFactor = consumeAtomic(manualOversamplingFactor)
  |     -> 0 をそのまま取得
  |
  +---> requestRebuild()  [RebuildDispatch.cpp:~560]
  |     task.buildInput.oversamplingFactor = paramSnapshot.oversamplingFactor
  |     -> 0 を task に設定
  |
  +---> captureRuntimeBuildSnapshot()  [RebuildDispatch.cpp:88-93] *PR-2投影未完了
  |     irLoaded/irFinalized/structuralHash = UI ConvolverProcessor から投影（正しい）
  |     oversamplingFactor = buildInput.oversamplingFactor（=0）← **DSP解決値が未投影**
  |     sampleRate = buildInput.sampleRate（実質的にDSP値と一致）
  |     baseLatencySamples = 未設定（=0）
  |
  +---> finalizeRuntimeBuildSnapshot()  [RebuildDispatch.cpp:101]
  |     buildInput.oversamplingFactor = std::max(0, ...)
  |     -> std::max(0, 0) = 0 を許容
  |
  +---> buildRuntimePublishWorld(current=新規DSP, next=旧DSP, sealedSnapshot)
  |     [RuntimeBuilder.cpp:216]  resource.oversamplingFactor = current->oversamplingFactor
  |     -> ここでは正しい DSP解決値（48kHz->8x）が設定される
  |     [RuntimeBuilder.cpp:311]  resource.oversamplingFactor = sealedSnapshot->buildInput.oversamplingFactor
  |     -> 正しい値を 0 で上書き！ <- *Phase1 の修正対象
  |
  +---> validateResources()  [RuntimePublicationValidator.cpp:119]
  |     os(0) < 1 -> return false -> InvalidResources
  |
  +---> publishWorld()  [RuntimePublicationCoordinator.h:~100]
  |     bridge_.validatePublicationNonRt() が false -> Rejected
  |
  +---> trySubmit()  [RuntimePublicationOrchestrator.cpp:~150]
        [DIAG] trySubmit: executor_.publish FAILED gen=1 result=1
        -> new DSP を retire、RuntimeStore 不変
```

### 2.2 ログとの対応

| ログ行 | 出力 | 意味 |
| --- | --- | --- |
| 15 | `requestRebuild(sr,bs): task queued generation=1 SR=48000.00` | gen=1 48kHz ビルド開始 |
| 60 | `rebuildThreadLoop: generation=1 build=86.4ms rebuildIR=0.0ms` | ビルド完了（86.4ms） |
| 61 | `generation=1 sr=48000.0 osFactor=0 processingRate=384000.0` | buildInput.osFactor=0（未解決）, DSPCore 内部は 8x 解決済み |
| 62 | `eventCode=6002 severity=1 value=0` | InvalidResources |
| 63 | `trySubmit: executor_.publish FAILED gen=1 result=1` | Rejected |
| 64 | `prepareToPlay: enter spb=1024 sr=192000.00` | SR 変更（48k->192k） |

---

## 3. 確定した事実と否定された仮説

### 3.1 確定した事実

#### 事実1: placeholder DSP は無音ではない

```cpp
// ConvolverProcessor.Runtime.cpp:217-224
// DSPCore::process() で state.convBypassed=false の場合、
// 常に convolverRt().process() が呼ばれる
if (!state.convBypassed)
    convolverRt().process(processBlock);  // 呼ばれる

// ConvolverProcessor.Runtime.cpp:98-173
// processBypassWithLatencyCompensation: 入力をリングバッファで遅延し出力にコピー
// 無音ではなく、dry信号（convolution未適用）が出力される
void ConvolverProcessor::processBypassWithLatencyCompensation(
    juce::dsp::AudioBlock<double>& block, const StereoConvolver& conv) noexcept
{
    // 1. 入力を delayBuffer に保存
    std::memcpy(buf + writePos, src, ...);
    // 2. 遅延した信号を出力へ戻す
    juce::FloatVectorOperations::copy(dstBuf, srcBuf + readPos, ...);
    // -> 無音ではない！ドライ信号が出力される
}
```

#### 事実2: publish 失敗時の動作

```cpp
// RuntimePublicationOrchestrator.cpp:148-155
auto result = executor_.publish(engine_, std::move(worldOwner));
if (result != PublishResult::Success) {
    // publish 失敗: activate/crossfade/retire は一切行わない
    if (!req.newDSP.isNull())
        lifetime_.retire(newDSPResolved);  // new DSP を破棄
    // RuntimeStore は更新されない -> 旧 world 継続
    return PublicationAdmission::Decision::RejectedShutdown;
}
```

```cpp
// AudioBlock.cpp:143 / AudioEngine.h:2578-2584
// audio callback は RuntimeWorld.engine.current から DSP を取得
DSPCore* dsp = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandleRef);
```

#### 事実3: RuntimeBuildSnapshot 二層構造の設計意図

```cpp
// RuntimeBuildTypes.h:35-58
struct RuntimeBuildSnapshot
{
    int generation = 0;
    BuildInput buildInput {};            // ユーザー意図 (oversamplingFactor=0=Auto)
    // ...
    // [PR-2] DSP semantic projection snapshot values
    int oversamplingFactor = 1;          // 設計上は DSPCore 解決値 (default: 1)
    // ...
};
```

```cpp
// RebuildDispatch.cpp:93 - 現在の実装（バグ）
snapshot.oversamplingFactor = buildInput.oversamplingFactor;  // = 0
// PR-2 コメント: "populated from DSPCore when snapshot is created"
//                -> DSPCore の値が入るべきだが、実際は buildInput のコピー
```

### 3.2 PR-2 Projection 導入の未完了（設計上の未完成領域）

`captureRuntimeBuildSnapshot()` の PR-2 投影フィールドの実態と消費状況:

| フィールド | 現在の値 | 由来 | 正しいか | 生産コードでの消費先 | 影響 |
| --- | --- | --- | --- | --- | --- |
| `irLoaded` | `uiConvolverProcessor.isIRLoaded()` | UI Convolver | おおよそ正しい | CrossfadeAuthority (oldHasIR/newHasIR) | クロスフェード要否判断 |
| `irFinalized` | `uiConvolverProcessor.isIRFinalized()` | UI Convolver | おおよそ正しい | **消費なし** | **なし** |
| `structuralHash` | `uiConvolverProcessor.getStructuralHash()` | UI Convolver | おおよそ正しい | CrossfadeAuthority (IR変更検出) | クロスフェード要否判断 |
| `oversamplingFactor` | **`buildInput.oversamplingFactor` (=0)** | **BuildInputコピー** | **❌ DSPCore値未投影** | CrossfadeAuthority (OS変化検出) | Phase2で修復が必要 |
| `sampleRate` | `buildInput.sampleRate` | BuildInputコピー | 実質的にDSP値と一致 | **消費なし** | **なし** |
| `baseLatencySamples` | **0（未設定）** | デフォルト | **未実装** | **消費なし** | **なし** |

PR-2 コメントは "populated from DSPCore when snapshot is created" としているが、実際は:

- `irLoaded`/`irFinalized`/`structuralHash`: UI ConvolverProcessor から投影（DSPCoreではない）
- `oversamplingFactor`/`sampleRate`: BuildInput の単純コピー（DSPCore未解決）
- `baseLatencySamples`: 未実装

**重要**: 生産コードで実際に消費されている projection フィールドは `irLoaded`、`structuralHash`、`oversamplingFactor` の3つのみ。`irFinalized`、`sampleRate`、`baseLatencySamples` は現在どこからも消費されていない。そのため、これらが未完成でも現時点の動作に影響はない。

したがって「RuntimeBuilder.cpp:311 だけが悪い」のではなく、「**PR-2 の projection 導入が未完成**」と表現する方が正確。ただし実害が生じているのは `oversamplingFactor` のみで、残りは将来の拡張領域。

### 3.3 否定された仮説（分析書の誤り）

| 分析書の主張 | 否定理由 |
| --- | --- |
| placeholder DSP は無音 | `processBypassWithLatencyCompensation` が入力を遅延コピーすることを確認。無音ではない |
| 音飛びの唯一の根本原因 | SR変更＋リビルドカスケード＋conv有効/無効切り替えの複合要因。単一原因と断定できない |
| Fix A: capture で Auto 解決 | `BuildInput.oversamplingFactor=0` は正当な Auto 意図。解決すると設計原則に反する |
| Fix B: request で再解決 | 解決ロジックが 3 箇所に分散し保守性低下 |

---

## 4. アーキテクチャ上の設計判断

### 4.1 Snapshot Authority 原則

ConvoPeq の ISR Runtime は **Snapshot = Authority** を基本原則とする。

```text
BuildInput.oversamplingFactor          = 0    ユーザー意図 (Auto)
  変更しない（これが Authority）

RuntimeBuildSnapshot.oversamplingFactor   DSP 投影値（設計上の役割）
  現在は buildInput のコピーだが、本来は DSPCore 解決値が入るべき
  将来の改善項目（Phase2）

RuntimePublishWorld.resource.oversamplingFactor  Resource 検証値
  validateResources() で 1-16 の 2のべき乗が要求される
  DSPCore 解決値を現在の current から使用
```

### 4.2 二層構造の維持

```text
+-----------------------------------------------------------+
| BuildInput.oversamplingFactor                              |
| = 0 (Auto = ユーザー意図)                                  |
| 変更禁止！これが Snapshot Authority                         |
+-----------------------------------------------------------+
| RuntimeBuildSnapshot.oversamplingFactor                    |
| = DSPCore 解決値（設計上の役割）                            |
| 現在は未実装（buildInput のコピー）                          |
| 将来: rebuildThreadLoop で更新する（Phase2）                |
+-----------------------------------------------------------+
| RuntimePublishWorld.resource.oversamplingFactor            |
| = validateResources() の検証対象                            |
| DSPCore 解決値を現在の current から使用                     |
+-----------------------------------------------------------+
```

### 4.3 `resource` フィールドと `dspProjection` フィールドの消費分析

`resource` フィールドの消費先:

| フィールド | 検証ルール | その他の消費先 |
| --- | --- | --- |
| `resource.oversamplingFactor` | `os < 1 \|\| os > 16 \|\| (os & (os-1)) != 0` → reject | `semanticHash.resourceHash`, `semanticHash.payloadHash`（fallback時のみ） |
| `resource.ditherBitDepth` | `dd != 0 && dd != 16 && dd != 24 && dd != 32` → reject | `semanticHash.resourceHash`, `semanticHash.payloadHash` |
| `resource.noiseShaperType` | `ns < 0 \|\| ns > 3` → reject | `semanticHash.resourceHash`, `semanticHash.payloadHash` |

**重要**: `ditherBitDepth=0` は検証ルールで明示的に許容されている（`if (dd != 0 ...)` で弾かれない）。`noiseShaperType=0` も同様。**`oversamplingFactor=0` だけが検証に失敗する唯一の値。**

`dspProjection` フィールドの消費先（生産コードのみ）:

| フィールド | CrossfadeAuthority | Orchestrator |
| --- | --- | --- |
| `irLoaded` | `oldHasIR` / `newHasIR` 判定 | `cfDecision.newHasIR` 設定 |
| `structuralHash` | IR変更検出 (`oh != nh`) | — |
| `oversamplingFactor` | OS変化検出 (`old != new`) | — |
| `irFinalized` | — | — |
| `sampleRate` | — | — |
| `baseLatencySamples` | — | — |

CrossfadeAuthority では3フィールドのみ使用。残り3フィールドは生産コードで未消費。

### 4.4 `semanticHash` の一貫性

```cpp
// sealedSnapshot 存在時: payloadHash = buildInput 全体のハッシュ（user intent）
//   → oversamplingFactor=0 を含む（正しい。Auto モードの表現）
// sealedSnapshot 不在時: payloadHash = resource 値から計算（bootstrap パスのみ）
// resourceHash: resource の全フィールドから計算（actual resource state）

// Phase1 適用後:
//   payloadHash（sealedSnapshot時）= hashBuildInput(buildInput) → 0 を含む（変化なし、正しい）
//   resourceHash = resource.oversamplingFactor を含む → 8（DSP解決値、正しい）
```

両ハッシュは異なる意味論を持つため、Phase1 前後で不整合は生じない。

### 4.5 なぜ BuildInput を変更してはいけないか

1. **重複検出との整合性**: `isRuntimeBuildSnapshotSealedAndCompatible()` は `buildInput.oversamplingFactor` を比較する。0（Auto）と 8（解決値）は異なる値として扱われ、同一設定なのに異なる snapshot と判定される。
2. **トレーサビリティ**: ログ出力 `[CONV_STATUS] osFactor=...` は `buildInput.oversamplingFactor` を使用しており、これが 0 であることで Auto モードだったことが確認できる。解決後はこの情報が失われる。
3. **UI 状態保存との整合性**: `AudioEngine.Parameters.cpp` の `setOversamplingFactor()` は 0=Auto を保存。

---

## 5. buildRuntimePublishWorld 実引数検証

### 5.1 全呼出箇所6箇所の追跡結果

`buildRuntimePublishWorld(DSPCore* current, DSPCore* next, ...)` の全呼出箇所:

| # | 呼出元 | current | next | sealedSnapshot | バグ影響 |
| --- | --- | --- | --- | --- | --- |
| 1 | `trySubmit()` Orchestrator.cpp:72 | **newDSPResolved (新規DSP)** | oldDSP (旧Runtime) | あり | **影響あり** |
| 2 | `prepareToPlay()` PrepareToPlay.cpp:145 | currentForPublish (既存active) | fadingForPublish | なし | 影響なし |
| 3 | `prepareToPlay()` PrepareToPlay.cpp:258 | getActiveRuntimeDSP() (placeholder) | nullptr | なし | 影響なし |
| 4 | `releaseResources()` ReleaseResources.cpp:146 | nullptr | nullptr | なし | 影響なし |
| 5 | `timerCallback()` Timer.cpp:471 | currentAfterFade | nullptr | なし | 影響なし |
| 6 | `publishIdleWorldOnly()` Transition.cpp:24 | currentAfterFade | nullptr | なし | 影響なし |

### 5.2 trySubmit() の実引数確認（最重要）

```cpp
// RuntimePublicationOrchestrator.cpp:53-73
auto* newDSPResolved = engine_.resolveDSPHandle(req.newDSP);   // 新規DSP
auto oldHandle = engine_.dspHandleRuntime_.getActiveRuntimeDSPHandle();
auto* oldDSP = (!oldHandle.isNull())
    ? engine_.resolveDSPHandle(oldHandle)
    : nullptr;

auto worldOwner = worldBuilder.buildRuntimePublishWorld(
    newDSPResolved,     // <- current = 新規DSP（Auto解決済み, 48kHz->8x）
    oldDSP,             // <- next    = 旧DSP
    convo::TransitionPolicy::HardReset, 0.0, false,
    &req.sealedSnapshot);
```

**結論: `current` は新規DSPであり旧Runtimeではない。**

- `current` = newDSPResolved = rebuildThreadLoop で構築された DSPCore
- `DSPCore::prepare()` で Auto 解決済み（48kHz->8x, 192kHz->4x）
- `current->oversamplingFactor` = 8（48kHz時）
- 直前行~216: `resource.oversamplingFactor = static_cast<int>(current->oversamplingFactor)` = 8（正しい）

---

## 6. 確定修正（3Phase計画）

### 6.1 Phase 1: 緊急修正（必須・即時）

**目的**: gen=1 publish 失敗を即座に防止
**変更**: `RuntimeBuilder.cpp:311` の上書き行を削除
**リスク**: 極小（変更1行、sealedSnapshotが渡される唯一の経路のみ影響）

```cpp
        // [100%] sealedSnapshot が存在する場合 ...
        if (useSealedSnapshot)
        {
            // Phase1: oversamplingFactor は直前行~216 の current->oversamplingFactor
            // （新規DSP解決値, 48kHz->8x）を維持。snapshot の 0 (=Auto) で上書きしない。
            // worldOwner->resource.oversamplingFactor = sealedSnapshot->buildInput.oversamplingFactor;
            // ~~上記行を削除~~
            worldOwner->resource.ditherBitDepth = sealedSnapshot->buildInput.ditherBitDepth;
            worldOwner->resource.noiseShaperType = sealedSnapshot->buildInput.noiseShaperType;
            worldOwner->timing.sampleRateHz = sealedSnapshot->buildInput.sampleRate;
        }
```

**根拠**:

- 全6呼出箇所中、sealedSnapshot が渡されるのは経路 #1（`trySubmit`）のみ
- 経路 #1 の `current` = newDSPResolved -> `current->oversamplingFactor` = Auto解決値（正しい）
- 他の5経路は sealedSnapshot なし -> 311行実行されず、影響なし
- `ditherBitDepth`/`noiseShaperType`/`sampleRate` は buildInput 値で正しいので投影継続

### 6.2 Phase 2: Snapshot 投影値の修復（推奨）

**目的**: `RuntimeBuildSnapshot.oversamplingFactor` を真の DSP 解決値に更新

```cpp
// AudioEngine.RebuildDispatch.cpp: rebuildThreadLoop 内、build() 成功後
if (buildResult.runtime != nullptr)
{
    dspGuard.ptr = buildResult.runtime;
    auto* newDSP = buildResult.runtime;

    // Phase2: DSP構築完了後に RuntimeBuildSnapshot の投影値を更新
    task.runtimeBuildSnapshot.oversamplingFactor = static_cast<int>(newDSP->oversamplingFactor);
```

**効果**:

- `RuntimeBuilder.cpp:231` の `dspProjection.oversamplingFactor = sealedSnapshot->oversamplingFactor` が正しい値に
- `CrossfadeAuthority.cpp:23` の比較が正確に（bootstrap world の projection=8 vs gen=1 world の projection=8 -> 変化なし = クロスフェード不要）
- gen=1成功時は oversampling 変化がないため、本来不要なクロスフェードが発生しなくなる

### 6.3 Phase 3: RuntimeBuilder の Snapshot Authority 統一（保留）

**目的**: `resource.oversamplingFactor` を Snapshot Authority から投影

```cpp
        if (useSealedSnapshot)
        {
            // Phase3（保留）: sealedSnapshot->oversamplingFactor が真の DSP 解決値になった場合の将来案
            // 現時点では current->oversamplingFactor（一次情報源）を維持
            // worldOwner->resource.oversamplingFactor = sealedSnapshot->oversamplingFactor;
```

**判断根拠（保留）**:

- RuntimeWorld 構築時点では `current->oversamplingFactor` の方が**一次情報源**として自然
- `DSPCore -> RuntimeBuildSnapshot -> RuntimeWorld` という二段投影は依存関係を増やす
- ISR Runtime の原則: **Authority は1箇所、Projection は複数可**
- 現状の `resource.oversamplingFactor = current->oversamplingFactor` で十分正確
- Phase2 で dspProjection が正しくなれば、CrossfadeAuthority と Admission は十分正確になる
- Phase3 は RuntimeBuildSnapshot の全 projection 項目が成熟し、DSP Reality を完全に保持することを確認してから判断

### 6.4 Phase 実施判断基準

| Phase | 必須度 | リスク | 効果 |
| --- | --- | --- | --- |
| Phase 1 | **必須（即時）** | 極小（1行削除） | gen=1 publish 失敗を防止 |
| Phase 2 | **推奨** | 小（再ビルドが必要） | DSP投影値の正確性向上、CrossfadeAuthority の誤検出防止 |
| Phase 3 | **保留** | — | 全 projection 項目の成熟を待つ |

### 6.5 変更しないこと

- `captureBuildParameterSnapshot()` -- 変更なし（Auto=0 をそのまま取得）
- `captureRuntimeBuildSnapshot()` -- 実装は現状維持。PR-2 projection は未完成だが、Phase2 で rebuildThreadLoop が `oversamplingFactor` を上書きすることで必要な最小限の修復は達成される。`irLoaded`/`irFinalized`/`structuralHash` は UI ConvolverProcessor からの投影で実用上問題ない。`baseLatencySamples` は未実装だが現時点では影響なし。
- `finalizeRuntimeBuildSnapshot()` -- 変更なし（`std::max(0, ...)` は許容）
- `AudioEngine.h` の `manualOversamplingFactor{0}` -- 変更なし（0=Auto は正規状態）
- 分析書の Fix A/B は**不採用**
- 分析書の Fix C/D は**Phase1 で不要になる**

### 6.6 最終推奨サマリー

| Phase | 対応 | ステータス |
| --- | --- | --- |
| **Phase1** | `RuntimeBuilder.cpp:311` の上書き行を削除 | **必須・即時実施** |
| **Phase2** | `rebuildThreadLoop` で `runtimeBuildSnapshot.oversamplingFactor = newDSP->oversamplingFactor` を設定 | **推奨** |
| **Phase3** | `resource.oversamplingFactor = sealedSnapshot->oversamplingFactor` への移行 | **保留**（全 projection の成熟を確認後判断） |

### 6.7 実装詳細（コード変更の正確な位置と内容）

#### Phase1: RuntimeBuilder.cpp:311

**ファイル**: `src/audioengine/RuntimeBuilder.cpp`
**行**: 308-319

```cpp
        // [100%] sealedSnapshot が存在する場合、resource/timing フィールドも atomic ではなく snapshot から設定
        if (useSealedSnapshot)
        {
            // ★ Phase1: この行を削除（直前行~216 の current->oversamplingFactor 維持のため）
            //   sealedSnapshot->buildInput.oversamplingFactor は Auto 時 0 であり、
            //   validateResources() の os<1 チェックで Rejected される原因。
            worldOwner->resource.oversamplingFactor = sealedSnapshot->buildInput.oversamplingFactor;
            // ~~上記行を削除~~
            worldOwner->resource.ditherBitDepth = sealedSnapshot->buildInput.ditherBitDepth;
            worldOwner->resource.noiseShaperType = sealedSnapshot->buildInput.noiseShaperType;
            worldOwner->timing.sampleRateHz = sealedSnapshot->buildInput.sampleRate;
        }
        else
        {
            worldOwner->resource.oversamplingFactor = (current != nullptr)
                ? static_cast<int>(current->oversamplingFactor) : 1;
```

**影響範囲**: この変更の影響を受けるのは経路 #1（`trySubmit()`）のみ。理由:

- sealedSnapshot が渡される全6経路中、経路 #1 以外では `sealedSnapshot` が `nullptr`
- `useSealedSnapshot` は `sealedSnapshot != nullptr` と同値 → 他の5経路ではこのブロック自体が実行されない

#### Phase2: AudioEngine.RebuildDispatch.cpp（rebuildThreadLoop 内）

**ファイル**: `src/audioengine/AudioEngine.RebuildDispatch.cpp`
**挿入位置**: CONV_STATUS ログブロック直後、commit 直前（~857-862行）

```cpp
            // 6. Commit on Message Thread
            // Release ownership from guard, pass to commitNewDSP

            // ★ Phase2: DSP構築完了後の RuntimeBuildSnapshot 投影値更新
            //   PR-2 設計: snapshot 投影値は DSPCore の実値を持つべき
            //   buildInput.oversamplingFactor (=0 for Auto) を DSP 解決値で上書きする。
            //   これにより RuntimeBuilder::buildRuntimePublishWorld() 内で
            //   dspProjection.oversamplingFactor = sealedSnapshot->oversamplingFactor
            //   が正しい値（例: 48kHz->8x）を持つようになる。
            //
            //   NOTE: 現在は oversamplingFactor のみ修正。
            //   他の PR-2 投影フィールド（irLoaded/irFinalized/structuralHash/
            //   sampleRate/baseLatencySamples）は以下の理由により意図的に deferred:
            //   - irLoaded/irFinalized/structuralHash: UI ConvolverProcessor 由来だが
            //     CrossfadeAuthority の実用上問題ない
            //   - sampleRate: buildInput の値が実質的に DSP 値と一致
            //   - baseLatencySamples: 生産コードで未消費
            //   これらは将来的な Phase2 拡張で必要に応じて追加すること。
            task.runtimeBuildSnapshot.oversamplingFactor = static_cast<int>(newDSP->oversamplingFactor);

            DSPCore* dspToCommit = dspGuard.ptr;
            dspGuard.ptr = nullptr;
            enqueuePublicationIntentForRuntimeCommit(dspToCommit, task.generation, task.runtimeBuildSnapshot);
```

**なぜこの位置か**:

- `newDSP->oversamplingFactor` は `DSPCore::prepare()` で Auto 解決済み（48kHz→8x, 192kHz→4x）
- CONV_STATUS ログより後なので、ログ出力には影響しない（ログは `buildInput` を表示）
- commit 直前なので、すべての事前チェック（warmup, obsolete）を通過した確定タスクのみ更新
- `enqueuePublicationIntentForRuntimeCommit` は `const RuntimeBuildSnapshot&` を受け取り、内部で `PublishRequest::sealedSnapshot` に**値コピー**するため、更新値が伝播

**所有権と伝播の確認**:

```cpp
// PublicationAdmission.h:17-21
struct PublishRequest {
    DSPHandle newDSP;
    int generation = 0;
    RuntimeBuildSnapshot sealedSnapshot;  // ★ 値コピー！参照ではない
};

// AudioEngine.Commit.cpp:665-669
convo::isr::PublicationAdmission::PublishRequest req;
req.newDSP = handle;
req.generation = generation;
req.sealedSnapshot = sealedSnapshot;  // ★ task.runtimeBuildSnapshot の値がコピーされる
```

`PublishRequest::sealedSnapshot` は値 (`RuntimeBuildSnapshot`) なので、`task.runtimeBuildSnapshot` を事前に更新しておけば、そのコピーに更新値が含まれる。`submitPublishRequest` → `trySubmit` は同期的に実行され、値コピーはその時点のスナップショットを取得する。

---

## 7. 検証方法

### 7.1 ビルド

```powershell
# "Release Build (cmd env retry)" タスクでビルド
```

### 7.2 ログ検証

```powershell
# eventCode=6002 (InvalidResources) が出力されないこと
Select-String -Path 'ConvoPeq.log' -Pattern 'eventCode=6002'
# -> ヒット0件（これが最重要検証項目）

# publish FAILED が出力されないこと
Select-String -Path 'ConvoPeq.log' -Pattern 'publish FAILED'
# -> ヒット0件（これも最重要）

# osFactor=0 は Auto モードの正規表現として残る可能性がある
# BuildInput=Authority を維持するため、osFactor=0 の有無は検証条件にしない
# osFactor=0 自体は正常値である
```

### 7.3 動作検証

```powershell
# CI検証
# "work21 EpochDomain CI Gate" タスク
# "Strict Atomic Dot-Call Scan" タスク

# 聴覚確認: アプリ起動直後（48kHz->192kHz切替時）の音飛び有無
```

### 7.4 Phase2 追加検証

```powershell
# dspProjection.oversamplingFactor が正しい DSP 解決値を持つこと
# 検証方法: CrossfadeAuthority の OS変化検出が正確になること
#
# 注意: CONV_STATUS の osFactor は buildInput 値を表示するため、
# Phase2 後も osFactor=0 のまま変化しない。
# osFactor は「ユーザー意図（Auto=0）」を表しており、これは正しい動作。
# processingRate は DSPCore 実値（newDSP->sampleRate * oversamplingFactor）
# を表示するため、こちらで解決値の確認が可能。
#   例: sr=48000.0 osFactor=0 processingRate=384000.0
#       （osFactor=0 は Auto, processingRate=384000 は 8x解決済み）
```

---

## 8. フォローアップ項目

### 8.1 即時対応

- Phase1: `RuntimeBuilder.cpp:311` の 1 行削除（緊急修正）

### 8.2 推奨（Phase2/3）

- Phase2: `rebuildThreadLoop()` 内で snapshot 投影値を DSP 解決値に更新
- Phase3: `RuntimeBuilder.cpp:311` を `sealedSnapshot->oversamplingFactor` に変更

### 8.3 調査推奨

- **placeholder DSP 使用期間の定量化**: gen=1 failure から gen=3 success までの時間を計測
- **音飛びの主原因の切り分け**: placeholder DSP（dry信号） vs SR変更カスケードの寄与度評価
- **SR変更時の gen 増加抑制**: prepareToPlay 後の不要な rebuild 連鎖の抑制

---

## 付録A: コード変更diff（Phase1）

```diff
--- a/src/audioengine/RuntimeBuilder.cpp
+++ b/src/audioengine/RuntimeBuilder.cpp
@@ -308,7 +308,8 @@ convo::aligned_unique_ptr<RuntimePublishWorld>
         // [100%] sealedSnapshot が存在する場合、resource/timing フィールドも atomic ではなく snapshot から設定
         if (useSealedSnapshot)
         {
-            worldOwner->resource.oversamplingFactor = sealedSnapshot->buildInput.oversamplingFactor;
+            // oversamplingFactor は直前行~216 の current->oversamplingFactor（新規DSP解決値）を維持
+            // (sealedSnapshot->buildInput.oversamplingFactor=0 は Auto意図であり resource 検証に不適切)
             worldOwner->resource.ditherBitDepth = sealedSnapshot->buildInput.ditherBitDepth;
             worldOwner->resource.noiseShaperType = sealedSnapshot->buildInput.noiseShaperType;
             worldOwner->timing.sampleRateHz = sealedSnapshot->buildInput.sampleRate;
```

## 付録B: 調査ログマップ

| 調査ファイル | 調査内容 | 使用ツール |
| --- | --- | --- |
| `RuntimeBuildTypes.h` | RuntimeBuildSnapshot 構造定義 | read_file |
| `AudioEngine.RebuildDispatch.cpp` | captureBuildParameterSnapshot, captureRuntimeBuildSnapshot, requestRebuild | read_file, Serena, CodeGraph, AiDex, semble |
| `RuntimeBuilder.cpp` | buildRuntimePublishWorld, sealedSnapshot 投影 | read_file, Select-String |
| `RuntimePublicationValidator.cpp` | validateResources 検証ロジック | read_file |
| `RuntimePublicationCoordinator.h` | publishWorld, Rejected 返却 | read_file |
| `RuntimePublicationOrchestrator.cpp` | trySubmit, publish 失敗時の retire | read_file, CodeGraph |
| `AudioEngine.Processing.AudioBlock.cpp` | audio callback DSP 解決パス | read_file, Select-String |
| `AudioEngine.Processing.DSPCoreFloat.cpp` | DSPCore::process, convBypass 判定 | read_file |
| `ConvolverProcessor.Runtime.cpp` | processBypassWithLatencyCompensation | read_file, ccc |
| `AudioEngine.Processing.PrepareToPlay.cpp` | placeholder DSP 作成 | read_file |
| `AudioEngine.h` | captureAudioThreadParameterSnapshot, resolveActiveRuntimeDSPFromRuntimeWorldOnly | read_file, Select-String |
| `CrossfadeAuthority.cpp` | dspProjection.oversamplingFactor 比較 | read_file |
| `EQProcessor.Core.cpp` | EQ_PREPARE capacity=0 | read_file |
| `ConvoPeq.log` | ログタイムライン検証 | read_file |
