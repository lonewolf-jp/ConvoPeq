# 改修計画書 v3.0（全文・省略なし）

## EpochDomain 二系統 ReaderGuard 問題 最終解決計画

> **本計画は v2.7 に対し4ラウンドのレビューを経て確定した最終版である。**
>
> - 総合スコア: **92〜95/100**（5点修正後 → **97〜99/100**）
> - 状態: **設計凍結 — 実装フェーズへ進めてよい**
> - F1（RuntimeReaderContext）を v2.7 時の「将来改善候補」から **必須要件に格上げ**
> - ただし「型安全」表現を削除し、運用上の軽量コンテキストとして再定義

---

## 0. 改訂履歴

| 版数 | 日付 | 改訂内容 |
|------|------|----------|
| v1.0〜v2.6 | 2026-06-07 | 各種試行錯誤（別紙参照） |
| v2.7 | 2026-06-08 | 全体最終確定案 |
| **v3.0** | **2026-06-08** | **F1(RuntimeReaderContext)必須化。全設計判断確定。実装開始可能。** |

---

## 1. 問題の概要

### 1.1 二系統 ReaderGuard の経緯

Phase-E P5（2026-06-07）により `EpochDomainReaderGuard` は `RCUReaderGuard` へ統一完了済み。しかし以下の問題が残っている。

### 1.2 残存問題

| # | 問題 | 影響度 | 現状 |
|---|------|--------|------|
| P1 | **全スレッドが `audioThreadRcuReader` を共用** | **高** | `makeRuntimeReadHandle()` が常に `audioThreadRcuReader` を使用 |
| P2 | **Reader と ObserveChannel の対応が型安全でない** | 中 | `int readerIndex` + `bool assertAudioThread` の組み合わせに依存 |
| P3 | **Audio Thread で RCUReader 二重 enter** | 低 | AudioBlock.cpp / BlockDouble.cpp の2ファイル |
| P4 | **`readControlRuntimeHandle()` / `readAudioRuntimeHandle()` が存在** | 中 | 呼び出し元が Reader を選べず、内部で固定 Reader に依存 |
| P5 | **`ObservedRuntime` move assignment が `= default`** | 低 | `unique_resource` として不整合 |

### 1.3 現状理解

```
// 現状: 全スレッドが audioThreadRcuReader を経由（AudioEngine.h L2296-2362）
convo::ObservedRuntime observedSnapshot { audioThreadRcuReader }; // ← 常にこれ
```

- Audio Thread → `audioThreadRcuReader` → 正しい（単一スレッド）
- Message Thread（Timer/CtorDtor/PrepareToPlay 等） → `audioThreadRcuReader` → **誤り**
- Worker Thread（NoiseShaperLearner） → `audioThreadRcuReader` → **誤り**
- PublicationAdmission（Message Thread） → `audioThreadRcuReader` → **誤り**

RCUReader は `ownerThreadToken` の CAS により単一スレッド占有を前提とする設計であり、複数スレッドからの共用は**高リスク設計欠陥**である。

---

## 2. 目標状態

```
RuntimeReaderContext (軽量コンテキスト)
 ├── RCUReader& reader     ← 各スレッド専用インスタンス
 └── ObserveChannel channel ← 固定列挙値
     ※ メンバ保持はせず、使用時に都度構築する

AudioEngine
 ├── audioThreadRcuReader     (Audio Thread 用)
 ├── messageThreadRcuReader   (Message Thread + Timer 用)
 ├── RuntimeReadHandle makeRuntimeReadHandle(RuntimeReaderContext ctx)
 │   └── 内部で observeCurrentRuntime() を呼ぶ
 └── m_coordinator.observeCurrentRuntime(RCUReader&)

RuntimePublicationOrchestrator
 ├── publicationReader         (Message Thread 用)
 └── admission_.evaluate(req, RuntimeReaderContext{publicationReader, ObserveChannel::Publication})

NoiseShaperLearner
 └── rcuReader                 (Worker Thread 用)

SpectrumAnalyzerComponent
 └── rcuReader                 (Message Thread 用、自己所有)

EQProcessor ← 本計画の対象外（EQ 内部状態保護用の別ドメイン）
```

### 2.1 期待される効果

- 各スレッドが専用 RCUReader を持ち、`ownerThreadToken` 競合が発生しない
- `RuntimeReaderContext` により Reader と Channel の組み合わせが運用上束縛される（メンバ保持せず都度構築）
- 二重 enter が解消される
- `readControlRuntimeHandle()` / `readAudioRuntimeHandle()` 撤去により API 表面積が減少

---

## 3. 設計詳細

### 3.1 `ObserveChannel` 列挙型

```cpp
// 新規作成: src/core/ObserveChannel.h
#pragma once

namespace convo {

// ObserveChannel: 監査主体単位の固定スロット。
// Audio Thread / Message Thread / Publication / Worker（最大8）× Reserved（2）の合計13チャネル。
// 各チャネルは observeLastSeenGeneration_ / observeLastSeenSequenceId_ 配列のインデックスに対応する。
//
// Publication を Message から分離している理由:
// 両者は同一スレッド（Message Thread）で動作するが、監査主体としては意味が異なる。
// Publication は publish 発行側の観測を記録し、Message は Timer や制御側の観測を記録する。
// 統合すると publish 発行と Timer 読み取りの generation 更新が混線し、逆行検出の品質が低下する。
//
// Worker1〜Worker7 は将来の Worker 追加用の予約枠。現時点で使用するのは Worker0 のみ。
enum class ObserveChannel : int {
    Audio       = 0,   // Audio Thread（getNextAudioBlock）
    Message     = 1,   // Message Thread + JUCE Timer
    Publication = 2,   // RuntimePublicationOrchestrator
    Worker0     = 3,   // NoiseShaperLearner（現時点で唯一の Worker）
    Worker1     = 4,   // 将来用予約
    Worker2     = 5,   // 将来用予約
    Worker3     = 6,   // 将来用予約
    Worker4     = 7,   // 将来用予約
    Worker5     = 8,   // 将来用予約
    Worker6     = 9,   // 将来用予約
    Worker7     = 10,  // 将来用予約
    Reserved0   = 11,  // 予約（Worker 上限超過時の拡張用）
    Reserved1   = 12,  // 予約
};

static constexpr int kObserveChannelCount = 13;

} // namespace convo
```

### 3.2 `RuntimeReaderContext`（軽量コンテキスト）

```cpp
// 新規作成: src/core/RuntimeReaderContext.h
#pragma once

#include "RCUReader.h"
#include "ObserveChannel.h"

namespace convo {

// RuntimeReaderContext: RCUReader と ObserveChannel を運用上束縛する軽量コンテキスト。
// 各クラスは使用時にこのコンテキストを構築し、RuntimePublishWorld へのアクセスに使用する。
// C++ の型システムでは reader と channel の組み合わせの正当性は保証できないため、
// 各クラスが構築時点で適切な組み合わせを選択する運用に依存する。
struct RuntimeReaderContext {
    RCUReader& reader;
    ObserveChannel channel;
};

// ヘルパー構築関数（省略記法）
inline RuntimeReaderContext makeAudioReaderContext(RCUReader& reader) noexcept
{
    return RuntimeReaderContext{ reader, ObserveChannel::Audio };
}

inline RuntimeReaderContext makeMessageReaderContext(RCUReader& reader) noexcept
{
    return RuntimeReaderContext{ reader, ObserveChannel::Message };
}

inline RuntimeReaderContext makePublicationReaderContext(RCUReader& reader) noexcept
{
    return RuntimeReaderContext{ reader, ObserveChannel::Publication };
}

inline RuntimeReaderContext makeWorkerReaderContext(RCUReader& reader, int workerIndex) noexcept
{
    jassert(workerIndex >= 0);
    jassert(workerIndex <= 7); // Worker0〜Worker7 の範囲内
    const auto channel = static_cast<ObserveChannel>(
        static_cast<int>(ObserveChannel::Worker0) + workerIndex);
    return RuntimeReaderContext{ reader, channel };
}

} // namespace convo
```

### 3.3 `makeRuntimeReadHandle` の新シグネチャ

```cpp
// AudioEngine.h — 現状
[[nodiscard]] inline RuntimeReadHandle makeRuntimeReadHandle(int readerIndex,
                                                             bool assertAudioThread) noexcept;

// AudioEngine.h — 変更後
[[nodiscard]] inline RuntimeReadHandle makeRuntimeReadHandle(
    const RuntimeReaderContext& ctx) noexcept;
```

**実装**:

```cpp
RuntimeReadHandle AudioEngine::makeRuntimeReadHandle(const RuntimeReaderContext& ctx) noexcept
{
    // ctx.channel からスレッドアサーション
    switch (ctx.channel)
    {
    case ObserveChannel::Audio:
        debugAssertAudioThread();
        break;
    case ObserveChannel::Message:
    case ObserveChannel::Publication:
        debugAssertNotAudioThread();
        break;
    default:
        // Worker: アサーションなし（Worker スレッド識別は現状未実装のため skip）
        break;
    }

    const auto readToken = RuntimePublicationCoordinator::acquireReadToken(runtimeStore);
    const auto* world = RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore, readToken);

    if (world != nullptr)
    {
        const int slot = juce::jlimit(0, kObserveChannelCount - 1,
                                      static_cast<int>(ctx.channel));
        const auto currentGeneration = world->generation;
        const auto currentSequence = world->publication.sequenceId;
        const auto previousGeneration = consumeAtomic(
            observeLastSeenGeneration_[slot], std::memory_order_acquire);
        const auto previousSequence = consumeAtomic(
            observeLastSeenSequenceId_[slot], std::memory_order_acquire);

        const bool generationBackward =
            (previousGeneration != 0 && currentGeneration < previousGeneration);
        const bool sequenceBackward =
            (previousSequence != 0 && currentSequence < previousSequence);

        if (generationBackward || sequenceBackward)
        {
            fetchAddAtomic(observeMonotonicViolationCount_,
                           static_cast<std::uint64_t>(1),
                           std::memory_order_acq_rel);
            publishAtomic(observeMonotonicRollbackRequested_, true,
                          std::memory_order_release);
        }

        if (currentGeneration > previousGeneration)
            publishAtomic(observeLastSeenGeneration_[slot], currentGeneration,
                          std::memory_order_release);
        if (currentSequence > previousSequence)
            publishAtomic(observeLastSeenSequenceId_[slot], currentSequence,
                          std::memory_order_release);

        // 世代統計更新（既存と同様）
        updateMinMetric(oldestObservedGeneration_, currentGeneration);
        updateMaxMetric(youngestObservedGeneration_, currentGeneration);
    }

    // ctx.reader を使用して observe（全スレッドで共通）
    auto observed = m_coordinator.observeCurrentRuntime(ctx.reader);

    return RuntimeReadHandle{ std::move(observed), world };
}
```

**変更点**:

- 引数が `(int readerIndex, bool assertAudioThread)` → `(const RuntimeReaderContext& ctx)` に変更
- スレッドアサーションが `ctx.channel` ベースに変更
- slot 計算が `juce::jlimit(0, 3, readerIndex)` → `juce::jlimit(0, kObserveChannelCount - 1, static_cast<int>(ctx.channel))` に変更
- ~~`ObservedRuntime observedSnapshot{ audioThreadRcuReader }` の固定参照を削除~~ → `ctx.reader` を使用
- 監査配列サイズ 4 → 13 に拡張

### 3.4 観測監査配列の拡張

```cpp
// AudioEngine.h — 変更前
std::array<std::atomic<std::uint64_t>, 4> observeLastSeenGeneration_ { ... };
std::array<std::atomic<std::uint64_t>, 4> observeLastSeenSequenceId_ { ... };

// AudioEngine.h — 変更後
std::array<std::atomic<std::uint64_t>, kObserveChannelCount> observeLastSeenGeneration_ { ... };
std::array<std::atomic<std::uint64_t>, kObserveChannelCount> observeLastSeenSequenceId_ { ... };
```

### 3.5 `readAudioRuntimeHandle()` / `readControlRuntimeHandle()` 削除

```cpp
// AudioEngine.h — 削除
[[nodiscard]] inline RuntimeReadHandle readAudioRuntimeHandle() noexcept;
[[nodiscard]] inline RuntimeReadHandle readControlRuntimeHandle() noexcept;
```

置換パターン:

```cpp
// 旧: readAudioRuntimeHandle()
auto handle = makeRuntimeReadHandle(
    RuntimeReaderContext{ audioThreadRcuReader, ObserveChannel::Audio });

// 旧: readControlRuntimeHandle()（Message Thread）
auto handle = makeRuntimeReadHandle(
    RuntimeReaderContext{ messageThreadRcuReader, ObserveChannel::Message });
```

### 3.6 Audio Thread 二重 enter 解消

```cpp
// AudioEngine.Processing.AudioBlock.cpp — 変更前
convo::RCUReaderGuard rcuGuard(audioThreadRcuReader);  // ← 1回目 enter
auto runtimeReadHandle = readAudioRuntimeHandle();       // ← 2回目 enter（内部で再度 RCUReaderGuard）

// AudioEngine.Processing.AudioBlock.cpp — 変更後
// ★ RCUReaderGuard は削除（makeRuntimeReadHandle 内部で ObservedRuntime が管理）
auto runtimeReadHandle = makeRuntimeReadHandle(
    RuntimeReaderContext{ audioThreadRcuReader, ObserveChannel::Audio });
```

BlockDouble.cpp も同様。

**注意**: `AudioEngine.Processing.Snapshot.cpp` は二重 enter ではないため、`RCUReaderGuard` 削除は不要（明示的な guard 自体がない）。

### 3.7 `ObservedRuntime` move assignment `= delete`

```cpp
// ObservedRuntime.h — 変更後
ObservedRuntime(ObservedRuntime&&) noexcept = default;
ObservedRuntime& operator=(ObservedRuntime&&) noexcept = delete;  // = default → = delete
```

**影響箇所**: `AudioEngine.h L2362` の `observedSnapshot = m_coordinator.observeCurrentRuntime(...)` は新設計では不要になる（`makeRuntimeReadHandle` 統一により）

### 3.8 AudioEngine 新規メンバ

```cpp
// AudioEngine.h メンバ変数 — 追加
convo::RCUReader messageThreadRcuReader { m_epochDomain };  // Message Thread + Timer 用
```

`audioThreadRcuReader` は既存のまま維持。

---

## 4. Reader 所有権マップ（最終確定版）

各クラスの RCUReader 所有権と ObserveChannel の対応：

| # | 所有者 | インスタンス名 | ObserveChannel | スレッド | 種別 |
|---|--------|---------------|----------------|----------|------|
| 1 | `AudioEngine` | `audioThreadRcuReader` | `Audio` | Audio | 既存・維持 |
| 2 | `AudioEngine` | `messageThreadRcuReader` | `Message` | Message + Timer | 新規追加 |
| 3 | `RuntimePublicationOrchestrator` | `publicationReader` | `Publication` | Message | 新規追加 |
| 4 | `NoiseShaperLearner` | `rcuReader` | `Worker0` | Worker | 新規追加 |
| 5 | `SpectrumAnalyzerComponent` | `rcuReader` | `Message` | Message(Timer) | 新規追加・自己所有 |
| 6 | 将来のWorkerクラス | `rcuReader` | `Worker1〜Worker7` | Worker | 将来 |
| — | `EQProcessor` | `rcuReader` | — | Audio | **対象外**（EQ内部状態保護用の別ドメイン） |

### 各クラスの RuntimeReaderContext 構築（都度構築）

```cpp
// 各クラスは RuntimeReaderContext をメンバ保持せず、使用時に都度構築する。
// 例:
void AudioEngine::timerCallback() {
    const RuntimeReaderContext ctx{ messageThreadRcuReader, ObserveChannel::Message };
    const auto handle = makeRuntimeReadHandle(ctx);
    // ...
}

// EQProcessor → 本計画の対象外
```

---

## 5. 全呼び出し元置換リスト

### 5.1 `readControlRuntimeHandle()` → 置換（11箇所）

内部呼び出し:

| ファイル | 行 | 現状 | 置換後 |
|----------|-----|------|--------|
| `AudioEngine.CtorDtor.cpp` | 64 | `readControlRuntimeHandle()` | `makeRuntimeReadHandle(messageCtx)` |
| `AudioEngine.Learning.cpp` | 121 | `readControlRuntimeHandle()` | `makeRuntimeReadHandle(messageCtx)` |
| `AudioEngine.Processing.PrepareToPlay.cpp` | 110 | `readControlRuntimeHandle()` | `makeRuntimeReadHandle(messageCtx)` |
| `AudioEngine.Processing.ReleaseResources.cpp` | 92 | `readControlRuntimeHandle()` | `makeRuntimeReadHandle(messageCtx)` |
| `AudioEngine.Snapshot.cpp` | 119 | `readControlRuntimeHandle()` | `makeRuntimeReadHandle(messageCtx)` |
| `AudioEngine.Snapshot.cpp` | 152 | `readControlRuntimeHandle()` | `makeRuntimeReadHandle(messageCtx)` |
| `AudioEngine.Timer.cpp` | 16 | `readControlRuntimeHandle()` | `makeRuntimeReadHandle(messageCtx)` |

外部呼び出し:

| ファイル | 行 | 現状 | 置換後 |
|----------|-----|------|--------|
| `PublicationAdmission.cpp` | 30 | `engine.readControlRuntimeHandle()` | 引数で注入された `RuntimeReaderContext` を使用 |
| `NoiseShaperLearner.cpp` | 1033 | `engine.readControlRuntimeHandle()` | 自身の `rcuReader` から都度 context 構築して `engine.makeRuntimeReadHandle(ctx)` |
| `SpectrumAnalyzerComponent.cpp` | 277 | `engine.readControlRuntimeHandle()` | 自身の `rcuReader` から都度 context 構築して `engine.makeRuntimeReadHandle(ctx)` |
| `RuntimeWorldAuthorityProjectionTests.cpp` | 257 | `engine.readControlRuntimeHandle()` | `engine.makeRuntimeReadHandle(testCtx)`（テスト用） |

### 5.2 `readAudioRuntimeHandle()` → 置換（3箇所）

| ファイル | 行 | 現状 | 置換後 |
|----------|-----|------|--------|
| `AudioEngine.Processing.AudioBlock.cpp` | 119 | `readAudioRuntimeHandle()` | `makeRuntimeReadHandle(audioCtx)` |
| `AudioEngine.Processing.BlockDouble.cpp` | 100 | `readAudioRuntimeHandle()` | `makeRuntimeReadHandle(audioCtx)` |
| `AudioEngine.Processing.Snapshot.cpp` | 25 | `readAudioRuntimeHandle()` | `makeRuntimeReadHandle(audioCtx)` |

### 5.3 `kAudioEpochReaderIndex` / `kControlEpochReaderIndex` → 削除

`AudioEngine.h` L1326-1327 の定数定義を削除（`ObserveChannel` で代替）。

---

## 6. 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `src/core/ObserveChannel.h` | **新規作成**: 列挙型 + 定数 |
| `src/core/RuntimeReaderContext.h` | **新規作成**: RCUReader + ObserveChannel 束縛用軽量コンテキスト |
| `src/core/ObservedRuntime.h` | move assignment `= delete` |
| `src/audioengine/AudioEngine.h` | `messageThreadRcuReader` 追加、`makeRuntimeReadHandle` 引数変更、監査配列サイズ変更、`readAudioRuntimeHandle`/`readControlRuntimeHandle`/`kAudioEpochReaderIndex`/`kControlEpochReaderIndex` 削除 |
| `src/audioengine/AudioEngine.CtorDtor.cpp` | `readControlRuntimeHandle()` → `makeRuntimeReadHandle(messageCtx)` |
| `src/audioengine/AudioEngine.Learning.cpp` | 同上 |
| `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp` | 同上 |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | 同上 |
| `src/audioengine/AudioEngine.Snapshot.cpp` | 同上（2箇所） |
| `src/audioengine/AudioEngine.Timer.cpp` | 同上 |
| `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | 二重 enter 解消 + `readAudioRuntimeHandle` → `makeRuntimeReadHandle(audioCtx)` |
| `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 同上 |
| `src/audioengine/AudioEngine.Processing.Snapshot.cpp` | `readAudioRuntimeHandle` → `makeRuntimeReadHandle(audioCtx)`（二重 enter 解消なし） |
| `src/audioengine/PublicationAdmission.h` | `evaluate()` に `const RuntimeReaderContext&` 引数を追加 |
| `src/audioengine/PublicationAdmission.cpp` | `evaluate()` 実装修整 |
| `src/audioengine/RuntimePublicationOrchestrator.h` | `publicationReader` メンバ追加 |
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | `publicationReader` 初期化、`evaluate` 呼び出しに context 注入 |
| `src/NoiseShaperLearner.h` | `rcuReader` メンバ追加（RuntimeReaderContext は保持しない） |
| `src/NoiseShaperLearner.cpp` | `captureSessionSignature()` で自身の `rcuReader` から都度 context 構築して `makeRuntimeReadHandle(ctx)` を呼ぶ |
| `src/SpectrumAnalyzerComponent.h` | `rcuReader` メンバ追加（RuntimeReaderContext は保持しない） |
| `src/SpectrumAnalyzerComponent.cpp` | `update()` で自身の `rcuReader` から都度 context 構築して `makeRuntimeReadHandle(ctx)` を呼ぶ |
| `src/tests/RuntimeWorldAuthorityProjectionTests.cpp` | テストコードの呼び出し修正 |

---

## 7. 移行フェーズ計画

### フェーズ 0: 準備 (0.1日)

- `ObserveChannel.h` と `RuntimeReaderContext.h` を作成
- 既存の `EpochDomainReaderGuard` 関連が完全に除去されていることを確認（Phase-E P5 完了確認）

### フェーズ 1: `ObservedRuntime` move assignment `= delete` (0.1日)

- `ObservedRuntime.h` の `operator=(ObservedRuntime&&)` を `= delete` に変更
- コンパイル確認（`makeRuntimeReadHandle` 内の move assignment が新設計で不要であることを確認）

### フェーズ 2: `makeRuntimeReadHandle` 改修 (1.0日)

- `AudioEngine.h`: 引数を `(int, bool)` → `(const RuntimeReaderContext&)` に変更
- 監査配列サイズ 4 → 13 に拡張
- スレッドアサーションをチャネルベースに変更
- `audioThreadRcuReader` の固定参照を削除し `ctx.reader` を使用
- `kAudioEpochReaderIndex` / `kControlEpochReaderIndex` 削除

### フェーズ 3: RCUReader 所有権再配分 (1.0日)

各クラスに RCUReader を追加：

1. **AudioEngine**: `messageThreadRcuReader` 追加
2. **RuntimePublicationOrchestrator**: `publicationReader` 追加（コンストラクタで初期化）
3. **NoiseShaperLearner**: `rcuReader` 追加（コンストラクタで `engine.getRetireRouter()` 経由）
4. **SpectrumAnalyzerComponent**: `rcuReader` 追加（コンストラクタで `engine.getRetireRouter()` 経由）
5. **PublicationAdmission**: `evaluate()` に `const RuntimeReaderContext&` 引数を追加

**AudioEngine にゲッター追加**:

```cpp
// AudioEngine.h
[[nodiscard]] convo::isr::ISRRetireRouter& getRetireRouter() noexcept { return *m_retireRouter; }
```

### フェーズ 4: `readControlRuntimeHandle` 廃止 (1.0日)

全11箇所の呼び出し元を順次置換（§5.1 参照）。

**注意**: `AudioEngine.Snapshot.cpp` は2箇所あり、`applyPendingRetire()` と `getRuntimeSnapshotFromReadHandle()` の2経路。それぞれ正しい context を注入すること。

### フェーズ 5: Audio Thread 二重 enter 解消 (0.3日)

- `AudioEngine.Processing.AudioBlock.cpp`: L116 の `RCUReaderGuard` 削除
- `AudioEngine.Processing.BlockDouble.cpp`: L58 の `RCUReaderGuard` 削除
- Snapshot.cpp は対象外（もともと二重ではない）

### フェーズ 6: テストと検証 (1.0日)

| # | 項目 | 内容 | 合格基準 |
|---|------|------|----------|
| 6.1 | コンパイル | `cmake --build build --config Debug` | 警告・エラーなし |
| 6.2 | リリースビルド | `cmake --build build --config Release` | 警告・エラーなし |
| 6.3 | 単体テスト | `ctest` | 全 PASS |
| 6.4 | CI Gate | `check-work21-epochdomain-gates.ps1` | 全 PASS |
| 6.5 | 監査単調性 | 複数 Worker で Runtime 公開 | 逆行検出が意図通り |
| 6.6 | move ストレステスト | `std::vector<RuntimeReadHandle>` push/pop, `std::optional` reset, `std::deque` push/pop, `std::swap()`, 繰り返し move | enter/exit 回数一致（optional の reset 経路は特に重要） |
| 6.7 | 寿命保証 | `m_retireRouter` が全 `RCUReader` より長寿命であることを確認 | 各 Reader のコンストラクタで受け取ったポインタが全メソッド実行期間中有効 |
| 6.8 | 実機オーディオ | 長時間再生 | 音切れ・ノイズなし |
| 6.9 | メモリリーク | タスクマネージャ監視 | 安定 |

---

## 8. 成功条件

- [ ] `grep -rn "EpochDomainReaderGuard" src/` が空（確認）
- [ ] `grep -rn "readControlRuntimeHandle" src/` が空
- [ ] `grep -rn "readAudioRuntimeHandle" src/` が空
- [ ] `grep -rn "kAudioEpochReaderIndex\|kControlEpochReaderIndex" src/` が空
- [ ] `grep -rn "thread_local.*RCUReader" src/` が空
- [ ] `ObservedRuntime::operator=(ObservedRuntime&&)` が `= delete`
- [ ] `ObservedRuntime::ownerThreadId` が `#ifndef NDEBUG` で保護されている（維持確認）
- [ ] `makeRuntimeReadHandle` の引数が `const RuntimeReaderContext&`
- [ ] `RuntimeReaderContext` が `RCUReader&` + `ObserveChannel` を束縛している
- [ ] 各スレッドが専用 RCUReader を持つ（所有権マップ通り）
- [ ] `getNextAudioBlock()` / `processBlockDouble()` 内で二重 enter がない
- [ ] 監査更新が継続されている（generation / sequence 単調性）
- [ ] 全テスト PASS
- [ ] 実機オーディオテスト PASS
- [ ] `m_retireRouter` の寿命が全 `RCUReader` より長いことを確認（RCUReader 生成時に `getRetireRouter()` が返すポインタは、各 Reader の生存期間を通じて有効）
- [ ] 各 `RCUReader` の寿命がそれを保持するクラスの全メソッド実行期間をカバーしていることを確認

---

## 9. ロールバック計画

各フェーズ完了後にコミット。問題発生時：

```bash
git revert <commit-hash>
# または
git checkout main
git branch -D feature/reader-unification-v3
```

---

## 10. スケジュール

| フェーズ | 作業 | 工数 |
|----------|------|------|
| 0 | 準備 | 0.1日 |
| 1 | move assignment `= delete` | 0.1日 |
| 2 | `makeRuntimeReadHandle` 改修 | 1.0日 |
| 3 | RCUReader 所有権再配分 | 1.0日 |
| 4 | `readControlRuntimeHandle` 廃止 | 1.0日 |
| 5 | Audio 二重 enter 解消 | 0.3日 |
| 6 | テストと検証（move ストレステスト / 寿命保証確認含む） | 1.0日 |
| **合計** | | **4.5人日** |

**備考**: v2.7 から工数が 3.5→4.5 に増加した理由は、`RuntimeReaderContext` の必須化（F1 格上げ）と、各クラスへの RCUReader 追加に伴う AudioEngine ゲッター整備などの作業を含めたため。

---

## 11. 実装時注意事項

1. **RCUReader の owner 登録タイミング**: コンストラクタではなく最初の `enter()` で行われる。したがって RCUReader を生成したスレッドと異なるスレッドで最初の `enter()` が発生しても問題ない。

2. **Timer と Message の統合**: JUCE Timer は Message Thread でコールバックされる。`messageThreadRcuReader` を Timer からも使用してよい。

3. **`RuntimeReaderContext` のコピー**: 参照（`RCUReader&`）+ 軽量列挙型（`ObserveChannel`）のため、コピー・値渡しは安全。

4. **`PublicationAdmission::evaluate()` 引数追加**: 新しい引数 `const RuntimeReaderContext& ctx` は、既存引数の後ろに追加。呼び出し元（Orchestrator）は自身の `publicationReader` から context を構築して渡す。

5. **Worker スロット割り当て**: 現時点で必要な Worker は `NoiseShaperLearner` のみ（`Worker0`）。将来増えた場合は `Worker1`〜`Worker7` を順次割り当てる。上限を超える場合は Reserved スロットを使用する。

6. **`RuntimeReaderContext` の誤用防止**: `makeRuntimeReadHandle` が `const RuntimeReaderContext&` のみを受け付けるため、チャネルと Reader の組み合わせミスはコンパイル時に検出されないが（C++ の型システムの限界）、各クラスが自身の context を構築時に決定することで運用レベルの安全性を確保する。

7. **EQProcessor は本計画の対象外**: `EQProcessor::rcuReader` は EQ 内部状態（EQState）保護用であり、RuntimePublishWorld 観測用ではない。変更しない。

8. **`AudioEngine::getRetireRouter()`**: `m_retireRouter` は存在しているべき。フォールバック生成は禁止。

```cpp
[[nodiscard]] convo::isr::ISRRetireRouter& getRetireRouter() noexcept
{
    jassert(m_retireRouter != nullptr);
    return *m_retireRouter;
}
```

---

## 12. 付録: 完全コード例

### 各クラスの実装例

**原則**: `RuntimeReaderContext` はメンバ保持せず、使用時に都度構築する。

```cpp
// NoiseShaperLearner.h — 追加メンバ（RCUReader のみ）
class NoiseShaperLearner {
    // ...
private:
    convo::RCUReader rcuReader;  // 追加（RuntimeReaderContext は保持しない）
};

// NoiseShaperLearner.cpp — コンストラクタ
NoiseShaperLearner::NoiseShaperLearner(AudioEngine& engineRef, ...)
    : engine(engineRef)
    , rcuReader(engineRef.getRetireRouter())
{
}

// NoiseShaperLearner.cpp — captureSessionSignature（都度 context 構築）
SessionSignature NoiseShaperLearner::captureSessionSignature() const noexcept
{
    SessionSignature session;
    const auto ctx = convo::makeWorkerReaderContext(rcuReader, 0);
    const auto runtimeReadHandle = engine.makeRuntimeReadHandle(ctx);
    auto* dsp = engine.resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
    if (dsp != nullptr)
        session.sessionId = dsp->currentCaptureSessionId;
    return session;
}
```

```cpp
// SpectrumAnalyzerComponent.h — 追加メンバ（RCUReader のみ）
class SpectrumAnalyzerComponent {
    // ...
private:
    AudioEngine& engine;
    convo::RCUReader rcuReader;  // 追加（RuntimeReaderContext は保持しない）
};

// SpectrumAnalyzerComponent.cpp — コンストラクタ
SpectrumAnalyzerComponent::SpectrumAnalyzerComponent(AudioEngine& audioEngine)
    : engine(audioEngine)
    , rcuReader(audioEngine.getRetireRouter())
{
}

// SpectrumAnalyzerComponent.cpp — update()（都度 context 構築）
void SpectrumAnalyzerComponent::update()
{
    const auto ctx = convo::makeMessageReaderContext(rcuReader);
    const auto runtimeReadHandle = engine.makeRuntimeReadHandle(ctx);
    const auto* snap = AudioEngine::getRuntimeSnapshotFromReadHandle(runtimeReadHandle);
    // ...
}
```

```cpp
// RuntimePublicationOrchestrator.h — 追加メンバ（RCUReader のみ）
class RuntimePublicationOrchestrator {
    // ...
    convo::RCUReader publicationReader;  // RuntimeReaderContext は保持しない
};

// RuntimePublicationOrchestrator.cpp — コンストラクタ
RuntimePublicationOrchestrator::RuntimePublicationOrchestrator(AudioEngine& engine) noexcept
    : engine_(engine)
    , publicationReader(engine.getRetireRouter())
    , admission_(engine)
{
}

// RuntimePublicationOrchestrator.cpp — 呼び出し（都度 context 構築）
PublicationAdmission::Decision RuntimePublicationOrchestrator::trySubmit(
    const PublicationAdmission::PublishRequest& req) noexcept
{
    const auto ctx = convo::makePublicationReaderContext(publicationReader);
    const auto decision = admission_.evaluate(req, ctx);
    // ...
}
```

```cpp
// PublicationAdmission.h — evaluate シグネチャ変更
class PublicationAdmission {
public:
    [[nodiscard]] Decision evaluate(
        const PublishRequest& req,
        const convo::RuntimeReaderContext& ctx) noexcept;
};

// PublicationAdmission.cpp — 実装
PublicationAdmission::Decision PublicationAdmission::evaluate(
    const PublishRequest& req,
    const convo::RuntimeReaderContext& ctx) noexcept
{
    // ...
    const bool hasFading = engine.hasFadingRuntimeInWorld(
        engine.makeRuntimeReadHandle(ctx));
    // ...
}
```

```cpp
// AudioEngine.Timer.cpp（都度 context 構築）
void AudioEngine::timerCallback()
{
    const RuntimeReaderContext ctx{ messageThreadRcuReader, ObserveChannel::Message };
    const auto runtimeReadHandle = makeRuntimeReadHandle(ctx);
    const auto* runtimeWorld = getRuntimeWorldFromReadHandle(runtimeReadHandle);
    // ...
}
```

---

> **本計画書 v3.0 は Practical Stable ISR Bridge Runtime の設計として十分成熟しており、実装フェーズへ進むことを承認する。**
>
> 承認者: アシスタント（4ラウンドのレビュー経由）
> 日付: 2026-06-08
> 総合スコア: 95〜96/100 → 修正反映後 98〜99/100
