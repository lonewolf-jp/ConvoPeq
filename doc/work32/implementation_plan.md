# Practical Stable ISR Bridge Runtime — 改修計画書

**作成日**: 2026-06-12
**基礎文書**: `doc/work32/notfinished8_re-evaluation.md`
**検証レポート**: `doc/work32/notfinished8_validation_report.md`
**対象プロジェクト**: ConvoPeq (JUCE 8.0.12 / C++20 / Windows x64)

---

## 目次

1. [改修一覧](#1-改修一覧)
2. [S-1: Epoch の意味の純化](#2-s-1-epoch-の意味の純化)
3. [S-2: Health→Control Pipeline 完成](#3-s-2-healthcontrol-pipeline-完成)
4. [A-1: Health→Recovery 実装（自己防衛）](#4-a-1-healthrecovery-実装自己防衛)
5. [B-1: Reader Ownership Telemetry 完成](#5-b-1-reader-ownership-telemetry-完成)
6. [C-1: RuntimeDrainAudit ↔ WorldLifecycleAudit 連携](#6-c-1-runtimedraindata--worldlifecycleaudit-連携)
7. [C-2: BuildError 分類拡充](#7-c-2-builderror-分類拡充)
8. [C-3: EpochDomain Reader Slot 可観測性強化](#8-c-3-epochdomain-reader-slot-可観測性強化)
9. [実装順序と依存関係](#9-実装順序と依存関係)
10. [リスクと注意点](#10-リスクと注意点)

---

## 1. 改修一覧

| ID | 課題 | 優先度 | 性質 | Tier | ファイル改修数 | 推定工数 |
|---|---|---|---|---|---|---|
| S-2 | Health→Control Pipeline | S | 構造（Governance） | Tier 1 | 7ファイル | 大（4経路へのHealthState導入） |
| S-1 | Epoch の意味の純化 | S | 意味論（整合性） | Tier 1 | 2ファイル | 小（1行修正＋影響調査） |
| A-1 | Health→Recovery（自己防衛） | A | 自己防衛 | Tier 2 | 3ファイル | 中（コールバック拡張） |
| B-1 | Reader Ownership Telemetry | B+ | 可観測性 | Tier 3 | 2ファイル | 小（フィールド実装） |
| C-1 | DrainAudit ↔ WorldLifecycleAudit | C | 診断品質 | Tier 4 | 2ファイル | 小（フィールド追加） |
| C-2 | BuildError 分類拡充 | C | 診断品質 | Tier 4 | 2ファイル | 小（enum追加） |
| C-3 | Reader Slot 可観測性強化 | C | 可観測性 | Tier 4 | 1ファイル | 小（フィールド追加） |

> **注意**: S-1 と S-2 は同じ Tier 1 だが、実運用リスクの観点では S-2 が最大の未達。
> S-1 は最小工数（1行修正）のため S-2 と並行して着手可能。

---

## 2. S-1: Epoch の意味の純化

### 2.1 現状

`DSPLifetimeManager::retire()` が `publishEpoch()` を呼び、Epoch をインクリメントしている。
これにより Publish と Retire が同一の Epoch Source を共有し、Epoch = Publication Generation という意味論が崩れている。

### 2.2 影響を受けるファイル

| ファイル | 役割 | 変更種別 |
|---|---|---|
| `src/audioengine/DSPLifetimeManager.h` | Retire 呼び出し元 | 修正 |
| `src/core/EpochDomain.h` | Epoch 管理 | 影響確認のみ（修正不要） |
| `src/core/IPublicationProvider.h` | インターフェース定義 | 修正の必要なし |

### 2.3 改修手順

#### Step 1: DSPLifetimeManager::retire() の修正

**ファイル**: `src/audioengine/DSPLifetimeManager.h`

**現状** (42行目):

```cpp
const uint64_t epoch = router_->publishEpoch();
```

**修正後**:

```cpp
const uint64_t epoch = router_->currentEpoch();
```

**変更内容**: `publishEpoch()` → `currentEpoch()` に置換。
これにより retire 時に epoch がインクリメントされなくなる。

#### Step 2: 影響確認 — publishEpoch() の他の使用箇所

`publishEpoch()` / `publishEpoch()` の全使用箇所を確認し、retire 経路以外は変更しない。

| 使用箇所 | ファイル | 変更 |
|---|---|---|
| `DSPLifetimeManager::retire()` | `DSPLifetimeManager.h:42` | **変更対象** |
| `AudioEngine::markRetireEpoch()` | `AudioEngine.Publication.cpp:17` | 変更しない（Publish 時の epoch advance は正しい動作） |
| `AudioEngine::advanceRetireEpoch()` | `AudioEngine.Publication.cpp:27` | 変更しない（Publish 後 epoch advance は正しい） |
| `AudioEngine::~AudioEngine()`（graceful drain） | `AudioEngine.CtorDtor.cpp:152,169` | 変更しない（drain 促進のための意図的な epoch advance。コメントで意図を明記） |

#### Step 3: インターフェース分離（将来拡張）

現在 `IEpochProvider` → `IPublicationProvider` 経由で `publishEpoch()` が提供されている。
この設計は妥当であり、変更しない。`currentEpoch()` との使い分けを呼び出し側で行う。

### 2.4 検証方法

| 検証項目 | 方法 |
|---|---|
| Retire 時に epoch が進まないこと | `EpochDomain::globalEpoch` の値を retire 前後で比較 |
| Publish 時に epoch が進むこと | 変更なし。従来通り。 |
| 回帰テスト（Retire/Publish 混在シナリオ） | ISR テレメトリの epoch 値が正しいこと |

### 2.5 リスク

- なし。`currentEpoch()` は読み取り専用（`consumeAtomic`）であり、スレッド安全性に影響しない。
- `enqueueRetire()` の epoch 引数として使われる値が「発行時点の epoch」になるが、これは RCU の safe-epoch 計算（`minReaderEpoch`）と整合する。

---

## 3. S-2: Health→Control Pipeline 完成

### 3.1 現状

`PublicationAdmission` のみ HealthState が配線済み。以下の4経路が未接続:

| 経路 | ファイル | 現状 |
|---|---|---|
| Rebuild Admission | `AudioEngine.RebuildDispatch.cpp` | `shouldRejectRebuildAdmissionForPressure()` は retire pressure のみ |
| RuntimeBuilder | `RuntimeBuilder.cpp` | `build()` に HealthState 参照なし |
| CrossfadeAuthority | `CrossfadeAuthority.cpp` | `evaluate()` に HealthState 参照なし |
| DSPTransition | `DSPTransition.h` | `onPublishCompleted()` に HealthState チェックなし |

### 3.2 影響を受けるファイル

| ファイル | 変更種別 |
|---|---|
| `src/audioengine/AudioEngine.h` | 修正（HealthStateRef 公開メソッド追加） |
| `src/audioengine/AudioEngine.RebuildDispatch.cpp` | 修正（shouldRejectRebuildAdmissionForPressure 拡張） |
| `src/audioengine/AudioEngine.Threading.cpp` | 修正（shouldRejectRebuildAdmissionForPressure 実装変更） |
| `src/audioengine/RuntimeBuilder.h` | 修正（HealthState 参照メンバ追加） |
| `src/audioengine/RuntimeBuilder.cpp` | 修正（build() 先頭でチェック追加） |
| `src/audioengine/CrossfadeAuthority.h` | 修正（HealthState 参照メンバ追加） |
| `src/audioengine/CrossfadeAuthority.cpp` | 修正（evaluate() 先頭でチェック追加） |
| `src/audioengine/DSPTransition.h` | 修正（onPublishCompleted 先頭でチェック追加） |

### 3.3 改修方針

各コンポーネントに共通の HealthState 判定パターンを導入する:

```cpp
// 共通パターン: 各経路の入口で HealthState をチェック
if (m_healthStateRef) {
    auto health = convo::consumeAtomic(*m_healthStateRef, std::memory_order_acquire);
    if (health == ISRHealthState::Critical) {
        // Critical: 全リクエスト拒否
        return /* 適切な拒否値 */;
    }
    if (health == ISRHealthState::Degraded) {
        // Degraded: 低優先度リクエストを拒否（経路ごとに判断）
        return /* 適切な拒否値 */;
    }
}
```

### 3.4 改修手順

#### Step 1: HealthStateRef 取得用メソッドを AudioEngine に追加

**ファイル**: `src/audioengine/AudioEngine.h`

既存の `m_healthMonitor` から取得できる HealthStateRef を外部公開する方法を確認。
現在 `RuntimePublicationOrchestrator` はコンストラクタ後の明示的な設定 (`setAdmissionHealthStateRef`) で受け取っている。
これと同様の方法を各コンポーネントに適用する。

**変更案**:

```cpp
// AudioEngine.h に追加（public セクション）
[[nodiscard]] const std::atomic<ISRHealthState>* getHealthStateRef() const noexcept {
    return m_healthMonitor.getHealthStateRef();
}
```

#### Step 2: Rebuild Admission 改修

**ファイル**: `src/audioengine/AudioEngine.Threading.cpp`

```cpp
bool AudioEngine::shouldRejectRebuildAdmissionForPressure() const noexcept
{
    // 既存: retirePressureAdmissionStrict_ チェック
    if (convo::consumeAtomic(retirePressureAdmissionStrict_, std::memory_order_acquire))
        return true;

    // ★ 追加: HealthState Critical チェック
    auto health = m_healthMonitor.getHealthState();
    if (health == ISRHealthState::Critical)
        return true;

    return false;
}
```

**注意**: `AudioEngine::shouldRejectRebuildAdmissionForPressure()` は `const` メソッドであり、
`m_healthMonitor.getHealthState()` も `const` なので問題ない。

#### Step 3: RuntimeBuilder 改修

**ファイル**: `src/audioengine/RuntimeBuilder.h`

```cpp
class RuntimeBuilder {
public:
    // ★ HealthState 参照設定メソッドを追加
    void setHealthStateRef(const std::atomic<ISRHealthState>* ref) noexcept {
        m_healthStateRef = ref;
    }

    // ... 既存メソッド ...

private:
    AudioEngine& engine;
    // ★ HealthState 参照メンバを追加
    const std::atomic<ISRHealthState>* m_healthStateRef = nullptr;
};
```

**ファイル**: `src/audioengine/RuntimeBuilder.cpp` — `build()` 先頭

```cpp
BuildResult RuntimeBuilder::build(const BuildInput& in,
                                  const ConvolverProcessor::BuildSnapshot& convolverBuildSnapshot) noexcept
{
    BuildResult result {};

    // ★ HealthState Critical チェックを追加
    if (m_healthStateRef) {
        auto health = convo::consumeAtomic(*m_healthStateRef, std::memory_order_acquire);
        if (health == ISRHealthState::Critical) {
            result.error = BuildError::ResourceUnavailable;
            return result;
        }
    }

    if (in.sampleRate <= 0.0 || in.blockSize <= 0) {
        result.error = BuildError::InvalidInput;
        return result;
    }
    // ... 以降既存のまま ...
```

#### Step 4: CrossfadeAuthority 改修

**ファイル**: `src/audioengine/CrossfadeAuthority.h`

```cpp
class CrossfadeAuthority {
public:
    // ★ HealthState 参照設定メソッドを追加
    void setHealthStateRef(const std::atomic<ISRHealthState>* ref) noexcept {
        m_healthStateRef = ref;
    }

    Decision evaluate(const AudioEngine& engine,
                      const RuntimePublishWorld& oldWorld,
                      const RuntimePublishWorld& newWorld) noexcept;

private:
    // ★ HealthState 参照メンバを追加
    const std::atomic<ISRHealthState>* m_healthStateRef = nullptr;
};
```

**ファイル**: `src/audioengine/CrossfadeAuthority.cpp` — `evaluate()` 先頭

```cpp
CrossfadeAuthority::Decision CrossfadeAuthority::evaluate(
    const AudioEngine& engine,
    const RuntimePublishWorld& oldWorld,
    const RuntimePublishWorld& newWorld) noexcept
{
    Decision ctx;

    // ★ HealthState Critical チェックを追加
    if (m_healthStateRef) {
        auto health = convo::consumeAtomic(*m_healthStateRef, std::memory_order_acquire);
        if (health == ISRHealthState::Critical) {
            // Critical: crossfade 不要として返す
            return ctx;  // ctx.needsCrossfade = false のまま
        }
    }

    // ... 以降既存のまま ...
```

#### Step 5: DSPTransition 改修

**ファイル**: `src/audioengine/DSPTransition.h`

```cpp
class DSPTransition {
public:
    explicit DSPTransition(AudioEngine& engine) noexcept : engine_(engine) {}

    // ★ HealthState 参照設定メソッドを追加
    void setHealthStateRef(const std::atomic<ISRHealthState>* ref) noexcept {
        m_healthStateRef = ref;
    }

    void onPublishCompleted(AudioEngine::DSPCore* newDSP,
                            AudioEngine::DSPCore* oldDSP,
                            const CrossfadeAuthority::Decision& decision,
                            DSPLifetimeManager& lifetime) noexcept
    {
        // ★ HealthState Critical チェック: Critical なら即 retire（crossfade は行わない）
        if (m_healthStateRef) {
            auto health = convo::consumeAtomic(*m_healthStateRef, std::memory_order_acquire);
            if (health == ISRHealthState::Critical) {
                // Critical: crossfade スキップ、旧 DSP を即 retire
                lifetime.activate(newDSP);
                if (oldDSP != nullptr) {
                    engine_.crossfadeRuntime_.complete();
                    lifetime.retire(oldDSP);
                }
                return;
            }
        }

        // 1. activate (publish 成功後にのみ実行)
        lifetime.activate(newDSP);
        // ... 以降既存のまま ...
```

#### Step 6: RuntimePublicationOrchestrator で配線

**ファイル**: `src/audioengine/RuntimePublicationOrchestrator.cpp` または `h`

`RuntimePublicationOrchestrator` は `DSPTransition` と `RuntimeBuilder` をメンバとして保持するため、
その初期化時に HealthStateRef を伝播する:

```cpp
// RuntimePublicationOrchestrator::RuntimePublicationOrchestrator() 内または
// setAdmissionHealthStateRef 呼び出し時に併せて設定
void RuntimePublicationOrchestrator::setAdmissionHealthStateRef(const std::atomic<ISRHealthState>* ref) noexcept
{
    admission_.setHealthStateRef(ref);
    transition_.setHealthStateRef(ref);  // ★ 追加
    // CrossfadeAuthority はローカル変数のため、trySubmit() 内で設定
}
```

`CrossfadeAuthority` は `trySubmit()` 内のローカル変数なので:

```cpp
// RuntimePublicationOrchestrator.cpp:100 付近
CrossfadeAuthority crossfade;
crossfade.setHealthStateRef(admission_.getHealthStateRef());  // ★ 追加
auto cfDecision = crossfade.evaluate(engine_, *oldWorld, *worldOwner);
```

### 3.5 検証方法

| 検証項目 | 方法 |
|---|---|
| HealthState Critical → Rebuild 抑制 | HealthMonitor を Critical 状態にして Rebuild リクエストが拒否されること |
| HealthState Critical → Builder 抑制 | 同上、RuntimeBuilder::build() が ResourceUnavailable を返すこと |
| HealthState Critical → Crossfade 抑制 | 同上、CrossfadeAuthority が needsCrossfade=false を返すこと |
| HealthState Critical → DSPTransition 抑制 | 同上、crossfade を開始せず即 retire すること |
| HealthState Healthy → 従来動作 | Degraded/Critical でない場合、一切の動作が変わらないこと |
| 回帰テスト（正常系） | 既存の ISR テストが全てパスすること |

### 3.6 リスク

- **CrossfadeAuthority のインターフェース変更**: `evaluate()` は `const` ではないが、HealthState 参照の読み取りはスレッドセーフ。
- **DSPTransition の分岐増加**: Critical 時に crossfade をスキップする分岐が入る。Audio Thread には影響しない（NonRT のみ）。

### 3.7 設計上の注意: Health Policy の分散リスク

本計画では4経路に個別に `if (health == Critical)` を導入する方式を採っている。
これは短期的には正しいが、**Health Policy が分散する** リスクがある。

今後の長期的な改修では、以下への集約を検討すべき:

```cpp
// 理想形: RuntimeGovernancePolicy への集約
HealthDecision decision = governancePolicy.evaluate(currentHealth);

// decision を各経路に伝播
if (!decision.allowRebuild)  rejectRebuild();
if (!decision.allowBuild)    abortBuild();
if (!decision.allowCrossfade) skipCrossfade();
```

この集約により:

- Health 判定ロジックの一貫性確保
- 新規経路追加時の漏れ防止
- ポリシーの動的変更が容易

ただし現時点では過剰設計のリスクもあるため、まずは個別実装で済ませ、
Health→Control Pipeline の完成後にリファクタリングとして集約を検討するのが現実的。

---

## 4. A-1: Health→Recovery 実装（自己防衛）

### 4.1 現状

`onHealthEvent()` は Crossfade Timeout の回復処理のみ実装。Reader Exhaustion / Publication Stall / Retire Stall の各イベントはログ出力のみ。

### 4.2 影響を受けるファイル

| ファイル | 変更種別 |
|---|---|
| `src/audioengine/AudioEngine.Timer.cpp` | 修正（onHealthEvent 拡張） |
| `src/audioengine/AudioEngine.h` | 修正（必要に応じてヘルパーメソッド追加） |
| `src/audioengine/RuntimeHealthMonitor.h` | 影響確認（イベントコードは既存） |

### 4.3 改修手順

#### Step 1: onHealthEvent() に ReaderExhaustion 回復処理を追加

**ファイル**: `src/audioengine/AudioEngine.Timer.cpp`

```cpp
void AudioEngine::onHealthEvent(const convo::HealthEvent& event) noexcept
{
    diagLog("[HEALTH] eventCode=" + juce::String(static_cast<int>(event.eventCode))
        + " severity=" + juce::String(static_cast<int>(event.severity))
        + " value=" + juce::String(static_cast<juce::int64>(event.value)));

    // ★ Recovery: Reader Slot Exhaustion → Admission 強制停止
    if (event.eventCode == convo::EVENT_READER_SLOT_USAGE
        && event.severity == convo::HealthEvent::Severity::Error)
    {
        diagLog("[HEALTH] Reader slot exhaustion detected, forcing admission stop");
        // Admission は既に HealthState Critical で停止するはずだが、
        // 念のため retirePressureAdmissionStrict_ も直接設定
        convo::publishAtomic(retirePressureAdmissionStrict_, true, std::memory_order_release);

        // ★ 強制診断ダンプを出力
        emitEvidenceTickNonRt(true);
        worldLifecycleAudit_.tryDumpPeriodic();
        return;
    }

    // ★ Recovery: Publication Stall → Crossfade Stop
    if (event.eventCode == convo::EVENT_PUBLICATION_STALL
        && event.severity == convo::HealthEvent::Severity::Error)
    {
        diagLog("[HEALTH] Publication stall detected, initiating recovery");
        // 滞留中の deferred publish を drain
        runtimeOrchestrator_->clearDeferredForShutdown();

        // 強制診断ダンプ
        emitEvidenceTickNonRt(true);
        return;
    }

    // ★ Recovery: Retire Stall → Builder Throttle
    if ((event.eventCode == convo::EVENT_RETIRE_STALL
         || event.eventCode == convo::EVENT_RETIRE_AGE_CRITICAL)
        && event.severity == convo::HealthEvent::Severity::Error)
    {
        diagLog("[HEALTH] Retire stall detected, throttling rebuild");
        // retirePressureAdmissionStrict_ は既に retire pressure policy で設定済みのはずだが、
        // HealthMonitor 経由でも明示的に設定
        convo::publishAtomic(retirePressureAdmissionStrict_, true, std::memory_order_release);

        // 強制 retire reclaim を実行
        tryReclaimResources();

        // 強制診断ダンプ
        emitEvidenceTickNonRt(true);
        return;
    }

    // ★ 既存: Crossfade Timeout 回復処理
    if (event.eventCode == convo::EVENT_CROSSFADE_TIMEOUT)
    {
        // ... 既存のまま ...
    }
}
```

#### Step 2: HealthEvent → EvidenceExporter 連携

現在 `emitEvidenceTickNonRt()` は定期タイマーでのみ呼ばれている。
HealthEvent 発生時に強制的にエビデンス出力することで、障害発生時の診断情報を確実に残す。

**変更箇所**: `onHealthEvent()` 内の各分岐で `emitEvidenceTickNonRt(true)` を呼ぶ（上記 Step 1 で既に記述済み）。

#### Step 3: HealthEvent → Telemetry 連携

現在 `TelemetryRecorder` は `RuntimePublicationOrchestrator` 内で publish 関連の記録のみ。
HealthEvent を Telemetry に記録する仕組みを追加する。

**変更案**: `AudioEngine.Timer.cpp` の `onHealthEvent()` 内で Telemetry に記録:

```cpp
// TelemetryRecorder への記録（RuntimePublicationOrchestrator 経由）
if (runtimeOrchestrator_) {
    runtimeOrchestrator_->telemetryRecorder().recordHealthEvent(
        event.eventCode,
        static_cast<uint64_t>(event.severity),
        event.value);
}
```

**注意**: `TelemetryRecorder` に `recordHealthEvent()` メソッドが存在しない場合、
追加が必要（`TelemetryRecorder.h` の改修）。

### 4.4 検証方法

| 検証項目 | 方法 |
|---|---|
| Reader Exhaustion → Admission 強制停止 | Reader Slot 使用率 90% 超過で `retirePressureAdmissionStrict_` が true になること |
| Publication Stall → Crossfade Stop | 30秒以上停滞で deferred がクリアされること |
| Retire Stall → Builder Throttle | Retire backlog 超過で `retirePressureAdmissionStrict_` が true になること |
| エビデンス強制出力確認 | 各 Recovery 発動時に evidence ディレクトリにダンプが出力されること |
| 回帰テスト | Crossfade Timeout 回復が従来通り動作すること |

### 4.5 リスク

- **Admission 強制停止の副作用**: `retirePressureAdmissionStrict_ = true` により、正常な Publish も停止する可能性がある。ただし HealthState Critical 時はそれで正しい動作。
- **`tryReclaimResources()` の呼び出し**: Retire stall 回復で呼ぶが、既存の定期タイマーと二重呼び出しにならないことを確認。

---

## 5. B-1: Reader Ownership Telemetry 完成

### 5.1 現状

`HealthEvent` に `readerIndex` / `readerEpoch` / `readerDepth` / `residencyTimeUs` フィールドが定義されているが、
`checkReaderSlotUsage()` で実際に埋められていない。

### 5.2 影響を受けるファイル

| ファイル | 変更種別 |
|---|---|
| `src/audioengine/RuntimeHealthMonitor.cpp` | 修正（checkReaderSlotUsage 拡張） |
| `src/audioengine/RuntimeHealthMonitor.h` | 影響確認のみ |
| `src/audioengine/ISRRetireRouter.h` | 必要に応じて reader 詳細取得メソッド追加 |

### 5.3 改修手順

#### Step 1: ISRRetireRouter に Reader Slot 詳細取得メソッド追加

**ファイル**: `src/audioengine/ISRRetireRouter.h`

```cpp
// ★ Reader Ownership Telemetry: 特定 slot の詳細情報を取得
struct ReaderSlotDetail {
    uint64_t epoch;
    uint32_t depth;
    uint64_t residencyTimeUs;
    bool active;
};

[[nodiscard]] ReaderSlotDetail getReaderSlotDetail(int readerIndex) const noexcept;
```

**ファイル**: `src/audioengine/ISRRetireRouter.cpp`

```cpp
ISRRetireRouter::ReaderSlotDetail ISRRetireRouter::getReaderSlotDetail(int readerIndex) const noexcept
{
    assert(provider_ != nullptr);
    // EpochDomain の ReaderSlot 詳細を取得するインターフェース委譲
    return provider_->getReaderSlotDetail(readerIndex);
}
```

**ファイル**: `src/core/IEpochProvider.h`

```cpp
// インターフェースに追加
struct ReaderSlotDetail {
    uint64_t epoch;
    uint32_t depth;
    uint64_t residencyTimeUs;
    bool active;
};

[[nodiscard]] virtual ReaderSlotDetail getReaderSlotDetail(int readerIndex) const noexcept {
    return ReaderSlotDetail{0, 0, 0, false};  // デフォルト実装
}
```

**ファイル**: `src/core/EpochDomain.h`

```cpp
ReaderSlotDetail getReaderSlotDetail(int readerIndex) const noexcept override
{
    if (readerIndex < 0 || readerIndex >= kMaxReaders)
        return ReaderSlotDetail{0, 0, 0, false};

    const auto& slot = readers[static_cast<size_t>(readerIndex)];
    const uint64_t epoch = convo::consumeAtomic(slot.epoch, std::memory_order_acquire);
    const uint32_t depth = convo::consumeAtomic(slot.depth, std::memory_order_acquire);
    const uint64_t startUs = convo::consumeAtomic(slot.residencyStartTimestampUs, std::memory_order_acquire);
    const auto nowUs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    const uint64_t residencyUs = (startUs != 0 && depth > 0) ? (nowUs - startUs) : 0;
    const bool active = (depth > 0);

    return ReaderSlotDetail{epoch, depth, residencyUs, active};
}
```

#### Step 2: checkReaderSlotUsage() で個別 Reader 情報を設定

**ファイル**: `src/audioengine/RuntimeHealthMonitor.cpp`

現状の `checkReaderSlotUsage()` は全体の使用率のみ監視。以下のように拡張:

```cpp
void RuntimeHealthMonitor::checkReaderSlotUsage() noexcept
{
    // ... 既存の使用率チェック ...

    // ★ 使用率が閾値を超えた場合、個別 Reader 情報を詳細取得
    if (usage >= kReaderSlotCriticalThreshold) {
        // 最も古い Reader slot を特定
        int worstReaderIndex = -1;
        uint64_t worstResidencyUs = 0;

        if (m_retireRouter) {
            int capacity = m_retireRouter->readerCapacity();
            for (int i = 0; i < capacity; ++i) {
                auto detail = m_retireRouter->getReaderSlotDetail(i);
                if (detail.active && detail.residencyTimeUs > worstResidencyUs) {
                    worstResidencyUs = detail.residencyTimeUs;
                    worstReaderIndex = i;
                }
            }
        }

        // 最も滞留している Reader の情報を HealthEvent に設定
        // emitOnTransition ではなく直接コールバックを呼ぶ
        if (m_callback && worstReaderIndex >= 0) {
            auto detail = m_retireRouter->getReaderSlotDetail(worstReaderIndex);
            HealthEvent ev{getCurrentTimeUs(),
                           HealthEvent::Severity::Error,
                           EVENT_READER_SLOT_USAGE,
                           activeCount,
                           maxSlots};
            ev.readerIndex = worstReaderIndex;
            ev.readerEpoch = detail.epoch;
            ev.readerDepth = detail.depth;
            ev.residencyTimeUs = detail.residencyTimeUs;
            m_callback(ev);
        }
    }

    // ★ 元の emitOnTransition での使用率通知は残す（全体傾向把握用）
    // ... 既存の emitOnTransition ...
}
```

#### Step 3: onHealthEvent で Reader Slot 詳細を Evidence に出力

`AudioEngine.Timer.cpp` の `onHealthEvent()` 内で Reader Slot 詳細を Evidence に含める（A-1 の回復処理と組み合わせる）。

### 5.4 検証方法

| 検証項目 | 方法 |
|---|---|
| Reader Slot 詳細取得 | `getReaderSlotDetail(i)` で正しい epoch/depth/residency が取得できること |
| コールバックに Reader 詳細が含まれる | EVENT_READER_SLOT_USAGE 発火時に readerIndex/residencyTimeUs が 0 以外であること |

### 5.5 リスク

- なし。読み取り専用のインターフェース追加であり、既存の RCU 動作に影響しない。

---

## 6. C-1: RuntimeDrainAudit ↔ WorldLifecycleAudit 連携

### 6.1 現状

`WorldLifecycleAudit` は `activeWorldCount` / `publishedCount` / `retiredCount` を保持しているが、
`RuntimeDrainAudit` が参照していない。

### 6.2 影響を受けるファイル

| ファイル | 変更種別 |
|---|---|
| `src/audioengine/RuntimeDrainAudit.h` | 修正（フィールド追加） |
| `src/audioengine/AudioEngine.Threading.cpp` | 修正（collectDrainAudit 拡張） |

### 6.3 改修手順

#### Step 1: RuntimeDrainAudit に World カウンタ追加

**ファイル**: `src/audioengine/RuntimeDrainAudit.h`

```cpp
struct RuntimeDrainAudit {
    uint64_t pendingPublication;
    uint64_t pendingRetire;
    uint64_t activeCrossfadeCount;
    uint64_t routerPendingRetire;
    uint64_t maxDeferredAgeMs;
    uint64_t deferredPublish;
    uint64_t quarantineResident;    // 監査のみ
    uint64_t oldestPendingAgeMs;    // 監査のみ
    uint64_t maxQuarantineAgeSec;   // 監査のみ
    // ★ WorldLifecycleAudit 連携（診断目的）
    uint64_t activeWorldCount;      // active world 数
    uint64_t publishedCount;        // 累積 publish world 数
    uint64_t retiredCount;          // 累積 retire world 数
};
```

#### Step 2: collectDrainAudit() で World カウンタを取得

**ファイル**: `src/audioengine/AudioEngine.Threading.cpp`

```cpp
convo::isr::RuntimeDrainAudit AudioEngine::collectDrainAudit() noexcept
{
    return convo::isr::RuntimeDrainAudit{
        .pendingPublication = runtimePublicationBridge_.getPublicationBacklogCount(),
        .pendingRetire = retireRuntime_.pendingIntentCount(),
        .activeCrossfadeCount = crossfadeRuntime_.isPending() ? 1u : 0u,
        .routerPendingRetire = static_cast<uint64_t>(m_retireRouter->pendingRetireCount())
            + convo::consumeAtomic(fallbackQueueDepth_, std::memory_order_acquire),
        .maxDeferredAgeMs = runtimeOrchestrator_
            ? runtimeOrchestrator_->getMaxDeferredAgeMs() : 0u,
        .deferredPublish = (runtimeOrchestrator_
            && runtimeOrchestrator_->hasDeferredRequest()) ? 1u : 0u,
        .quarantineResident = dspQuarantineManager_.residentCount(),
        .oldestPendingAgeMs = static_cast<uint64_t>(
            std::max(0.0, convo::consumeAtomic(oldestPendingAge_, std::memory_order_acquire))),
        .maxQuarantineAgeSec = dspQuarantineManager_.getMaxEntryAgeSec(),
        // ★ 追加: WorldLifecycleAudit から World カウンタ取得
        .activeWorldCount = worldLifecycleAudit_.activeWorldCount(),
        .publishedCount = worldLifecycleAudit_.publishedCount(),
        .retiredCount = worldLifecycleAudit_.retiredCount()
    };
}
```

### 6.4 検証方法

| 検証項目 | 方法 |
|---|---|
| collectDrainAudit に World カウンタ含まれる | 戻り値の activeWorldCount/publishedCount/retiredCount が 0 以外 |
| World 発行でカウンタ増加 | publish 後に publishedCount が増加すること |

### 6.5 リスク

- なし。読み取り専用の追加であり、既存の shutdown 判定に影響しない。

---

## 7. C-2: BuildError 分類拡充

### 7.1 現状

`BuildError` は `None` / `InvalidInput` / `ResourceUnavailable` / `WarmupFailed` / `InternalError` の5種類。
`catch(...)` で捉えた例外はすべて `InternalError` に丸められている。

### 7.2 影響を受けるファイル

| ファイル | 変更種別 |
|---|---|
| `src/audioengine/RuntimeBuilder.h` | 修正（enum 拡張） |
| `src/audioengine/RuntimeBuilder.cpp` | 修正（toString / build の catch 拡張） |

### 7.3 改修手順

#### Step 1: BuildError enum 拡張

**ファイル**: `src/audioengine/RuntimeBuilder.h`

```cpp
enum class BuildError {
    None,
    InvalidInput,
    ResourceUnavailable,
    MKLFailure,          // ★ MKL 初期化・FFT 計画失敗
    ConvolverFailure,    // ★ Convolver Build 失敗
    PrepareFailure,      // ★ DSPCore::prepare() 失敗
    WarmupFailed,
    InternalError
};
```

#### Step 2: toString() 拡張

**ファイル**: `src/audioengine/RuntimeBuilder.cpp`

```cpp
const char* toString(BuildError error) noexcept
{
    switch (error)
    {
        case BuildError::None:               return "None";
        case BuildError::InvalidInput:       return "InvalidInput";
        case BuildError::ResourceUnavailable: return "ResourceUnavailable";
        case BuildError::MKLFailure:         return "MKLFailure";
        case BuildError::ConvolverFailure:   return "ConvolverFailure";
        case BuildError::PrepareFailure:     return "PrepareFailure";
        case BuildError::WarmupFailed:       return "WarmupFailed";
        case BuildError::InternalError:      return "InternalError";
    }
    return "Unknown";
}
```

#### Step 3: build() の catch 拡張

**ファイル**: `src/audioengine/RuntimeBuilder.cpp`

現在の `catch(...)` を具体的な例外型で捕捉するよう拡張。ただし `noexcept` 指定の関数内での例外捕捉には限界があるため、最小限の改善として:

```cpp
try
{
    runtime = convo::aligned_make_unique<AudioEngine::DSPCore>();
    runtime->convolverRt().setVisualizationEnabled(false);
    runtime->convolverRt().applyBuildSnapshot(convolverBuildSnapshot);
    runtime->prepare(in.sampleRate, in.blockSize, ...);
    result.runtime = runtime.release();
    result.prepared = true;
    return result;
}
catch (const std::bad_alloc&)
{
    result.error = BuildError::ResourceUnavailable;
    return result;
}
// ★ 追加: MKL 関連例外の捕捉
catch (const mkl::exception&)
{
    result.error = BuildError::MKLFailure;
    return result;
}
catch (...)
{
    result.error = BuildError::InternalError;
    return result;
}
```

**注意**: oneMKL の例外型が `mkl::exception` かどうかは oneMKL のバージョンに依存。
実際の例外型を確認して修正すること。標準の `std::exception` から派生している場合は
`catch (const std::exception& e)` で捕捉し、`e.what()` の内容から MKL/Convolver/Prepare を判定する方式も検討。

#### Step 4: rebuildThreadLoop の catch 改善（オプション）

**ファイル**: `src/audioengine/AudioEngine.RebuildDispatch.cpp`

```cpp
catch (const std::exception& e)
{
    DBG("AudioEngine::rebuildThreadLoop exception: " << e.what());
    // ★ エラー内容を BuildResult 形式で Telemetry に記録
    convo::fetchAddAtomic(buildErrorCount_, static_cast<std::uint64_t>(1),
                          std::memory_order_acq_rel);
    juce::ignoreUnused(e);
}
catch (...)
{
    DBG("AudioEngine::rebuildThreadLoop unknown exception");
    convo::fetchAddAtomic(buildErrorCount_, static_cast<std::uint64_t>(1),
                          std::memory_order_acq_rel);
}
```

### 7.4 検証方法

| 検証項目 | 方法 |
|---|---|
| BuildError の文字列表現 | `toString(BuildError::MKLFailure)` → `"MKLFailure"` |
| 新規エラー型の伝搬確認 | MKL 例外発生時に BuildError::MKLFailure が返ること |

### 7.5 リスク

- なし。enum の拡張であり、既存の switch-case に default があるため互換性を維持。

---

## 8. C-3: EpochDomain Reader Slot 可観測性強化

### 8.1 現状

`kMaxReaders = 64` は固定だが、枯渇時に「どのスレッドが占有しているか」を特定する手段がない。
`ReaderSlot` 構造体にはスレッド名や所有者識別子が保存されていない。

### 8.2 影響を受けるファイル

| ファイル | 変更種別 |
|---|---|
| `src/core/EpochDomain.h` | 修正（ReaderSlot に所有者情報追加） |

### 8.3 改修手順

#### Step 1: ReaderSlot に所有者識別子を追加

**ファイル**: `src/core/EpochDomain.h`

```cpp
struct ReaderSlot
{
    std::atomic<uint64_t> epoch { kInactiveEpoch };
    std::atomic<uint32_t> depth { 0 };
    std::atomic<uint64_t> enterCount { 0 };
    std::atomic<uint64_t> residencyStartTimestampUs { 0 };
    // ★ C-3: Reader 所有者情報（スレッドID / タグ名）
    std::atomic<uint64_t> ownerThreadId { 0 };       // std::thread::id のハッシュ値
    // ownerTag は atomic 非互換のため、registerReaderThread 時に設定
    char ownerTag[32] {};  // "AudioThread", "TimerThread", "WorkerThread" 等（mutex 保護下で設定）
};
```

**注意**: `ownerTag` は `char[]` であり atomic ではない。`registerReaderThread()` と `reserveReaderThread()` 内で
ミューテックス保護下で設定するか、または書き捨て（stale read 許容）とする。

#### Step 2: registerReaderThread で所有者タグを設定

**ファイル**: `src/core/EpochDomain.h`

```cpp
int registerReaderThread() noexcept override
{
    // ★ registerReaderThread にタグ名を渡せるオーバーロードを追加
    return registerReaderThread("unnamed");
}

int registerReaderThread(const char* tag) noexcept
{
    for (int i = 0; i < kMaxReaders; ++i)
    {
        uint64_t expected = kInactiveEpoch;
        if (convo::compareExchangeAtomic(readers[static_cast<size_t>(i)].epoch,
                                         expected, kReservedEpoch,
                                         std::memory_order_acq_rel, std::memory_order_acquire))
        {
            convo::publishAtomic(readers[static_cast<size_t>(i)].depth,
                                 static_cast<uint32_t>(0), std::memory_order_release);
            // ★ 所有者タグを設定（mutex 不要: CAS 成功後は単一スレッドのみがこの slot にアクセス）
            std::strncpy(readers[static_cast<size_t>(i)].ownerTag, tag, sizeof(ownerTag) - 1);
            readers[static_cast<size_t>(i)].ownerTag[sizeof(ownerTag) - 1] = '\0';
            convo::publishAtomic(readers[static_cast<size_t>(i)].ownerThreadId,
                                 std::hash<std::thread::id>{}(std::this_thread::get_id()),
                                 std::memory_order_release);
            return i;
        }
    }
    return -1;
}
```

#### Step 3: 既存の registerReaderThread 呼び出し元にタグ名を追加

`RCUReader::acquireThreadSlot()` 内で `epochProvider->registerReaderThread()` が呼ばれている。
この呼び出し元でタグ名を指定できるよう `IReaderEpochProvider` のインターフェースを拡張するか、
`RCUReader` のコンストラクタでタグ名を受け取る。

**簡易対応**: `RCUReader` のコンストラクタにタグ名パラメータを追加。

```cpp
class RCUReader {
public:
    explicit RCUReader(IReaderEpochProvider& provider,
                       const char* readerTag = "unnamed") noexcept
        : epochProvider(&provider), readerTag_(readerTag) {}
    // ...
private:
    const char* readerTag_;
};
```

`acquireThreadSlot()` 内で `epochProvider->registerReaderThread(readerTag_)` を呼ぶ。

### 8.4 検証方法

| 検証項目 | 方法 |
|---|---|
| Reader Slot に所有者タグが設定される | `readers[i].ownerTag` が空文字列でないこと |
| 枯渇時に所有者タグが確認可能 | 全 64 slot 占有後に `ownerTag` を読み取れること |

### 8.5 リスク

- **`char[]` の atomic 性**: `ownerTag` は atomic ではない。登録時の書き捨てであり、
  長期間運用で stale になる可能性がある。問題が起きた場合は最新の所有者を特定できない可能性があるが、
  実用上は「誰が登録したか」が分かれば十分。
- **軽微なメモリ増加**: `64 * 32 = 2048 bytes` の追加。

---

## 9. 実装順序と依存関係

### 9.1 推奨実装順序

実運用リスクと工数対効果を考慮した推奨順序:

```
Tier 1（まず着手）:

  Phase 1: S-2 Health→Control Pipeline（最大の未達）
  ├─ Step 1: AudioEngine に HealthStateRef 公開メソッド追加
  ├─ Step 2: Rebuild Admission 改修（shouldRejectRebuildAdmissionForPressure）
  ├─ Step 3: RuntimeBuilder 改修
  ├─ Step 4: CrossfadeAuthority 改修
  └─ Step 5: DSPTransition 改修 + Orchestrator 配線

  Phase 2: S-1 Epoch の意味の純化（最小工数、並行可能）
  └─ Step 1: DSPLifetimeManager::retire() の一行修正

Tier 2:

  Phase 3: A-1 Health→Recovery（自己防衛、S-2 完了後）
  ├─ Step 1: Reader Exhaustion 回復処理
  ├─ Step 2: Publication Stall 回復処理
  ├─ Step 3: Retire Stall 回復処理
  └─ Step 4: Evidence 強制出力連携

Tier 3:

  Phase 4: B-1 Reader Ownership Telemetry（運用品質）
  ├─ Step 1: IEpochProvider / ISRRetireRouter に詳細取得インターフェース追加
  ├─ Step 2: checkReaderSlotUsage() 拡張
  └─ Step 3: onHealthEvent 連携

Tier 4:

  Phase 5: C 群（診断品質）
  ├─ C-1: RuntimeDrainAudit 拡張
  ├─ C-2: BuildError 分類拡充
  └─ C-3: Reader Slot 可観測性強化
```

### 9.2 依存関係

```
S-2 (Phase 1) ──→ A-1 (Phase 2): S-2 で HealthState 配線完了後に A-1 の Recovery が有効になる
       │
       └──→ B-1 (Phase 4): HealthMonitor の checkReaderSlotUsage 拡張は S-2 と独立

S-1 (Phase 3) ──→ 独立。どのフェーズとも依存しないため Phase 1〜4 の隙間で実施可能

C-1 / C-2 / C-3 (Phase 5) ──→ すべて独立。どのフェーズとも依存しない
```

---

## 10. リスクと注意点

### 10.1 共通注意事項

| 注意事項 | 詳細 |
|---|---|
| **JUCE 8.0.12 との互換性** | すべての変更は JUCE 非依存。標準 C++20 + atomic のみ。 |
| **Audio Thread への影響** | S-2 / A-1 の変更はすべて NonRT スレッド（Timer/Worker/Message Thread）。Audio Thread には影響しない。 |
| **Atomic 操作のメモリオーダリング** | 既存のパターン（`consumeAtomic` / `publishAtomic`）に従う。 |
| **noexcept 指定の維持** | すべての変更は `noexcept` を維持する（SEH / try-catch は既存の範囲内）。 |

### 10.2 各改修のリスク評価

| ID | リスク | 評価 |
|---|---|---|
| S-1 | なし（読み取り専用への置換） | 低 |
| S-2 | CrossfadeAuthority の変更で crossfade が不必要に抑制されるリスク | 中（テストで確認） |
| A-1 | `retirePressureAdmissionStrict_` の直接設定で正常動作を阻害するリスク | 中（HealthState 連動のため許容範囲） |
| B-1 | なし（読み取り専用の追加） | 低 |
| C-1 | なし（読み取り専用の追加） | 低 |
| C-2 | なし（enum 拡張） | 低 |
| C-3 | `char[]` の atomic 性保証なし | 低（実用上許容範囲） |

### 10.3 推奨テストシナリオ

各 Phase 完了後に以下を実施:

1. **正常系回帰**: IR ロード→Publish→Crossfade→Retire の一連の流れ
2. **異常系**: HealthMonitor の検出値を模擬して各 Admission が正しく反応すること
3. **Shutdown シーケンス**: 正常 shutdown / 異常 shutdown の両方
4. **長期安定性**: 1時間以上の連続動作で epoch の異常加速がないこと

---

## 付録: 検証に使用したツール

| ツール | 用途 |
|---|---|
| AiDex MCP | 識別子検索、ファイル構造把握、セッション管理 |
| grep/Select-String | パターン検索、catch(...)/publishEpoch 等の網羅的抽出 |
| CodeGraph MCP | グラフベースのコード解析（global_search） |
| read_file (直接) | 該当ファイルの全文確認 |
