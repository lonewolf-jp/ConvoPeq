# notfinished8.md ソースコード検証レポート

**作成日**: 2026-06-12
**検証方法**: AiDex MCP + grep/Select-String + CodeGraph MCP によるソースコード徹底調査
**検証者**: GitHub Copilot (DeepSeek V4 Flash)

---

## 凡例

- ✅ 文書の主張がコードと一致（確認済み）
- ❌ 文書の主張が不正確（コード上は実装済み）
- ⚠️ 部分的に対応されているが不十分
- 太字 **追加発見**: 文書に記載がなく、今回の調査で新たに発見した漏れ

---

## 項目1: Health Monitor が診断系に留まり、制御系へ十分反映されていない

### 文書の主張

> RuntimeHealthMonitor は異常を発見→ログ出力までだが、異常を発見→新しい Publish を拒否するまで到達していない。

### 検証結果: ⚠️ **部分的に不正確（Publication Admission は既に配線済み）**

#### PublicationAdmission → HealthState: ✅ **既に配線完了**

**コード証拠** (`AudioEngine.CtorDtor.cpp:60`):

```cpp
runtimeOrchestrator_->setAdmissionHealthStateRef(m_healthMonitor.getHealthStateRef());
```

**コード証拠** (`PublicationAdmission.cpp:27-32`):

```cpp
if (m_healthStateRef) {
    auto health = convo::consumeAtomic(*m_healthStateRef, std::memory_order_acquire);
    if (health == ISRHealthState::Critical) {
        return Decision::RejectedPressure;  // Critical: 全 publish 拒否
    }
    if (health == ISRHealthState::Degraded) {
        return Decision::RejectedPressure;  // Degraded: 低優先度拒否
    }
}
```

→ **HealthState Critical/Degraded → Publication 拒否は既に実装されている。**

#### Rebuild Admission → HealthState: ❌ **未配線**

**コード証拠** (`AudioEngine.Threading.cpp:21-23`):

```cpp
bool AudioEngine::shouldRejectRebuildAdmissionForPressure() const noexcept
{
    return convo::consumeAtomic(retirePressureAdmissionStrict_, std::memory_order_acquire);
}
```

`retirePressureAdmissionStrict_` は retire queue の深さに基づいて設定されるフラグであり（`AudioEngine.Retire.cpp:285`）、**HealthMonitor の `ISRHealthState` とは独立した制御**。HealthState Critical でも Rebuild が通る可能性がある。

#### Crossfade 開始 → HealthState: ❌ **未配線**

**コード証拠** (`CrossfadeAuthority::evaluate()`, `CrossfadeAuthority.cpp:14-71`):

- `evaluate()` 内に HealthState 参照なし
- DSPCore 直読ではなく `dspProjection` 投影値のみで判断
- `DSPTransition::onPublishCompleted()` (`DSPTransition.h:51-97`) でも HealthState チェックなし

#### Runtime Builder → HealthState: ❌ **未配線**

**コード証拠** (`RuntimeBuilder::build()`, `RuntimeBuilder.cpp:440-461`):

- HealthState 参照なし
- HealthState Critical でも DSPCore の build が実行され得る

#### **追加発見: HealthMonitor コールバックが診断ログのみ**

**コード証拠** (`AudioEngine.Timer.cpp:534-565`):

```cpp
void AudioEngine::onHealthEvent(const convo::HealthEvent& event) noexcept
{
    diagLog("[HEALTH] eventCode=" + ...);  // ← ログ出力のみ
    // Crossfade Timeout の場合のみ回復処理あり
    if (event.eventCode == convo::EVENT_CROSSFADE_TIMEOUT) { ... }
}
```

→ Reader stuck / Publication stall / Retire stall の各イベントは **ログ出力のみで制御系へのフィードバックなし**。

---

## 項目2: Reader Slot 枯渇時の回復戦略が存在しない

### 文書の主張

> RCUReader は slot 取得失敗時に fail-closed になるが、なぜ枯渇したか・どう復旧するかが無い。

### 検証結果: ✅ **概ね妥当。複数の追加不足を確認**

#### 既存対応（確認済み）

**`RCUReader::enter()` fail-closed** (`RCUReader.h:42-86`):

```cpp
const int tid = acquireThreadSlot();
if (tid >= 0) {
    epochProvider->enterReader(tid);
    rootEnterSucceeded_ = true;
} else {
    rootEnterSucceeded_ = false;  // Fail-Closed
    // ... ownerThreadToken を戻す ...
}
```

**`checkReaderSlotUsage()` 閾値監視** (`RuntimeHealthMonitor.cpp:195-220`):

- 50% → Warning, 75% → Error, 90% → Error（三者三様の閾値）
- capacity を動的取得（router 経由）

**HealthState への反映** (`RuntimeHealthMonitor.cpp:115-139`):

```cpp
if (m_prevReaderSlotState == MonitorState::Error && newState != ISRHealthState::Critical)
    newState = ISRHealthState::Critical;
```

#### **追加発見1: Reader Slot 枯渇時に EvidenceExporter が呼ばれない**

`checkReaderSlotUsage()` は `emitOnTransition()` を呼んで `m_callback` を発火するのみ。
`onHealthEvent()` コールバック内に Reader Slot 枯渇時の Evidence 出力や Telemetry 記録の処理がない。

#### **追加発見2: HealthEvent の reader フィールドが未設定**

`RuntimeHealthMonitor.h:41-44` で定義されたフィールド:

```cpp
int32_t readerIndex{-1};
uint64_t readerEpoch{0};
uint32_t readerDepth{0};
uint64_t residencyTimeUs{0};
```

`checkReaderSlotUsage()` の `emitOnTransition` 呼び出し:

```cpp
emitOnTransition(m_prevReaderSlotState, MonitorState::Error,
    HealthEvent::Severity::Error, EVENT_READER_SLOT_USAGE,
    activeCount, maxSlots);
```

→ `slot` パラメータに `maxSlots`（容量）を渡しているが、**個別 Reader の詳細情報（readerIndex/readerEpoch/readerDepth/residencyTimeUs）は未設定のまま**。

#### **追加発見3: 強制診断ダンプ機構なし**

Reader Slot 枯渇イベント発生時に `WorldLifecycleAudit::emitSnapshot()` や `EvidenceExporter::exportEvidence()` を自動呼び出しする機構がない。

---

## 項目3: WorldLifecycleAudit が監査専用で Shutdown Authority に参加していない

### 文書の主張

> WorldLifecycleAudit は Diagnostic 限定。Shutdown 完了判定の重要材料だが、Shutdown 判定に参加していない。

### 検証結果: ✅ **妥当**

#### コードコメントでの明示

**`WorldLifecycleAudit.h:24`**:

```cpp
// ★ P3-B: World ライフサイクル監査（Diagnostic 限定）
//   Shutdown 完了判定の Authority にはしない。
//   Shutdown 判定は RuntimeDrainAudit + ShutdownRuntime FSM が担当。
```

#### RuntimeDrainAudit に World カウンタ不在

**`RuntimeDrainAudit.h:18-30`**:

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
    // ★ activeWorldCount / publishedCount / retiredCount なし
};
```

#### collectDrainAudit() 実装にも含まれず

**`AudioEngine.Threading.cpp:47-63`**:

```cpp
return convo::isr::RuntimeDrainAudit{
    .pendingPublication = runtimePublicationBridge_.getPublicationBacklogCount(),
    .pendingRetire = retireRuntime_.pendingIntentCount(),
    .activeCrossfadeCount = crossfadeRuntime_.isPending() ? 1u : 0u,
    // ... WorldLifecycleAudit の値は含まれない
};
```

#### **追加発見: 定期ダンプ間隔が60秒固定**

`WorldLifecycleAudit.h:91`:

```cpp
static constexpr uint64_t kDumpIntervalUs = 60'000'000; // 60秒ごとにダンプ
```

Shutdown 直前の状況把握には長すぎる間隔。Shutdown シーケンス中に 60秒待てない。

---

## 項目4: DSPLifetimeManager の retire が epoch を進めている

### 文書の主張

> Retire のたびに publishEpoch() しており、大量 retire 時に epoch が異常加速する。

### 検証結果: ✅ **妥当。追加で別箇所の使用も確認**

#### Retire 経路で publishEpoch() 使用を確認

**`DSPLifetimeManager::retire()` (`DSPLifetimeManager.h:42`)**:

```cpp
const uint64_t epoch = router_->publishEpoch();  // ← fetchAddAtomic(globalEpoch, 1)
router_->enqueueRetire(static_cast<void*>(dsp), &AudioEngine::destroyDSPCoreNode, epoch);
```

#### EpochDomain::publishEpoch() の実装

**`EpochDomain.h:144-148`**:

```cpp
uint64_t publishEpoch() noexcept override
{
    return convo::fetchAddAtomic(globalEpoch, 1, std::memory_order_acq_rel);  // 確実にインクリメント
}
```

#### Publication 経路も同一

**`AudioEngine.Publication.cpp:17`**:

```cpp
[[nodiscard]] uint64_t AudioEngine::markRetireEpoch() noexcept
{
    return m_retireRouter->publishEpoch();  // 同一経路
}
```

→ **Publish と Retire が同一の epoch source を共有しており、大量 retire で epoch が不必要に加速する。**

#### **追加発見: ctor/dtor でも publishEpoch 使用**

**`AudioEngine.CtorDtor.cpp:151-169`**:

```cpp
m_retireRouter->publishEpoch();  // ~AudioEngine の graceful drain ループ内
```

graceful drain 促進の意図は理解できるが、epoch 加速の副作用が発生する。

---

## 項目5: RuntimeBuilder が例外依存

### 文書の主張

> RuntimeBuilder は catch(...) を使っており、原因を消す。

### 検証結果: ⚠️ **部分的に改善済みだが不足あり**

#### RuntimeBuilder::build() の例外処理

**`RuntimeBuilder.cpp:440-461`**:

```cpp
try {
    runtime = convo::aligned_make_unique<AudioEngine::DSPCore>();
    runtime->convolverRt().setVisualizationEnabled(false);
    runtime->convolverRt().applyBuildSnapshot(convolverBuildSnapshot);
    runtime->prepare(in.sampleRate, in.blockSize, ...);
    result.runtime = runtime.release();
    result.prepared = true;
    return result;
} catch (const std::bad_alloc&) {
    result.error = BuildError::ResourceUnavailable;  // ✅ 具体的な分類
    return result;
} catch (...) {
    result.error = BuildError::InternalError;  // ⚠️ 詳細不明
    return result;
}
```

#### BuildError の現状分類

**`RuntimeBuilder.h:8-13`**:

| 分類 | 有無 |
|---|---|
| `None` | ✅ |
| `InvalidInput` | ✅ |
| `ResourceUnavailable` | ✅ |
| `WarmupFailed` | ✅ |
| `InternalError` | ✅ (catch(...) の行き先) |
| **`MKLFailure`** | **❌ なし** |
| **`ConvolverFailure`** | **❌ なし** |
| **`PrepareFailure`** | **❌ なし** |

→ `catch(...)` は `std::bad_alloc` 以外の全例外を `InternalError` に丸めている。MKL 初期化失敗や Convolver Build 失敗の切り分けが不可。

#### **追加発見: rebuildThreadLoop の catch(...) でエラー消失**

**`AudioEngine.RebuildDispatch.cpp:842-847`**:

```cpp
catch (const std::exception& e)
{
    DBG("AudioEngine::rebuildThreadLoop exception: " << e.what());
}
catch (...)
{
    DBG("AudioEngine::rebuildThreadLoop unknown exception");
}
```

→ **DBG ログのみで BuildError に変換されず、上位にエラーが伝搬しない。** エラーが完全に飲み込まれている。

---

## 項目6: CrossfadeRuntime に実クロスフェード数の上限制御が見当たらない

### 文書の主張

> 同時 Crossfade 数の上限制御が見当たらない。

### 検証結果: ⚠️ **自然な制限は存在するが明示的制限なし**

#### 暗黙的な自然制限

| 制限要因 | 説明 |
|---|---|
| `crossfadeRuntime_` 単一インスタンス | 1つの pending crossfade のみ管理可能 |
| `fadingRuntimeDSPSlot` 単一スロット | 1つの fading DSP のみ保持可能 |
| `PublicationAdmission::evaluate()` → `DeferredFadingActive` | fading 中の新規 publish を defer |

#### 明示的制限の不在

| 確認項目 | 状態 |
|---|---|
| `maxConcurrentCrossfades` 定数 | ❌ なし |
| `CrossfadeAuthority::evaluate()` の同時実行数チェック | ❌ なし |
| `CrossfadeAuthorityRuntime::registerCrossfade()` の上限 | ❌ なし |

**`ISRDSPHandle.h:173-176`**:

```cpp
class CrossfadeAuthorityRuntime {
    // ...
    CrossfadeId registerCrossfade(DSPHandle from, DSPHandle to);
    // ...
private:
    std::vector<CrossfadeRecord> crossfadeRecords_;  // ← 無制限ベクタ
    std::atomic<CrossfadeId> nextCrossfadeId_{1};
};
```

#### **追加発見: CrossfadeAuthority::evaluate() に HealthState チェックなし**

`CrossfadeAuthority.cpp:14-71` の `evaluate()` 関数は `dspProjection` 値のみで判断。HealthState Critical でも Crossfade が必要と判断される可能性がある。

#### **追加発見: DSPTransition で crossfade 開始前に HealthState チェックなし**

`DSPTransition::onPublishCompleted()` (`DSPTransition.h:51-97`) は、`decision.needsCrossfade` が true なら HealthState に関係なく crossfade を開始する。

---

## 項目7: EpochDomain が固定64 Reader

### 文書の主張

> kMaxReaders = 64 固定。枯渇時に誰が占有しているか・何秒保持しているかが取得できない。

### 検証結果: ✅ **妥当。追加で複数の不足を確認**

#### kMaxReaders = 64 固定を確認

**`EpochDomain.h:22`**:

```cpp
static constexpr int kMaxReaders = 64;
```

#### HealthEvent フィールド定義は存在

**`RuntimeHealthMonitor.h:41-44`**:

```cpp
int32_t readerIndex{-1};
uint64_t readerEpoch{0};
uint32_t readerDepth{0};
uint64_t residencyTimeUs{0};
```

#### **追加発見1: checkReaderSlotUsage() で個別 Reader 情報未設定**

`checkReaderSlotUsage()` (`RuntimeHealthMonitor.cpp:195-220`) は全体の使用率のみ監視。個別 Reader の `readerIndex`, `readerEpoch`, `readerDepth`, `residencyTimeUs` を設定していない。

```cpp
emitOnTransition(m_prevReaderSlotState, MonitorState::Error,
    HealthEvent::Severity::Error, EVENT_READER_SLOT_USAGE,
    activeCount, maxSlots);  // ← 個別情報なし
```

#### **追加発見2: EpochDomain に Reader 所有権名追跡機構なし**

slot を占有しているスレッドの識別情報（スレッドID・名前等）が保存されていない。枯渇時に「どのスレッドが占有しているか」を特定する手段がない。

**`EpochDomain.h:22-35`**:

```cpp
struct ReaderSlot {
    std::atomic<uint64_t> epoch;
    std::atomic<uint32_t> depth;
    std::atomic<uint64_t> residencyStartTimestampUs{0};  // ← 滞留開始時刻のみ
    // ★ スレッド名 / 所有者識別子なし
};
```

---

## 総合評価サマリ

| # | 項目 | 文書の正確性 | 文書未指摘の追加発見数 |
|---|---|---|---|
| 1 | Health→Control | ⚠️ **不正確**: Publication Admission は既に配線済み | 2件 |
| 2 | Reader Slot 枯渇 | ✅ 妥当 | 3件 |
| 3 | Audit→Shutdown | ✅ 妥当 | 1件 |
| 4 | Retire→publishEpoch | ✅ 妥当 | 1件 |
| 5 | RuntimeBuilder例外 | ⚠️ **部分的に改善済み** | 1件 |
| 6 | Crossfade上限 | ⚠️ 自然制限は存在 | 2件 |
| 7 | Reader固定64 | ✅ 妥当 | 2件 |

### 文書の誤り（最重要）

**項目1**: 文書は「HealthState → Publication Admission」が未実装と主張しているが、**実際は `PublicationAdmission.cpp:27-32` で実装済み**。

### 追加発見トップ5（重要度順）

1. **Rebuild Admission が HealthState を見ていない** — `shouldRejectRebuildAdmissionForPressure()` は retire queue pressure のみ参照
2. **`rebuildThreadLoop` の `catch(...)` でエラー消失** — BuildError に変換されずログのみ
3. **`onHealthEvent()` コールバックが診断ログのみ** — Reader Slot 枯渇イベント時の制御なし
4. **`checkReaderSlotUsage()` で個別 Reader 情報未設定** — 定義済みフィールドが活用されていない
5. **`CrossfadeAuthorityRuntime::registerCrossfade()` に上限なし** — ベクタに無制限追加可能

---

*本レポートは AiDex MCP (Tree-sitter + SQLite), grep/Select-String, CodeGraph MCP を用いたソースコード実証調査に基づきます。*
