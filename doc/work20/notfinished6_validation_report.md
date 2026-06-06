# `doc/work20/notfinished6.md` 検証レポート

> 検証日: 2026-06-06
> 使用ツール: CodeGraph MCP (50,866 entities), ccc (8,701 chunks, 619 files), grep, 実ファイル直接読取
> 検証ファイル: `src/audioengine/DSPLifetimeManager.h`, `DSPTransition.h`, `RuntimePublicationOrchestrator.h/.cpp`,
> `PublicationAdmission.h`, `PublicationExecutor.h/.cpp`, `CrossfadeAuthority.h/.cpp`,
> `RuntimeBuilder.h/.cpp`, `RuntimePublicationCoordinator.h`, `AudioEngine.h`, `AudioEngine.Processing.ReleaseResources.cpp`

---

## 凡例

| 判定 | 意味 |
|------|------|
| ✅ **CORRECT** | ドキュメントの主張が実装と一致 |
| ⚠️ **PARTIALLY CORRECT** | 一部正しいが、実装が部分的に解消済みまたは文脈が異なる |
| ❌ **OUTDATED / INCORRECT** | ドキュメントの主張が現在の実装と矛盾（すでに改修済み） |
| 🔴 **NEW FINDING** | ドキュメント未記載だが発見した追加課題 |

---

## S1. DSPLifetimeManager が AudioEngine ラッパー

### 判定: ✅ CORRECT

**証拠**: `src/audioengine/DSPLifetimeManager.h` 全行

```cpp
class DSPLifetimeManager {
    void activate(AudioEngine::DSPCore* dsp) noexcept
        { engine_.setActiveRuntimeDSP(dsp); }           // ラッパー
    void retire(AudioEngine::DSPCore* dsp) noexcept
        { engine_.retireDSP(dsp); }                     // ラッパー
    AudioEngine::DSPCore* getActive() const noexcept
        { return engine_.getActiveRuntimeDSP(); }       // ラッパー
};
```

コメント自体が「Phase-A: AudioEngine のラッパーに過ぎない」と明記。

**状態**: Phase-A のラッパー段階。`activeHandle_` / `fadingHandle_` の状態保持なし。AudioEngine への委譲のみ。

---

## S2. Execution Path がまだ DSPCore* ベース

### 判定: ✅ CORRECT

**証拠**: `RuntimePublicationOrchestrator.cpp` line 52-53, `DSPTransition.h` line 47-48

```cpp
// RuntimePublicationOrchestrator::trySubmit():
auto* newDSPResolved = engine_.resolveDSPHandle(req.newDSP);  // DSPHandle → DSPCore*
auto* oldDSP = engine_.getActiveRuntimeDSP();                  // DSPCore*

// DSPTransition::onPublishCompleted():
void onPublishCompleted(AudioEngine::DSPCore* newDSP,         // DSPCore* パラメータ
                        AudioEngine::DSPCore* oldDSP, ...)
```

**補足**: `DSPLifetimeManager` もすべて `DSPCore*` ベース。`DSPHandle` は `ISRDSPHandle` に存在するが、Execution Path の引数型としてはまだ浸透していない。

---

## S3. getActiveRuntimeDSP() Authority Leakage

### 判定: ✅ CORRECT（重大）

**証拠**: 全ソース検索結果（grep + ccc）

| ファイル | 行 | 用途 |
|----------|-----|------|
| `RuntimePublicationOrchestrator.cpp` | 53 | `auto* oldDSP = engine_.getActiveRuntimeDSP();` — **Decision 層** |
| `DSPLifetimeManager.h` | 36 | `getActive()` → 委譲 |
| `AudioEngine.h` | 1487, 1492, 2756 | 内部 Execution |
| `AudioEngine.CtorDtor.cpp` | 60, 70, 80 | 初期化/破棄 |
| `AudioEngine.Processing.Latency.cpp` | 84 | Execution |
| `AudioEngine.Processing.PrepareToPlay.cpp` | 244 | world build 時 |
| `AudioEngine.Processing.ReleaseResources.cpp` | 94, 100, 138 | **Shutdown 経路** |

**問題**: `trySubmit()` 内での `getActiveRuntimeDSP()` 呼び出しは Semantic Decision に該当。本来は `RuntimeWorld` の projection 値のみで判断すべき。

---

## ④ Crossfade Runtime State が AudioEngine に残存

### 判定: ✅ CORRECT

**証拠**: `AudioEngine.h` lines 1589-1596

```cpp
std::atomic<bool> dspCrossfadePending { false };
std::atomic<int> dspCrossfadeStartDelayBlocks { 0 };
std::atomic<bool> firstIrDryCrossfadePending { false };
std::atomic<bool> dspCrossfadeUseDryAsOld { false };
std::atomic<int> dspCrossfadeDryHoldSamples { 0 };
convo::LinearRamp dspCrossfadeGain;
std::atomic<double> queuedFadeTimeSec { 0.030 };
std::atomic<CrossfadeId> activeCrossfadeId_ { 0 };
```

これらの全原子変数が `AudioEngine` クラスのメンバとして直接宣言されている。`DSPTransition::onPublishCompleted()` から直接 `engine_.dspCrossfadeGain` のようにアクセスされている。

---

## ⑤ RuntimeStore Publish Authority

### 判定: ⚠️ PARTIALLY CORRECT

**証拠**: `RuntimePublicationCoordinator.h` の `publishWorld()` は publish 経路の唯一の実体。
ただし `RuntimePublicationOrchestrator` → `PublicationExecutor` → `coordinator.publishWorld()` の経路が確立されている。

**懸念**: `RuntimeStore::publishAndSwap()` の呼び出し元は `RuntimePublicationCoordinator` のみ（6件確認）で封鎖はできている。

---

## ⑥ Shutdown Path が旧 Runtime 経路を保持

### 判定: ✅ CORRECT

**証拠**: `AudioEngine.Processing.ReleaseResources.cpp` lines 94-172

```cpp
auto* const activeRaw = getActiveRuntimeDSP();          // 直接取得
setActiveRuntimeDSP(nullptr);                           // 直接設定
auto* const fadingRaw = exchangeFadingRuntimeDSP(nullptr); // 直接交換
retireDSP(activeToRelease);                             // 直接 retire
retireDSP(fadingToRelease);
```

`DSPLifetimeManager` を経由せず、AudioEngine の旧来 API を直接使用。

---

## ⑦ DSPTransition が巨大化

### 判定: ✅ CORRECT

**証拠**: `DSPTransition::onPublishCompleted()` (lines 47-99) ー 以下の責務を単一メソッドが保持:

1. `lifetime.activate(newDSP)` — Activate
2. Crossfade registration (`crossfadeAuthorityRuntime_.registerCrossfade`)
3. `exchangeFadingRuntimeDSP` — Fading スロット操作
4. 全 crossfade atomic 更新 (dspCrossfadeGain, dspCrossfadePending 等 8個)
5. `lifetime.retire(oldDSP)` — Retire

52 行の単一関数で 5 責務を処理している。

---

## ⑧ DSPProjection Coverage

### 判定: ⚠️ PARTIALLY CORRECT

**証拠**: DSPProjection は `AudioEngine.h` line 206 に定義済みで 6 フィールド完備:

```cpp
struct DSPSemanticProjection {
    bool irLoaded = false;          // ✓ CrossfadeAuthority 使用
    bool irFinalized = false;       // (未使用)
    uint64_t structuralHash = 0;    // ✓ CrossfadeAuthority 使用
    int oversamplingFactor = 1;     // ✓ CrossfadeAuthority 使用
    double sampleRate = 48000.0;    // (未使用)
    int baseLatencySamples = 0;     // (未使用)
} dspProjection;
```

`CrossfadeAuthority::kEvaluateRelevantFieldNames` は 3 フィールドのみ:

```cpp
static constexpr std::array<const char*, 3> kEvaluateRelevantFieldNames {{
    "irLoaded", "structuralHash", "oversamplingFactor"
}};
```

**問題**: ドキュメントが指摘するように、`kEvaluateRelevantFieldNames` と `DSPSemanticProjection` の定義に不整合がある。`irFinalized` / `sampleRate` / `baseLatencySamples` が Projection にあるが、Coverage Contract に含まれていない。ただし現状 CrossfadeAuthority が使わないフィールドがあるのは問題ではない（将来追加時に Coverage Contract を更新すればよい）。

---

## ① Legacy Commit Path 残存

### 判定: ❌ OUTDATED / INCORRECT

**証拠**:

```bash
grep -R "applyRuntimeCommitFromIntent" → 0件
grep -R "processPendingCommit" → 0件
grep -R "PendingCommitData" → 0件
grep -R "pendingCommitFlag_" → 1件（コメントのみ）
grep -R "pendingCommit_" → 0件
```

唯一のヒットは `AudioEngine.Threading.cpp:305` のコメント:

```cpp
// [PR-3] Old pendingCommitFlag_ removed. Check Orchestrator deferred queue.
```

Legacy Commit Path はすでに削除済み。`RuntimePublicationOrchestrator` の deferred queue に置き換わっている。

---

## ② Coordinator が Facade 止まり

### 判定: ⚠️ PARTIALLY CORRECT

**証拠**: `RuntimePublicationOrchestrator::trySubmit()` は確かに以下の完全な publish シーケンスを保持:

```
Admission → Build → Publish → Crossfade Decision → Transition → Epoch advance
```

しかし `trySubmit()` 内で `engine_.getActiveRuntimeDSP()` を呼び出しており（line 53）、Semantic Decision 層に AudioEngine の旧 API が混入している。また `engine_.resolveDSPHandle()` (line 52) も AudioEngine 経由。

**評価**: Coordinator は Facade ではないが（実処理を持っている）、依然 AudioEngine への back-reference を判断ロジックに使っている。

---

## ③ Publication と DSP Lifetime 密結合

### 判定: ⚠️ PARTIALLY CORRECT

**証拠**: `PublicationExecutor::publish()` は publish のみを行い activate/retire を行わない:

```cpp
PublishResult PublicationExecutor::publish(...) {
    coordinator.publishWorld(std::move(worldOwner));
    return PublishResult::Success;
}
```

しかし `RuntimePublicationOrchestrator::trySubmit()` では publish 成功直後に:

```cpp
transition_.onPublishCompleted(newDSPResolved, oldDSP, cfDecision, lifetime_);
```

と同一関数内で publish + DSP ライフサイクルを連続実行している。

`PublicationResult` 構造体は未導入。成功/失敗の区別は `PublishResult` enum で行っているが、DSPTransition への結果伝達は戻り値ではなく同一関数内の逐次呼び出し。

---

## ④ RuntimeWorld/DSPCore 二重モデル

### 判定: ✅ CORRECT

**証拠**: Decision 層（`RuntimePublicationOrchestrator::trySubmit()`）で:

```cpp
auto* oldDSP = engine_.getActiveRuntimeDSP();     // DSPCore 直接取得
auto* oldWorld = engine_.runtimeStore.observe();  // RuntimeWorld 経由
```

両方の経路から意味情報を取得可能。`CrossfadeAuthority` は `dspProjection` のみ使用するが、呼び出し元の Orchestrator では両方の経路が利用可能。

---

## ⑤ Observe Path 多重化

### 判定: ✅ CORRECT

**証拠**: 以下の複数経路を確認:

- `RuntimeStore::observe()` → `RuntimePublishWorld`（coordinator 経由）
- `getActiveRuntimeDSP()` → `DSPCore*`（AudioEngine 直接）
- `resolveFadingRuntimeDSPFromRuntimeWorldOnly()` → 間接経由
- `resolveActiveRuntimeDSPFromRuntimeWorldOnly()` → 間接経由

---

## ⑥ Publish決定権が AudioEngine 側

### 判定: ✅ CORRECT

**証拠**: `PublicationAdmission::evaluate()` は Coordinator 内にあるが、各 rebuild dispatch 箇所（`AudioEngine.RebuildDispatch.cpp`）で:

```cpp
// line 226, 422, 465
if (!acceptsRuntimePublication())
```

が複数存在する。`acceptsRuntimePublication()` は `AudioEngine` のメソッドであり、Admission の判断を AudioEngine が先行して行っている。

また `acceptsRuntimePublication()` の実装:

```cpp
// AudioEngine.Commit.cpp:113
bool AudioEngine::acceptsRuntimePublication() const noexcept
```

は AudioEngine の内部状態（shutdown phase 等）を直接参照しており、Admission の判断が Coordinator ではなく Engine に残っている。

---

## ⑦ Crossfade Authority 二重化

### 判定: ⚠️ OUTDATED

**証拠**: `CrossfadeAuthority::evaluate()` のみが `dspProjection` を使用して判断。

```bash
grep "CrossfadeContext\|computeCrossfadeContext" src/ → 0件
```

CrossfadeContext / computeCrossfadeContext は完全に削除済み。Crossfade 判断は CrossfadeAuthority のみ。

ただし `DSPTransition::onPublishCompleted()` 内で crossfade 登録 (`registerCrossfade`) を行っており、ドキュメントが指摘する「判断の二重化」は解消済みだが、「登録の分散」は残っているわけではない（登録＝DSPTransition のみ）。

---

## ⑧ Semantic/Execution 分離未完成

### 判定: ✅ CORRECT

**証拠**: `RuntimeBuilder.cpp` lines 230-235:

```cpp
worldOwner->dspProjection.irLoaded = current->convolverRt().isIRLoaded();        // DSPCore 直接
worldOwner->dspProjection.irFinalized = current->convolverRt().isIRFinalized();  // DSPCore 直接
worldOwner->dspProjection.structuralHash = current->convolverRt().getStructuralHash(); // DSPCore 直接
worldOwner->dspProjection.baseLatencySamples = engine.estimateRuntimeLatencyBaseRateSamples(current, false); // DSPCore 直接
```

Builder が `sealedSnapshot` から値を取る経路もあるが（RuntimeBuilder.cpp:222-227 に `if (sealedSnapshot != nullptr)` 分岐）、current DSP から直読する経路も残っている。`RuntimePublicationOrchestrator::trySubmit()` では `sealedSnapshot` を渡しているので snapshot 経路が使用されるが、他の呼び出し元（例: ReleaseResources, PrepareToPlay）では current DSP 直読経路が使われる可能性がある。

---

## ⑨ RuntimeWorld Construction Authority

### 判定: ❌ OUTDATED / INCORRECT（すでに封鎖済み）

**証拠**: `AudioEngine.h` lines 121-144:

```cpp
struct RuntimeState : convo::isr::SealedObject<RuntimeState> {
    struct BuilderToken {
    private:
        friend class AudioEngine;
        friend class convo::RuntimeBuilder;   // ★ Builder のみ生成可能
        constexpr BuilderToken() noexcept = default;
    };
    RuntimeState() = delete;                            // デフォルト構築禁止
    explicit RuntimeState(BuilderToken) noexcept {...}   // Token 必須
    static auto createForBuilder(BuilderToken token) noexcept { ... }
};

using RuntimePublishWorld = RuntimeState;
static_assert(!std::is_default_constructible_v<RuntimePublishWorld>,
              "RuntimePublishWorld must not be default-constructible outside builder path");
```

また `AudioEngine.h:319`:

```cpp
friend class convo::RuntimeBuilder;
```

**検証**:

```bash
grep "new RuntimePublishWorld\|make_unique<RuntimePublishWorld>" src/ → 0件
```

RuntimePublishWorld の生成経路は完全に Builder に封鎖済み。

---

## 🔴 NEW FINDING: `acceptsRuntimePublication()` の分散判断

**発見ツール**: grep / ccc / CodeGraph

**問題点**: `acceptsRuntimePublication()` は AudioEngine のメンバ関数として実装され、以下の箇所で呼ばれている:

| 呼び出し元 | 行 | 問題 |
|-----------|-----|------|
| `AudioEngine.RebuildDispatch.cpp` | 226, 422, 465 | RebuildDispatch が直接 Admission bypass |
| `AudioEngine.Commit.cpp` | 183 | Commit path 内 |

本来は `PublicationAdmission::evaluate()` が全ての Admission 判断を行うべきだが、RebuildDispatch 側で先行判断している。

---

## 🔴 NEW FINDING: `DSPTransition::onTransitionComplete()` 内の DSPLifetimeManager 局所生成

**発見ツール**: 実ファイル直接読取

**問題点**: `DSPTransition::onTransitionComplete()` (DSPTransition.h line 112):

```cpp
DSPLifetimeManager lifetime(engine_);   // ★ 毎回ローカル生成
lifetime.retire(done);
```

メンバとして持っている `RuntimePublicationOrchestrator` の `lifetime_` を使うべきだが、DSPTransition が自分でローカル生成している。これは `RuntimePublicationOrchestrator` の lifetime_ と二重管理になる可能性がある。

---

## 🔴 NEW FINDING: `dspCrossfadeUseDryAsOld` / `firstIrDryCrossfadePending` の二重 publish

**発見ツール**: 実ファイル直接読取

**問題点**: `DSPTransition::onPublishCompleted()` で以下の publish が Crossfade 経路と非 Crossfade 経路の両方で行われている:

```cpp
// Crossfade 経路（line 93-94）:
publishAtomic(engine_.dspCrossfadeUseDryAsOld, false, ...);
publishAtomic(engine_.firstIrDryCrossfadePending, false, ...);

// 非 Crossfade 経路（line 100-101）:
publishAtomic(engine_.dspCrossfadeUseDryAsOld, false, ...);
publishAtomic(engine_.firstIrDryCrossfadePending, false, ...);
```

コード重複。`CrossfadeRuntime` への抽出で解消可能。

---

## 🔴 NEW FINDING: `CrossfadeAuthority` が `engine` 参照を受ける設計

**発見ツール**: 実ファイル直接読取

**問題点**: `CrossfadeAuthority::evaluate()` の引数:

```cpp
Decision evaluate(const AudioEngine& engine, ...) noexcept;
```

Authority の判断に engine 参照が必要ということは、Authority が完全に独立していない証拠。engine はフェード時間設定 (atomic) の読み取りに使われているが、これらも `RuntimeWorld` の projection に含めるべき。

---

## 総括: 検証結果マトリクス

| # | ドキュメントの指摘 | 優先度 | 判定 | 備考 |
|---|-------------------|--------|------|------|
| S1 | DSPLifetimeManager ラッパー | S | ✅ CORRECT | Phase-A 維持 |
| S2 | DSPCore* 依存 | S | ✅ CORRECT | Execution Path 全体 |
| S3 | getActiveRuntimeDSP() Leakage | S | ✅ CORRECT | **Orchestrator で使用中** |
| A1 | Shutdown Path 旧経路 | A | ✅ CORRECT | releaseResources 直接使用 |
| A2 | Crossfade State in AudioEngine | A | ✅ CORRECT | 8個の atomic が残存 |
| A3 | RuntimeStore Publish Authority | A | ⚠️ PARTIALLY | Coordinator 封鎖は完了 |
| B1 | DSPTransition 肥大化 | B | ✅ CORRECT | 52行/5責務 |
| B2 | Projection Coverage | B | ⚠️ PARTIALLY | 実装済み、Contract 未同期 |
| ① | Legacy Commit Path | S | ❌ **OUTDATED** | すでに削除完了 |
| ② | Coordinator Facade | S | ⚠️ PARTIALLY | 実処理はあるが engine 依存 |
| ③ | Publ. & Lifetime 密結合 | S | ⚠️ PARTIALLY | Executor 分離済、結果オブジェクト未導入 |
| ④ | World/DSPCore 二重 | A | ✅ CORRECT | 両経路から読取可能 |
| ⑤ | Semantic/Execution 分離未完成 | A | ✅ CORRECT | Builder 直読経路残存 |
| ⑥ | Publish決定権 Engine側 | A | ✅ CORRECT | acceptsRuntimePublication 残存 |
| ⑦ | Crossfade Authority 二重化 | A | ⚠️ **OUTDATED** | 解消済み |
| ⑧ | Observe Path 多重化 | A | ✅ CORRECT | 複数読取経路 |
| ⑨ | Construction Authority | B | ❌ **OUTDATED** | BuilderToken + friend 完備 |

### 実装の棚卸し：改修が必要な箇所

#### Phase-A レベル（S 優先度で実装が追いついていないもの）

1. **`RuntimePublicationOrchestrator::trySubmit()` の `getActiveRuntimeDSP()` 排除**
   - ファイル: `src/audioengine/RuntimePublicationOrchestrator.cpp` line 53
   - 現状: `auto* oldDSP = engine_.getActiveRuntimeDSP();`
   - 代替案: `runtimeStore.observe()` 経由で Projection から取得、または LifetimeManager から Handle 取得

2. **`acceptsRuntimePublication()` の Admission 移行**
   - ファイル: `src/audioengine/AudioEngine.RebuildDispatch.cpp` lines 226, 422, 465
   - 現状: RebuildDispatch が直接 AudioEngine の状態を確認
   - 代替案: `PublicationAdmission::evaluate()` に全判断を集約

3. **Shutdown Path の DSPLifetimeManager 経由化**
   - ファイル: `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` lines 94-172
   - 現状: `getActiveRuntimeDSP()`, `setActiveRuntimeDSP()`, `exchangeFadingRuntimeDSP()`, `retireDSP()` を直接使用
   - 代替案: `lifetimeManager.shutdown()` 一本化

4. **DSPTransition の crossfade atomic 直接アクセス排除**
   - ファイル: `src/audioengine/DSPTransition.h` lines 89-101
   - 現状: `engine_.dspCrossfadeGain`, `engine_.dspCrossfadePending` 等を直接 publish
   - 代替案: `CrossfadeRuntime` クラス抽出

#### Phase-B レベル（推奨）

1. **DSPLifetimeManager への状態保持追加**
   - ファイル: `src/audioengine/DSPLifetimeManager.h`
   - 現状: `activeHandle_` / `fadingHandle_` なし
   - 代替案: Handle 保持 + `getActiveHandle()` API 追加

2. **DSPTransition::onTransitionComplete() の lifetime ローカル生成排除**
   - ファイル: `src/audioengine/DSPTransition.h` line 116
   - 現状: 毎回 `DSPLifetimeManager lifetime(engine_)` を生成
   - 代替案: Coordinator の `lifetime_` を参照

3. **dspCrossfadeUseDryAsOld / firstIrDryCrossfadePending の重複 publish 統合**
   - ファイル: `src/audioengine/DSPTransition.h` lines 93-94, 100-101

#### Phase-C レベル

1. **PublicationResult 構造体導入による Publication / DSP Lifetime の結果連携**
2. **CrossfadeAuthority の engine 依存排除（フェード時間設定も Projection へ）**

---

## 結論

`doc/work20/notfinished6.md` は全体として高品質な分析であり、**19 の指摘中 12 が妥当（CORRECT/PARTIALLY CORRECT）** でした。しかし **4 つの指摘（Legacy Commit Path、Crossfade Authority 二重化、Construction Authority、CrossfadeContext 重複）は現在の実装ではすでに解消済み** であり、ドキュメントが古い情報に基づいていることを示しています。

最も重大な未解決課題は **「getActiveRuntimeDSP() の Decision 層での使用」**（S3）と **「acceptsRuntimePublication() の Engine 側残存」**（追加発見）であり、これらは AudioEngine が依然として Runtime Authority の一部を保持していることを示しています。
