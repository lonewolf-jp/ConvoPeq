# notfinished6.md 妥当性検証レポート

**調査日**: 2026-06-06
**調査者**: GitHub Copilot (AI Assistant)
**調査手法**: Serena MCP (シンボル/パターン検索) + ccc (意味検索) + CodeGraph MCP (関数呼び出し関係) + grep (テキスト検索)

---

## 総括

| 分類 | 項目 | 件数 |
| --- | --- | --- |
| ✅ **文書の主張は正確（現在も残存）** | ③④⑤⑥⑦⑧⑩⑫⑬⑭⑮⑯⑰⑱⑲⑳ | **16項目** |
| ⚠️ **部分的に改善済み** | ②⑪ | **2項目** |
| ❌ **文書の主張は誤り（既に修正/実装済み）** | ①⑨ | **2項目** |

---

## ① Legacy Commit Path がまだ生存している

**判定: ❌ 文書の主張は誤り（既に削除済み）**

### 文書の主張

- `processPendingCommit()` → `applyRuntimeCommitFromIntent()` が依然として主経路
- `pendingCommit_` / `pendingCommitFlag_` / `pendingCommitMutex_` が残存

### 検証結果

| シンボル | ソースコード上の存在 | 備考 |
| --- | --- | --- |
| `processPendingCommit()` | **存在しない** | 全検索ツールで確認不可 |
| `applyRuntimeCommitFromIntent()` | **存在しない** | 同上 |
| `pendingCommit_` | **存在しない** | 同上 |
| `pendingCommitFlag_` | **存在しない** | コメントのみ `AudioEngine.Threading.cpp:304` に残存 |
| `pendingCommitMutex_` | **存在しない** | 全検索ツールで確認不可 |

### 実際の状況

主経路は `RuntimePublicationOrchestrator::submitPublishRequest()` に移行済み。PR-3 適用により削除されたと推定される。

---

## ② Coordinator が Executor Authority になっていない

**判定: ⚠️ 部分的に改善済みだが不完全**

### 文書の主張

- `coordinator.publishWorld(...)` が呼び出し先として使われているだけ
- Executor権威になっていない

### 検証結果

`RuntimePublicationOrchestrator` は存在し、内部で下記の流れを実装:

```
submitPublishRequest()
  → trySubmit()
    → admission_.evaluate()
    → executor_.publish()
    → transition_.onPublishCompleted()
```

しかし `coordinator.publishWorld()` は **8箇所以上** から直接呼ばれている:

| 呼び出し元 | ファイル・行 |
| --- | --- |
| `AudioEngine.Commit.cpp:617` | `enqueuePublicationIntentForRuntimeCommit` 内 |
| `AudioEngine.Init.cpp:53` | ブートストラップ |
| `AudioEngine.Processing.PrepareToPlay.cpp:136, 248` | prepareToPlay 経路 |
| `AudioEngine.Processing.ReleaseResources.cpp:133` | releaseResources 経路 |
| `AudioEngine.Timer.cpp:410` | タイマー経路 |
| `DSPTransition.h:102` | DSPTransition 内 |
| `PublicationExecutor.cpp:24` | PublicationExecutor の publish() 内 |

また `AudioEngine::publishWorld()` ラッパー（`AudioEngine.h:2725`）は**デッドコード**（呼び出し元ゼロ）。

---

## ③ Crossfade Decision が完全に RuntimeWorld 化されていない

**判定: ✅ 文書の主張は正しい**

### 検証結果

| API | 引数 | 状態 |
|-----|------|------|
| `evaluateOnly(AudioEngine&, DSPCore*, DSPCore*)` | DSPCore ベース | **残存・使用中** |
| `evaluateAndRegister(AudioEngine&, DSPCore*, DSPCore*, DSPHandle, DSPHandle)` | DSPCore ベース | **残存** |
| `evaluateFromWorlds(const AudioEngine&, const RuntimePublishWorld&, const RuntimePublishWorld&)` | RuntimeWorld ベース | **存在するが未使用** |
| `computeDecision(const AudioEngine&, const DSPCore*, const DSPCore*)` | DSPCore ベース | **残存** |

`RuntimePublicationOrchestrator::trySubmit()`（`RuntimePublicationOrchestrator.cpp:34`）内で `evaluateOnly(DSPCore*)` が使用されており、`evaluateFromWorlds()` は使用されていない。

---

## ④ DSPCore が Semantic Authority に残っている

**判定: ✅ 文書の主張は正しい**

`CrossfadeAuthority::computeDecision()`（`CrossfadeAuthority.cpp:31`）は DSPCore を直読。`evaluateOnly()`/`evaluateAndRegister()` も DSPCore 依存。

---

## ⑤ DSPLifetimeManager が単なるラッパー

**判定: ✅ 文書の主張は正しい**

`DSPLifetimeManager.h` の実装（確認済み）:

```cpp
void activate(DSPCore* dsp) { engine_.setActiveRuntimeDSP(dsp); }
void retire(DSPCore* dsp) { engine_.retireDSP(dsp); }
DSPCore* getActive() { return engine_.getActiveRuntimeDSP(); }
CrossfadeId beginCrossfade(from, to) { return engine_.dspHandleRuntime_.beginCrossfade(from, to); }
```

すべて `engine_` への委譲のみ。真の分離は未達成。

---

## ⑥ AudioEngine が Runtime Authority を保持している

**判定: ✅ 文書の主張は正しい**

### `getActiveRuntimeDSP()` 使用箇所（ソースコード内）

| ファイル | 行 | 用途 |
| --- | --- | --- |
|---------|-----|------|
| `AudioEngine.CtorDtor.cpp` | 59, 69, 79 | デストラクタ検証 |
| `AudioEngine.Processing.Latency.cpp` | 83 | レイテンシ処理 |
| `AudioEngine.Processing.PrepareToPlay.cpp` | 243 | prepareToPlay |
| `AudioEngine.Processing.ReleaseResources.cpp` | 93, 99, 137, 171 | リソース解放 |
| `DSPLifetimeManager.h` | 35 | getActive() |
| `DSPTransition.h` | 30 | トランジション処理 |
| `RuntimePublicationOrchestrator.cpp` | 30 | Orchestrator |

### `setActiveRuntimeDSP()` 使用箇所

| ファイル | 行 |
| --- | --- |
|---------|-----|
| `AudioEngine.CtorDtor.cpp` | 72 |
| `AudioEngine.Processing.PrepareToPlay.cpp` | 235 |
| `AudioEngine.Processing.ReleaseResources.cpp` | 102 |
| `DSPLifetimeManager.h` | 16 |

---

## ⑦ Fading DSP 管理が旧Runtime構造

**判定: ✅ 文書の主張は正しい**

### `exchangeFadingRuntimeDSP()` 使用箇所

| ファイル | 行 |
| --- | --- |
|---------|-----|
| `AudioEngine.CtorDtor.cpp` | 74 |
| `AudioEngine.Processing.ReleaseResources.cpp` | 105 |
| `AudioEngine.Timer.cpp` | 386, 418 |
| `DSPTransition.h` | 42, 81 |

---

## ⑧ PublicationExecutor が実質薄い委譲

**判定: ✅ 文書の主張は概ね正しい**

`PublicationExecutor::publish()`（`PublicationExecutor.cpp:24`）は内部で `coordinator.publishWorld()` に委譲している。

---

## ⑨ RuntimeWorld Immutability が未完成

**判定: ❌ 文書の主張は部分的に不正確**

### 検証結果

| 機能 | 状態 |
| --- | --- |
|------|------|
| `sealRecursively()` | **実装済み**（`ISRSealedObject.h:54`） |
| `isSealed()` | **実装済み**（`ISRSealedObject.h:66`） |
| `isSealedRecursively()` | **実装済み**（`ISRSealedObject.h:73`） |
| publish前のseal | **使用中**（`RuntimePublicationCoordinator.h:94` で `worldOwner->sealRecursively()`） |
| sealチェック | **使用中**（`AudioEngine.Commit.cpp:235` で `world.isSealedRecursively()`） |

文書は「sealRecursively/freeze/isSealed 未整備」と主張しているが、`sealRecursively` は実装・使用ともに完了している。ただし `freeze` という名前の関数は見つからず（`assertMutable` 相当のものは存在する可能性あり）、その点では文書の指摘に一部妥当性がある。

---

## ⑩ RuntimeBuildSnapshot Authority が完全確立していない

**判定: ✅ 文書の主張は概ね妥当**

`dspProjection` は `RuntimeBuilder.cpp:219-224` で DSPCore から値をコピー:

```cpp
worldOwner->dspProjection.irLoaded = current->convolverRt().isIRLoaded();
worldOwner->dspProjection.irFinalized = current->convolverRt().isIRFinalized();
worldOwner->dspProjection.structuralHash = current->convolverRt().getStructuralHash();
worldOwner->dspProjection.oversamplingFactor = static_cast<int>(current->oversamplingFactor);
worldOwner->dspProjection.sampleRate = current->sampleRate;
worldOwner->dspProjection.baseLatencySamples = engine.estimateRuntimeLatencyBaseRateSamples(current, false);
```

Builder での投影自体は責務上妥当だが、完全な分離（Snapshot Only Authority）には至っていない。

---

## ⑪ Publication Success 前に DSP Activation が発生し得る設計痕跡

**判定: ⚠️ RuntimePublicationOrchestrator は正しい順序を守っている**

`trySubmit()` の実装順序:

1. Admission → 2. Build → 3. Publish → 4. **Success確認** → 5. activate (`transition_.onPublishCompleted`) → 6. epoch advance

文書の懸念は現在のコードでは解消済み。

---

## ⑫ Warmup が Publication Pipeline の外にある

**判定: ✅ 文書の主張は妥当**

`executeWarmup` という関数は存在しないが、`RuntimeBuilder::validateWarmup()` が `RebuildDispatch.cpp:789` で呼ばれている。Warmup 専用コンポーネントへの分離は未達成。

---

## ⑬ Retire Epoch Authority が Coordinator に集約されていない

**判定: ✅ 文書の主張は正しい**

`advanceRetireEpoch()` は `AudioEngine` のメソッドとして残存（`AudioEngine.Threading.cpp:44`）。
`RuntimePublicationOrchestrator::trySubmit()` 内で `engine_.advanceRetireEpoch()` を直接呼んでいる。
Coordinator/Executor 管理下にない。

---

## ⑭ Latency Adjustment が Publication の副作用として残存

**判定: ✅ 文書の主張は妥当**

`estimateRuntimeLatencyBaseRateSamples()` は `AudioEngine` のメソッドとして残存（`AudioEngine.h:3059`）。
`RuntimeBuilder.cpp:224` で `engine.estimateRuntimeLatencyBaseRateSamples(current, false)` として呼ばれる。

---

## ⑮ UI通知が Publication Pipeline に混在

**判定: ✅ 文書の主張は正しい**

| 関数 | 使用箇所数 | 主要ファイル |
| --- | --- | --- |
| `sendChangeMessage()` | 10箇所以上 | `Parameters.cpp`, `StateIO.cpp`, `Timer.cpp`, `UIEvents.cpp`, `EQProcessor.Core.cpp` 他 |
| `triggerAsyncUpdate()` | 2箇所 | `RebuildDispatch.cpp:339`, `Timer.cpp:429` |
| `enqueueLearningCommand()` | 4箇所 | `Learning.cpp`, `Timer.cpp`, `UIEvents.cpp` |

---

## ⑯ Deferred Publication Queue が RuntimeStore Authority になっていない

**判定: ✅ 文書の主張は正しい**

`deferredRequest_` は `PublicationAdmission.h` 内のローカル状態（`PublicationAdmission.h:53`）:

```cpp
std::optional<PublishRequest> deferredRequest_;
bool hasDeferred_ = false;
```

RuntimeStore 管理下にはない。文書が主張する「Coordinator内部」からは `PublicationAdmission` に移動済みだが、Store管理には至っていない。

---

## ⑰ PublicationRequest が DSPCore を直接保持

**判定: ✅ 文書の主張は正しい**

```cpp
struct PublishRequest {
    void* newDSP = nullptr;  // AudioEngine::DSPCore*
    int generation = 0;
    RuntimeBuildSnapshot sealedSnapshot;
};
```

DSPCore raw pointer (`void*`) を保持。文書の指摘通り。

---

## ⑱ RuntimeBuilder が DSPCore 依存のまま

**判定: ✅ 文書の主張は正しい**

```cpp
buildRuntimePublishWorld(
    AudioEngine::DSPCore* current,   // ← DSPCore 直参照
    AudioEngine::DSPCore* next,
    convo::TransitionPolicy policy,
    double fadeTimeSec,
    bool active,
    const convo::RuntimeBuildSnapshot* sealedSnapshot);
```

11箇所から呼ばれている。

---

## ⑲ Crossfade Registration Authority が二重化

**判定: ✅ 文書の主張は正しい（発見）**

### 2つの経路を確認

**経路1: CrossfadeAuthority 経由**

```
CrossfadeAuthority::doRegister()
  → engine.crossfadeAuthorityRuntime_.registerCrossfade(from, to)
```

（`CrossfadeAuthority.cpp:94`）

**経路2: DSPTransition 直接（CrossfadeAuthority をバイパス）**

```
DSPTransition.h:35
  → engine_.crossfadeAuthorityRuntime_.registerCrossfade(oldHandle, newHandle)
```

（`DSPTransition.h:35`）

---

## ⑳ Publish Entry Point が完全統一されていない

**判定: ✅ 文書の主張は正しい**

`coordinator.publishWorld()` への直接呼び出しは8箇所以上:

- `AudioEngine.Commit.cpp:617`
- `AudioEngine.Init.cpp:53`
- `AudioEngine.Processing.PrepareToPlay.cpp:136, 248`
- `AudioEngine.Processing.ReleaseResources.cpp:133`
- `AudioEngine.Timer.cpp:410`
- `DSPTransition.h:102`
- `PublicationExecutor.cpp:24`
- テストファイル: `PartialPublicationRejectTests.cpp` (7箇所), `RuntimePublicationCoordinatorTests.cpp` (4箇所)

---

## 文書未記載の追加発見事項

### A. `publishRuntimeStateNonRt()` はデッドコード

- 宣言: `AudioEngine.h:2718`
- 定義: `AudioEngine.Commit.cpp:602`
- **呼び出し元: ゼロ**
- 関連項目: ⑧

### B. `AudioEngine::publishWorld()` ラッパーもデッドコード

- 宣言: `AudioEngine.h:2725`
- **呼び出し元: ゼロ**（全員直接 `coordinator.publishWorld()` を使用）
- 関連項目: ⑳

### C. `dspProjection` の値は未だに DSPCore からコピー

- `RuntimeBuilder.cpp:219-224` で5フィールドを DSPCore から直接読み取り
- `CrossfadeAuthority::evaluateFromWorlds()` は `dspProjection` を使用可能だが未使用
- 関連項目: ④⑩⑱

### D. `CrossfadeAuthority::evaluateOnly()` が Orchestrator の主経路

- `RuntimePublicationOrchestrator.cpp:34` で使用
- `evaluateFromWorlds()` は実装されているが一度も呼ばれていない
- 関連項目: ③④

### E. `setMixedPhaseState` が ConvolverProcessor 内で多数使用

- `ConvolverProcessor.MixedPhase.cpp` 全体で10箇所以上
- Publication pipeline 外での状態変更
- 関連項目: ⑮

### F. `commitNewDSP` / `prepareCommit` / `executeCommit` は削除済み

- アクティブなソースコード上には存在せず
- CodeQLデータベースのバックアップ（`storage/`, `.musubi/`）にのみ残存
- 関連項目: ①

---

## レビューフィードバックによる改訂（2026-06-06）

本レポートに対する第三者レビューを反映し、以下の改訂を行う。

### 誤判定だった項目の撤回

#### ① Legacy Commit Path 残存 — **撤回**

調査結果により `processPendingCommit` / `applyRuntimeCommitFromIntent` / `pendingCommit_` / `pendingCommitFlag_` / `pendingCommitMutex_` は全削除済み。旧Commitモノリスから `RuntimePublicationOrchestrator` への移行は完了している。「Legacy Commit Path残存」ではなく「Publication Entry Pointの統一未完」へ論点修正すべき。

#### ⑨ RuntimeWorld Immutability 未実装 — **撤回**

`sealRecursively()` / `isSealed()` / `isSealedRecursively()` が存在し、`worldOwner->sealRecursively()` まで実行済み。「Immutability未実装」ではなく「Immutability実装済み、ただし運用上の完全強制が十分かは別監査」が正しい評価。

### 総括表の更新

| 分類 | 項目 | 件数 |
| --- | --- | --- |
| ✅ **文書の主張は正確（現在も残存）** | ③④⑤⑥⑦⑧⑩⑫⑬⑭⑮⑯⑰⑱⑲⑳ | **16項目** |
| ⚠️ **部分的に改善済み** | ②⑪ | **2項目** |
| ❌ **文書の主張は誤り（撤回すべき）** | ①⑨ | **2項目** |

### 優先度再編成

レビューにより「真に重要な未達成事項」を優先度A/B/Cに再編成する。

#### 優先度A — ISR Bridge Runtime 完成に必須（7項目）

DSPCore Authority → RuntimeWorld Authority への最後の移行区間。

| # | 項目 | 本質 | 現状 |
| --- | --- | --- | --- |
| A1 | `evaluateFromWorlds()` 未使用 | **RuntimeWorld Authority 未成立** | `evaluateOnly(DSPCore*)` が主経路。`evaluateFromWorlds()` は死んでいる |
| A2 | CrossfadeAuthority DSPCore依存 | RuntimeWorld ではなく DSPCore が判断権威 | `computeDecision(DSPCore*, DSPCore*)` が残存 |
| A3 | RuntimeBuilder DSPCore依存 | DSPCore → RuntimeWorld の投影構造が逆転 | `buildRuntimePublishWorld(DSPCore*, DSPCore*, ...)` が11箇所 |
| A4 | dspProjection DSPCore直読 | DSPCore が Authority であり続けている | `current->convolverRt().isIRLoaded()` 等を RuntimeBuilder が直読 |
| A5 | PublishRequest が DSPCore 保持 | Authority が二重（DSP Pointer + Snapshot） | `struct PublishRequest { void* newDSP; ... }` |
| A6 | publishWorld() 直接呼び出し散在 | Single Publication Entry 未達成 | 8箇所以上から `coordinator.publishWorld()` を直呼 |
| A7 | Crossfade Registration 二重化 | Authority が二重 | DSPTransition が CrossfadeAuthority をバイパスして直接 `registerCrossfade()` |

#### 優先度B — 重要だがAよりはリスクが低い（3項目）

| # | 項目 | 本質 |
| --- | --- | --- |
| B1 | Retire Epoch Authority の移譲 | `engine_.advanceRetireEpoch()` — Coordinator が権威でない |
| B2 | DSPLifetimeManager 実体化 | 現状は `AudioEngine` の façade に過ぎない |
| B3 | Deferred Queue Store統合 | `PublicationAdmission` ローカル状態のまま |

#### 優先度C — 純化・整理（4項目）

| # | 項目 |
| --- | --- |
| C1 | Latency Service 分離 |
| C2 | Warmup Service 分離 |
| C3 | UI通知のイベント化 |
| C4 | デッドコード整理（`publishRuntimeStateNonRt()` / `AudioEngine::publishWorld()`） |

### 完成度評価の見直し

レビュー結論を反映し、Practical Stable ISR Bridge Runtime の完成度を **85〜90%** に上方修正する。

**根拠**:

- Legacy Commit Path が既に除去済み（旧評価では未達成と想定）
- RuntimeWorld Immutability が既に実装済み（同上）
- 残件の中心が「旧モノリス除去」から「Authority の純化」に移行している

**現在の最大課題**:
機能不足ではなく、**DSPCore Authority → RuntimeWorld Authority への最後の移行区間**。
