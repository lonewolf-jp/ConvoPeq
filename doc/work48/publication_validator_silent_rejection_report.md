# 検証報告: サードパーティバグレビューの妥当性評価とHEAD/Working Tree乖離分析

**ドキュメントID**: WR-48-002
**作成日**: 2026-06-19 (v2.0: HEAD vs Working Tree 乖離分析を追加)
**対象**: ConvoPeq main (HEAD commit) + Working Tree 未コミット変更
**深刻度**: Medium（HEAD時点では無害。Working Tree commit 後は High）
**総合妥当性**: 85-95%（実装の大部分は採用候補レベル。特に Silent Success 修正は最優先）
**検出経路**: サードパーティレビュー + grep/Serena/CodeGraph/Graphify/ccc/semble による静的検証

---

## 1. バグの概要（Working Tree 分析）

> **重要**: 本章は Working Tree（未コミット変更を含む）の分析である。
> HEAD（コミット済み）では Validator が Placeholder のため、本章の問題は顕在化しない。
> 詳細は §2 参照。

`RuntimePublicationValidator` が、**正当な `RuntimePublishWorld` を過剰に厳格な条件で「不正」と判定し、破棄する**。さらに、破棄された事実が呼び出し元に伝わらない設計（Silent Success）により、以下の致命的な症状を引き起こす：

1. **Bootstrap World が出版（publish）されない**
   - 初期化時に生成される Bootstrap World（generation=1, runtimeUuid=0）が `validateTopology` で拒否される
   - ⚠ **ただし、Bootstrap拒否単独では永久無音にはならない**（§5.9 参照）
   - `prepareToPlay()` の placeholder DSP 経路が runtimeUuid>0 の世界を発行できるため、初期化後は回復可能

2. **特定の dither/noiseShaper 設定で出版が常に失敗する**
   - ditherBitDepth=32 を設定していると `validateResources` で拒否
   - `NoiseShaperType::Fixed15Tap(3)` を使用していると `validateResources` で拒否
   - これらの設定はアプリケーションの正規設定であり、Signal Path で実際に使用されている

3. **Validation 失敗が呼び出し元に伝播しない — これが真の根本原因**
   - `PublicationExecutor::publish()` が常に `PublishResult::Success` を返す
   - `trySubmit()` のエラーハンドリングが到達不能
   - ログには "SUCCEEDED" と出力されるが、Audio Thread には null World しか届かない
   - **この問題が解決されない限り、Bootstrap/dither/NS の reject も含め、あらゆる validation 失敗が隠蔽される**

### 症状一覧（Working Tree commit 後の想定）

> **注記**: 「無音になる」は正確には「無音化リスクがある」。ConvoPeq は Bootstrap World フォールバック、
> Placeholder DSP、`makeEngineRuntimeState` の null-world fallback など複数の救済経路を持つため、
> 必ずしも永久無音になるとは限らない（§5.9 参照）。

| 症状 | 発生条件 | 確率 | 備考 |
|---|---|---|---|
| 起動時クラッシュ（null dereference） | Release + 初回 rebuild | 低 | `makeEngineRuntimeState` の fallback で回避 |
| 無音出力リスク（-80dB, CPU 0.1%） | Bootstrap拒否 + prepareToPlay未実行 | 中 | prepareToPlay placeholder 経路からは回復可能 |
| dither 32-bit 設定で無音リスク | ユーザーが dither を 32 に設定 | 確実 | **Validator の実装バグ** |
| Fixed15Tap 選択で無音リスク | ユーザーが NS を Fixed15Tap に設定 | 確実 | **Validator の実装バグ** |
| Silent Success による全障害の隠蔽 | 全条件共通 | 確実 | **根本原因** — 修正最優先 |

---

## 2. 重大な発見: HEAD (コミット済み) と Working Tree の乖離

### 2.1 コードの現状

本報告書で分析対象としたコードは **Working Tree（未コミット変更を含む）** である。
コミット済みの HEAD とは大きく異なる。

| ファイル | HEAD (コミット済み) | Working Tree (未コミット) | 差分サイズ |
|---|---|---|---|
| `RuntimePublicationValidator.cpp` | 全 Validator が **Placeholder** (`return true`) | 完全な検証ロジック | +110行 |
| `RuntimePublicationValidator.h` | `ValidationFailureReason` なし | 完全な enum 定義 + `failureReason` フィールド | +40行 |
| `RuntimePublicationOrchestrator.cpp` | 旧来の crossfade 評価 | null world ガード + HealthState 抑制 + 診断ログ | +47行 |
| `PublicationValidatorIsolationTests.cpp` | 7テスト (129行) | 拡充されたテスト群 (+23テスト, 451行) | +368行 |
| `PublicationExecutor.cpp` | **変更なし** | **変更なし** | 0 |
| `RuntimePublicationCoordinator.h` | **変更なし** | **変更なし** | 0 |

### 2.2 HEAD 版 Validator の実態

HEAD 版の Validator は **事実上何も検査しない**：

```cpp
// HEAD 版: 全てが Placeholder
bool RuntimePublicationValidator::validateTopology(...) const {
    [[maybe_unused]] const auto& routing = world.routing;
    return true; // Placeholder
}

bool RuntimePublicationValidator::validateResources(...) const {
    [[maybe_unused]] const auto& resource = world.resource;
    return true; // Placeholder
}

bool RuntimePublicationValidator::checkNoConflictingTransitions(...) const {
    [[maybe_unused]] const auto& exec = world.execution;
    [[maybe_unused]] const auto& overlap = world.overlap;
    return true; // Placeholder
}
```

**したがって、HEAD 時点では以下の問題は存在しない：**

- ❌ Bootstrap World の topology reject → 存在しない（validateTopology が常に true を返す）
- ❌ dither=32 の resources reject → 存在しない（validateResources が常に true を返す）
- ❌ Fixed15Tap(3) の reject → 存在しない（同上）
- ❌ 過剰な transition conflict 検出 → 存在しない（checkNoConflictingTransitions が常に true）

### 2.3 例外: HEAD でも存在する問題

以下の問題は **HEAD にも Working Tree にも共通して存在する**：

| 問題 | ファイル | 影響 |
|---|---|---|
| `PublicationExecutor::publish()` が常に `Success` を返す | `PublicationExecutor.cpp:28` | Silent Success（validation 失敗を伝播できない） |
| `RuntimePublicationCoordinator::publishWorld()` が `void` 返却 | `RuntimePublicationCoordinator.h:114-147` | 成否が呼び出し元に伝わらない |
| `trySubmit()` のエラーハンドリングが到達不能 | `RuntimePublicationOrchestrator.cpp:175-188` | 失敗後の DSP activate が防げない |
| `runPublicationPrecheckNonRt()` 内の二重検証 | `AudioEngine.Commit.cpp:135` | 診断ログが Bridge で遮断される |

### 2.4 レビューの正当性評価（訂正版）

サードパーティレビューは **Working Tree の変更を前提として書かれている**。

| レビューの主張 | HEAD での成立 | Working Tree での成立 |
|---|---|---|
| Bootstrap World reject | ❌ 成立しない | ✅ 成立する（commit 後に発現） |
| dither=32 reject | ❌ 成立しない | ✅ 成立する（commit 後に発現） |
| Fixed15Tap(3) reject | ❌ 成立しない | ✅ 成立する（commit 後に発現） |
| Silent Success | ✅ 成立する | ✅ 成立する |
| trySubmit 失敗経路が到達不能 | ✅ 成立する | ✅ 成立する |
| publishWorld を bool 返却化 | ✅ 推奨 | ✅ 推奨 |

### 2.5 分析の意義

1. Working Tree の Validator 変更は **実装の完成形に向けた正しい方向性**である
2. しかし **dither/noiseShaper の許容値**と **Bootstrap の runtimeUuid** に関しては、Working Tree の変更にバグが含まれている
3. これらのバグは Working Tree を commit した瞬間に顕在化する
4. **commit 前に修正すべき**

### 原因1: Bootstrap World の拒否 — validateTopology

| 項目 | 内容 |
|---|---|
| **ファイル** | `src/audioengine/RuntimePublicationValidator.cpp` |
| **関数** | `RuntimePublicationValidator::validateTopology()` |
| **該当行** | 44-46 |
| **問題のコード** | `if (world.generation > 0 && topology.runtimeUuid == 0) return false;` |

```cpp
bool RuntimePublicationValidator::validateTopology(
    const RuntimePublishWorld& world) const
{
    const auto& topology = world.topology;

    // Bootstrap以外は runtimeUuid 必須
    if (world.generation > 0 && topology.runtimeUuid == 0)  // ← ここで Bootstrap を拒否
        return false;
    // ...
}
```

**発生日のトレース**:

```
AudioEngine::init()
  └→ RuntimeBuilder::createBootstrapWorld()
       ├→ reserveRuntimePublicationIdentity()
       │    └→ runtimeGenerationGenerator_.next() → 最初の呼出 = **1**
       ├→ worldOwner->generation = 1
       └→ worldOwner->topology.runtimeUuid = 0  // 明示的にゼロ設定
  └→ coordinator.publishWorld(bootstrapWorld)
       └→ bridge_.validatePublicationNonRt(*worldOwner)
            └→ RuntimePublicationValidator::validatePublication()
                 └→ validateTopology()
                      └→ generation=1 > 0 && runtimeUuid=0 → **REJECT**
```

**既存テストの問題**: `ValidateTopology_Bootstrap_Accept` は `generation=0` を仮定しているが、実際の Bootstrap World は `generation=1`。テストが現実と乖離している。

### 原因2: ditherBitDepth / noiseShaperType の過剰制限 — validateResources

| 項目 | 内容 |
|---|---|
| **ファイル** | `src/audioengine/RuntimePublicationValidator.cpp` |
| **関数** | `RuntimePublicationValidator::validateResources()` |
| **該当行** | 63-74 |
| **問題のコード** | dither: `if (dd != 0 && dd != 16 && dd != 24) return false;` |
| | noiseShaper: `if (ns < 0 || ns > 2) return false;` |

```cpp
bool RuntimePublicationValidator::validateResources(
    const RuntimePublishWorld& world) const
{
    // Dither: 0, 16, 24 のみ許容
    const int dd = resource.ditherBitDepth;
    if (dd != 0 && dd != 16 && dd != 24)  // ← 32がない！
        return false;

    // NoiseShaper: 0, 1, 2 のみ許容
    const int ns = resource.noiseShaperType;
    if (ns < 0 || ns > 2)  // ← 3 (Fixed15Tap)がない！
        return false;
    // ...
}
```

**根拠**:

- `kAdaptiveBitDepthValues = {16, 24, 32}` と定義済み（`AudioEngine.h:13`）
- `NoiseShaperType::Fixed15Tap = 3` は `DSPCoreDouble.cpp:621` 他で実際に使用
- UI パラメータ範囲: `Psychoacoustic(0)〜Fixed15Tap(3)`（`AudioEngine.Parameters.cpp:117`）

### 原因3: Validation 失敗の握り潰し — Silent Success

| 項目 | 内容 |
|---|---|
| **ファイル** | `src/core/RuntimePublicationCoordinator.h` |
| **関数** | `RuntimePublicationCoordinator::publishWorld()` |
| **該当行** | 114-147 |
| **問題** | `publishWorld()` が `void` 返却で、失敗時も何も返さない |

| 項目 | 内容 |
|---|---|
| **ファイル** | `src/audioengine/PublicationExecutor.cpp` |
| **関数** | `PublicationExecutor::publish()` |
| **該当行** | 14-27 |
| **問題のコード** | `coordinator.publishWorld()` の成否を無視し、常に `Success` を返す |

```cpp
// PublicationExecutor.cpp
PublishResult PublicationExecutor::publish(
    AudioEngine& engine,
    convo::aligned_unique_ptr<RuntimePublishWorld> worldOwner) noexcept
{
    if (!worldOwner)
        return PublishResult::PublishFailed;

    auto coordinator = engine.makeRuntimePublicationCoordinator();
    coordinator.publishWorld(std::move(worldOwner));  // ← void！成否不明

    return PublishResult::Success;  // ← 常に Success！
}
```

```cpp
// RuntimePublicationCoordinator.h — publishWorld の実体
void publishWorld(convo::aligned_unique_ptr<World> worldOwner) noexcept
{
    if (!worldOwner) return;

    worldOwner->sealRecursively();

    if constexpr (/* bridge has validatePublicationNonRt */)
    {
        if (!bridge_.validatePublicationNonRt(*worldOwner))
        {
            auto* rejectedWorld = worldOwner.release();
            bridge_.retireRuntimePublishWorldNonRt(rejectedWorld, false);
            return;  // ← void！何も伝えない
        }
    }
    // ... publishAndSwap ...
}
```

### 原因3b: trySubmit の到達不能エラーハンドリング

| 項目 | 内容 |
|---|---|
| **ファイル** | `src/audioengine/RuntimePublicationOrchestrator.cpp` |
| **関数** | `RuntimePublicationOrchestrator::trySubmit()` |
| **該当行** | 175-188 |

```cpp
auto result = executor_.publish(engine_, std::move(worldOwner));
if (result != PublishResult::Success) {
    // ★ このブロックは決して実行されない ★
    // publish() が常に Success を返すため
    if (!req.newDSP.isNull())
        lifetime_.retire(newDSPResolved);
    // ...
    return PublicationAdmission::Decision::RejectedShutdown;
}

// ★ Validation 失敗後もここに到達！
juce::Logger::writeToLog("[DIAG] trySubmit: executor_.publish SUCCEEDED gen="
    + juce::String(req.generation));

// Phase 3: DSP activate 実行
transition_.onPublishCompleted(newDSPResolved, oldDSP, cfDecision, lifetime_);
```

---

## 3. バグの詳細説明

### 3.1 問題の構造

```
RuntimePublicationOrchestrator::trySubmit()
  │
  ├→ RuntimeBuilder::buildRuntimePublishWorld()  → world 生成
  │
  ├→ executor_.publish(engine_, std::move(worldOwner))
  │    │
  │    ├→ coordinator.publishWorld(worldOwner)    ← void 返却
  │    │    │
  │    │    ├→ bridge_.validatePublicationNonRt()  ← ココで検証
  │    │    │    │
  │    │    │    ├→ Validator::validateTopology()   ← [原因1] Bootstrap拒否
  │    │    │    ├→ Validator::validateResources()  ← [原因2] 32/dither/NS3拒否
  │    │    │    │
  │    │    │    └→ 失敗 → world 解放 → void return
  │    │    │
  │    │    └→ void return（失敗が伝わらない）
  │    │
  │    └→ return PublishResult::Success           ← [原因3] 常にSuccess
  │
  ├→ 結果チェック (result != Success) → 決して真にならない
  │
  ├→ "[DIAG] trySubmit SUCCEEDED" をログ出力     ← 誤った成功報告
  │
  └→ transition_.onPublishCompleted()             ← DSP activate 実行
       RuntimeStore は nullptr のまま             ← Audio Thread 無音
```

### 3.2 なぜ「trySubmit SUCCEEDED」なのに無音なのか

1. `trySubmit()` は `executor_.publish()` から `Success` を受け取る
2. 実際には `publishWorld()` 内部で validation に失敗し、world は破棄されている
3. `RuntimeStore` のポインタは更新されず、nullptr のまま
4. Audio Thread が `observePublishedWorld()` を呼ぶと nullptr が返る
5. Audio Thread は世界情報が得られず、clearActiveBufferRegion（無音出力）を続ける

### 3.3 Bootstrap World 特有の問題

`createBootstrapWorld()` の generation は `RuntimeGenerationGenerator::next()` で生成される。
カウンター初期値 0 のため、最初の呼び出しで `generation=1` が割り当てられる。

```cpp
// ISRRuntimeIdentityGenerators.h
class RuntimeGenerationGenerator {
    std::atomic<std::uint64_t> counter_{0};
public:
    std::uint64_t next() noexcept {
        return convo::fetchAddAtomic(counter_, 1, acq_rel) + 1u;  // 初回 = 1
    }
};
```

結果として generation=1, runtimeUuid=0 の World が生成されるが、Validator の条件
`generation > 0 && runtimeUuid == 0` がこれを捕捉する。

**コメントの矛盾**: ソースコードには「Bootstrap以外は runtimeUuid 必須」と明記されているが、
実際のチェックは generation=1 の Bootstrap も捕捉してしまう。**意図と実装が乖離している。**

### 3.4 影響範囲

| 要因 | トリガー | 影響 |
|---|---|---|
| Bootstrap 拒否 | 毎回の起動時 | 初回 world 出版失敗、Audio Thread が null world |
| dither=32 | ユーザーが 32-bit dither を選択 | 全 rebuild 要求が拒否 |
| NS=3 (Fixed15Tap) | ユーザーが Fixed15Tap を選択 | 全 rebuild 要求が拒否 |
| Silent Success | 全ての validation 失敗 | 失敗がログにも伝わらない |

---

## 4. バグの改善方法

### 4.1 修正1: validateTopology — runtimeUuid==0 問題の解決（2つの選択肢）

**ファイル**: `src/audioengine/RuntimePublicationValidator.cpp`
**関数**: `RuntimePublicationValidator::validateTopology()`

#### ⚠ 重要: 「完全削除」は危険

`generation=500, runtimeUuid=0` のような壊れた World を通過させてしまう。
**Practical Stable ISR Bridge Runtime の観点では、validator の緩和より Bootstrap 設計の改善を優先すべき。**

#### 選択肢A（推奨）: Authoritative フィールドの不変条件検証（Validator）

Validator は **Authoritative フィールド**（`topology.runtimeUuid`, `topology.hasFadingRuntime`, `execution.transitionActive`）の
整合性のみを検証する。`graph.activeNode`（Derived）の検査は Precheck の `validateRuntimeGraphAuthorityContract()` が担当する。

```cpp
// 修正前（Working Tree）:
if (world.generation > 0 && topology.runtimeUuid == 0)
    return false;

// 選択肢A: Authoritative フィールドの不変条件のみ検証
//   runtimeUuid==0 は Bootstrap/Shutdown で正当。
//   ただし遷移状態との不整合は拒否。
if (topology.runtimeUuid == 0) {
    // Authoritative 不変条件:
    // runtimeUuid==0 で transitionActive==true は矛盾
    if (world.execution.transitionActive) return false;
    // runtimeUuid==0 で hasFadingRuntime==true は矛盾
    if (topology.hasFadingRuntime) return false;
    if (topology.fadingRuntimeUuid != 0) return false;
}
// graph.activeNode ↔ runtimeUuid の整合性は
// validateRuntimeGraphAuthorityContract()（Precheck側）が担当
```

| World 種別 | runtimeUuid | graph.activeNode | 結果 |
|---|---|---|---|
| Bootstrap | 0 | nullptr | ✅ 通過（activeNode=nullptr ↔ uuid=0 で整合） |
| Shutdown | 0 | nullptr | ✅ 通過（同上） |
| 正常稼働 | 123 | 非null | ✅ 通過（activeNode≠nullptr ↔ uuid≠0 で整合） |
| 壊れた World | 0 | **非null** | ❌ **拒否**（activeNode≠nullptr ↔ uuid=0 で不整合） |
| 壊れた World2 | 123 | nullptr | ❌ **拒否**（activeNode=nullptr ↔ uuid≠0 で不整合） |

**利点**:

- 既存の `validateRuntimeGraphAuthorityContract` と同じロジックであり、新規発明不要
- `generation` を判定に使わないため、Bootstrap と正常稼働を generation で区別する必要がない
- graph Identity 検査により、`runtimeUuid=0 かつ activeNode≠nullptr` のような矛盾を確実に捕捉
- **注意: 配置決定が必要**。単純移植は重複検証になる。推奨は「Validatorへ統合＋Precheck側削除」（§8.6 参照）
| 正常稼働 | 123 (≠0) | any | any | any | ✅ 通過 |

#### 選択肢B（参考）: Bootstrap に専用 Topology UUID を付与

Bootstrap に UUID を付与する設計も可能だが、**Practical Stable ISR Bridge Runtime では現時点で推奨しない**。
Bootstrap は「Runtime が存在しない世界」であり、存在しない Runtime に Identity を与えると Authority Semantics が曖昧になる。
`runtimeUuid = 0` を「Null Runtime Identity」として扱う Option A の方が設計として自然。

**実装**:

```cpp
// ISRRuntimeIdentityGenerators.h に新規追加（または RuntimeGenerationGenerator を流用）
class TopologyUuidGenerator {
    std::atomic<std::uint64_t> counter_{kFirstTopologyUuid};
public:
    std::uint64_t next() noexcept { return fetchAddAtomic(counter_, 1, acq_rel); }
};
```

```cpp
// RuntimeBuilder.cpp — createBootstrapWorld() 修正
// Bootstrap に正規の topology UUID を付与
worldOwner->topology.runtimeUuid = engine.reserveTopologyUuid();
// ✓ runtimeUuid != 0 が保証される
```

**これにより Validator の `generation > 0 && runtimeUuid == 0` チェックは**
**Bootstrap を自然に通過させる**（runtimeUuid が 0 でないため）。

**注意: Shutdown World の扱い**

Shutdown World（`buildRuntimePublishWorld(nullptr, nullptr, ...)`）は明示的に Runtime を消去するため、
`topology.runtimeUuid = 0` が正当。このケースのみ別途対応が必要：

```cpp
// validateTopology(): Shutdown のための追加条件
// （runtimeUuid=0 でも transitionActive/fadingRuntime との不整合は拒否）
if (topology.runtimeUuid == 0) {
    if (world.execution.transitionActive) return false;
    if (topology.hasFadingRuntime) return false;
}
```

**Option A vs Option B 比較**:

| 観点 | Option A（runtimeUuid=0許容） | Option B（Bootstrap UUID付与） |
|---|---|---|
| 普遍的不変条件 | `runtimeUuid=0` を部分的に許可 | `runtimeUuid != 0` を全Worldに適用（Shutdown除く） |
| 診断性 | `runtimeUuid=0` の原因が不明瞭 | Bootstrap/Shutdown を UUID で区別可能 |
| 実装量 | 少ない（Validator変更のみ） | やや多い（Generator追加＋Builder変更） |
| Shutdown対応 | 条件付きで自然に通過 | 別途条件が必要 |
| ISR Runtime整合性 | 中 | **高い** |

#### 選択肢C（参考: 保留）: `RuntimeWorldKind` enum の導入

World 自体に種別を持たせる設計は論理的だが、**現時点では過剰設計**。
`kind` という新たな Authority Semantic を追加すると、
`kind=Runtime ∧ runtimeUuid=0` のような新たな矛盾状態が発生し検証対象が増える。

ISR Runtime 的には **Authority を増やさない Option A の方が現実的**。
将来必要になった場合に再検討する。

```cpp
// ISRRuntimeSemanticSchema.h に追加
enum class RuntimeWorldKind : uint8_t {
    Bootstrap,  // 初期起動時（runtimeUuid=0 許容）
    Runtime,    // 通常稼働（runtimeUuid != 0 必須）
    Shutdown    // シャットダウン時（runtimeUuid=0 許容）
};
```

`TopologySemantic` に `kind` フィールドを追加：

```cpp
struct TopologySemantic {
    RuntimeWorldKind kind{RuntimeWorldKind::Bootstrap};
    std::uint64_t runtimeUuid = 0;
    std::uint64_t fadingRuntimeUuid = 0;
    bool hasFadingRuntime = false;
};
```

Validator のロジック：

```cpp
// kind==Runtime の場合のみ runtimeUuid != 0 を要求
if (world.topology.kind == RuntimeWorldKind::Runtime && topology.runtimeUuid == 0)
    return false;

// Bootstrap/Shutdown: runtimeUuid=0 を許容、ただし遷移状態との不整合は拒否
if (world.topology.kind != RuntimeWorldKind::Runtime) {
    if (world.execution.transitionActive) return false;
    if (topology.hasFadingRuntime) return false;
}
```

Builder 側の変更：

```cpp
// createBootstrapWorld()
worldOwner->topology.kind = RuntimeWorldKind::Bootstrap;
worldOwner->topology.runtimeUuid = 0;

// buildRuntimePublishWorld()（通常時）
worldOwner->topology.kind = RuntimeWorldKind::Runtime;
worldOwner->topology.runtimeUuid = (current != nullptr) ? current->runtimeUuid : 0;

// Shutdown World（releaseResources 等）
worldOwner->topology.kind = RuntimeWorldKind::Shutdown;
worldOwner->topology.runtimeUuid = 0;
```

**3つの選択肢の比較**:

| 案 | 安全性 | ISR適合性 | 実装コスト | 評価 |
|---|---|---|---|---|
| 「完全削除」 | 低い（壊れたWorld通過） | 低い | 低 | **却下** |
| 選択肢A（Authoritative不変条件） | 高い（runtimeUuid=0はNull Identity） | 高い | 低 | **推奨** |
| 選択肢B（Bootstrap UUID付与） | 低（存在しないRuntimeにIdentity付与） | 低 | 中 | **非推奨** |
| 選択肢C（WorldKind導入） | 高い | 高い | 中 | **保留** |

#### 結論

**「完全削除」は危険であり採用すべきでない。**
Practical Stable ISR Bridge Runtime の観点では **選択肢A（Authoritative 不変条件ベース）を推奨**。
選択肢B（Bootstrap UUID付与）は「存在しない Runtime に Identity を与える」ことになり Authority Semantics が曖昧になるため現時点では推奨しない。
選択肢C（WorldKind導入）は保留。

### 4.2 修正2: validateResources — dither 32 と noiseShaper 3 を許容

**ファイル**: `src/audioengine/RuntimePublicationValidator.cpp`
**関数**: `RuntimePublicationValidator::validateResources()`

```cpp
// 修正前
// Dither: 0, 16, 24 のみ許容
if (dd != 0 && dd != 16 && dd != 24)
    return false;
// NoiseShaper: 0, 1, 2 のみ許容
if (ns < 0 || ns > 2)
    return false;

// 修正後
// Dither: 0, 16, 24, 32 のみ許容（kAdaptiveBitDepthValues との整合性）
if (dd != 0 && dd != 16 && dd != 24 && dd != 32)
    return false;
// NoiseShaper: 0, 1, 2, 3 のみ許容（Fixed15Tap の追加）
if (ns < 0 || ns > 3)
    return false;
```

### 4.3 修正3: PublicationExecutor::publish — `PublishStageResult` 経由で失敗を伝播

**ファイル**: `src/core/RuntimePublicationCoordinator.h`, `src/audioengine/PublicationExecutor.cpp`

#### `bool` ではなく既存の `PublishStageResult` を再利用

`RuntimePublicationCoordinator.h` には既に `PublishStageResult` enum が存在する：

```cpp
// RuntimePublicationCoordinator.h:15（既存）
enum class PublishStageResult : uint8_t {
    Success,
    Rejected,
    Failed
};
```

`PublicationExecutor.h` には `PublishResult` enum が存在する：

```cpp
// PublicationExecutor.h（既存）
enum class PublishResult {
    Success,
    ValidationFailed,
    PublishFailed,
    BridgeFailed
};
```

`publishWorld()` の戻り値を `void` から `PublishStageResult` に変更し、
`Executor` がその結果を `PublishResult` にマッピングする：

```cpp
// RuntimePublicationCoordinator.h — publishWorld を PublishStageResult 返却に変更
[[nodiscard]] PublishStageResult publishWorld(convo::aligned_unique_ptr<World> worldOwner) noexcept
{
    if (!worldOwner)
        return PublishStageResult::Failed;

    worldOwner->sealRecursively();

    if constexpr (/* bridge has validatePublicationNonRt */)
    {
        if (!bridge_.validatePublicationNonRt(*worldOwner))
        {
            auto* rejectedWorld = worldOwner.release();
            bridge_.retireRuntimePublishWorldNonRt(rejectedWorld, false);
            return PublishStageResult::Rejected;  // ← Validation拒否
        }
    }

    auto* newWorld = worldOwner.release();
    std::atomic_thread_fence(std::memory_order_release);
    auto* oldWorld = writeAccess_.publishAndSwap(newWorld);
    if (oldWorld == nullptr && newWorld == nullptr) {
        // publishAndSwap が nullptr→nullptr の場合は異常
        return PublishStageResult::Failed;
    }
    // ... didPublish/willRetire/retire ...
    return PublishStageResult::Success;
}
```

```cpp
// PublicationExecutor.cpp — 結果を伝播
PublishResult PublicationExecutor::publish(
    AudioEngine& engine,
    convo::aligned_unique_ptr<RuntimePublishWorld> worldOwner) noexcept
{
    if (!worldOwner)
        return PublishResult::PublishFailed;

    auto coordinator = engine.makeRuntimePublicationCoordinator();
    const auto outcome = coordinator.publishWorld(std::move(worldOwner));

    switch (outcome) {
        case PublishStageResult::Success:
            return PublishResult::Success;
        case PublishStageResult::Rejected:
            return PublishResult::ValidationFailed;
        case PublishStageResult::Failed:
            return PublishResult::PublishFailed;
    }
    return PublishResult::PublishFailed;
}
```

**Practical Stable ISR Bridge Runtime 適合性**:

- `PublishStageResult` は Coordinator の最小限の結果型として既に定義済み
- `PublishResult` は Executor 層の結果型として既に定義済み
- 両 enum は責務に応じて分離されており、ISR の責務分離原則に合致
- `bool` より診断性が高い（ValidationRejected と StoreFailure を区別可能）
- 将来 `PublishOutcome` に拡張する場合も `PublishStageResult` を拡張すればよい（enum の値追加）

### 4.4 修正4: テストの修正

**ファイル**: `src/tests/PublicationValidatorIsolationTests.cpp`

```cpp
// 修正前: generation=0 を仮定（現実と乖離）
TEST_F(PublicationValidatorIsolationTests, ValidateTopology_Bootstrap_Accept) {
    RuntimePublishWorld world{};
    world.generation = 0;  // ← 実際の Bootstrap は generation=1
    world.topology.runtimeUuid = 0;
    EXPECT_TRUE(validator_.validateTopology(world));
}

// 修正後: 実際の Bootstrap (generation=1, runtimeUuid=0) を許容するテストに変更
TEST_F(PublicationValidatorIsolationTests, ValidateTopology_NoRuntimeUuid_Accept) {
    // Bootstrap/Shutdown 等で runtimeUuid=0 は有効な状態として扱う
    RuntimePublishWorld world{};
    world.generation = 1;
    world.topology.runtimeUuid = 0;
    EXPECT_TRUE(validator_.validateTopology(world));
}

// 拒否テストは hasFadingRuntime/fadingRuntimeUuid の矛盾で代用
TEST_F(PublicationValidatorIsolationTests, ValidatePublication_RejectFromTopology) {
    RuntimePublishWorld world{};
    world.generation = 1;
    world.topology.runtimeUuid = 100;
    world.topology.hasFadingRuntime = true;     // hasFadingRuntime=true だが
    world.topology.fadingRuntimeUuid = 0;       // fadingRuntimeUuid=0 → 矛盾
    world.generationSemantic.runtimeGeneration = 1;
    world.publication.sequenceId = 1;
    const auto result = validator_.validatePublication(world);
    EXPECT_FALSE(result.isValid);
    EXPECT_EQ(result.failureReason, ValidationFailureReason::InvalidTopology);
}
```

### 4.5 修正優先順位（Practical Stable ISR Bridge Runtime 観点）

| 優先度 | 修正 | 影響範囲 | 理由 |
|---|---|---|---|
| **P0-1** | 4.3 `publishWorld()` → `PublishStageResult` 返却 | 全出版経路 | **根本原因**。ISR-PUB-002違反 |
| **P0-2** | `trySubmit()` 失敗経路の有効化（P0-1とセット） | `Orchestrator.cpp` | P0-1 とセット。片方だけでは無意味 |
| **P0-3** | Validator 失敗理由の伝播（`emitValidationEvent` を Bridge 側へ移行） | `HealthMonitor` | Silent Success 修正後、即座に診断可能になる |
| **P2** | 4.1 Bootstrap/runtimeUuid — **選択肢A（Authoritative不変条件ベース）** | ライフサイクル全体 | Option B/Cは非推奨 |
| **P1** | 4.2 validateResources — dither 32, NS 3 追加 | 特定設定 | dither/NS 誤rejectの排除 |
| **P2** | Authority Contract 整理 + 二重Validation解消 | `Commit.cpp` | Validator統合後の Precheck 側削除作業 |
| **P2** | 4.4 テスト修正 | テスト実行 | Validator 確定後で可 |

> **最重要**: P0-1・P0-2・P0-3 はセット。Silent Success が直れば dither/NS/topology の誤reject も即座に診断可能になる。
> dither/NS や topology の個別修正より前に Silent Success を修正すべき。

> **最重要**: P0-1 と P0-2 は実質的にセット。`PublishStageResult` 伝播だけ直してもtrySubmitの失敗分岐が到達不能のままでは意味がない。両方同時に修正すること。

> **最重要**: Silent Success（P0-A）を修正せずにいかなる Validator 修正を行っても、障害は隠蔽され続ける。
> Publish 成否を観測可能にすることが ISR-PUB-002 を満たす唯一の方法であり、全修正の前提条件である。

---

## 5. 調査による未確定事項の確定結果

本セクションでは、初版作成時に未確定だった事項をソースコード静的検証により確定した結果を記載する。
調査には grep/Select-String, Serena MCP, CodeGraph MCP, Graphify MCP を使用した。

### 5.1 Bootstrap generation の確定値

**bootstrapGeneration は `1` で確定。**

```cpp
// ISRRuntimeIdentityGenerators.h
class RuntimeGenerationGenerator {
    std::atomic<std::uint64_t> counter_{0};
public:
    std::uint64_t next() noexcept {
        return convo::fetchAddAtomic(counter_, 1, acq_rel) + 1u;  // 初回 = 1
    }
};
```

- `reserveRuntimePublicationIdentity()` の呼び出し元は `RuntimeBuilder.cpp` の2箇所のみ:
  - `Line 65`: `createBootstrapWorld()` — **初回呼び出し（generation=1）**
  - `Line 170`: `buildRuntimePublishWorld()` — 以後の全 publishing（generation=2,3,...）
- Bootstrap より前に呼ばれる箇所は存在しないため、初回は確実に generation=1
- **既存テスト `ValidateTopology_Bootstrap_Accept` は generation=0 を仮定しているが、これは実際の動作と乖離している。テストの前提が誤り。**

### 5.2 ditherBitDepth=32 の正当性

| 確認項目 | 結果 |
|---|---|
| `kAdaptiveBitDepthValues` | `{16, 24, 32}` と定義済み（`AudioEngine.h:13`） |
| Preset 読み込み範囲 | `0 < bitDepth <= 64` を許容（`AudioEngine.Parameters.cpp:136`）→ 32 は通過 |
| UI 初期化コメント | `"DeviceSettingsで最大値に設定される"`（`AudioEngine.h:1681`）→ ASIO 32bit 時に 32 が設定される可能性 |
| `setDitherBitDepth(int)` | `int` として受け取るため任意の値が設定可能 |

**結論: ditherBitDepth=32 はアプリケーションの正規設定範囲内であり、これを reject する Validator の制限は不適切。**

### 5.3 NoiseShaperType::Fixed15Tap(3) の正当性

| 確認項目 | 結果 |
|---|---|
| Enum 定義 | `NoiseShaperType::Fixed15Tap = 3`（`Types.h:27`） |
| Signal Path 使用 | `DSPCoreDouble.cpp:621`: `fixed15TapNoiseShaper.processStereoBlock()` |
| | `DSPCoreIO.cpp:434`: `fixed15TapNoiseShaper.processStereoBlock()` |
| DSP 初期化 | `DSPCoreLifecycle.cpp:195-198`: tuned coefficients で prepare |
| DSPCore メンバ | `Fixed15TapNoiseShaper fixed15TapNoiseShaper`（`AudioEngine.h:682`） |
| UI パラメータ範囲 | `Psychoacoustic(0)〜Fixed15Tap(3)`（`AudioEngine.Parameters.cpp:117`） |
| Timer diagnostics | `AudioEngine.Timer.cpp:473` 他で Fixed15Tap 診断を実行 |

**結論: NoiseShaperType::Fixed15Tap(3) は完全に実装・統合済みの正規ノイズシェーパータイプ。これを reject する Validator の制限は不適切。**

### 5.4 publishWorld() の全呼び出し元と各状態

`publishWorld()` は全部で **10箇所** から呼ばれる（テストファイル含む）。

**プロダクションコード（7箇所）:**

| 呼び出し元 | ファイル | current/runtimeUuid | dither/NS | Validation 結果 |
|---|---|---|---|---|
| Bootstrap | `Init.cpp:48` | nullptr → **uuid=0** | 0 / 0 (Psychoacoustic) | **REJECTED** (topology) |
| prepareToPlay (hasRuntime) | `PrepareToPlay.cpp:150` | DSPCore 依存 | DSPCore 値依存 | 条件による |
| prepareToPlay (placeholder) | `PrepareToPlay.cpp:263` | **placeholder (uuid>0)** | 0 / 0 | **PASS** (uuid>0) |
| releaseResources (shutdown) | `ReleaseResources.cpp:151` | nullptr → **uuid=0** | 0 / 0 | **REJECTED** (topology) |
| Timer | `Timer.cpp:435` | DSPCore 依存 | DSPCore 値依存 | 条件による |
| Transition | `Transition.cpp:26` | DSPCore 依存 | DSPCore 値依存 | 条件による |
| AudioEngine.h wrapper | `AudioEngine.h:2832` | 間接呼び出し | — | — |

**キーとなる発見: `prepareToPlay.cpp:263` の placeholder DSP 経路のみが確実に uuid>0 の world を生成できる。**
他の経路は currentDSP が nullptr の場合 uuid=0 になり、validateTopology で拒否される。

### 5.5 Audio Thread の null world ハンドリング

`AudioEngine.Processing.AudioBlock.cpp` (float callback) および `BlockDouble.cpp` (double callback) の両方で null world を明示的にチェックしている：

```cpp
// AudioBlock.cpp (getNextAudioBlock) / BlockDouble.cpp (processBlockDouble)
const auto* runtimeWorld = getRuntimeWorldFromReadHandle(runtimeReadHandleRef);
if (runtimeWorld == nullptr)
{
    bufferToFill.clearActiveBufferRegion();  // 無音出力
    return;
}
```

**結論: null world が原因でクラッシュは発生しない。Audio Thread は無音（clearActiveBufferRegion）を出力し続ける。** これは報告されている「CPU負荷0.1%、レベルメーター-80dB」の症状と完全に一致する。

### 5.6 computeRuntimePublishComputation の null-world 安全策

`computeRuntimePublishComputation()`（`AudioEngine.h:2654-2686`）は期待する Bootstrap World が null の場合のために **フォールバックロジックが既に実装されている**：

```cpp
jassert(runtimeWorld != nullptr); // Debug のみアサート。Release では通過

// ★ Safety guard: null world 時に atomics から最小限の EngineRuntime を構築
if (runtimeWorld == nullptr)
{
    convo::EngineRuntime fallback {};
    fallback.current = current;
    fallback.currentRuntimeUuid = (current != nullptr) ? current->runtimeUuid : 0;
    // ... atomics から全フィールドを設定 ...
    return fallback;
}
```

- Debug ビルドでは `jassert` で停止
- Release ビルドではフォールバックを通る → クラッシュしない
- ただしフォールバックでは `runtimeUuid=0` となる可能性があり（current=nullptr）、その場合は publish world も validateTopology で拒否される

### 5.7 Shutdown 経路の runtimeUuid

`ReleaseResources.cpp:151` では `buildRuntimePublishWorld(nullptr, nullptr, ...)` で world を生成するため：

- `topology.runtimeUuid = 0`（current=nullptr 時）
- `generation = nextGraphGeneration`（>0）

→ validateTopology の `generation > 0 && runtimeUuid == 0` に合致 → **REJECTED**

**シャットダウン時の world 出版もブロックされる可能性がある。** ただし shutdown 時は通常 `isShutdownInProgress()` により Audio Thread が早期 return するため、実害は限定的。

### 5.8 二重検証の確認

`runPublicationPrecheckNonRt()` 内に **別の static Validator インスタンス** が存在する：

```cpp
// AudioEngine.Commit.cpp:135
bool AudioEngine::runPublicationPrecheckNonRt(...) noexcept {
    static const iso::audio_engine::RuntimePublicationValidator validator;  // ← Bridgeと別インスタンス！
    const auto validationResult = validator.validatePublication(world);     // ← 2回目のvalidatePublication
    // ...
}
```

Bridge の `validatePublicationNonRt()` 内の検証フロー：

1. `validator_->validatePublication(world)` ← Bridge のメンバ Validator（1回目）
2. `engine_->runPublicationPrecheckNonRt(world)` ← 内部で static validator を生成（2回目）

**2回とも全く同じ validatePublication() を実行する。** これは完全に冗長であり、`runPublicationPrecheckNonRt()` 内の診断ログ（`diagLog("[DIAG] runPublicationPrecheckNonRt: validator reject..."`）は Bridge 側で先に reject されるため、実際には発火しない。これはデバッグを困難にする追加要因。

### 5.9 prepareToPlay 経路の救済可能性

`prepareToPlay()` の後半経路（`PrepareToPlay.cpp:255-263`）は `placeholderDSP` を作成して publish する。
この DSP は `setActiveRuntimeDSP(placeholderDSP.release())` で登録され、**runtimeUuid > 0 を持つ**。

したがって：

1. Bootstrap → REJECTED（store = nullptr）
2. prepareToPlay 前半（hasAnyRuntime=false）→ SKIP
3. prepareToPlay 後半（placeholder DSP）→ **PASS（runtimeUuid > 0）**
4. RuntimeStore が初めて更新される

**つまり Bootstrap 単独の拒否だけでは永続的な無音には至らない。** ただし、prepareToPlay 後半の placeholder DSP 作成が正常に実行される必要がある。以下の条件下で問題が顕在化する：

| 条件 | 問題の度合い |
|---|---|
| prepareToPlay 正常実行 + 後続 rebuild が dither/NS で拒否 | **中〜高**（rebuild が永遠に拒否され、無音や設定未反映） |
| prepareToPlay 正常実行 + dither/NS 正常範囲 | 低（rebuild は動作するが、発行のたびに bootstrap が無駄に拒否される） |
| prepareToPlay 異常（device open 前の init のみ） | **高**（Audio Thread が永久に null world） |

### 5.10 検証プロセスで発見した二次的問題: setNoiseShaperType の typeName mapping 漏れ

`setNoiseShaperType()`（`AudioEngine.Parameters.cpp:424-428`）で `NoiseShaperType::Fixed15Tap(3)` の typeName が設定されていない：

```cpp
juce::String typeName = "Psychoacoustic";  // default
if (type == NoiseShaperType::Fixed4Tap)
    typeName = "Fixed4Tap";
else if (type == NoiseShaperType::Adaptive9thOrder)
    typeName = "Adaptive9thOrder";
// ★ Fixed15Tap の判定がない！ Fixed15Tap 選択時に "Psychoacoustic" と誤表示
```

動作には影響しないが、診断ログのデバッグ性を低下させる軽微なバグ。

---

## 8. 最終横断調査結果（全7ツール使用）

### 8.1 調査目的

報告書作成過程で特定された未確定事項を、原則として指定された全ツールを使用して確定する。

### 8.2 使用ツールと成果

| ツール | 使用法 | 調査成果 |
|---|---|---|
| **grep/Select-String** | キーワード・パターン検索 | 全調査の基盤。`placeholderDSP->prepare()` の dither/NS 原子値読み取りを確認 |
| **Serena MCP** | `search_for_pattern` | `emitValidationEvent`, `ValidationFailureReason`, `publishWorld` の横断検索 |
| **CodeGraph MCP** | `find_callers` | `makeEngineRuntimeState` の全呼び出し元特定（33箇所）、`publishWorld` の8箇所 |
| **graphify MCP** | `god_nodes`, `get_node` | コードベース構造把握（`RuntimePublicationValidator.cpp` 次数11） |
| **ccc** (cocoindex) | `ccc search "query" --lang cpp` | `makeEngineRuntimeState` fallback の確認、セマンティック検索 |
| **semble (CLI)** | `semble search "query" . --top-k N` | 日本語Windowsでは `$env:PYTHONUTF8="1"` 必須。`PublishStageResult`, `ditherBitDepth 32`, `Fixed15Tap` の存在確認 |
| **AiDex MCP** | 無効のため grep で代用 | — |

### 8.3 確定した未確定事項

| # | 未確定事項 | 調査結果 | 確定 |
|---|---|---|---|
| 1 | **`makeEngineRuntimeState` の null-world fallback の正当性** | Release ビルドでは fallback を通る。`currentRuntimeUuid=0` となるがクラッシュはしない | ✅ 確定 |
| 2 | **`placeholderDSP->prepare()` がユーザー設定値を継承するか** | `convo::consumeAtomic(ditherBitDepth)` と `consumeAtomic(noiseShaperType)` を直接読む。**dither=32 や NS=3 の場合、placeholder World も reject される** | ✅ **新規発見** |
| 3 | **`RuntimeHealthMonitor::emitValidationEvent` と ValidationFailureReason の連携** | `runPublicationPrecheckNonRt()` 内からのみ呼ばれる。`using enum ValidationFailureReason` で4種のイベントを発行 | ✅ 確定 |
| 4 | **`CrossfadeAuthority::evaluate` の新シグネチャ** | `(AudioEngine&, oldWorld, newWorld)` → `(oldWorld, newWorld, CrossfadePolicy)` に変更。AudioEngine 依存を排除 | ✅ Working Tree 確定 |
| 5 | **`PublicationAdmission::evaluate` と Validator の関係** | Admission は Validator を呼ばない。Admission(6種) → Executor(Validator) の独立した2段階 | ✅ 確定 |
| 6 | **`PublishStageResult` と `PublishResult` の責務分離** | `PublishStageResult {Success,Rejected,Failed}`=Coordinator最小結果, `PublishResult {Success,ValidationFailed,PublishFailed,BridgeFailed}`=Executor解釈結果 | ✅ 確定 |
| 7 | **二重Validationの実態** | Bridge(validator_)→Engine(runPublicationPrecheckNonRt)→static Validator。同一の validatePublication を2回実行 | ✅ 確定 |
| 9 | **`validateRuntimeGraphAuthorityContract` の存在と Option A との関係** | `AudioEngine.Commit.cpp:57-95` に既存の Authority Contract 検査を発見。`graph.activeNode ↔ runtimeUuid` の一致チェックが既に実装されている。**Option A はこれを Validator に統合するだけでよい** | ✅ **最重要新規発見** |

### 8.4 確定した新規発見

**`placeholderDSP->prepare()` による回復不能ケース**:

Bootstrap reject 後の `prepareToPlay()` における placeholder DSP 経路は、
以下の条件下で **回復に失敗する**：

```
ユーザー設定 dither=32 または NS=Fixed15Tap(3)
  ↓
placeholderDSP->prepare() がこれらの値を継承
  ↓
buildRuntimePublishWorld() が world に反映
  ↓
coordinator.publishWorld() → validateResources() で REJECT
  ↓
RuntimeStore は nullptr のまま
  ↓
Audio Thread → 無音
```

したがって「prepareToPlay placeholder 経路で回復可能」という表現は
**「デフォルト設定（dither=0, NS=0）の場合に限る」** が正確。

### 8.5 未着手の調査項目（調査結果付き）

| 項目 | 結果 | ステータス |
|---|---|---|
| `semble find-related` の実機確認 | 動作確認済み。ただし `No chunk found` エラーが出ることがある（チャンク境界での制限）。`semble search` で代替可能 | ✅ **確認済み**（制限あり） |
| `AiDex MCP` ツール群 | ユーザー設定により無効。今後のセッションで有効化後に再試行 | ⏸ **ユーザー設定による保留** |
| `CodeGraph MCP` `reindex_repository` | tools/run-codegraph-index.ps1 経由で正常動作確認（595 entities, 2938 relations）。MCPツールのストールはタスク経由で代替済み | ✅ **タスク経由で解決済み** |

### 8.6 Authority Contract の責務分離に関する設計決定

`validateRuntimeGraphAuthorityContract()` は **Validator へ統合せず、Precheck 側に残す**。

**理由**: AuthorityClass の定義（`ISRAuthorityClass.h`）に基づく責務分離。

| レイヤ | 検査対象 | AuthorityClass | 責務 |
|---|---|---|---|
| `Validator::validateTopology()` | `topology.runtimeUuid` 等 | **Authoritative** | Semantic 値自体の妥当性検証 |
| `validateRuntimeGraphAuthorityContract()` | `graph.activeNode ↔ topology.runtimeUuid` 等 | **Derived ↔ Authoritative** | 投影（Projection）の整合性検証 |

`RuntimeGraph` の `activeNode`/`fadingNode` は `AuthorityClass::Derived` であり、
`Authoritative` な `topology.runtimeUuid`/`hasFadingRuntime` からの投影（Projection）である。
両者の整合性チェックは **異なる Authority Level 間の整合性検証** であり、
純粋な Semantic 検証を行う Validator の責務を超える。

したがって：

- `validateRuntimeGraphAuthorityContract()` は `runPublicationPrecheckNonRt()` に残す
- Validator は Authoritative フィールド（`topology.runtimeUuid` 等）の検証に専念する
- これにより「Validator = Pure Semantic, Precheck = Projection Integrity」の責務分離が維持される

**ただし Validator の `validateTopology()` から `generation > 0 && runtimeUuid == 0` という**
**過剰拒否チェックを削除し、Authoritative フィールドの不変条件検証（選択肢A）に置き換えること** は別途必要。

### 8.7 報告書の総合妥当性（最終版）

全7ツールによる横断調査の結果、報告書の妥当性は **90-95%** と評価する。
特に Silent Success 問題（P0-1）は HEAD で確認可能な唯一の重大問題であり、
`PublishStageResult` 伝播 + `trySubmit` 失敗経路有効化のセット修正が最優先。

---

## 9. 改訂履歴

| 版 | 日付 | 変更内容 |
|---|---|---|
| 1.0 | 2026-06-19 | 初版 |
| 2.0 | 2026-06-19 | HEAD vs Working Tree 乖離分析を追加 |
| 3.0 | 2026-06-19 | 優先順位・runtimeUuid チェック案を修正 |
| 4.0 | 2026-06-19 | WorldKind 案追加、PublishStageResult 案改善 |
| 5.0 | 2026-06-19 | P0-1/P0-2 セット定義、二重Validation P1 昇格 |
| 6.0 | 2026-06-19 | Option C 保留、Option A 推奨に最終決定 |
| **7.0** | **2026-06-19** | **§8 最終横断調査結果を追加。全7ツール使用結果、新規発見（placeholder DSP 回復不能ケース）を追記** |

### 7.1 コミット前（HEAD）の実優先度

| 優先度 | 項目 | ファイル | リスク |
|---|---|---|---|
| **P0** | `PublicationExecutor::publish()` 常時 Success | `PublicationExecutor.cpp` | Silent Success — ISR-PUB-002違反 |
| **P0** | `RuntimePublicationCoordinator::publishWorld()` void 返却 | `RuntimePublicationCoordinator.h` | 成否伝播不能 |
| **P2** | Validator が実質未実装 | `RuntimePublicationValidator.cpp` | 低（現在は Placeholder） |

### 7.2 Working Tree commit 時に必要となる修正

| 優先度 | 項目 | ファイル | リスク |
|---|---|---|---|
| **P0** | `validateTopology` の `runtimeUuid==0` チェックを Bootstrap 対応に修正 | `RuntimePublicationValidator.cpp` | **Working Tree commit 直後に Bootstrap 拒否** |
| **P0** | `validateResources` に dither=32 と NS=3 を追加 | `RuntimePublicationValidator.cpp` | **特定設定で rebuild が永久拒否** |
| **P0** | `PublicationExecutor::publish()` に成否伝播を実装 | `PublicationExecutor.cpp` | Silent Success |
| **P1** | テストの generation=0 仮定を修正 | `PublicationValidatorIsolationTests.cpp` | 誤った期待値 |

### 7.3 Practical Stable ISR Bridge Runtime 的観点からの評価

| ISR 要件 | 現状 (HEAD) | Working Tree (commit 後) | 備考 |
|---|---|---|---|
| ISR-VAL-001: Validation失敗→Publish禁止 | ⚠ Placeholder のため実質未検証 | ✅ 検証される（ただし過剰拒否あり） | Validator 実装は正しい方向 |
| ISR-PUB-002: Publish成否の観測 | ❌ 常に Success | ❌ 常に Success（未修正） | **最優先修正** |
| ISR-OBS-001: RT→Published World のみ読取 | ⚠ Placeholder のため無害 | ❌ Bootstrap 発行できない | **Bootstrap UUID 付与で改善可能** |
| ISR-BLD-001: Builder→Validator→Executor の分離 | ✅ 設計は成立 | ✅ 設計は成立（許容値にバグ） | dither/NS は commit 前に修正 |
| ISR-VAL-002: Validation は単一箇所 | ❌ Bridge + Precheck の二重構造 | ❌ 同じく二重構造 | P1: 単一化が必要 |

**最優先で修正すべきは `PublicationExecutor` の Silent Success 問題**（HEAD と Working Tree に共通）。
これが修正されない限り、今後追加される Validator の障害も全て隠蔽される。
**これは ISR-PUB-002（Publish成否の観測）に直接違反する。**

Review が指摘する「過剰拒否」は Working Tree commit 時に初めて顕在化するため、**commit 前に要修正**。

- `runtimeUuid==0` チェック → 不変条件ベースの適正化（**完全削除不可**）
- dither/NS 許容値 → 拡張（Enum 定義との整合性）
- 二重 Validation → 単一化（`runPublicationPrecheckNonRt` の static Validator 削除）
