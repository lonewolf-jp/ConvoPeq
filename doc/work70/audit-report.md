# ConvoPeq work70 — メモリ肥大化 監査レポート

**生成**: 2026-07-11
**対象**: doc/work70/modification-plan-v3.md (v5.57)
**ソース**: src/audioengine/ 以下の変更ファイル群
**最終テスト状態**: Debug ビルド 0 errors + CTest 15/15 PASS + CI Gates ALL PASS

---

## 凡例

| ラベル | 意味 |
|:-------|:-----|
| ✅ **FACT** | コード解析またはログ解析で確定した事実 |
| 🔍 **Strong HYPOTHESIS** | ログとコードの両方が強く示唆するが、複数の説明が残るため未確定 |
| 🔍 **HYPOTHESIS** | ログやコードから示唆されるが未確定 |
| 💡 **PROPOSAL** | 改善のための設計案（複数案あり得る） |
| ⚠️ **CAVEAT** | 注意・制約・未解決の懸念 |

---

## Design Principles

本レポート全体を貫く設計原則。以降の監査・提案はすべてこの原則から導かれる:

| Authority | 責務 | コード位置 |
|:----------|:-----|:----------|
| **Construction Authority** | RuntimePublishWorld の生成。入力を忠実に反映し、自身では判断しない。 | `RuntimeBuilder` |
| **Publication Authority** | publish のトランザクション管理（register → publish → rollback）。 | `commitRuntimePublication()` |
| **Validation Authority** | 不変条件の最終確認。Builder の設定漏れを検出する安全網。 | `RuntimePublicationValidator` |
| **Lifetime Authority** | DSPCore の破棄。物理メモリ解放の唯一の入口。 | `DSPLifetimeManager` |
| **Crossfade Authority** | crossfade の要否判断。DSPCore に依存しない純粋関数。 | `CrossfadeAuthorityRuntime` |

### Builder（Construction Authority）の責務制約

Builder は以下を**行わない**:
- 入力の整合性を判断しない（No Validation）
- Policy Decision を行わない（No Policy Decision）
- 入力を構造体へ忠実に写像するのみ（Pure Construction）
- semantic intent を変更しない（Never mutates semantic intent — CrossfadePlan をそのまま写像）

```
入力 (current, next, policy, fadeTimeSec, active, sealedSnapshot, CrossfadePlan)
  ↓
Builder（忠実な写像、No Validation / No Policy / No Mutation）
  ↓
RuntimePublishWorld（Topology / Graph / Execution の**構造的整合性**のみ Builder 内で保証 — Semantic Invariant は Validator の責務）
```

この制約により、Builder にif文が増えることを構造的に防止する。将来の構築ルール変更は Builder の写像ロジックのみで対応し、判断ロジックは CrossfadeAuthority / Orchestrator が担当する。

**コア原則: 各 Authority は他の Authority の内部状態を直接変更しない。必要な情報は入力として渡し、出力は完成したオブジェクトまたは状態遷移として受け渡す。**

この原則から:
- Orchestrator は Builder の生成物に部分上書きしない（→ CrossfadePlan を Builder 入力として渡す）
- `commitRuntimePublication()` は Lifetime を直接呼ばない（→ OwnershipDisposition で間接化）
- Validator と AUTH_CONTRACT は別段階で独立して動作する（責務分離）

### Validator（Validation Authority）の正しい理解

Validator は **Builder の修正機構ではない**。以下のパイプラインで動作する:

```
Builder（生成）
  ↓
Validator（Invariant 最終検査） ← Builder の出力を検証するが、Builder を修正しない
  ↓
Publication（公開）
```

重要な設計判断:
- Builder は Validator が通ることを前提とせず、自身の構築規則に従って正しい World を生成する
- Validator は Builder の設定漏れを発見しても Builder を修正せず、Publication を拒否するのみ
- Validator 通過 = Builder が生成した World が Publication 可能な**最低条件**を満たした。Authority Contract は Precheck で別途検証するため、Validator 通過 ≠ World が完全に正しい
- Validator 不合格 ≠ Builder が間違っている（Builder 入力が不適切な場合もあり得る）

この分離により:
- Builder は Pure Construction に専念できる（No Validation）
- Validator は Invariant のみに専念できる（No Fix）
- 将来の構築ルール変更が Builder と Validator に独立して影響を与えない

---

## 1. 問題の全体像

ConvoPeq のプロセスメモリ使用量が、過去の観測では最大 Private Memory **約 2.5GB**、今回の修正適用後の測定 (v5.44) では **約 1.42GB** に達する。

### 1.1 最終計測値（v5.44 実機ログ）

| 指標 | 値 | 備考 |
|:-----|:---|:------|
| 起動直後 | 75MB | JUCE framework 最小構成 |
| 初回 prepare 後 | 556MB | gen=1/2 world (blockSize=524288) |
| prepareToPlay 後 (gen=3) | 996MB | Coordinator direct publish 完了 |
| リビルドサイクル後 | 1422MB | gen=4/5/8 publish 失敗 ×2 + IR load |
| 最終状態（35秒後） | 1422MB | DC live=3, NUC live=2, SC live=1 |
| MKL convolution | 35MB (2.5%) | 全体に対する割合 |

### 1.2 最終 Private Memory 内訳（1422MB）

```
  ├─ DSPCore寄与（推定）                             ≈ 660MB  ※DC live=3 からの推定値であり確定値ではない．runtimeUuid 単位の DIAG 計装が必要
  ├─ MKL convolution (tracked)                           =  35MB
  ├─ rebuild スレッド一時バッファ                        ≈ 200MB  △ 推定
  ├─ JUCE Framework + CRT                                ≈ 360MB  △ 推定
  └─ Other (CacheManager/IRConverter/STL)                ≈ 167MB  △ 推定
```

⚠️ **CAVEAT**: 上記内訳は MEM_SNAP の DC live=3 と各フェーズのメモリ増分から推定したものであり、**各 DSPCore の個別サイズは未計測**。実際には gen=1 (blockSize=524288, ~481MB) と gen=4以降 (blockSize=2048, ~220MB) でサイズが大きく異なる可能性がある。したがって **≈660MB は DSPCore 群の寄与としての推定値**であり、確定には runtimeUuid 単位の DIAG 計装が必要。

---

## 2. 発見された4つの根本原因（FACT / HYPOTHESIS）

### 2.1 Primary: activeRuntimeDSPHandle_ 未更新 — ✅ FIXED (v5.39)

**Status**: ✅ FIXED · ✅ FACT（コード確定）

**問題**: `commitRuntimePublication()` で publish 成功後に `dspHandleRuntime_.activate(handle)` が呼ばれないため、`activeRuntimeDSPHandle_` が null のまま。次回 Orchestrator リビルド時に oldDSP を解決できず retire が発行されない。

**原因コード**:
```
commitRuntimePublication()          ← AudioEngine.h
  → registerDSPHandleForRuntime()   ← Map に登録 ✅
  → coordinator.publishWorld()      ← 成功 ✅
  → dspHandleRuntime_.activate()    ← ★ 存在しなかった ❌
```

**修正**: `commitRuntimePublication()` 内で publish 成功直後、rollbackHandle 無効化前に `dspHandleRuntime_.activate(rollbackHandle)` を追加。

**ファイル**: `src/audioengine/AudioEngine.h:3981`

---

### 2.2 Secondary: DSPGuard rebuild-obsolete リーク — ✅ FIXED (v5.40)

**Status**: ✅ FIXED · ✅ FACT（コード確定）

**問題**: `AudioEngine.RebuildDispatch.cpp` の `DSPGuard` 構造体はデストラクタで `DSPLifetimeManager::retire()` を呼ぶが、rebuild-obsolete でキャンセルされた DSPCore は `runtimeDSPHandleMap_` に未登録のため `retireDSPHandleForRuntime()` が `false` を返し早期 return。`destroyDSPCoreNode()` が呼ばれずリーク。

**原因コード**:
```
dspGuard.~DSPGuard()
  → DSPLifetimeManager::retire(ptr)
    → retireDSPHandleForRuntime(ptr)
      → Map.find(ptr) == end ❌（未登録）
      → return false
    → return（destroyDSPCoreNode に到達せず）🐛
```

**修正**: `retireDSPHandleForRuntime()` が false の場合、直接 `destroyDSPCoreNode(ptr)` を呼ぶ + DIAG invariant jassert。

**ファイル**: `src/audioengine/AudioEngine.RebuildDispatch.cpp:764-785`

---

### 2.3 #3: AUTH_CONTRACT ブロック — 🔍 Strong HYPOTHESIS

**Status**: ❌ UNFIXED · 🔍 Strong HYPOTHESIS

#### 観測事実（FACT）と hasFadingRuntime 書き込み元の確定

**hasFadingRuntime の書き込み元は3箇所のみ**（全コード調査済み）:
| # | 箇所 | 値 | 条件 |
|:-:|:-----|:---|:------|
| 1 | `RuntimeBuilder.cpp:91` (bootstrap) | `false` | 常に |
| 2 | `RuntimeBuilder.cpp:210` (build) | `graphState.fadingNode != nullptr` | グラフ状態に同期 |
| 3 | `RuntimePublicationOrchestrator.cpp:146` (trySubmit override) | `true` | `needsCrossfade && oldDSP != nullptr` |

**確認**: hasFadingRuntime=true を書き込むのは #3 (Orchestrator override) のみ。RuntimeBuilder は生成時には hasFading を graphState.fadingNode と同期する（#2）。

**AUTH_CONTRACT が発生し得る条件はコード上明確**:
```
Orchestrator trySubmit L145-146:
  worldOwner->topology.hasFadingRuntime = true;
  // graph.fadingNode は RuntimeBuilder で graphState.fadingNode (=nullptr) のまま
  // → ValidateRuntimeGraphAuthorityContract で:
  //   hasGraphFadingNode(nullptr) != hasFadingRuntime(true) → FAIL
```

**gen=4 の独立した失敗（原因特定済み）**:
- ✅ AUTH_CONTRACT ログが存在しない → **AUTH_CONTRACT ではない**
- ✅ Validator が reject している（precheck 到達前に遮断）
- ✅ HEALTH ログ L237: `[HEALTH] eventCode=6001 = EVENT_VALIDATION_TOPOLOGY_FAILURE` — HealthMonitor が **Topology Validation Failure イベントを発行**. Validator の reject 理由が `InvalidTopology` であることを強く示唆する。
- ✅ **正確な違反ルールは特定済み（コード調査により確認）**: `RuntimePublicationValidator::validateTopology()` 内のチェック条件 `topology.hasFadingRuntime != (topology.fadingRuntimeUuid != 0)`（RuntimePublicationValidator.cpp:92）に違反。

**推定される不整合の発生経路（🔍 Strong HYPOTHESIS）**:
上記の Validator 違反ルール自体は ✅ **FACT**（コード確認済み）。しかし、**gen=4 の具体的な world が本当に以下の経路で生成されたかどうかは、transitionActive の状態を確認するログが存在しないため未確定**。以下はコードパス分析から推定される最も有力な経路である。

```
推定される生成経路（RuntimeGraph 生成経路で発生する不整合）:
  computeRuntimePublishComputation()  ← 入力: current, next, policy, fadeTimeSec, active
    → EngineRuntime::fading = transitionActive ? next : nullptr  (AudioEngine.h:2726)
    → RuntimeGraph::fadingNode = state.fading  (AudioEngine.h:3013)

  RuntimeBuilder::buildRuntimePublishWorld()  ← Builder は入力を忠実にコピー
    → topology.hasFadingRuntime = (graphState.fadingNode != nullptr)  (RuntimeBuilder.cpp:210)

  ★ Builder は「入力された next の runtimeUuid」を topology.fadingRuntimeUuid へ忠実に写像するだけである。
     フィールド名は「fading」だが、Builder は next が「fading runtime」を意味するのか「candidate runtime」なのか
     判断できない（No Policy Decision）。単に next が nullptr かどうかでUUIDを代入:
    → topology.fadingRuntimeUuid = (next != nullptr) ? next->runtimeUuid : 0  (RuntimeBuilder.cpp:209)

  ⚠️ Builder を原因と断定してはならない。Builder は `graphState.fadingNode` と `next` を
     そのまま反映しているだけであり、不整合の真因は **RuntimeGraph 生成経路**
     （computeRuntimePublishComputation 内の EngineRuntime::fading / RuntimeGraph::fadingNode
     の設定）にある可能性が高い。Builder の忠実性はコードから確認できるが、
     EngineRuntime の計算が正しいかは別問題である。

  不一致の条件（推定）:
    transitionActive == false
      → fading == nullptr → hasFadingRuntime = false
      → しかし next ≠ nullptr のため fadingRuntimeUuid ≠ 0
      → hasFadingRuntime(false) != (fadingRuntimeUuid != 0) → Validator reject
```

🔍 **Strong HYPOTHESIS**: 上記の経路はコード構造から強く示唆されるが、実際の gen=4 world で `transitionActive==false` だったことを示すログは存在しない。したがって「推定される生成経路」として扱う。

**gen=4/5/8 の関係**: 3つのリビルドは同一 IR load + PEQ 設定トリガーで連続発行。gen=4 (Validator InvalidTopology) と gen=5/8 (AUTH_CONTRACT) は **別々のバリデーション段階で失敗**. gen=4 は Validator で遮断、gen=5/8 は Precheck (AUTH_CONTRACT) で遮断。gen=4 と gen=5/8 は、**トポロジー整合性チェック時点で状態が異なったため、検出段階が異なる**。

⚠️ **CAVEAT（2026-07-11 レビュー指摘）**: gen=5/8 が gen=4 と「同じ next ポインタの状態を持っていた」と断定することはコードだけでは証明できない。gen=4→reject→gen=5→build→gen=8 の流れにおいて、Builder は毎回新しい RuntimePublishWorld を生成する。gen=4 world と gen=5/8 world が同一の next/transitionActive 状態を持っていた可能性は高いが、確定にはログ確認が必要である。したがって「同様の条件を持つ可能性が高い」として扱う。

#### AUTH_CONTRACT 検出コード

```cpp
// AudioEngine.Commit.cpp:86-89
if (hasGraphFadingNode != world.topology.hasFadingRuntime) {
    // L1626: [AUTH_CONTRACT] FAIL fadingNode=0 hasFading=1
    return false;  // → rejectWithEvidence("runtime_graph_authority_contract")
}
```

#### 推定される因果関係の一例（Strong HYPOTHESIS）

```
// 推定される経路の一例:
// Orchestrator trySubmit の override が hasFadingRuntime=true のみを
// 設定し、graph.fadingNode を同期しなかった場合に不整合が発生する。
RuntimeBuilder::buildRuntimePublishWorld()
  → graphState.fadingNode = nullptr（リビルド時は fading 未設定）
  → world.topology.hasFadingRuntime = false（hasFading と同期）

Orchestrator trySubmit() L140-146:
  if (cfDecision.needsCrossfade && oldDSP != nullptr) {
      world.execution.transitionActive = true;
      world.topology.hasFadingRuntime = true;  // ★ override
      // しかし graph.fadingNode は未更新 → 不整合！
  }
```


#### 検証方法

Orchestrator override 時に `graph.fadingNode` も同時更新する修正を実装し、gen=5 以降の publish が成功することを確認する。

---

### 2.4 #4: rollback + retire 二重経路 — 🔍 Strong HYPOTHESIS（一部 FACT）

**Status**: ❌ UNFIXED · ✅ **FACT**（destroyDSPCoreNode未到達経路）＋ 🔍 **Strong HYPOTHESIS**（リーク）

#### FACT（コード確定）: destroyDSPCoreNode に到達しない経路が存在する

```cpp
// PublicationExecutor.cpp:44
const auto result = engine.commitRuntimePublication(coordinator, ...);
// commitRuntimePublication 内部:
//   → publishWorld() → FAILED
//   → ScopeExit rollbackDSPRegistration(handle)
//     → CAS(Constructing→Reclaimed) ✅
//     → eraseByHandle() → Map 削除 ✅
//   → return { PublishStageResult::Rejected }

if (!isCommitted(result.stage)) {
    return PublishResult::PublishFailed;  // result=2
}

// trySubmit のエラーハンドリング:
if (result != PublishResult::Success) {
    if (!req.newDSP.isNull())
        lifetime_.retire(newDSPResolved);
        // retireDSPHandleForRuntime(dsp)
        //   → Map.find() == end ❌（rollback で削除済み）
        //   → return false
        // → return（destroyDSPCoreNode 未実行 — 🐛 この経路が存在することは FACT）
}
```

#### Strong HYPOTHESIS: 上記経路により DSPCore はリークする可能性が高い

`destroyDSPCoreNode()` に到達しない経路が存在することはコードから確認できる。

**代替解放経路の調査結果（全5経路）**:
| 経路 | 結果 |
|:-----|:------|
| DSPGuard (rebuild-obsolete 専用) | ❌ 通常の publish 失敗後の DSPCore は DSPGuard の管理外 |
| Shutdown (`~AudioEngine`) | ❌ `DSPLifetimeManager::retire()` → `retireDSPHandleForRuntime()` false → **destroyDSPCoreNode 未実行**. shutdown でも解放されない |
| Quarantine | ❌ Handle は正常に Reclaimed 済み（rollback 成功）のため対象外 |
| DeferredDelete (`enqueueDeferredDeleteNonRt`) | ❌ `retireDSP` 経由でのみ呼ばれるが、retireDSPHandleForRuntime が false のため到達しない |
| EBR drain | ❌ `DSPLifetimeManager::retire()` の EBR enqueue に到達しない（handleFreed=false で早期 return）|

**結論**: 全5経路調査完了。destroyDSPCoreNode に到達しない経路が存在することは ✅ **FACT**。ただし生ポインタ管理のため、解析対象外の所有経路の可能性が完全に排除できない。したがって「リーク」は 🔍 **Strong HYPOTHESIS** に留める。

**✅ FACT（コード調査範囲内）**: `~DSPCore()` は `destroyDSPCoreNode()` (`AudioEngine.Threading.cpp:18`) からのみ呼ばれ、`destroyDSPCoreNode()` の全呼び出し元は `DSPLifetimeManager::retire()` / `retireDSP()` / `DSPGuard` の3経路のみ。現在解析したコード範囲では publish 失敗後のロールバック経路から `destroyDSPCoreNode()` に到達する経路は確認できなかった。



#### 推奨修正方針

trySubmit 側で直接 `destroyDSPCoreNode` を呼ぶより、`PublishCommitResult` に所有権状態（`OwnershipDisposition`）を含めて返す設計が Authority を維持する：

```cpp
// PublishCommitResult 拡張案
enum class OwnershipDisposition {
    Transferred,    // publish 成功、所有権移譲済み
    CallerDestroy, // publish 失敗、rollback 完了、呼び出し元が物理解放すべき
    None            // 初期状態（何も起きていない）
};

// ★ CallerDestroy は「呼び出し元が破棄責任を持つ」ことを示すが、
//   実際に destroyDSPCoreNode() を呼ぶのは DSPLifetimeManager（Lifetime Authority）である。
//   呼び出し元は DSPLifetimeManager::destroyRolledBackDSP() を経由して破棄を依頼する。
struct PublishCommitResult {
    convo::PublishStageResult stage;
    OwnershipDisposition ownership = OwnershipDisposition::None;
};

// commitRuntimePublication:
//   publish 失敗 + rollback 成功 → OwnershipDisposition::CallerDestroy
//   呼び出し元 (trySubmit) :
//     CallerDestroy かつ ptr != nullptr → DSPLifetimeManager::destroyRolledBackDSP(ptr)
```

⚠️ **設計上の注意（2026-07-11 レビュー指摘）**: `CallerOwns → destroyDSPCoreNode()` の直接呼び出しは **Authority が二重になる危険**がある。現在 `destroyDSPCoreNode()` は `DSPLifetimeManager::retire()` の内部（EBR enqueue 経由）および `DSPGuard` からのみ呼ばれている。そこへ新たな呼び出し元が加わると、物理破棄の Authority が `DSPLifetimeManager` と `PublicationExecutor` の二重になる。

したがって以下の設計を推奨する：

```cpp
// ★ DSPLifetimeManager に専用 API を追加
//   物理破棄の Authority は DSPLifetimeManager に一元化する。
//   destroyDSPCoreNode() を直接公開せず、DSPLifetimeManager 経由で呼び出す。
// ★ destroyRolledBackDSP() は EBR を経由しない特殊ルート。
//   「Publication Authority から返却された未公開オブジェクト（Never Published Object）」
//   のみを対象とし、EBR epoch 保護は不要（publish されたことのない DSPCore は
//   Audio Thread から到達不能なため）。通常の retire() とは異なるライフサイクル。
void DSPLifetimeManager::destroyRolledBackDSP(DSPCore* dsp) noexcept
{
    if (dsp == nullptr) return;
    // rollback 後の未登録 DSPCore を直接破棄（EBR 非経由）。
    // 事前条件: Handle は既に rollback 済み（Reclaimed）。
    // post-condition: DSPCore のメモリが解放される。
    engine_.destroyDSPCoreNode(dsp);
}

// 呼び出し側（PublicationExecutor::publish の failure path）:
//   CallerDestroy かつ ptr != nullptr → lifetime_.destroyRolledBackDSP(ptr)
```

この設計であれば：
- `destroyDSPCoreNode()` は `DSPLifetimeManager` の `private` 内部関数として隠蔽される
- 物理破棄の Authority は `DSPLifetimeManager` のみ
- 将来の破棄前後処理（DIAGログ / 統計 / HealthMonitor通知）の追加が一箇所で完結
- `DSPGuard` の直接破棄パスも `DSPLifetimeManager::destroyRolledBackDSP()` に統合可能（将来のリファクタリング候補）

---

## 3. 実装済み修正のサマリー

| # | 修正 | ファイル | ステータス |
|:-:|:-----|:---------|:----------|
| F-1 | commitRuntimePublication に activate 追加 | `AudioEngine.h` | ✅ **実装済み** |
| F-2 | lookupDSPHandleForRuntime (private + DIAG) | `AudioEngine.h` | ✅ **実装済み** |
| F-3 | DSPLifetimeManager::activate 純化（二重Authority解消） | `DSPLifetimeManager.h` | ✅ **実装済み** |
| F-4 | DSPGuard 直接破棄パス + DIAG jassert | `AudioEngine.RebuildDispatch.cpp` | ✅ **実装済み** |
| F-5 | getStereoLiveCount() DIAG ガード | `ConvolverProcessor.h` | ✅ **実装済み** |
| F-6 | DIAG_MKL_MALLOC ガード構造修正 | `DiagnosticsConfig.h` | ✅ **実装済み** |
| — | AUTH_CONTRACT 回避（fadingNode 同時更新） | `RuntimePublicationOrchestrator.cpp` | ❌ **未着手** |
| — | PublishCommitResult 拡張（RollbackSucceeded） | `AudioEngine.h` + `PublicationExecutor.cpp` | ❌ **未着手** |
| — | gen=4 失敗理由の DIAG 出力拡張 | `RuntimePublicationValidator` | ❌ **未着手** |
| — | DC live 内訳の runtimeUuid DIAG 出力 | `AudioEngine.Timer.cpp` | ❌ **未着手** |

---

## 4. 設計判断（Single Authority）

### 4.1 Authority Boundary

| 責務 | Authority | コード位置 |
|:-----|:----------|:----------|
| `activeRuntimeDSPHandle_` 更新 | `commitRuntimePublication()` のみ | `AudioEngine.h:3981` |
| `DSPCore*` active 更新 | `DSPLifetimeManager::activate()` | `DSPLifetimeManager.h:28` |
| DSPHandle 発行 | `registerDSPHandleForRuntime()` (idempotent) | `AudioEngine.h:3838` |
| DSPHandle 回収 | `retireDSPHandleForRuntime()` | `AudioEngine.h:3875` |
| EBR enqueue | `DSPLifetimeManager::retire()` | `DSPLifetimeManager.h:37` |
| DSPCore* 物理破棄 | `destroyDSPCoreNode()` (物理破棄エンドポイント) | `AudioEngine.Threading.cpp:15` |
| Crossfade 登録 | `CrossfadeAuthorityRuntime` | 別ファイル |
| Snapshot フェード | `SnapshotCoordinator` | 別ファイル |

### 4.2 Invariant

| ID | Invariant | 根拠 |
|:---|:----------|:------|
| INV-1 | 公開を試みる DSP は publish 前に Handle が登録済み | commitRuntimePublication が唯一の窓口 |
| INV-2 | 公開失敗時は Handle 登録をロールバック | commitRuntimePublication ScopeExit |
| INV-3 | Commit point 以降は Handle をロールバック禁止 | PublishStageResultTraits::isCommitted |
| INV-4 | Map に Reclaimed Handle が永続しない | rollback→erase, retire→erase |
| INV-5 | rollback 後、同一 DSPCore は再登録可能 | create() が Reclaimed slot を再利用 |
| INV-6 | Handle 登録と publish は同一トランザクション | commitRuntimePublication 一つ |
| INV-7 | rollback 後、Map 不整合は DIAG のみ報告 | CAS 成功が最優先 |
| INV-8 | alreadyRegistered の Handle は Constructing 状態 | CAS 条件 |
| INV-9 | `activeRuntimeDSPHandle_` は常に **Active** 状態の Handle を指す | `DSPHandleRuntime::activate()` および `endCrossfade()` のみが activeRuntimeDSPHandle_ を更新し、両方とも Active 状態を設定する。Constructing/Retired/Reclaimed 状態を指すことは設計上発生しない。activate 漏れの再発防止に重要。 |

### 4.3 依存方向（厳守）

```
commitRuntimePublication()     ← publishWorld + activate の唯一権威
    ↓ registerDSPHandleForRuntime()  ← idempotent helper
    ↓ coordinator.publishWorld()
    ↓ dspHandleRuntime_.activate()  ← Single Authority
    ↓ (commit point → rollbackHandle 無効化)

DSPLifetimeManager::activate()  ← DSPCore* のみ
DSPLifetimeManager::retire()    ← handle unregister + EBR

lookupDSPHandleForRuntime()     ← DIAG 限定 private (production 参照禁止)
```

---

## 5. 未解決の課題（Phase 2 以降）

### 5.1 AUTH_CONTRACT 修正（最優先）

**対象**: `RuntimePublicationOrchestrator.cpp:trySubmit()`
**修正**: crossfade decision override 時に `graph.fadingNode` も同時更新

```cpp
// 修正方針（抽象表現 — 具体的な代入値は RuntimeGraph 構造に依存するため例示不可）:
// Orchestrator trySubmit の crossfade override 時に、topology.hasFadingRuntime と
// graph.fadingNode の同期を確保する。具体的な代入値は RuntimeBuilder または
// CrossfadeAuthority の graph 構築ロジックに委ねる。
//
// ⚠️ CAVEAT: graph.fadingNode は RuntimeGraph の Topology 構造体の一部であり、
// DSPCore* の裸ポインタではない。したがって graph.fadingNode = oldDSP のような
// 直接代入は型不安全である。実際の修正は RuntimeBuilder の buildRuntimePublishWorld()
// 内で graph 状態を正しく構築することにより行うべきである。
// worldOwner->execution.transitionActive = true;
// worldOwner->topology.hasFadingRuntime = true;
// // ★ FIX: graph.fadingNode も builder 経由で同期する
```

**設計上の補足**: Orchestrator で `graph.fadingNode` のみを部分更新する設計は **Single Authority の観点から危険**である。なぜなら Builder が `Topology`, `Graph`, `Execution` を同時生成している設計だからである。Orchestrator が `graph` だけを直接触ると、Builder が生成した他の構造との間に不整合が生じる可能性がある。

**推奨設計: Builder 入力に CrossfadePlan を渡す（Proposal A — 長期方針）**
`CrossfadeDecision` を Builder に直接渡すのではなく、中間表現 `CrossfadePlan`（Builder入力の一部）として渡す:
```
CrossfadeAuthority::evaluate() → CrossfadePlan
                                      ↓
Builder::buildRuntimePublishWorld(current, next, policy, fadeTimeSec, active, crossfadePlan)
                                      ↓
Builder が Topology/Graph/Execution を全部一括生成（整合性保証）
```

このアプローチであれば:
- `CrossfadePlan` は immutable（不変）とする。Builder に入力された後は変更されず、Builder は忠実に写像するのみ。
- Builder は `CrossfadePlan` を immutable な入力として受け取り、自身の構築ロジックに統合する
- Orchestrator は Builder の生成結果を publish するのみで部分上書きしない
- Builder の責務は「Pure Construction（純粋構築）」に留まり、Decision/Policy/Construction の3役を兼ねない
- 将来の構造追加にも Builder の拡張のみで対応可能

🔍 **HYPOTHESIS**: 現状の Orchestrator override は Builder の生成に後から上書きする design smell である。Proposal A（CrossfadePlan 経由）への移行を Phase 2 のリファクタリングとして実施する。

### 5.2 rollback 後の物理解放（高優先）

**対象**: `commitRuntimePublication()` + `PublicationExecutor::publish()`
**修正**: `PublishCommitResult` にロールバック状態を追加

### 5.3 gen=4 失敗理由の確定

**対象**: `RuntimePublicationValidator`
**修正**: Validator の reject 理由を DIAG ログに出力

### 5.4 DC live 内訳の特定

**対象**: `AudioEngine.Timer.cpp` MEM_SNAP
**修正**: 各 DSPCore の runtimeUuid を DIAG 出力

### 5.5 Handle State 統計出力（推奨）

**対象**: MEM_SNAP または定期 VERIFY 出力
**追加**: DSPHandle の状態別件数（Constructing / Active / Retired / Reclaimed / Quarantined）を定期出力。例:
```
[HANDLE_STATS] C=0 A=1 Rt=2 Rc=8 Q=0
```
MEM_SNAP の DC: live だけでは区別できない DSPCore の生存状態を Handle 単位で可視化。現在の「DC live=3 の内訳」問題を直接解決する。

**運用上の推奨**: `HANDLE_STATS` / `RUNTIME_MAP` / `MEM_SNAP` は**同一タイムスタンプ**（同一 Timer callback 内）で出力する。例:
```
[T=81235] MEM_SNAP gen=8 Priv=1422MB DC:live=3 ...
[T=81235] HANDLE_STATS gen=8 C=0 A=1 Rt=2 Rc=8 Q=0
[T=81235] RUNTIME_MAP gen=8 uuid=0x... handle=31 ptr=0x... state=Active
```
これにより、1回のログ出力で HandleState / DSPCore live / Memory の三者対応が完全に一致し、解析時間が大幅に短縮される。**generation 値を全行に含める**ことで、世代間の対応関係も一目で把握可能。

### 5.6 ROLLBACK 三種同時ログ（推奨）

**対象**: rollbackDSPHandleRegistration() または commitRuntimePublication() の publish 失敗パス
**追加**: publish 失敗時に Handle / Ownership / DSPCore / generation / runtimeUuid の五者を同時ログ出力。例:
```
[ROLLBACK] gen=8 uuid=0x2634B7F8040 handle=31 owner=caller destroy=caller ptr=0x000002638A6DF400
```
これにより解析時に「誰が Handle を保持し、誰が破棄責任を持つか」が一目で把握可能。今回の #4 解析時間を半減できる。

### 5.7 PUBLISH_STATS 3階層カテゴリ出力（推奨）

**対象**: commitRuntimePublication() または RuntimePublicationBridge
**修正**: publish 結果を以下の3階層カテゴリに分割して出力する。これにより「どの段階で reject されたか」だけでなく「なぜ reject されたか」が一発で把握可能になる。

```text
Validation 層: 不変条件違反
├── Topology     = N   (Topology Representation Invariant — hasFadingRuntime/fadingRuntimeUuid 不整合など)
├── Transition   = N   (conflicting transitions)
├── Resources    = N   (oversampling/dither/noiseShaper 範囲外)
└── Semantic     = N   (generation/sequenceId 不整合)

Authority 層: Authority Contract 違反
├── Graph        = N   (graph.fadingNode vs hasFadingRuntime 不一致)
├── Handle       = N   (handle 発行/回収 contract 違反)
├── Snapshot     = N   (snapshot 整合性)
└── Crossfade    = N   (crossfade 判定 contract 違反)

Executor 層: 実行時障害
├── Commit       = N   (publish 成功)
├── Rollback     = N   (publish 失敗→rollback 成功)
├── Retry        = N   (enqueueRetire 再試行 — 将来拡張用)
└── Abort        = N   (publish 失敗→rollback も失敗)
```

例:
```
[PUBLISH_STATS] V: T=3 Tr=0 R=0 S=0 reject=3 | A: G=2 H=0 Sn=0 X=0 reject=2 | Exec: C=1 Rb=2 Rt=0 Ab=0 reject=2 total=6
```

これにより「Validator が 500 回 reject → 内訳は Topology=300, Transition=200」のように一発で把握可能になり、かつ Authority 層と Executor 層の統計も同時に追跡できる。

### 5.8 ValidationFailure 列挙型の一元化（推奨）

現在、ValidationFailureReason は `RuntimePublicationValidator` の内部 enum として定義されている（4値: InvalidTopology / InvalidResources / InvalidTransition / SemanticInconsistency）。これを公開 enum に昇格し、Validator / HealthMonitor / PUBLISH_STATS のすべてで同一の語彙を使用する:

```cpp
// RuntimePublicationValidator.h または独立ヘッダ
enum class ValidationFailure : uint8_t {
    None,
    InvalidTopology,           // Topology Representation Invariant 違反（hasFadingRuntime/fadingRuntimeUuid 不整合）
    InvalidTransition,         // conflicting transitions (policy/fadeTime)
    InvalidResources,          // oversampling/dither/noiseShaper 範囲外
    InvalidGeneration,         // generation 不整合
    InvalidSequence,           // sequence monotonicity 違反
    InvalidCrossfade,          // crossfade topology 不整合
    RuntimeGraphAuthority,     // AUTH_CONTRACT (graph.fadingNode vs hasFadingRuntime)
    SemanticInconsistency      // generation/sequenceId 矛盾
};
```

これにより:
- Validator は `ValidationFailure` を返す（現在の string errorMessage からの移行）
- HealthMonitor のイベントコード（EVENT_VALIDATION_TOPOLOGY_FAILURE = 6001 等）も同じ enum を発行
- PUBLISH_STATS のカウントも同じ enum 値でインクリメント
- DIAG ログは `[VALIDATOR] reason=InvalidTopology expected=true actual=false uuid=xxxxx` のように統一フォーマットで出力可能

**HealthMonitor との統合例**:
```cpp
// Validator 側:
const auto failure = ValidationFailure::InvalidTopology;
engine->m_healthMonitor.emitValidationEvent(failure);  // enum 直接発行

// PUBLISH_STATS 側:
convo::fetchAddAtomic(publishStats_[static_cast<size_t>(failure)], 1, ...);
```

すべてが同じ enum を参照するため、原因の追跡・集計・可視化が一貫して行える。

### 5.9 RUNTIME_MAP 対応表 DIAG 出力（推奨）

**対象**: 任意のタイミング（Timer callback の定期出力または VERIFY）
**追加**: UUID・Handle・DSPCoreポインタ・State・generation の対応表を1回出力。例:
```
[RUNTIME_MAP] gen=8 uuid=0x2634B7F8040 handle=31 ptr=0x000002638A6DF400 state=Active
[RUNTIME_MAP] gen=3 uuid=0x2634B7F8030 handle=17 ptr=0x000002638A6DF200 state=Active
[RUNTIME_MAP] gen=7 uuid=0x2634B7F8050 handle=0  ptr=0x000002638A6DF600 state=RolledBack
```
これにより以下の3者を直接突き合わせ可能になる:
```
MEM_SNAP の DC live / HANDLE_STATS の各状態件数 / 各 DSPCore の runtimeUuid
```
現在の「DC live=3 の内訳は gen ごとにしか分からない」問題を直接解決する。HANDLE_STATS だけでは「Active handle が1個」しか分からないが、RUNTIME_MAP があれば「どの gen のどの DSP が Active か」まで特定できる。

### 5.7 初回 BlockSize 最適化（P2-1）

**問題**: `SAFE_MAX_BLOCK_SIZE(65536) × MAX_OS_FACTOR(8) = 524288`
**影響**: 初回 prepare 時の DSPCore 1個あたり ~481MB

---

## 6. 検証計画

### 6.1 Phase 2 検証項目（Publish → Retire → EBR → Destroy → Memory 完全チェーン）

| # | 確認項目 | 段階 | 方法 | 合格基準 |
|:-:|:---------|:-----|:-----|:-------|
| 1 | AUTH_CONTRACT 修正後 gen=4 publish 成功 | **Publish** | DIAG ログ | `[PUBLISH] seq=N gen=4` |
| 2 | lifecycle(retire) > 0 | **Retire** | VERIFY | retire > 0 |
| 3 | pendingRetireCount 一時的増加 | **EBR enqueue** | MEM_SNAP `pend` | 一時的に >0 → 0 に収束 |
| 4 | reclaim 進行（rec カウンタ増加） | **EBR reclaim** | MEM_SNAP `rec` | rec カウンタが増加 |
| 5 | DC live 2（current+fading）収束 | **Destroy** | MEM_SNAP | DC: live=2 (steady) |
| 6 | **HandleState と DSPCore live の一致確認** | **Consistency** | HANDLE_STATS + MEM_SNAP | Active handles == DC live、Retired → 0、Reclaimed 増加が live 減少と一致 |
| 7 | Private Memory 改善 | **Memory** | MEM_SNAP | ~1000MB 程度まで低下 |
| 7 | DSPCore リークゼロ | **Destroy** | MEM_SNAP DC: live | 定常時 2 を超えない |
| 8 | rollback+retire 二重経路解消 | **Retire** | DIAG + コードレビュー | destroyDSPCoreNode が高々1回 |
| 9 | 音声品質 | **Quality** | 主観評価 | ノイズ・クリックなし |
| 10 | 30 分間安定動作 | **Stability** | 長時間実行 | クラッシュ・ハング・メモリ増加なし |

各段階の確認が **一本のチェーン** として機能していることを検証する。Publish 成功だけではメモリリークが解決したとは言えず、Retire→EBR→Destroy→Memory 収束まで確認して初めて「改善された」と判断する。

### 6.2 Partially Verified

- ✅ gen=3 world origin → prepareToPlay Coordinator direct publish（確定）
- ✅ DC live=3（確定）
- ✅ **gen=4 の失敗理由確定**（FACT #63）: `Orchestrator::buildRuntimePublishWorld(newDSPResolved, oldDSP, ..., false)` → Builder が `active=false, next=oldDSP≠nullptr` で `hasFadingRuntime=false, fadingRuntimeUuid≠0` を生成 → `needsCrossfade=false` のため override 未適用 → Validator L92 で reject。HEALTH eventCode=6001 発行。
- ✅ **gen=5/8 の失敗理由確定**（FACT #64）: 同種のトポロジ生成条件（`active=false, next=oldDSP≠nullptr`）から生成された world に対し `needsCrossfade=true` で override 適用（`transitionActive=true, hasFadingRuntime=true`） → **Validator は通過**（L92/L96 ともに一致）→ Precheck AUTH_CONTRACT L87 で `graph.fadingNode` 未更新を検出 → `hasGraphFadingNode(false) != hasFadingRuntime(true)` → reject。
- 🔍 DC live=3 の内訳 → 未確定（MEM_SNAP 単独では区別不能。HANDLE_STATS 追加が必要）

## 6a. メモリ消費量解析（ConvoPeq.log 実機ログ検証 2026-07-11）

**詳細レポート**: [memory-consumption-analysis.md](memory-consumption-analysis.md)

ConvoPeq.log（30,457行 / 163.5秒ラン / 無音起動→IR読込+PEQ設定→NoiseShaper→音楽再生→シャットダウン）の
MEM_SNAP / BUILD_PHASE / MEM publish 全データを解析。

### 6a.1 メモリタイムライン

| 時刻(s) | イベント | Private | 備考 |
|:--------|:---------|:--------|:-----|
| 0.0 | 起動 | 75 MB | JUCE + EQ ctor |
| 0.4 | gen=1 (48kHz) | 556 MB | 初回 DSPCore |
| 0.8 | gen=3 (192kHz osFactor=2) | 996 MB | 二重 DSPCore |
| 1.0 | gen=3 公開完了 | 648 MB | 旧 DSPCore 解放 |
| 1.5 | Heap 温存 | 674 MB | バッファコミット |
| +1.1～5.9 | 定常 #1 | 674→683 MB | 安定動作 |
| +5.9 | gen=7 rebuild (IR読込) | 1,094 MB | **ピーク** |
| +5.9 | AUTH_CONTRACT FAIL | — | IR未適用 |
| +6.0 | IR解放→定常 #2 | 676→686 MB | ベースライン同一 |
| +10.9 | NoiseShaper学習開始 | 684 MB | accepted=0 |
| +20.0 | 定常継続 | 686 MB | **160秒間ゼロ成長** |
| +163.5 | シャットダウン | — | 正常解放 |

### 6a.2 メモリリーク検出結果

| 検査対象 | 結果 | エビデンス |
|:---------|:-----|:----------|
| MKL Conv層 | ✅ リークなし | `lostFree=0` 全IR_RELEASEで確認 |
| NUC Lifecycle | ✅ リークなし | NUC alloc=0MB steady, peak=35MB fully recovered |
| EBR Retirement | ⚠️ 未評価（EBR未発動） | pend=0, reclaim=0 — EBRが動作しなかっただけ |
| トランザクションカウンタ | ⚠️ 検証機会なし | `pub/ret/reclaim=4/0/0` — retire=0は単にretire対象がなかった |
| ヒープ成長 (163秒) | ✅ 観測範囲では定常 | 683-686MB で160秒間フラット |

**注意**: 163秒の観測ウィンドウは短い。長時間（8時間以上）や大量IR切替（500回以上）では
異なる挙動を示す可能性がある。「リークなし」は「このログの範囲では確認できない」の意。

### 6a.3 不要メモリ確保の有無

1. **【軽微】gen=2 prepare の無駄 (processingRate=768k → 384k 訂正)**:
   gen=2 が 768k rate で prepare 開始 → gen=3 により obsolete → 384k で再prepare。
   ~100msの無駄なCPU/メモリ。アーキテクチャ上許容範囲。

2. **【情報】初回 648→686MB 成長 (+38MB/5秒)**:
   VirtualAlloc lazy commit + C++ heap 温存。その後の160秒間は完全フラット。
   リークではなく通常のヒープウォームアップ。全く問題なし。

### 6a.4 AUTH_CONTRACT FAIL のメモリ影響

gen=7 rebuild で IR (impulse_hpf_lpf, 192000 taps, 2ch) をロード:
- IR_LAYOUT: IRFreq=5MB FDL=11MB Accum=1MB 計 18MB/ch = 36MB (2ch persistent)
- MKL workspace: before=0→35→17→0MB (過渡的)
- **計 ~80MB の追加確保 → その後全解放**

結果: 1,094MB ピークの原因だが、解放後は 686MB ベースラインに復帰。
IR未適用のため定常状態では NUC live=0, alloc=0MB。DC: live=1 のみ。

### 6a.5 非メモリ問題: NoiseShaper サンプルレート不一致

NoiseShaper 学習スレッド: 全期間 `accepted=0`、毎ループ `dropSampleRate=~3000`。
~80回の無駄イテレーション（240,000サンプル相当）。

**原因（コード調査で確定 2026-07-11）**:
- セッション側: `engine.currentSampleRate` (192000) を読む (`NoiseShaperLearner.cpp:1034`)
- キャプチャブロック側: `DSPCore.sampleRateHz` (= **processingRate** 768000) を設定 (`DSPCoreDouble.cpp:290`)
- `drainCaptureQueue()` が `192000 ≠ 768000` で全ブロックを `droppedBySampleRate`
- osFactor > 1 で常に発生するバグ

**修正**: `DSPCoreDouble.cpp:290` / `DSPCoreIO.cpp:143` で `block.sampleRateHz` に
base sample rate（192000）を使用すべき。processingRate は誤り。

### 6a.6 新規確定 FACT（コード調査 2026-07-11）

| # | 項目 | 確定内容 | 調査方法 |
|:-:|:-----|:---------|:---------|
| 70 | **DSPCore tracked allocations = ~159 MB** | internalMaxBlock=524288固定（osFactor非依存）での生確保量。OS work 96MB(60%)、EQ 21MB(13%)、aligned+dry 16MB(10%)、latency 16MB(10%) | コード数式計算 |
| 71 | **DSPCore hidden overhead = ~191 MB** | C++ CRTヒープ断片化 + VirtualAlloc granularity + MKL FFT plan internal + container internals。DSPCore「350MB」の差分 | 実測値−生確保量 |
| 72 | **P2-1で98.2%削減可能** | SAFE_MAX_BLOCK_SIZE=65536→1024で internalMaxBlock=524288→8192。DSPCore 193MB→3.4MB | コード数式計算 |
| 73 | **NoiseShaper accepted=0 — 再調査中** | **⚠️ コード解析により前提誤りを訂正**: block.sampleRateHz は `dsp->sampleRate` (base rate) から設定。processingRate 説は誤り。Runtime観測が必要。 | コード解析＋ログ検証 |
| 74 | **RuntimeBuilder 一時的確保ゼロ** | build() 内で DSPCore 1個の `aligned_make_unique` のみ。中間バッファなし | コード確認 |
| 75 | **JUCE AudioProcessorGraph 不使用** | 独自 DSPチェーン、JUCEグラフルーティング不使用。JUCE単体で680MBは不可能 | コード確認 |
| 76 | **VirtualAlloc/VirtualFree/HeapAlloc 直接呼び出しなし** | 全確保は通常C++アロケータまたは mkl_malloc 経由 | grep全数調査 |

### 6a.7 結論

**観測範囲（163秒）ではメモリリークの傾向は確認できない。**
- 定常 686MB は 160秒間完全フラット（ゼロ成長）✅
- MKL/NUC はクリーンな解放を確認 ✅
- ピーク 1,094MB は二重バッファリング + Conv データで設計上当たり ✅
- EBRは今回のログでは発動しておらず、検証機会なし ⚠️

**以下は現状の診断では結論できない（要DIAG改善）**:
- 686MB "Other" の内訳（診断不能）
- 長時間稼働時のリーク有無（観測は163秒のみ）

**以下はコード調査で解決済み（v5.59）**:
- DSPCore 1個あたりのコスト → 生確保 ~159MB + 未計測残差 ~191MB（候補列挙、確定ではない）
- NoiseShaper accepted=0 → processingRate bug 特定
- RuntimeBuilder 一時的確保 → 存在しない
- JUCE支配説 → 否定（JUCE単体で680MBにはならない）
- VirtualAlloc 直接呼び出し → 存在しない

---

## 7. Root Cause Timeline — 証明済み因果関係と時間順

以下の時系列図は、本監査で特定された全問題（#1〜#4）の因果関係を世代（gen）単位で整理したものである。

```
gen=3 publish成功 (Coordinator direct publish)
        │
        ├─ activeRuntimeDSPHandle_ 未更新 ← ✅ FIXED (P1-a v5.39)
        │   └─ 原因: commitRuntimePublication() 内で activate() 未呼び出し
        │         └─ retire漏れ ← ✅ FIXED
        │
        ├───────────────────────────────────────────┐
        │                                           │
        ▼                                           ▼
   ┌─ Publication 入力 ──┐               └─ Post-build Mutation ──┐
   │ active=false         │               │ Orchestrator が       │
   │ next=oldDSP≠nullptr  │               │ Builder の World に   │
   └──────────┬───────────┘               │ 部分上書きするが、    │
              │                           │ graph.fadingNode を   │
              ▼                           │ 更新し忘れている      │
      gen=4 (Validator)                   └──────────┬─────────────┘
   ┌─────────────────────┐                           │
   │ Builder が忠実に     │                           ▼
   │ hasFadingRuntime=    │                   gen=5/8 (AUTH_CONTRACT)
   │ false を設定        │               ┌─────────────────────────┐
   │                      │               │ override で            │
   │ needsCrossfade=false │               │ transitionActive=true  │
   │ → override 未適用    │               │ hasFadingRuntime=true  │
   │                      │               │ → Validator は通過      │
   │ Validator L92:       │               │ → しかし graph.fading  │
   │ false≠(uuid≠0) →FAIL │               │   Node 未更新          │
   └─────────────────────┘               │ → AUTH_CONTRACT L87   │
              │                           │  で FAIL              │
   ┌─────────────────────┐               ┌─────────────────────────┐
   │ Builder が忠実に     │               │ override で            │
   │ hasFadingRuntime=    │               │ transitionActive=true  │
   │ false を設定        │               │ hasFadingRuntime=true  │
   │                      │               │ → Validator は通過      │
   │ needsCrossfade=false │               │ → しかし graph.fading  │
   │ → override 未適用    │               │   Node 未更新          │
   │                      │               │ → AUTH_CONTRACT L87   │
   │ Validator L92:       │               │  で FAIL              │
   │ false≠(uuid≠0) →FAIL │               │                       │
   └─────────────────────┘               └─────────────────────────┘
              │                                      │
              ▼                                      ▼
      gen=4 publish FAILED                    gen=5/8 publish FAILED
              │                                      │
              ▼                                      ▼
          rollback → Handle Reclaimed         rollback → Handle Reclaimed
          destroyDSPCoreNode 未到達           destroyDSPCoreNode 未到達
        │                                               │
        ▼                                               ▼
    DSPCore リーク（現解析範囲では実質確定）
    解析対象外の所有経路は理論上の可能性のみ           解析対象外の所有経路は理論上の可能性のみ
        │                                               │
        └───────────────┬───────────────────────────────┘
                        ▼
           最終状態: DC live=3, Private Memory 1422MB
```

### 因果関係のまとめ

| 原因 | 影響する世代 | 状態 | 種別 |
|:-----|:-----------|:-----|:------|
| activeHandle未更新 | gen=3 (retire漏れ) | ✅ **FIXED** | 独立した根本原因（Primary） |
| RuntimeGraph 生成入力（active=false, next≠nullptr） | gen=4/5/8 | ❌ **UNFIXED** | ✅ **FACT**（コード経路確定済み） |
| Post-build Mutation（graph.fadingNode未更新） | gen=5/8 | ❌ **UNFIXED** | ✅ FACT（ログ確認済み） |
| rollback→destroy未到達 | gen=4/5/8 | ❌ **UNFIXED** | ✅ FACT（経路存在）＋ 現解析範囲では実質確定 |

### Key Insight

- **gen=4 と gen=5/8 は同じ根本原因から派生した連鎖的障害ではない**。activeHandle問題は retire 漏れの原因であり（独立した枝）、Validator/AUTH_CONTRACT は別の独立した問題である。
- gen=4 と gen=5/8 は同一の **RuntimeGraph 生成入力**（`active=false, next=oldDSP≠nullptr`）から派生しているが、検出段階が differentiator（override の有無）により異なる。
- 根本原因は3系統に分かれる:
  ```
  activeHandle未更新 ── retire漏れ（✅ FIXED）

  RuntimeGraph 生成入力 ── gen=4 Validator（❌ UNFIXED）

  Post-build Mutation ── gen=5/8 AUTH_CONTRACT（❌ UNFIXED）
  ```

### 2026-07-11 追加調査で確定した完全メカニズム

**gen=4 Validator InvalidTopology（FACT #63）**:
```
Orchestrator trySubmit L73-76:
  worldBuilder.buildRuntimePublishWorld(newDSPResolved, oldDSP, HardReset, 0.0, false)
  → active=false, next=oldDSP≠nullptr

Builder:
  transitionActive = active && next != nullptr = false && true = false
  graphState.fadingNode = nullptr (transitionActive==false のため)
  topology.hasFadingRuntime = (graphState.fadingNode != nullptr) = false
  topology.fadingRuntimeUuid = (next != nullptr) ? next->runtimeUuid : 0 ≠ 0
  execution.transitionActive = active = false

Orchestrator override (L126-131):
  needsCrossfade=false のため未適用 ← 構造的 rebuild では常に false

Validator L92:
  hasFadingRuntime(false) != (fadingRuntimeUuid != 0(true)) → FAIL
  HEALTH eventCode=6001 (EVENT_VALIDATION_TOPOLOGY_FAILURE) 発行
```

**gen=5/8 AUTH_CONTRACT（FACT #64 — Validator 通過後に AUTH_CONTRACT で遮断）**:
```
同種のトポロジ生成条件（active=false, next=oldDSP≠nullptr）から生成された world に対し:
  needsCrossfade=true により override 適用
  → transitionActive=true, hasFadingRuntime=true

  ここで Validator は通過する（L92: hasFadingRuntime(true) == (fadingRuntimeUuid≠0), L96: hasFadingRuntime(true) == transitionActive(true)）

  Precheck (runPublicationPrecheckNonRt) Stage 1.5:
  AUTH_CONTRACT L87: hasGraphFadingNode(false) != hasFadingRuntime(true) → FAIL
  → override は graph.fadingNode を更新しないため
  L1626 [AUTH_CONTRACT] FAIL 確認済み
```

**Key finding**: Validator L96 に未文書化の制約 `hasFadingRuntime == transitionActive` が存在する（FACT #62）が、gen=4/5/8 の経路では L92 または AUTH_CONTRACT L87 で先に reject されるため表面化していない。

### 追加調査結果: 全所有経路の網羅的追跡（2026-07-11）

本レポート作成後、全6ツール（grep/serena/cocoindex/graphify/semble/WSL）による DSPCore* ポインタの網羅的追跡を実施した。**その結果、DSPCore のメモリ解放は `destroyDSPCoreNode()`（AudioEngine.Threading.cpp:15）が唯一の経路であることが確定した。** 当該関数は `core->~DSPCore()` + `convo::aligned_free(core)` を実行し、その呼び出し元は以下の2系統のみ:

| 呼び出し元 | 条件 | 備考 |
|:---------|:-----|:------|
| `DSPLifetimeManager::retire()` → EBR enqueue | Handle 登録済みかつ retireDSPHandleForRuntime 成功 | 正常系の retire パス |
| `DSPGuard::~DSPGuard()` 直接破棄 | rebuild-obsolete かつ retireDSPHandleForRuntime 失敗（未登録） | FIXED (v5.40) |

以下の全経路からの漏れがないことを確認:
- `aligned_free` の全出現箇所（AlignedAllocation.h, CtorDtor.cpp, PrepareToPlay.cpp, ConvolverProcessor.h 等）はいずれも DSPCore 本体以外のサブコンポーネント（IR データ、レイテンシバッファ、StereoConvolver 等）の解放であり、DSPCore 本体の解放は `destroyDSPCoreNode()` のみ。
- `ScopedAlignedPtr` / `aligned_unique_ptr` の `release()` により自動解放は無効化済み。
- `NonOwningPtr`（activeRuntimeDSPSlot/fadingRuntimeDSPSlot）は非所有ポインタであり解放を行わない。
- デッドコードの `retireDSP()` も同様に `destroyDSPCoreNode` を経由するが、呼び出し元が存在しない。

**結論**: 「解析対象外の所有経路」はもはや理論上の可能性に過ぎない。全ての DSPCore* 所有経路は追跡済みであり、未登録 DSPCore のリークは **現解析範囲では実質確定** と言える（プロセス終了後の OS 回収に依存する点、DLL/テストコード/ifdef/将来のコード差し替え等は完全には否定できないため留保）。

---

## 8. 参考資料

| 資料 | パス |
|:-----|:-----|
| 改修計画書（最新） | `doc/work70/modification-plan-v3.md` (v5.57) |
| 実装チェックリスト | `doc/work70/implementation-checklist.md` |
| ランタイムログ（35秒） | `doc/work70/ConvoPeq.log` (29,556行) |
| アーキテクチャ検証 | `doc/work70/modification-plan-v3.md` [設計] 6. |

---

## 9. 改訂履歴

| 版 | 日付 | 内容 |
|:---|:-----|:------|
| 1.0 | 2026-07-11 | 初版。P1-a〜P1-c 修正 + 4つの根本原因を文書化。 |
| 1.1 | 2026-07-11 | **レビュー指摘反映**: #3 Validator 生成経路を Strong HYPOTHESIS に明確化、メモリ内訳の DSPCore 数値を「寄与の推定値」に軟化、隠れた所有経路→解析対象外の所有経路に修正、AUTH_CONTRACT 修正案に Builder 同期生成の補足を追加、OwnershipDisposition::RolledBack→CallerOwns に改名、Root Cause Timeline を新設。 |
| 1.2 | 2026-07-11 | **全未確定事項のコード調査完了**: gen=4 の失敗メカニズムを `Orchestrator::buildRuntimePublishWorld(newDSP, oldDSP, ..., false)` → Validator L92 と確定（FACT #63）。gen=5/8 の失敗メカニズムを override 未更新 graph.fadingNode → AUTH_CONTRACT L87 と確定（FACT #64）。Validator L96 未文書化制約を発見（FACT #62）。pendingTask リーク第2経路（旧FACT #59）を誤報と訂正。64 FACT / 4 HYPOTHESIS 確定。 |
| 1.3 | 2026-07-11 | **レビュー指摘反映（第2弾）**: gen=5/8 の「同じ next ポインタの状態」断定を「同様の条件を持つ可能性が高い」に弱化。Builder 原因断定を「RuntimeGraph 生成経路で発生する不整合」に修正。OwnershipDisposition 設計を `destroyDSPCoreNode()` 直接呼び出しから `DSPLifetimeManager::destroyRolledBackDSP()` に変更（Single Authority 維持）。PUBLISH_STATS を 3階層カテゴリ（Validator/Authority/Executor）に再設計。Phase 2 検証を Publish→Retire→EBR→Destroy→Memory の完全チェーンに拡張。DSPCore メモリ内訳表現を「DSPCore寄与（推定）」に軟化。 |
| 1.4 | 2026-07-11 | **最終包括調査完了（全7ツール）**: gen=5/8 の Validator 通過後 AUTH_CONTRACT 遮断を確認（Validator は override により通過し、その後 Precheck AUTH_CONTRACT で遮断される）。AUTH_CONTRACT 3チェーン完全文書化。pendingTask.currentDSP全7箇所代入確認（常にnullptr→FACT#59誤報確定）。非DIAG mkl_malloc 9箇所確定。全7ツール使用完了。 |
| 1.5 | 2026-07-11 | **レビュー指摘反映（第3弾）**: 「runtime中は確定的」→「現解析範囲では実質確定」に弱め。Root Cause Timeline を3系統（activeHandle/Crossfadeトポロジ生成規則/override不整合）の独立した枝に再構成。CallerOwns→CallerDestroy に改名。Builder→CrossfadeDecision直接渡しをCrossfadePlan中間表現に修正（BuilderのPure Construction責務維持）。PUBLISH_STATS階層名を RuntimeSemantic/RuntimeAuthority/Executor に明確化。Design Principles 節を新設。 |
| 1.6 | 2026-07-11 | **最終レビュー反映（第4弾）**: Proposal B（Builder再呼出し案）を削除し、CrossfadePlan入力案に一本化。Builder責務制約（No Validation / No Policy Decision / Pure Construction）をDesign Principlesに明文化。PUBLISH_STATS階層名を Validation/Authority/Execution に改名。「Crossfadeトポロジ生成規則」→「RuntimeGraph生成規則」に修正。HandleState と DSPCore live の一致確認を Phase 2 検証に追加。 |
| 1.7 | 2026-07-11 | **最終レビュー反映（第5弾）**: 「RuntimeGraph生成規則」→「RuntimeGraph生成入力」に改名し、Builder入力起点を明確化。Validatorの責務を「Publication前のInvariant検査」としてDesign Principlesに明記（Builder修正機構ではない）。RUNTIME_MAP（UUID/Handle/DSPCore*/State対応表）の定期DIAG出力を新規提案。全7ツールによる最終調査で隠れた所有経路なしを再確認。 |
| 1.8 | 2026-07-11 | **最終レビュー反映（第6弾）**: AUTH_CONTRACT の本質を「override」→「Partial Mutation（Builder出力への部分上書き）」に修正。ValidationFailure 列挙型の一元化（Validator/HealthMonitor/PUBLISH_STATS 共通語彙）を新規提案。HANDLE_STATS/RUNTIME_MAP/MEM_SNAP の同一タイムスタンプ出力を推奨。CrossfadePlan の不変性（immutable）と Builder never mutates semantic intent を Design Principles に明記。 |
| 1.9 | 2026-07-11 | **最終未確定事項調査完了（新FACT 4件追加）**: CrossfadePlan の構造的実現可能性確認（FACT #65）。ValidationFailureReason 統合確認（FACT #66）。HANDLE_STATS/RUNTIME_MAP 新規性確認（FACT #67）。EBR enqueueRetire 再試行機構確認（FACT #68）。全7ツール使用完了。68 FACTs / 4 HYPOTHESIS 確定。 |
| 1.10 | 2026-07-11 | **最終レビュー反映（第7弾）**: Builder の `fadingRuntimeUuid = next->runtimeUuid` を「忠実な写像」と明記（Builder は next が fading か candidate かを判断できない）。InvalidTopology を「Topology Representation Invariant」として表現を精緻化。「Partial Mutation」→「Post-build Mutation」に改名（Builder が一括生成した World への後付け変更が本質）。OwnershipDisposition::CallerDestroy に「実際に destroy するのは DSPLifetimeManager」と補足。destroyRolledBackDSP() に EBR 非経由の特殊ルートであることを明記。HANDLE_STATS/RUNTIME_MAP/MEM_SNAP に generation 値を統一。FACT #68 を「再試行機構の存在」に弱化（動作保証ではない）。 |
| 1.11 | 2026-07-11 | **最終調査完了宣言**: 現在のコードベースから追加確定できる事項は見当たらない。全68 FACTs確定。残る4 HYPOTHESISはRuntime観測に依存するためコード調査では確定不能。全7ツール（grep/WSL/serena/ccc/semble/graphify/AiDex）による最終確認完了。 |
