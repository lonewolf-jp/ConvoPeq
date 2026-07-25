# ConvoPeq Runtime OS 改修計画書

**策定日**: 2026-07-24
**改訂日**: 2026-07-25（第14版レビュー反映）
**現状基準**: v0.6.10 (277 source files, ~3.17MB)
**前提設計**: Practical Stable ISR Bridge Runtime（RTは判断しない・所有しない・解放しない / Authority単一 / Observer副作用禁止）

---

# Part 1. 設計書（実装計画）

> 実装を担当するプログラマに必要な情報のみを記載する。

---

## ISR 不変条件（完了定義）

以下の条件が全て成立した時点で ISR 完了とする。設計フェーズ完了ではなく、**不変条件の成立**で定義する。

```
ISR Invariants（全て成立 = ISR 完了）
  □ Publish経路は RuntimeCoordinator のみ（publishAndSwap → Coordinator 経由のみ）
  □ Retire経路は RuntimeCoordinator のみ（retireIntent → Coordinator 経由のみ）
  □ Crossfade判定は RuntimeCoordinator のみ（endCrossfade → Coordinator 経由のみ）
  □ RuntimeWorld は完全 Immutable（mutable 0件）
  □ RT ownership なし（Audio Thread は所有しない・判断しない）
  □ Observer 副作用なし（Publish/Retire/Delete を EventBus から発行しない）
  □ Validation 必須（Publish 前に Validator を通過）
  □ Rollback 禁止（一度 Committed された Transaction は戻せない）
```

---

## Authority Matrix（最終形）

| コンポーネント | 唯一の責務 | 禁止事項 |
|---|---|---|
| RuntimeIntent | UI/Automation/Preset/IR → Intent変換 | 直接Builder呼び出し |
| RuntimeBuilder | RuntimeWorld構築のみ | Publish/Retire/Crossfade/Store |
| RuntimePublicationValidator | RuntimeWorld検証のみ | 副作用（Publish/Retire/Delete） |
| RuntimeCoordinator | Publish/Crossfade/Retire **Decision Authority**（判定のみ）。実装はExecutorに委譲 | DSP実行/delete |
| RuntimeStore | RuntimeWorld保持・Atomic Publishのみ | Policy/Crossfade/Retire/Validation |
| Audio Thread | RuntimeWorld読取・DSP実行のみ | 所有/解放/判断 |
| CrossfadeRuntime | CrossfadePlan実行のみ | Decision保持 |
| EpochDomain | Retire/Reclaimのみ | delete |
| DeletionWorker | deleteのみ | 判断 |
| HealthMonitor | 観測/診断/回復/検証のみ | 直接Publish |

---

## 実装順序と依存関係

```
Phase 0（現状固定 + CIゲート）
  │
  ├──→ Phase 1（RuntimeIntent）
  │         │
  │         ▼
  │    Phase 2（Builder純化 + RuntimeCapability 導入可）
  │         │
  │         ▼
  │    Phase 3（Validator強化）
  │         │
  │         ▼
  │    Phase 4（Transaction導入）
  │         │
  │         ▼
  │    Phase 5A（Publish Authority）
  │         │
  │    ┌────┤
  │    │    │
  │    ▼    ▼
  │  5B   5C（Retire / Crossfade Authority）  ← 並列可
  │    │    │
  │    └────┘
  │         │
  │         ▼
  │    Phase 5D（friend class 削除）
  │         │
  │    ┌────┼─────────────────────┐
  │    │    │                     │
  │    ▼    ▼                     ▼
  │  Phase 6（Immutable）   Phase 7+8（Crossfade/Retire統合）
  │    │                          │
  │    ├──→ Phase 12（Schema）    ├──→ Phase 11（Shutdown）
  │    └──→ Phase 13（Generation）│
  │                               │
  │    Phase 9（Store確認）← Phase 5A 完了後
  │    Phase 10（Health拡張）← Phase 5A 完了後
  ▼
完了
```

### 優先順位

| 順位 | Phase | 理由 |
|---|---|---|
| S-1 | Phase 0 | 全ての前提。Authority Matrix の CI 化 |
| S-2 | include削減 + 循環依存チェック | 即時改善可能 |
| A-1 | Phase 1 (Intent) | 上流から着手 |
| A-2 | Phase 2 (Builder + RuntimeCapability) | Intent 受信後 |
| A-3 | Phase 3 (Validator強化) | Builder 出力の検証 |
| A-4 | Phase 4 (Transaction) | 上記 3 Phase 完了後 |
| A-5 | Phase 5A (Publish Authority) | ★最重要 |
| A-6 | Phase 5B + 5C (Retire + Crossfade) | 並列可 |
| B-1 | Phase 5D (friend削除) | 一本化完了後 |
| B-2 | Phase 7+8 (Crossfade/Retire統合) | Phase 5B 完了後 |
| B-3 | Phase 6 (Immutable) | Phase 5A-5C 完了後 |
| C-1 | Phase 9 (Store確認) | Phase 5A 完了後 |
| C-2 | Phase 10 (Health拡張) | Phase 5A 完了後 |
| C-3 | Phase 11 (Shutdown統一) | Phase 5A + Phase 8 完了後 |
| C-4 | Phase 12 (Schema) | Phase 6 完了後 |
| C-5 | Phase 13 (Generation) | Phase 6 完了後 |

---

## Phase 0 — 現状固定 + CIゲート

**目的**: Authority 分散を可視化し、CI ゲートを稼働させる。

**作業**:
1. `config/authority_inventory.json` 拡張（Publish/Retire/Crossfade/Atomic/World 更新箇所一覧）
2. CI スクリプト追加:
   - `tools/check_circular_includes.sh` — 循環依存ゼロ（ADR-009）
   - `tools/check_mutable_world.sh` — RuntimeWorld 配下 mutable のみ禁止（ADR-002）
   - `tools/check_coordinator_atomic.sh` — Coordinator 外部 atomic 現状記録（ADR-010）
3. ベースラインスナップショット作成

**CIゲート**:
```bash
check_circular_includes:  python3 tools/check_circular_includes.py src/ → 0 サイクル
check_mutable_world:      grep -rn " mutable " src/audioengine/FrozenRuntimeWorld.h 2>/dev/null | wc -l → 0
# ★ 注: RuntimeWorld は独立したファイルではなく RuntimePublishWorld (=RuntimeState) として AudioEngine.h に定義。
#   mutable が存在する箇所は FrozenRuntimeWorld.h のみ CI で監視する。
check_coordinator_atomic: grep -rn "std::atomic" src/ --include="*.h" | grep -v "RuntimeCoordinator|RuntimePublicationCoordinator" | wc -l → 現状件数（629）を記録。Phase 5A 後に新規追加禁止
```

**受け入れ条件**:
- [ ] Authority Inventory JSON がパース可能
- [ ] CI ゲート 3 本が動作し、現状 FAIL を正しく検出
- [ ] ベースライン保存済み

---

## S-2: include削減 + 循環依存チェック

**目的**: AudioEngine.h の include を整理し、循環依存がないことを確認。

**削減対象**（実測確認済みのみ）:
```cpp
"LatticeNoiseShaper.h"     // 前方宣言 + .cpp 移動の可能性
"TruePeakDetector.h"       // 使用箇所が限定的
"LoudnessMeter.h"          // 同上
"SimplePeakLimiter.h"      // include 自体は存在
"GenerationManager.h"      // 使用箇所確認要
```

**注意**: include 数の数値目標は撤廃。代わりに「循環依存ゼロ」を CI ゲートとする。

---

## Phase 1 — RuntimeIntent 層新設

**目的**: Builder 呼び出し経路を RuntimeIntent で統一。

**新規ファイル**:
```cpp
// src/audioengine/RuntimeIntent.h
// ★ 第10版レビュー反映: generation は Intent に持たせない。
//   ISR では generation は Builder が PublishCandidate を生成した時点で決定される。
//   Single Source は PublishCandidate::BuildMetadata::builderGeneration のみ。
//   Intent は correlationId と priority だけで十分。
enum class IntentKind { Parameter, Preset, IRLoad, Automation, Host, Midi, Osc, System };

// ★ 第12版レビュー反映: bool ではなく enum で優先度を表現。
//   将来 Background/Urgent 等が追加できるよう拡張性を確保。
//   ★ 第14版: Realtime → Urgent に変更。Intent は Decision を持たないため、
//   「即 Publish」を連想させる Realtime は不適切。Urgent は「優先処理依頼」の意味に限定される。
enum class IntentPriority : uint8_t {
    Background,
    Normal,
    High,
    Urgent
};

struct RuntimeIntent {
    IntentKind kind;

    // ★ 第11版レビュー反映: Payload はコピー可能・軽量であること。
    //   重いデータ（IR 波形等）は shared_ptr<const XXX> を経由する。
    //   サイズ制限は RuntimeIntent 全体を実装時の static_assert で確認する。
    //   ★ 第14版: Payload 単位の数値目標は仕様に書かない（ABI 依存のため）。
    //
    // ★ 第12版レビュー反映: Payload 数が増加した場合（10種類以上等）、
    //   variant の compile time/include/rebuild が悪化する。
    //   その場合は type-erasure（shared_ptr<const IntentPayloadBase>）
    //   への移行を検討する。現段階では variant で十分。
    variant<ParameterPayload, PresetPayload, IRLoadPayload, AutomationPayload,
            HostPayload, MidiPayload, OscPayload, SystemPayload> payload;
    CorrelationId correlation;
    IntentPriority priority{IntentPriority::Normal};
};
// ★ 注: generation は PublishCandidate::BuildMetadata::builderGeneration が Single Source。
//   Intent の生成時点では未確定のため、Intent に generation を持たせない。
```

**RuntimeIntentJournal の分離**:
- `RuntimeIntent.h`（audioengine/）: Intent の定義と発行。**Runtime の責務**
- `RuntimeIntentJournal.h`（DiagnosticsDomain 内）: 記録のみ。**Diagnostics の責務**
- Replay は別コンポーネント（ReplayEngine）に分離

**★ レビュー反映: ReplayEngine の Intent 再投入経路（Authority 境界）を Phase 1 で明文化**

ReplayEngine が Intent を再投入する場合、経路を明確にする:
```
ReplayEngine
    ↓
IntentIngress          // ★ 第10版: IntentIngress に改名（enqueue のみの責務）
    ↓ enqueue
IntentQueue（SPSC）
    ↓ dequeue
Coordinator
    ↓
Builder
```

- ReplayEngine は **IntentIngress 経由のみ** で Intent を再投入する
- ReplayEngine は **Coordinator に直接アクセスしない**（Authority 境界を維持）
- IntentIngress は **Queue へ積むだけ**。Decision Authority は持たない
- Coordinator が dequeue して Builder に渡す。Decision は Coordinator のみ
- これにより Replay が Authority を持つ危険を排除する

**CIゲート**:
```
grep -r "submitRebuildIntent" src/audioengine/ --include="*.cpp" | grep -v "RuntimeIntent" → 0
```

---

## Phase 2 — RuntimeBuilder 純化 + RuntimeCapability

**目的**: Builder の責務境界を明確化し、必要に応じて RuntimeCapability を導入。

**RuntimeCapability**: Phase 2 着手時に DSP 側の constexpr 分岐が必要かを再確認。**必要なければ保留**。

---

## Phase 3 — RuntimePublicationValidator 強化

**目的**: 5 種 → 10 種に拡張。

**追加検証**: validateProjection / validateRouting / validateLatency / validateDSPGraph / validateCrossfade

**★ レビュー反映: variant 肥大化対策（RuntimeIntent）**
Phase 1 で作成する RuntimeIntent の variant は、将来 8 種類の Payload が入る可能性がある。コンパイル時の型サイズを抑えるため、実装時に `static_assert` で各 Payload のサイズを確認する:
```cpp
// ★ 第10版レビュー反映: static_assert の数値は ABI/MSVC/Clang 依存で変わる。
//   設計仕様書に「128byte以下」と書かず、実装指針としてコメントに留める。
//   ビルド時の static_assert で実際のサイズを確認する方針。
//   例:
//   constexpr size_t kMaxIntentInlineSize = 128;
//   static_assert(sizeof(RuntimeIntent) <= kMaxIntentInlineSize);
```

**★ レビュー反映: Validator 分割余地の確保**

現在は `RuntimePublicationValidator` に 10 種を格納するが、将来の拡張性を考慮し、以下の分割パターンをコメントで文書化する:

```
// 将来の分割パターン（Phase 3+ で必要に応じて適用）:
//
// RuntimePublicationValidator (Facade)
//   ├── GraphValidator          (validateDSPGraph, validateTopology)
//   ├── LatencyValidator        (validateLatency)
//   ├── ProjectionValidator     (validateProjection, validateRouting)
//   ├── ResourceValidator       (validateResources, validateCrossfade)
//   └── SemanticValidator       (validateSemanticConsistency, validateOwnership, validateAuthority)
//
// 現段階: 全検証を RuntimePublicationValidator に集約。
// 分割の判断基準: Validator 同士が互いを参照し始めたら分割。
//   例: GraphValidator が LatencyValidator を呼び始めたら Facade 分割の合図。
//   責務数（4 つ等）はあくまで目安。依存方向が重要。
```

**将来拡張**: validateOwnership / validateAuthority も将来的に必要になる見込み。

---

## Phase 4 — RuntimeTransaction 導入

**目的**: Intent → BuilderResult → Validation → Plan → Token を Transaction として束ねる。

**★ 第4版レビュー反映: Transaction は World を所有しない（ISR 思想に整合する設計に再設計）**
### 将来拡張: RuntimeTransaction の分割パターン

将来、Transaction が巨大化した場合、以下のように分割できる:
```cpp
// 将来の分割パターン（責務ベースで判断）:
struct PublishTransaction {
    RuntimeIntent intent;
    std::unique_ptr<PublishCandidate> candidate;
    PublishToken token;
};
struct RetireTransaction {
    RuntimeIntent intent;
    RetirePlan plan;
    RetireToken token;
};
struct CrossfadeTransaction {
    RuntimeIntent intent;
    CrossfadePlan plan;
    CrossfadeToken token;
};
// 現段階: 統合型 RuntimeTransaction を使用。
// 分割の判断基準:
//   - Publish/Retire/Crossfade の生成タイミングが異なり始めたら分割
//   - 各 Transaction のフィールドが肥大化し、個別管理が必要になった場合
//   - sizeof は ABI/STL実装/パディングで変わるため、分割基準には使わない。
//
// ★ 第12版レビュー反映: さらに Transaction と PublicationDecision を分離可能。
//   以下のように「決定結果」を PublicationDecision に切り出すことで、
//   Transaction は Builder 成果物 + Validation のみを束ねる責務に純化できる:
//     RuntimeTransaction { Candidate, Validation }
//     PublicationDecision { PublishToken, CrossfadePlan, RetirePlan }
//   現段階では統合。将来、PublicationDecision が独立ライフサイクルを持ったら分割。
```
### 設計原則

ISR の Transaction は **Publish Candidate（Builder の生成物）を保持する**ものであり、**World の実体や所有権とは直接結びつかない**。

```
旧案（❌ 依存が逆戻りする）:
  Intent → Transaction → World（直接所有 / shared_ptr）
  ↑ Transaction が World を知ることで、依存方向が逆転する

新案（✅ ISR 思想に整合）:
  Intent → Transaction → PublishCandidate（Builder の生成物）
  ↑ Transaction は PublishCandidate を保持（所有権は unique_ptr move）
  ↑ PublishCandidate は World の不変ビューを含むが、Transaction は World の実体を直接認識しない
  ↑ Coordinator が Candidate を受け取り、Store に渡す
```

**PublishCandidate（Builder の生成物）**:

★ 第5版レビュー反映: PublishCandidate は「Builder 成果物」のみを保持する。
CrossfadePlan / RetirePlan は「Publication Decision」に近い情報のため、
Transaction 側に分離する。これにより Builder 責務がさらに純化される。

```
Builder 生成物:              Publication 計画（Transaction 側）:
  RuntimePublishWorld          CrossfadePlan
  BuildMetadata                RetirePlan
  builderGeneration            (Validation, Token 等)
```

```cpp
// src/core/PublishCandidate.h

// ★ レビュー反映: BuildMetadata を復活。snapshotId + builderGeneration をまとめる。
//   Phase 13 の RuntimeMetadata と整合するため、Candidate 内のメタデータは
//   「Builder が Publish するために必要な情報」のみを BuildMetadata に集約。
struct BuildMetadata {
    uint64_t builderGeneration;   // Builder の論理世代（Builder Authority）
    CorrelationId correlation;    // 元 Intent の追跡用（Diagnostics にも使用）
    // ★ 第14版: snapshotId は廃止。採番 Authority の曖昧さを排除。
    //   Publish 順序は Coordinator が publishSequence (RuntimeMetadata) で管理する。
    //   Builder は generation のみを Authority として持つ。
    // 注意: buildDurationSec / peakCpuLoad 等は DiagnosticsDomain の責務

struct PublishCandidate {
    // Builder が生成した不変の World ビュー。
    // 所有権は Candidate が持つ（unique_ptr）。
    // Coordinator は Candidate を受け取り、World を Store に渡した後、
    // Candidate は破棄される。
    std::unique_ptr<const RuntimePublishWorld> world;

    // Builder が生成した構築メタデータ（Builder が Publish するために必要な情報のみ）
    BuildMetadata metadata;
};
```

// ★ 第14版レビュー反映: Phase4 から PublicationDecision を導入する。
//   Transaction は Builder 成果物 + Validation のみを束ねる責務に純化。
//   PublicationDecision が Coordinator の Decision 結果（Token+Plan）を保持する。
//
//   RuntimeTransaction { Intent, Candidate, Validation }
//       ↓ Coordinator::commit()
//   PublicationDecision { PublishToken, CrossfadePlan, RetirePlan }
//
//   これにより Builder 成果物と Coordinator 決定が明確に分離される。

```cpp
// src/core/PublicationDecision.h
struct PublicationDecision {
    PublishToken token;
    CrossfadePlan crossfadePlan;
    RetirePlan retirePlan;
    CorrelationId correlation;
};

// src/core/RuntimeTransaction.h
// ★ 第11版レビュー反映: seal() は private。friend class RuntimeCoordinator で制限。
//   Coordinator だけが状態遷移を確定する (Decision Authority)。
class RuntimeTransaction {
public:
    RuntimeIntent intent;
    std::unique_ptr<PublishCandidate> candidate;
    ValidationResult validation;

private:
    friend class RuntimeCoordinator;
    // Coordinator::commit() 内でのみ呼ばれる。tx.seal() → Coordinator が PublicationDecision を生成
    void seal() noexcept;
};
enum class TransactionResult { Committed, Rejected, Deferred };  // RolledBack は廃止（ISR Invariant: Rollback 禁止）
```

**寿命シナリオ（修正後）**:
```
Builder が所有 → PublishCandidate 生成 → Transaction が candidate を受け取り move
↑ unique_ptr で所有権が移動。Builder は候補を渡すと解放。
↑ Transaction → Coordinator → Store の順で所有権が移動。
↑ Store が World を保持し、Audio Thread が参照する。
```

**★ unique_ptr 採用の根拠（第4版レビュー反映）**:
- Builder は `aligned_unique_ptr<const RuntimePublishWorld>` を返す（現状コードと一致）
- 所有権は move で移動。shared_ptr による寿命共有は不要
- Builder と Transaction が別スレッドで生存する場合でも、unique_ptr move は安全（PublishCandidate が中間バッファとなるため）
- `shared_ptr` を使用する唯一の理由は「Builder 破棄後も World を参照したい」場合だが、ISR 設計では Store が World を保持するため、Builder の寿命に関係なく World は生存する

**★ レビュー反映: PublishCandidate の設計判断**
- CrossfadePlan / RetirePlan は Candidate に含めない。Builder は Specification → World の写像のみ。CrossfadePlan は Coordinator が World から生成する。RetirePlan は EpochDomain の責務。
- candidate は `unique_ptr<PublishCandidate>` で所有権を移動。Builder が所有 → Transaction が受け取り → Coordinator が Store に渡す。

**既存 FSM の統合**: ISRRuntimeSemanticSchema.h の `SemanticTransactionState` を内部状態として利用。

---

## Phase 5A — Publish Decision Authority（Coordinator 一本化 第一段階）

**目的**: RuntimeCoordinator が Publish を**決定する**唯一の Decision Authority になる。Coordinator 自身は Publish を**実行しない**。実行は PublishExecutor に委譲。

**責務境界（レビュー反映）**:
```
Decision（Coordinator）
    ↓ CoordinatorDecisionTag 発行
Executor（PublishExecutor）
    ↓ Publish 実行
RuntimeStore
    ↓ Atomic Publish
```

**God Object 防止**: Coordinator は Decision（判定・タグ発行）のみ。実装は専用 Executor に委譲。

**★ 第14版: Executor は Policy を持たない（Execute only）**
```
Coordinator
    Decision
Executor
    Execute only
Store
    Ownership
```
これにより God Object 化を防止する。Executor が Policy（どう実行するか）を判断することはない。

```cpp
// src/core/RuntimeCoordinator.h
class RuntimeCoordinator {
    CoordinatorDecisionTag issueTag(AuthorityKind kind, CorrelationId cid);
    TransactionResult commit(RuntimeTransaction&& tx);
    // tryRollback は廃止（ISR Invariant: Rollback 禁止）
    // Transaction は一度 Committed されたら戻せない。Rejected の場合のみ破棄可能。
};

// src/core/PublishExecutor.h    — Publish 実装
// src/core/CrossfadeExecutor.h  — Crossfade 実装
// src/core/RetireExecutor.h     — Retire 実装
```

**CoordinatorDecisionTag**: ★ 第14版: CoordinatorPrivateTag から改名。Decision Authority の通過証。
nonce/issuer は不要。

**★ 第11版レビュー反映: CoordinatorDecisionTag は Coordinator のみが生成可能**
```cpp
// ★ CoordinatorDecisionTag は Coordinator が発行する内部タグ。
//   認証・セキュリティ目的ではなく、Decision が Coordinator を経由したことを保証する。
//   Executor（PublishExecutor 等）はこのタグを受け取って初めて実行可能になる。
struct CoordinatorDecisionTag {
private:
    friend class RuntimeCoordinator;
    explicit CoordinatorDecisionTag(AuthorityKind kind, CorrelationId cid) noexcept
        : authorityKind(kind), correlation(cid) {}
    AuthorityKind authorityKind;
    CorrelationId correlation;  // デバッグ時追跡用（認証目的ではない）
};
//   これにより Executor が自分でタグを生成したり、AuthorityKind を偽装したりできない。
```

**★ レビュー反映: CoordinatorDecisionTag の用途を明確化**
CoordinatorDecisionTag は、Decision が Coordinator を経由したことを保証する内部タグであり、
認証・セキュリティ目的ではない。nonce/issuer を後から追加する必要はない。

**★ レビュー反映: AuthorityKind を追加**
Publish/Retire/Crossfade を区別できるように AuthorityKind enum を追加する:
```cpp
enum class AuthorityKind {
    Publish,
    Retire,
    Crossfade
};
// CoordinatorDecisionTag に AuthorityKind を含めることで、
// どの種類の Decision であるかを明確にできる。
```

**CIゲート（★ レビュー反映: grep ベースの暫定性を明記）**:
```bash
# ★ 暫定 CI（grep ベース）: リネームやマクロですり抜ける可能性あり。
#   また、コメント/テストコード/サンプルコードも引っかかるため、
#   実運用では src/tests/ doc/ 等を grep --exclude-dir で除外する。
#   Phase 5A 後に AST ベース（clang-tidy / clang-query）へ移行を検討。
#   現状は「可視化」として有効。「設計保証」には不十分。

grep -r "publishAndSwap" src/ --include="*.cpp" | grep -v "RuntimeCoordinator" → 0
# ★ 注意: retireIntent は \bretireIntent\b（単語境界）が必要。
#   emitRetireIntent / dequeuePendingRetireIntents / acknowledgeRetireCoordination は
#   設計上正当な関数名だが、単純部分一致で誤検出するため除外パターンを追加。
grep -rn "publishAndSwap\|\bretireIntent\b\|\bdelete\b" src/ --include="*.cpp" \
  | grep -v "RuntimeCoordinator\|DeletionWorker\|EpochDomain\|emitRetireIntent\|dequeuePendingRetireIntents\|acknowledgeRetireCoordination\|enqueueDeferredDelete\|deleteEQStatePtr\|deleteBandNodePtr\|=\s*delete\b" → 0  // Authority 漏れ検出
# ★ 注意: \bdelete\b は C++ の演算子 delete や deleteXxx 関数名にも一致する。
#   上記除外パターンで ISR 以外の delete を除去しているが、新規追加時は注意。
```

**注意**: SnapshotCoordinator は RT 層の Fade 擔当。**削除対象ではない**。
ただし SnapshotCoordinator::retireCurrentAndTarget() および enqueueWithRetry() は
直接 IEpochProvider::enqueueRetire() を呼んでいるため、Retire Initiation Authority を持つ。
Phase 5B では SnapshotCoordinator の Retire 経路も Coordinator 経由への移行を検討する。

**★ レビュー反映: 監視対象 API は実装変更に応じて更新する**
上記の grep パターンは暫定。実装のリネームや API 変更に応じて更新が必要。

---

## Phase 5B — Retire Authority

**目的**: Retire 決定権を Coordinator に一元化。

**注意**: SnapshotCoordinator が `retireCurrentAndTarget()` および `enqueueWithRetry()` を介して
直接 `IEpochProvider::enqueueRetire()` を呼んでいる。これにより SnapshotCoordinator は
Retire Initiation Authority を持つ。Phase 5B では以下の移行を検討する:
- SnapshotCoordinator の Retire 判断（いつ・どの Snapshot を retire するか）を
  RuntimeCoordinator の Decision 経由にする
- 実行 (=EpochDomain への enqueue) は SnapshotCoordinator 自身が行う権限を残す
  （RT 層 Fade 完了直後に同期的に retire する設計上、Coordinator 往復が XRUN リスクになるため）

**CIゲート**: `grep -r "\bretireIntent\b" src/ | grep -v "RuntimeCoordinator\|SnapshotCoordinator" → 0`

---

## Phase 5C — Crossfade Authority

**目的**: Crossfade 決定権を Coordinator に一元化。

**CIゲート**: `grep -r "endCrossfade" src/ | grep -v "RuntimeCoordinator" → 0`

---

## Phase 5D — friend class 削除

**目的**: friend class を段階的に削減する。

**★ レビュー反映: 全面禁止ではなく「理由が説明できる friend のみ許容」に変更**

現在の 13 箇所を分類し、削除可能/許容可能を判定する:

| friend | 所在 | 判定 | 理由 |
|---|---|---|---|
| AudioEngine → RuntimeState | AudioEngine.h:142 | 削除可 | BuilderToken で制御可能に |
| RuntimeBuilder → RuntimeState | AudioEngine.h:143 | 削除可 | Builder は public API 経由のみ |
| RuntimeBuilder → AudioEngine | AudioEngine.h:579 | 削除可 | Public メソッドで代替可能 |
| AudioEngine → InnerState | AudioEngine.h:2051 | 要検討 | 内部状態アクセスのため |
| RuntimePublicationOrchestrator | AudioEngine.h:3428 | 保留 | Coordinator 移行後に再評価 |
| PublicationExecutor | AudioEngine.h:3429 | 保留 | Executor パターン移行後に再評価 |
| DSPTransition | AudioEngine.h:3430 | **許容** | RT 層の DSP 処理。friend が自然 |
| NoiseShaperLearner | AudioEngine.h:3887 | **許容** | 学習アルゴリズム。内部状態アクセス必要 |
| EQEditProcessor | AudioEngine.h:3888 | **許容** | エディタ処理。内部状態アクセス必要 |
| RuntimePublicationStateOwner | RuntimePublicationState.h:92 | 削除可 | State オーナーパターン |
| RuntimePublicationOrchestrator | RuntimePublicationState.h:101 | 保留 | 「唯一の書込権限者」 |
| RuntimePublicationCoordinator | RuntimePublicationCoordinator.h:30 | 保留 | テンプレート内 |
| RuntimeStore | RuntimeStore.h:48 | 削除可 | WriteAccess で制御済み |

**運用方針**:
- **削除対象**: BuilderToken / WriteAccess で制御可能なものは Phase 5D で削除
- **許容対象**: NoiseShaperLearner / EQEditProcessor / DSPTransition は「理由が説明できる friend」として許容
- **保留**: Phase 5A〜5C の Coordinator 移行後に再評価
- **CI ゲート**: 削除対象（BuilderToken / WriteAccess で制御可能な friend）が **0 件** になること。許容対象のみ残ること。数は結果。

**★ 第14版レビュー反映: friend が許可される条件を ADR に文書化**
friend が許容されるのは以下のいずれかを満たす場合のみ:
1. **RT 性能維持**: DSPTransition 等、RT スレッドで呼ばれ、public API 経由ではレイテンシが増加する場合
2. **循環依存回避**: 2 つのクラスが互いの完全型を必要とするが、public API では解決できない場合
3. **WriteAccess では表現不能**: BuilderToken や WriteAccess のような専用アクセストークンが設計上不適切な場合
これらの条件は Phase 5D で再評価する。

---

## Phase 6 — RuntimeWorld 完全 Immutable

**目的**: RuntimeWorld/FrozenRuntimeWorld の mutable を全廃。**ISRWorld 配下のみ対象。**

**許容**: ISRRetire.h, ConvolverProcessor.h, DeferredRetireFallbackQueue.h の mutable（World とは無関係）

---

## Phase 7+8 — Crossfade/Retire 統合（同一マイルストーン）

Crossfade は Retire と強く結び付くため同一マイルストーン。

- Phase 7: CrossfadeAuthority 責務境界の文書化（既に dspProjection のみ使用）
- Phase 8: RetireIntent → EpochDomain → DeletionWorker の一本化

---

## Phase 9 — RuntimeStore 確認

既に publishAndSwap のみ。Phase 5A 完了後の確認フェーズ。

---

## Phase 10 — Runtime Health 拡張

自己修復シナリオ（RetireStallRecovery 等）の具体実装。

---

## Phase 11 — Shutdown Pipeline 統一

ISRShutdown 11 phases を全コンポーネントで一貫使用。

---

## Phase 12 — Runtime Schema 導入

RuntimeSchema v1/v2 変換 + SchemaMigrator。

---

## Phase 13 — RuntimeMetadata 導入（旧 RuntimeEpoch）

**型レベル分離（レビュー反映）**: 論理順序と物理時刻を型レベルで分離し、誤用を防止する。

```cpp
// src/core/RuntimeMetadata.h

// 論理順序（因果関係を保証）
// ★ 第14版レビュー反映: generation / publishSequence / retireSequence の役割を ADR に明記。
//   generation:      Runtime Builder の論理世代。Builder が PublishCandidate 生成時に決定。
//   publishSequence: Publish 実行順。Coordinator が採番。
//   retireSequence: Retire 実行順。EpochDomain が採番。
//
//   将来「generation++ と publishSequence++ を同時に更新する」誤用を防ぐため、
//   更新責務はそれぞれ単一の Authority に限定する。

struct LogicalGeneration {
    uint64_t value;
    bool isAfter(const LogicalGeneration& other) const noexcept;
};
struct LogicalSequence {
    uint64_t publishSequence;
    uint64_t retireSequence;
    // ★ 第5版レビュー反映: 必要になれば CrossfadeSequence を追加。
    //   Crossfade も Authority であるため、将来的に独立シーケンスが必要になる可能性あり。
    // uint64_t crossfadeSequence;
};

// 物理時計（論理順序とは独立）
struct PhysicalTimestamp {
    uint64_t us;  // microseconds since epoch（std::chrono::system_clock::time_point 相当）
    // ★ 第12版注: system_clock 相当であることを明記。steady_clock との混同を防止する。
};

// 統合型
struct RuntimeMetadata {
    LogicalGeneration generation;
    LogicalSequence sequence;
    PhysicalTimestamp timestamp;  // 論理順序とは独立。比較に使用しない。
};
```

**★ レビュー反映: Publish/Retire/Crossfade ごとのメタデータ分離**

将来的に以下のように分割する可能性がある:

```cpp
// 将来の分割パターン（必要に応じて適用）:
struct PublishMetadata {
    LogicalGeneration generation;
    LogicalSequence publishSequence;
    PhysicalTimestamp timestamp;
};
struct RetireMetadata {
    LogicalGeneration generation;
    LogicalSequence retireSequence;
    PhysicalTimestamp timestamp;
};
struct CrossfadeMetadata {
    LogicalGeneration generation;
    CrossfadeId crossfadeId;
    PhysicalTimestamp timestamp;
};
// 現段階: 統合型 RuntimeMetadata を使用。
// 分割の判断基準: Publish/Retire/Crossfade で異なるメタデータ字段が必要になった場合。
```

**★ レビュー反映: Phase 13 で LogicalGeneration へ移換する際の移行戦略**

LogicalGeneration は `uint64_t` のエイリアスとして導入する:
```cpp
// 移行期間中のみ使用。完了後は一括置換。
// ★ 第11版レビュー反映: 移行期間中は以下のいずれかを使用する。
//   方案A（推奨）: struct Generation { [[deprecated]] LogicalGeneration value; };
//   方案B: using Generation [[deprecated("Phase 13 完了後に LogicalGeneration へ一括置換")]] = LogicalGeneration;
//   方案A の方がコンパイラの deprecated 警告が出やすいが、
//   コード量が多い。方案B でも環境によっては警告が十分でない場合がある。
//   実際の導入時に方案A/B を決定する。
using Generation [[deprecated("Phase 13 完了後に LogicalGeneration へ一括置換")]] = LogicalGeneration;
```

これにより:
- Phase 13 以降は `LogicalGeneration` 型で型安全性を確保
- 既存コードは `Generation` エイリアスで引き続き動作（deprecated 警告付き）
- Phase 13 完了後に `Generation` → `LogicalGeneration` へ一括置換
- **注意**: `Generation` と `LogicalGeneration` が永久に共存しない。期限付きで管理する。
- **完了条件**: Phase 13 完了時に `using Generation` を削除し、全コードを `LogicalGeneration` へ置換する。

**設計判断**: timestamp を同列に置くと論理順序と物理時間が混ざるため、型で分離。`LogicalGeneration::isAfter()` は generation 単独比較のみ許可し、timestamp との混在を防止。

---

## CI ゲート定義書（全体）

### 設計方針（レビュー反映）

**grep ベースの限界**: 現状の CI ゲートは grep/regex ベース。リネームやマクロで容易にすり抜ける。grep は**簡易監視**としては有効だが、**設計保証**には不十分。

**★ レビュー反映: 全ゲートに「暫定」フラグを明記**

**移行計画**:
- Phase 0〜5A: grep ベースで即時導入（**暫定 CI**。可視化目的）
- Phase 5A 後: AST ベース（clang-tidy / clang-query）への移行を検討
- 最終目標: Authority 漏れを AST で保証

### ゲート一覧

| ID | 条件 | 導入 Phase | コマンド | 暫定/確定 | 移行先 |
|---|---|---|---|---|---|
| ADR-001 | publishAndSwap → Coordinator 限定 | 5A | `grep publishAndSwap src/ \| grep -v RuntimeCoordinator \| wc -l → 0` | **暫定** | clang-tidy |
| ADR-002 | RuntimeWorld 配下 mutable 禁止 | 0 | `grep -rn " mutable " src/audioengine/FrozenRuntimeWorld.h 2>/dev/null \| wc -l → 0` | 確定 | clang-tidy |
| ADR-003 | RuntimeGraph.h <250行 | 0 | `wc -l src/audioengine/RuntimeGraph.h → 250未満` | 確定 | — |
| ADR-004 | generation 単独比較禁止 | 13 | `grep -rn "generation ==" src/ \| grep -v "epoch\|isAfter" \| wc -l → 0` | **暫定** | clang-tidy |
| ADR-005 | friend class 削除完了 | 5D | 削除対象（BuilderToken / WriteAccess で制御可能な friend）が 0 件。許容対象のみ残る | 確定 | — |
| ADR-006 | Validator 検証項目網羅 | 3 | `grep -c "validateProjection(\|validateRouting(\|validateLatency(\|validateDSPGraph(\|validateCrossfade(" src/audioengine/RuntimePublicationValidator.cpp → 5以上` | **暫定** | AST ベース |
| ADR-007 | Diagnostics 逆依存禁止 | 0 | `grep -rn '#include.*AudioEngine.h' src/core/DiagnosticsDomain.h \| wc -l → 0` | 確定 | include-what-you-use |
| ADR-008 | Failure Injection 9系統 | 3 | `grep -c "FailureInject\|faultInject" src/tests/ \| wc -l → 9以上` | **暫定** | — |

**★ 第9版検証: FailureInjection は現状 0件**。Phase 3 で実装時に命名規則を ADR として先に固定する必要あり。

**★ 第10版レビュー反映: FailureInjection 命名規則を ADR で固定**
命名が揺れると grep CI がすぐ壊れる。以下を ADR として先に固定する:
```cpp
// ADR-012: FailureInjection 命名規則
// - 注入ポイント: `FailureInjectPoint`（クラス名、ファイル名）
// - 注入関数: `bool injectFailure(InjectKind kind) noexcept`
// - 注入種別: `enum class InjectKind { ... }`
// - 禁止: `FaultInject`, `FaultInjector`, `InjectFailure`, `InjectFault` 等の別名
```
| ADR-009 | 循環依存ゼロ | 0 | `python3 tools/check_circular_includes.py src/ → 0 サイクル` | 確定 | include-what-you-use |
| ADR-010 | Coordinator 外部 Authority 関連 atomic 現状記録 → Phase 5A 後新規追加禁止 | 0 | Phase 0: `grep -rn "std::atomic" src/ --include="*.h" \| grep -v "RuntimeCoordinator\|RuntimePublicationCoordinator" \| wc -l → 現状件数（629）記録。Phase 5A 後: 新規追加禁止 | **暫定→確定** | clang-tidy |
| ADR-011 | Authority 漏れ検出 | 5A | `grep -rn "publishAndSwap\|\bretireIntent\b\|\bdelete\b" src/ --include="*.cpp" \| grep -v "RuntimeCoordinator\|DeletionWorker\|EpochDomain\|emitRetireIntent\|dequeuePendingRetireIntents\|acknowledgeRetireCoordination\|enqueueDeferredDelete\|deleteEQStatePtr\|deleteBandNodePtr\|=\s*delete\b" \| wc -l → 0` | **暫定** | AST ベース Authority チェック |

---

# Part 2. 未確定事項・要決定事項

> 実装着手前に確定させるべき設計判断。

---

## U-1: RuntimeCapability の必要性

**現状**: DSP 側の version 分岐はシリアライズ/キャッシュフォーマット版本が主。constexpr 分岐の迫切性は低い。

**決定**: Phase 2 着手時に、DSP 側に constexpr 分岐が必要な箇所が実際に存在するかを再確認。**必要なければ保留**。

---

## U-2: Contract / Schema / Capabilities / Invariant の境界

**問題**: RuntimeContract / RuntimeSchema / RuntimeCapability / RuntimeInvariant が近い責務を持っている。今後 Contract が全てを吸収する危険がある。

**決定**: **ADR で境界を先に固定**する。各コンポーネントの責務を ADR として文書化し、境界を明確にする。

**候補 ADR 境界**:
- RuntimeContract: schemaVersion + capability + invariant + compatibility の組み合わせを保持
- RuntimeSchema: v1/v2 変換のみ。他の責務を持たない
- RuntimeCapability: constexpr 機能フラグのみ。Schema/Invariant とは独立
- RuntimeInvariant: 7 flags の Strict 検証のみ

---

## U-3: 循環依存の未検証

**現状**: .h 間の include リレーションは未解析。致命的な循環依存は存在しないと推定されるが、正式検証は未実施。

**決定**: Phase 0 の `check_circular_includes.sh` で検証。**Phase 0 着手時に確定**。

---

## U-4: Coordinator 外部 atomic の管理方針

**現状**: Coordinator 外部の std::atomic は約 629 箇所（AudioEngine.h 内 235箇所含む）。数値目標は撤廃し、現状記録 + 新規追加禁止に変更。

**決定**: 数値目標ではなく、以下の管理指標に変更する。
- **Phase 0〜5A**: grep ベースで現状を可視化（段階的削減）
- **Phase 5A 後**: 「Coordinator 外部への新規 atomic 追加禁止」を CI で保証
- **段階的削減**: AudioEngine.h の 235 箇所を主な削減対象とし、Phase 5A で Coordinator に集約した後に再評価

**根拠**: atomic は多い＝悪ではない。重要なのは Authority が分散していないこと。Coordinator 内部の atomic は正規な使用。

---

## U-5: Transaction と PublishCandidate の関係（★ 新規）

**問題**: Transaction が World を直接保持する設計は ISR 思想に反する。

**決定**: Transaction は `PublishCandidate`（Builder の生成物）を保持する。PublishCandidate は `unique_ptr<const RuntimePublishWorld>` を含むが、Transaction は World の実体を直接認識しない。

**根拠**:
- ISR の Transaction は「Publish 候補」を束ねるもの。「World そのもの」を束ねるものではない。
- Builder → Transaction → Coordinator → Store の順で所有権が移動する。
- `shared_ptr` は不要。所有権は move で管理する。

---

## U-6: friend class の運用方針（★ 新規）

**問題**: friend class を全面禁止するか、例外を許容するか。

**決定**: **「理由が説明できる friend のみ許容」**に変更。全面禁止は現実的でない。

**許容対象**:
- NoiseShaperLearner: 学習アルゴリズム。内部状態への直接アクセスが自然
- EQEditProcessor: エディタ処理。内部状態への直接アクセスが必要
- DSPTransition: RT 層の DSP 処理。friend が自然

**削除対象**: BuilderToken / WriteAccess で制御可能なものは削除。

**CI ゲート**: 削除対象（BuilderToken / WriteAccess で制御可能な friend）が **0 件** になること。許容対象のみ残ること。

---

# Part 3. Appendix

> 設計判断の裏付け、現状分析、レビュー履歴等の参考情報。

---

## A-1. 現状定量解析（2026-07-24 コードベース実測）

```
AudioEngine.h 4429行 / 4000→3000目標 ❌
class/struct 624定義（AudioEngine.h 59 / 50目標）
AuthorityClass:: 70件 / publishWorld 30件
AudioEngine.h atomic 235箇所（retire 24 / crossfade+latency 17 / publication 12 / generation 6 / other 170）
MemoryOrder 2358件（acquire 1074 / release 754 / acq_rel 253 / relaxed 277）
std::mutex 32件（AudioEngine.h 10件）
AudioEngine.h include 72（53プロジェクト + 19システム）
Complexity>20 8件（calcEQResponseCurve 47 / loadFromTextFile 47 / emitShutdownTrace 40 / 他5）
TODO/FIXME 1件（ISRShutdown 関連のみ） ✅
```

---

## A-2. ファイル構成現状

| ファイル | 状態 | 備考 |
|---|---|---|
| RuntimeStore.h | ✅ | Token必須は未実装 |
| EpochDomain.h | ✅ | 64 readers（**src/core/**） |
| DeletionQueue.h | ✅ | Custom MPMC 4096（**src/core/**） |
| CommandBuffer.h | ✅ | SPSC 1024（**src/core/**） |
| RuntimePublicationCoordinator.h | ✅ | 既存（**src/core/**） |
| SnapshotCoordinator.h | ✅ | RT層 Fade 擔当（削除しない）（**src/core/**） |
| RuntimeCoordinator.h | ❌ | Phase 5A で作成 |
| RuntimeMetadata.h | ❌ | Phase 13 で作成（旧 RuntimeEpoch） |
| RuntimeCapability.h | ❌ | Phase 2 で作成（任意） |
| AuthorityToken.h | ❌ | Phase 5A で作成（CoordinatorDecisionTag） |
| RuntimeContract.h | ❌ | Phase 6 で作成 |
| RuntimeInvariant.h | ❌ | Phase 6 で作成 |
| RuntimeSchema.h | ❌ | Phase 12 で作成 |
| DiagnosticsDomain.h | ❌ | Phase 10 で作成 |
| CrossfadeAuthority.h | ✅ | dspProjection 使用（DSPCore 非依存） |
| CrossfadeRuntime.h | ✅ | SPSC 32 |
| RuntimePublicationValidator.h | ✅ | 5種実装済み |
| RuntimeBuilder.h | ⚠ | Builder + Publish 混在（215行） |
| RuntimeGraph.h | ✅ | 95行 |
| FrozenRuntimeWorld.h | ✅ | 一部 Immutable 化済 |
| ISRRuntimeSemanticSchema.h | ✅ | 598行 |
| DeferredRetireFallbackQueue.h | ✅ | SPSC ring 化候補（**src/core/**） |

---

## A-3. CI ゲート現状

| ADR | 条件 | 判定 |
|---|---|---|
| ADR-001 | publishAndSwap → Coordinator 限定 | ✅ PASS |
| ADR-002 | RuntimeWorld 配下 mutable 禁止 | ⚠ 全 4 箇所（RuntimeWorld 配下は 0 件、CIゲートは PASS） |
| ADR-003 | RuntimeGraph.h <250行 | ✅ PASS |
| ADR-004 | generation 単独比較禁止 | ⚠ NOT_VERIFIED |
| ADR-005 | friend class 削除 | ❌ FAIL（13 箇所） |
| ADR-006 | Validator 10種 | ⚠ 5種実装済み、残 5種未実装 |
| ADR-007 | Diagnostics 逆依存禁止 | ❌ NOT_IMPLEMENTED |
| ADR-008 | Failure Injection 9系統 | ⚠ NOT_IMPLEMENTED（**現状 0件**） |
| ADR-009 | 循環依存ゼロ | ⚠ 未検証 |
| ADR-010 | Coordinator 外部 atomic 現状記録 → 新規追加禁止 | ⚠ Phase 0: 記録。Phase 5A 後: 禁止 |

---

## A-4. リスクと対策

| リスク | Phase | 確率 | 対策 |
|---|---|---|---|
| Phase 5A で Publish 経路を壊す | 5A | 高 | ShadowCompare で新旧二重実行 |
| Phase 5D で Accessor/mutable 増加 | 5D | 中 | 段階的削除 + CI 監視 |
| include 削減でリンクエラー | S-2 | 中 | 1ファイルずつ削除 + ビルド確認 |
| Phase 5B で XRUN 増加 | 5B | 中 | ISRRetire テストで事前検証 |
| 開発リソース（1人）で数年規模 | 全般 | 高 | Phase 5A 完了を第一マイルストーン |

---

## A-5. 進捗トラッキング

| Phase | ステータス | 備考 |
|---|---|---|
| Phase 0 | 📝 未着手 | 最初に着手 |
| S-2 | 📝 未着手 | Phase 0 と並行 |
| Phase 1 | 📝 未着手 | Intent→Builder |
| Phase 2 | 📝 未着手 | Builder 純化 |
| Phase 3 | 📝 未着手 | Validator 強化 |
| Phase 4 | 📝 未着手 | Transaction 導入 |
| Phase 5A | 📝 未着手 | ★最重要 |
| Phase 5B | 📝 未着手 | Retire Authority |
| Phase 5C | 📝 未着手 | Crossfade Authority |
| Phase 5D | 📝 未着手 | friend 削除 |
| Phase 6 | 📝 未着手 | Immutable |
| Phase 7+8 | 📝 未着手 | Crossfade/Retire 統合 |
| Phase 9 | ✅ 達成済み | Store 確認 |
| Phase 10 | 📝 未着手 | Health 拡張 |
| Phase 11 | 📝 未着手 | Shutdown 統一 |
| Phase 12 | 📝 未着手 | Schema 導入 |
| Phase 13 | 📝 未着手 | Metadata 導入 |

### マイルストーン

| ID | 定義 |
|---|---|
| M0 | Phase 0 完了 + CI ゲート稼働 |
| M1 | Phase 1〜3 完了（上流整備） |
| M2 | Phase 4 + 5A 完了（Transaction + Publish Authority） |
| M3 | Phase 5B〜5D 完了（Authority 一本化 + friend 削除） |
| M4 | Phase 6 + 7+8 完了（Immutable + Crossfade/Retire 統合） |
| M5 | Phase 10〜13 完了（最終 CI 全 PASS） |

---

## A-6. 新規作成ファイル一覧

| ファイル | 責務 | Phase |
|---|---|---|
| `src/core/RuntimeCoordinator.h` | Authority（窓口。Executor に委譲） | 5A |
| `src/core/PublishExecutor.h` | Publish 実装 | 5A |
| `src/core/CrossfadeExecutor.h` | Crossfade 実装 | 5A |
| `src/core/RetireExecutor.h` | Retire 実装 | 5A |
| `src/core/RuntimeMetadata.h` | generation/publishSequence/retireSequence + timestampUs | 13 |
| `src/core/RuntimeTransaction.h` | Intent + PublishCandidate + CrossfadePlan + RetirePlan + Validation + Token | 4 |
| `src/core/PublishCandidate.h` | Builder 生成物（World 不変ビュー + BuildMetadata） | 4 |
| `src/core/PublicationDecision.h` | Coordinator の Decision 結果（Token+Plan）。Transaction とは別 | 4 |
| `src/core/RuntimeCapability.h` | constexpr Baseline/Full/SafeMode | 2（任意） |
| `src/core/CoordinatorDecisionTag.h` | CoordinatorDecisionTag（Coordinator のみ生成可能） | 5A |
| `src/core/RuntimeContract.h` | schemaVersion+capability+invariant+compatibility | 6 |
| `src/core/RuntimeInvariant.h` | 7flags Strict | 6 |
| `src/core/RuntimeSchema.h` | v1/v2 変換 + SchemaMigrator | 12 |
| `src/core/DiagnosticsDomain.h` | Telemetry/Health/EventBus/Journal/Exporter | 10 |
| `src/core/ObserverEventBus.h` | Publish/Retire/Delete 禁止 | 10 |
| `src/core/DeletionWorker.h` | delete 唯一 | 8 |
| `src/core/ReplayEngine.h` | Intent Replay（Journal とは分離） | 10 |
| `src/audioengine/RuntimeIntent.h` | IntentKind + struct RuntimeIntent | 1 |
| `src/audioengine/IntentIngress.h` | Intent 発行（enqueue のみ。ReplayEngine はこれ経由のみ） | 1 |
| `src/audioengine/RuntimeIntentJournal.h` | Intent Journal（記録のみ） | 10 |
| `src/audioengine/RuntimeCoordinatorFacade.h` | AudioEngine 互換 API（移行後削除） | 0 |
| `tools/check_circular_includes.sh` | ADR-009 | 0 |
| `tools/check_mutable_world.sh` | ADR-002 | 0 |
| `tools/check_coordinator_atomic.sh` | ADR-010 | 0 |
| `tools/run_all_ci_gates.sh` | 全 CI ゲート一括実行 | 0 |

---

## A-7. 既存ファイル変更一覧

| ファイル | 変更内容 | Phase |
|---|---|---|
| AudioEngine.h | include削減 / Publish/Retire/Crossfade 経路再配線 / generation統合 | S-2, 5A-5C, 13 |
| AudioEngine.RebuildDispatch.cpp | Intent 経由に置換 | 1 |
| AudioEngine.Commit.cpp | Intent 経由に置換 | 1 |
| RuntimeBuilder.cpp | Intent 受付 + RuntimeCapability 分岐 | 1, 2 |
| RuntimePublicationValidator.cpp | 検証範囲拡張（10種） | 3 |
| RuntimeTransaction.cpp | seal() 実装 + FSM 統合 | 4 |
| ISRRuntimePublicationCoordinator.cpp | RuntimeCoordinator へ移行（5A 後削除） | 5A |
| ISRDebugRuntime.cpp | ShadowCompare Production 化 | 5A |
| RuntimeHealthMonitor.cpp | Recover サイクル追加 | 10 |
| ISRShutdown.cpp | パイプライン統一 | 11 |
| RuntimeStore.h | Token 必須化 | 5A |
| EpochDomain.h | retireSequence 管理追加 | 8 |
| DeferredRetireFallbackQueue.h | SPSC ring 化または統合 | 8 |

---

## A-8. 削除対象ファイル

| ファイル | Phase | 理由 |
|---|---|---|
| ISRRuntimePublicationCoordinator.h/.cpp | 5A | RuntimeCoordinator へ統合 |
| RuntimeCoordinatorFacade.h | 全 Phase 完了後 | 移行用。最終不要 |

**削除しない**: SnapshotCoordinator.h/.cpp（RT 層の Fade 擔当）

---

## A-9. 要調査事項の確定（コードベース調査結果）

### E-1: RuntimeCapability の導入根拠
version 分岐はシリアライズ/キャッシュフォーマット版本が主。DSP 機能分岐の迫切性は低い。→ U-1 参照。

### E-2: RuntimeGraphBuilder の必要性
不要。Builder の責務範疇内で十分（`makeRuntimeGraphState()` に委譲済み）。

### E-3: ShadowCompare の Production 化
ISRDebugRuntime に実装済み。追加作業不要。

### E-4: SemanticTransactionState との整合性
既存 FSM を基に拡張。新しい FSM は作り直さない。

### E-5: CrossfadePlan 生成責務（★ 第5版レビュー反映）

**変更**: CrossfadePlan は PublishCandidate から Transaction 側に分離された。
生成責務は Builder → Coordinator に変更。Builder は World 構築のみ。

```
旧: Builder が CrossfadePlan を生成 → PublishCandidate に格納
新: Coordinator が CrossfadePlan を生成 → Transaction の crossfadePlan に格納
```

**根拠**: CrossfadePlan は「Publication Decision」に近い情報。Builder は Specification → World の写像のみ。Crossfade の判断は Coordinator の責務。

### E-6: 循環依存
Phase 0 で正式検証。→ U-3 参照。

### E-7: Coordinator 外部 atomic 数
約 629 箇所（2026-07-25 実測）。数値目標は根拠が薄弱。→ U-4 参照。管理指標に変更。

### E-8: Transaction の World 依存（★ 新規・レビュー反映）
Transaction は `PublishCandidate` を保持する設計に変更。World を直接保持しない。→ U-5 参照。
ISR の Transaction は「Publish 候補」を束ねるもの。「World そのもの」を束ねるものではない。

### E-9: friend class の運用方針（★ 新規・レビュー反映）
全面禁止ではなく「理由が説明できる friend のみ許容」に変更。→ U-6 参照。
NoiseShaperLearner / EQEditProcessor / DSPTransition は許容。CI ゲートは削除対象が 0 件。

---

## A-10. レビュー反映サマリ

### 第1版レビュー反映（2026-07-25）

| # | 指摘 | 対応 |
|---|---|---|
| ① | Phase 5 が巨大すぎる | 5A/5B/5C/5D の 4 サブフェーズに分割 |
| ② | AuthorityToken の過設計 | CoordinatorPrivateTag に縮小 |
| ③ | friend 削除は早すぎる | Phase 5D に後ろ倒し |
| ④ | mutable CI は誤検知 | RuntimeWorld 配下のみに限定 |
| ⑤ | include 数の目標は危険 | 「循環依存ゼロ」に変更 |
| ⑥ | atomic 数の目標は不適切 | 「Coordinator 外部 ≤ 100」から「現状記録 + 新規追加禁止」に変更 |
| ⑦ | RuntimeCapability の導入が遅すぎる | Phase 12→Phase 2 に前倒し |
| ⑧ | IntentJournal は分離すべき | Intent と Journal を分離 |

### 第2版レビュー反映（2026-07-25）

| # | 指摘 | 対応 |
|---|---|---|
| ① | RuntimeCoordinator が God Object | Executor に委譲 |
| ② | Transaction が World を所有 | 参照のみ保持 |
| ③ | RuntimeEpoch の timestamp 混乱 | RuntimeMetadata に改名 |
| ④ | IntentJournal の Replay 大きすぎ | ReplayEngine として分離 |
| ⑤ | Contract/Schema 境界曖昧 | ADR で境界固定 |
| ⑥ | Authority 漏れ CI なし | ADR-011 を追加 |
| ⑦ | Phase 7 と 8 が分離 | 同一マイルストーンに統合 |
| ⑧ | 完了定義がフェーズベース | 不変条件で定義 |

### 第3版レビュー反映（2026-07-25）

| # | 指摘 | 対応 |
|---|---|---|
| ① | Transaction の World 参照が寿命切れリスク | `shared_ptr<const RuntimePublishWorld>` に変更。寿命戦略を明文化 |
| ② | Coordinator が Decision Authority であることを明記 | Authority Matrix と Phase 5A の記述を「Decision Authority」に統一 |
| ③ | CI ゲートが grep ベースで弱い | grep は簡易監視用。AST/clang-tidy への移行計画を追記 |
| ④ | atomic ≤100 が根拠薄弱 | 数値目標を管理指標に変更。「Coordinator 外部への新規追加禁止」に重点移行 |
| ⑤ | RuntimeMetadata の型分離 | LogicalGeneration / LogicalSequence / PhysicalTimestamp に型レベル分離 |
| ⑥ | Validator に Ownership/Authority 検証が必要 | 将来拡張として validateOwnership / validateAuthority を追記 |

### 第14版レビュー反映（2026-07-25）★ 本次

| # | 指摘 | 対応 | 優先度 |
|---|------|------|--------|
| ① | RuntimeTransaction に Builder成果物と Decision が混在 | **Phase4 から PublicationDecision を導入**。Transaction{Candidate+Validation}/Decision{Token+Plan} を分離 | 高 |
| ② | BuildMetadata の snapshotId 採番 Authority が曖昧 | **snapshotId を廃止**。Builder Authority は generation のみ。Publish 順序は Coordinator が publishSequence で管理 | 高 |
| ③ | RuntimeIntent Payload サイズ目標が ABI 依存 | **Payl単位の 64byte 目標を削除**。サイズ制限は RuntimeIntent 全体の static_assert で確認 | 中 |
| ④ | IntentPriority::Realtime が Decision を連想 | **Urgent に改名**。Intent は Decision を持たない | 中 |
| ⑤ | CoordinatorPrivateTag が C++ タグディスパッチに紛らわしい | **CoordinatorDecisionTag に改名** | 中 |
| ⑥ | CI の atomic 全数監視が Authority と無関係 | **Authority 関連 atomic（Publication/Generation/Retire）のみ監視する方針に修正**。全数は参考指標 | 中 |
| ⑦ | friend 許容条件が不明確 | **ADR に 3 条件（RT性能/循環依存/WriteAccess不可）を追記** | 低 |
| ⑧ | RuntimeMetadata の generation/sequence 役割が不明 | **ADR に役割と更新 Authority を明記**（Builder/generation, Coordinator/publishSequence, EpochDomain/retireSequence） | 中 |
| ⑨ | Executor が Policy を持ち得る可能性 | **Executor は Execute only（Policy なし）と ADR に明記** | 低 |

### 第13版ソースコード整合性検証反映（2026-07-25）

| # | 乖離 | 対応 | 影響 |
|---|------|------|------|
| ① | Mutex数が122→32 | **A-1 を 32 に修正**。std::mutex 実測値に更新 | 中 |
| ② | AudioEngine.h class/struct定義が68→59 | **A-1 を 59 に修正** | 低 |
| ③ | AuthorityClass 採点が70→74 | **A-1 を 74 に修正** | 低 |
| ④ | Coordinator外部 atomic が344→622 | **U-4/E-7/ADR-010/Phase0 CI を 622 に統一** | 中 |
| ⑤ | ADR-002 / Phase0 CI が src/core/RuntimeWorld.h を参照 | **src/core/RuntimeWorld.h は存在しないため除去**。RuntimePublishWorld は RuntimeState として AudioEngine.h に定義 | 高 |

### 第12版レビュー反映（2026-07-25）

| # | 指摘 | 対応 | 優先度 |
|---|------|------|--------|
| ① | RuntimeTransaction と PublicationDecision の分離余地 | **将来分離パターンをコメント追加**。Transaction{Candidate+Validation}/PublicationDecision{Token+Plan} | 中 |
| ② | RuntimeIntent::isHighPriority が bool | **IntentPriority enum（Background/Normal/High/Realtime）に変更** | 高 |
| ③ | RuntimeIntent variant 肥大化時の対策 | **type-erasure 移行方針をコメント追加**。Payload数増加時は shared_ptr<const IntentPayloadBase> へ | 中 |
| ④ | snapshotId の採番責務 | **Builder/Coordinator 両方あり得ることをコメント追加**。現状は Builder 採番 | 低 |
| ⑤ | CoordinatorPrivateTag に CorrelationId | **CorrelationId を追加**。デバッグ時追跡用（認証目的ではない） | 中 |
| ⑥ | Executioner → Executor 統一 | **修正済み** | 低 |
| ⑦ | PhysicalTimestamp に clock 種別 | **system_clock 相当であることを明記**。steady_clock との混同防止 | 低 |
| ⑧ | 他は良好 | **変更不要** | — |

### 第11版レビュー反映（2026-07-25）

| # | 指摘 | 対応 | 優先度 |
|---|------|------|--------|
| ① | RuntimeTransaction::seal() が public のまま | **private に変更。friend class RuntimeCoordinator で制限**。API と設計を一致 | 高 |
| ② | CoordinatorPrivateTag が誰でも生成できる | **private コンストラクタに変更**。friend RuntimeCoordinator で生成限定 | 高 |
| ③ | RuntimeIntent variant の肥大化方針 | **「Payload はコピー可能・軽量」を ADR として追記**。重いデータは shared_ptr 経由 | 中 |
| ④ | BuildMetadata に CorrelationId がない | **CorrelationId を追加**。SchemaVersion/CapabilityVersion 等の将来追加コメントも追記 | 高 |
| ⑤ | Validator 分割基準が責務数ベース | **「Validator 同士が互いを参照し始めたら分割」に変更**。依存方向が重要 | 中 |
| ⑥ | grep CI がテストコードまで引っかかる | **`grep --exclude-dir` で tests/doc を除外する注記を追加** | 低 |
| ⑦ | Generation alias の deprecated が不十分 | **struct Generation 方案もコメントで追記**。導入時に方案A/B を決定 | 低 |
| ⑧ | 他は良好 | **変更不要** | — |

### 第10版レビュー反映（2026-07-25）

| # | 指摘 | 対応 | 優先度 |
|---|------|------|--------|
| ① | RuntimeIntent に generation があると同期漏れ | **generation を Intent から削除**。PublishCandidate::BuildMetadata::builderGeneration が Single Source | 高 |
| ② | RuntimeIntentPublisher の名前が Publish と混同する | **IntentIngress に改名**（enqueue のみの責務） | 中 |
| ③ | seal() の呼び出し元が曖昧 | **Coordinator::commit() 内で seal() を呼ぶ**。Authority 境界を Coordinator に統一 | 中 |
| ④ | sizeof(RuntimeIntent)<=128 が設計仕様に書かれている | **static_assert は実装指針に留める**。ABI 依存の数値を設計書に書かない | 中 |
| ⑤ | ADR-006 が `validateProjection(.*world` で壊れやすい | **`validateProjection(` だけの存在確認に戻す**。AST 移行までの暫定 | 低 |
| ⑥ | FailureInjection 命名規則を ADR 固定 | **ADR-012 として命名規則を追加**（`FailureInjectPoint`/`injectFailure`/`InjectKind`） | 高 |
| ⑦ | 他は良好 | **変更不要** | — |

### 第9版ソースコード整合性検証反映（2026-07-25）

| # | 不整合 | 対応 | 影響 |
|---|--------|------|------|
| ① | AudioEngine.h 行数が 4424→4429 | **4429 に修正** | 低 |
| ② | TODO/FIXME が 0→1件 | **1件に修正**（ISRShutdown 関連） | 低 |
| ③ | SnapshotCoordinator パスが src/audioengine/ | **src/core/ に修正** | 中 |
| ④ | RuntimeTransition パスが src/core/ | **src/audioengine/ に修正** | 中 |
| ⑤ | EpochDomain パスが src/audioengine/ | **src/core/ に修正** | 中 |
| ⑥ | CommandBuffer パスが src/audioengine/ | **src/core/ に修正** | 低 |
| ⑦ | DeferredDeletionQueue パスが src/audioengine/ | **src/ に修正** | 低 |
| ⑧ | DeferredRetireFallbackQueue パスが src/audioengine/ | **src/core/ に修正** | 中 |
| ⑨ | FailureInjection が 0件（Planは9系統） | **Phase 3 で命名規則を ADR 固定** | 高 |
| ⑩ | RuntimeBuilder.h 行数未記載 | **215行を追記** | 低 |
| ⑪ | ISRRuntimeSemanticSchema.h 行数未記載 | **598行を追記** | 低 |

### 第8版レビュー反映（2026-07-25）

| # | 指摘 | 対応 | 優先度 |
|---|------|------|--------|
| ① | RuntimeIntentPublisher の Authority が曖昧 | **enqueue/dequeue の図を追加**。Publisher は Queue へ積むだけ。Coordinator が dequeue して Builder に渡す | 高 |
| ② | RuntimeTransaction の分割判断基準が曖昧 | **生成タイミングが異なり始めたら分割**に変更。具体的な判断基準を追加 | 中 |
| ③ | PublishCandidate の snapshotId が散在 | **BuildMetadata を復活**。snapshotId + builderGeneration を BuildMetadata に集約 | 中 |
| ④ | CoordinatorPrivateTag の用途が不明 | **認証・セキュリティ目的ではないことを明記**。内部タグであることの説明を追加 | 低 |
| ⑤ | Validator の static_assert が片手落ち | **全 Payload に対して static_assert を実施**。8 Payload + RuntimeIntent 全体 | 中 |
| ⑥ | ADR-006 が宣言のみで一致 | **実際の呼び出し（validateProjection(...world)）を確認する形に改善** | 中 |
| ⑦ | Phase 13 の deprecated alias が期限不明 | **完了条件に Generation alias 削除を追加** | 低 |

### 第7版レビュー反映（2026-07-25）

| # | 指摘 | 対応 | 優先度 |
|---|------|------|--------|
| ① | Transaction分割条件がsizeof 64B | **責務ベースに変更**。Publish/Retire/Crossfade が独立ライフサイクルになったら分割 | 高 |
| ② | RuntimeIntent variant サイズが片手落ち | **sizeof(RuntimeIntent) <= 128** に変更。全Payload対象 | 高 |
| ③ | PublishCandidate に snapshotId がない | **snapshotId を最初から含める** | 中 |
| ④ | Generation alias が永久に残る | **[[deprecated]] 付きに変更**。Phase 13 完了後に一括置換 | 中 |
| ⑤ | ADR-006 が validateProjection で誤検出 | **括弧付き `validateProjection(` に変更** | 中 |
| ⑥ | Authority検出 CI がAPI変更に脆弱 | **監視対象APIは実装変更に応じて更新する注記を追加** | 低 |
| ⑦ | ReplayEngine の Intent 再投入経路が未明 | **Phase 1 で Authority 境界を明文化**。ReplayEngine → RuntimeIntentPublisher 経由のみ | 高 |
| ⑧ | CoordinatorPrivateTag が AuthorityKind を区別できない | **AuthorityKind enum（Publish/Retire/Crossfade）を追加** | 中 |
| ⑨ | friend CI が件数ベース | **削除対象が0件・許容対象のみ残ること**に変更 | 中 |
| ⑩ | RuntimeCapability 保留判断 | **妥当と確認**。constexpr dispatch の必要性証拠なし | — |

### 第6版レビュー反映（2026-07-25）

| # | 指摘 | 対応 | 優先度 |
|---|------|------|--------|
| ① | PublishCandidate が Builder成果物全部になる可能性 | **将来拡張パターンをコメントで残す**。現段階では問題なし | 低 |
| ② | BuildMetadata に Diagnostics 寄りの情報が混在 | **BuildMetadata の責務を明確化**。Builder が Publish するために必要な情報のみ | 低 |
| ③ | RuntimeIntent payload の variant サイズ | **static_assert で各 Payload のサイズを確認する方針を追記** | 低 |
| ④ | LogicalGeneration 移行戦略 | **using Generation = LogicalGeneration; の移行期間を追記** | 低 |
| ⑤ | Phase 0 の atomic ゲートが矛盾 | **「現状件数（622）を記録」に統一**。Phase 5A 後に新規追加禁止 | 中 |
| ⑥ | PublicationDecision 抽象化 | **将来拡張パターンとして PublicationDecision 構造体を追記** | 低 |

### 第5版レビュー反映（2026-07-25）

| # | 指摘 | 対応 | 優先度 |
|---|---|---|---|
| ① | PublishCandidate に CrossfadePlan/RetirePlan が混在 | **Transaction 側に分離**。PublishCandidate は Builder 成果物のみ。Builder 責務をさらに純化 | 中 |
| ② | RuntimeIntent::generation が uint64_t | **Phase 13 で LogicalGeneration へ移換予定のコメントを追加** | 低 |
| ③ | freeze() の名称が曖昧 | **seal() に変更**。不変化の意味を明確化。代替案: transitionToFrozen() | 低 |
| ④ | Validator 分割条件が定量的 | **責務の増加を主基準に変更**。メソッド数はあくまで目安 | 低 |
| ⑤ | RuntimeMetadata に CrossfadeSequence がない | **追加余地のコメントを追加** | 低 |
| ⑥ | ADR-006 が validate\|check 件数で弱い | **期待検証項目ごとの存在確認に変更** | 中 |
| ⑦ | Phase 順序 | **現在の順序を維持**（評価済み） | — |
| ⑧ | E-5 CrossfadePlan 生成責務が矛盾 | **Builder → Coordinator に変更**。CrossfadePlan は Publication Decision  | 中 |
| ⑨ | TransactionResult::RolledBack が Invariant と矛盾 | **RolledBack を廃止**。ISR Invariant: Rollback 禁止に整合 | 高 |
| ⑩ | tryRollback() が Invariant と矛盾 | **tryRollback() を廃止**。Committed された Transaction は戻せない | 高 |

### 第4版レビュー反映（2026-07-25）

| # | 指摘 | 対応 | 優先度 |
|---|---|---|---|
| ① | Transaction が World を直接保持する設計 | **PublishCandidate パターンに再設計**。Transaction は Builder の生成物を保持し、World の実体とは直接結び付けない | 高 |
| ② | Builder が shared_ptr を保持する所有権モデル | **unique_ptr move パターンに変更**。所有権は move で移動。Store が World を保持するため shared_ptr は不要 | 高 |
| ③ | RuntimeIntent の巨大化対策 | **Small Object Optimization の注記を追加**。将来的に shared_ptr<const PayloadBase> への移行パスをコメントで残す | 中 |
| ④ | Validator の将来的な分割余地 | **分割パターンをコメントで文書化**（GraphValidator / LatencyValidator / ProjectionValidator / ResourceValidator / SemanticValidator） | 中 |
| ⑤ | RuntimeMetadata の将来拡張 | **PublishMetadata / RetireMetadata / CrossfadeMetadata の分割パターンをコメントで追加** | 中 |
| ⑥ | friend は全面禁止ではなく例外を許容 | **「理由が説明できる friend のみ許容」に変更**。CI ゲートは削除対象が 0 件 | 中 |
| ⑦ | grep ベース CI が暫定であることの明記 | **全ゲートに「暫定/確定」フラグを追加**。移行計画を明確化 | 低 |
| ⑧ | Phase 順序の見直し | **現在の順序を維持**。Transaction 導入後 Builder を直す方が実装は楽 | 低 |
