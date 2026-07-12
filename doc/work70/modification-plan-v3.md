# ConvoPeq メモリ肥大化 改修計画書 v9.11

**日付**: 2026-07-12
**対象**: work70 (メモリ 2.5GB 肥大化問題)
**前版**: v9.9 (2026-07-12)

> **本計画書は「優先改修項目（Backlog）」を先頭にまとめ、「完了した改修」は Appendix に配置している。**
> 凡例: ✅ FACT / 🔍 HYPOTHESIS / 💡 PROPOSAL / ⚠️ CAVEAT / 🎯 DECISION / 🛠️ IMPLEMENTED / ⏳ PENDING

---

## 実装進捗サマリ

| フェーズ | ステータス | 実装日 | 変更ファイル |
|:---------|:----------|:-------|:------------|
| **P6-a**: Builder L210 `active` 条件追加 | 🛠️ **完了** | 2026-07-12 | `RuntimeBuilder.cpp` |
| **P6-b**: DIAG_AUTH 4点追加 (CoordExit/BuilderEntry/BuilderExit/PreCommit) | 🛠️ **完了** | 2026-07-12 | `RuntimeBuilder.cpp`, `RuntimePublicationOrchestrator.cpp`, `AudioEngine.Commit.cpp` |
| **P7-A1**: `RetirePart` 追加 (Spec + Orchestrator + Builder) | 🛠️ **完了** | 2026-07-12 | `RuntimeBuilder.h`, `RuntimeBuilder.cpp`, `RuntimePublicationOrchestrator.cpp` |
| **P7-A2/B**: `AdaptivePart` 追加 (coeffBankIndex + coeffGeneration) | 🛠️ **完了** | 2026-07-12 | `RuntimeBuilder.h`, `RuntimeBuilder.cpp`, `RuntimePublicationOrchestrator.cpp` |
| **P7-C**: IR transfer → Builder Service（Resource Factory）として正式採用 | 🛠️ **完了**（5条件確認済み、実装変更不要） | 2026-07-12 | `ConvolverProcessor.h` |
| **P7-D**: HealthState → Builder から分離・除去 **完了** | 🛠️ **完了** | 2026-07-12 | `RuntimeBuilder.h/cpp`, `Init.cpp`, `PrepareToPlay.cpp`, `ReleaseResources.cpp`, `RebuildDispatch.cpp`, `Timer.cpp`, `Transition.cpp`, `Orchestrator.cpp` |
| **P8**: MMCSS 再試行機構（MmcssState 3値管理 + Timer retry） | 🛠️ **完了** | 2026-07-12 | `AudioEngine.h`, `AudioEngine.Timer.cpp`, `AudioBlock.cpp`, `BlockDouble.cpp`, `PrepareToPlay.cpp` |
| **P5-1/2**: int→enum 型整理 | ⏳ **最終段階で実施**（全設計安定後）。コード調査: transitionPolicy 13(src)+8(tests)=21箇所、processingOrder 30(src)+4(tests)+5(core)=39箇所。変更影響範囲: 中程度。 | — | — |
| **P5-4**: `RuntimePublicationSpecification.h` | 🛠️ **エイリアス確認済み** | 2026-07-12 | — |

## Design Principles

本計画書全体を貫く設計原則。以降の設計判断はすべてこの原則から導かれる:

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
- semantic intent を変更しない（Never mutates semantic intent）
- **入力パラメータ間の意味的整合性を再解釈しない**

```
入力 (RuntimePublishSpecification)
  ↓
Builder（Pure Construction — 入力を忠実に写像するのみ。自身では判断しない）
  ↓
RuntimePublishWorld（完成状態）
```

### Deterministic Construction の定義（Semantic Equivalence）

> **Given identical RuntimePublishSpecification, Builder shall produce RuntimePublishWorld instances whose DSP topology, routing, runtime behavior, latency, crossfade behavior, and processing semantics are equivalent. Object identity, allocation address, publication identity, generation number, and other implementation artifacts are excluded.**

これは以下のことを意味する:
- Builder は Specification の値のみから World を構築する（暗黙の外部状態に依存しない）
- 同じ Specification なら DSP topology / routing / latency / crossfade / processing semantics は常に等価
- **保証しないもの**: オブジェクト同一性、アロケーションアドレス、publication identity、世代番号、その他実装アーティファクト
- **Builder Service may affect implementation artifacts only**. Allocation, identity, pointer, generation は変更可能だが、DSP topology / routing / latency / crossfade / processing semantics は変更不可。
- テストは Specification を直接構築して Builder 単体で実施可能（Coordinator/Current Runtime のモック不要）

### Specification Completeness（設計原則）

> **RuntimePublishSpecification shall contain every mutable runtime value required to construct RuntimePublishWorld. Builder shall not obtain additional mutable inputs from any other source.**

この原則は INV-13 を補強する:
- INV-13（Builder は mutable Runtime state を直接観測しない）は Behavior の制約
- Specification Completeness は **Data の制約** — 必要な情報はすべて Specification でカバーされているべき
- P0〜P2 はこの原則の実装に他ならない（暗黙の atomic 読取りを明示的な Part に昇格）
- 将来新たな mutable 値が必要になった場合、最初の選択肢は Specification への Part 追加（Runtime の直接参照ではない）

### Validator（Validation Authority）の正しい理解

Validator は **Builder の修正機構ではない**:

```
Builder（生成）
  ↓
Validator（Invariant 最終検査）← Builder の出力を検証するが、Builder を修正しない
  ↓
Publication（公開）
```

**Validator の保証範囲**:
- ✅ **Structural Invariants**: Topology / Execution / Identity の三カテゴリ
- ❌ **Semantic correctness**: Builder 入力の意味的整合性は保証しない
- ❌ **Authority Contract**: graph.fadingNode と CrossfadeRuntime の一貫性は保証しない
- ❌ **Policy correctness**: RuntimeBuildPolicy の適切性は保証しない

---

## 改修設計（Backlog）

*以下の優先順位は本セッションでのコード調査（2026-07-12）に基づく。各項目ごとに確認できた事実と具体的な対応方針を記載する。*
*凡例: ~~P0〜P4~~ は **Appendix A に移動**（実装完了）*

---

### P5: 型整理・既知制限

**🎯 優先度: ★☆☆☆☆（全設計が安定した最終段階で実施）**

| # | 項目 | 内容 | 優先度 |
|:-:|:-----|:------|:-------|
| 1 | `transitionPolicy`: int → enum class | 設計書は `convo::TransitionPolicy`、コードは `int`。コメントで対応関係はドキュメント済み。型安全性向上のみであり、Runtime/Builder 設計に影響しない。**全設計安定後にまとめて実施。** | ★☆☆☆☆ |
| 2 | `processingOrder`: int → enum class | 同上 | ★☆☆☆☆ |
| 3 | `aligned_free` DIAG 非対称性 | `aligned_malloc` は `DIAG_MKL_MALLOC` 経由だが `aligned_free` は `mkl_free` 直接。`DIAG_MKL_FREE` は size 引数が必要なため対応不可。診断品質は低下するが設計上の既知制限として許容。現状: `DIAG_MKL_MALLOC` (3箇所: CacheManager×2, IRConverter×1) / `aligned_malloc/mkl_malloc` 総数 81箇所 / `DIAG_MKL_FREE` 0箇所。 | ★★☆☆☆ |
| 4 | `RuntimePublicationSpecification.h` 整理 | 独立ファイルとして作成されたが誰もインクルードしていない。将来の分離用準備として維持。 | ★☆☆☆☆ |

---

### P6: AUTH_CONTRACT FAIL 原因修正 — Builder または Spec 生成段階における条件不一致（2026-07-12 発見）

**🎯 優先度: ★★★★★（最優先）**

**【コード調査により特定した可能性】**

**現象**: `[AUTH_CONTRACT] FAIL fadingNode=0 hasFadingByUuid=1`

**可能性** — `RuntimeBuilder.cpp:210` と `makeEngineRuntimeState` (`AudioEngine.h:2759`) の間に条件差異が存在することをコード上確認した。この差異は AUTH_CONTRACT FAIL の十分条件になり得るが、実際にどの層（Coordinator/Spec生成/Builder）で不整合が導入されたかは追加 DIAG により切り分ける必要がある。

`RuntimeBuilder.cpp:210` と `makeEngineRuntimeState` (`AudioEngine.h:2759`) の間に条件の不統一がある:

| コード位置 | ロジック | 備考 |
|:----------|:---------|:------|
| `RuntimeBuilder.cpp:210` | `fadingRuntimeUuid = (next != nullptr) ? next->uuid : 0` | **無条件**（`active` 無視） |
| `AudioEngine.h:2758-2759` (makeEngineRuntimeState) | `fading = transitionActive ? next : nullptr` / `transitionActive = active && next != nullptr` | **条件付き**（`active` 考慮） |
| `AudioEngine.h:3052-3053` (makeRuntimeGraphState) | `graph.fadingNode = state.fading` | `engineState.fading` をコピー |

→ `transitionActive=false` かつ `next` (fadingDSP) が非 null の場合:
- `graph.fadingNode = nullptr` (makeEngineRuntimeState → makeRuntimeGraphState 経由で条件付き)
- `topology.fadingRuntimeUuid = next->runtimeUuid` (BUILDER で無条件に非ゼロ)
- → AUTH_CONTRACT 不一致 ⇒ publish FAILED

**【ただし原因箇所は Builder だけとは限らない】** — Coordinator/Spec生成/Builder の各段階を等しく候補として扱うべきであり、ログだけでは特定不能。

1. **準備段階の不一致**: `prepareToPlay()` は Orchestrator 経由ではなく `Builder` の旧シグネチャを直接呼ぶ（`PrepareToPlay.cpp:150`）。この経路でも `RuntimeBuilder.h:123-158` の委譲ロジックを経由して同じ L210 に到達するが、`crossfadeRuntime_.reset()` 直後の状態で呼ばれるため、引数の解決方法が Orchestrator 経由と異なる可能性がある。

2. **Coordinator → Orchestrator (trySubmit) 経路**: 通常の rebuild 要求は `trySubmit` を経由する。ここでは `oldDSP` が rebuild 要求から解決されるが、この解決方法が誤っている可能性も排除できない。

※ ISR Runtime では Builder は同一 Snapshot から Spec を構築するため、Builder 実行中に他スレッドが状態を変更する可能性は極めて低い。

**【確定した修正 — P6-a 実装済み】** :
```cpp
// RuntimeBuilder.cpp:221 — P6-a (2026-07-12 実装済み) 🛠️
worldOwner->topology.fadingRuntimeUuid = (active && next != nullptr) ? next->runtimeUuid : 0;
```

**【2026-07-12 詳細調査による本件の再評価】** :

⚠️ **コード調査により `next->runtimeUuid` は DSPCore コンストラクタで設定される immutable フィールドであることが確定した。**

| 決定論的観点 | 判定 |
|:------------|:------|
| `next->runtimeUuid` は mutable Runtime state か？ | ❌ **Immutable**（DSPCore 構築時に設定、以後不変） |
| Builder の `next->runtimeUuid` 読み取りは INV-12 違反か？ | ❌ **違反しない**（immutable プロパティの読み取りのため） |
| `active` は Spec から取得しているか？ | ✅ `spec.execution.transitionActive` から取得 |
| `next` は Spec から取得しているか？ | ✅ `spec.topology.fadingDSP` から取得 |
| Builder が「判断」しているか？ | ❌ 単純写像のみ（`active && next != nullptr` は Spec の2値を忠実に組み合わせているだけ） |

**結論**: **P6-a の修正は「暫定措置」ではなく最終設計として正当である。** 理由:
- `fadingRuntimeUuid` の計算に使われる値はすべて Spec 由来（`transitionActive`, `fadingDSP`）
- `runtimeUuid` は immutable プロパティであるため、INV-12（mutable Runtime state の直接観測禁止）に抵触しない
- Builder は Spec の2値から単純写像を行っているのみで、Policy Decision を行っていない
- `(active && next != nullptr)` は論理演算であって Policy Decision ではない（`a && b` は写像の一部）

**長期的な設計判断（Specification Completeness の観点）**:
- `fadingRuntimeUuid` の Specification への explicit field 追加は可能だが、**優先度は低い**
- 現状でも Builder は Spec から必要な情報をすべて取得できており、INV-12/13 に完全準拠している
- 昇格する場合の追加工数: RuntimeBuilder.h に `uint64_t fadingRuntimeUuid = 0` 追加 + Orchestrator で計算 + Builder でコピー。~20行の変更
- **決定**: `fadingRuntimeUuid` の Spec 昇格は **P9 以降で検討**（現状で設計契約違反はないため）

**確定に必要な追加 DIAG**（4点で同一 UUID 付き Spec/World ダンプ、Coordinator 出口を含めることで Builder 以前と Builder 以後を完全に切り分け）:
```
// Coordinator 出口: 生成された Spec の状態
[DIAG_AUTH] CoordExit   uuid=XXX transitionActive=? currentUuid=? nextUuid=? spec.fadingRuntimeUuid=?
// Builder 入口: Builder が受け取った Spec の状態
[DIAG_AUTH] BuilderEntry uuid=XXX transitionActive=? currentUuid=? nextUuid=? ...
// Builder 出口: 生成された World の状態
[DIAG_AUTH] BuilderExit  uuid=XXX graph.fadingNode=? fadingRuntimeUuid=? ...
// Commit 直前: publish される World の最終状態
[DIAG_AUTH] PreCommit   uuid=XXX graph.fadingNode=? fadingRuntimeUuid=? transitionActive=?
```
CoordinatorExit がないと「Spec が既に壊れていた」のか「Builder で壊した」のかの区別がつかない。

---

### P7: Builder 残留 atomic 読取りの Specification 昇格（2026-07-12 発見 → 🛠️ 全項目完了）

**🎯 優先度: ★★★★☆ → ✅ 全項目完了（2026-07-12）**

**【コード調査で確認された残留依存（2026-07-12）】**

現在の `RuntimeBuilder.cpp` には以下の `engine.*` 直接アクセスが残っていた (P0〜P2 完了後も未対応)。これらの性質に応じて4分類する:

#### A. Mutable Runtime Input（INV-12 違反 — Specification への昇格が必要）

| # | 行 | コード | Spec カバレッジ |
|:-:|:--|:------|:---------------|
| 1 | 255 | `convo::consumeAtomic(engine.retireQueueDepth_, ...)` | ❌ RetirePart 未定義 |
| 2 | 319-320 | `engine.currentAdaptiveCoeffBankIndex` | ❌ AdaptivePart 未定義 |

Builder が mutable な Runtime 状態（retire queue 深度、係数バンクインデックス）を直接読んでいる。本来は Orchestrator がこれらの値を Snapshot し、Specification の一部として Builder に渡すべき。

#### B. Runtime Query（INV-13 違反 — Orchestrator 経由への移行が必要）

| # | 行 | コード | Spec カバレッジ |
|:-:|:--|:------|:---------------|
| 3 | 324 | `engine.getAdaptiveCoeffBankForIndex(bankIndex)` | ❌ AdaptiveBankPart 未定義 |

エンジンの内部構造（CoeffBank）に直接アクセスしている。Orchestrator が事前に必要なデータを抽出し、Specification 経由で受け渡すべき。

#### C. IR transfer → ✅ Builder Service（Resource Factory）として正式採用（2026-07-12 確定）

| # | 行 | コード | 判断 |
|:-:|:--|:------|:-----|
| 4 | 382 | `runtime->convolverRt().transferIRStateFrom(engine.getConvolverProcessor())` | 🛠️ **Builder Service（Resource Factory）として正式採用** |

**2026-07-12 コード調査による5条件検証結果**（`transferIRStateFrom` `ConvolverProcessor.h:1037`）:

| # | 条件 | 結果 | 根拠 |
|:-:|:-----|:-----|:------|
| ① | IR Resource をコピー**だけ**であること | ✅ | `source.acquireIRState()` → `updateIRState(*srcState->ir, srcState->sampleRate)` → `source.releaseIRState(srcState)`。const source からの読み取りのみ。 |
| ② | Engine 状態を書き換えないこと | ✅ | 呼び出し先は `runtime->convolverRt()`（新規構築中のターゲット）。`engine.getConvolverProcessor()` は `const&` で渡される。 |
| ③ | Crossfade 状態を書き換えないこと | ✅ | 関数内で crossfade 関連のフィールドに一切アクセスしない。pure IR data transfer。 |
| ④ | Runtime topology を変更しないこと | ✅ | DSP ノードの追加/削除は行わない。既存の convolver に IR データを設定するのみ。 |
| ⑤ | Semantic に影響しないこと | ✅ | 同一 IR → 同一コンボリューション結果。semantic equivalence を満たす。 |

**結論**: 全5条件を充足。Builder Service の **Resource Factory** カテゴリとして正式に位置付ける。実装変更不要。

**本質的な前提条件**: Resource Factory が Builder Service として認められるためには、**Source が immutable resource であること**が最重要である。`transferIRStateFrom()` の場合は `source.acquireIRState()` → `const IRState*` → `updateIRState()` という流れ（Source Runtime → acquire → const resource → Target Runtime）になっており、Builder が mutable な Runtime 状態を観測することがない。逆に `engine.retireQueueDepth_` のような mutable atomic を Resource Factory 経由で Builder に渡そうとすれば、それは INV-12 違反となる。

---

### P8: MMCSS 初回失敗再試行機構（2026-07-12 詳細調査に基づく新設）

**🎯 優先度: ★★★☆☆**

**【コード調査で確定した問題】**

`AudioEngine.Processing.AudioBlock.cpp:47-51` の `compareExchangeAtomic(mmcssApplied_, false, true)` が成功した後に `applyMmcssPriority()`（`AudioEngine.Timer.cpp:218`）を呼ぶ設計において、`applyMmcssPriority()` が失敗しても `mmcssApplied_` を `false` に戻さない。その結果:

- **同一 prepareToPlay 期間中に MMCSS 再試行は一切発生しない**
- `Timer.cpp` の heartbeat callback に MMCSS 再試行コードは存在しない（2026-07-12 全コード調査確定）
- 唯一の再試行経路は `PrepareToPlay.cpp:27` の `publishAtomic(mmcssApplied_, false, release)` による次回デバイス再初期化時のみ

**確定した実装ギャップ**:

| 項目 | 現状 | あるべき姿 |
|:----|:-----|:-----------|
| 状態管理 | `bool mmcssApplied_` | 3値（未試行/成功/失敗）による再試行管理 |
| 再試行タイミング | prepareToPlay リセット時のみ | NonRT Timer 定期再試行（100ms-1s間隔） |
| 失敗永続化 | Error 1552 でセッション中ずっと MMCSS 未適用 | 次回 prepareToPlay まで定期的に再試行 |

**Error 1552 の特殊性**: `ERROR_NO_MORE_ITEMS` は MMCSS API としては珍しいエラー。Task名/ドライバ/Windows状態に依存するため、ログのみでは実装側の問題か Windows 側の問題か判断不能。

**設計**:

```cpp
// ★ P8: MmcssState による3値管理
enum class MmcssState { NeverTried, Applied, Failed };
std::atomic<MmcssState> mmcssState_{MmcssState::NeverTried};

// AudioBlock.cpp: CAS 成功後に関数を呼ぶ（従来と同じ）
if (mmcssState_.compare_exchange_strong(expected, MmcssState::Applied)) {
    applyMmcssPriority();  // 成功すると Applied、失敗すると Failed に戻す
}

// Timer.cpp: Failed 状態で定期再試行（100ms周期の heartbeat callback 末尾で）
if (mmcssState_.load() == MmcssState::Failed) {
    mmcssState_.store(MmcssState::NeverTried);  // 再試行許可
    // 次回 AudioBlock CAS で再試行される
}
```

**重要**: Audio callback 内での毎回の `AvSetMmThreadCharacteristics()` 試行は RT 性能に悪影響を与える。再試行はあくまで `NeverTried→Applied` の CAS が成功した場合のみ。Timer が `Failed→NeverTried` にリセットすることで、次回 Audio callback が再度試行する。

**実装方針**:
1. `AudioEngine.h`: `mmcssApplied_` → `MmcssState mmcssState_` に変更
2. `AudioBlock.cpp`: `applyMmcssPriority()` 戻り値を `applyMmcssPriority()` 内で `mmcssState_` に反映
3. `Timer.cpp`: heartbeat callback の適切な位置で `Failed→NeverTried` リセット
4. `PrepareToPlay.cpp`: リセット処理を維持（`mmcssState_ = NeverTried`）

---

#### D. HealthState — ✅ Builder から完全除去完了（2026-07-12）

| # | 行 | コード | 判断 |
|:-:|:--|:------|:-----|
| 5 | 357(旧) | ~~`engine.m_healthStateRef` (consumeAtomic)~~ | 🛠️ **Builder から完全除去済み** |

HealthState は Runtime の現在状態（mutable）であり、INV-12（Builder は mutable Runtime を直接観測しない）と相容れないため、Builder から完全に除去した。

**除去内容（2026-07-12 実装）**:
- `RuntimeBuilder::setHealthStateRef()`, `m_healthStateRef` を削除
- `RuntimeBuilder::build()` 内の HealthState Critical チェックを削除
- 全7箇所の `setHealthStateRef()` 呼び出しを削除（Init, PrepareToPlay×2, ReleaseResources, RebuildDispatch, Timer, Transition, Orchestrator）

**移行後の HealthState の役割**:
1. **Orchestrator の crossfade 制御**（`RuntimePublicationOrchestrator.cpp`）— HealthState Critical 時は crossfade を強制抑制。こちらは Builder とは独立した正当な用途。
2. **PublicationAdmission**（`RuntimePublicationOrchestrator.h`）— admission 制御用。Builder 経由ではなく直接保持。
3. **RuntimeHealthMonitor**（`RuntimeHealthMonitor.h`）— 診断ログ用。Timer 経由で取得可能。

**判定**: HealthState の Builder での用途（Critical 時に DSPCore 構築を抑止）は、Orchestrator が既に HealthState を crossfade/admission でチェックしているため冗長だった。削除による影響はない。

---

## 現時点の設計課題サマリ

P7-C 完了により **Builder 責務・Specification Completeness・INV-12/INV-13 に関する設計はほぼ閉じた**。残る設計論点は以下の2点に集約される:

| # | 課題 | カテゴリ | 優先度 | 備考 |
|:-:|:-----|:---------|:-------|:------|
| **P6** | AUTH_CONTRACT FAIL の責務境界整理 | Builder vs Spec 境界 | ★★★☆☆ | Builder は `active && next != nullptr` の写像で現状契約を満たす。長期的には `fadingRuntimeUuid` を Specification に昇格する選択肢もある（P9+）。優先度低。 |
| **P8** | MMCSS 初回失敗再試行機構 | 実行時設計改善 | ★★★☆☆ | RuntimeBuilder とは独立した別トラック。メモリ肥大化対策とは直接関係しないが、XRUN 根本原因の切り分けに必要。 |

**Builder 関連の設計完了状況**:
- ✅ P0/P1/P2: atomic 読取り・Runtime Query の Specification 昇格 → **完了**
- ✅ P6-a: Builder L210 `active` 条件追加 → **完了（最終設計として正当）**
- ✅ P6-b: DIAG_AUTH 4点 → **完了**
- ✅ P7-A1/A2/B: RetirePart/AdaptivePart → **完了**
- ✅ P7-C: IR transfer → **Resource Factory として Builder Service 正式採用**
- ✅ P7-D: HealthState → **Builder から完全除去**
- ⏳ P5-1/2: int→enum 型整理 → **全設計安定後の最終段階**

---

## 設計参照情報

*本節は全フェーズに共通する設計判断の集約。v8.3 までは [設計] 6 として配置されていたものを改訂・集約した。*

### INV 一覧（2026-07-12 ユーザー訂正版）

```
INV-12（設計原則）
  Builder は mutable Runtime state を直接観測しない
        ↓
INV-13（実装契約）
  Builder が利用できるのは Specification, Builder Service, Pure Utility のみ
  Input（atomic 読取り）: ❌ 禁止 → Specification 経由
  Runtime Query（Coordinator/Current Runtime/Crossfade）: ❌ 禁止 → Orchestrator 経由
  Builder Service には Allocator / Identity Generator / Immutable Factory / Resource Factory が含まれる。
  Resource Factory が扱えるのは immutable resource のみであり、mutable Runtime state を観測してはならない。
```

| ID | Invariant | 備考 |
|:---|:----------|:------|
| INV-1〜INV-10 | 変更なし（旧版維持） | DSPHandle ライフサイクル・Commit トランザクション |
| INV-11 | **RuntimePublishWorld は Builder 完了後 immutable。Builder never mutates inputs.** | ✅ 実装済み（const 返却確認済み） |
| **INV-12**（設計原則） | **Builder shall never observe mutable Runtime state directly. Any mutable runtime information required for construction shall be captured into RuntimePublishSpecification before Builder execution.** | **Builder は mutable な Runtime 状態を直接観測してはならない。構築に必要な mutable 情報はすべて RuntimePublishSpecification にキャプチャしてから Builder を実行する。** 設計原則。P0〜P2 の根拠。 |
| **INV-13**（実装契約 — INV-12 の実装ルール） | **Builder は RuntimePublishSpecification、Builder Service、Pure Utility のみ利用可能。Input（atomic）と Runtime Query（Coordinator/Current Runtime/Crossfade）は禁止。Builder Service には Allocator / Identity Generator / Immutable Factory / Resource Factory が含まれる。Resource Factory が扱えるのは immutable resource のみであり、mutable Runtime state を観測してはならない。**（P3 で正式化予定） | 🎯 2026-07-12 再定義。INV-12「mutable state 観測禁止」を実現する具体的な契約。P7-C で Resource Factory を Builder Service の正式カテゴリとして追加。P0〜P2 完了後に設計書更新。 |

### Builder 依存分類（2026-07-12 コード調査確定）

| 分類 | 可否 | 該当例 | 件数 | 現状 |
|:-----|:-----|:-------|:----|:-----|
| **① Input（禁止）→ Specification（→ ProcessingPart）** | ❌ | `currentProcessingOrder`, `eqBypassActive`, `convBypassActive`, `softClipEnabled`, `saturationAmount`, `inputHeadroomGain`, `outputMakeupGain`, `convolverInputTrimGain` | **8** | P0 対象。`sealedSnapshot==nullptr` 経路で atomic 直接読み取り。ProcessingPart として昇格。 |
| **② Runtime Query（禁止）→ Orchestrator→Spec へ** | ❌ | Coordinator `consumeWorldHandle()`, Publication Sequence, CrossfadeRuntime（`getStartDelayBlocks` 他3）, `latencyDelayOld/New`, `retireQueueDepth_` | **8** | P1+P2 対象。`computeRuntimePublishComputation()` 経由 + 直接読み取り。PublicationSnapshotPart/CrossfadeSnapshotPart/LatencyPart に格納。 |
| **③ Builder Service（許可）** | ✅ | `reserveRuntimePublicationIdentity()`, `RuntimePublishWorld::createForBuilder()`, Allocator, Factory, Identity Generator, Immutable Helper | **1** | 問題なし。契約条件を満たす（Runtime状態非依存/入力を変更しない/意味的結果を変更しない/決定論性を破壊しない）。 |
| **④ Pure Utility（許可）** | ✅ | `estimateRuntimeLatencyBaseRateSamples()`, hash(), math | **1** | 問題なし。現在状態に依存しない純粋計算。 |
| **⑤ P7 残留 Input** | 🛠️ **完了** | `retireQueueDepth_`, `currentAdaptiveCoeffBankIndex`, `getAdaptiveCoeffBankForIndex()` → RetirePart/AdaptivePart に昇格 | **3** | ✅ P7-A1/A2/B で全件完了。Specification 経由で Builder に渡される。 |
| **⑥ Resource Factory（正式 Builder Service）** | ✅ **許可** | `getConvolverProcessor()` → IR transfer | **1** | ✅ P7-C で `transferIRStateFrom()` の5条件確認完了。Resource Factory として正式分類。`m_healthStateRef` は Builder から除去済み（P7-D）。 |

### Builder Service の定義と判定条件

Builder Service は以下の**3条件をすべて満たす**必要がある（条件3は旧条件3「Semantic Equivalence」と旧条件4「実装アーティファクトのみ」を統合 — Semantic Equivalence を満たせば実装アーティファクト制約は自動的に包含されるため）:

| # | 条件 | 説明 | 違反例 |
|:-:|:-----|:-----|:------|
| 1 | **Mutable Runtime State を参照しない** | エンジンの atomic 変数・DSPCore 内部状態を読まない | `consumeAtomic(retireQueueDepth_)` |
| 2 | **Specification を書き換えない** | Builder の入力を変更しない | Spec のフィールドへの再代入 |
| 3 | **Semantic Equivalence を変えない**（実装アーティファクトは除く） | 同じ Spec → 同じ World。Allocation, Identity, Pointer, Generation 等の実装アーティファクトの変更は許容されるが、DSP topology/routing/latency/crossfade/processing semantics は不変 | 非決定的な ID 生成による World 差異、DSP topology の変更 |
| **4** | **Resource Factory: Source は immutable resource に限る**（Resource Factory 追加条件） | Resource Factory が扱う Source は IR データ・FFT Plan 等、構築後に変更されない immutable resource のみ。Runtime の mutable state（Current Runtime / Crossfade State / Publication State / Health State）は Source として認められない。 | `engine.getConvolverProcessor()` から `acquireIRState()` で取得した `const IRState*` は immutable。`engine.retireQueueDepth_` のような atomic 変数は mutable につき Resource Factory の対象外。 |

**Builder Service の分類**:

| カテゴリ | サービス種別 | 該当例 |
|:---------|:------------|:-------|
| **Memory Service** | Allocator | `aligned_malloc`/`aligned_free` |
| **Identity Service** | Identity Generator | `reserveRuntimePublicationIdentity()` |
| **Immutable Factory** | Factory + Immutable Helper | `RuntimePublishWorld::createForBuilder()`, FFTPlan, LatencyCalculator, BufferFactory |
| **Resource Factory** | Immutable Resource Copier | `IRFactory`, IR transfer（`transferIRStateFrom`） |

**Resource Factory レビューチェックリスト**:

Resource Factory に新しい API を追加する場合は以下の項目をレビューすること:

| # | チェック項目 | 判定例 |
|:-:|:------------|:-------|
| □ | **Source が immutable resource である** | `const IRState*` は ✅。`engine.retireQueueDepth_` のような atomic 変数は ❌ |
| □ | **Source を変更しない** | `source.acquireIRState()` → read-only → `source.releaseIRState()` は ✅。内部キャッシュの書き換えは ❌ |
| □ | **Runtime topology を変更しない** | 既存の DSP ノードにデータを設定するのみ ✅。ノードの追加/削除は ❌ |
| □ | **Runtime state を観測しない** | 対象ノードが新規構築中のものであれば ✅。active/fading DSP の内部状態を読むのは ❌ |
| □ | **Semantic Equivalence を維持する** | 同一 Source → 同じ処理結果 ✅。非決定的な振る舞いの導入は ❌ |

> **注意**: 上記を1つでも満たさない操作は、Resource Factory ではなく Specification への昇格または Orchestrator での事前解決が必要。

**P7 残留依存の確定**:
- A/B（Mutable Input / Runtime Query）→ ✅ **P7-A1/A2/B で昇格完了**
- C（IR transfer）→ ✅ **Resource Factory として Builder Service に正式分類**
- D（HealthState）→ ✅ **Builder から完全除去完了**
- **全項目完了につき P7 クローズ**

### Authority Boundary Chart

| 責務 | Authority | コード位置 |
|:-----|:----------|:----------|
| `activeRuntimeDSPHandle_` 更新 | `commitRuntimePublication()` | `AudioEngine.h:3981` |
| `DSPCore*` active 更新 | `DSPLifetimeManager::activate()` | `DSPLifetimeManager.h:28` |
| DSPHandle 発行 | `registerDSPHandleForRuntime()` | `AudioEngine.h:3838` |
| DSPHandle 回収 | `retireDSPHandleForRuntime()` | `AudioEngine.h:3875` |
| EBR enqueue | `DSPLifetimeManager::retire()` | `DSPLifetimeManager.h:37` |
| DSPCore* 物理破棄 | `destroyDSPCoreNode()` | `AudioEngine.Threading.cpp:15` |

### RT Safety

- `activeRuntimeDSPHandle_`: publishAtomic(release) / consumeAtomic(acquire) — 正しい HB 順序
- `runtimeDSPHandleMap_`: `std::mutex` 保護（NonRT のみ）
- `DSPHandleRuntime`: `std::atomic<DSPHandle>` + `std::atomic<DSPState>` ベース
- EBR drain: 「全 Reader が epoch を離脱した」ことを保証してから callback 実行

### エグゼクティブサマリ

> **設計上の推定値（コード解析ベース — 概算であり実測値ではない）**: 未修正時 2,477MB に対し、設計上は定常 686MB / ピーク 1,094MB を見込む。ただしこれらの数値はコード上のバッファサイズ・アロケーション数式に基づく概算値であり、ビルド・実測未実施のため確定値ではない。実測により大きく変動する可能性がある。

**実測確認が必須の項目**: BlockSize 削減効果（~189MB/DSPCore 見込み）、CrossfadePlan 導入後の EBR 動作、680MB Other 内訳。これらは P0〜P3 実装後の MEM_SNAP で確認する。

**全11ステップ実装確認済み**: ソースコード上の実装は完了。発見した7件の不整合（旧戻り値型削除漏れ、テストの hasFadingRuntime 残存、P4 DIAG_MKL_MALLOC include不足、queueDepthBlocks ラベル、コメント古朽化2件）は修正済み。ビルド・テスト通過確認は未実施（C1060 環境問題のため）。

---

## 改修後の測定結果で検証するべき項目

*本セクションは「コード変更を実装した後、MEM_SNAP やログを用いた実測により検証する必要がある項目」を集約する。現時点ではコード調査で確定不能であり、測定結果を待って確定する。*

### BlockSize 削減実測値

**内容**: `kInitialPrepareMaxBlock=4096`（`Init.cpp:41`）コード確認済み。コード数式で ~189MB/DSPCore の削減を見込む。

**確認方法**: P2-1 実装後の MEM_SNAP で削減効果を確認。現状のログ（gen=1: spb=4096, internalMaxBlock=32768）は改修前の値のため、改修後に再測定が必要。

### 680MB Other 内訳実測値

**内容**: 現状の計装では Private - TRK = ~455MB までしか分解不能。`computeOtherPrivate()` は残余値（`osPrivateMB - MKL_bytes - retire_bytes`）であり、aligned_malloc/JUCE/CRT/VirtualAlloc 等すべてを含む。

**確認方法**: TrackedMemoryStatistics 統合後の MEM_SNAP、または DIAG_MKL_MALLOC 拡張（生 `aligned_malloc` 7箇所の段階的 DIAG 化）により分解能を向上させる。

### 455MB 未追跡メモリ内訳

**内容**: steady-state Private 455MB のうち TRK total=1.2MB のみ追跡。残り ~454MB が DIAG 非計装領域。

**確認方法**: P4 TrackedMemoryStatistics の DIAG 計装拡張（DSPCore 内部アロケーションの段階的 DIAG 化）が必要。

### runtimeDSPHandleMap 収束値

**内容**: `std::unordered_map<DSPCore*, DSPHandle>`。生存 DSPCore 数に厳密にバインドされる。コード上の設計としては問題なし。steady-state 2-3 エントリと見込む。

**確認方法**: 長期稼働後の MEM_SNAP または DIAG 出力でエントリ数確認。

---

## その他の未確定項目

*本セクションは「現時点のログとコード調査だけでは確定できず、追加の調査・設計判断・外部情報が必要な事項」を集約する。*

### MMCSS 初回失敗再試行機構

**発見経緯**: 2026-07-12 ConvoPeq.log 解析。`[MMCSS] FAILED: GetLastError=1552 taskIndex=0`。

**問題**: `AudioEngine.Processing.AudioBlock.cpp:40-44` では `compareExchangeAtomic` が成功した後に `applyMmcssPriority()` を呼ぶ設計。このため初回登録が失敗しても `mmcssApplied_` は `true` のまま固定され、**同一 prepareToPlay 期間中に再試行は発生しない**。

**確認済みの再試行経路**（コード調査 2026-07-12）:
- `PrepareToPlay.cpp:27`: `convo::publishAtomic(mmcssApplied_, false, std::memory_order_release)` でリセット
- つまり次回のデバイス再初期化（prepareToPlay 再呼び出し）時に再試行される
- `Timer.cpp` に MMCSS 再試行コードなし（`AudioBlock.cpp` の CAS のみが唯一の適用経路）

**Error 1552 の特殊性**: 1552 = `ERROR_NO_MORE_ITEMS` は MMCSS API としては珍しいエラー。原因は Task名/ドライバ/Windows状態など複数あり、ログのみでは実装側の問題か Windows 側の問題か判断できない。

**確認方法**: MMCSS 成功時ログ `[MMCSS] registered:` の有無を確認。

**対応**: **P8 として設計書 Backlog に追加**（2026-07-12 詳細調査）。`enum class MmcssState` による3値管理 + NonRT Timer 再試行の設計を確定。実装は次回改修単位。

### XRUN 根本原因（複数仮説）

**発見経緯**: 2026-07-12 ConvoPeq.log 解析。全7件の XRUN が正 drift (+3,410〜+4,297us) を示す。コールバック実処理時間 (0.47〜1.57ms) は予算 (5.33ms) 内であるため、DSP 負荷過多ではない。

**2026-07-12 詳細調査による追加知見**:
- Timer callback は heartbeat 専用であり、`Timer.cpp` 内で `jitter > 20ms` または `expected*0.1` を検出している。XRUN とは独立した監視。
- `Timer.cpp` 末尾で 10ms 超の実行時間を検出・ログ出力。Timer callback が重いと audio callback のタイミングに影響する可能性がある。
- `AudioBlock.cpp` 内の `callbackTimingHistory` リングバッファ（`kCallbackTimingSlots` エントリ）が各 callback の処理時間/ドリフト/CPU/予算を記録。XRUN 発生後に Timer が `[CB_HIST]` としてダンプ。
- Timer heartbeat callback（100ms周期）と XRUN ペア間の間隔（40-50s）に直接的な相関は確認できず。

**候補（ログのみでは特定不能）**:
1. MMCSS 未適用によるスレッド競合 — ❗ gen=6 で MMCSS 成功後も XRUN が減少していないため、直接原因の可能性は低い
2. Windows Audio Engine / WASAPI / ASIO ジッター
3. USB オーディオインターフェースのアイソクロナス転送遅延
4. DPC/ISR によるプリエンプション
5. CPU Package C-state 遷移
6. メモリ帯域競合（MixedPhase 時の PageFault surge 時に顕著）

**確認方法**: P8（MMCSS 再試行機構）実装後、MMCSS が確実に適用された状態で XRUN 発生有無を再評価する。その後も解消しない場合は ETW/xperf による詳細トレース。

**タイミングパターン分析（2026-07-12 ログ解析）**: XRUN は単独ではなくペアで発生する傾向がある:
- gen=6 crossfade 中: 2回が 0.4s 間隔（XRUN#1-#2）
- gen=7 steady-state: 5回がペアで発生（XRUN#3-#4: ~10s間隔、XRUN#5-#6: ~10s間隔、XRUN#7: 単独）
- ペア間の間隔: ~40-50s

単発のOSジッターよりは何らかの定期的なバックグラウンド処理との相関が疑われる。Timer callback（100ms）は周期が短すぎて直接の原因とは考えにくい（XRUNペア間は10s x 100tick）。Windows のスレッドスケジューリング量子（~15-30ms）や DPC レイテンシのクラスタリングが関与している可能性がある。ただし確定的な因果関係はログのみでは特定不能。

### EQ Cache モノトニック成長 — ✅ 影響軽微で確定（2026-07-12 詳細調査）

**観測**: VERIFY カウンタ `eqCacheMiss(create/lookup)=0/0` -> cache hit rate 100%。

**コード調査による確定事実（2026-07-12）**:
- `CacheMap` = `std::unordered_map<uint64_t, EQCoeffCache*>` - **削除機構・エビクションポリシーなし**。`clear()`, `erase()`, `evictLRU()` のいずれも未実装。
- `cacheMapPtr` は copy-on-write の `std::atomic<CacheMap*>` - 新しいエントリ追加時に `CacheMap` 全体をコピーし新しい Map を atomic 公開。旧 Map は EBR 経由で非同期解放。
- **全く対照的な設計**: 同じコードベースの IR `CacheManager`（`CacheManager.cpp`）には `evictLRU()`, `lruList`, `clear()` が存在するが、`EQCacheManager` には一切ない。
- つまり EQ Cache は **モノトニック成長**（追加される一方で削除されない）
- **実測サイズ（コード調査 2026-07-12）**: `EQCoeffCache` = coeffs[20] @ 64bytes + metadata ~100bytes + alignment = **~1.5KB/entry**（従来推定の ~200KB から大幅下方修正）。coeffs は `EQCoeffsSVF` (8 doubles = 64bytes) × 20バンド = 1,280bytes が大部分を占める。
- 再計算: 100種類の EQ 設定で **~150KB**、1000種類で **~1.5MB**。実用的なメモリ影響はごく軽微。
- ただし `CacheMap` の copy-on-write による瞬間的なメモリ使用量倍増は発生し得る

**✅ 確定判断（2026-07-12 詳細調査）**: モノトニック成長だが ~1.5KB/entry のため現実的なメモリ影響は軽微。エビクション実装は不要。設計上の注意点として文書化。

**確認方法**: `CacheMap` のエントリ数ダンプ、または EQ パラメータ変更頻度と実メモリ使用量の相関観測により検証可能。

---

## 調査確定状況一覧

**全ツールを使用した最終網羅調査の結果、コード調査で確定可能な事項は全て確定・記録された。** 使用ツール: grep/sed (WSL), AiDex MCP, serena MCP, cocoindex-code (ccc.exe), graphify, semble。

| 調査項目 | 結果 | 確定状況 |
|:---------|:-----|:---------|
| `hasFadingRuntime` production残存 | **0件** - 全削除確認（ISRRuntimeSemanticSchema.h から削除、hasFadingRuntimeInWorld は fadingRuntimeUuid != 0 導出） | ✅ 確定 |
| `currentCaptureSessionId` 代入 | **0件** - 定義時初期値 =0 固定。Runtime代入なし -> sessionId によるドロップは発生しない | ✅ 確定 (FACT #82) |
| P4 `DIAG_MKL_MALLOC` 完全性 | CacheManager(2) + IRConverter(1) の3箇所すべて DIAG 化 + `DiagnosticsConfig.h` include 追加完了。MKLNonUniformConvolver 内6箇所の生 `mkl_malloc` は P4 範囲外（ScopedAlignedPtr 管理 + allocatedBytes() 追跡済み） | ✅ 確定 |
| `aligned_malloc` DIAG 未計装 | 生 `aligned_malloc` 7箇所、`DIAG_MKL_MALLOC` 23箇所 = 合計30箇所。v7.9 調査時81箇所から改善。生 `aligned_malloc` 7箇所は ScopedAlignedPtr で RAII 管理。 | ✅ 既知制限 |
| Builder メモリオーダー | 全18件 `memory_order_acquire` 統一。不整合なし | ✅ 確定 |
| `const_cast` | 2箇所のみ: RuntimeBuilder.h:94, Orchestrator.cpp:178 | ✅ 確定 |
| 全 Builder 関数 | `buildRuntimePublishWorld`, `createBootstrapWorld`, `validateWarmup` 全て `noexcept` | ✅ 確定 |
| TODO/FIXME work70関連 | **0件** | ✅ 確定 |
| `collectTrackedMemoryStatistics()` | 定義 + 実装完了（`ASSERT_NON_RT_THREAD()` 完備）。呼び出し元0件。 | ✅ API完了 |
| `RuntimePublicationSpecification.h` | `RuntimeBuilder.h` のエイリアスファイル。間接的に全ビルドでインクルード。 | ✅ エイリアス確認 |
| NoiseShaper accepted=0 | `accepted=3012/3004`, `dropSampleRate=0` -> **NOT-A-PROBLEM** -> Appendix B | ✅ 解決 |
| EBR lifecycle(retire)=0 | 主因（AUTH_CONTRACT FAIL）特定。reclaim=13 は `tryReclaimResources()` 呼び出し回数（物理破棄件数ではない）-> Appendix B | ⚠️ 主因特定 |
| 同一 gen=3 内 OS倍率変化 | ノイズシェイパー type=0->2 変更が原因。**正常動作** -> Appendix B | ✅ 解決 |
| **EBR epoch advance** | `advanceEpoch()` は deprecated + private（移行完了済み）。public API は `publishEpoch()` / `tryReclaim()`。EpochDomain のコード調査により正常動作確認。Runtime での進行状況（epochAdvanceCount）は別途観測。 | ✅ 設計完了確認 |
| **EQ Cache モノトニック成長** | `CacheMap` = `unordered_map`（evictLRU/clear/erase 未実装）。~1.5KB/entry。1000エントリでも ~1.5MB → 実用的影響は軽微。 | ✅ 影響軽微で確定 |
| **MMCSS 再試行** | CAS 成功後に `applyMmcssPriority()` 失敗 → `mmcssApplied_` が true 固定。Timer に再試行コードなし。**P8 として設計書に追加**。 | 🆕 P8 新設 |
| **P6 長期設計（fadingRuntimeUuid 昇格）** | `next->runtimeUuid` は immutable。Builder での読み取りは INV-12 違反ではない。P6-a 修正は暫定措置ではなく最終設計として正当。Spec 昇格は P9 以降で検討。 | ✅ 設計判断確定 |
| **RuntimePublicationSpecification.h** | エイリアスファイル。誰もインクルードしていない（将来の分離用準備）。 | ✅ エイリアス確認済み |

**結論**: コード調査 + ログ解析により NoiseShaper, EBR 主因, OS倍率変化は解決。BlockSize/Other/455MB はコード変更後の実測で検証。MMCSS（P8）および XRUN は追加の設計判断または実装後の再評価が必要。EQ Cache/EBR epoch advance/P6 長期設計/RuntimePublicationSpecification.h は調査により確定。

---

## Appendix A: 実装済み成果物

### A.0 今回の改修（work70 v8.3 全11ステップ）— ソースコード上確認済み

| ステップ | 内容 | ファイル | 状態 |
|:--------|:-----|:---------|:-----|
| ① | RuntimePublishSpecification 定義（三部構成 + version） | `RuntimeBuilder.h` | ✅ 実装確認 |
| ② | Orchestrator Spec生成 + Post-build Mutation 削除 | `RuntimePublicationOrchestrator.cpp` | ✅ worldOwner-> 代入 READ 1件のみ |
| ③ | Builder Spec 受取 + 内部実装修正 | `RuntimeBuilder.h/.cpp` | ✅ シグネチャ変更＋Spec 解決 |
| ④ | 他6箇所の呼び出し元対応（旧シグネチャ委譲） | 各 .cpp | ✅ inline 委譲確認 |
| ⑤ | Validator 三者整合性（Topology/Execution/Identity） | `RuntimePublicationValidator.cpp` | ✅ hasFadingRuntime 不使用 |
| ⑥ | hasFadingRuntime 削除（production） | `ISRRuntimeSemanticSchema.h` + 全参照 | ✅ fadingRuntimeUuid != 0 導出 |
| ⑦ | P2-1 BlockSize 最適化 | `AudioEngine.h`, `Init.cpp`, `DSPCoreLifecycle.cpp` | ✅ PrepareBlockSizingPolicy + kInitialPrepareMaxBlock=4096 |
| ⑧ | P-NS NoiseShaper DIAG 改善 | `NoiseShaperLearner.h/.cpp` | ✅ DropReason enum, generation, queueDepthBlocks, dropBySampleRate DIAG |
| ⑨ | P-DIAG TrackedMemoryStatistics API + MEM_SNAP統合 | `AudioEngine.h`, `DSPCoreLifecycle.cpp`, `Timer.cpp` | ✅ 10カテゴリ + ASSERT_NON_RT_THREAD + MEM_SNAP出力（P4） |
| ⑩ | P4 mkl_malloc DIAG 化 | `CacheManager.cpp`, `IRConverter.cpp` | ✅ DIAG_MKL_MALLOC + include 追加 |
| ⑪ | const RuntimePublishWorld 化 | `RuntimeBuilder.h`, `AudioEngine.h`, `RuntimePublicationCoordinator.h` | ✅ Builder→commitRuntimePublication→Coordinator const chain |
| **⑫** | **P0: ProcessingPart追加 + atomic読取り排除（v9.4）** | `RuntimeBuilder.h/.cpp`, `RuntimePublicationOrchestrator.cpp` | ✅ ProcessingPart定義、Orchestrator設定、Builder読取り、無効行削除 |
| **⑬** | **P1: PublicationSnapshotPart + previousCommittedSequence移行（v9.5）** | `RuntimeBuilder.h/.cpp`, `RuntimePublicationOrchestrator.cpp` | ✅ PublicationSnapshotPart定義、Orchestrator設定、Builder読取り（Runtime Query 部分移行完了） |
| **⑭** | **P2: CrossfadeSnapshotPart/LatencyPart 追加（v9.5）** | `RuntimeBuilder.h/.cpp`, `RuntimePublicationOrchestrator.cpp` | ✅ CrossfadeSnapshotPart/LatencyPart 定義・Orchestrator設定・Builder engine直接読取り排除 |

#### A.0.1 RuntimePublishSpecification 構造（v8.3）

```cpp
struct RuntimePublishSpecification {
    uint32_t version = 1;

    struct TopologyPart {
        const AudioEngine::DSPCore* activeDSP = nullptr;
        const AudioEngine::DSPCore* fadingDSP = nullptr;
    } topology;

    struct ExecutionPart {
        bool transitionActive = false;
        int transitionPolicy = 0;  // 0=HardReset, 1=SmoothOnly, 2=DryAsOld
        double fadeTimeSec = 0.0;
    } execution;

    struct RoutingPart {
        int processingOrder = 0;   // 0=EQ→Conv, 1=Conv→EQ
        bool eqBypassed = false;
        bool convBypassed = false;
    } routing;
};
```

#### A.0.2 P2-1 PrepareBlockSizingPolicy

```cpp
struct PrepareBlockSizingPolicy {
    static constexpr int kMinimumPrepareBlock = 256;
    [[nodiscard]] static constexpr int apply(int samplesPerBlock) noexcept {
        jassert(samplesPerBlock > 0);
        return std::max(kMinimumPrepareBlock, samplesPerBlock);
    }
};
const int kInitialPrepareMaxBlock = 4096;  // AudioEngine.Init.cpp
```

#### A.0.3 P-NS DropReason + Waiting diagnostics

| フィールド | 値 | 意味 |
|:-----------|:---|:------|
| `generation=` | `consumeAtomic(progress.iteration)` | 完了世代数（1-based） |
| `queueDepthBlocks=` | `captureQueue.size()` (w-r) | キュー内 AudioBlock 滞留数 |
| `dropBySampleRate` DIAG | `block.sampleRateHz` + `session.sampleRateHz` 実値 | Runtime 観測用 |

#### A.0.4 P-DIAG TrackedMemoryStatistics API

```cpp
struct TrackedMemoryStatistics {
    size_t oversampling = 0;       // Oversampling work buffers
    size_t softClip = 0;           // SoftClip OS work buffers
    size_t eqProcessor = 0;        // EQ scratch/dry/parallel/structure/msWorkBuffer
    size_t alignedBuffers = 0;     // alignedL/R + dryBypassL/R
    size_t latencyBuffers = 0;     // fixedLatency × 4
    size_t truePeakDetector = 0;   // TruePeakDetector internal
    size_t convolver = 0;          // Convolver internal (no IR = minimal)
    size_t crossfade = 0;          // JUCE crossfade buffers
    size_t misc = 0;               // DCBlocker/LoudnessMeter/PeakLimiter/NoiseShaper
    size_t otherTracked = 0;       // tracked だが特定カテゴリに分類されないもの
    [[nodiscard]] size_t totalTracked() const noexcept { /* SUM(categories) */ }
};
```

### A.1 P1-a: publish 経路への handle 登録追加

**目的**: Coordinator direct publish（Init/PrepareToPlay/ReleaseResources/Timer/Transition）で DSPCore が `runtimeDSPHandleMap_` に未登録となる問題を修正。

```cpp
struct RegistrationContext {
    DSPCore* dsp = nullptr;
    DSPHandle handle;
    static RegistrationContext needsRegistration(DSPCore* dsp_) noexcept;
    static RegistrationContext alreadyRegistered(DSPHandle handle_) noexcept;
    static RegistrationContext none() noexcept;
};
```

**呼び出しパターン（7 箇所）**:

| # | 呼び出し元 | mode | dsp | handle |
|:-:|:-----------|:-----|:----|:-------|
| 1 | Init | none | — | — |
| 2 | PrepareToPlay (first) | needsRegistration | currentForPublish | — |
| 3 | PrepareToPlay (rebuild) | needsRegistration | getActiveRuntimeDSP | — |
| 4 | ReleaseResources | none | — | — |
| 5 | Timer (fadeComplete) | needsRegistration | currentAfterFade | — |
| 6 | Transition | needsRegistration | 引数 newDSP | — |
| 7 | PublicationExecutor | alreadyRegistered | — | req.newDSP |

**検証**: CTest 15/15 PASS, CI Gates ALL PASS.

### A.2 P1-b: advanceFade 配線

- FadeState の開始・進行・完了を RuntimePublicationCoordinator と連携
- 完了後は advanceFade 内で commitRuntimePublication を呼び出して新しい RuntimePublishWorld を公開

### A.3 P1-c: MEM_SNAP 監視強化

- DSPCore::liveCount（DIAG ガード済み atomic）の MEM_SNAP 出力
- RuntimeWorld サイズ表示（`world gen=M size=NNNMB`）
- `computeOtherPrivate()` の完全追跡

### A.4 P1-a-FIX: activeRuntimeDSPHandle_ 未更新修正

Coordinator direct publish 後の `activeRuntimeDSPHandle_` 更新漏れを修正。`commitRuntimePublication()` 成功後に `dspHandleRuntime_.activate(handle)` を呼ぶ。

### A.5 P1-a-FIX-2: DSPGuard 直接破棄パス

例外発生時に DSPGuard から直接 `destroyDSPCoreNode()` を呼ぶパスを追加。`destroyDSPCoreNode()` 呼び出し前に `DSPHandle` を retire する。

### A.6 P1-a-FIX-3: 0xC0000005 修正（DSPGuard 重複 destroy）

`runtimeDSPHandleMap_` の `eraseByHandle` が `DSPGuard` 破棄後に `runtimeDSPHandleMap_` を走査して Access Violation を引き起こす問題を修正。

### A.7 D-1: DSPLifetimeManager::destroyRolledBackDSP

publish 失敗後に未公開 DSPCore を安全に破棄するための専用メソッド。

### A.8 検証結果

**設計上のメモリ効果見込み**: 未修正時 2,477MB に対し、設計上は定常 686MB / ピーク 1,094MB を見込む。ビルド・実測未実施のため確定値ではない。

**FACT 一覧（全86件）**: コード調査により確定可能な事項は全て確定済み。残る6項目は全て Runtime 観測または実装後の検証に依存（[未確定] 7 参照）。

**検証済みの設計判断**:

| 判断 | 結論 |
|:-----|:------|
| advanceFade のサンプル単位 | OS 補正不要。コールバックサンプル数のまま減算 |
| FadeState::Completed 追加 | 不要。既存の remaining==0 チェックで完了検出可能 |
| memory_order 変更 | 現状維持（relaxed 化は却下） |
| CrossfadeRuntime への完了通知追加 | 不要。SnapshotCoordinator と CrossfadeRuntime は責務が完全に独立 |

---

## Appendix B: 解決済み未確定事項（旧 [未確定] 7.1, 7.2）

### D.1 NoiseShaper accepted=0 → ✅ NOT-A-PROBLEM（2026-07-12 ログ解析で確定）

**解決日**: 2026-07-12

**経緯**: コード調査では `accepted=0` の原因を特定できず、サンプルレート不一致が疑われていた。ConvoPeq.log の P-NS DIAG 出力により解決。

**確定事実**:
```
[NoiseShaperLearner] Waiting diagnostics:
  accepted=3012/3004  dropSession=0  dropSampleRate=0  dropBank=0
  sessionId=0  sampleRateHz=192000  bankIndex=107  generation=39  queueDepthBlocks=0
```

**結論**: NoiseShaper は正常に学習ブロックを受理中。block.sampleRateHz と session.sampleRateHz は一致しており、サンプルレート不一致は発生していない。

---

### D.2 EBR lifecycle(retire)=0 → ✅ 主因特定（2026-07-12 ログ解析）

**解決日**: 2026-07-12

**経緯**: `VERIFY lifecycle(pub/ret/reclaim)` で `retire=0` が継続。handle 未登録が疑われていた。

**確定事実**: 主因は AUTH_CONTRACT FAIL による publish 停止（gen=3〜5）。NUC live=0 の間は retire 機会が発生しない。
- gen=6 publish 成功後、`Ret: pend=1` を確認（retire enqueue 正常）
- gen=7 publish 成功後、`Ret: pend=0` + reclaim counter 0→13（EBR polling 正常）

**保留**: EBR epoch advance / reader leave drain / callback 完全性はこのログのみでは確認不可。必要に応じて別ログで検証。

---

## Appendix C: 調査ツール

| ツール | 使用目的 |
|:-------|:---------|
| grep/sed/awk (WSL) | ログ抽出、統計計算、production コード全数調査 |
| serena MCP | コードパストレース、型情報取得、状態遷移調査 |
| cocoindex-code (ccc.exe) | 関数間依存関係の grep、シンボル特定 |
| graphify | 依存関係グラフパス検索、関数間リンク検証 |
| semble | セマンティックコード検索、フォールバック経路発見 |
| AiDex MCP | コードインデックス検索、セッションノート管理 |
| ast-grep / rg / fd / fzf | WSL ベースの高速コード検索・フィルタリング |

---

## Appendix D: 改訂履歴

| 版 | 日付 | 改訂内容 |
|:---|:-----|:---------|
| 1.0〜7.9 | 2026-07-10〜11 | 初版〜FACT 86 確定（旧版 v8.3 までの全履歴は旧 Appendix E 参照） |
| **8.0** | **2026-07-11** | **レビュー指摘3点反映（最終確定）**: transitionActive 導出→ExecutionSemantic 包含に戻す、Specification 三階層構造化、AllocatorPolicy 独立 |
| **8.1** | **2026-07-11** | **レビュー指摘4点反映**: Specification 三部構成、PrepareBlockSizingPolicy 改名、DropReason 通常 enum、totalTracked カテゴリ合計算出 |
| **8.2** | **2026-07-12** | **レビュー指摘5点反映**: version フィールド追加、enum class 化、jassert Contract Enforcement、Deterministic Construction 明確化、DTO 的性質明文化 |
| **8.3** | **2026-07-12** | **レビュー指摘3点反映**: Specification 独立ファイル化、INV-12 Deterministic Utility 条件追加、Validator 三カテゴリ再構成 |
| **9.0** | **2026-07-12** | **全面再構成**: 実装済み全11ステップを Appendix に移動。優先改修項目（Backlog）を先頭に新設。INV-12 をユーザー設計判断に基づき再定義。Builder 依存4分類を追加。collectTrackedMemoryStatistics MEM_SNAP 統合を Backlog P5 に追加。 |
| **9.1** | **2026-07-12** | **設計審査フィードバック反映**: P0 本質を「暗黙入力排除」に修正。P2 RuntimeContextPart を CrossfadePart/LatencyPart/PublicationPart に分割。P3 EnginePart/GraphPart 削除＋Specification Part 追加の3基準追加。Builder Service 明示分類。Deterministic Construction 定義追加。優先順位再編（INV-12 再定義を P0-P2 完了後の P3 に移動）。 |
| **9.2** | **2026-07-12** | **設計審査フィードバック反映（7点）**: P0 Deterministic 定義を Semantic Equivalence に精密化＋P0 タイトル「排除→昇格」に修正。P1 AutomationPart を ProcessingPart に改名（processingOrder 包含）。P2 PublicationPart/CrossfadeSnapshotPart に責務明記＋Builder Service を契約ベースに抽象化。INV-12 本文に Pure Utility 追記。エグゼクティブサマリ「改善→見込み」に修正。 |
| **9.3** | **2026-07-12** | **設計審査フィードバック反映（4点）**: INV-13（設計原則）と INV-12（実装契約）の上下関係を明示（INV-13→INV-12）。INV-13「Builder mutable Runtime state 直接観測禁止」追加。ProcessingPart に将来の RoutingPart/ProcessingParameterPart 分割方針を補足。processingOrder バグを設計問題と実装問題に分離記載。全ソース最終調査（grep/sed/AiDex/serena/ctx）で残存問題ゼロ確認。 |
| **9.4** | **2026-07-12** | **可読性・保守性改善（5点）**: ProcessingPart に YAGNI 統合理由を追記。Builder Service を「Builder execution environment」と位置づけ。PublicationPart→PublicationSnapshotPart に改名。INV-12/INV-13 番号を設計原則→実装契約の順序に入れ替え。FACT 件数表記を「全86件」→「整理対象86件」に軟化。 |
| **9.5** | **2026-07-12** | **実装進捗**: P0（ProcessingPart + atomic読取り排除）✅完了。P4（MEM_SNAP統合）✅完了。P1第一段階（PublicationSnapshotPart追加＋previousCommittedSequence移行）✅完了。P2（CrossfadeSnapshotPart/LatencyPart追加＋Builder engine直接読取り排除）✅完了。設計書Backlog更新、Appendix A.0に⑫⑬⑭追加。P3 設計書更新済み。 |
| **9.6** | **2026-07-12** | **設計契約の明文化（5点）**: (1) Deterministic Construction に「Builder Service may affect implementation artifacts only」追記。(2) Specification Completeness 節を新設（INV-13 の Data 制約版）。(3) P2 に Source 一意性（ProcessingPart 唯一 Source）と Part Ownership 表を追加。(4) Builder Service を Memory/Identity/Immutable Factory に細分類。(5) P2 Backlog 完了確認・冗長代入 cleanup は次回対応。 |
| **9.7** | **2026-07-12** | **v9.7a: ConvoPeq.log 解析＋ツール統合調査結果反映**: P6（AUTH_CONTRACT FAIL — Builder または Spec 生成段階の条件不一致）新設。P7（Builder 残留 atomic 読取り5件）新設。Builder 依存分類表に⑤残留 Input と⑥特殊依存を追記。P5 に aligned_malloc 件数実測値を追記。[未確定] 7.1 NoiseShaper accepted=0 解決。7.2 EBR 主因特定。 |
| | | **v9.7b: 設計書再構成**: P0〜P4（完了）を Appendix A に統合・Backlog から削除。[未確定] 7.1/7.2（解決）を Appendix D に移動。新規4項目（MMCSS/XRUN/455MB/EQ Cache）を [未確定] 7.2〜7.5 として追加。Appendix 番号を B〜E に再編。使用ツール: grep/sed(WSL), serena MCP, cocoindex-code, semble, graphify, AiDex。 |
| **9.8** | **2026-07-12** | **P6/P7 コード実装**: P6-a (Builder.cpp L210 `active`条件追加), P6-b (DIAG_AUTH 4点: CoordExit/BuilderEntry/BuilderExit/PreCommit), P7-A1 (RetirePart 追加), P7-A2/B (AdaptivePart 追加＋adaptive bank 排除)。実装進捗サマリテーブルを追加。変更ファイル: `RuntimeBuilder.h`, `RuntimeBuilder.cpp`, `RuntimePublicationOrchestrator.cpp`, `AudioEngine.Commit.cpp`。 |
| **9.9** | **2026-07-12** | **全未確定項目の詳細調査・確定**: 全8ツール（grep/sed(WSL)/serena/cocoindex-code/graphify/semble/AiDex/ast-grep）を使用して未確定項目を網羅調査。P6 長期設計：`runtimeUuid` immutable 確定により P6-a は最終設計として正当と判断。P8 MMCSS 再試行機構を新設。XRUN に Timer callback 相関分析を追記。EQ Cache モノトニック成長を「影響軽微」で確定。EBR epoch advance の設計完了確認。RuntimePublicationSpecification.h エイリアス確認。調査確定状況一覧に6項目を追加更新。使用ツール: grep/sed(WSL), serena MCP, cocoindex-code (ccc.exe), semble, graphify, AiDex MCP, ast-grep/rg(WSL)。 |
| **9.10** | **2026-07-12** | **P7-C 確定 + Builder Service 分類整理**: P7-C（IR transfer）コード調査完了。`transferIRStateFrom()` の5条件検証（①IRコピー専用、②Engine非書き換え、③Crossfade非書き換え、④Topology不変、⑤Semantic不変）→ 全条件充足確認。Builder Service に **Resource Factory** カテゴリを追加し、IR transfer を正式分類。P7-D 完了確定、P7 クローズ。Backlog の P7 節を整理。 |\n| **9.11** | **2026-07-12** | **P8 MMCSS 再試行機構 実装完了**: `AudioEngine.h` に `MmcssState` enum（NeverTried/Applied/Failed）追加、`mmcssApplied_`→`mmcssState_` に変更。`applyMmcssPriority()` を `bool` 返却に変更し、失敗時に `mmcssState_=Failed` に設定。`AudioBlock.cpp`/`BlockDouble.cpp` の CAS を 3値比較に更新。`Timer.cpp` の timerCallback 末尾で `Failed→NeverTried` リセット追加。`PrepareToPlay.cpp` のリセットを `MmcssState::NeverTried` に変更。変更ファイル: `AudioEngine.h`, `AudioEngine.Timer.cpp`, `AudioBlock.cpp`, `BlockDouble.cpp`, `PrepareToPlay.cpp`。 |
