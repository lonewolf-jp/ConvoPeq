# ConvoPeq メモリ肥大化 改修計画書 v6.0

**日付**: 2026-07-11
**対象**: work70 (メモリ 2.5GB 肥大化問題)
**前版**: v5.60-2 (2026-07-11)

> **本計画書は「これから行う改修」を先頭にまとめ、「完了した改修」は Appendix に配置している。**
> 凡例: ✅ FACT / 🔍 Strong HYPOTHESIS / 🔍 HYPOTHESIS / 💡 PROPOSAL / ⚠️ CAVEAT

---

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
- **Validator 通過は「Validator が保証する Invariant が成立している」ことのみを意味する。Authority Contract (別段階の Precheck) までは保証しない。**
- Validator 不合格 ≠ Builder が間違っている（Builder 入力が不適切な場合もあり得る）

**Validator と Authority Contract / Precheck は独立した検証段階である**:
- Validator 通過 ≠ Authority Contract 成立（gen=5/8 で実証済み）
- Authority Contract 通過 ≠ Precheck 成立（Precheck 内で別の制約が追加され得る）
- 各段階は独立して動作し、前段の結果に依存しない

この分離により:
- Builder は Pure Construction に専念できる（No Validation）
- Validator は Invariant のみに専念できる（No Fix）
- 将来の構築ルール変更が Builder と Validator に独立して影響を与えない

---

## [設計] 0. エグゼクティブサマリ

### 0.1 改修完了状況

| Phase | 状態 | 内容 | 参照 |
|:------|:-----|:------|:-----|
| P1-a | ✅ **完了** | Coordinator direct publish に handle 登録追加 | Appendix A.1 |
| P1-b | ✅ **完了** | advanceFade 配線 | Appendix A.2 |
| P1-c | ✅ **完了** | MEM_SNAP 監視強化 | Appendix A.3 |
| P1-a-FIX | ✅ **完了** | activeRuntimeDSPHandle_ 未更新修正 | Appendix A.4 |
| P1-a-FIX-2 | ✅ **完了** | DSPGuard 直接破棄パス | Appendix A.5 |
| P1-a-FIX-3 | ✅ **完了** | 0xC0000005 修正（DSPGuard 重複destroy） | Appendix A.6 |
| D-1 | ✅ **完了** | destroyRolledBackDSP | Appendix A.7 |
| Phase 2 | ✅ **完了** | Publish→Retire→EBR→Destroy→Memory 検証 | — |
| P3 (AoS→SoA) | ✅ **完了済み** | 主要変換は既に完了、残作業はコードレビューのみ | — |

総合的なメモリ効果: 未修正時の **2,477MB** から **定常 686MB / ピーク 1,094MB** に改善（72%削減）。

**ただし `lifecycle(retire)=0` は継続中**。これは handle 未登録が原因ではなく、gen=3 以降の publish が全滅（AUTH_CONTRACT）したため retire 機会そのものが発生しなかったため（詳細: [未確定] 7.2→[設計] 1）。

### 0.2 残存課題と優先順位

| 優先度 | 課題 | 内容 | 難易度 | リスク | 期待効果 |
|:-------|:-----|:------|:-------|:-------|:---------|
| **1** | **CrossfadePlan 導入** | AUTH_CONTRACT / gen=4/5/8 失敗の根本修正。Builder に CrossfadePlan を入力として渡し、Post-build Mutation を排除。 | 中 | 低 | ✅ 正しい IR 適用可能に |
| **2** | **P2-1 BlockSize 最適化** | 初回 SAFE_MAX_BLOCK_SIZE=65536 による 524288 バッファ肥大の修正。DSPCore 生確保量を 159MB→3.4MB に削減。 | 中 | 低 | ✅ DSPCore あたり ~189MB 削減 |
| **3** | **P-NS: NoiseShaper 診断改善** | block.sampleRateHz と session.sampleRateHz の DIAG ログ出力を追加し、accepted=0 の真因を特定可能にする。 | 低 | 低 | 🔍 調査基盤改善 |
| **4** | **P-DIAG: MEM_SNAP バケット改善** | 680MB "Other" の内訳を把握するためのメモリバケットタグ追加。少なくとも DSPCore/RuntimeWorld/MKL/CRT を分離。 | 中 | 低 | 🔍 680MB 内訳可視化 |
| **5** | **P4: mkl_malloc DIAG 化** | CacheManager(2) + IRConverter(1) の未計装 mkl_malloc 追跡。メモリ削減効果なし、透明性向上。 | 低 | 低 | — |
| — | EBR / lifecycle / runtimeDSPHandleMap / BlockSize実測 / 680MB内訳 | 🔍 **全て [未確定] 7 に集約済み**。Runtime 観測依存のためコード調査では確定不能。 | — | — | — |
| — | NoiseShaper accepted=0 | 🔍 **コード調査では確定不能。Runtime DIAG が必要**（→ [設計] 4. P-NS）。 | — | — | — |

### 0.3 検証済みの設計判断

| 判断 | 結論 | 根拠 |
|:-----|:------|:------|
| advanceFade のサンプル単位 | **OS 補正不要。** コールバックサンプル数のまま減算 | `getNextAudioBlock()` の `numSamples` は callback block size |
| `FadeState::Completed` 追加 | **不要。** 既存の remaining==0 チェックで完了検出可能 | `tryComplete()` は `remainingCount() > 0` を直接チェック |
| `memory_order` 変更 | **現状維持（relaxed 化は却下）。** 現状の release/acquire が正しい HB を提供 | remaining→tryComplete の同期に release/acquire が必要 |
| CrossfadeRuntime への完了通知追加 | **不要。** SnapshotCoordinator と CrossfadeRuntime は責務が完全に独立 | 別セクション参照 |

---

## [設計] 1. CrossfadePlan 導入による AUTH_CONTRACT 修正

### 1.1 問題: Post-build Mutation が Authority Contract 違反を引き起こす

✅ **FACT**: gen=4/5/8 の publish 失敗は以下のメカニズムで発生する:

```
gen=4: Orchestrator buildRuntimePublishWorld(newDSP, oldDSP, HardReset, 0.0, false)
  → active=false, next=oldDSP≠nullptr
  → Builder: hasFadingRuntime=false, fadingRuntimeUuid=oldDSP_uuid≠0
  → needsCrossfade=false → override 未適用
  → Validator L92: hasFadingRuntime(false) != fadingRuntimeUuid≠0 → FAIL

gen=5/8: 同種のトポロジに crossfade rebuild のため override 適用
  → transitionActive=true, hasFadingRuntime=true （override 設定）
  → Validator 通過
  → Precheck AUTH_CONTRACT L87: graph.fadingNode(nullptr) != hasFadingRuntime(true) → FAIL
  → 原因: override は graph.fadingNode を更新しない（Post-build Mutation の限界）
```

**根本原因**: Builder が一括生成した `RuntimePublishWorld` に対して Orchestrator が部分上書き（Post-build Mutation）を行う設計。Builder の Pure Construction 責務と矛盾し、`graph.fadingNode` のような内部状態の更新漏れが発生する。

### 1.2 修正案: CrossfadePlan を Builder 入力として渡す

**ConvoPeq.md の CrossfadeAuthority 定義に基づく**。

```cpp
// 新規: CrossfadePlan — Builder への完全な Graph Topology 構築指示
//   Builder はこの構造体に記述された内容をそのまま RuntimePublishWorld に写像する。
//   自身で Crossfade 要否を判断せず、常に plan の指示に忠実に従う。
//   将来 FadePolicy の種類が増えても（HardReset/SoftReset/Instant/Progressive 等）
//   フィールド追加で対応可能。enum 拡張不要。
enum class FadePolicy {
    None,       // クロスフェードなし（即時切り替え）
    Standard,   // 標準クロスフェード（Generator が fading を管理）
};

struct CrossfadePlan {
    FadePolicy policy = FadePolicy::None;
    double fadeTimeSec = 0.0;
    // Builder は fadingNode を plan から受け取り、そのまま graphState に設定する。
    // 自身で fadingNode の有無を判断しない。
    RuntimeGraphNode* fadingNode = nullptr;
};

// RuntimeBuilder のシグネチャ変更:
RuntimePublishWorld buildRuntimePublishWorld(
    DSPCore* current, DSPCore* next,
    RuntimeBuildPolicy policy, double fadeTimeSec, bool active,
    const GlobalSnapshot* sealedSnapshot,           // 既存のデフォルト引数
    const CrossfadePlan& plan = CrossfadePlan{}     // 新規追加
) noexcept;
```

**変更影響**: 7箇所の呼び出し中、Orchestrator のみが plan を渡す。残り6箇所（Init/PrepareToPlay/ReleaseResources/Timer/Transition/PublicationExecutor）はデフォルト値（policy=None, fadingNode=nullptr）を使用。

**期待効果**: Builder は Plan に書かれた Graph Topology をそのまま構築するのみ。自身で判断しない。Post-build Mutation が不要になり、AUTH_CONTRACT FAIL が解消される。

✅ **FACT（2026-07-11 コード調査確定）**: CrossfadePlan 追加は構造的に実現可能。`buildRuntimePublishWorld()` は既に `sealedSnapshot = nullptr` のデフォルト引数パターンを持つ。同様のパターンで `CrossfadePlan plan = CrossfadePlan{}` を追加可能。

**将来の拡張性**: 現在は `FadePolicy::None / Standard` の2値だが、将来 `HardReset` / `SoftReset` / `Instant` / `Progressive` 等が追加されても CrossfadePlan のフィールド拡張のみで対応可能。Builder のシグネチャは不変。

### 1.3 実装ステップ

| Step | 内容 | ファイル |
|:-----|:------|:---------|
| 1 | `CrossfadePlan` 構造体を `RuntimeBuilder.h` に追加 | `RuntimeBuilder.h` |
| 2 | `buildRuntimePublishWorld()` に `CrossfadePlan` 引数を追加（デフォルト値付き） | `RuntimeBuilder.h/.cpp` |
| 3 | Orchestrator `trySubmit()` 内で CrossfadeAuthority の決定を plan に詰めて Builder に渡す | `RuntimePublicationOrchestrator.cpp` |
| 4 | Builder 内で `plan` に記述された Graph Topology（fadingNode/policy）をそのまま `graphState` に写像 | `RuntimeBuilder.cpp` |
| 5 | 残り6箇所の呼び出しを確認（デフォルト引数で動作。変更不要） | 各 .cpp |
| 6 | gen=4/5/8 が publish 成功することを DIAG ログで確認 | — |

---

## [設計] 2. P2-1: 初回 BlockSize 最適化

### 2.1 問題

初回 `DSPCore::prepare()` 時の `internalMaxBlock=524288`（`SAFE_MAX_BLOCK_SIZE 65536 × MAX_OS_FACTOR 8`）。
この値は全バッファ（Oversampling/SoftClip/EQ/TruePeakDetector等）のサイズを決定する。

**本質的な問題**: `DSPCore::prepare()` は `inputMaxBlock = max(SAFE_MAX_BLOCK_SIZE, samplesPerBlock)` で計算する。初回 prepare（Init 直後）では `maxSamplesPerBlock` が `SAFE_MAX_BLOCK_SIZE=65536` のままのため、`inputMaxBlock=65536` となる。これにより `internalMaxBlock=524288` の巨大バッファが確保される。

**修正方針**: `SAFE_MAX_BLOCK_SIZE` そのものを削減するのではなく、**初回 prepare 時に使用する `maxSamplesPerBlock` を実用的な値（prepareToPlay 確定後に再設定される値）に変更する**。

✅ **FACT（定量化完了）**: 初回 prepare 時の `internalMaxBlock` を 524288→8192 に削減した場合、DSPCore 内部バッファ（tracked allocations）は:
- 生確保量: ~193 MB → ~3.4 MB (**tracked allocations で ~189 MB節約**)
- ⚠️ ただしこの削減量は DSPCore 内部のコード追跡可能なバッファのみ。隠れオーバーヘッド（~191MB）は比例縮小の可能性があるが、実 Private Bytes 削減量は実装後の MEM_SNAP で確認する。

### 2.2 原因チェーン

1. `AudioEngine.Init.cpp:38`: `maxSamplesPerBlock = SAFE_MAX_BLOCK_SIZE (65536)`（デバイス未初期化の安全策）
2. `AudioEngine.Init.cpp:57`: 初回 Structural rebuild を submit
3. `DSPCoreLifecycle.cpp:140-142`: `inputMaxBlock = max(65536, samplesPerBlock) = 65536`、`internalMaxBlock = 65536 × 8 = 524288`
4. この 524288 が全バッファのサイズとなる（osFactor に非依存）

### 2.3 実装案

| 案 | 難易度 | リスク | 期待削減 | 説明 |
|:---|:-------|:-------|:---------|:------|
| **A（推奨）**: 初回 prepare 時に使用する `maxSamplesPerBlock` を小さな値（例: 512）に設定し、prepareToPlay で実SR/BSに再設定 | 中 | 低 | ~189MB/DSPCore | 最も実装自由度が高く、`SAFE_MAX_BLOCK_SIZE` 自体は変更しない |
| B: `SAFE_MAX_BLOCK_SIZE` の低減 | 高 | 中 | ~189MB/DSPCore | クロスフェードバッファ等他の依存箇所に波及。独立性が低い |
| C: 初回 prepare で internalMaxBlock 上限を実ブロックサイズに制限（osFactor 未確定のため 1x 相当） | 低 | 低 | ~189MB/DSPCore | 最も安全、後続の prepareToPlay で再設定される |

---

## [設計] 3. P4: 未計装 mkl_malloc DIAG 化

### 3.1 範囲

✅ **FACT**: DIAG 非対応の `mkl_malloc`/`mkl_free` は以下の 3 箇所のみ:
- `CacheManager.cpp:190` — IR データキャッシュ
- `CacheManager.cpp:228` — IR データコピー
- `IRConverter.cpp:187` — IR 変換バッファ

`AlignedAllocation.h` の汎用ラッパー（`aligned_malloc`/`aligned_free`）は広範囲で使用されるため P4 の対象外。

### 3.2 期待効果
- 直接のメモリ削減効果はない
- MKL アロケーションの完全可視化（`lostFree` / `alloc` の完全性向上）
- 優先度: **低**

---

## [設計] 4. P-NS: NoiseShaper 診断改善

### 4.1 問題

NoiseShaper learner の `Waiting diagnostics` ログでは `accepted=0, dropSampleRate=~3000` が全期間継続する。コード調査によりブロック側・セッション側のサンプルレート設定は正しい（ともに base rate 192000）ことが確認されたが、Runtime で実際に **drop 判定に使用された block.sampleRateHz と session.sampleRateHz の実値**が正しいかは未確認。

`currentCaptureSessionId` は初期値0以降書き込みが存在しないため、`sessionId` によるドロップは発生しない（FACT #77）。

### 4.2 変更内容

`captureSessionSignature()` および `drainCaptureQueue()` の DIAG ログに以下を追加:

| # | 追加内容 | ファイル | 行数 |
|:-:|:---------|:---------|:-----|
| 1 | `block.sampleRateHz` を `[NoiseShaperLearner] drainCaptureQueue` ログに出力 | `NoiseShaperLearner.cpp` | +1 |
| 2 | `session.sampleRateHz` と `block.sampleRateHz` の実値同時出力（Waiting diagnostics行に追記） | `NoiseShaperLearner.cpp:916` | +1 |
| 3 | GlobalSnapshot に `block.sampleRateHz` の統計情報を追加（可能な場合） | `AudioEngine.h` | +3 |

### 4.3 期待効果

- `block.sampleRateHz` の実値が 192000 ≠ 192000 のように不一致なら問題確定
- 両者が一致している場合、ドロップ原因は `segmentBuffer.pushBlock()` の暗黙的失敗や他の要因に絞り込める
- 目的は「sampleRate mismatch の確認」ではなく「**drop 判定に使われた block/session 値を Runtime で可視化すること**」
- 優先度: **中**（NoiseShaper の継続学習が完全に停止しているため）

---

## [設計] 5. P-DIAG: MEM_SNAP メモリバケット改善

### 5.1 問題

定常 683-686MB のうち、現状の MEM_SNAP は NUC/DC/SC/Ret 以外をすべて「Other」として一括計上する。以下の内訳が診断不能:
- DSPCore サブシステムごとの内訳（Oversampling/SoftClip/EQ/Buffers/Convolver/TruePeakDetector 等）
- RuntimeWorld (active generation state)
- RuntimeBuilder 一時バッファ
- MKL 内部状態
- C++ CRT ヒープ断片化
- JUCE framework

### 5.2 変更内容（推奨: `DSPCore::memoryUsage()` 診断API方式）

**設計判断**: `sizeof(DSPCore)` では内部の vector/AudioBuffer/Oversampler/Convolver 等のヒープ確保まで含まれない。そのため以下の2案を比較:

| 案 | 実装量 | 正確さ | 将来保守性 |
|:---|:-------|:-------|:----------|
| **A（推奨）**: `DSPCore::memoryUsage()` 診断専用APIを追加 | 中 | ◎ | ◎ サブシステム追加時に追従容易 |
| B: 個別フィールドを MEM_SNAP にベタ書き | 低 | ○ | △ サブシステム追加時に修正漏れリスク |

**案Aの API 設計案**:
```cpp
struct DSPCoreMemoryBreakdown {
    size_t totalTracked = 0;       // 合計
    size_t oversampling = 0;       // Oversampling work buffers
    size_t softClip = 0;           // SoftClip OS work buffers
    size_t eqProcessor = 0;        // EQ scratch/dry/parallel/structure/msWorkBuffer
    size_t alignedBuffers = 0;     // alignedL/R + dryBypassL/R
    size_t latencyBuffers = 0;     // fixedLatency × 4
    size_t truePeakDetector = 0;   // TruePeakDetector internal
    size_t convolver = 0;          // Convolver internal (no IR = minimal)
    size_t crossfade = 0;          // JUCE crossfade buffers
    size_t misc = 0;               // DCBlocker/LoudnessMeter/PeakLimiter/NoiseShaper
};
DSPCoreMemoryBreakdown DSPCore::memoryUsage() const noexcept;
```

MEM_SNAP 出力の拡張例:
```
[MEM_SNAP] gen=3
  DC: live=1 | OS:64MB SC:32MB EQ:21MB Buf:16MB Lat:16MB TPD:8MB Conv:0MB XF:2MB
  DC-total:159MB | World: gen=3 size=??MB | Ret: slots=0 tot=0MB | NUC: live=0 alloc=0MB
```

**優先度**: 中。ただし `DSPCore::memoryUsage()` の設計・実装には工数がかかるため、P1-c の拡張範囲に入れるか否かは別途判断。

| # | フィールド | 取得元 | 意味 |
|:-:|:-----------|:-------|:------|
| 1 | `DC: live=N alloc=NNNMB` | DSPCore::liveCount + バッファ合計 | DC に紐づく全 tracked バッファ確保量（コード数式 ~159MB を DIAG で実測） |
| 2 | `World: gen=M size=NNNMB` | RuntimePublishWorld の世代 + サイズ | アクティブ RuntimeWorld のメモリコスト |
| 3 | `Ret: slots=N tot=NNNMB` | EBR キュー内エントリ数 + 合計バイト | `m_pendingRetireBytes_` の値が正しく設定される前提（現状は ISRRetireRouter のデッドコード） |
| 4 | `MKL: nuc=N lost=N zero=N` | NUC 診断（既存） | 既存の NUC フィールドでカバー済み。拡張不要。 |

### 5.3 期待効果
- 680MB "Other" のうち DSPCore 実測値が特定可能になる（現状はコード数式からの推定のみ）
- RuntimeWorld の世代交代に伴うメモリ増減が追跡可能
- 優先度: **中**（680MB 内訳の透明性向上）

---

## [設計] 6. アーキテクチャ検証: Authority 境界・Invariant・RT Safety

*本節はすべてのフェーズに共通する設計参照情報。改修完了後も維持すべき不変条件を示す。*

### 4.1 Authority Boundary Chart — Single Authority 設計

| 責務 | Authority | コード位置 | 備考 |
|:-----|:----------|:----------|:------|
| **`activeRuntimeDSPHandle_` 更新** | `commitRuntimePublication()` | `AudioEngine.h:3981` | publish 成功後にのみ更新．唯一の変更箇所． |
| **`DSPCore*` active 更新** | `DSPLifetimeManager::activate()` | `DSPLifetimeManager.h:28` | `setActiveRuntimeDSP(dsp)` のみ．Handle層には触れない． |
| **DSPHandle 発行** | `registerDSPHandleForRuntime()` | `AudioEngine.h:3838` | idempotent |
| **DSPHandle 回収** | `retireDSPHandleForRuntime()` | `AudioEngine.h:3875` | Map lookup → erase → Handle.retire/reclaim |
| **EBR enqueue** | `DSPLifetimeManager::retire()` | `DSPLifetimeManager.h:37` | ISRRetireRouter 経由 |
| **DSPCore* 物理破棄** | `destroyDSPCoreNode()` | `AudioEngine.Threading.cpp:15` | EBR callback または DSPGuard 直接破棄 |

### 4.2 DSPHandle ライフサイクル状態遷移（確定）

```text
         [Constructing] ← 登録直後．rollback 可能な唯一の状態
            ↙        ↘
   publish成功     publish失敗
        ↓              ↓
     [Active]      rollbackRegistration()
        ↓          CAS: Constructing→Reclaimed
    [Retired]           ↓
   (crossfade)    [Reclaimed] ← スロット再利用可能
        ↓
   [CrossfadingOut] / [CrossfadingIn] → endCrossfade() → [Active/Retired]
```

**Invariant**: Constructing のみ rollback 可能 / Reclaimed のみ create() 再利用 / Quarantined からの復帰は不可。

### 4.3 RT Safety 証明

- `activeRuntimeDSPHandle_`: publishAtomic(release) / consumeAtomic(acquire) — 正しい HB 順序
- `runtimeDSPHandleMap_`: `std::mutex` 保護（NonRT のみ）— RT スレッドは map 非アクセス
- `DSPHandleRuntime`: `std::atomic<DSPHandle>` + `std::atomic<DSPState>` ベース — ミューテックス不使用
- EBR drain: 「全 Reader が epoch を離脱した」ことを保証してから callback 実行

### 4.4 Invariant 一覧

| ID | Invariant | 根拠 |
|:---|:----------|:------|
| INV-1 | 公開前の DSP は Handle 登録済み | commitRuntimePublication が唯一の窓口 |
| INV-2 | 公開失敗時は Handle 登録をロールバック | rollbackRegistration (CAS) |
| INV-3 | Commit point 以降は Handle をロールバック不可 | `rollbackHandle = DSPHandle::null()` |
| INV-4 | Reclaimed Handle が Map に永続しない | retireDSPHandleForRuntime で削除 |
| INV-5 | rollback 後、同一 DSPCore は再登録可能 | Constructing→Reclaimed により再利用可能 |
| INV-6 | Handle 登録と publish は同一トランザクション内 | commitRuntimePublication で完結 |
| INV-7 | rollback 成功後は Map 不整合を許容（次回登録で修復） | CAS 成功が最優先 |
| INV-8 | alreadyRegistered の Handle は Constructing 状態 | CAS 条件 |
| INV-9 | activeRuntimeDSPHandle_ は常に Active 状態 | activate() のみ更新 |
| INV-10 | publish 成否を OwnershipDisposition で明示 | Transferred / CallerDestroy |

---

## [未確定] 7. 未解決課題（Runtime 観測依存）

*本セクションは「設計書の時点で確定できず、将来の Runtime 観測または実装後に検証が必要な事項」を集約する。*

### 7.1 NoiseShaper accepted=0 の真因

🔍 **HYPOTHESIS**: `accepted=0` の原因はコード調査の範囲では特定できなかった。

**確定済みの事実**:
- セッション側 `session.sampleRateHz` は `engine.currentSampleRate`（192000）から設定（NoiseShaperLearner.cpp:1034）
- ブロック側 `block.sampleRateHz` は `dsp->sampleRate`（192000 = base rate）から設定（AudioEngine.h:3552 → DSPCoreLifecycle.cpp:104）
- 両者はコード上**一致するはず**（192000 = 192000）

**誤りと判明した仮説**: 従来の「`processingRate` bug」説（`dsp->sampleRate` が processingRate 768000 と誤って設定されている）はコード検証の結果誤り。`dsp->sampleRate` は正しく base rate を保持する。

**新規確定事実（2026-07-11 コード調査）**:
- ✅ `currentCaptureSessionId` は **初期化（=0）以降、書き込みが存在しない**（AudioEngine.h:943 のみ）。したがって `block.sessionId` は常に 0。セッション側も `session.sessionId==0` のため `sessionIdCompatible` チェックは常に通過。ログの `droppedBySession=0` と整合。
- ✅ ブロック側 `block.sampleRateHz` とセッション側 `session.sampleRateHz` はともに `dsp->sampleRate`（base rate）または `engine.currentSampleRate` から設定され、コード上一致する。
- ✅ `audioCaptureQueue` は 4096 エントリの共有リングバッファ。gen=3 は ~10 秒間動作し、全エントリを上書き可能。したがって gen=1（48000Hz）のブロック残留説は成立しない。

**結論**: コード調査では真因の特定は不可能。Runtime DIAG により `block.sampleRateHz` の実値を出力する改善が必要（→ [設計] 4. P-NS）。

**残存する可能性**（いずれも Runtime 観測が必要）:
- `a`) `segmentBuffer.pushBlock()` の暗黙的な失敗（キューから pop できても segment キューが満杯で push できない）
- `b`) セッション開始タイミングの競合：キューと `startLearning` の間の瞬間的な不整合
- `c`) ConvoPeq.md 上の未確認な条件分岐

**確認方法**: [設計] 4. P-NS の DIAG 改善適用後、`block.sampleRateHz` と `session.sampleRateHz` の実値を確認。

### 7.2 EBR 経路の未検証

⚠️ **CAVEAT**: 今回の ConvoPeq.log では EBR（Epoch-Based Reclamation）は一度も発動しなかった。

| 指標 | 値 | 意味 |
|:-----|:----|:------|
| `pend` | 0 | Retire 待ちエントリなし |
| `reclaim` | 0 | EBR reclaim が一度も実行されず |
| `retire` (lifecycle) | 0 | `retireDSPHandleForRuntime()` が一度も成功せず |

✅ **FACT（2026-07-11 コード調査で確定）**: EBR 機構自体は**完全に実装されている**。
- Reader thread 登録: `registerReaderThread()`（ISRRetireRouter.cpp:47）
- Reader epoch 参加: `enterReader(readerIndex)`（ISRRetireRouter.cpp:59）
- Reader epoch 離脱: `exitReader(readerIndex)`（ISRRetireRouter.cpp:65）
- Epoch 進行: `ISRRetireRouter::currentEpoch()` → `EpochDomain` globalEpoch fetch_add
- Enqueue: `enqueueRetire(ptr, deleter, epoch)`（DSPLifetimeManager.h:50）
- Reclaim: Timer 50ms周期で `tryReclaim()` → `getMinReaderEpoch()` → safe entries freed
- QueueFull フォールバック: `tryReclaim()` で backlog 消化後、再試行

**⚠️ 重要訂正（2026-07-11 v6.2）**: 従来の「未発動の原因は handle 未登録のみ」という説明は**誤り**。ConvoPeq.log は P1-a/FIX 適用済みバイナリ（v5.44）で収録されているにもかかわらず `lifecycle(retire)=0` が継続しているため、handle 未登録が原因ではない。

**正しい因果関係**:
1. ✅ P1-a/FIX により handle 登録は正しく行われていた
2. gen=3 の Coordinator direct publish は成功し、DSPCore#3 が active に
3. ❌ しかし gen=4/5/6/7/8 のリビルドが**全滅**（AUTH_CONTRACT / Validator reject）
4. DSPTransition::onPublishCompleted() が一度も呼ばれず、retire の機会がなかった
5. したがって EBR enqueueRetire() にエントリが投入されず → pend=0, reclaim=0

`retire=0` の原因は「handle 未登録」ではなく、**「gen=3 以降に publish 成功が一度もなく、DSPCore#3 を置き換える後続 publish が存在しなかった」** ことにある。DSPCore#3 が active のまま置き換えられなかったため、retire する対象がそもそも発生しなかった。

**確認方法**: CrossfadePlan 導入（[設計] 1）により gen=4+ の publish が成功するようになった後、IR 切替 → DSPTransition → `lifetimeMgr.retire(oldDSP)` → `retireDSPHandleForRuntime`（今度は handle が登録済みなので true）→ EBR enqueue → pending>0 → Timer tryReclaim → reclaim>0 → private 減少 の完全チェーンを確認する。

### 7.3 lifecycle(retire) の実測値

🔍 **HYPOTHESIS**: 現状の lifecycle(retire)=0 は「handle 未登録」ではなく「publish 成功自体が gen=3 以降存在しない」ことが原因。

**確定済みのチェーン**（ConvoPeq.log v5.44 = P1-a/FIX 適用済み）:
- gen=1: Coordinator direct publish（v5.44 の P1-a により handle 登録済み ✅）
- gen=3: prepareToPlay 経由の Coordinator direct publish（handle 登録済み ✅）
- gen=4/5/6/7/8: **全滅**（AUTH_CONTRACT / Validator reject）→ DSPTransition 未実行
- → DSPCore#3（gen=3）は active のまま置き換えられず、retire 対象が発生しない
- → `lifecycle(retire)=0` はこの状況を正しく反映している（前提条件の問題であって EBR 経路の問題ではない）

**CrossfadePlan 導入後（[設計] 1）の期待**:
- gen=4+ の publish が成功する
- DSPTransition::onPublishCompleted() が oldDSP を retire
- `retireDSPHandleForRuntime(oldDSP)` → Map find → erase → Handle.retire/reclaim → EBR enqueue
- → lifecycle(retire)>0 に増加

✅ **FACT**: handle 登録さえ行われれば、retire → EBR → destroy の後段経路に既知のブロック要因は存在しない（QueueFull 時も tryReclaim フォールバックあり）。

**確認方法**: CrossfadePlan 導入後の MEM_SNAP で lifecycle(retire)>0, pending>0, reclaim>0 のシーケンスを確認。

### 7.4 runtimeDSPHandleMap 収束値

⚠️ **CAVEAT**: steady-state では 2-3 エントリに収束する見込み（FACT #38）。ただし transition 中（crossfade / rollback / rebuild prepare の重なり）は一時的に増加し得る。MAX_DSP_SLOTS=256 に対して十分小さく、枯渇リスクは極めて低い。実測による確認が必要。Runtime 観測依存。

### 7.5 BlockSize 削減効果の実測

🔍 **HYPOTHESIS**: P2-1 実装により SAFE_MAX_BLOCK_SIZE=65536→1024 で DSPCore 生確保量が 98.2%削減（~189MB/DSPCore）することをコード数式で定量化済み（[設計] 2）。ただしこの効果は隠れオーバーヘッド（~191MB）を含まない理論値であり、実メモリ削減量は P2-1 実装後の MEM_SNAP で確認する。メモリバケット DIAG（[設計] 5）と併用することで効果検証が容易になる。

### 7.6 680MB "Other" の内訳

⚠️ **CAVEAT**: 定常 683-686MB のうち、現状の MEM_SNAP は NUC/DC/SC/Ret 以外をすべて「Other」として一括計上する。以下のカテゴリが混在しており個別の内訳は診断不能:

**Application-owned**: DSPCore + 全サブコンポーネント、RuntimeWorld、Transition/crossfade、Snapshot、IO/Capture buffer
**Allocator-owned**: C++ CRT heap（断片化・フリーリスト）、VirtualAlloc reservation/commit、スレッドスタック
**External library**: MKL internal（FFT plan/scratch/TLS）、IPP internal
**Framework**: JUCE（数十MB程度 — 680MBの主因ではない）

**確認方法**: [設計] 5. P-DIAG の MEM_SNAP バケット改善により部分的な可視化が可能。

---

## Appendix A: 実装済み改修内容

*以下は完了した改修の詳細。コードレビュー・テスト結果を含む。*

### A.1 P1-a: publish 経路への handle 登録追加

**目的**: Coordinator direct publish（Init/PrepareToPlay/ReleaseResources/Timer/Transition）で DSPCore が `runtimeDSPHandleMap_` に未登録となる問題を修正。

**設計**: `commitRuntimePublication()` + `RegistrationContext` により、register → publish → 失敗時 rollback のトランザクションを保証。

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

| # | 呼び出し元 | mode | dsp | handle | 備考 |
|:-:|:-----------|:-----|:----|:-------|:------|
| 1 | Init | none | — | — | Bootstrap world |
| 2 | PrepareToPlay (first) | needsRegistration | currentForPublish | — | Coordinator direct |
| 3 | PrepareToPlay (rebuild) | needsRegistration | getActiveRuntimeDSP | — | Coordinator direct |
| 4 | ReleaseResources | none | — | — | シャットダウン |
| 5 | Timer (fadeComplete) | needsRegistration | currentAfterFade | — | Coordinator direct |
| 6 | Transition | needsRegistration | 引数 newDSP | — | Coordinator direct |
| 7 | PublicationExecutor | alreadyRegistered | — | req.newDSP | Orchestrator → DSPTransition |

**検証**: CTest 15/15 PASS, CI Gates ALL PASS.
//   ★ v5.37: 現状 O(n) linear scan（MAX_DSP_SLOTS=256 のため問題なし）。名前は eraseByHandle
//     だが実装は full scan であることを明記。将来の reverse map で O(1) に最適化予定。
//     現状: O(n) full scan。将来: HandleRegistry reverse map → O(1)。
private:
[[nodiscard]] bool eraseByHandle(convo::isr::DSPHandle handle) noexcept
{
    std::lock_guard<std::mutex> lock(runtimeDSPHandleMapMutex_);
    for (auto it = runtimeDSPHandleMap_.begin(); it != runtimeDSPHandleMap_.end(); ++it)
    {
        if (it->second == handle)
        {
            runtimeDSPHandleMap_.erase(it);
            return true;
        }
    }
    // Map に存在しなくても CAS 成功済みのため rollback は成功（INV-4/INV-7 対象外）
    return true;
}

// ★ 後方互換用: DSPCore* 版（bool のまま）
//   ★ v5.22: 互換レイヤー。将来の二方向 Map 導入後、DSPHandle 版に統一し削除予定。
//   ★ v5.24: [[deprecated]] を付与し、新規コードでの使用を禁止。
//   TODO: HandleRegistry リファクタリング時に削除
[[deprecated("Use rollbackDSPHandleRegistration(DSPHandle) instead")]]
bool rollbackDSPHandleRegistration(DSPCore* dsp) noexcept
{
    if (dsp == nullptr) return false;
    convo::isr::DSPHandle handle;
    {
        std::lock_guard<std::mutex> lock(runtimeDSPHandleMapMutex_);
        const auto it = runtimeDSPHandleMap_.find(dsp);
        if (it == runtimeDSPHandleMap_.end()) return false;
        handle = it->second;
    }
    if (!handle.isNull())
    {
        if (!dspHandleRuntime_.rollbackRegistration(handle))
            return false;
    }
    {
        std::lock_guard<std::mutex> lock(runtimeDSPHandleMapMutex_);
        for (auto it = runtimeDSPHandleMap_.begin(); it != runtimeDSPHandleMap_.end(); ++it)
        {
            if (it->second == handle)
            {
                runtimeDSPHandleMap_.erase(it);
                break;
            }
        }
    }
    return true;
}

// ISRDSPHandle.cpp に追加
// ★ work70: 登録のロールバック。
//   Constructing → Reclaimed への CAS により、「登録されたが公開されなかった」状態を
//   スロット利用可能プールへ戻す。
//   ▲ 重要: Constructing のみ rollback 対象。Constructing 以外の状態（Active/Retired/Reclaimed）
//     にある slot は rollback してはならない（publish 成功後または既に寿命終了）。
//     CAS が失敗した場合（Constructing 以外の状態）は false を返す。
//   ★ 設計判断 (v5.18): CAS を先に実行し、成功後に instance=nullptr を設定する。
//     理由: state（DSPState）が slot 再利用の authority である。CAS で Constructing→Reclaimed
//     に遷移した後に instance をクリアしても、他スレッドは Reclaimed を確認してから
//     create() でスロットを再利用するため安全。逆順（instance=null → CAS）では
//     Constructing 状態のまま instance が null になる window が生じ、resolve() が
//     不正な null instance を返す可能性がある。
//   ★ v5.31 コード確認確定: `resolve()`（ISRDSPHandle.cpp:40-55）は generation 一致確認後、
//     state（Reclaimed/Quarantined のみ無効）を authority として有効性を判定する。
//     instance は state が有効な場合にのみ返される（reg.instance を直接返す）。
//     したがって resolve の観点からは state が唯一の authority であり、instance=nullptr の
//     設定が state 変更の前後どちらでも安全（state 次第で resolve が instance を参照しない）。
//     この前提が変更される（resolve が state を見ずに instance のみを返す）場合には
//     rollbackRegistration の順序も再評価が必要。
//   ★ v5.33: instance=nullptr を削除。rollback を「state のみの操作」に純化。
//     「rollback が instance を消す」という責務は不要。理由:
//     - resolve() は state を authority とする（state が Reclaimed なら instance を返さない）
//     - create() は常に instance を上書きする（reclaimed→Constructing 時に dspInstance をセット）
//     - rollback と create の間に instance!=nullptr の時間窓が存在しても resolve は state で防御
//     reclaim() は retire→解放の最終段階として instance=nullptr + state=Reclaimed を設定する
//     （rollback とは責務が異なる — rollback は「未公開のまま利用可能プールに戻す」、
//      reclaim は「メモリ解放済み」）。この責務分離により ISR の state machine がより明確になる。
//   ★ v5.35: create() に jassert 追加推奨。rollback 後も instance が残っていることを確認:
//     jassert(reg.instance == nullptr || reg.state == DSPState::Reclaimed);
//     これにより「rollback 後に instance が不正に残っている」状態を Debug で検出できる。
//     Production では jassert は無効化されるためオーバーヘッドなし。
//   ★ v5.37: rollbackRegistration は Constructing 専用。将来 Publishing/PreActive 等の
//     中間状態が追加された場合は本関数の前提が変わるため再設計が必要。
//     Only Constructing may be rolled back. Future intermediate states require redesign.
[[nodiscard]] bool DSPHandleRuntime::rollbackRegistration(DSPHandle handle) noexcept
{
    if (handle.isNull() || handle.slot >= MAX_DSP_SLOTS) return false;
    auto& reg = registry_[handle.slot];
    DSPState expected = DSPState::Constructing;
    // ★ v5.33: state のみ CAS（instance は不変）。create() が上書きするため不要。
    return convo::compareExchangeAtomic(reg.state, expected, DSPState::Reclaimed,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire);
}
```

✅ **FACT（2026-07-10 コード調査確定）**: `registerDSPHandleForRuntime()` の `emplace` は find→return の後のみ実行される。`std::unordered_map::emplace` は重複キーの場合でも既存エントリを変更しない（insert 相当）。noexcept 関数内で例外は発生せず、`create(dsp)` で取得した Handle が孤立することはない。

✅ **FACT（2026-07-10 コード調査確定）**: `DSPHandleRuntime::create()` は `DSPState::Reclaimed` のスロットを再利用する（ISRDSPHandle.cpp:25）。`Constructing → Reclaimed` への rollback はこの「スロット利用可能プール」に戻す動作であり、振る舞いとして正しい。`Reclaimed` が「寿命終了」と「利用可能」の両方の意味を持つことは状態名の限界だが、内部的な振る舞いは一貫している。

#### 呼び出しパターン

**Coordinator direct publish 経路** (Init/PrepareToPlay/ReleaseResources/Timer/Transition):
```cpp
auto coordinator = makeRuntimePublicationCoordinator();
auto worldOwner = worldBuilder.buildRuntimePublishWorld(dsp, ...);
const auto result = commitRuntimePublication(
    coordinator, std::move(worldOwner),
    RegistrationContext::needsRegistration(dsp));
if (!PublishStageResultTraits::isCommitted(result.stage)) return;
```

**Orchestrator 経路** (PublicationExecutor):
```cpp
// Orchestrator は enqueuePublicationIntentForRuntimeCommit() 内で事前に
// registerDSPHandleForRuntime(newDSP) を実行済み。その結果の DSPHandle を
// RegistrationContext::alreadyRegistered() として渡し、publish 失敗時の rollback に使用する。
// ★ v5.31: handle は Constructing 状態であることが前提。rollbackRegistration() は Constructing からの
//   CAS(→Reclaimed) のみ成功する。
auto coordinator = engine.makeRuntimePublicationCoordinator();
const auto result = engine.commitRuntimePublication(
    coordinator, std::move(stateOwner),
    RegistrationContext::alreadyRegistered(registeredHandle));
if (!PublishStageResultTraits::isCommitted(result.stage)) return;
```

**v5.27 の変更**: `committed` は `PublishCommitResult` のメンバではなくなった。代わりに `PublishStageResultTraits::isCommitted(result.stage)` で commit 判定を行う。これにより committed と stage の二重管理を解消した。

#### 各 publish 経路のパラメータ

以下の 7 箇所の publishWorld 呼び出しについて、commitRuntimePublication の RegistrationContext パラメータを確定した。C++20 designated initializer で構築する:

| # | ファイル | RegistrationContext | 意味 |
|:-:|:---------|:-------------------|:------|
| 1 | AudioEngine.Init.cpp:48 | `RegistrationContext::none()` | Bootstrap world、DSP 未存在 |
| 2 | PrepareToPlay.cpp:155 | `RegistrationContext::needsRegistration(dsp)` | 初回 prepareToPlay、新規 DSP |
| 3 | PrepareToPlay.cpp:275 | `RegistrationContext::needsRegistration(dsp)` | placeholderDSP publish |
| 4 | ReleaseResources.cpp:155 | `RegistrationContext::none()` | Hard reset、DSP なし |
| 5 | Timer.cpp:900 | `RegistrationContext::needsRegistration(dsp)` | fadeCompleted ブロック |
| 6 | Transition.cpp:26 | `RegistrationContext::needsRegistration(dsp)` | publishIdleWorldOnly |
| 7 | PublicationExecutor.cpp:33 | `RegistrationContext::alreadyRegistered(h)` | Orchestrator が事前登録済み（FACT 確認済） |

CAVEAT: PublicationExecutor (#7) は Orchestrator 経路専用。enqueuePublicationIntentForRuntimeCommit() 内で publish 前に registerDSPHandleForRuntime(newDSP) 済み（FACT 確認済）のため handle のみ指定。

DSPTransition の重複登録について: DSPTransition::onPublishCompleted() の crossfade 経路 (needsCrossfade=true) で registerDSPHandleForRuntime(newDSP) が再度呼ばれるが、idempotent のため安全。即時 retire 経路 (needsCrossfade=false、今回の構造的 rebuild) では重複登録は発生しない。将来的なクリーンアップとして DSPHandleRuntime に findHandleByInstance() を追加し再登録不要とする方式も検討可能だが、P1-a の範囲外。

#### PublishOrigin — DIAG 専用に分離

`PublishOrigin` は DIAG 用途であり、本番 API には含めない。代わりに以下で対応する:

```cpp
// ★ DIAG ビルドのみ: commitRuntimePublication の呼び出しをトレース
#ifdef CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
struct ScopedPublishTrace {
    const char* origin_{nullptr};  // 呼び出し元識別子
    uint64_t dspPtr_{0};
    uint64_t generation_{0};

    ScopedPublishTrace(const char* origin, uint64_t dspPtr, uint64_t gen)
        : origin_(origin), dspPtr_(dspPtr), generation_(gen)
    {
        CONVOPEQ_DIAG_LOG("[HANDLE] origin=%s DSP=0x%llx gen=%llu action=begin_publish\n",
                          origin, (unsigned long long)dspPtr, (unsigned long long)gen);
    }
    ~ScopedPublishTrace()
    {
        if (origin_)
            CONVOPEQ_DIAG_LOG("[HANDLE] origin=%s DSP=0x%llx gen=%llu action=end_publish\n",
                              origin_, (unsigned long long)dspPtr_, (unsigned long long)generation_);
    }
};

// 呼び出し側:
ScopedPublishTrace trace("PrepareToPlay", reinterpret_cast<uint64_t>(dsp), worldGen);
const auto result = commitRuntimePublication(
    coordinator, std::move(worldOwner),
    RegistrationContext{.dsp = dsp});
if (!PublishStageResultTraits::isCommitted(result.stage)) return;
```
#else
// release ビルドでは trace 無し
#endif
```

✅ **FACT（2026-07-10 コード調査確定）**: `CONVOPEQ_DIAG_LOG` マクロは存在しない。プロジェクトでは `juce::Logger::writeToLog(juce::String::formatted(...))` または匿名名前空間の `diagLog(const juce::String&)` 関数（AudioEngine.Commit.cpp:17）を使用する。`diagLog` は `AudioEngine.Commit.cpp` 内の匿名名前空間に限定されるため、`ScopedPublishTrace` を別ファイルで使用する場合は直接 `juce::Logger::writeToLog` を呼ぶ。

このアプローチにより:
- 本番 API（`commitRuntimePublication`）に DIAG 用パラメータが混入しない
- 各呼び出し元が識別子を文字列で指定できる（enum より柔軟）
- RAII によりスコープ単位のトレースが自動化される
- 呼び出し元の記述量は `PublishOrigin` enum と同等

**HANDLE DIAG 拡張提案（v5.35）**: 以下の 4 アクションを追加することで handle lifecycle を state 単位で追跡可能にする:

```text
[HANDLE] action=register   slot=3  gen=5  state_before=Reclaimed state_after=Constructing
[HANDLE] action=rollback   slot=3  gen=5  CAS=success
[HANDLE] action=retire     slot=3  gen=5  lookup=success state_before=Active state_after=Retired
[HANDLE] action=reclaim    slot=3  gen=5  state_before=Retired  state_after=Reclaimed
[HANDLE] action=rollback_skip  slot=3  gen=5  reason=committed  // ★ v5.37: commit point 到達後の ScopeExit 無効化を追跡
```

`rollback_skip` は commit point 到達後に `rollbackHandle = DSPHandle::null()` で無効化されたケースを記録する。これにより ScopeExit が「rollback 不要だった」のか「無効化されて rollback しなかった」のかをログだけで区別できる（DIAG ビルドのみ）。

#### 責務境界まとめ

| コンポーネント | 責務 | AudioEngine 知識 |
|:-------------|:-----|:----------------|
| `commitRuntimePublication()` (AudioEngine private) | トランザクション: register → publish → rollback | Coordinator 引数で受取り、内部で register + publish + rollback を実行 |
| `RuntimePublicationCoordinator` (template) | publishWorld の実行 | なし（汎用テンプレート） |
| 各 publish 経路 (Init/PrepareToPlay/Timer等) | Coordinator 生成 + world 構築 + commitRuntimePublication 呼び出し | AudioEngine 内部メソッド |
| コアテンプレート (`src/core/RuntimePublicationCoordinator.h`) | Bridge 所有、publishWorld 実行。getBridge() は **持たない** | なし |
| `PublicationExecutor` | Orchestrator 経路の publish。RegistrationContext.handle で登録済み Handle を渡す | 型変換 + Handle 転送 |

✅ **FACT（2026-07-10 コード調査確定）**: `RuntimePublicationCoordinator` テンプレートは `Bridge bridge_` を **private メンバ**として保持する。public `getBridge()` は存在せず、Bridge にアクセスする唯一の方法は Coordinator メソッド（`publishWorld()` 内部での利用）のみ。したがって「Bridge 経由で Handle 登録を委譲する」設計はこのテンプレート制約と衝突する。この制約を尊重し、`commitRuntimePublication()` は AudioEngine のメソッドとして実装する。

#### `runtimeDSPHandleMap` の抽象化（将来の拡張性）

現在 `runtimeDSPHandleMap_` は `std::unordered_map<DSPCore*, DSPHandle>` + `std::mutex` で実装されている。この順方向 Map では Handle からの逆引きが O(n) になり、rollbackDSPHandleRegistration(DSPHandle) の Map erase が full scan となる。現状の MAX_DSP_SLOTS=256 では問題にならないが、恒久設計としては二方向 Map（`DSPCore* ↔ DSPHandle`）への移行を推奨する。P1-a 完了後のリファクタリングとして実施:

```cpp
// ★ 将来のリファクタリング候補: HandleRegistry クラス（二方向 Map）
//   現状: unordered_map<DSPCore*, DSPHandle>（順方向のみ）— Handle 逆引きが O(n)
//   将来: unordered_map<DSPCore*, DSPHandle> + unordered_map<DSPHandle, DSPCore*>
//         二方向 Map により register/retire/rollback 全てが O(1) で動作する。
//   P1-a 完了後の HandleRegistry リファクタリング時に実施。
class HandleRegistry {
public:
    DSPHandle register_(DSPCore* dsp);
    bool retire(DSPCore* dsp);
    bool rollback(DSPCore* dsp);
    RollbackResult rollback(DSPHandle handle);  // ★ O(1) erase が可能
private:
    std::mutex mutex_;
    std::unordered_map<DSPCore*, DSPHandle> forward_;
    std::unordered_map<DSPHandle, DSPCore*> reverse_;  // ★ Handle→DSPCore* 逆引き
};
```

これにより:
- rollbackDSPHandleRegistration(DSPHandle) の Map erase が O(1) になる
- 将来的な ConcurrentHashMap 置き換えが HandleRegistry 内部のみで完結
- 全マップ操作の一貫性（register/rollback/retire で同一ロック）がコード構造で保証される
- DIAG ログの追加も HandleRegistry 内に集中できる
- DSPCore* 版 rollback が不要になり、削除可能

**P1-a の最小変更方針**: P1-a では既存の `registerDSPHandleForRuntime()` 等の inline 関数を直接呼び出す。**private helper は導入しない。** HandleRegistry への抽象化は P1-a 完了後の後続リファクタリングで一括実施する。これにより P1-a の変更範囲を最小化する。

★ v5.27: private helper 案は削除。HandleRegistry リファクタリング時に一元導入するため、P1-a の段階で helper 層を入れても後で置き換えが必要になる。

**PublicationExecutor の置き換え**:
`PublicationExecutor::publish()` は `aligned_unique_ptr<FrozenRuntimeWorld>` のみ保持し、DSPCore* は **持たない**。したがって `commitRuntimePublication()` には **RegistrationContext{}（dsp なし）** を渡す。ただし Orchestrator が事前に `enqueuePublicationIntentForRuntimeCommit()` 内で登録した `DSPHandle`（`req.newDSP`）を `RegistrationContext.handle` として渡す必要がある。

**データフロー変更**: `PublicationExecutor::publish()` のシグネチャに `DSPHandle existingHandle` パラメータを追加する:
```cpp
// PublicationExecutor.h
[[nodiscard]] PublishResult publish(
    AudioEngine& engine,
    convo::aligned_unique_ptr<convo::FrozenRuntimeWorld> frozen,
    convo::isr::DSPHandle existingHandle) noexcept;  // ★ work70: 新規追加
```

**Orchestrator 側の呼び出し**:
```cpp
// RuntimePublicationOrchestrator.cpp:158 変更後
auto result = executor_.publish(engine_, std::move(frozen), req.newDSP);
```

**PublicationExecutor 側の使用**:
```cpp
const auto result = engine.commitRuntimePublication(
    coordinator, std::move(stateOwner),
    RegistrationContext::alreadyRegistered(existingHandle));
if (!PublishStageResultTraits::isCommitted(result.stage)) return;
```

`FrozenRuntimeWorld` に `toPublishWorld()` を追加する設計も可能だが、最小変更の原則から v5.7 では `releaseState()` の直接使用を維持する。`toPublishWorld()` は v5.6 で提案したリファクタリング候補であり、P1-a 実装後の改善として検討する。

**影響を受ける既存コード**:

| ファイル | 関数 | 変更前 | 変更後（RegistrationContext） |
|:---------|:------|:-------|:-----------------------------|
| `AudioEngine.Init.cpp:48` | initialize() | `coordinator.publishWorld(...)` | `commitRuntimePublication(c, w, RegistrationContext::none())` |
| `PrepareToPlay.cpp:155` | prepareToPlay() | `coordinator.publishWorld(...)` | `commitRuntimePublication(c, w, RegistrationContext::needsRegistration(dsp))` |
| `PrepareToPlay.cpp:275` | prepareToPlay() | `coordinator.publishWorld(...)` | `commitRuntimePublication(c, w, RegistrationContext::needsRegistration(dsp))` |
| `ReleaseResources.cpp:155` | releaseResources() | `coordinator.publishWorld(...)` | `commitRuntimePublication(c, w, RegistrationContext::none())` |
| `AudioEngine.Timer.cpp:900` | timerCallback() | `coordinator.publishWorld(...)` | `commitRuntimePublication(c, w, RegistrationContext::needsRegistration(dsp))` |
| `AudioEngine.Transition.cpp:26` | publishIdleWorldOnly() | `coordinator.publishWorld(...)` | `commitRuntimePublication(c, w, RegistrationContext::needsRegistration(dsp))` |
| `PublicationExecutor.cpp:33` | publish() | `coordinator.publishWorld(...)` | `engine.commitRuntimePublication(c, w, RegistrationContext::alreadyRegistered(h))` |

⚠️ **CAVEAT**: 本設計書の対象は production コードのみ。テストコード（`PartialPublicationRejectTests.cpp` 等）は `publishWorld()` を直接呼び続けてよい（テスト固有の要件）。

**Timer 経路の DSPCore* 解決**（fadeCompleted ブロック内: Timer.cpp:897）:
- publish 対象は `resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle)` の戻り値 `currentAfterFade`
- これを `commitRuntimePublication(coordinator, worldOwner, RegistrationContext::needsRegistration(currentAfterFade))` として呼び出す
- **検証**: `currentAfterFade` が publish する RuntimeWorld の生成元 DSPCore と一致することを DIAG ログで確認する

#### 設計ルール: commitRuntimePublication() 利用の徹底

**production コードでは `coordinator.publishWorld()` を直接呼んではならない。** Runtime publish は必ず `commitRuntimePublication()`（AudioEngine の private メソッド）を経由すること。これにより:
- registerDSPHandleForRuntime() の呼び忘れを構造的に防止
- 「Runtime publication を試行する DSP は publish 前に Handle 登録済みである」invariant がトランザクション構造で保証される
- 新規 publish 経路が追加されても自動的に invariant を満たす

**production コード中の全 `publishWorld()` 呼び出し（7 箇所）が `commitRuntimePublication()` に置き換わっていることを、コードレビュー + CI 静的チェックで確認する。** 1 箇所でも直接呼び出しが残ると invariant が成立しなくなる。

#### Architecture Decision: Publish の唯一の入口

**Decision**: 以降、production コードにおける `RuntimePublishWorld` の publish は全て `AudioEngine::commitRuntimePublication()` を経由する。`RuntimePublicationCoordinator::publishWorld()` を直接呼び出すことは**設計違反**であり、CI 静的チェックで禁止する。

**Rationale**:
- `commitRuntimePublication()` は register → publish → rollback のトランザクションを保証する唯一の関数である
- 直接 `publishWorld()` を呼ぶと register 漏れが構造的に発生する（今回の不具合の根本原因）
- 新規 publish 経路が追加されても、`commitRuntimePublication()` を通すことで自動的に INV-1（事前登録）が保証される

**Scope**: AudioEngine 内の全 publish 経路（Init/PrepareToPlay/ReleaseResources/Timer/Transition/PublicationExecutor）。テストコードは対象外。将来追加される publish 経路もこの制約に従う。

**`publishWorld()` へのアクセス制御**: `RuntimePublicationCoordinator` テンプレートの `publishWorld()` を `private` にすることはテンプレートの汎用性を損なうため行わない。代わりに以下の防御で制御する:
1. **CI 静的チェック（clang-tidy AST matcher）**で production コード中の `publishWorld()` 直接呼び出しを禁止（MemberCallExpr, callee === "publishWorld"）。
2. **コードレビュー**で全 7 箇所の置換を確認。

★ v5.35: `[[deprecated]]` 属性は Coordinator 汎用テンプレートに付与しない。clang-tidy のみで制御する。

**CI 静的チェック**（**参考実装**: 恒久対策は clang-tidy AST matcher に移行予定）:
```bash
# ★ v5.27: 以下は「参考実装」として記載する。production CI には clang-tidy を導入する。
#   grep ベースのチェックはコメント・改行跨ぎ・関数ポインタを検出できないため信用できない。
#
# 恒久対策（clang-tidy AST matcher）:
#   clang-tidy --checks='-*,misc-publish-world-direct-call' \
#     --match='MemberCallExpr, callee === "publishWorld"'
#   これによりコメント内の publishWorld( を誤検出せず、実呼び出しのみを検出可能。
#   また coordinator\n    .publishWorld( のような改行跨ぎも正しく検出できる。
#   MemberCallExpr のため auto fn = &...publishWorld 関数ポインタも検出可能。
#
# ★ v5.27 追記: コード調査により production コードの publishWorld() 直接呼び出しは
#   7 箇所全て確定済み（FACT #43）。関数ポインタ経路の隠れ呼び出しも存在しないことを確認。
#   したがって P1-a 実装後のリグレッション検知には clang-tidy を導入するまで
#   コードレビュー＋手動確認で十分対応可能。
#
# 参考: grep ベースの簡易チェック（検出漏れのリスクあり）:
#   grep -rn "publishWorld(" src/ --include="*.cpp" --include="*.h" \
#     | grep -v "tests/" | grep -v "Test\|test"
```

⚠️ **CAVEAT**: 関数ポインタ経由 (`auto fn = &RuntimePublicationCoordinator::publishWorld;`) は grep では検出不可能。clang-tidy の MemberCallExpr による AST マッチングが恒久対策。P1-a 実装時点ではコードベースに関数ポインタ経路が存在しないことを確認済み（FACT #43）。

**`RegistrationContext` の使い分けについて**:
- `RegistrationContext::needsRegistration(dsp)`: commitRuntimePublication が register + publish を実行。呼び出し元は register 不要。
- `RegistrationContext::alreadyRegistered(handle)`: 呼び出し元が事前登録済み。handle を rollback 用に保持。※ handle は Constructing 状態が前提。
- `RegistrationContext::none()`: 登録不要（Bootstrap world / Hard reset）。
- ★ v5.33: 静的ファクトリにより「dsp と handle の同時指定」という不正状態を構造的に防止。
- **DIAG ビルドでは `ScopedPublishTrace` を使用してこの対応関係をログに残す**

#### Future extension: PublishCommitResult と commitRuntimePublication の分割

**PublishCommitResult**: `commitRuntimePublication()` の戻り値。`stage` のみを保持する（`committed` は `PublishStageResultTraits::isCommitted(stage)` で導出。二重管理防止のため struct には含めない）:
```cpp
struct PublishCommitResult {
    PublishStageResult stage;  // ★ commit point は PublishStageResultTraits::isCommitted(stage) で判定
    // ★ v5.27: committed は stage から導出可能なため削除。handle も DIAG 専用のため本番 API に含めず。
    //   本番 API を最小限に保ち、DIAG 詳細は ScopedPublishTrace または別機構で提供。
};

// ★ commit point 判定 — Coordinator 実装から分離して独立
//   ★ v5.26: commitRuntimePublication の committed 判定はこの関数に委譲。
//     Coordinator が SuccessWithWarning / DeferredSuccess 等を追加しても
//     PublishStageResultTraits の修正のみで対応可能。AudioEngine 側の変更不要。
struct PublishStageResultTraits {
    [[nodiscard]] static bool isCommitted(PublishStageResult stage) noexcept {
        // ★ 現行実装では Success のみが commit point。
        //   将来拡張時はここに条件を追加する。
        return stage == PublishStageResult::Success;
    }
};
```
内部実装では `PublishStageResultTraits::isCommitted(stage)` で commit 判定を行う。呼び出し側は `result.stage` を `Traits::isCommitted()` に渡す。これにより将来 `PublishStageResult` に値が追加されても `PublishStageResultTraits` の修正のみで対応可能。

**内部関数分割**: 同様に、`commitRuntimePublication()` が肥大化した場合の対策として、以下の内部ヘルパーに分割する方式を将来のリファクタリング候補として提案する:
```cpp
// 将来の分割候補（P1-a では単一関数のまま）
[[nodiscard]] DSPHandle performRegistration(DSPCore* dsp) noexcept;
[[nodiscard]] PublishStageResult performPublish(RuntimePublicationCoordinator& coordinator,
    convo::aligned_unique_ptr<RuntimePublishWorld> world) noexcept;
bool performRollback(DSPCore* dsp, const DSPHandle& handle) noexcept;
```
ただし P1-a の実装時点では `commitRuntimePublication()` は 20 行未満であり、分割の必要性は低い。DIAG/統計/タイミング計装が追加された段階で検討する。

### A.2 P1-b: advanceFade 配線

**目的**: `SnapshotCoordinator::advanceFade()` の 0 call sites 問題を修正。

**変更**: `AudioBlock.cpp` の `getNextAudioBlock()` 内に `m_coordinator.advanceFade(numSamples)` を追加。

**検証位置**: DSP 処理完了直後（Audio callback 終了直前）。既存の `m_coordinator` が使用可能な位置。

**検証**: CTest 15/15 PASS ✅

### A.3 P1-c: MEM_SNAP 監視強化

**目的**: MEM_SNAP の診断情報を拡充。

**変更内容**: MEM_SNAP フォーマット（`Timer.cpp:907-929`）の gen フィールドは既に currentGeneration を出力済み。追加作業:
- DSPCore::liveCount（DIAG ガード済み atomic）の MEM_SNAP 出力
- StereoConvolver::liveCount（DIAG ガード済み atomic）の MEM_SNAP 出力
- retiringGeneration（DSPLifetimeManager に new atomic field、retire() のみ更新可）

### A.4 P1-a-FIX: activeRuntimeDSPHandle_ 未更新修正

**問題**: `commitRuntimePublication()` 内で publish 成功時に `dspHandleRuntime_.activate(handle)` が呼ばれておらず `activeRuntimeDSPHandle_` が null のままだった。

**修正**: `commitRuntimePublication()` に activate() 呼び出しを追加。

**効果**: Handle の state が Active になり後続の lookup が正常動作。

### A.5 P1-a-FIX-2: DSPGuard 直接破棄パス

**問題**: rebuild-obsolete な DSPCore が DSPGuard 経由でも解放されない（handle 未登録のため `retireDSPHandleForRuntime()` が false）。

**修正**: `AudioEngine.RebuildDispatch.cpp` の DSPGuard デストラクタで `retireDSPHandleForRuntime()` が false を返した場合、`destroyDSPCoreNode(ptr)` を直接呼び出して解放。

**Invariant**:
- Precondition: DSPCore は未登録（Audio Thread から到達不能のため EBR epoch 保護不要）
- Safety: `retireDSPHandleForRuntime()` が false の場合のみ `destroyDSPCoreNode` を呼ぶ
- No double-free: ptr はスコープローカル

### A.6 P1-a-FIX-3: 0xC0000005 修正（DSPGuard 重複destroy）

**問題**: フォーマッタ/外部編集により DSPGuard のデストラクタ内で `destroyDSPCoreNode(ptr)` が重複して記述されていた。重複により未登録 DSPCore の二重解放（`~DSPCore()`＋`aligned_free()` を 2 回実行）が発生し、rebuild 処理中に 0xC0000005 アクセス違反を引き起こしていた。

**修正**: 重複した `destroyDSPCoreNode` 呼び出しを削除。

### A.7 D-1: DSPLifetimeManager::destroyRolledBackDSP

**目的**: rollback 後の DSPCore 物理破棄の Authority を DSPLifetimeManager に一元化。

**設計**: `destroyRolledBackDSP(DSPCore* dsp)` は内部で `destroyDSPCoreNode(dsp)` を呼ぶ専用 API。`commitRuntimePublication()` から OwnershipDisposition::CallerDestroy が返った際に使用する。

## Appendix B: 検証結果アーカイブ

### B.1 メモリ消費量解析（ConvoPeq.log）

**詳細**: [memory-consumption-analysis.md](memory-consumption-analysis.md)

**要約**:
| 測定項目 | 値 |
|:---------|:----|
| ピーク Private | 1,094 MB（gen=7 rebuild, IR load） |
| 定常 Private | 683–686 MB（160秒間フラット） |
| 観測時間 | 163.5 秒 |
| リーク傾向 | 観測範囲内では確認できず |
| EBR Retirement | **発動せず**（pend=0, reclaim=0）。retire=0 は正常ではなく「発動機会がなかった」だけ。 |
| Transaction Counters | pub/ret/reclaim=4/0/0。reclaim=0 は EBR が不要だったことを示すが、EBR 経路の検証は未完了。 |
| Heap warmup | +38MB（648→686MB、初回5秒間）。通常の VirtualAlloc lazy commit + CRT ヒープ拡張。 |

**DSPCore コスト内訳**（BUILD_PHASE からの逆算ではなくコード数式で追跡）:
- 生確保（tracked allocations）: **~159 MB**（Oversampling 64MB, SoftClip 32MB, EQ 21MB, Aligned+dry 16MB, Latency 16MB, Others 10MB）
- 未計測残差: **~191 MB**（候補列挙 — CRT断片化, VirtualAlloc granularity, MKL FFT plan等）
- 合計: **~350 MB**（BUILD_PHASE 観測値と一致）

### B.2 副次的観測（設計上の注意点 — 確定済み）

**Unnecessary allocation (obsoleted gen=1 at 768k)**: gen=2→3 遷移中、`processingRate=768000` で prepare が開始された後、gen=3 により 384k に訂正された。~100ms の無駄なCPU/メモリだが、アーキテクチャ上の二重バッファリングに起因する過渡現象であり、設計上の問題ではない。

**残存課題**: EBR caveat / lifecycle(retire)実測値 / runtimeDSPHandleMap収束値 / BlockSize実測 / 680MB内訳 → **すべて [未確定] 7** に集約済み。NoiseShaper accepted=0 は [設計] 4. P-NS で DIAG 改善対応。

### B.3 Root Cause Timeline

gen=4/5/8 の publish 失敗は以下の 2 段階メカニズム:

```
gen=4: Orchestrator buildRuntimePublishWorld(newDSP, oldDSP, HardReset, 0.0, false)
  → active=false, next=oldDSP≠nullptr
  → Builder: hasFadingRuntime=false, fadingRuntimeUuid=oldDSP_uuid≠0
  → needsCrossfade=false → override 未適用
  → Validator L92: hasFadingRuntime(false) != fadingRuntimeUuid≠0 → FAIL

gen=5/8: 同種のトポロジに crossfade rebuild のため override 適用
  → transitionActive=true, hasFadingRuntime=true（override 設定）
  → Validator 通過
  → Precheck AUTH_CONTRACT L87: graph.fadingNode(nullptr) != hasFadingRuntime(true) → FAIL
```

**根本原因**: Post-build Mutation（Builder 出力への部分上書き）が `graph.fadingNode` を更新しないため。

**修正方針**: CrossfadePlan を Builder 入力として渡す（本編 [設計] 1 参照）。

### B.4 lifecycle(retire)=0 の調査経緯

**確認済みパス**: `registerDSPHandleForRuntime` の唯一の呼び出し元は `enqueuePublicationIntentForRuntimeCommit()`（AudioEngine.Commit.cpp:685）。Coordinator direct publish（Init/PrepareToPlay/ReleaseResources/Timer/Transition）は register を経由しない。7箇所の retire 呼び出しのうち、register 経由で成功するのは DSPTransition（Orchestrator経由）のみ。

**rebuild-obsolete 代替経路不在確認**: DSPGuard 以外に rebuild-obsolete DSPCore を追跡するリスト・プールは存在しない。pendingTask.currentDSP は全7箇所で nullptr。

## Appendix C: 最終調査結果

### C.1 FACT 一覧（全78件）

全 78 の確定事実（FACT）は v6.0b までにコード調査で確定済み。主要カテゴリ:

| カテゴリ | 件数 | 代表例 |
|:---------|:-----|:--------|
| リーク原因特定 | 10+ | lifecycle(retire)=0（真因: gen=3以降のpublish全滅）, rebuild-obsolete, DSPGuard |
| コード構造確認 | 20+ | publishWorld 7箇所, RuntimeBuilder zero temp, JUCE不使用 |
| メモリ解析 | 10+ | DSPCore cost 159MB+191MB, P2-1定量化 |
| DIAG 基盤 | 5+ | MEM_SNAP gen 出力済み, lookupDSPHandleForRuntime |
| 設計検証 | 15+ | CrossfadePlan feasible, ValidationFailureReason統合, EBR機構存在 |

### C.2 残存 HYPOTHESIS — 未解決課題一覧

残る未解決課題は **すべて [未確定] 7** に集約済み。以下は参照用の要約: NoiseShaper accepted=0 真因（🔍 HYPOTHESIS）、EBR 経路未検証（⚠️ CAVEAT — **訂正: handle 未登録が原因ではない。gen=3 以降の publish 全滅による retire 機会欠如**）、lifecycle(retire)実測値（🔍 HYPOTHESIS — **同上の理由**）、runtimeDSPHandleMap収束値（⚠️ CAVEAT）、BlockSize実測（🔍 HYPOTHESIS）、680MB Other 内訳（⚠️ CAVEAT）。

**詳細**: [未確定] 7 を参照。

### C.3 結論

コード調査により確定可能な事項は **全て確定済み**（**78 FACTs**）。残る 6 項目は全て Runtime 観測または実装後の検証に依存するため、コード調査による追加確定は不可能。これらは **[未確定] 7. 未解決課題** に集約済み。

**EBR に関する重要な訂正（v6.2）**: ConvoPeq.log は P1-a/FIX 適用済みで収録されている。`lifecycle(retire)=0` は「handle 未登録」が原因ではなく、gen=3 以降の全 publish が失敗したため retire 機会そのものが発生しなかったことが真因。

**NoiseShaper accepted=0 の「processingRate bug」説は誤りと判明**（v5.60-2 コード再調査）。`adaptiveCaptureSampleRateHz` は正しく `dsp->sampleRate`（base rate=192000）から設定。真因はキューラップ動作/タイミング問題等であり Runtime 観測が必要。

**lifecycle(retire)=0 パス完全確認**: `registerDSPHandleForRuntime` の唯一の呼び出し元は `enqueuePublicationIntentForRuntimeCommit()`。Coordinator direct publish は未登録。

## Appendix D: 調査ツール

| ツール | 使用目的 |
|:-------|:---------|
| grep/sed/awk (WSL) | ログ抽出、統計計算、production コード全数調査 |
| serena MCP | コードパストレース、型情報取得、状態遷移調査 |
| cocoindex-code (ccc.exe) | 関数間依存関係の grep、シンボル特定 |
| graphify | 依存関係グラフパス検索、関数間リンク検証 |
| semble | セマンティックコード検索、フォールバック経路発見 |
| AiDex MCP | コードインデックス検索、セッションノート管理 |
| ast-grep / rg / fd / fzf | WSL ベースの高速コード検索・フィルタリング |

## Appendix E: 改訂履歴

| 版 | 日付 | 改訂内容 |
|:---|:-----|:---------|
| 1.0～5.57 | 2026-07-10～11 | 初版～最終調査完了宣言（全68 FACTs） |
| 5.58 | 2026-07-11 | メモリ消費量解析完了（FACT #69） |
| 5.59 | 2026-07-11 | メモリ解析の誤り訂正（JUCE支配説撤回等） |
| 5.60 | 2026-07-11 | 未確定事項最終調査（FACT #70-76） |
| 5.60-2 | 2026-07-11 | NoiseShaper FACT #73 誤り訂正（processingRate bug 説撤回） |
| **6.0** | **2026-07-11** | **全体再構成: 完了済み改修を Appendix に移動。将来の改修設計（CrossfadePlan / P2-1 / P4）を先頭に配置。** |
| **6.0b** | **2026-07-11** | **未確定事項の最終コード調査完了。全7ツール使用。新規FACT 2件**: (77) `currentCaptureSessionId` 未書き込み確定、(78) EBR機構完全性確認。NoiseShaper accepted=0 はコード調査では確定不能と明確化。EBR/lifecycle(retire) の既知後段ブロック要因不存在を確認。[未確定] 5→7 に再構成。 |
| **6.1** | **2026-07-11** | **調査結果を設計に反映**: [設計] 4. P-NS / [設計] 5. P-DIAG 新設。既存[設計] 4→6に繰り下げ。 |
| **6.2** | **2026-07-11** | **FACT #78 誤り訂正: EBR 未発動の原因は handle 未登録ではなく、gen=3 以降の publish 全滅による retire 機会欠如。** P1-a/FIX 適用済みのログであることを考慮すると、handle 未登録説は矛盾。正しい因果: gen=4+ 全滅 → DSPTransition 未実行 → retire 対象なし → EBR にエントリ未投入。7.2/7.3 を全面修正。 |
| **6.3** | **2026-07-11** | **レビュー指摘4点反映**: (1) BlockSize削減量に「tracked allocationsのみの理論値」と注釈追加、(2) 680MB Otherを4カテゴリに分類、(3) NoiseShaper DIAG目的を「drop判定実値のRuntime観測」に訂正、(4) runtimeDSPHandleMapにsteady-state/transition区別を追記。 |
| **6.4** | **2026-07-11** | **設計の純粋性向上（5点反映）**: (1) CrossfadePlanをbool→FadePolicy+fadingNode完全構造体に拡張、(2) Builder責務を「planの写像」に修正、(3) Validator節を「通過≠Authority Contract成立」と明確化、(4) P2-1を「初回prepareのmaxSamplesPerBlock最適化」に修正、(5) MEM_SNAPをサブシステム単位に細分化しmemoryUsage()API設計案を提示。 |
