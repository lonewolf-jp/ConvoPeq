# D153 — T3c Read-only Implementation Audit（実装前 baseline 照合）

```text
Production source changes: 0
Test source changes:       0
CMake changes:             0
Build:                     NOT RUN
CTest:                     NOT RUN

Source baseline:
  ConvoPeq.md = Generated: 2026-08-31 15:32:58（mtime 2026-08-31T06:33:05Z）
  newer source = 0（src/**.{cpp,h} で ConvoPeq.md より新しいもの 0 件・ctx_execute 実測）
  git worktree: 5 files M（AudioEngine.RebuildDispatch/Threading, ISRRuntimePublicationCoordinator.h/.cpp,
                ISRSemanticValidationTests）= セッション開始時既存の G-4.x/F6 変更。D153 自身の追加変更 0。
```

**ツール:** WSL rg/sed（rtk）・ctx_execute（mtime/git）・AiDex（RecoveryLifecycleWord=0 確認）・serena（tryInsert=h:391-412 0-based 位置確認）・ccc grep（D151 で field 棚卸し一致確認済）・MSVC STL `stl/inc/atomic` raw（本セッション取得）・MicrosoftDocs cpp-docs raw（本セッション取得）・プロジェクト一次資料 ISRDSPHandle.h/.cpp。

---

## 1. §2 .h 11 項目照合（ISRRuntimePublicationCoordinator.h）

| # | 項目 | 現行実測 | D152 想定との判定 |
|---|---|---|---|
| 1 | LogicalRecoveryObligation | h:347-362 | **MATCH** |
| 2 | ObligationState | h:306-316・8 値（NoObligation..ResolvedSuperseded） | **MATCH** |
| 3 | ObligationDeliveryState | h:325-329・3 値 | **MATCH** |
| 4 | consecutiveFailureCount | h:361 `std::atomic<std::uint8_t>` | **MATCH**（旧構造として存在＝expected baseline） |
| 5 | id | h:348 `std::atomic<LogicalRecoveryObligationId>` | **MATCH**（同上） |
| 6 | state | h:350 `std::atomic<ObligationState>` | **MATCH**（同上） |
| 7 | delivery | h:356 plain | **MATCH**（同上） |
| 8 | tryInsert | h:392-413（serena: 391-412 0-based） | **MATCH** |
| 9 | findByKey | h:382-389 | **MATCH** |
| 10 | resolve | h:423-436 | **MATCH** |
| 11 | accessor | h:441-446（slot/consecutiveFailureCount(i)）/ h:542-544 | **MATCH** |
| — | **RecoveryLifecycleWord** | **0 hit**（AiDex + rg 実測） | **NOT FOUND = expected baseline**（実装漏れではない） |

## 2. §3 T1-T7 再トレース（cpp）

| T | 現行実装（実測） | 判定 |
|---|---|---|
| T1 | 置換対象 = markTransientFailure 本体 cpp:1059-1085（scan→Live 判定→delivery 書込→fetch_add→resolve の check-then-act）+ 6 call site（§4） | **MATCH**（D152 の分割前提成立） |
| T2 | 新設対象なし（adjudicateRecoveryFailureSignals 0 hit・expected baseline）。挿入先 runCoordinatorPhase 実在 | **MATCH** |
| T3 | resolve h:423-436: id scan（h:425）→ state CAS（h:428）→ **counter.store(0) 別語（h:429）** → liveCount−1（h:430） | **MATCH**（D152 が統合対象とする「別 atomic 構造」が現存することを実証） |
| T4 | tryInsert h:392-413: id.store（h:399）→ identity（h:400）→ **state.store(Live,release)（h:401）** → delivery（h:402）→ **recoveryGeneration（h:404）← Live 公開後** → counter.store（h:407）→ liveCount++（h:408）。加えて caller cpp:956-959 が handle/epoch/intentId/buildSource を **Live 公開後**に書込 | **MATCH（現行=旧順序）** — D152 の payload-before-CAS は計画的変更。前提（payload 全 writer/publisher=CL、跨スレッド payload reader ゼロ）は成立（§3-B 下記） |
| T5 | coalesce cpp:927-940: findByKey → **1 発** CAS(Live→Live)（cpp:932）→ 失敗時 coalesceOnLive=false → tryInsert（cpp:947） | **MATCH（現行は state 単語 CAS のため 1 発設計で正しい）**。D152 R1（全文 CAS 化後は contention 再試行必須）は将来実装の要求であり、現行コードとの矛盾なし。前提（CAS 失敗時 expected 更新→再評価可能）は compare_exchange_strong 仕様で成立 |
| T6 | delivery attach cpp:979（Transport）/996（None）/1001（Durable）— plain 書込・CL | **MATCH** |
| T7 | redrive cpp:1106-1116（scan: state cpp:1110・delivery cpp:1112・id cpp:1114）+ cpp:1122-1167（付着 1159/1167 plain・CL） | **MATCH** |

### §3-B（T4 前提の精密検証）
- payload writer: tryInsert 内（identity h:400・recoveryGeneration h:404）+ caller（handle/epoch/intentId/buildSource cpp:956-959）= **全て CL**。
- payload reader: redrive cpp:1143-1157（CL）、submit cpp:975（CL）、resolve は id のみ（h:425）。Builder は durable slot の**コピー**を読む（table を読まない）。
- → 「payload writer/Live publisher とも CL」不変（停止条件 4 非発火）。現行の Live 後 payload 書込は CL 単一スレッドゆえ安全だが、D152 の順序強化は正（将来の affinity 変更への防衛）。

### §3-A（T5 最重要監査）
現行 1 発 CAS は「失敗＝terminal」だが、T3c 全文 CAS では「失敗＝pending/delivery contention」も成立 → **再試行しないと誤 NEW 化（ΔL 二重計上）**。D152 T5 擬似コード（失敗→expected 再評価→Live なら再試行、terminal のみ tryInsert）は現行トポロジー（findByKey→CAS→tryInsert の分岐構造 cpp:927-947）にそのまま適用可能であることを実測確認。**前提成立**。

## 3. §4 production 6 call sites 実測（D153-A）

`markTransientFailure(` 全 hit をコード行のみ抽出（コメント/定義/宣言除外）:

| # | file | line | 呼び出し |
|---|---|---|---|
| 1 | AudioEngine.RebuildDispatch.cpp | 1006 | `runtimePublicationBridge_.markTransientFailure(recovery->obligationId)` |
| 2 | 同 | 1033 | 同 |
| 3 | 同 | 1091 | 同 |
| 4 | 同 | 1115 | 同 |
| 5 | RuntimePublicationOrchestrator.cpp | 311 | `engine_.runtimePublicationBridge_.markTransientFailure(req.recoveryObligationId)` |
| 6 | 同 | 401 | 同 |

- **production 実測 = 6**（builder 4 + orchestrator 2）。D152 仕様 6 と **一致**。
- 非 call: 定義 cpp:1059・宣言 h:483・テスト約 30（ISRSemanticValidationTests.cpp のみ）。
- 過去資料の「5 sites（builder2+orchestrator3）」は**現行ソースで不支持**（G-4.4-P1 で builder 側 +2 済み・RebuildDispatch:1006/1033 は D136-A 追加、メモ [[d135-8-9-gate-g4-4-p1-stranding-fix]] と整合）。
- `postRecoveryFailureSignal(` = **0**（現時点で正常）。
- **D153-A = PASS（6=6）**。

## 4. §4' tryInsert call site（D153-B）

`tryInsert(` 全 hit: 定義 h:392 + **call site cpp:947 のみ**。expected=1 / **actual=1**。
**D153-B = PASS**。

## 5. §5 runCoordinatorPhase 実配置（Threading.cpp 実測）

```text
:258  void AudioEngine::runCoordinatorPhase() noexcept
:260-264  { DSPLifetimeManager ...; runtimePublicationBridge_.processIntent(*this, lifetimeMgr); }
:266-270  （redrive 注釈）
:270  runtimePublicationBridge_.redriveDeferredRecoveryObligations();
```
- 挿入点 = `:264`（processIntent 閉じ括弧）〜 `:266`（redrive 注釈）の間。D152 §8 の順序（processIntent → adjudicate → redrive）は現行構造に一意に埋め込める。
- `adjudicateRecoveryFailureSignals` 既存 call site = **0**（expected baseline）。
- CL 実スレッド経路: ISRCoordinatorLoop.cpp:8（"ConvoPeq.CoordinatorLoop"）→ :39 runCoordinatorPhase。test 以外の呼出し候補 = この 1 経路のみ（rg 実測）。
- jassert 再検討 = 指示どおり実施せず（D152 §8 で不採用確定）。

## 6. §6 telemetry 現行位置記録（修正せず記録のみ）

| 項目 | 現行位置 | D152 要求 |
|---|---|---|
| `recoveryRetryExhaustedCount_` 増加 | **cpp:1079**（`fetchAddAtomic(..., release)`）— resolve 呼び出し cpp:1080 の**前**、CAS 成否未確認 | **T2 枯渇 CAS 勝者のみ**へ移動（cpp:1079 の位置が修正対象。本監査では変更しない） |
| `recoveryConsecutiveFailureCount(i)` | h:542-544（→ table h:444-446 counter.load） | `recoveryAdjudicatedFailureCount(i)` へリネーム（テスト使用 0 実測・D151 表 D） |
| 新設 4 カウンタ（saturated/droppedStale/droppedTerminal/droppedInvalid） | 現存せず | 実装時に h:1006-1016 ブロックへ追加 |
| liveCount_ helper | `convo::fetchAddAtomic`（h:408）/ `fetchSubAtomic`（h:430）/ `consumeAtomic`（h:393/439）— AtomicAccess.h:91/60 実在確認 | 維持（CAS 勝者 gate のみ追加） |

## 7. §7 V1-V10 実装前 baseline

| V | 検査 | baseline 実測 | 実装後 target |
|---|---|---|---|
| V1 | production `markTransientFailure(` 呼び出し | **6**（§3 の表） | 0 |
| V2 | `.delivery =` plain write | **7**（h:402, cpp:979/996/1001/1071/1159/1167） | 0（desired ローカル構築のみ） |
| V3 | 旧 atomic field 宣言 | **3**（h:348 atomic id / h:350 atomic state / h:361 atomic counter）+ plain delivery h:356 | 0 |
| V4 | id/state 直接 atomic アクセス | **16**（h:384/397/399/401/425/428 + cpp:932/938/955/1064/1066/1110/1114/1128/1136/1350） | 0（W helper 経由のみ） |
| V5 | `_InterlockedCompareExchange128` in src | **0** | 1（wrapper）— **D153-D により要修正**（§8） |
| V6 | RecoveryLifecycleWord / alignas(16) W | **0**（AiDex+rg） | 定義 + static_assert 4 点 |
| V7 | adjudicateRecoveryFailureSignals call site | **0** | Threading.cpp + tests のみ |
| V8 | 6 サイト現状 | markTransientFailure 6（§3） | postRecoveryFailureSignal 6 |
| V9 | CTest | **NOT RUN — D153 read-only** | 40/40×2 + NT-1..5 |
| V10 | 旧 counter/delivery 構造 | **存在**（cpp:1071 delivery plain + cpp:1074 fetch_add） | 消滅 |

## 8. Critical findings

### D153-A — 6 sites 矛盾: **解消（PASS）**
現行ソース実測で production 6（builder 4 + orchestrator 2）。D152 と一致。過去資料の 5 は G-4.4-P1 追加前の旧時点。

### D153-B — tryInsert 唯一 call site: **PASS**（1/1）

### D153-C — 型安全/pragma 前提: **PASS（注記 2）**
- include: `<atomic>`(h:2) `<cstdint>`(h:5) `<type_traits>`(h:6) 既存。`<intrin.h>` 無し（wrapper 採用時のみ追加）。namespace `convo::isr` h:34、クラス h:90 → W 定義は h:34-90 間に配置可能。
- **注記 1**: `LogicalRecoveryObligation` は**既に** alignas(16) メンバを内包（CoalesceIdentity→DSPHandle h:298、ISRDSPHandle.h:29 `struct alignas(16) DSPHandle`）→ 構造体のアライメントは現行で 16。W 追加で alignment class は変わらない。C4324 pragma（h:971-994 は PendingRecoveryAdmission のみ）を D152 §4 のとおり新 struct 囲りに追加する推奨は妥当（前例 ISRDSPHandle.h:27-28）。
- **注記 2**: D152 §1 の static_assert 4 点（sizeof/alignof/trivially_copyable/standard_layout）は DSPHandle 前例（ISRDSPHandle.h:207-212）と同型で MSVC で検証可能。`is_always_lock_free` の static_assert は**含めない**（D152 §1 は正しく含んでいない）。

### **D153-D —（新規・ブロッカー）D152 §2.3 の MSVC 16B atomic 主張が現行コード一次資料と矛盾**
D152 §2.3 は「MSVC STL は 16B を lock pool で実装 → `std::atomic<16B>` 却下・intrinsic wrapper 必須」と記載。現行コード側の一次資料は**反対**を記録している:
1. **プロジェクト自身の前例** ISRDSPHandle.cpp:18-22:「MSVC では 16 バイト atomic の is_lock_free()/is_always_lock_free() が false を返す（**STL の保身的判定**）。**実際は InterlockedCompareExchange128 (CMPXCHG16B) で lock-free に動作**するため、MSVC ではアサートを回避する」— そして `std::atomic<DSPHandle>`（16B・alignas16）を実運用（runtime 検証パターン確立）。
2. **MSVC STL 本体ソース**（本セッション raw 取得）: `#ifdef _WIN64 / struct _Atomic_storage<_Ty&, 16> { // lock-free using 16-byte intrinsics` — 16B 専用ロックフリー特殊化が実在（`_InterlockedCompareExchange128` 使用）。compile-time の `_Is_always_lock_free<=8` は**保身的定数**にすぎず、アライメント充足時の runtime 実装は lock-free。
→ **帰結**: D152 §2.3 の棄却根拠は事実誤認。実装選択肢は (a) `std::atomic<RecoveryLifecycleWord>`（alignas(16)・DSPHandle 前例の runtime `is_lock_free()` 検証・標準 memory_order API・V5 条件は「intrinsic 直接使用 0」に反転）または (b) 明示 intrinsic wrapper（安全性同等・根拠記述のみ修正）。**いずれにせよ仕様書の誤った事実記述を修正しないと実装者・将来の監査者を誤誘導する**（停止条件 #10「D152 の仕様に現行コード上の未解決矛盾」に該当）。

## 9. 停止条件チェック（10 項目）

| # | 条件 | 結果 |
|---|---|---|
| 1 | call topology 一致 | **一致**（§2/§5 実測） |
| 2 | production site = 6 | **一致**（D153-A） |
| 3 | tryInsert call site = 1 | **一致**（D153-B） |
| 4 | payload writer / Live publisher 所有者前提 | **成立**（§2 §3-B: 全て CL） |
| 5 | T5 coalesce retry 前提 | **成立**（§3-A） |
| 6 | delivery writer = CL 以外に存在 | **現行は cpp:1071 のみ RebuildThread**（=消去対象として D152 想定どおり。想定外の第 4 の書込者なし） |
| 7 | resolve の想定外呼び出し経路 | **なし**（Orchestrator 320/354/371 + cpp:1352 = D152 想定集合と一致） |
| 8 | liveCount +1/−1 authority | **一致**（h:408/h:430、D152 表と同一構造） |
| 9 | kMaxObligationConsecutiveFailures == 4 | **確認**（h:369 実測） |
| 10 | D152 仕様の現行コード矛盾 | **発火**（D153-D） |

## 10. Verdict

```text
D153 = NO-GO（条件付き・単一ブロッカー）
  停止条件 #10 が D153-D で発火。
  それ以外（G152-01, 03..14 / V1..V8, V10 / 停止条件 1-9）は全て PASS・baseline 記録済み。

解除手順:
  D152-R1（仕様修正・read-only）:
    §2.3 の「lock pool」記述を撤回し、MSVC STL 16B 特殊化（lock-free・is_always_lock_free は保身的 false）
    と ISRDSPHandle 前例を引用。実装選択を確定:
      推奨 (a) std::atomic<RecoveryLifecycleWord> + alignas(16) + DSPHandle 前例の runtime
          is_lock_free() 検証（Debug/Release 両構成・ISRDSPHandle.cpp:12-27 パターン）
      許容 (b) 明示 _InterlockedCompareExchange128 wrapper（根拠記述のみ修正）
    §12 V5 条件を選択結果に整合（(a) なら「intrinsic 直接使用 0・std::atomic<W> 1 宣言・runtime 検証 1 箇所」）。
  → D153-R（修正節のみの軽量再監査）→ GO → T3c production implementation → build/CTest。
```

**Phase 2 実装凍結は D152-R1 + D153-R 完了まで継続。**
