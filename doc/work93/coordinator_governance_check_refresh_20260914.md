# work93 — Coordinator State-Machine Governance Check Refresh

- **日付**: 2026-09-14
- **Type**: governance check refresh（検査側のみ変更・production code 変更 0）
- **発見元**: H-02/SR-03 push 前の standard PUSH GATE 実行（isr-run-tiered-verification.ps1 -Tier standard）
- **受理判定**: ユーザー承認 (a) 採択（2026-09-14）—「検査を現行契約へ同期する」方向。
  **production code を gate に合わせて旧 regex 形へ戻すことは禁止**（本 work の第一前提）。

---

## 1. 背景と root cause（証拠連鎖）

PUSH GATE が `isr-verify-runtime-coordinator-state-machine.ps1` の 2 violations で FAIL した。
対象 2 ファイル（ISRRuntimePublicationCoordinator.h/.cpp）は `186335a5..HEAD` で
一切変更がなく（git diff --name-only 実測）、H-02/SR-03（commit 9c09582e / 7923d3e1 /
d02f07be）とは無関係の **既存の check-vs-code drift** と確定。

| 項目 | commit | 日付 |
|---|---|---|
| check script 最終更新 | c8ca439b | 2026-08-11 |
| coordinator 再設計（dash2 §1.4/§1.6.1/§1.7） | 3b43a35d (work89) | 2026-09-10 10:31 |
| 過去 gate PASS 実績 (tiered-verify-standard-final.log) | — | 2026-08-11 16:17（再設計**前**のコードで PASS） |

→ work89 クローズ時に検査追従が行われなかった残存ギャップ。D135-3 push gate（09-13）は
別構成のため顕在化しなかった。

## 2. 旧検査条項 × 现行契約（disk 一次証拠）

### violation 1: 「commit must enforce monotonic sequence（raw > regex）」
- 旧 regex: `static_cast<std::uint64_t>(sequenceId) > static_cast<std::uint64_t>(prevSeqId)`
- 现行コード（ISRRuntimePublicationCoordinator.cpp:99-107）:
  `convo::isr::isAfter(sequenceId, prevSeqId) && convo::isr::isAfter(epoch, prevEpoch)
   && mappedGeneration > prevGen`（非単調時 Faulted + return）
- 根拠コメント: dash2 §1.6.1（Phase H）— wraparound-safe modular comparison への置換
  （SequenceArithmetic.h、isAfter(a,b)==(a>b) at 非 wrap 値、seq/epoch は +1 増加のため
  semantics-preserving）。
- **更新後条項**: isAfter(seq)+isAfter(epoch)+gen raw 比較の 3 条件必須（epoch 単調性の
  明示検証は追加強化）。

### violation 2: 「retire must update backlog through setRetireBacklogCount(backlog+1)」
- 现行コード（dash2 §1.4 B0-3・意図的设计、コメント cpp:136-146 明記）:
  - `retire()` は retireBacklogCount_ を**直接増やさない**（元 setter 上書き設計の commit 毎
    無制限増加 → Layer 2 isFullyDrained シャットダウン失敗回帰を防止）
  - accounting の本体 = semantic event 対:
    `onRetireAccepted()`（fetchAddAtomic(retireBacklogCount_…) + noteRetireBacklogChanged）/
    `onRetireConsumed()`（old>0 underflow guard + fetchSubAtomic、違反時 Faulted）
  - `setRetireBacklogCount` = **TEST-ONLY**（cpp:263 コメント「production からの絶対値上書きは禁止」）
  - 実測 drain 判定の authority = Layer 1（AudioEngine::isFullyDrained が
    pendingRetireCount + pendingIntentCount を直参照。cpp:526 で Layer 2 従来形も併存確認）
- **更新後条項**（4 検証）: (i) retire() body に直接 fetchAdd(retireBacklogCount_) が**ない**こと
  （負検証）、(ii) onRetireAccepted の accounting 実在、(iii) onRetireConsumed の underflow
  guard 実在、(iv) setRetireBacklogCount の TEST-ONLY マーカー。

## 3. 変更範囲

```text
変更:  .github/scripts/isr-verify-runtime-coordinator-state-machine.ps1 （条項 2 → 同期形へ）
       .github/scripts/isr-verify-v4-dsp-handle-policy.ps1             （§6 の同型 drift を同期）
       .github/scripts/isr-verify-c1-c15-minimal.ps1                   （§7 の同型 drift を同期）
       .github/scripts/isr-verify-v7-rt-nonrt-retire-bridge.ps1        （§8 の同型 drift を同期）
       .github/scripts/isr-verify-memory-ordering-contract.ps1         （§9 の同型 drift を同期）
       .github/scripts/isr-verify-publication-single-path.ps1          （§10 の同型 drift を同期）
       doc/work93/coordinator_governance_check_refresh_20260914.md      （本書）
禁止(遵守): ISRRuntimePublicationCoordinator.* / AudioEngine* / ReleaseResources* の動作変更なし /
            H-02・SR-03 commit への混入なし / evidence/ の新規変更の commit なし（ゲート出力は作業ツリー残置） /
            既存事前ステージ分（.gitignore / build.bat / doc work68）の巻込みなし
```

## 6. gate 再実行中に出た同型 drift（v4-dsp-handle-policy・追記）

standard PUSH GATE 再実行は coordinator check PASS まで進み、次段
`isr-verify-v4-dsp-handle-policy.ps1` の「Shutdown reclaim path must route through
runtimePublicationBridge_.reclaim」で FAIL。調査結果は本 work と同一クラス:

- 要求 regex は ReleaseResources.cpp 直下に `runtimePublicationBridge_.reclaim(` を期待（08-11 形）。
- 现行コード（work88 dash2 §2.2 Step 12-14）: ReleaseResources.cpp:507/514 →
  `tryShutdownQuiescentReclaim(handle)`（AudioEngine.h:4541・Proof→Permit→reclaim 一括 helper・
  AC-2 により caller-side shutdown 判断撤去）→ AudioEngine.h:4547
  `runtimePublicationBridge_.reclaimShutdownQuiescent(..., std::move(*permit))`
  → Coordinator `reclaimShutdownQuiescent`（h:844・ReclaimPermit consume・single-use・
  cross-runtime/stale reject）。
- **意味（Coordinator Reclaim Authority 一本・bypass 禁止）は保持、API 名と配置進化的強化**
  （Permit 認可が追加され旧形より強い）。旧 regex は形式だけ失効。

同期（検査側のみ・強度以上）: (i) ReleaseResources に `tryShutdownQuiescentReclaim(` の
単一入口を要求、(ii) AudioEngine.h に `runtimePublicationBridge_\.reclaimShutdownQuiescent(` を
要求。production code 変更 0。

## 7. 同型 drift その3: isr-verify-c1-c15-minimal.ps1 C4（追記）

v4 通過後、`isr-verify-c1-c15-minimal.ps1` C4 が failCount=1（evidence: `bridgeCommit=0`）。

- 旧要求: src/ 全域に `runtimePublicationBridge_.commit(` ≥1（08-11 時点の bridge commit 形）。
- 现行トポロジ（#5/#7 Sprint-2「Bridge responsibility: validate / didPublish / willRetire only」・
  #21 validator 委譲）: commit は Bridge の責務から外れ、単一 authority 経路
  **RuntimeWorldAuthority.h:299 `coordinator_.commit(PublishAuthority::Granted, ...)`** に収束。
  coordinator.cpp:65 の自己呼出は 4-arg overload → 完全 overload 委派（定義内部）で追加 call site ではない。
- 同期（検査側のみ・強度以上）: `$c4BridgeCommit` 要求を
  `coordinator_\.commit\(PublishAuthority::Granted` ≥1 に置換（authority token 束縛を明示検証）。
  legacy 3 パターン=0 / forbidden=0 / bridgeRetire≥1 は不変維持。
- targeted 結果: **c1-c15 15/15 PASS（fail=0 manual=0）**。

## 8. 同型 drift その4: isr-verify-v7-rt-nonrt-retire-bridge.ps1 R9（追記）

- 旧要求: AudioEngine.Commit.cpp に `worldAuthority_.lifetime().emitRetireIntentRT(` ≥1
  （08-11 時点形・RT callback 検出からの intent 発行）。
- 现行コード（D132 M2 / Step 12）: RT 側は identity 非携帯の SPSC 純シグナル
  `crossfadeRuntime_.notifyRampComplete()`（AudioBlock.cpp:466）で edge 通知のみを行い、
  intent 発行本体は NonRT bridge 経路 `lifetime().emitRetireIntentNonRT(intent)`
  （Commit.cpp:485）へ収束。**RT 上での intent 発行自体が撤去された（設計上より強い**
  方向の意味保持: RT は signal のみ / 発行は NonRT authority）。
- 同期（検査側のみ・両端束縛で強度以上）:
  (i) AudioBlock.cpp に `notifyRampComplete(` の検出シグナル実在を要求、
  (ii) Commit.cpp に `emitRetireIntentNonRT(` を要求。
- targeted 結果: **v7 PASS**（R9 RT-detect to NonRT-retire bridge policy verified）。

## 9. 同型 drift その5: isr-verify-memory-ordering-contract.ps1（追記）

- 旧要求（commit() の単調性条項・FUTURE-4 形）:
  (a) `const auto prevWorld = static_cast<const RuntimeState*>(`（currentWorld_ read）、
  (b) `static_cast<std::uint64_t>(sequenceId) > static_cast<...>(prevSeqId)`、
  (c) 同 epoch 形 → violations 3 件。
- 现行コード: baseline は明示 `prevWorld` 引数（dash2 §1.7 CW-3b — currentWorld_ 参照自体の撤去）、
  seq/epoch 比較は wraparound-safe `convo::isr::isAfter(...)`（§1.6.1 Phase H）。
  bake 条項（pubWorld->publication = PublicationSemantic{）と mappedGeneration raw 比較は现行に存在。
- 同期（検査側のみ・強度以上）: (a)→`prevWorld ? prevWorld->publication.sequenceId` baseline 要求、
  (b)(c)→`convo::isr::isAfter(sequenceId, prevSeqId)` / `convo::isr::isAfter(epoch, prevEpoch)` 要求。
  意味（strict monotonic・fail-closed）は保持、coordinator refresh と同一形へ揃う。
- targeted 結果: **memory ordering contract PASS**。

## 10. 同型 drift その6: isr-verify-publication-single-path.ps1（追記）

- violations 2 件: (a) 7 引数 semantic commit regex → 现行は CW-3b の `const RuntimeState* prevWorld`
  追加 8 引数形、(b) publish-to-commit 呼出数を AudioEngine.Commit.cpp 内 bridge/legacy 形で
  expected=1 を要求 → 现行は RuntimeWorldAuthority.publishAndSwap の
  `coordinator_.commit(PublishAuthority::Granted, ...)` 1 箇所に収束（bridge/legacy=0）。
- 同期（検査側のみ・意味同一以上）: (a) 8 引数形（prevWorld 任意マッチではなく明示 allow）へ、
  (b) `authorityCommitCount == 1 @ RuntimeWorldAuthority.h` + `legacy==0 and bridge==0` を
  分離検証（従来より強い単一経路強制）。報告に authorityCommitCount を追加。
- targeted 結果: **publication single-path PASS**（legacy=0 bridge=0 authority=1）。

本 work での同期合計: coordinator state machine 2 条項 / v4 1 条項 / c1-c15 C4 1 条項 /
v7 R9 1 条項 / memory-ordering 3 条項 / publication-single-path 2 条項 —
すべて「検査→実装」方向で、production code 変更 0。

## 4. 検証結果

| ゲート | 結果 |
|---|---|
| targeted: isr-verify-runtime-coordinator-state-machine.ps1 単体 | **PASS**（violations=0・CHECK_EXIT=0・report ready=true） |
| negative-control: 旧 regex 条項は削除済みで code を触っていない（差分 ps1 のみ） | 成立（production diff 0） |
| standard PUSH GATE（再実行） | GATE_RESULT_PLACEHOLDER |
| push | PUSH_RESULT_PLACEHOLDER |

## 5. 残余リスクと注記

- 本 refresh は**検査の意味（回帰検出力）を維持した同期**である: 旧形の raw > 比較へ
  逆戻りすると check が赤になる（§1.6.1 の wraparound-safe 性が契約として強制される）。
  retire 側も「直接 +1 復活」「TEST-ONLY マーカー消失」「underflow guard 撤去」の各退行を
  検出する。
- gate スクリプトの prune 副作用（evidence/ の削除・再書込）は実行後に D 分を HEAD へ
  復元する手順を要する（2026-09-14 時点で実施済）。
- coordinator 以外の check に同型 drift が潜在するかは本 work のスコープ外。次回 gate で
  網羅的に再検証される（FAIL 時は同方針: 検査→実装の同期方向で起票）。
