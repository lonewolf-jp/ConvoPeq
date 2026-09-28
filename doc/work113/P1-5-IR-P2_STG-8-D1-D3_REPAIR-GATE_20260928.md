# STG-8-D1/D2/D3 Repair Gate（2026-09-28）

Discovery 記録（`P1-5-IR-P2_STG-8_DISCOVERY_20260928.md`）は変更せず保持する。
本書は Contract Audit → 実装 → regression → authority の Repair Gate 記録である。

## 0. 開始時 authority（Contract Audit 開始時・再生成不要を確認）
- SHA-256: `955582FBF3ABBEE532E67EF908DF083D807F34DE550D85EA89A76FA93DB0732C`
- size: 5,589,694 B / Generated: 2026-09-28 14:33:45 / HEAD: `29dbdca9`
- `--check`: FRESH / NEWER_SRC_COUNT: 0

## 1. Contract Audit（production 変更なし・全項目 PASS）

### D1（Deferred Discard / RedriveBudgetExhausted）
1. obligationId 寿命: `slotRequestSnapshot` は discard 前の値コピー（`:877`）。
   resolve は id-based lookup のため slot 消滅後も有効。
2. 順序 DSP-retire→slot 消滅→resolve: retire（handle map）と resolve（table word）は
   独立資源・双方冪等のため順序不問。候補順序は安全。
3. 既存 caller: Route A/B（Published `:324/:358`）、Route C（StaleSuperseded `:398`、
   ShutdownDiscarded `:443`、Retry no-op `:429`）、Route D（Failed `:1137`）、
   shutdown discard-all（`:1470`）。新規は StaleSuperseded/ShutdownDiscarded のみで意味不変。
4. 意味: StaleDiscard 系→StaleSuperseded、ShutdownDiscard→ShutdownDiscarded。
   Ready-path `:398` 前例と対称。
5. Retry 非終端: 候補は Retry を渡さない（`:1043-1044` 到達不能）。
6. 冪等性: `RecoveryAdmissionTable::resolve`（`ISRRuntimePublicationCoordinator.h:513-536`）
   は full-word CAS Live→terminal、winner のみ −1、非 Live/CAS 敗北/id 不一致→false。
7. winner semantics: 最初の CAS 勝者のみ true（D152 T3）。id 単調（`nextId_` single-writer）
   のため ABA なし。
8. pending/delivery/lifecycle: resolve は pending/adjudicated を zero 化し delivery を保持。
   `isFullyDrained` の L==0 要件（`Coordinator.cpp:561`）と整合。
9. Boundary 内: 既存 public API（`resolveRecoveryObligation`）のみ使用。

### D2（Deferred Overwrite）
1. 旧 O は置換前（`:551` 前）に `deferredSlot_->request` から取得可能（同一スレッド）。
2. 旧 retire→旧 resolve は独立のため順序不問。
3. 同一/別 obligation の分離: ガードは **oblId のみ** で判定。
   同一 O の新 generation 再 defer（coalesce 再送）は同一 Live obligation の新表現であり
   終端化しない（gen 比較を含めると誤終端になるため含めない）。
4/5. 別 obligation の場合のみ旧 O を `StaleSuperseded` で終端化。
6. 新 O の delivery/lifecycle 不変。7. `deferredOverwriteCount` 不変。
8. helper 化: 3 処点の outcome mapping が各々異なる（D1-Discard は reason 対応、
   D1-Budget/D2 は StaleSuperseded＋D2 は distinct-check）ため inline 実装とし、
   helper による責務混在を避けた。空 slot 時は `has_value` guard で保護。

### D1/D2 共通禁止事項の遵守
Coordinator authority 拡張・lifecycle state 追加・ID 体系変更・resolve 意味変更・
Retry 意味変更・delivery model 変更・P3 再実装・RT 変更・atomic 追加・
ownership/retire authority 追加・shutdown 別 authority 追加 — いずれもなし。
Boundary は `RuntimePublicationOrchestrator.cpp` のみ。

### D3（RejectedPressure・独立契約）
1. Retry semantics（MUST-2、ΔL=0、Live 維持）。2. `resolveIfRecovery(Retry)` no-op（`:1043-1044`）。
3. rearm は durable-Building＋id 一致が条件（`:1185-1186`）のため transport では不発。
4. `postRecoveryFailureSignal` は bounded CAS increment（4 で飽和・inert）、
   id-scoped・terminal/reuse-aware のため全 durable 状態で安全。
   durable-Building 併存時も `tryAttachDurableRecovery` の AlreadyRepresented 短絡
  （`:1373-1376`、D144/D146）により二重表現なし。
5. failure budget 消費の正当性: `RejectedNotFinalized`（`:414-415`）との完全対称
   ＋ D105-R18 transient-failure pattern（budget 消費・exhaustion→Failed が唯一の
   Failed 経路）。持続圧力（4 連続）後の `ResolvedFailed` 終端は silent strand＋
   capacity leak より誠実であり、定数・マッピングの変更なし（意味変更ではなく既存
   category への合流）。
6. delivery 直接変更不要（adjudicate が None 化 `:1141-1145`）。delivery model 不変。
7. redrive が同 id＋buildSource で再 attach（ΔL=0）し Builder を wake（`:1252-1265`）。
   圧力解除後は Accepted→Published→resolve まで決定的に進行。
8. liveness: 圧力解除後の次 coordinator tick で adjudicate＋redrive が同期実行される。
   テストは health ではなく throttle flag で pressure を強制・解除する。
9. `kMaxObligationConsecutiveFailures=4` の意味不変。
10. `NotFinalized` との対称性成立（共に build 済み DSP の admission 拒否＋DSP retire）。
    現状の非対称（signal 有無）こそが defect。

### Regression oracle（確定）
- D1: deferred O1→bump（submit なし）→Discard→`!hasDeferred && L==0`。
- D2: O1 defer→別 O2 上書き→`L==1（O2 のみ）&& hasDeferred` 安定。
  同一 O 再送→`L==1 && hasDeferred` 維持（D2b guard 証明）。
- D3: throttle ON→submit→redriveCount 増加→throttle OFF→`L==0`＋sequence 前進。

## 2. 実装（Repair GO 範囲内・commit なし）
production diff は `src/audioengine/RuntimePublicationOrchestrator.cpp` のみ（+41、4 hunks）:
- `:429-437` D3: Pressure 分岐に `postRecoveryFailureSignal(O)` 追加（rearm 維持）。
- `:514-528` D2: overwrite 時に旧 O（新 O と異なる場合のみ）`StaleSuperseded` 終端化。
- `:915-927` D1-Budget: RedriveBudgetExhausted で slot O を `StaleSuperseded` 終端化。
- `:963-980` D1-Discard: Discard で slot O を終端化
 （`ShutdownDiscard→ShutdownDiscarded`、他→`StaleSuperseded`）。
- 新規 raw atomic 0。RT/Coordinator/lifecycle/delivery 変更なし。header 変更なし。

test diff（CTest 新規登録なし・harness 内 subtest）:
- 新規 `src/tests/AudioEngineHarness/STG8RecoveryObligationTests.cpp`
 （D1/D2/D2b/D3 の4 subtest＋`runSTG8RecoveryObligationTests()`）。
- `DeferredPublicationTestAccess.h`（+22: coordinator/throttle/bump の test-only 到達。
  production 意味変更なし・friend 枠内）。
- `PublishPipelineIntegrationTests.cpp`（+7: 宣言＋呼出）。
- `CMakeLists.txt`（+1: TU 登録）。

## 3. Regression（§7）
- Debug harness: STG-8-D1/D2/D2b/D3 全 PASS、`FAIL:` 0 件。
  （D2 は初回 setup 不備〈同一 DSP 二重登録は同一 handle〉を synthetic handle
  `DSPHandle{91,1}` で修正。RECOVERY-6 は null のみ拒否することを実読確認済み。）
- Release harness: STG-8 全 PASS＋STG-1/2/4-1/6-D1/7-D1 全 PASS＋
  `AudioEngineHarness: all publish pipeline tests PASS`。
- CTest: **42/42 PASS**（Total 150.52s、`AudioEngineHarness` 128.28s 含む）。
- 重点項目: recovery capacity（L 会計 全 oracle）/ overwrite（D2＋overwriteCount）/
  coalesce（D2b＋既存 C16 含む CTest）/ redrive（D3＋既存 R18/T-R20-2 含む CTest）/
  shutdown drain（既存 ShutdownRetireIntentDrain 等 含む CTest）。
- 環境事象: PC フリーズ後の Release 再ビルドで C1033（vc140.pdb 破損）。
  該当 2 PDB を削除→再生成し解決（build 成果物の回復・source 無関係）。
  Debug ビルド・全テスト結果に影響なし。

## 4. Post-repair authority（再生成・流用なし）
- SHA-256: `56751E2E5DAE9E02DC8EB4D72968044BC57B0CD501F64D83A50F29174A5F2E5A`
- size: 5,612,326 B / Generated: 2026-09-28 15:58:53 / HEAD: `29dbdca9`（不変）
- `--check`: FRESH / NEWER_SRC_COUNT: 0

## 5. ISR / authority audit
- RT path / RuntimeWorld / RuntimeStore / Publication / Crossfade / Retire-Epoch /
  Coordinator の変更なし。新規 raw atomic 0。
- Practical Stable ISR 原則（RT no wait/lock/allocation/delete/decision、
  Coordinator sole authority、Retire through Epoch、RuntimeWorld immutable after publish、
  Overflow != silent loss、Shutdown = complete drain）に抵触なし。
  終端化は既存単一 Authority 経由のみ。

## 6. Worktree / commit policy
- commit 未実施（commit approval 待ち）。push NOT AUTHORIZED 継続。
- `AGENTS.md` / `headroom-proxy-start.ps1` の既存無関係差分は非接触・非混入。
- 本書は新規作成（untracked）。DISCOVERY 記録は不変。
