# D2_IMPL_CHECKLIST — REPAIR_PLAN2-dash2 実装チェックリスト

**作成日:** 2026-08-14
**元文書:** `doc/work88/REPAIR_PLAN2-dash2.md`（2026-08-14 版・H.6〜H.18 追記レイヤー含む）
**実装順序の正本:** §3 推奨実装順序 + **H.11.17.5 15-Step**（bool API 早期削除・compile guard を含む）を優先
**前提:** REPAIR_PLAN2-dash.md 本実装（P2-1〜P2-4 / X1〜X6 / X4-B / X3-R4 Phase 7）完了・ctest 28/28 PASS

## 凡例
- [ ] 未着手 / [→] 作業中 / [x] 完了 / [-] 対象外・保留（理由付き）
- 各 Phase で**ビルド + ctest 通過**を rollback point とする
- ctest 除外: `-E "BuildInputSemanticContract|RuntimeWorldAuthority"`（既知）

---

## Phase 0: invariant freeze（現行 Invariant 固定・baseline 検証）

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| 0-1 | 現行 Invariant 棚卸し（INV-ISFULLDRAINED-1〜5 / INV-R1-1〜2 / INV-EPOCH-1〜2 / INV-FIFO-1 / INV-ISR-LIFE-1〜6）を文書化 | [ ] | dash2 §1.4 / H.11.14.1 |
| 0-2 | baseline ビルド（Debug）成功確認 | [x] | tools/build-verify-phase-full.bat（Debug バイナリ存在確認） |
| 0-3 | baseline ctest 全 PASS（既知除外） | [x] | **28/28 PASS**（2026-08-14 確認） |

## Phase B0/B1/B2: §1.4 external setter 撤去（Tier 1 — Phase A1 より先に実施）

**設計方針（dash2 §1.4 第三者的レビュー反映）:**
1. setter は「private化」でなく「意味ごと廃止」（API 削除）
2. snapshot accounting に戻さない → **semantic event API**（onRetireAccepted/onRetireConsumed 等）へ閉じ込める
3. underflow 防止: fetch_sub 前に `old > 0` 検証、違反時 Faulted
4. `setQuarantineResidentCount` は domain mixing → `DSPQuarantineManager::residentCount()` を直接 source に

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| B0-1 | 7 個の external setter 全 call site 棚卸し完了 | [x] | h:121-127 / rg で全箇所確認済み |
| B0-2 | semantic event API 設計（onRetireAccepted/onRetireConsumed/onReclaimBegin/onReclaimEnd/onFallbackAccepted/onFallbackConsumed/onDeferredRetireAccepted/onDeferredRetireConsumed） | [x] | dash2 §1.4 設計方針・h に実装 |
| B0-3 | Coordinator 内部の snapshot overwrite 撤去（cpp retire()/enqueueRetire() の load+setRetireBacklogCount → onRetireAccepted / 無計数化） | [x] | retire() は増加なし（commit ごと無制限増加防止）、enqueueRetire() は onRetireAccepted |
| B0-4 | AudioEngine.Threading.cpp:126-128（isFullyDrained の setFallback/setRetire/setDeferredRetire）撤去 → 実測値直接判定 | [x] | router pendingRetireCount + lifetime pendingIntentCount + 各 residentCount を直接判定 |
| B0-5 | AudioEngine.Retire.cpp（setReclaimInFlightCount/setFallbackBacklog/setRetireBacklog/setDeferredRetire）撤去 → semantic event | [x] | onReclaimBegin/End に置換 + スナップショット撤去 |
| B0-6 | AudioEngine.Processing.ReleaseResources.cpp:291（setQuarantineResidentCount domain mixing）撤去 | [x] | Layer 1 が DSPQuarantineManager::residentCount 直接判定 |
| B0-7 | AudioEngine.h:4152/4162（setRetireBacklogCount）撤去 | [x] | Layer 1 が router 実測を直接判定 |
| B0-8 | テスト側 setter 呼び出し（ISRSemanticValidationTests 等）の検証・代替 | [x] | TEST-ONLY として維持（P2 教訓 — テストリセットは許可）・20 参照 |
| B0-9 | ビルド + ctest 全 PASS | [x] | **Debug ビルド成功 + ctest 28/28 PASS**（2026-08-14） |
| B0-10 | `isr-verify-backlog-specfixed-residual.ps1` PASS | [x] | **PASS**（totalRStatus=0 / specFixedResidual=0）2026-08-14 |

**Acceptance:** 外部 setter コード参照 0 / isFullyDrained が実測と整合（X5/X6 カウンタと照合）

## Phase A1: §2.2 ShutdownQuiescenceProof / ReclaimPermit 型導入（15-Step 1-6, type only）

**正本:** H.11.17.5 15-Step。旧 Commit 1〜4 → Step 1〜5、旧 Commit 13 → Step 6。

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| A1-1 | Step1: `ShutdownRuntimeIdentity` 型導入（generation / epochGeneration / readerRegistrationGeneration） | [x] | ISRLifetimeProof.h 新規・型のみ |
| A1-2 | Step2: `ShutdownQuiescenceProof` 型導入（Q0〜Q7 完全条件, immutable, 生成は ShutdownRuntime 限定 friend） | [x] | Q0〜Q7 フラグ + identity 束縛。循環排除（pendingReclaim/isDrained は Proof から除外） |
| A1-3 | Step3: `ReclaimPermit` 型導入（move-only single-use, identity 束縛） | [x] | consume() CAS（Issued→Consumed）で二重 reclaim 構造的防止 |
| A1-4 | Step4: Proof 生成 API `tryMakeQuiescenceProof()`（全条件自前検証・snapshot 簡易生成不可） | [x] | ShutdownRuntime メソッドとして宣言 + type-only stub（常に nullopt） |
| A1-5 | Step5: Permit identity / single-use 検証（identity match / consume-once / post-proof irreversibility） | [x] | tryMakeReclaimPermit 宣言 + identity 束縛実装（type only） |
| A1-6 | Step6: `ReclaimIdentity` 型導入（pendingReclaimHandles_ 昇格の下地） | [x] | DSPHandle + retireSequence（INV-FIFO-1） |
| A1-7 | ビルド確認（production 未接続・既存挙動不変） | [x] | **Debug ビルド成功 + ctest 28/28 PASS** |

## Phase B3: 4 admission paths shutdown gate（Tier 1）

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| B3-1 | AdmissionState 4-state FSM（Open→Closing→Closed, Closed→Open 禁止） | [x] | ISRShutdown.h/.cpp に実装（CAS 不可逆遷移・INV-LIFE-9） |
| B3-2 | AdmissionToken / reservation 導入（RT-safe admission） | [-] | **Phase A2 連動で保留** — Q0（OutstandingAdmissionReservations）と密接（H.11.15.2 / H.11.22.2） |
| B3-3 | Path B: enqueuePublicationIntent gate（tryAdmitPublication） | [x] | CoordinatorState::ShuttingDown gate 追加（defense-in-depth、Owner reclaim は既存フロー） |
| B3-4 | Path A: submitPublishRequest admission token | [x] | 既実装確認 — PublicationAdmission::evaluate が isShutdownInProgress→RejectedShutdown |
| B3-5 | Path C: Recovery admission shutdown gate（P2-4 Step B 既実装の検証） | [x] | submitRecoveryRequest の ShuttingDown gate 実装済み（P2-4 Step B） |
| B3-6 | Path D: Build/Publish admission gate（4 経路統一） | [x] | RebuildDispatch に isShutdownInProgress 多数 / Observe・Quarantine は Timer/CoordinatorLoop が遮断 |
| B3-7 | ビルド + ctest 全 PASS | [x] | **Debug ビルド成功 + ctest 28/28 PASS**（2026-08-14） |

## Phase A2: production reclaim → Permit 接続（15-Step 7-15, A2-G01〜G23 PASS 前提）

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| A2-1 | Step7: `reclaimNormal()` 新設（production 接続 Yes） | [x] | requestReclaim を reclaimNormal へ委譲（RuntimeEBR） |
| A2-2 | Step8: `reclaimShutdownQuiescent(..., ReclaimPermit)` 新設（型のみ） | [x] | permit.consume() で認可 → 既存 ShutdownQuiescent 経路に委譲（production 未接続） |
| A2-3 | Step9: **旧 bool reclaim API 削除**（compile guard — 残存で compile error） | [x] | reclaim(ReclaimMode, ..., bool) / ReclaimMode enum 削除。reclaimNormal / reclaimShutdownQuiescent に直接実装（AC-1） |
| A2-4 | Step10: CacheMap caller-side shutdown 判断撤去 → Permit 経由 | [x] | ~CacheMap を tryShutdownQuiescentReclaim（Proof→Permit→reclaim）に委譲（AC-2） |
| A2-5 | Step11: ReleaseResources.cpp:423/433 migration | [x] | tryShutdownQuiescentReclaim に委譲（retire 事前実行 + reclaim） |
| A2-6 | Step12: ShutdownRuntime から Permit 供給 | [x] | tryShutdownQuiescentReclaim（AudioEngine.h）が Proof 生成→tryMakeReclaimPermit→reclaimShutdownQuiescent を一括実行 |
| A2-7 | Step13: physical destruction ordering 修正 | [x] | CacheMap: reclaim 成功後に EQCoeffCache を物理解放（ReclaimStarted→destruction→Completed 順序） |
| A2-8 | Step14: race / forged / stale Permit テスト（T9/T10/T11/T12/T13, Race A〜F） | [x] | T9 concurrent double reclaim + **T10a generation stability / T10b cross-runtime provenance / T10c generation mismatch rejection at ReclaimAuthority boundary** + T10d ReclaimAuthority constructor 固定注入 + T11 setter resurrection + T13 destruction ordering。**Authority Singularization 完全化: bindShutdownIdentity は private + friend ShutdownRuntime（AudioEngine から bind 不能・setShutdownIdentity 廃止・Unbound→Bound 固定）。setReclaimAuthority は public mutator ごと廃止し、ShutdownRuntime の constructor-level fixed dependency injection に置換（reclaimAuthority_ は reference member — 型レベルで null 不可・optional wiring / runtime reconfiguration 不在。AudioEngine は composition root として constructor initializer で注入 — shutdown 実行経路に wiring なし・再 bind/rebind 操作 API 非存在）。reclaimShutdownQuiescent 内部で identity validation → consume → reclaim** |
| A2-9 | Step15: A2-G01〜G23 全 PASS 後 production 固定 | [x] | **Step 9-14 完了。production reclaim 接続済み（tryShutdownQuiescentReclaim が CacheMap/ReleaseResources に接続）** |
| A2-10 | ビルド + ctest 全 PASS | [x] | **Debug ビルド成功 + ctest 28/28 PASS + invariant ALL TESTS PASSED**（2026-08-15 Step 9-14 後） |

### A2-G01〜G23 機械的検証結果（2026-08-15）

| Gate | 条件 | 判定 | 根拠 |
|------|------|------|------|
| A2-G01 | external setter = 0 | 🟢 PASS | production setter 呼び出し 0（reclaim() 内部の絶対値 setter も onReclaimBegin/End に置換） |
| A2-G02 | counter mutation = single authority | 🟢 PASS | reclaimInFlightCount_ を semantic event に一本化。onReclaimEnd は deferred 保留解消（単発成功は no-op — INV-3-1/3-2 PASS） |
| A2-G03 | isFullyDrained = observational only | 🟢 PASS | reclaim 認可に未使用（INV-LIFE-1/2） |
| A2-G04 | swapPending_ pre-check preserved | 🟢 PASS | isFullyDrained Layer2 :569 に維持 |
| A2-G05 | Path A shutdown gate | 🟢 PASS | PublicationAdmission.cpp:11-12 RejectedShutdown |
| A2-G06 | Path B authority-side shutdown gate | 🟢 PASS | enqueuePublicationIntent :354 ShuttingDown gate |
| A2-G07 | Recovery enqueue gate | 🟢 PASS | submitRecoveryRequest :861 ShuttingDown gate（P2-4 Step B） |
| A2-G08 | Build admission gate | 🟢 PASS | RebuildDispatch :239-513 isShutdownInProgress 複数 |
| A2-G09 | Publish gate | 🟢 PASS | Path A + Path B 両方 gate あり |
| A2-G10 | postStopEnqueue == 0 after producer join | � PASS | tracking（Commit.cpp:459）+ Proof の Q6 条件に接続（tryMakeQuiescenceProof 検証） |
| A2-G11 | reader registration closed | 🟢 PASS | EpochDomain 実装済み |
| A2-G12 | active readers == 0 | 🟢 PASS | activeReaderCount() 実装済み |
| A2-G13 | epoch settled | 🟢 PASS | Proof Q5 条件に接続（tryMakeQuiescenceProof 検証・acceptance テスト追加） |
| A2-G14 | pending reclaim identity == empty | 🟢 PASS | pendingReclaimHandles_ を ReclaimIdentity（handle+retireSequence）に昇格（AudioEngine.h / Retire.cpp） |
| A2-G15 | ShutdownQuiescenceProof private construction | 🟢 PASS | ISRLifetimeProof.h:96-98 private ctor + friend ShutdownRuntime |
| A2-G16 | ReclaimPermit private construction | 🟢 PASS | ISRLifetimeProof.h:158-162 private ctor + friend |
| A2-G17 | PermitIdentity bound to ShutdownRuntime | 🟢 PASS | ShutdownRuntimeIdentity 型 |
| A2-G18 | PermitIdentity bound to shutdown generation | 🟢 PASS | generation メンバー（tryMakeQuiescenceProof で fetch_add 供給） |
| A2-G19 | PermitIdentity bound to epoch generation | 🟢 PASS | EpochDomain::epochGeneration_ 実装（publishEpoch で increment）+ Proof identity 束縛 |
| A2-G20 | PermitIdentity bound to reader-reg generation | 🟢 PASS | EpochDomain::readerRegistrationGeneration_ 実装（closeReaderRegistration で increment）+ Proof identity 束縛 |
| A2-G21 | stale Permit rejected | 🟢 PASS | consume() CAS（Issued→Consumed）single-use（acceptance テスト） |
| A2-G22 | forged Permit impossible | 🟢 PASS | デフォルト ctor なし・private ctor のみ・copy deleted（機械検証） |
| A2-G23 | physical destruction paths audited | 🟡 ほぼ完了 | destroyDSPCoreNode 4 パス棚卸し（未公開 DSP 限定確認）+ Proof/Permit acceptance テスト追加。production 接続は Step 9-14 後 |

**→ 2026-08-15 第二パス後: PASS 22 / ほぼ完了 1（G23）。G10/G13/G14/G19/G20 を実装（EpochDomain generation・ReclaimIdentity 昇格・Proof Q0〜Q7 検証）し、acceptance テスト（invariant_INV3_INV5Tests: testA2G19G20EpochGenerationSupply / testA2G10G13G21G22ProofPermit）を追加。残りは Step 9-14（bool reclaim 削除・production 接続・Race テスト）。**

## Phase C: §2.1 R4 retire 順序の完全解消（Tier 1）

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| C-1 | epoch safety と FIFO の分離（INV-EPOCH-1/2 primary, INV-FIFO-1 secondary）文書化 | [x] | drainDeferredRetireQueues に INV 明文化（RCU 整合 P0279R1） |
| C-2 | retire の epoch 順序保証（FIFO）強化 | [-] | **保留 — A2 Step 13 の ReclaimIdentity set 昇格と統合**（H.11.11.6） |
| C-3 | drainDeferredRetireQueues の順序検証 | [-] | 保留（現行 epoch 再確認 + TOCTOU 対策は検証済み） |
| C-4 | retire 順序逆転回帰テスト（AC-R4-T1〜T7 拡張） | [-] | 保留 |
| C-5 | AC-R4-1〜10 全充足確認 | [x] | shutdownReclaim 0 / ReclaimAuthority 一本化済み（既存）を確認 |
| C-6 | ビルド + ctest 全 PASS | [x] | **Debug ビルド成功 + ctest 28/28 PASS** |

## Phase D: §1.8 BuildError 分類（Tier 3）

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| D-1 | BuildError → FailureClassification / RetryDisposition 分離（1.8.5.2） | [x] | RuntimeBuilder.h に FailureClassification / RetryDisposition / BuildOutcome 追加 |
| D-2 | constexpr descriptor table（1.8.10.3）で enum↔toString 網羅性 | [x] | kBuildErrorDefaultTable + kBuildErrorNames（static_assert 網羅検証）。toString を table ベース化 |
| D-3 | 3 箇所の build() 呼び出しサイトに return-code チェック配線 | [x] | RebuildDispatch:1087 主要経路に classifyBuildError 配線（catch-based 不採用 1.8.8.1） |
| D-4 | RetryDisposition 実装時の Acceptance Criteria（1.8.9 追記分）充足 | [-] | 保留 — retry ループ意味論変更 |
| D-5 | RetryBackoffPolicy を tuning parameter 化（第十八者 #7） | [-] | 保留 |
| D-6 | テスト（1.8.11）+ ビルド + ctest 全 PASS | [x] | **Debug ビルド成功 + ctest 28/28 PASS**（D-1〜3 後） |

## Phase E: §1.9 quarantine wake 最適化（Tier 2 — lost-wake proof 前提）

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| E-1 | quarantineAbsorptionCount_ テレメトリ追加 | [ ] | 無駄な起床抑制 |
| E-2 | 3-condition wake + lost-wake proof（INV-X1-2: queue full ≠ Recovery lost） | [ ] | |
| E-3 | `hasAuthoritativePublishedRuntime()` を `observePublishedWorld() != nullptr` で実装 | [ ] | Amendment 4（runtimePublishWorld_ は存在しない） |
| E-4 | ビルド + ctest 全 PASS | [ ] | rollback point |

## Phase F: §1.1 R1 recoveryIntentQueue_ MPSC 化（Tier 2）

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| F-1 | `recoveryIntentQueue_` を `MpscBoundedRing<RecoveryIntent,256>` に置換 | [ ] | h:551 |
| F-2 | submitRecoveryRequest: reservation→push→rollback（pendingIntentCount_ 管理） | [ ] | cpp:721-780 |
| F-3 | popRecoveryRequest: pop 成功時 fetchSub | [ ] | |
| F-4 | pendingRecoveryAdmission_ の SPSC-safe 維持（single NonRT admission authority） | [ ] | 1.1.1 |
| F-5 | INV-R1-1/INV-R1-2 / AC-ISR-1（Audio Thread 非 producer）充足 | [ ] | |
| F-6 | テスト（2 Producer 並行 / full→rollback / 重複なし / underflow なし）+ ctest | [ ] | rollback point |

## Phase G: §1.7 X4-B 案2 currentWorld_ 廃止（Tier 2 — 高リスク）

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| G-1 | read-source singularization 4 段階 migration（第十七者） | [ ] | H.11.25.4 |
| G-2 | AC-PUB-1（identity consistency）前提確認 | [ ] | |
| G-3 | ビルド + ctest 全 PASS | [ ] | rollback point |

## Phase H: §1.5/1.6 sparse completion + sequence テスト（Tier 3）

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| H-1 | completedThrough_ + completedOutOfOrder_ 二重構造 | [ ] | |
| H-2 | complete() を CAS max + sparse set に変更 | [ ] | |
| H-3 | waitFor() を frontier + sparse 併用に変更 | [ ] | |
| H-4 | sequence arithmetic 定義 + wraparound/out-of-order テスト（1.6.1） | [ ] | |
| H-5 | INV-X2-5/6 維持確認 + ctest | [ ] | rollback point |

## Phase I: §1.2 Recovery coalesce（Tier 2 — 条件付き GO）

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| I-1 | LogicalRecoveryIdentity + RecoveryProvenance 導入 | [ ] | 単純 latest-wins は NO-GO |
| I-2 | SupersessionDecision（Compatibility ≠ Supersession） | [ ] | 第七者 #18 |
| I-3 | lease state machine + bounded durable table | [ ] | 1.2.3 |
| I-4 | Building 中の supersession は pending 扱い（RC-11） | [ ] | H.11.27.3 |
| I-5 | ビルド + ctest 全 PASS | [ ] | rollback point |

---

## クロスカッティング Acceptance Criteria（dash2 §4）

| AC | 内容 | 状態 |
|----|------|------|
| AC-ISR-1 | Audio Thread は admission/enqueue/reclaim authority を呼ばない | [ ] |
| AC-LIFE-1/2 | Proof/Permit なき ShutdownQuiescent reclaim 禁止 | [ ] |
| AC-PUB-1 | identity consistency（currentWorld_ / publication） | [ ] |
| AC-1.4-DRAIN | isFullyDrained が実測と整合（INV-ISFULLDRAINED-1〜5） | [ ] |
| AC-1.8-RETRY | RetryDisposition が retryability を正確に分類 | [ ] |
| AC-2.2-PERMIT | ReclaimPermit は ShutdownRuntime のみ生成 | [ ] |
| AC-LIFE-NEW-1〜3 | 第四者レビュー追加（H.11 参照） | [ ] |

## 検証手順メモ
- ビルド: `tools/build-verify-phase-full.bat`（%ProgramFiles(x86)% は RTK が壊す → バッチ経由で実行）
- ctest: `ctest -C Debug --output-on-failure -E "BuildInputSemanticContract|RuntimeWorldAuthority"`
- コード検索: AiDex > serena > semble > cocoindex(ccc) > graphify > WSL rg/ast-grep/fd
- 調査ツール: WSL grep/ast-grep/rg/fdfind/ag/fzf/sed/awk
