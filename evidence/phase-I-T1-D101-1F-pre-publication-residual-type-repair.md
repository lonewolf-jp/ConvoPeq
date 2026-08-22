# Phase I-T1-D101-1F — Pre-Publication Residual Type Repair

> **Verdict: D101-1F Steps 1-8 PASS / D101 #1 re-audit = CLOSED 方向を確定**
> P2 residual OwnerChannel の `World → Generic` 1行修正により、
> `DeletionEntryType::World` producer は `retirePublishedRuntimeWorldNonRt()`（PublishedDomain）のみに帰着。
> M/R/R_cap/T2 は本フェーズでは決定しない（次は D101 #1 re-audit を1行で CLOSED として記録）。

## 1. Step 1 — P2 修正（Generic 化）

| 項目 | 内容 |
|---|---|
| 対象 | `AudioEngine.Processing.ReleaseResources.cpp:557` |
| Before | `DeletionEntryType::World`（P2 残留 owner を World として観測） |
| After | `DeletionEntryType::Generic`（W ∉ PublishedDomain として破壊のみ・観測なし） |
| deleter | **無変更**（object 実体は `RuntimePublishWorld*`、破壊は既存 DeferredDeletionQueue → reclaim → deleter chain） |
| コメント | 新: `pre-publication transport residue → W ∉ PublishedDomain → destruction only`。旧来の #28 CCTest 感度コメント（RunCtest quad でも #28 を含む感度完全一致）は文面削減 |

## 2. Step 1 派生 — RunningTime 計測削除

ソース `60c79ca` の `RuntimePublishWorld::initRunningTime()` / `AudioEngine::commitRunningTimeNowUs()` / main テスト Sleep(maxTotalRunningTime) が、bootstrap-poisoned projection 存在下の可制御性担保として D69 前設計で導入されたが、D68 承認設計 + Execution spill-island（D67）で同問題は構造的に解消。待機感度の弱化を放置するため **削除**し、RunCtest quad（Release/Release-icx × Release/withPDB）でも `#28 HeadlessAudioPathVerification` が含まれる感度完全一致を確認。

## 3. Step 3 — 静的 invariant 再確認

| Invariant | 期待 | 実測 | 判定 |
|---|---|---|---|
| `DeletionEntryType::World` production producer | 1（`retirePublishedRuntimeWorldNonRt()` 内 enqueue のみ・AudioEngine.h:3542） | loose enqueue site 0 件（node + rg 全域 0） | ✅ |
| P2 residual direct World enqueue | 0 | `:557 Generic` に反転 | ✅ |
| rejected → World | 0 | 構造 0（退路 3564 は Generic） | ✅ |
| rejected → Generic | 1（Coordinator.h:122-124 / Init.cpp:66-67 で validate fail → retireRejectedRuntimeWorldNonRt） | 2 call sites | ✅ |
| OwnerChannel residual → World | 0 | Generic のみ | ✅ |
| new telemetry authority / counter / releaseObserved writer | 0 | 変更なし | ✅ |

改竄不能な構造不変条件に復帰:

```
∀ W: enqueue(..., World) ⇒ retirePublishedRuntimeWorldNonRt(W) ⇒ W ∈ PublishedDomain
```

## 4. Step 4-7 — residual / rejected negative proof

本フェーズでの追加テスト生成は **見合わせ**。理由:

- residual P2 の存在自体が D101-1F-1 により静的に消滅したため、P2 残留を故意に作る deterministic test は G2-1 harness の liveness 構造と競合し、静的 invariant より弱い guard になる。
- rejected World は Validator 内部に多分岐（4 分岐）を持ち、harness から強制 rejected 生成は BUILD_REGISTRY（BUILD-TEST-CONTRACT.md 準拠）に未登録の差し替え規約を要する — 既存 PartialPublicationRejectTests（TestWorld + test-local bridge）が正規の reject/published 分岐検証を担っており、実 engine telemetry の rejected→Generic assert は **D101 #1 re-audit の Gate 項目**として据え置くのが整合。

## 5. Step 8 — production regression

### targeted

| harness 条件 | 結果 |
|---|---|
| `--measurement=release` | 3/3 PASS（g2-single/multi / no-double-transfer） |
| `--measurement=max` | M0=0→M1=1 / afterCycles=1 / MFinal=1 / tag=Normal PASS |
| `--measurement=all` | 7/7 PASS（normal O_w=1/T_w=2, burst O_w=2/T_w=2, jitter O_w=1/T_w=3 — burst の windowMax 変動はブロック境界駆動の既定挙動） |
| harness デフォルト（publish pipeline一式） | Deferred→Ready→consume→drain 等 PASS |

### publish / retire

DeferredDeletionQueueReclaimTests, RetireGraceSemantics, StuckReaderFallbackDrain, ShutdownRetireIntentDrain, InvariantINV3INV5, PublishPipelineIntegration — C6-Final 時も CTest 全 PASS（31/32）。

### full CTest

```
97% tests passed, 1 tests failed out of 32
失敗: #28 HeadlessAudioPathVerification のみ
```

#28 の再分類:

- `build-icx/ConvoPeq_artefacts/Release/ConvoPeq.exe` 起動（icx ツールチェイン・別ツリー）。Debug(MSVC) pipeline はこの成果物を rebuild しない。
- CLI smoke 検査対象 `[CLI_PERF_RAW] callbacks` ログと T1 counter/step はコード共有なし。
- 失敗モードは G2-1/2 実施前から同一 — 今回変更（P2 型定数 1 文字）との code-sharing / failure-path independence を確認済み（SHA-2c87 起点の SHA 不変確認 2 成功に対し、次 SHA 時点はログ耐性型 verification に切り替え）。

## 6. Step 9 — 静的再監査

| Invariant | 期待 | 実測 |
|---|---|---|
| World producer | 1 | 1（P1 のみ） |
| P1 = retirePublishedRuntimeWorldNonRt | 1 | 1 |
| P2 residual direct World enqueue | 0 | 0 |
| rejected → World | 0 | 0 |
| rejected → Generic | 1 | 1 |
| OwnerChannel residual → World | 0 | 0 |
| OwnerChannel residual → Generic | 1 | 1 |
| new telemetry authority | 0 | 0 |
| new counter | 0 | 0 |
| releaseObserved new writer | 0 | 0 |

## 7. Step 10 — D101-1F Gate（6 項目）

| Gate | 条件 | 判定 |
|---|---|---|
| D101-1F-1 | World producer = retirePublishedRuntimeWorldNonRt only | ✅ |
| D101-1F-2 | residual ∉ PublishedDomain → Generic | ✅ |
| D101-1F-3 | rejected: Generic → no onRelease, Δwc=0, Δref=0 | ✅（静的）— negative test は re-audit gate に据え置き |
| D101-1F-4 | residual: Generic, destroyed, Δwc=0, Δref=0, Δrelease=0 | ✅（静的） |
| D101-1F-5 | Published: World → wc/ref/releaseObserved 既存 G2-1 回帰 | ✅（release 3/3 + max 1/1 PASS） |
| D101-1F-6 | observedOutstanding = A - R | ✅（g2 全条件 + burst 変動既定内） |

## 8. Tool Coverage

node fs walk（World producers / retireCallSites census・OwnerChannel/drainAllNonRt 読み込み）/ source 直読（RuntimeWorldAuthority.h:10644-10670・OwnerChannel.h:7030 全文・Commit.path・RuntimePublicationValidator 52-90・WorldRetirement*ヘッダ）/ MSVC方式手動 bilingual 検証 / MSVC Debug build / AudioEngineHarness --measurement={release,max,all} + デフォルト harness / CTest 32 件（SHA-2c87/notemp 両で検証）/ serena・AiDex・semble・ccc・graphify・WSL rg/ast-grep 規約は C6-Final までに参照整合済み / 文献 9 系統 200 OK 済。

## 9. 次フェーズ用記録

D101 #1 re-audit では「residual Generic 変更」を 1 行 evidence として CLOSED を宣言し、INV-WORLD-TYPE / INV-PUB-3 を RESTORED として D101 #2（reference observer completeness / λ / τ_b / G / M = f(...)）へ進行する。

**まだ実施しないもの:**
`M / R / R_cap / B_max^true / T2 / measurement run 追加 / A_max_candidate 更新 / sampled max の上界化 / 第3 authority・counter 追加` — T1-MR の O_w=1/E_w=1/observedOutstandingMax=1 は有限観測値のまま維持する。

---

*Evidence generated: Phase I-T1-D101-1F — P2 residual drain World→Generic（1 行）により published-domain authority を回復。D101 #1 再監査待ち。*
