# STG-11-D6 Repair Contract Audit — Discovery / D6-1 選定

- Document: `doc/work113/P1-5-IR-P2_STG-11-D6_REPAIR-CONTRACT-AUDIT_20260930.md`
- Work item: **STG-11-D6** — Remaining Bug Discovery / Repair（D6-1 選定）
- Date: 2026-09-30
- Authority: リポジトリルート `ConvoPeq.md`（`ConvoPeq(20260930-132503).md` と同一の統一 authority として扱う）
- Commit / push: **禁止**

---

## 0. 判定

```
NO-GO（D6-1 該当なし。実装に進まない）
```

repository-wide に残存バグ候補を実コードから再評価した結果、
「実害が確認でき、かつ現在の作業ツリーを壊さずに修正できる」1 件は存在しなかった。
推測での選定・実装は Owner の「推測で決めないでください」に反するため行わない。
Contract Audit のみ作成し、Implementation / Gate は作成しない
（D4 指示の前例「Contract Audit が NO-GO の場合は Implementation / Gate は作成不要」に準拠）。

D1〜D5 の未 commit 状態はそのまま保持する。D5 の READY FOR COMMIT は維持する。

---

## 1. Authority（実ファイルから再取得）

| 項目 | 値 |
| --- | --- |
| authority file | リポジトリルート `ConvoPeq.md` |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| SHA-256 | `4014C78F3B2095981817275CC8EE220B6FC2570A04C8A2F1357C751B` |
| size | 5,854,098 B |
| Generated | `2026-09-30 22:17:20` |
| NEWER_SRC_COUNT / `--check` | 0 / FRESH（exit 0） |

`ConvoPeq(20260930-132503).md` という名前のファイルは存在しないが、
Owner 指示により別 authority 扱い・古い snapshot へのフォールバックは行わず、
ルート `ConvoPeq.md` を唯一の authority とした。

### D1〜D5 反映確認（`ConvoPeq.md` 内の実在確認）

| 項目 | marker | 存在 |
| --- | --- | --- |
| D1 | `m_ownedRetireRouter { m_epochDomain }`（:68858） | YES |
| D2 | `SnapshotCoordinator::quarantineRetireSink`（:59817） | YES |
| D3 | `recordAffinityFailure(0, audioMask, err);`（:21971） | YES |
| D4 | `recordSuccessObserved(3, 0, 0, 0);`（:21983） | YES |
| D5 | `recordMmcssEventObserved(5, 0u, 0u, 0u, 0u);`（:12603） | YES |

---

## 2. Discovery 方法

優先順位（Owner §2: データ消失 → UAF/所有権 → RT block/alloc → lifetime →
状態遷移 → shutdown/drain → silent loss → persistence → 数値 → diagnostics）に沿い、
以下を実コード（call chain / ownership / lifetime / thread context）で再評価した：

- STG-11 Discovery の残存項目（D4 / N4 / R1〜R8）
- WORK92 inventory の未確定項目
- retire / reclaim / drain 経路の残差（D1/D2 周辺）
- RT 到達域の backend / mutex / alloc（D3〜D5 周辺の残差）
- shutdown / drain pipeline
- D5-R1 / D5-R2（Owner §4 の再評価指示）

subagent は本環境で利用不可（free tier 制限）のため、調査は直接実施した。

---

## 3. 候補評価表（全件 elimination または記録）

| # | 候補 | 実コード位置 | 結論 | 根拠 |
| --- | --- | --- | --- | --- |
| 1 | D4: `closeReaderRegistration` が RT reader の slot 再取得を壊す | `EpochDomain.h:630` / `ReleaseResources.cpp:274` | **ELIMINATED** | D167-2 の reconfigure/terminal 境界により、reconfigure pass（`:53-58` → `releaseResourcesForReconfigure()`）は `closeReaderRegistration` を呼ばない（当該関数内に 0 件）。close は terminal pass のみ。STG-11 Discovery 時の仮説は現行設計で不成立 |
| 2 | N4: `submitObserve` / `submitQuarantine` overflow | `ISRRuntimePublicationCoordinator.cpp:611` / `:1474` | **ELIMINATED** | reservation-before-push＋rollback、quarantine は専用 fallback ring＋drop counter（Critical 昇格駆動）、observe は条件付き drop＋counter。いずれも文書化された policy どおり。STG-8 との非交差確認は別 scope の大型作業であり D6-1 にしない |
| 3 | R7(a): close 後の既存 reader 再取得 | `RCUReader.h:159` / `EpochDomain.h:48` | **ELIMINATED** | (1) reconfigure では close しない（#1）。(2) terminal 後は audio 停止が前提。(3) `kMaxReaders=64` に対し production reader は 10 件未満（audio/message/publication/convolver/EQ/learner）のため枯渇到達不能 |
| 4 | R7(b): RT 2 経路が `rootEnterSucceeded()` 未確認 | `ConvolverProcessor.Runtime.cpp:240` / `EQProcessor.Processing.cpp:488` | **ELIMINATED** | enter 失敗には slot 枯渇か post-terminal close が必要だが、両方到達不能（#3）。`ObservedRuntime::get()` は fail-closed（nullptr 返却）を確認済み。実害なし |
| 5 | `m_retireSink == nullptr` 時の leak | `SnapshotCoordinator.cpp:30` | **ELIMINATED** | `AudioEngine` ctor（`CtorDtor.cpp:42`）で使用前に必ず設定。production 到達不能（文書化どおり leak-only、UAF なし） |
| 6 | `resetFadeStateAndRetireTarget` の bool 破棄 | `SnapshotCoordinator.cpp:98-113` | **ELIMINATED** | production caller 0 件（decl＋def のみ）。Owner §3 により自動昇格させない。dead code として記録維持 |
| 7 | R1/R2/R3/R5/R6 | Discovery §4 | **ELIMINATED** | Discovery で Already Safe / Observability Gap と確定済み。再評価で覆す新証拠なし |
| 8 | R4 Candidate C | — | **ELIMINATED** | Design Debt（P3）。memory-ordering 再設計級の大型変更であり「現在の作業ツリーを壊さず」が不成立。D6-1 にしない |
| 9 | R8 intermittent | — | **ELIMINATED** | オンデマンド再現不能。D6-1 の deterministic test が書けない。R8 として追跡継続 |
| 10 | `retireRT` / `releaseRT` の D-full silent loss | `ISRRetireRouter.cpp:282` / `RefCountedDeferred.h:40` | **ELIMINATED** | `releaseRT` の caller 0 件。`RefCountedDeferred` は DEPRECATED（唯一利用者移行済み）。production 到達不能 |
| 11 | `enqueueRetireEpochBounded` の bool 破棄 | `AudioEngine.Retire.cpp:30` | **ELIMINATED** | caller 0 件（decl＋def のみ）。dead code |
| 12 | `terminalReclaim` の同期破棄・mutex+heap | `ISRRetireRouter.cpp:509` | **ELIMINATED** | 同期破棄分岐は `epochSafe && !isRt`（NonRT のみ）。store の mutex+heap は NonRT producer のみ到達（B-R2-2/R3-5 の caller 列挙が authoritative）。RT 到達の証明なし |
| 13 | Q/E 格納時の seqId=generation=0 | `ISRRetireRouter.cpp:342-384` / `RetireQuarantineStore.h:70` | **ELIMINATED** | drain の safety 判定は `epoch` のみ使用（`RetireQuarantineStore.h:103-107`）。seqId/generation は metadata であり provenance に影響しない。D1 の epoch provenance 問題と異なり無害 |
| 14 | `emitRetireIntent` の mutex（B-1） | `ISRRetire.cpp:23` | **ELIMINATED** | fast path は lock-free（Vyukov MPSC）。mutex は輻輳時 fallback のみ。production caller は全件 NonRT（Timer/Release/Coordinator worker。B-1 記録で検証済み） |
| 15 | `switchImmediate` の retire 漏れ | `SnapshotCoordinator.h:84` | **ELIMINATED** | `enqueueWithRetry`＋`quarantineRetireSink` fallback あり（`:90-108`）。処理済み |
| 16 | shutdown drain の budget 切れ | `ReleaseResources.cpp:300-440` | **ELIMINATED** | graceful（5s）→ final drain → EmergencyDrain → Q/E/T 強制 drain の多段構成。具体的な欠落なし |
| 17 | Commit 経路の RT logging | `AudioEngine.Commit.cpp` | **ELIMINATED** | commit caller は Transition/Prepare/Timer/PublicationExecutor（いずれも NonRT）。audio callback からの到達なし |
| 18 | WORK92 inventory 未確定分 | inventory 文書 | **ELIMINATED** | 全件 CLOSED（marker 確認済み）。再開すべき具体項目なし |
| 19 | `revertMmcssPriorityOnAudioThread` dead code | `AudioEngine.Timer.cpp:474` | **EXCLUDED** | Owner §3 により自動昇格させない。caller なし。記録維持 |
| 20 | `finalizeMmcssShutdown` | `AudioEngine.Timer.cpp:504` | **EXCLUDED** | D5 で NonRT と判定済み。新たな具体的 defect なし。Owner §3 どおり変更しない |

---

## 4. D5-R1 / D5-R2 の再評価（Owner §4）

### D5-R1 — MMCSS OS API の RT 契約

「audio thread 上で実行する必要がある」と「RT-safe である」は別問題であることを確認し記録する。
`AvSetMmThreadCharacteristicsW`（MMCSS service への LPC）/ `AvRevert` は厳密な RT 契約
（待たない・block しない）を満たさないが、以下により**実害の証明なし**として D5 再実装は行わない：

- 実行は thread 初回 callback および shutdown teardown のみ（定常区間ではない）
- 移動不可能性（calling-thread 登録＋driver 所有）が D5 Audit で立証済み
- 初回 buffer の deadline 内で完了する bounded 性は付帯事実（根拠にはしない）

### D5-R2 — observation の複数 writer 整合性

Audio thread 間の overlap 経路を調査した結果：device switch では旧 thread の join
（`stopAudioOnly`）/ JUCE の旧 device 停止が新 thread 開始に先行し、
writer 間 interleave の具体的 concurrency path は存在しない。
したがって sequential deterministic test（D3-T5c / D4 / D5 の 5 ラウンド）で十分であり、
**D5 defect としない**。将来 driver の overlap が実証された場合は再評価する。

---

## 5. Contract 判定

| 確認項目 | 結果 |
| --- | --- |
| authority（D1〜D5 反映） | PASS（§1） |
| ownership / lifetime の残存欠陥 | なし（§3 #1〜#17 で elimination） |
| thread / RT-nonRT boundary の残存欠陥 | なし（同上） |
| synchronization / capacity の残存欠陥 | なし（同上。R4 は大型のため除外） |
| failure path / shutdown path の残存欠陥 | なし（同上） |
| 新 authority / queue / worker の必要性 | なし（追加する欠陥がないため証明不要） |
| invariant impact | なし（変更を行わない） |
| regression surface | なし（変更を行わない。D1〜D5 の 45/45 は維持） |

```
NO-GO（D6-1 該当なし）
```

---

## 6. 推奨（次工程の投入候補。D6-1 ではない）

| 候補 | 理由 | 規模 |
| --- | --- | --- |
| R8 intermittent 3 件の追跡 | OBS-5 / OBS-D4-3 / OBS-D5 なし（D5 では未観測）。再現時の crash dump 取得が条件 | 観測継続 |
| R4 Candidate C（atomic state word） | Design Debt（P3）。RT 経路の memory ordering 再設計が必要 | 大型・別 STG |
| dead code 一括整理（`revertMmcssPriorityOnAudioThread` / `resetFadeStateAndRetireTarget` / `enqueueRetireEpochBounded` / `releaseRT` 系） | いずれも到達不能を確認済み。単独 STG ではなく一括で処理すべき | 小型・別機会 |
| N4 × STG-8 非交差確認 | overflow policy の体系確認。大型調査 | 別機会 |

---

## 7. 停止状態

- HEAD = `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- branch = `main...origin/main [ahead 9]`、staged = 0
- **commit / push は未実施**
- D1〜D5 の未 commit 変更・D5 の READY FOR COMMIT はそのまま保持
- Implementation / Gate は作成しない（NO-GO のため）
- D6-2 へは進まない。Owner の指示を待つ
