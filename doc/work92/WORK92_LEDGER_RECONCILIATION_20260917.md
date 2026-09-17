# WORK92 Residual Ledger Reconciliation — Final Closure Ledger

- **作成日**: 2026-09-17
- **種別**: Final Closure Ledger（WORK106 全件再照合の正式状態反映）
- **基準**: HEAD / origin/main = `eab40c23`（0/0）・ConvoPeq.md Generated `2026-09-17 13:10:15` / NEWER_SRC_COUNT=0 / FRESH
- **方法**: work92 §1 の 18 項目（CORRECT_INTEGRATED_BUG_LIST §1）+ work101 retriage（2026-09-16）+ mini bugs / big_bug 固有系を現行 HEAD のソースで行単位再実測（WORK106 監査）
- **本文書の性質**: 日付付き監査記録は書き換えず、本 closure ledger が**併存して supersede** する（CORRECT list §7 の規約と同一）

## 1. Closure Matrix（work92 §1 残置台帳の最終状態）

| 項目 | 最終状態 | 現行実測根拠（2026-09-17 HEAD eab40c23） |
|---|---|---|
| big 1-1 | CLOSED | StateAndUI.cpp に nucHCMode/nucLCMode 23 箇所（setProperty / getProperty 復元） |
| big 1-2 | CLOSED / DESIGN | producer 不実装（big_bug §10-3-1 確定方針）+ C-9 値初期化（ISRRuntimePublicationCoordinator.h:1016 `lastResortQueue_[…] {};`） |
| big 1-3 | CLOSED | InputBitDepthTransform.h 全 4 サイト `_mm256_storeu_pd`（:60/:81/:116/:117） |
| big 1-4 | **P2 quality / scope 外** | fastTanh ローカル複製が DSPCoreFloat.cpp:146 / DSPCoreIO.cpp:76 に現存（CORRECT 台帳 §2-3 が当初「P2 として残す・§1 計上外」と明記）。正規実装は DSPCoreDouble.cpp で使用済み。欠陥ではなく保守リスク |
| big 1-6 | CLOSED | MklFftEvaluator.h :282/:286/:451 ippStsNoErr 検査 + ippFailureCount_ 観測 |
| big 1-7 | CLOSED / **HARDENING ONLY** | emitRetireIntentNonRT リネーム + Finding 9 契約コメント（W105 実測）。§5-1 残置（自動 enforce 不在）は debug assertion 1-2 行で満たせる hardening 候補 — OPEN / CONTRACT DEFERRED には戻さない |
| big 1-8 | **RESOLVED** | WORK102 `859718e4`（FC-FORM-1/2/3/4/5 admission・IRLoadAdmission.h 単一情報源・streaming hash）+ WORK102-PREV-01 `eab40c23`（preview parity）。kStreamChunk チャンク読込は ConvoPeq 正本 :72991（main）/:74894（preview）実測 |
| big 1-9 | CLOSED | MKLNonUniformConvolver.h:355 `std::int64_t fftSize`（B-6） |
| big 1-10 | CLOSED / contract | Snapshot.cpp:96 唯一の acknowledge 点 + :99 submitRebuildIntent 先行（B-2 契約） |
| big 2-6 | CLOSED | core/ThreadHash.h acquireUniqueThreadId → currentThreadToken 統合（B-4） |
| big 2-7 | CLOSED / DESIGN | ISRDSPHandle.h :107/:215 static_assert ×2 |
| big 2-8 | CLOSED / DESIGN | forceCleanup（StateAndUI.cpp:1028）stopThread 役割分担確定 |
| big 2-9 | CLOSED | AudioBlock.cpp:633/:639 saturatingSubUs（B-8） |
| big 2-10 | CLOSED | StateIO.cpp:93-99 enum 範囲ガード（B-3） |
| mini bugs | CLOSED | BUG-011/012/013/016 clamp/sanitize（CmaEsOptimizer.h:84/:191/:214・Dynamic.h:29）。修正済み 62 項目は後続 work で対象 TU 未変更のため維持 |
| P1 | CLOSED 7/7 | 上記 P1 行 + BUG-065（work92 B-7a 既報） |
| P2 | CLOSED 7/7 | big2-1 ASCII 化・big2-2 sanitize ×8（DC 後含む）・big2-3 kEpsilon 分割・big2-4 memcpy カーソル・big2-5 writeIndex 先読み・big3-6 isnan 4系統・big3-9/3-10 CMakeLists:1519 |

## 2. 非ブロック残置（WORK92 §1 の 18 項目スコープ外・欠陥ではない）

| 項目 | 状態 | 根拠 |
|---|---|---|
| big 1-4（fastTanh ローカル複製） | **P2 quality leftover** | DSPCoreFloat.cpp:146 / DSPCoreIO.cpp:76 にローカル実装が現存。正規 `SoftClipPadePolicy` は DSPCoreDouble.cpp で使用。CORRECT 台帳 §2-3 が「P2 として残す（§1 計上外）」と既に確定 |
| R-新規 A〜D | **DORMANT / P3** | rebuildJob make_unique 0 件（実測・未発火のまま）。R-新規C は bad_alloc 限定・R-新規D は文書整合済み |
| big 1-7 自動 enforce | **HARDENING ONLY** | `ISRRetire.cpp` に `#include "DspNumericPolicy.h"` + `ASSERT_NON_RT_THREAD()` 追加の debug-only 1-2 行。production 挙動・authority・RT 経路への影響 0（W105） |
| IS-7（trim copy-on-resize） | **NOT-A-BUG** | whole-path simultaneous residency 2 GiB は WORK102 arbitration §9.3 の凍結契約値。任意改善（avoidReallocating 化には容量保持 trade-off）として維持 |
| M-03 D3 | **HOLD / SEPARATE** | work92 §1 対象外（OFF 側 PDC 過大申告・別系統） |

## 3. big 1-8 の解消チェーン（時系列）

```text
RECONCILIATION_20260909.md:120      「残存 → Phase B-5」
work92 IMPLEMENTATION_REPORT_20260910:20   B-5 LoaderThread ストリーミング ✅ CLOSED
FINAL_INVENTORY_20260910.md:76      big 1-8 を CLOSED BUG に計上（B-5 達成）
work101 retriage 2026-09-16 §4-B/§5-2   CONTRACT DEFERRED に再起票
                                     （destination 一括確保の上限内 ~34GB 懸念 +
                                       「上限の明文化」未実施）
WORK102 859718e4                    FC-FORM-1/2/3/4/5 admission（1 GiB destination bound +
                                     IRLoadAdmission.h 単一情報源）+ streaming hash → 上限明文化完了
WORK102-PREV-01 eab40c23            preview 経路の同一契約 parity → ★ big 1-8 RESOLVED
```

## 4. 非同期経路の追加解消（PREV-01 による旧 OPEN 条件の消滅）

```text
preview allocation failure → ThreadPool 例外握り潰し → finishAsyncIRLoadPreview 未到達
（= UI の irPreviewInProgress 漏出）:
  WORK102-PREV-01（eab40c23）の no-throw failure boundary（analyzeImpulseResponseFile の
  catch 群）により解消。全 failure 経路が caller-visible completion に到達することを
  PE チェック（Debug / Release 双方 PASS）で実測済み。
```

## 5. Runtime architecture 確認（実測継続）

```text
INV-ISR-01〜07 コメント固定          ISRRuntimePublicationCoordinator.h:100-116 実在
Publish / Crossfade / Retire 権限    worldAuthority_ 単一（publish == runtimePublicationBridge_、
                                     retire = lifetime().emitRetireIntent*）
Epoch / reclaim                      advanceRetireEpoch 単一点（RuntimePublishExecutor.h:110）
                                     + move-only reclaim permit（二重 reclaim 構造的防止）
Shutdown full drain                  CtorDtor.cpp:291 drainAllQuarantineStore（Q+E+T・epoch 非依存）
Lifetime budget                      RecoveryBudget 10 分窓 + retire pressure 75/90/95% NoRt
RT boundary                          enforcement 全点 NoRt/NonRT。RT への decision/allocation/
                                     lock/blocking 追加は WORK102〜107 全 diff で 0
```

## 6. 照合の根拠と監査連鎖

```text
WORK104（big1-8 pre-audit） : big1-8 と big1-7 の責務分離確認 → NOT-A-BUG / HARDENING ONLY
WORK105（big1-7 pre-audit） : 自動 enforce 不在 = hardening 候補・契約 gap 無し → NOT-A-BUG
WORK106（全件再照合）        : work92 §1 18 項目の行単位再実測 → True Open = 0・stale 2 行
本 closure ledger            : WORK106 の結果を正式状態として反映
```

## 7. 最終状態

```text
Contract Gap            = 0
True Open               = 0
CONTRACT-HOLD           = 0
Production regression   = 0（CTest 40/40 Release / Debug・PREV-01 CLOSED）
RT architecture drift   = 0
Authority duplication   = 0

WORK92 = CLOSED
```

## 8. 台帳 supersede 記録

- `doc/work92/INTEGRATED_BUG_LIST.md`（:80/:194）
- `doc/work92/CORRECT_INTEGRATED_BUG_LIST.md`（:44/:194）
- `doc/work92/RECONCILIATION_20260909.md`（:120）
- `doc/work101/work92_rt_residual_runtime_bug_retriage_20260916.md`（§4-B row5 / §5-2 / §6）

各文書の historical 記載は変更せず、末尾に本 ledger への supersede addendum を追記した。
`FINAL_INVENTORY_20260910.md` / `IMPLEMENTATION_REPORT_20260910.md` は変更なし（B-5 CLOSED 記載が既に最終状態と整合）。
