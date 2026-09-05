# D162-2-G4 Evidence — Final Deterministic Soak（60-gen × 2・S3=ON / V-D=ON・production 変更 0）

- Work item: D162-2-G4（D162-2-G ステージ 4/4・最終確定用 deterministic soak）
- Date: 2026-09-04
- 基準ソース: **ConvoPeq.md `Generated: 2026-09-04 20:28:50` 相当の G2 適用済み working tree**（G3 と同一・変更 0）
- 判定: **PASS（#1 / #2 両方）** → **D162-2-G 完了（G0→G1→G2→G3→G4）**

---

## 1. 静的 preflight（最終再確認・全 ✓）

| 項目 | 結果 |
| --- | --- |
| `destroyRolledBackDSP` | V-D caller = **0**（ReleaseResources.cpp 内実呼出 0）・rollback 経路のみ残存（Orchestrator.cpp:292） |
| S3 | `shutdown-clear → retireRegisteredDSP()` が `deferredSlot_.reset()`（Orchestrator.cpp:630-633 → :634）**より前に実行** |
| V-D | active/fading とも `lifetimeMgrForFinalDSP.retire()` のみ（ReleaseResources.cpp:543/:552） |
| `if (false && ...)` | src 全走査 **0 件** |
| `runtimeDSPHandleMap_` erase | authority retire 内（AudioEngine.h:4367） |
| 変更状態 | G3 から source 変更なし（git diff は G2 状態と同一） |

## 2. G4-1 Build / G4-2 CTest

| Gate | 結果 |
| --- | --- |
| G4-1 Build（Debug / Release / RWDI） | 全 **EXIT=0** |
| G4-2 CTest Debug | **40/40 PASS**（37.71s） |
| G4-2 CTest Release | **39/40** — 失敗は **AudioEngineHarness 0xC0000374 のみ**（D162-1P/B/E/F/G1-G3 と同一 pre-existing class）→ **新規失敗 = 0** → 判定条件「既知失敗のみ = PASS」を充足 |

## 3. Run #1 / #2 判定マトリクス

| 判定項目 | #1 | #2 | 必須 |
| --- | --- | --- | --- |
| Build | PASS | — | ✓ |
| CTest | PASS（新規失敗 0） | — | ✓ |
| RWDI 60-gen exit | **0x00000000** | **0x00000000** | ✓ |
| crash dump | **0** | **0** | ✓ |
| Signature A / B / C | **0 / 0 / 0** | **0 / 0 / 0** | ✓ |
| E-3 / INV-D162-8 | **0** | **0** | ✓ |
| EBR overflow | **0** | **0** | ✓ |
| EBR final pending | **0** | **0** | ✓ |
| residual | **0** | **0** | ✓ |
| direct destroy | **0** | **0** | ✓ |
| stale map | **0（MISS 実測）** | **0（MISS 実測）** | ✓ |
| shutdown complete | **1** | **1** | ✓ |
| footprint remaining | **全件 0** | **全件 0** | ✓ |
| E-4 accounting | **完全収支** | **完全収支** | ✓ |
| destroy duplication | **0** | **0** | ✓ |
| generation 1:1 | **✓** | **✓** | ✓ |
| 新規 XRUN class | **なし** | **なし** | ✓ |

**#1 / #2 両方 PASS → D162-2-G4 = PASS。**

## 4. S3 / V-D 経路分離（run 別）

### 4.1 S3

| 項目 | #1 | #2 |
| --- | ---: | ---: |
| `CLEAR_SHUTDOWN_DISPOSITION` | **8** | **6** |
| shutdown-clear retire 呼出 | 8 | 6 |
| shutdown-clear retired=1（単独 executor） | **0**（not observed — §6） | **0** |
| shutdown-clear retired=0（no-op・防護動作） | 8 | 6 |
| E-4d retired=1（先行 disposition） | 8 | 6 |
| closure（destroy + remaining=0） | **8/8** | **6/6** |
| S3 direct destroy | **0** | **0** |

### 4.2 V-D

| 項目 | #1 | #2 |
| --- | ---: | ---: |
| V-D active target | **1** | **1** |
| V-D fading target | 0 | 0 |
| V-D retired=**1** | **1** | **1** |
| V-D EBR enqueue Success | 1（epoch 232） | 1（epoch 237） |
| V-D D117_DESTROY / remaining | 1 / 0 | 1 / 0 |
| direct destroy | 0 | 0 |
| stale map | 0 | 0 |

#2 の V-D 対象（0000025F2D72C080）は **address reuse 個体**（同一アドレスに旧 lifecycle
gen=56・新 lifecycle gen=61 が存在）だったが、generation 検証付き registry が新 lifecycle
を正しく解決し、旧 lifecycle との混同なし。#1 の V-D 対象は gen=59。

### 4.3 個体単位 closure 例（#1・V-D）

```text
[D162-2G2_VD_RETIRE] dsp=000001EC91BA8080 target=active-final
[D117_RETIRE]        dsp=000001EC91BA8080 retired=1
[D117_RETIRE]        dsp=000001EC91BA8080 enqueue=0 epoch=232
[D117_DESTROY]       dsp=000001EC91BA8080          ← ~AudioEngine enter (198327) 後 198334 = dtor D5/D8 内
[DSP_FOOTPRINT_RELEASED] dsp=000001EC91BA8080 remaining=0
```

S3 対象 8 個体（#1）: 000001EC91BA6080 / AEA98080 / B54C3080 / B9DBD080 / BD7AA080 /
BEB18080 / BEC10080 / C0A59080 — 全て E-4d retired=1 → EBR Success → destroy → remaining=0。
S3 と V-D の対象は **run 内で重複なし（独立個体）**。

## 5. generation 1:1 突合（最優先項目）

| 項目 | #1 | #2 |
| --- | --- | --- |
| destroyed gen 範囲 | **4–64 連続** | **3–63 連続** |
| destroyed gen 数 | **61** | **61** |
| 同一 gen の destroy > 1 | **0 件** | **0 件** |
| gen sequence gap | **なし** | **なし** |
| 構築 = 破壊（orphan/mismatch） | **一致** | **一致** |

**両 run で CREATE された generation と DESTROY された generation が 1:1 で完全対応。**
orphan / mismatch / 二重 destroy なし。

## 6. stale map の確認（指示 §6）

| run | V-D retire 直後の E-2 `retireByHandle` | 判定 |
| --- | --- | --- |
| #1 | `[D117_RETIRE_BY_HANDLE] handle=31 lookup=MISS (not in runtimeDSPHandleMap_)`（~AudioEngine 内） | **authority erase 完了 → no-op** |
| #2 | `[D117_RETIRE_BY_HANDLE] handle=34 lookup=MISS`（同上） | 同上 |

**HIT による STOP 条件は発生せず。** HIT（#1: 10 件 / #2: 11 件）は全て V-D 以外の
生存 DSP への正常 authority destroy（gen 1:1 チェックにより二重 destroy なしと裏取り済み）。

## 7. E-4 accounting（各 run 完全収支）

| run | CREATE | CONSUME | OVERWRITE | CLEAR | DISCARD | 収支 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| #1 | 26,254 | 26,209 | 35 | 8 | 2 | **26,254 = 26,254 ✓** |
| #2 | 25,672 | 25,628 | 36 | 6 | 2 | **25,672 = 25,672 ✓** |

数字そのものの run 間差（G3: 23,676）は timing variance であり、必要条件は
「**各 run で完全収支**」→ **両 run 充足**。

## 8. EBR / shutdown / XRUN

| 項目 | #1 | #2 |
| --- | --- | --- |
| EBR pend 最終 / 最大 | 0 / 2（運転中一時・運転内消化） | 0 / 2（同） |
| EBR overflow | 0 | 0 |
| E-3 / INV-D162-8 | 0 | 0 |
| shutdown sequence complete | 1 | 1 |
| 最終 destroy 位置 | `~AudioEngine enter` 後 = **dtor body D5/D8 内**（INV-D162-8 適合） | 同 |
| XRUN 件数 / max Callback | **15 / 2.15ms** | **12 / 1.70ms** |
| XRUN Pressure / shutdown 窓 | 全 Pressure=0・VerifyDrained 前に完結 | 同 |

XRUN は既知 jitter クラス（Expected 5.33ms に対する軽微超過・E baseline max 1.57ms /
G3 max 2.15ms と同域）で **S3/V-D ON 固有の新規クラスなし** → STOP 条件非該当。

## 9. crash dump 分類

| dump | 由来 | 分類 |
| --- | --- | --- |
| AudioEngineHarness.exe.7920.dmp（21:17） | G4-2 Release CTest AudioEngineHarness 0xC0000374 | pre-existing（D162-1P/B/E/F/G1-G3 文書済み） |
| rpcs3.exe.11936.dmp（21:18） | **外部アプリ（rpcs3 エミュレータ）** — 本環境のテスト対象外 | 無関係 |
| G4 60-gen #1 / #2 | — | **dump 0 件** |

## 10. S3 standalone retired=1 の扱い（指示 §9）

```text
S3 standalone retired=1:  not observed（G1〜G4 通じて）
S3 execution:             observed（#1: 8 回 / #2: 6 回・block 実行時到達）
S3 closure:               observed through preceding E-4d retire + authority no-op
```

通常 soak では Timer C2/C3/C4 → requestDeferredClear が RebuildThread 生存中に latch 処理
されるため、E-4d が常に先行 disposition を実行する。異常系の人工生成は行っていない
（指示どおり）。S3 block は no-op 帰還を含め **14 回**（G1-G4 累積）実証済み。

## 11. G4 判定

**#1 PASS + #2 PASS → D162-2-G4 = PASS。**

同一 source / 同一設定 / 同一 soak profile に対し、2 run 連続で同一の lifecycle closure
（exit 0x0・Signature 0・E-3 0・EBR closure・residual 0・generation 1:1・二重 destroy 0・
E-4 完全収支・shutdown complete・footprint remaining 全件 0）が成立することを確認した。
lifecycle closure 自体は deterministic（both runs identical accounting structure）であり、
disposition trigger count（S3 8/6・DISCARD 2/2 等）のみ timing variance を示す — これは
soak profile の負荷揺らぎ由来で closure 契約には影響しない。

---

## 12. D162-2-G 総括（G0 → G1 → G2 → G3 → G4）

| Stage | 判定 | 成果 |
| --- | --- | --- |
| G0 preflight | PASS | S3/V-D switch 特定・V-D-a stale-map latent 発見・V-D-b 設計確定 |
| G1 S3 re-enable | PASS | S3 = authority（EBR）経由で ON 化・destroy 61 = F+1（無処分 1 件解消） |
| G2 V-D re-enable | PASS | V-D direct destroy 廃止 → authority retire 統一・V-D retired=1 実測・stale map MISS 実測 |
| G3 joint reproof | PASS | S3+V-D 同時成立・別個体閉包・二重 destroy 0・変更 0 |
| G4 final soak | PASS | 60-gen × 2 で lifecycle closure の決定論的再現 |

**D162-2-G（S3/V-D staged re-enable / final validation）完了。** S3/V-D のいずれも
authority（`DSPLifetimeManager::retire` → map erase → EBR → destroy）に統一され、
INV-D162-1〜9 全契約下で shutdown 時 DSP 破壊が EBR 単経路で閉じることを 60-gen × 2 で
確定した。Direct destroy は V-D 経路から消滅（rollback 経路の正当 caller のみ残存）。
