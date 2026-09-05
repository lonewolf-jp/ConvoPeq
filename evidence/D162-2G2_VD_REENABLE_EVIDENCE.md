# D162-2-G2 Evidence — V-D-b Staged Re-enable (S3=ON / V-D=ON via authority retire)

- Work item: D162-2-G2（V-D direct destroy 廃止 → EBR authority 統一・ステージ 2/4）
- Date: 2026-09-04
- 基準ソース: G1 適用済み working tree に G2 変更を適用（ConvoPeq.md 再生成 `Generated: 2026-09-04 20:28:50`）
- production 変更: **1 ファイル**（`src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` のみ）
- 判定: **PASS**（§9 判定表）

---

## 1. G2 diff（V-D-b）

VerifyDrained 破壊 block（実ファイル行 523-556）のみ変更:

```cpp
// before（B-era 休眠コード）
if (false && activeDSPToDestroy != nullptr)
    lifetimeMgrForFinalDSP.destroyRolledBackDSP(activeDSPToDestroy);
if (false && fadingDSPToDestroy != nullptr)
    lifetimeMgrForFinalDSP.destroyRolledBackDSP(fadingDSPToDestroy);

// after（V-D-b）
if (activeDSPToDestroy != nullptr)                       // ★ D162-2-G2
{
    // DIAG: [D162-2G2_VD_RETIRE] dsp=%p target=active-final
    lifetimeMgrForFinalDSP.retire(activeDSPToDestroy);   // map erase + registry Retired
                                                         // + requestReclaim + EBR enqueue
}
if (fadingDSPToDestroy != nullptr && fadingDSPToDestroy != activeDSPToDestroy)
{                                                        // ★ 同一 DSP 二重 retire 防止ガード
    // DIAG: [D162-2G2_VD_RETIRE] dsp=%p target=fading-final
    lifetimeMgrForFinalDSP.retire(fadingDSPToDestroy);
}
```

- `destroyRolledBackDSP()` 関数自体は削除していない（DSPLifetimeManager.cpp:149 残存・
  rollback 経路で今後も使用）。
- S3（G1 の `shutdown-clear` retire）は維持（Orchestrator.cpp:632 確認）。
- その他の production source は未接触（git status の他ファイル変更は E/G1 由来の既存分）。

## 2. 静的チェック（指示 §3）

| 項目 | 結果 |
| --- | --- |
| V-D active: `if (activeDSPToDestroy != nullptr) → retire` | ✓（ReleaseResources.cpp:536/:543） |
| V-D fading: `if (fadingDSPToDestroy != nullptr && != active) → retire` | ✓（:545/:552） |
| S3: shutdown-clear retire 維持 | ✓（Orchestrator.cpp:632） |
| V-D: destroyRolledBackDSP 呼び出しなし | ✓（src 全走査で 0） |
| V-D: `if (false &&` なし | ✓（src 全走査で 0 件 — 全ソースから消滅） |
| map erase が authority 内で実行 | ✓（AudioEngine.h:4367 `runtimeDSPHandleMap_.erase(dsp)` — retire → retireDSPHandleForRuntime 経路） |

## 3. Gate ladder

| Gate | 結果 |
| --- | --- |
| G2-1 Build（Debug 236 / Release / RWDI 138） | 全 **EXIT=0** |
| G2-2 CTest | Debug **40/40 PASS**（34.52s）/ Release **39/40**（AudioEngineHarness 0xC0000374 = G1/B/E/F と同一 pre-existing・Test #36 HeadlessAudioPathVerification は PASS → 39/39 相当） |
| G2-3 Debug 6-gen | **exit 0x00000000**・dump 0 |
| G2-4 RWDI 6-gen | **exit 0x00000000**・dump 0 |
| G2-5 RWDI 60-gen | **exit 0x00000000**・dump 0 |

## 4. V-D 固有観測（指示 §5 A–D）

### A. V-D retire の実測（3 run 全てで retired=1）

| run | V-D target | target 種別 | retired | enqueue | epoch |
| --- | --- | --- | --- | --- | --- |
| Debug 6-gen | 00000259B52640C0 | **active-final** | **1** | 0=Success | 35 |
| RWDI 6-gen | 0000023133E21080 | **active-final** | **1** | 0=Success | 33 |
| RWDI 60-gen | 000002385A4EF080 | **active-final** | **1** | 0=Success | 236 |

- V-D active target count: **1 / 1 / 1**（各 run 1 件）
- V-D fading target count: **0 / 0 / 0**（fading は通常 crossfade 経路で先に retire 済みのため、
  VerifyDrained 時点で残存する final fading は今回未発生 — 構造上は同一 authority を通る）
- V-D retire(retired=1) count: **1 / 1 / 1** — **G2 の本質的条件（V-D が retired=1 で動作）を実測達成**
- V-D retire(retired=0) count: 0 / 0 / 0
- V-D EBR enqueue count: 1 / 1 / 1（全て Success）

### B. direct destroy = 0

- `[D162-2B_DESTROY]`（旧 direct destroy DIAG）: **0 件**（全 run）— V-D-b の本質検証条件を達成。

### C. D117 destroy 対応（個体単位閉包）

```text
[D162-2G2_VD_RETIRE] dsp=000002385A4EF080 target=active-final   （world clear 後・releaseResources 内）
[D117_RETIRE]        dsp=000002385A4EF080 retired=1
[D117_RETIRE]        dsp=000002385A4EF080 enqueue=0 epoch=236    （EBR 破壊権取得 Success）
[D117_DESTROY]       dsp=000002385A4EF080                        （dtor body D5/D8 drain 内 digest・
                                                                  ~AudioEngine enter (line 200007) の後）
[DSP_FOOTPRINT_RELEASED] dsp=000002385A4EF080 remaining=0
```

Debug 6-gen / RWDI 6-gen も同型（retired=1 → enqueue Success → DESTROY → remaining=0）。
**3/3 run で retire → EBR → destroy → footprint released の完全閉包。**

### D. stale map の不存在（V-D-a との決定的差分・実測証明）

RWDI 6-gen / 60-gen とも、V-D retire（map erase 完了）直後の ~AudioEngine E-2 が
**`[D117_RETIRE_BY_HANDLE] handle=... lookup=MISS (not in runtimeDSPHandleMap_)`** を記録。

- V-D-a（direct destroy）では map entry が残存するため、この lookup は HIT し破壊済み
  DSP へ enqueue し得た（G0 N-1 の latent 経路）。
- V-D-b では authority retire が map を erase するため **MISS = no-op** となり、
  stale map entry 0 を実行時证明した。**stale map = 0 ✓**

### 補足観測（address reuse の正しい処理）

60-gen の V-D 対象アドレス `000002385A4EF080` は、run 前半に破壊された別 DSP と同じ
アドレスが再利用された個体であった（epoch 220 帯の旧 lifecycle と epoch 236 帯の新 lifecycle
が同一アドレスに共存）。generation 検証付き handle registry のもとで V-D retire は新 lifecycle
の map entry を正しく解決し（retired=1）、旧 lifecycle との混同は発生しなかった —
D162-2-C §10 で指摘された address reuse 環境での authority 経路の健全性を裏付ける観測。

## 5. 全体観測（60-gen）

| 指標 | Debug 6-gen | RWDI 6-gen | RWDI 60-gen |
| --- | --- | --- | --- |
| exit code | 0x00000000 | 0x00000000 | 0x00000000 |
| E-4 accounting | 1626 = 1623+2+**1** | 2709 = 2705+3+**1** | **26603 = 26558+37+7+1** |
| CLEAR / CLEAR_MIDRUN / CLEAR_SHUTDOWN_DISPOSITION | 1/1/1 | 1/1/1 | 7/7/7（S3 維持） |
| D117_DESTROY / FOOTPRINT_RELEASED | 7 / 7 | 6 / 6 | **61 / 61** |
| residual | 0 | 0 | **0**（destroy 61 = 構築 60 + placeholder 1） |
| EBR pend 最終 / 最大 | 0 / 1 | 0 / 1 | 0 / 2（運転中一時・運転内消化） |
| EBR overflow | 0 | 0 | 0 |
| E-3（INV-D162-8） | 0 | 0 | **0** |
| Signature A / B / C | 0 / 0 / 0 | 0 / 0 / 0 | **0 / 0 / 0** |
| shutdown sequence complete | 1 | 1 | 1 |
| XRUN | 2 | 2 | **17**（Callback ≤1.87ms・E baseline 17 と同水準 / F 15 — 新規クラスなし） |
| DC live | 収束 | 収束 | 1–3 収束 |
| Priv 最終 | — | — | **442MB**（F=445 / E=461 と同水準） |

60-gen の最終 destroy（dsp=000002385A4EF080）は `~AudioEngine: enter`（line 200007）後
line 200014 で実行 = **dtor body 内 D5/D8 drain での digest**（INV-D162-8 適合）。

## 6. crash dump 分類

| dump | 時刻 | 由来 | 分類 |
| --- | --- | --- | --- |
| ConvoPeq.exe.25884.dmp | 20:15 | Release CTest HeadlessAudioPathVerification → 旧 build-icx binary の tolerated static-teardown AV（G1 の 3668 と同一クラス） | pre-existing・G2 無関係 |
| AudioEngineHarness.exe.4228.dmp | 20:15 | Release AudioEngineHarness 0xC0000374（D162-1P/B/E/F/G1 文書済み） | pre-existing・G2 無関係 |
| G2 soak 3 run | — | dump **0 件** | — |

## 7. STOP 条件チェック（指示 §6）

| STOP 条件 | 観測 |
| --- | --- |
| Signature A（AudioSegmentBuffer → aligned_free → mkl_free） | **0 件** → 発生せず |
| Signature C（新規 AV/UAF/double-free） | **0 件** → 発生せず |
| E-3 pendingRetireCount != 0 | **0 件** → 発生せず |
| V-D direct destroy（destroyRolledBackDSP 破壊） | **0 件** → 発生せず |
| EBR closure failure | **なし**（3 run 全閉包） |
| stale map entry 残存 | **なし**（retireByHandle MISS を実測） |

## 8. Signature B（detectStuckReaders）

Debug 6-gen 含め全 run で 0 件 — 発生すらせず（G1 と同様）。記録対象なし。

## 9. G2 判定

| PASS 基準 | 結果 |
| --- | --- |
| Build | **PASS**（3 config EXIT=0） |
| CTest | **PASS**（Debug 40/40・Release 39/40 = pre-existing のみ・新規失敗なし） |
| Debug 6-gen | **PASS**（exit 0 / Signature A/C なし） |
| RWDI 6-gen | **PASS**（exit 0 / Signature A/C なし） |
| RWDI 60-gen | **PASS**（exit 0 / Signature A/C なし） |
| V-D direct destroy = 0 | **達成** |
| V-D authority retire 実測 | **達成**（3 run で retired=1） |
| V-D EBR closure 完全 | **達成**（destroy + remaining=0） |
| stale map entry = 0 | **達成**（retireByHandle MISS 実測） |
| residual / EBR pend / ovf / E-3 | **0 / 0 / 0 / 0** |
| S3 disposition・E-4 会計・footprint closure・shutdown completion（G1 条件） | **維持** |

**D162-2-G2 = PASS。** G3（S3+V-D 同時の本命再証明）への進行条件を満たす。
現在の source 状態は既に S3=ON / V-D=ON であるため、**G3 は production 変更 0 で
Gate ladder のみを実行する形**になる。
