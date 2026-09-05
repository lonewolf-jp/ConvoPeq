# D162-2-G3 Evidence — S3 + V-D 同時再証明（S3=ON / V-D=ON・production 変更 0）

- Work item: D162-2-G3（D162-2-G ステージ 3/4・S3+V-D 同時の本命再証明）
- Date: 2026-09-04
- 基準ソース: **ConvoPeq.md `Generated: 2026-09-04 20:28:50` 相当の G2 適用済み working tree**
- production 変更: **0**（G2 tree をそのまま使用）
- 判定: **PASS**（§7 判定表）

---

## 1. 静的 preflight（指示 §1-2・全 ✓）

| 項目 | 結果 |
| --- | --- |
| 基準ソース | G2 適用済み working tree・git diff に G1/G2 以外の新規 production 変更なし |
| `destroyRolledBackDSP` 関数残存 | ✓（DSPLifetimeManager.cpp:149・rollback 経路 Orchestrator.cpp:292 のみが正当 caller） |
| **V-D caller の destroyRolledBackDSP = 0** | ✓（ReleaseResources.cpp 内はコメント 1 件のみ・実呼出 0） |
| S3 `shutdown-clear → retireRegisteredDSP()` 維持 | ✓（Orchestrator.cpp:632） |
| V-D active/fading が authority `retire()` のみ | ✓（ReleaseResources.cpp:543/:552） |
| `if (false && ...)` | ✓ src 全走査 **0 件** |
| S3 が `deferredSlot_.reset()` **前**に実行 | ✓（Orchestrator.cpp:622-633 retire block → :634 reset） |
| `runtimeDSPHandleMap_` erase が authority 内 | ✓（AudioEngine.h:4367） |

## 2. Gate ladder

| Gate | 結果 |
| --- | --- |
| G3-1 Build（Debug / Release / RWDI） | 全 **EXIT=0** |
| G3-2 CTest | Debug **40/40 PASS**（35.31s）/ Release **39/40**（AudioEngineHarness 0xC0000374 = G1/G2 と同一 pre-existing class・Test #36 HeadlessAudioPathVerification PASS） |
| G3-3 Debug 6-gen | **exit 0x00000000**・dump 0 |
| G3-4 RWDI 6-gen | **exit 0x00000000**・dump 0 |
| G3-5 RWDI 60-gen | **exit 0x00000000**・dump 0 |

## 3. S3 観測（分離報告）

### 3.1 カウント

| 項目 | Debug 6-gen | RWDI 6-gen | RWDI 60-gen |
| --- | ---: | ---: | ---: |
| `CLEAR_SHUTDOWN_DISPOSITION` | 1 | 1 | **5** |
| `shutdown-clear` retire 呼出 | 1 | 1 | 5 |
| shutdown-clear で retired=**1** | 0 | 0 | 0 |
| shutdown-clear で retired=**0**（no-op） | 1 | 1 | 5 |
| E-4d（timer-clear-midrun）retired=1 | 1 | 1 | 5（全て S3 対象と同一個体の先行 retire） |

### 3.2 個体単位 closure（60-gen・5 件全て）

| S3 target dsp | E-4d retired=1（先行） | shutdown-clear | destroy | released |
| --- | --- | --- | --- | --- |
| 000002B8D3C60080 | ✓ (retired=1) | no-op (retired=0) | ✓ | remaining=0 |
| 000002B8E09D0080 | ✓ | no-op | ✓ | remaining=0 |
| 000002B8F1909080 | ✓ | no-op | ✓ | remaining=0 |
| 000002B8F4D7C080 | ✓ | no-op | ✓ | remaining=0 |
| 000002B8F590B080 | ✓ | no-op | ✓ | remaining=0 |

**5/5 で CREATE → E-4d retire(retired=1・EBR Success) → CLEAR → CLEAR_SHUTDOWN_DISPOSITION
→ shutdown-clear retire(no-op・INV-D162-3) → D117_DESTROY → FOOTPRINT_RELEASED remaining=0**
の完全閉包。

### 3.3 「S3 が唯一の disposition 実行者（retired=1）」ケースについて

G1 と同様、本 G3 の 3 run でも **S3 単独 retired=1 のケースは未観測**。
全 S3 発火が「Timer C2/C3/C4 → requestDeferredClear → latch → RebuildThread E-4d retire →
shutdown clear で no-op 帰還」経路だった。これは soak 負荷プロファイル下では
deferred 保持 DSP が shutdown 境界を RebuildThread 生存のまま通過しないためと考えられる
（EmergencyDrain body も実行されている — G3 60-gen で
`EmergencyDrain phase completed in 6.2ms` を確認）。

→ **G3 PASS 判定には影響しない**（S3 block は実行時到達・安全帰還を 7 回累積実証・
disposition closure は E-4d との合成で 100% 達成）。ただし「S3 が retired=1 になる
変異（EmergencyDrain/C1 直接で slot 保持 DSP が RebuildThread 停止後に残留するケース）」は
引き続き未観測事項として記録する。この変異は C1 fallback / EmergencyDrain の発火条件が
subordinate であり、意図的に作らない限り production で到達しない構造である
（Timer C2/C3/C4 は RebuildThread 生存中に latch → E-4d が処理するため）。

## 4. V-D 観測（分離報告）

| 項目 | Debug 6-gen | RWDI 6-gen | RWDI 60-gen |
| --- | ---: | ---: | ---: |
| V-D active target count | 1 | 1 | **1** |
| V-D fading target count | 0 | 0 | 0 |
| V-D retire(retired=1) | **1** | **1** | **1** |
| V-D retire(retired=0) | 0 | 0 | 0 |
| V-D EBR enqueue | 1 (Success) | 1 (Success) | 1 (Success, epoch 235) |
| V-D D117_DESTROY | 1 | 1 | 1 |
| V-D footprint released | 1 (remaining=0) | 1 (remaining=0) | 1 (remaining=0) |
| **direct destroy** | **0** | **0** | **0** |
| stale map | 0 | 0 | 0 |

60-gen の V-D 個体閉包:
```text
[D162-2G2_VD_RETIRE] dsp=000002B8D19AB080 target=active-final
[D117_RETIRE]        dsp=000002B8D19AB080 retired=1
[D117_RETIRE]        dsp=000002B8D19AB080 enqueue=0 epoch=235
[D117_DESTROY]       dsp=000002B8D19AB080          ← ~AudioEngine enter (188386) 後 188393 = dtor D5/D8 内 digest
[DSP_FOOTPRINT_RELEASED] dsp=000002B8D19AB080 remaining=0
```

## 5. 両経路の独立性（指示 §5）

- **同一 run 内での同時成立**: 60-gen で S3 5 件 + V-D 1 件がそれぞれ独立に
  authority retire → EBR closure を達成。
- **異なる lifecycle 個体**: S3 対象 5 個体（000002B8D3C6/E09D/F190/F4D7/F590...080）と
  V-D 対象 1 個体（000002B8D19AB080）は **完全に異なる DSP** — 二重処分・cross 処分なし。
- **二重 destroy チェック**: `DSP_DESTROY_FOOTPRINT` の gen リストで重複 0 件
  （destroy が gen 単位で 1:1）— 二重処分の構造的・実行時証明。

## 6. 全体 closure（60-gen）

| 指標 | Debug 6-gen | RWDI 6-gen | RWDI 60-gen |
| --- | ---: | ---: | ---: |
| exit code | 0x00000000 | 0x00000000 | **0x00000000** |
| E-4 accounting | 2440 = 2436+3+**1** | 2709 = 2705+3+**1** | **23676 = 23633+34+5+4**（完全収支） |
| destroyed / released | 7 / 7 | 7 / 7 | **60 / 60** |
| residual | 0 | 0 | **0**（構築 gens 4-63 の 60 個体 = destroyed 60 個体・1:1 対応・重複 destroy 0） |
| EBR pend 最終 / 最大 | 0 / 1 | 0 / 1 | 0 / 2（運転中一時・運転内消化） |
| EBR overflow | 0 | 0 | **0** |
| E-3（INV-D162-8 violation） | 0 | 0 | **0** |
| Signature A / B / C | 0 / 0 / 0 | 0 / 0 / 0 | **0 / 0 / 0** |
| shutdown sequence complete | 1 | 1 | **1** |
| footprint released remaining | 全件 0 | 全件 0 | **全件 0** |
| XRUN | 0 | 3 | **26**（§6.1） |
| DC live / Priv | 収束 | 収束 | 1–3 収束 / **440MB**（F=445・E=461・G2=442 と同水準） |

### 6.1 XRUN 26 件の評価（新規クラスなし）

- Callback 最大 **2.15ms**（Expected=5.33ms に対する軽微な超過のみ）— G1=1.69ms 相当域・
  E baseline max 1.57ms / G2 max 1.87ms と **同一の callback jitter クラス**。
- Interval 8.0-8.7ms・Pressure=0・RetireDepth=0-1・shutdown 窓（VerifyDrained 前まで）で
  完結 — baseline（F 15 / E 17 / G1 9 / G2 17）との差は IR reload × intent burst の
  タイミング揺らぎによるもの（Gen=35 に 12 件集中 = burst 帯の jitter 集中）で、
  S3/V-D ON に起因する新規クラスは**認められない**。

## 7. G3 判定

| PASS 基準 | 結果 |
| --- | --- |
| S3 authority retire ✓ | **✓**（E-4d との合成で 5/5 closure・S3 block 実行時到達 7 件累積） |
| V-D authority retire ✓ | **✓**（retired=1 → EBR Success → destroy → remaining=0・3/3 run） |
| S3/V-D direct destroy = 0 | **✓** |
| EBR closure ✓ | **✓**（両経路とも destroy + remaining=0） |
| stale map = 0 | **✓** |
| E-4 accounting 完全収支 | **✓**（23,676 完全収支） |
| residual = 0 | **✓**（構築 60 = 破壊 60・1:1・二重 destroy 0） |
| E-3 = 0 | **✓** |
| Signature A/B/C = 0 | **✓** |
| Debug/RWDI 6-gen PASS | **✓** |
| RWDI 60-gen PASS | **✓** |

**D162-2-G3 = PASS。** G4（60-gen × 2 final deterministic soak）への進行条件を満たす。

## 8. 補足・残置事項

1. **S3 単独 retired=1 の未観測**（§3.3）: soak プロファイルでは E-4d が常に先行処理する
   ため S3 は no-op 帰還となる。この変異の実測には「RebuildThread 停止後に deferred 保持
   DSP が残留する異常系」の強制が必要で、production 到達経路が存在しない構造であること
   を含め、次監査での分類判断材料として記録する（G4 の blocker ではない）。
2. crash dump 2 件（ConvoPeq.exe.33364 / AudioEngineHarness.exe.24952・20:44）は
   Release CTest 由来の pre-existing（旧 build-icx binary static teardown AV /
   AudioEngineHarness 0xC0000374）— G3 soak 由来 0 件。
3. `ConvoPeq.md` は G2 tree 時点（20:28:50）のまま — G3 は変更 0 のため再生成不要。
