# D162-2-G3 Work Report — S3 + V-D 同時再証明（本命 Gate）

- Work item: D162-2-G3（D162-2-G ステージ 3/4・S3=ON / V-D=ON・**production 変更 0**）
- Date: 2026-09-04
- 基準ソース: ConvoPeq.md `Generated: 2026-09-04 20:28:50` 相当の G2 適用済み working tree
- 判定: **PASS**（詳細: evidence/D162-2G3_JOINT_REPROOF_EVIDENCE.md）

## 0. Executive Summary

G2 適用済み source（S3=ON / V-D=ON）を **production 変更 0 のまま**、同一 run 内で
S3 と V-D の両 disposition 経路が独立に authority retire → EBR closure まで成立することを
再証明した。**3 soak（Debug 6-gen / RWDI 6-gen / RWDI 60-gen）すべて exit 0x00000000・
crash dump 0**。60-gen では S3 対象 5 個体 + V-D 対象 1 個体が全て完全閉包
（destroy 60 = 構築 60・1:1・二重 destroy 0・residual 0）。Signature A/B/C・E-3 違反・
direct destroy・stale map は全て 0 件。

## 1. 静的 preflight（全 ✓）

- `destroyRolledBackDSP`: 関数残存（rollback 経路のみ caller）・**V-D caller 0 件**
- S3 `shutdown-clear → retireRegisteredDSP()` 維持（slot reset **前**に実行）
- V-D active/fading とも authority `retire()` のみ・`if (false &&` は src 全走査 0 件
- `runtimeDSPHandleMap_.erase` は authority 内（AudioEngine.h:4367）
- git diff に G1/G2 以外の production 変更なし

## 2. Gate ladder 結果

| Gate | 結果 |
| --- | --- |
| G3-1 Build（Debug/Release/RWDI） | 全 **EXIT=0** |
| G3-2 CTest | Debug **40/40**・Release **39/40**（AudioEngineHarness 0xC0000374 = pre-existing のみ・新規失敗なし） |
| G3-3 Debug 6-gen | **exit 0x00000000**・dump 0 |
| G3-4 RWDI 6-gen | **exit 0x00000000**・dump 0 |
| G3-5 RWDI 60-gen | **exit 0x00000000**・dump 0 |

## 3. 両経路の同時成立と独立性（最重要観測）

### 3.1 分離カウント（60-gen）

| 項目 | S3 | V-D |
| --- | ---: | ---: |
| target count | **5**（active-final は V-D 側のみ） | **1**（active-final） |
| retired=**1** | 5（全て E-4d timer-clear-midrun 由来） | **1**（V-D authority retire 直接） |
| retired=0（no-op） | 5（S3 shutdown-clear の二重呼び防護） | 0 |
| EBR enqueue Success | 5 | 1（epoch 235） |
| D117_DESTROY / FOOTPRINT_RELEASED | 5 / 5（remaining=0） | 1 / 1（remaining=0） |
| direct destroy | **0** | **0** |
| stale map | 0 | 0 |

### 3.2 独立性の実証

- S3 対象 5 個体（000002B8D3C6/E09D/F190/F4D7/F590...080）と V-D 対象 1 個体
  （000002B8D19AB080）は**完全に異なる lifecycle 個体** — cross 処分なし。
- `DSP_DESTROY_FOOTPRINT` の gen 重複チェックで **二重 destroy 0 件**
  （構築 gens 4-63 の 60 個体が 1:1 で破壊・residual 0）。
- V-D 破壊は `~AudioEngine: enter` 後の dtor body D5/D8 drain 内（INV-D162-8 適合）。
- S3 破壊は E-4d retire の EBR enqueue を運転中に消化（shutdown 境界を跨がず E-3=0）。

### 3.3 残置観測（G4 blocker ではない）

**S3 が retired=1 の唯一 disposition 実行者になるケースは G1/G2/G3 を通じて未観測。**
soak プロファイルでは Timer C2/C3/C4 が RebuildThread 生存中に必ず latch → E-4d で
処理するため、S3 は no-op 帰還となる（EmergencyDrain body 自体は実行確認済み）。
S3 block の実行時到達・安全帰還は 7 件累積で実証済みであり、disposition closure は
E-4d との合成で 100% 達成。変異の実測には RebuildThread 停止後の deferred 残留という
production 到達経路のない異常系の強制が必要。

## 4. 全体 closure（60-gen）

| 指標 | 実測 |
| --- | --- |
| exit code | **0x00000000** |
| E-4 accounting | CREATE 23,676 = CONSUME 23,633 + OVERWRITE 34 + CLEAR 5 + DISCARD 4（**完全収支**） |
| residual | **0**（destroy 60 = 構築 60・gens 4-63 1:1・二重 destroy 0） |
| EBR pend 最終 / ovf | 0 / **0** |
| E-3 / INV-D162-8 | **0 件** |
| Signature A / B / C | **0 / 0 / 0** |
| shutdown sequence complete | 1 |
| XRUN | 26 件 — Callback ≤2.15ms（Expected 5.33ms）の既知 jitter クラス・shutdown 窓 0 件・baseline（F 15/E 17/G1 9/G2 17）と同クラスで新規性なし（Gen=35 burst 帯への集中 = タイミング揺らぎ） |
| DC live / Priv | 1–3 収束 / 440MB（F=445・E=461・G2=442 同水準） |

## 5. 判定

**D162-2-G3 = PASS。** ユーザー判定基準の全項目（S3/V-D authority retire・direct destroy 0・
EBR closure・stale map 0・E-4 完全収支・residual 0・E-3 0・Signature A/B/C 0・
6-gen/60-gen PASS）を満たした。

**次工程: D162-2-G4** — S3=ON / V-D=ON のまま production 変更 0 で
RWDI 60-gen × 2 final deterministic soak（lifecycle 突合・決定論確認）を実施する。
