# D165 — D162-2-I2 Long-run / Device-cycle Validation（Work Report）

```text
D165 — D162-2-I2 Long-run / Device-cycle Validation
Date: 2026-09-06
Type: operational validation（実装フェーズではない・コード変更 0）
Baseline: git 9cacee1f / G4 production tree（S3=ON・V-D=ON）
         build-diag RWDI binary sha256 72cce20a…3ad2（DIAG=ON・A0 で明示固定）
Verdict: **PASS + RESIDUAL**
次: 本 track（teardown/lifecycle validation）は D165 で closed。DS residual は別 defect として CR 化推奨。
```

## 目的

D164（I1 PASS）の「D120 teardown race 非再現」を、**より長い generation lifetime と複合再構成負荷**へ拡張検証する。
I0 §6 定義の「60-gen long soak + IR reload + rebuild burst + device cycle + repeated start/stop」を
I2-WA / I2-DS / I2-R×6 の 3 形態で実施。コード変更は 0（DS residual を修正してから検証しない）。

## 実施

| Gate | 内容 | 結果 |
| --- | --- | --- |
| A0 | 拡張 freeze（D164 教訓：binary hash に加え **CMake cache 診断フラグ**・compiler・S3/V-D source 現行確認を同一 freeze record に保存） | 完了（`d165_a0_freeze.json`・DIAG=ON 実測） |
| B | I2-WA 60-gen long soak 424s | exit 0x0・dump 0・TV=0・zone clean・**gen 4→64・61 obj 完全 closure** |
| B | I2-DS 60-gen long soak 428s | exit 0x0・dump 0・TV=7（既知 residual）・zone clean・closure 完全（2 obj） |
| B | I2-R1..R6 restart cycle（WA/DS 交互） | 6/6 exit 0x0・dump 0・zone clean（WA=TV0・DS=TV7 が決定論的） |
| D | 4-channel reconciliation（(pointer,generation) identity） | 8/8 全項目 ✓・benign pattern を log-level で再確認 |
| 判定 | PASS 基準 10 項目 | 全て ✓ → **PASS + RESIDUAL** |

## 4-channel 実測（8 runs）

exit（PowerShell `$p.ExitCode` authority）= 0x0×8 ／ new dump = 0×8（baseline 7 不変）／
shutdown zone = clean×8（T0/T1/T2 なし）／ lifecycle closure = per-object (pointer,gen) で
destroy 1:1・remaining=0・EBR ovf=0・final pend=0・dup destroy 0・stale HIT 0・E-3/Sig 0×8。
XRUN shutdown 窓 = 0（WA total 36 は全て startup transient）。

## ★ D165 での重要な新規発見 — DS-F2（rebuild-intake 完全停止）

I2-DS の 60-gen soak が **generation を 1 度も進められなかった**（gen 5 停滞・destroy 2 obj）：

```text
REBUILD_REQUESTED 61 件（全 decision=accepted）／ REBUILD_DISPATCHED 0 件／
rebuildThreadLoop 0 行（WA は 120 行）
```

root cause 連鎖（source 実読）：

1. `--cli-device-type DirectSound` → `MainWindow.cpp:457` `setCurrentAudioDeviceType`
2. JUCE が旧 device（WA）close → **AudioEngine::releaseResources() pass 1**（SHUTDOWN_BEGIN 前）
3. pass 1 内 `ReleaseResources.cpp:87` `closeAdmission()` → packedState_ = Closed
4. `prepareToPlay(DirectSound)` で rebuild thread 再 start・generation reset（`PrepareToPlay.cpp:87/97`）
5. 以後の burst intent 60 件は telemetry 上 `REQUESTED accepted` まで記録されるが、
   `RebuildDispatch.cpp:319` `if (!shutdownRuntime_.tryAdmit(1)) return;` で**無出力で握り潰し**（Dropped 未記録）
6. shutdown 時 pass 2 で backward transition 7 件が guard reject = **transitionViolations=7（DS-F1）**

つまり **DS-F1（D164 発見）と DS-F2（D165 発見）は同一 root cause**（二重 releaseResources +
ShutdownRuntime admission/phase に re-prepare 用の reset 経路が存在しない —
`reopenAdmission`/`resetForRestart` 相当は grep で 0 件）。影響の違い:

| | DS-F1（D164） | DS-F2（D165・本発見） |
| --- | --- | --- |
| 現象 | shutdown 時 backward transition 7 件 reject | device switch 後、**rebuild が一切走らない**（IR reload/burst 無効化） |
| 性質 | telemetry-only | **機能的**（shutdown 後再 prepare 系の全経路で発生し得る） |
| lifecycle への影響 | なし | なし（既存 DSP の closure は完全） |

## 判定

> **D165 = PASS + RESIDUAL。**
> D120 teardown race の非再現は I2 workload（60-gen + burst + device cycle + restart）まで拡張確定
> （crash 0・dump 0・zone clean×8・lifecycle closure 8/8）。DS residual（DS-F1/F2）は
> teardown/lifecycle integrity に干渉しないため判定を変更しない。

**本 track は D165 で closed。** DS-F2 は teardown ではなく device-reconfigure 機能の不備として
新規 defect（CR 候補）に切り出すこと。GUI 上の device switch でも同様に rebuild が停止する可能性が
あり、要調査（本監査範囲外）。修復方針（参考）: `prepareToPlay` 時の ShutdownRuntime
phase/admission reset、または releaseResources を真の shutdown と reconfigure で区別する設計 —
いずれも D165 では実装しない（コード変更 0 遵守）。

## 変更範囲（実測）

production source 0 ／ test 0 ／ CMakeLists 0（H5 +10 のまま）／ build.bat / tool 0 ／
CMakeCache 変更 0（D164 で復元済の値を freeze 確認のみ）。
新規: evidence/D165/ 配下（freeze json・runner ps1・reconciliation json・trace json×8）+ D165_*.log×8。

## 成果物

- 正本: [evidence/D165/D165_I2_LONGRUN_VALIDATION.md](C:\VSC_Project\ConvoPeq\evidence\D165\D165_I2_LONGRUN_VALIDATION.md)
- [d165_a0_freeze.json](C:\VSC_Project\ConvoPeq\evidence\D165\d165_a0_freeze.json)・[d165_b_reconciliation.json](C:\VSC_Project\ConvoPeq\evidence\D165\d165_b_reconciliation.json)・[d165_soak.ps1](C:\VSC_Project\ConvoPeq\evidence\D165\d165_soak.ps1)・d165_trace_*.json×8
- run log: evidence/D165_{WA,DS,R1..R6}.log×8
