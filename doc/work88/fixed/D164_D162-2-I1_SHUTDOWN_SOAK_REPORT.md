# D164 — D162-2-I1 Corrected Operational Shutdown Soak（Work Report）

```text
D164 — D162-2-I1 Corrected Operational Shutdown Soak
Date: 2026-09-06
Type: operational validation（実装フェーズではない・コード変更 0）
Baseline: git 9cacee1f / G4 production tree（S3=ON・V-D=ON）/ build-diag RWDI DIAG binary
Verdict: **PASS — D120 historical teardown race は I1 profile で再現せず（26/26）**
次: D165 = I2 Long-run / Device-cycle Validation（I0 §6 定義）
```

## 目的（I0 方針の維持）

D120-7 で観測された probable teardown race（zone T1 主容疑）を「修正済みと証明する」のではなく、
**旧 D120 crash profile を現行 binary で再実行して再現有無を測定する**。I0 §3.3 の分類
（historical probable → current 非再現）を結論としても維持する。

## 実施と結果

| Gate | 内容 | 結果 |
| --- | --- | --- |
| A0 | baseline freeze（git/binary/CrashDumps=7/CLI flag 定義/S3・V-D 現行 src 確認） | 完了 |
| A0-drift | **binary config drift 発見** — 初回 26 runs の binary が RUNTIME_DIAGNOSTICS=OFF（H5 reconfigure が ON を消失）→ channel 4 観測不能 → 初回分 invalid_v1 へ quarantine | 検出・記録 |
| A0r | DIAG=ON 復元（build-diag cache フラグのみ・source/CMake 0）+ rebuild + smoke（MEM_SNAP 119） | 回復確認 |
| A1 | Profile A plain ×8 | **8/8 PASS** |
| A2 | Profile B IR+rebuild ×6（D120 で 3/3 crash の最重要） | **6/6 PASS** |
| A3 | Profile C IR reload×3+burst ×6 | **6/6 PASS** |
| A4 | Profile D device cycle ×6（Windows Audio/DirectSound 交互・CLI_AUDIO_DEV_SWITCH 実測） | **6/6 PASS** |
| B | 4-channel reconciliation（exit/dump/shutdown trace/lifecycle closure） | 26/26 全項目 ✓ |
| C | PASS/STOP 判定 | **PASS** |

4-channel 全項目（26 runs）: exit 0x0×26（PowerShell `$p.ExitCode` authority）・新規 dump 0・
zone clean×26（T0/T1/T2 ゼロ）・E-3 0・Signature A/B/C 0・EBR ovf 0・final pend 0・
residual（remaining）0・per-object destroy 1:1・duplicate destroy 0・registered DSP direct destroy 0・
forbidden stale-map HIT 0・XRUN shutdown 窓 0（76 件全て startup transient）。

lifecycle closure の積極証拠: V-D retire（target=active-final）全 run・S3 shutdown-clear retire
（B/C 各 run 1 件・origin=shutdown-clear）・retire→enqueue→destroy→remaining=0 の連鎖確認。

## 判定

> **D164 = PASS。** 現行構成では D120 historical teardown race は I1 profile において再現しなかった。
> 絶対的不存在の証明ではない（I0 §3.3 維持）。zone T1（mainWindow.reset() 内 crash）は 1 回も観測されず。

**分岐: I1 PASS → D165（I2 Long-run / Device-cycle Validation）へ。**

## 副次発見（D165/運用への引き継ぎ）

0. **★ DirectSound 二重 releaseResources pass（transitionViolations=7・新規 observation）**:
   `--cli-device-type DirectSound` の実切替（WA→DS）が旧 device close を介して起動時に
   releaseResources pass 1 を走らせ、shutdown 時 pass 2 の 7 backward 遷移がガードに却下
   （ちょうど 7・決定論的・WA/B/C/A は 0）。root = ShutdownRuntime phase の re-prepare reset 経路なし
   （ISRShutdown.h:335）。**影響 telemetry-only**（2nd pass でも drain 完走・closure 完全・exit 0x0）、
   D120 型 race とは無関係。I1-D が Windows Audio 固定だったため初観測。
   → residual register 候補。修正（phase reset）は次編集 window/D165 scoping 判断・本作業では実装 0。
   詳細 = evidence/D164/D164_I1_SHUTDOWN_SOAK_EVIDENCE.md §4a。
1. **A0 freeze 項目の穴**: binary hash だけでは測定機構の同一性を担保できない。
   **CMake cache の診断フラグ値（CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS 等）を freeze 必須項目化**すべき。
   H5 のような「build-diag を触る作業」の直後は特に（D162-1P_diag_build.bat が文書化済みの
   cache-wipe pitfall と同一系統）。
2. **origin ラベル mojibake**（`D162-2B_RETIRE origin=桳瑵…`="shutdown-clear"）は G4 以前からの
   pre-existing cosmetic（%s ラベルの UTF-16/8 変換）。機能影響なし・STOP 非該当。修正は
   次の編集 window の doc/comment 系に混ぜて可（優先度低）。
3. analyzer の gen 1:1 判定は `[PUBLISH] seq` マーカー単独では不十分（startup/intermediate gen を
   網羅しない）。per-object (pointer, gen) accounting が正（本監査で訂正済み）。

## 変更範囲（実測）

production source 0 / test 0 / CMakeLists 0（H5 +10 のまま）/ build.bat 0 / tool 0。
build-diag cache の DIAG フラグ OFF→ON 復元のみ（派生 build tree・G4 config への一致）。

## 成果物

- evidence/D164/D164_I1_SHUTDOWN_SOAK_EVIDENCE.md（本報告の正本・per-run 表含む）
- evidence/D164/d164_a0_freeze.json（drift 記録 + 復元 binary hash）
- evidence/D164/d164_b_reconciliation.json（26 runs 4-channel 明細）
- evidence/D164/d164_diag_restore_build.bat（復元手順）
- evidence/D164/invalid_v1/（初回 26 log・quarantine 保存）
- evidence/D164_soak.ps1（runner・G4 .ps1 の exit capture 部流用）+ evidence/D164_{A,B,C,D}_run*.log×26 + D164_smoke.log
