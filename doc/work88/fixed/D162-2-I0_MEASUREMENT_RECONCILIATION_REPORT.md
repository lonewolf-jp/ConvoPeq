# D162-2-I0 Work Report — Operational Validation Readiness / Measurement Reconciliation

- Work item: D162-2-I0（D116/D120 旧測定の廃棄・再定義。**read-only / production 変更 0 / test 0 / build 0 / soak 0**）
- Date: 2026-09-04
- Baseline: ConvoPeq.md `Generated: 2026-09-04 20:28:50`
- 判定: **PASS / GO** — 詳細: evidence/D162-2-I0_MEASUREMENT_RECONCILIATION.md

## 0. Executive Summary

D116 系の shutdown exit-code 測定を**正式に廃止**し、D120 probable teardown race を
「historical probable — 未再確立」に分類した。新規診断コードは不要（既存 D123/D127-E trace
で crash phase 3-zone 分類が可能）であり、I1（Corrected Operational Shutdown Soak）の
測定契約・Profile A-D・PASS/STOP 基準を固定した。

## 1. D116 測定の正式廃止

| 過去記録 | 問題 | 処理 |
| --- | --- | --- |
| D116-7 restart 6/6「exit 0」 | `D116_restart_cycles.sh:10` が `cmd //c "... & echo %ERRORLEVEL%"` — `%ERRORLEVEL%` は parse-time 展開のため常に事前値 0 を表示（測定無効） | **正式廃止** |
| D116-OP2 `APP_EXIT=0` | 実行 block（OP2 report :43）に exit capture 構造が文書化されていない | **非検証扱い**（D162-1P report も「タイミング依存」と記録済み） |
| D120-7 probe（exit 139/0） | 測定修正済み（D120-9 明記） | **有効**（historical evidence） |
| G1-G4 の `EXITCODE=0x…` | PowerShell `$p.ExitCode` 正規 capture | **有効** |

**効力**: 今後の shutdown exit code 評価は本監査で確定する Operational Validation Contract
にのみ基づく。旧 bash `%ERRORLEVEL%` パターンの使用を禁止する。

## 2. D120 race の分類（解決済みとは宣言しない）

> **Historical probable teardown race — current G4 configuration has not reproduced it,
> but operational recurrence has not yet been re-established with corrected measurement.**

- Historical: frozen binary で plain 6s 2/5 crash・IR+rebuild 12s 3/3 crash（logger 切断後で phase 不観測）。
- Current: G4（S3/V-D ON）60-gen × 2 は clean だが、profile shape が D120 の短時間反復と異なる。
- 構造事実: D162-2-G により仮説 (d)「破棄漏れ DSPCore」は解消（S3/V-D authority 化）。
  仮説 (a)-(c) は未検証のまま → I1 の Profile A/B が直接の再現試験になる。

## 3. Operational Validation Contract（確定）

- **4 channel**: ① exit code（PowerShell 正規 capture）② crash dump（`%LOCALAPPDATA%\CrashDumps` 巡查）③ shutdown trace（既存 D123 3 行）④ lifecycle closure（DIAG イベント突合・G4 手法）。
- **shutdown trace zone 分類**: T0（teardown 前 crash）/ **T1（`mainWindow.reset()` 内 crash = D120-7 主容疑 zone）** / T2（teardown 後・logger 切断前）/ clean。新規診断コード不要（MainApplication.cpp:179-193 に既存・G4 log で動作実測）。
- **測定項目**: exit code / crash 三系統判定 / zone / lifecycle 個体閉包 / generation 1:1 / EBR pend=0・ovf=0 / registered DSP direct destroy 0 / stale map（MISS/HIT 分類・処分済み HIT は禁止）/ E-3 0 / Signature A/B/C / XRUN（shutdown window・Pressure・Callback max を記録・件数単独で判定しない）/ memory（DC live・Priv・residual）。

## 4. I1 実行条件（固定）

| Profile | 内容 | 回数 |
| --- | ---: | ---: |
| A: plain shutdown | startup → audio → clean shutdown | ×8 |
| B: IR+rebuild+shutdown | D120 crash 頻度最高条件の再現 | ×6 |
| C: repeated IR/rebuild | 短時間反復 reload → shutdown | ×6 |
| D: device open/close/reopen | restart cycle 形式 | ×6 |

- binary: `build-diag RWDI`（G4 と同一 config）・baseline 20:28:50 を開始時再確認。
- PASS: 全 run が全合格条件を満たす。STOP: exit≠0 / dump / zone T0-T2 / E-3≠0 / Signature≠0 / EBR 異常 / residual≠0 / gen mismatch / dup destroy / registered direct destroy / 処分済み HIT — 1 回でも発生したら該当 profile 即停止・原因分類へ。
- 合計 ~26 run ≒ 8 分。A×8・B×6 で 0 crash なら D120 rate（2/5・3/3）に対する合理的水準の非再現記録になる。

## 5. 判定

**D162-2-I0 = PASS / GO。** I1（Corrected Operational Shutdown Soak）の実施条件が成立。
I2（Long-run / Device-cycle Validation）は I1 PASS 後。DEFER 項目（H0 確定分）には触れない。
