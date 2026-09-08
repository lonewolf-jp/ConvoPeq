# D166 — DS-F2 Device-Reconfigure Admission / Lifecycle Contract Audit（Work Report）

```text
D166 — DS-F2 Device-Reconfigure Admission / Lifecycle Contract Audit
Date: 2026-09-06
Type: read-only architectural / source audit（production/test/CMake/build.bat/tool 変更 0・binary rebuild 0）
Baseline: git 9cacee1f / ConvoPeq.md 実測 Generated 2026-09-05 12:02:18（指示記載 03:04:39 と差異・実ファイル値を採用）
         binary 72cce20a…3ad2（DIAG=ON）／S3=ON／V-D=ON
Verdict: **GO — Case A (Defect confirmed)・修復契約 = Option 2 目標 / Option 1 最小增量**
```

## 判定

> **Case A — DS-F2 は「device reconfigure 後の admission 永続閉鎖」として source-level で確定。**
> 3 系統の独立証拠: (1) packedState_ 書込は CAS 4 箇所のみで Closed→Open 遷移コード・reopen API ともに
> 0 件（INV-LIFE-9 が明示禁止）— reopen 経路不存在を証明。(2) closeAdmission の唯一 caller が
> releaseResources:87 で、releaseResources は JUCE 契約上 reconfigure でも呼ばれる
>（AudioEngineProcessor.cpp:55-67 委譲・isEnginePrepared guard は二重 pass 抑制のみ）。
> (3) D165 実測（DS 61/0・WA 120/120・gen 停滞）が state graph と機械的に一致。
> Case B（別経路による再 Open・別原因）は棄却。

## 主要発見

1. **先行監査との接続（最重要）**: DS-F2 は本監査で初発見ではなく、**D162-2-I1-D-R0（2026-09-05）が
   D-profile 実測とともに確定済み**（R0 §5「admission lifetime = engine lifetime と JUCE re-prepare 契約の衝突」
   ・PrepareToPlay.cpp:292-294 の明示コメント「publication 復活問題は I2 scope 外（R0 §5/§7-C/D）」）。
   **D166 は R0 §9-3 が「別監査（G-series 相当）」として明示 defer した監査の実施**であり、
   R0 の全構造判定（§2-§5・§7 A/B/C/D 比較・§9 GO 条件）を今回の全 grep で再確認・整合。
2. **releaseResources の caller semantics 確定**: JUCE AudioProcessor lifecycle（reconfigure と terminal
   shutdown が同一入口）／test harness stop()／~AudioEngine は呼ばない（独立 terminal 経路）。
   callee 側から terminal 性は未知 → 全 pass を terminal 扱いする現行実装が defect の核。
3. **prepareToPlay は operational session restart**: rebuild thread 再起動・generation reset・
   停滞監視 reset を実施（session restart の意図が明確）ながら ShutdownRuntime にのみ触れない
   （PrepareToPlay.cpp に shutdownRuntime_ 参照 0 件）— 意図的設計ではなく漏れと判定。
4. **telemetry 会計 defect の独立成立**: Build 経路のみ tryAdmit 失敗が完全無出力
   （RequestAccepted → 無出力消滅）。Publication 経路は RejectedShutdown を返す・Recovery は durable
   延期のため会計は成立。`Suppressed(AdmissionClosed)` 1 event 追加で修復可能（D167 併施可）。
5. **GUI も同一 state machine**: DeviceSettings.cpp の 4 site + settings window + **起動時 loadSettings
   （保存 setup ≠ 既定 device なら起動直後に switch）** — CLI 固有ではなく GUI 通常操作に構造適用。
6. **CoordinatorState との非対称**: intent-loop 側には ShuttingDown→Bootstrapping 復帰あり
  （ISRRuntimePublicationCoordinator.cpp:563-574）、admission/phase には復帰概念なし — 責務分離欠落が本質。

## 修復契約（3 案比較 → 選定）

| 案 | INV-LIFE-9 | 評価 |
| --- | --- | --- |
| Option 1: releaseResources を terminal/reconfigure 区別・reconfigure では terminal pipeline 不実行 | 保全 | **採用（first increment）** |
| Option 2: ShutdownRuntime=terminal 専用・DeviceRuntime（新 FSM）=reconfigure 責務 | 保全 | **採用（目標架构・段階導入）** |
| Option 3: Closed→Reinitializing→Prepared→Open | 反転 | **不採用**（Q7/G-H/ReclaimPermit identity 前提を崩す・R0 §7-C と同結論） |

D167 への入力: terminal 判定信号の設計・reader registration/publication 継続の意味論変化監査・
trace 清浄化（TV=7 消滅）の regression 条件を evidence 版 §10 に記録済み。

## GO 条件照合

NO-GO 条件 5 項目（caller semantics / session semantics / Closed→operational 別経路 / CLI≠GUI /
telemetry semantics）は**全て確定** → **GO**。次は Repair Contract approval → D167 minimal implementation
→ tests → CTest → targeted device-switch validation → D168 regression soak。

## 変更範囲（実測）

production source 0 / test 0 / CMake 0 / build.bat 0 / tool 0 / binary rebuild 0。
新規: evidence/D166/D166_DS_F2_ADMISSION_RECONFIG_AUDIT.md（本報告の正本・12 節構成）＋ doc/work88 報告。

## 成果物

- 正本: [evidence/D166/D166_DS_F2_ADMISSION_RECONFIG_AUDIT.md](C:\VSC_Project\ConvoPeq\evidence\D166\D166_DS_F2_ADMISSION_RECONFIG_AUDIT.md)
- 本報告: [doc/work88/D166_DS_F2_ADMISSION_RECONFIG_AUDIT_REPORT.md](C:\VSC_Project\ConvoPeq\doc\work88\D166_DS_F2_ADMISSION_RECONFIG_AUDIT_REPORT.md)
