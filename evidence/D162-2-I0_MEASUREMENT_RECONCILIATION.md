# D162-2-I0 — Operational Validation Readiness / Measurement Reconciliation（read-only）

```text
Date:        2026-09-04
Type:        read-only audit / production source 変更 0 / test 0 / CMake 0 / build 0 / CTest 0 / soak 0
Baseline:    ConvoPeq.md Generated 2026-09-04 20:28:50（G2 tree・G3/G4/H0 で変更 0 のまま・唯一の baseline）
Position:    D162-2-H0 PASS/GO を受けた実運用検証の準備監査。I1（Corrected Operational Shutdown Soak）の
             測定契約を固定する。I0 では soak を実行しない。
参照:        evidence/D120_LIFECYCLE_AUDIT.md（§D120-7/9）・ evidence/D116_restart_cycles.sh /
             evidence/D116_restart_summary.txt / doc/work88/D116_OP2_OPERATIONAL_VALIDATION.md /
             src/MainApplication.cpp:177-195 / G4 証跡（evidence/D162-2G4_*）
```

---

## 0. 総合判定

> ## **D162-2-I0 = PASS / GO**
>
> 1. **D116 系 exit-code 測定を正式に廃止**（restart 6/6「exit 0」は `%ERRORLEVEL%` の
>    parse-time 展開により常に 0 を表示 = 測定無効。OP2 の `APP_EXIT=0` も capture 方法が
>    文書化されておらず非検証扱い）。
> 2. **D120 probable teardown race は「解決済み」と宣言しない** — historical evidence と
>    current evidence を分離し、I1 で修正済み測定による再確立を行う。
> 3. **新規診断コードは不要** — D123/D127-E shutdown trace が現行 source に既存
>    （MainApplication.cpp:177-195）で、G4 soak log で動作実測済み。crash phase を
>    3 zone に分類できる。
> 4. **Operational Validation Contract を確定**（exit code / dump / shutdown trace /
>    lifecycle closure の 4 系統観測 + I1 Profile A-D + PASS/STOP 基準）。
> 5. I1 実施条件が全て揃った → **GO**。

---

## 1. Baseline

- `ConvoPeq.md Generated: 2026-09-04 20:28:50`（H0 で FRESH 確認済み・本監査で変更なし）
- production / test / CMake / build 変更: **0**

## 2. D116 測定の正式廃止（I0-1 前提）

### 2.1 廃止対象 1: restart 6/6「exit 0」（D116-7）

**証拠（evidence/D116_restart_cycles.sh:10）**:

```bash
APPEXIT=$(cmd //c "build\ConvoPeq_artefacts\Release\ConvoPeq.exe ... & echo %ERRORLEVEL%" | tr -d '\r')
```

- `cmd /c` の `& echo %ERRORLEVEL%` は**コマンドライン解析時に展開**されるため、
  アプリ終了後の実際の exit code ではなく常に**事前値（0）**を表示する。
- `D116_restart_summary.txt` の 6 cycle 分 `exitcode=0` は全てこの問題を含む
  → **測定として無効・正式廃止**（D120-7 の訂正記録を本監査が正式化）。

### 2.2 廃止対象 2: OP2 `APP_EXIT=0`（D116-OP2）

- `doc/work88/D116_OP2_OPERATIONAL_VALIDATION.md:43` の実行 block は
  `cmd //c "...exe --cli-run ..."` で **exit code の capture 構造が文書内に存在しない**
  （`& echo %ERRORLEVEL%` すらない）。「APP_EXIT=0」（:54）の取得方法は記録されておらず
  **非検証データ**として扱う。
- 補強: `doc/work88/D162-1P_DIAGNOSTIC_ATTRIBUTION_REPORT.md:104` が既に
  「D116-OP-2 当時は APP_EXIT=0 で非発現 = タイミング依存」と記録しており、
  D162-1P 時代に Release 起動/終了 0xC0000005 が再現していた事実と整合。

### 2.3 廃止の範囲と効力

| 過去記録 | 処理 |
| --- | --- |
| D116-7 restart 6/6 exit 0 | **廃止**（測定無効） |
| D116-OP2 APP_EXIT=0 | **非検証扱い**（capture 方法文書なし） |
| D116 系の「crash/hang なし」評価 | exit-code 根拠部分は無効。MEM_SNAP / telemetry 系の観測は有効（別 channel） |
| D120-7 の probe（exit 139/0 分類・修正済み測定） | **有効**（historical evidence として維持） |
| G シリーズ（G1-G4）の `EXITCODE=0x…` | **有効**（PowerShell `$p.ExitCode` による正規 capture） |

**廃止の効力**: これ以降、shutdown exit code の評価は本監査で確定する
Operational Validation Contract（§4）にのみ基づく。旧 bash `%ERRORLEVEL%` パターンの使用を禁止。

## 3. D120 probable teardown race の分類（I0-2）

### 3.1 Historical evidence（D120-7・frozen production binary）

```text
plain 6s × 5:        exit 139, 139, 0, 0, 0        → 2/5 crash
IR+rebuild 12s × 3:  exit 139, 139, 139            → 3/3 crash
```

- crash は `[CLI] Auto-exit flush: shutting down` **後**に発生。当時は直後に
  `setCurrentLogger(nullptr)` で logger が切断され crash phase を観測できなかった
  （全 run 最終行が同一になる理由）。
- IR+rebuild ありで頻度増 → teardown 時の DSPCore/NUC/queue 残存量と正相関の示唆。
- 仮説候補（未検証のまま）: (a) RetireRouter drain と coordinator loop 停止の順序、
  (b) asyncSink logging と teardown、(c) audio device close と DSPCore destroy の HB、
  (d) 破棄漏れ DSPCore の shutdown reclaim 競合。

### 3.2 Current evidence（D162-2-G4・60-gen × 2）

- exit 0x00000000 × 2 / dump 0 / residual 0 / generation 1:1 / E-3 0 / Signature A/B/C 0 /
  direct destroy 0 / stale MISS。lifecycle closure は決定論的に成立。
- **ただし G4 profile は 60-gen × 6s interval の長尺 run であり、D120 が crash を再現した
  「短時間 plain / 短時間 IR+rebuild × 反復」profile とは shape が異なる。**

### 3.3 分類（指示どおり）

> **Historical probable teardown race — current G4 configuration has not reproduced it,
> but operational recurrence has not yet been re-established with corrected measurement.**

- 「D120 crash は解決済み」とは**宣言しない**。
- 解決の可能性を支持する構造事実（D162-2-G の効果）: 無処分 deferred DSP 1 件/run の解消
  （S3）、最終 published DSP の authority destroy（V-D）→ D120-7 仮説 (d) の
  「破棄漏れ DSPCore」が構造的に消滅。ただし仮説 (a)-(c) は検証されていない。
- I1 で再現測定を行い、再現の有無を初めて確定する。

## 4. Operational Validation Contract（I0 の本体）

### 4.1 測定系統（4 channel・全て既存機構のみ）

| # | Channel | 方法（I1 で固定） | 根拠（現行実装） |
| --- | --- | --- | --- |
| 1 | **Exit code** | PowerShell `Start-Process -PassThru -Wait` + `$p.ExitCode -band 0xFFFFFFFF` を `EXITCODE=0x%08X` で記録。**旧 bash `%ERRORLEVEL%` パターン禁止** | G シリーズ全 soak で使用・実績済み |
| 2 | **Crash dump** | 各 run の前後で `%LOCALAPPDATA%\CrashDumps` の mtime 巡查 → 新規 `ConvoPeq.exe.*.dmp` を run に帰属。WER Event 1000 は補助にしない（D120: 記録されない case 多数） | 同 dir で実績（G1-G4 期の dump が全てここに生成） |
| 3 | **Shutdown trace** | log の最終 3 行ブロック: `SHUTDOWN_BEGIN` → `SHUTDOWN: mainWindow.reset() completed` → `LOGGER_DETACH / SHUTDOWN_END`（§4.2 の zone 分類） | **既存 D123/D127-E trace**（MainApplication.cpp:179/:190/:193・diagnostic only・新規コード不要） |
| 4 | **Lifecycle closure** | DIAG log から `[D162-2E_DEFERRED]` 7 イベント収支 / `[D162-2B_RETIRE]` origin 別 / `[D117_RETIRE]` retired・enqueue / `[D117_DESTROY]` / `[DSP_DESTROY_FOOTPRINT]` gen / `[DSP_FOOTPRINT_RELEASED]` remaining / `[MEM_SNAP]` pend・ovf・DC・Priv | G シリーズ解析手法の踏襲 |

### 4.2 Shutdown trace zone 分類（crash phase 特定）

log 末尾の trace により crash phase を即時分類できる（D120 時に不可能だった観測）:

| zone | log 状態 | 意味 |
| --- | --- | --- |
| T0 | `[CLI] Auto-exit flush` はあるが `SHUTDOWN_BEGIN` なし | teardown 開始前（CLI 経路）で crash |
| **T1** | `SHUTDOWN_BEGIN` あり / `mainWindow.reset() completed` なし | **`mainWindow.reset()` 内（~MainWindow / ~AudioEngine / teardown 全体）で crash — D120-7 の主容疑 zone** |
| T2 | `reset completed` あり / `LOGGER_DETACH` なし | teardown 完了後・logger 切断前の crash |
| clean | 3 行全て出現 | shutdown sequence 完了（その後の crash は exit code / dump でのみ検出） |

- zone T1 を観測した run は D120-7 仮説の直接検証データになる（さらに前行の
  `[D162-2E]`/`[D117_*]`/`INV-D162-8` で shutdown 内位相を特定可能）。
- zone clean で exit ≠ 0 の場合: logger 切断後の static teardown 系 — dump 解析必須
  （Signature A/B/C 分類へ）。

### 4.3 測定契約（I1 で毎 run 記録する項目）

| 項目 | 合格条件 / 記録内容 |
| --- | --- |
| Exit code | `0x00000000` のみ合格 |
| Crash 判定 | process exit code + WER/dump dir 巡查 + harness/log 結果の**三系統**で判定 |
| Shutdown trace | 3 行ブロック完全出現（zone clean）・zone T0/T1/T2 は STOP トリガ |
| Lifecycle | CREATE → DISPOSITION → RETIRE → EBR → DESTROY → FOOTPRINT_RELEASED の個体閉包 |
| Generation | destroyed gen の連続性・1:1・重複 0（G4 と同一手法） |
| EBR | `[MEM_SNAP]` 最終 pend=0 / ovf=0 全 run |
| Direct destroy | registered DSP について 0（`[D162-2B_DESTROY]` 0 件・DSPGuard/rollback 未登録分は契約内で許容・個数記録） |
| stale map | `retireByHandle` の MISS / HIT を分類記録。**HIT は「未処分 DSP の正常 authority destroy」のみ許容**（gen 1:1 で裏取り）・処分済み DSP への HIT = 禁止 |
| E-3 | `INV-D162-8` DIAG 0 件 |
| Signature A/B/C | A（AudioSegmentBuffer→aligned_free→mkl_free）/ B（detectStuckReaders jassert）/ C（その他）を dump・log で分類 |
| XRUN | 件数に加え **shutdown window の有無 / Pressure 値 / Callback max** を記録（件数単独では判定しない） |
| Memory | `[MEM_SNAP]` DC live / Priv / NUC alloc・peak / residual（`remaining≠0` 0 件） |

## 5. I1 実行条件（Profile A-D）

**前提**: G4 production tree（S3=ON / V-D=ON）を**変更せず**使用。binary は
`build-diag RWDI`（G4 と同一 config — lifecycle closure の観測可能性を確保）。
`ConvoPeq.md` baseline が 20:28:50 から変わっていないことを I1 開始時に再確認。

| Profile | 内容 | CLI 例（既存 flag のみ） | 回数 |
| --- | --- | --- | --- |
| **A: plain shutdown** | startup → audio run → clean shutdown（D120 の plain 6s 相当） | `--cli-run --cli-log-file … --cli-exit-ms 15000` | **×8** |
| **B: IR + rebuild + shutdown** | IR load → rebuild → shutdown（D120 の 12s 相当・crash 頻度最高だった条件） | `--cli-run --cli-ir evidence\D162-1P_active.wav --cli-intent-burst-count 3 --cli-intent-burst-interval-ms 2000 --cli-exit-ms 15000` | **×6** |
| **C: repeated IR/rebuild** | 短時間で IR reload × 反復 → shutdown | `--cli-run --cli-ir … --cli-ir-reload-count 3 --cli-ir-reload-interval-ms 3000 --cli-intent-burst-count 3 --cli-intent-burst-interval-ms 3000 --cli-exit-ms 18000` | **×6** |
| **D: device open/close/reopen** | device cycle → shutdown（restart cycle 形式。`--cli-device-type` で切り替え・プロセス反復） | Profile A を device-type 変えて連続実行（各 run が 1 device cycle） | **×6** |

- 合計 ~26 run × 15-18s ≒ 8 分程度。
- 連続実行間隔 2s（D116 restart の反復条件を維持）。
- 各 run で §4.3 の全項目を記録（スクリプト化時は G4 soak .ps1 を雛形に exit capture 部だけ流用）。
- **D120 との統計比較**: A 2/5・B 3/3 の過去 crash rate に対し、A×8・B×6 で 0 crash なら
  「historical race の非再現」を有意に近い形で記録できる（完全証明ではなく合理的水準）。

### 5.1 I1 PASS 条件

- 全 run が §4.3 の全合格条件を満たす（exit 0x0 / dump 0 / zone clean / lifecycle closure /
  generation 1:1 / EBR 0 / direct destroy 0（registered）/ E-3 0 / Signature A/B/C 0）。
- XRUN: shutdown window 0 件・Pressure=0・Callback max が既知 jitter 域（≤ ~3ms・Expected
  5.33ms）— 件数は分布記録のみ。
- 上記が Profile A-D 全てで成立 → **I1 PASS**。

### 5.2 I1 STOP 条件（1 回でも発生したら該当 profile 即停止・原因分類へ）

```text
exit != 0
crash dump generated（ConvoPeq.exe 新規）
SHUTDOWN_BEGIN without reset-completed（zone T1）
E-3 != 0
Signature A/B/C != 0
EBR overflow > 0
final EBR pending != 0
residual != 0
generation mismatch
duplicate destroy
registered DSP direct destroy
stale-map HIT at forbidden terminal path（処分済み DSP への HIT・gen 1:1 破れ）
```

- STOP 時は dump + symbolize + zone/phase 特定 → D162-2 の residual register か
  新規 defect かを分類してから再開判断（自動的に次 profile に進まない）。

## 6. I2 への接続（本監査では実施しない）

I1 PASS 後、長時間 soak（60-gen class）・IR reload・rebuild burst・device cycle・
repeated start/stop を組み合わせた Long-run / Device-cycle Validation（I2）に進む。
Practical Stable ISR Bridge の運用評価軸（Retire→Epoch 通過 / shutdown 完全 Drain /
Overflow がデータ喪失に直結しない / Coordinator=Authority）を I2 の判定軸として引き継ぐ。

## 7. 今はやらないもの（H0 判定の維持）

RecoveryEpisodeId / Supersession / E_max・O_max・Phase-II / build-error telemetry counter /
MKLFailure・ConvolverFailure・PrepareFailure 生成経路 / CW-8 production caller 強制接続 /
S3 standalone retired=1 の人工生成 / Release CTest pre-existing 0xC0000374 の D162-2-G への
混同 — 全て DEFER / residual のまま。**I0/I1 で触れない。**

## 8. GO / NO-GO

| 条件 | 該当 |
| --- | --- |
| 旧測定の廃止根拠が確定 | ✓（restart_cycles.sh `%ERRORLEVEL%` + OP2 capture 文書なし） |
| crash 判定の 3 channel が現行環境で機能 | ✓（PowerShell exit capture・CrashDumps dir 実績・D123 trace 動作実測） |
| shutdown phase 観測が既存 code で可能 | ✓（D123/D127-E 3 行 trace・新規診断コード不要） |
| lifecycle closure 測定手法 | ✓（G シリーズで確立済みを契約化） |
| I1 の profile / PASS / STOP が固定 | ✓（§5） |
| production / test 変更 | 0 |

# **判定: D162-2-I0 = PASS / GO — I1（Corrected Operational Shutdown Soak）実施条件が成立。**
