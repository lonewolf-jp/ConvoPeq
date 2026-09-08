# D168 — Regression Validation（Work Report）

```text
Task:    D168 — D167 修復後の regression validation（CLI / GUI / WA / DS / restart / long-run 統合）
Date:    2026-09-06
Type:    runtime validation（production 変更 0・D167 実装済み binary のみで実施）
Binary:  build-diag RWDI sha256 14875cafb9be5a05（D167 実装入り・DIAG=ON）
Baseline: D167 DS 24/24 gen2→10 TV=0・D165 DS 61/0 gen5停滞 TV=7
Verdict: **全 run PASS — DS-F2 修復は長時間 workload でも安定・regression なし**
```

---

## 1. Runs 一覧

| Run | 内容 | ExitCode | Dump | TV | 時間 |
| --- | --- | --- | --- | --- | --- |
| WA long-run | Windows Audio・IR reload 60 + intent burst 60・420s | 0x0 | 0 | **0** | 424s |
| DS long-run | DirectSound switch + IR reload 60 + burst 60・420s | 0x0 | 0 | **0** | 426s |
| R1-R6 | WA/DS 交互 restart cycle ×6 | 全 0x0 | 全 0 | 全 **0** | 34-36s each |
| GUI | 保存設定（Voicemeeter ASIO 192kHz）での起動時 restore・90s | 0x0 | 0 | **0** | 94s |

## 2. WA long-run（switch なし対照）

- REBUILD_REQUESTED 119 / DISPATCHED 118（差 1 は latest-wins merge 窓内の collapsed）
- generation 62 まで進行（D165 WA 120/120 と同型の正常系維持）
- terminal pass 1 回のみ・reconfigure pass 0 回（switch なしのため正しい）
- XRUN 13 件 = 既知クラス（起動トランジント・Interval 8ms 系）
- メモリ: 381→445MB で plateau（leak なし）

## 3. DS long-run（D165 DS-F2 failure signature の長時間反転）★

| 指標 | D165 I2-DS | **D168 DS** |
| --- | --- | --- |
| REQUESTED / DISPATCHED | 61 / **0** | **121 / 120** |
| switch 後の dispatch | 0 | **118** |
| generation | 5 停滞 | **60 まで進行** |
| TV | 7 | **0** |

- reconfigure pass 1 回（switch 時）+ terminal pass 1 回（終端のみ）— D167 境界の正常動作。
- `admission_closed` telemetry 0 件 = reconfigure 経路の drop が構造的に消滅（D166 §9 予測どおり）。
- メモリ: 379→749MB（sample 190 = rebuild 最中の peak）→ **595MB で plateau**。
  DSP churn に相関する一時的増加であり、60-gen 完了後は安定。D165 は rebuild が
  動いていないため 281MB であった — 差分は 60 世代分の rebuild workload 相当。
- XRUN 1 件（Gen=6・Interval 1808ms = IR reload 直後の既知 transient）。

## 4. R1-R6 restart cycle（WA/DS 交互 ×6）

- 全 6 run: exit 0x0・dump 0・TV=0・phase=ShutdownComplete。
- WA run（R1/3/5）: REQ 6 / DIS 6・gen 7 まで進行・switch 1 回（CLI 前置 switch）。
- DS run（R2/4/6）: REQ 6-7 / DIS 6・gen 3 まで進行・reconfigure pass 1 回・switch 1 回。
- R6 の REQ 7 / DIS 6 差分 1 は exit 直前 3 秒窓内の intent（telemetry 記録後に
  exit 到達・会計としては terminal suppress 対象外の merge 窓残）— 異常ではない。
- restart を跨いでも admission / lifecycle が毎回正常に立ち上がることを 6 連続で実証。

## 5. GUI 起動時 restore 経路（D166 §7 の検証）

保存設定（R6 終了時に Voicemeeter ASIO 192kHz が保存済み）で GUI 起動:

- **起動直後に保存 device が復元され audio callbacks が 90 秒間流動**
  （`CLI_PERF_RAW callbacks=... sampleRateHz=192000.0`・`expected=5333` = 192kHz/1024 相当）。
- world publish（rev=4・worldGen=2-4）・rebuild(req/queued)=5/5 完走。
- 終端は terminal pass 1 回のみ・TV=0・exit 0x0・dump 0。
- メモリ 382→396MB plateau（leak なし）。
- **reconfigure pass 0 回** — 起動時 restore は「save 時 device ≠ 起動初期 default」でも
  `loadSettings` が engine 起動前に `initialise()` する順序のため engine の
  releaseResources を通らない（D166 §7 の想定「起動直後 switch」は JUCE device
  初期化が engine 起動前に完結する実装順序であり DS-F2 経路に到達しないことが実測で確定）。
  engine が release/prepare を経るのは以後の GUI device 変更時のみ = D167 境界で保護済み。

## 6. 既知残差（pre-existing・regression ではない）

1. `~AudioEngine` dtor の `[FAULT] coordinator in Faulted state` ログ 1 行:
   D165_DS/WA にも存在する既知残差（D162-2 台帳の residual intents 系）。
   D168 全 run で 1 行のみ・実害（exit code / dump / TV）なし。
2. XRUN: 既知クラス（IR reload 直後 transient）のみ。

## 7. D167 exit criteria 追補照合（D168 分）

```text
[D168-1] WA long-run:      dispatch 継続・gen 62・TV=0・plateau ✅
[D168-2] DS long-run:      121/120・gen 60・TV=0 ✅（D165 反転の長時間確認）
[D168-3] restart cycle ×6: 全 0x0/TV=0 ✅
[D168-4] GUI startup:      90s callbacks・TV=0 ✅（DS-F2 経路非到達も実測確定）
[D168-5] regression:       D165 PASS 項目の非退行なし ✅
```

## 8. 成果物

- runner: [evidence/D168/d168_soak.ps1](C:\VSC_Project\ConvoPeq\evidence\D168\d168_soak.ps1)
- logs: evidence/D168_{WA,DS,R1..R6,GUI}.log（18k-38k 行 ×9）
- traces: evidence/D168/d168_trace_*.json ×8（全 TV=0）
- 本報告: doc/work88/D168_REGRESSION_VALIDATION_REPORT.md

**結論: DS-F2 修復（D167）は targeted / long-run / restart / GUI の全軸で安定。
D162-2-I2 long-run track は閉包。残課題は D167 報告 §8 の 2 件（in-flight rebuild ×
terminal race・duplicate-prepare collapse abort）を別 track（D169 候補）へ起票。**
