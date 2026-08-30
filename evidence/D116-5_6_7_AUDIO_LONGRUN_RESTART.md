# D116-5 / D116-6 / D116-7 evidence (Part 2 — Operational Validation続き)

**Date:** 2026-08-29
**前提:** `D116_OPERATIONAL_VALIDATION_AUDIT.md`（D116-1〜4, CONDITIONAL PASS）の続報
**方針:** 新規発見に対しても Phase-II 実装へ進まず、観測と記録に限定する

---

## D116-5 — Audio Quality Validation

### 自動化可能部分（客観観測・実デバイス 192kHz/1024samples）

| Run | 構成 | 結果 |
| --- | --- | --- |
| D2 | intent burst ×10（同一IR・content変化なし） | **9 publications**、異常 0、exit 0。publish 毎 gen/worldId 連番交替 |
| D3 | IR swapper（A/B/C 3種を4s周期でfile差替え）+ reload ×8 + intent burst ×8 | **7 publications**、**fingerprint 2種を確認**（`0xc7a79ef54e7b3e1d`, `0xfdd5894302e804e5`）= 異なるIR内容でのrebuild+publish成立、異常 0、exit 0 |

- 使用IR（生成資産）: `evidence/D116_irA.wav`（減衰noise burst）, `D116_irB.wav`（pre-delay付き長減衰）, `D116_irC.wav`（複音+noise tail）— md5互いに異なる
- **技術注記**: CLI telemetry モードでは `convolverParamsChanged` 通知が抑制されるため、reload単独では rebuild が発生しない（`[DIAG] convolverParamsChanged: suppressed while CLI telemetry mode is enabled`）。これはCLIモード固有の抑制であり、本検証では意図的に burst で rebuild を強制して IR 切替 publish を実現した。production UI 経路はこの抑制外。
- **同一IR reload の重複排除**（B-1, fingerprint一致 → intent 不発）を再確認。latest-wins ではない契約通りの動作。

### 実耳確認（A. Impulse / B. Sine / C. Music）→ **operator HOLD（未実施）**

人間の聴覚による click/pop/dropout/image shift 判定は本環境では実施不可能。客観代理指標としては:
- callback 処理時間: long-run 全期間で procTimeUsMax **最大 2653 µs < block budget 5330 µs**（余裕 ~50%）
- XRUN/underrun/dropout を示すテレメトリ: **0 件**（全 run 合計）
- crossfade 境界の sample-level 正当性は CTest の PublishPipelineIntegrationTests / T1-T4Measurement が担保（D116-2 で 40/40 PASS）

---

## D116-6 — Long-run Validation（6分28秒・continuous audio + 周期 publication + IR定期変更）

### 実行条件

- 継続時間 381s、IR swapper（A/B/C 6s周期）+ reload ×60 + intent burst ×60（6s間隔）
- **59 publications**、758 テレメトリサンプル、**AUTH_CONTRACT違反 / stall / XRUN / underrun / EMERGENCY / overflow / quarantine = 0 件**、exit 0

### メモリ時系列（2s間隔・191サンプル）→ **HOLD 該当**

| 時点 | Private Memory |
| --- | --- |
| 初期（10サンプル平均） | 612 MB |
| 中盤 | 5,165 MB |
| 終盤（10サンプル平均） | 9,409 MB |
| 最大 | 9,418 MB |
| **後半傾き** | **+1,453 MB/min（線形・単調増加）** |
| shutdown時 | 9,418 → 9,329 MB と解放開始（終了時に保持分を解放） |

**ユーザー定義 HOLD 条件「memory: 増加 → 増加 → 増加 → …」に該当 → D116-6 は HOLD。**

### 起因の切り分け（修正は行わない・観測のみ）

| Run | reload | burst/rebuild | IR内容変化 | メモリ推移 |
| --- | --- | --- | --- | --- |
| D2 | 無 | ×10 | 無 | **フラット**（2177MB 定常） |
| B-1 | ×6 | 無 | 無 | **フラット**（673MB） |
| Control | ×30 | ×30 | **無**（同一ファイル） | **増加**: 711 → 4,980 MB（+4,269MB/192s ≈ +1,334MB/min） |
| Long-run | ×60 | ×60 | 有（A/B/C循環） | **増加**: +8,797MB/381s（+1,453MB/min） |
| Restart cycles（×6） | 無 | ×3/cycle | 無 | **フラット**（823-824MB） |

**結論（観測事実）**:
1. 増加には「**IR reload と rebuild/publish の組合せ**」が必要。どちらか単独ではフラット。
2. **IR内容の変化は不要**（同一内容の reload+rebuild でも増加）→ fingerprint/内容差分の問題ではなく reload+rebuild 機構の滞留。
3. 増加率は **約 145 MB / (reload+rebuild pair)** で両実行間で一致（4269/30 ≈ 142、8797/59 ≈ 149）。
4. 保持されたメモリは **shutdown 時に解放が開始**される（プロセス終了で回収）。実行中の reclaim 進行は観測されず（drops 4回の小幅変動のみ）。
5. 増加中も音声処理は budget 内・異常テレメトリ 0 → **silent な memory 蓄積**（D116-8 の「overflow が silent loss になっていないか」の観点では、機能的損失は観測されず、滞留はメモリ側）。

**因果仮説（監査用・未検証）**: reload 経路で確保された IR/partition データが、rebuild により旧 world が retire された後も quarantine / deferred deletion / 旧 generation 参照経由で実行中に回収されず、shutdown drain まで保持されている。内部カウンタ（pendingRetireCount / quarantineResident）は診断ビルドが必要なため直接確認は未実施。

---

## D116-7 — Shutdown / Restart（6 cycles）

| Cycle | exit code | publications | perf samples | 異常 | max Private MB | 先頭cb avg µs | 末尾cb avg µs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 2 | 29 | 0 | 823.1 | 465.1 | 797.2 |
| 2 | 0 | 2 | 29 | 0 | 824.4 | 442.9 | 809.5 |
| 3 | 0 | 2 | 29 | 0 | 823.8 | 456.0 | 777.0 |
| 4 | 0 | 2 | 29 | 0 | 823.2 | 431.1 | 777.9 |
| 5 | 0 | 2 | 29 | 0 | 823.8 | 436.3 | 792.2 |
| 6 | 0 | 2 | 29 | 0 | 823.3 | 484.2 | 823.8 |

- **6/6 サイクルで startup → audio callback 稼働 → publication → shutdown 完了（exit 0）**
- max メモリが 823.1–824.4 MB の狭い帯域に収束 → **Cycle N の残留状態が Cycle N+1 に持ち越されていない**（プロセス分離 + 起動時メモリ一定）
- callback 処理時間もサイクル間で劣化なし
- **D116-7: PASS**

---

## D116 総合判定の更新

| Gate | 判定 |
| --- | --- |
| D116-5 Audio | **CONDITIONAL**（客観代理指標 PASS / 実耳確認 operator HOLD） |
| D116-6 Long-run | **HOLD** — reload+rebuild 継続時に ~145MB/pair のメモリ滞留（単調増加、実行中 reclaim なし、shutdown で解放） |
| D116-7 Restart | **PASS**（6/6、残留なし） |

**D116 全体: CONDITIONAL PASS から据え置き、ただし D116-6 のメモリ滞留が新規監査対象**（D117 相当の原因監査を推奨: 診断ビルドによる pendingRetireCount / quarantineResident / DSPHandleRuntime live count の直接時系列取得）。

- **commit は引き続き凍結**（D116-6 HOLD の解消または許容判断が確定するまで）
- icx crash（0xc0000005 ×3, `build-icx`）は別トラック凍結のまま（MSVC結果の横展開は禁止）
- Phase-II 要素（RecoveryEpisodeId / SemanticRecoveryTarget / supersession / MPSC化等）には**進んでいない**。本発見は観測のみであり、対応方針は原因監査後に契約ベースで決定すべきもの

## 生成物（Part 2）

- `evidence/D116_scenarioD_irswitch.log`, `D116_scenarioD2.log`, `D116_scenarioD3.log`
- `evidence/D116_longrun.log`, `D116_longrun_memory.csv`
- `evidence/D116_control_static.log`, `D116_control_static_mem.csv`
- `evidence/D116_restart_cycle{1..6}.log`, `D116_restart_cycle{1..6}_mem.csv`, `D116_restart_summary.txt`
- `evidence/D116_ir{A,B,C}.wav`, `D116_ir_swapper.py`, `D116_ir_swapper*.log`, `D116_restart_cycles.sh`, `D116_memory_sampler*.ps1`
