# D116 — Operational / Deployment Validation (D116-1 〜 D116-4) evidence

**Date:** 2026-08-29
**対象:** Release executable の運用条件検証（Phase-I baseline の operational 延長）
**前提:** D115-A PASS（D113-A comment-only 変更が実行コードへ非侵入）を commit-ready 拡大解釈しない

---

## D116-1 — Pre-flight / Environment Freeze（read-only）

| 項目 | 値 |
| --- | --- |
| Git HEAD | `956bda6f2385e11e13d45f0e2de7e0b52f5d26e5`（2026-08-28 02:18:37 +0900 "commit"） |
| Working tree | 42 エントリ（下記分離参照） |
| D114 baseline | Debug 40/40, Release 40/40（`evidence/D114_PHASE_I_OPERATIONAL_VALIDATION_AUDIT.md`） |
| D115-A evidence | `evidence/D115-A_POST_CHANGE_READ_ONLY_DIFF_AUDIT.md`（8/8 PASS） |
| D113-A diff | `evidence/D113A_AudioEngine.h.diff`（単一 hunk, コメントのみ） |
| Build system | CMake + Ninja Multi-Config（`CMAKE_GENERATOR:INTERNAL=Ninja Multi-Config`） |
| Compiler | MSVC `cl` via VS 18 Enterprise `vcvars64.bat` + IPP 2026.1 include（`tools/build_with_vcvars.bat`） |
| JUCE | 8.0.12（CMakeLists.txt v0.6.10） |
| OS | Windows 11 Home, NT 10.0.26200.0 |
| Audio devices | Bravo-HD USB Audio Device / NVIDIA HD Audio / High Definition Audio / VB-Audio Voicemeeter VAIO（いずれも Status=OK） |
| Runtime diagnostics | `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF`（D114 baseline と同一構成） |

### 変更の分離（D116-1 GO 条件）

- **D113-A（本検証対象）**: `src/audioengine/AudioEngine.h` — CacheMap dtor コメント hunk のみ（D115-A で実行コード同一性証明済み）
- **D113/D114 期の既監査済み未コミット変更**: `AudioEngine.RebuildDispatch.cpp`, `ISRRuntimePublicationCoordinator.{h,cpp}`, `RuntimePublicationOrchestrator.cpp`, `ISRSemanticValidationTests.cpp`, `doc/work88/I4_DESIGN_CONTRACT.md`, `ConvoPeq.md`（ソース統合一時ファイル）
- **非ソース変更**: `.gitignore`, `AGENTS.md`, `tools/*.bat`, `.zcode/`, `evidence/*`, build ログ削除
- **GO条件判定**: D113-A 以外の新規変更なし ✅ / unexpected source/test change = 0 ✅ / environment reproducible ✅（build手順・CTest手順は D114 と同一 wrapper 使用）

---

## D116-2 — Release Build Validation

| 項目 | 結果 |
| --- | --- |
| Release build | **PASS** — `tools/build_with_vcvars.bat Release nopause`、exit 0、error/FAILED 0 件、`build\ConvoPeq_artefacts\Release\ConvoPeq.exe` 生成（47,778,304 bytes, 2026-08-29 00:52） |
| Release CTest | **PASS 40/40（100%）** — Total 32.10 sec。D114 baseline（40/40）を再現 |

- ログ: `evidence/D116_release_build.log`, `evidence/D116_ctest_release.log`
- failure 時の修正不作業ルールは発動せず（全 PASS）

---

## D116-3 — Runtime Scenario Test（CLI 自動化による実行）

実行方式: アプリ実装の CLI 自動化フラグ（`--cli-run`, `--cli-log-file`, `--cli-exit-ms`, `--cli-ir`, `--cli-intent-burst-count/interval-ms`）を使用。オーディオデバイス実オープン・192kHz/1024 samples/block で音声コールバック稼働。

### Scenario A — Cold Start（2 回実施）

| 観測項目 | 実行1 | 実行2 |
| --- | --- | --- |
| crash | なし | なし |
| hang（Responding=False） | なし | なし |
| publication failure | なし | なし |
| audio callback 稼働 | ✅ callbacks 進行, sampleRateHz=192000, blockSamples=1024 | 同左 |
| 処理時間 | procTimeUsAvg ≈ 424-622 µs / block budget 5.33 ms（≈ 10%）、max 1.68 ms | 同左 |
| 自動終了 | `[CLI] Auto-exit flush: shutting down` → **exit code 0** | **exit code 0** |
| メモリ | 起動直後 26 MB → 初期割当後 ~520 MB で一定 | — |

- ログ: `evidence/D116_scenarioA.log`, `evidence/D116_scenarioA2.log`, メモリ: `evidence/D116_scenarioA_memory.csv`
- **注記**: 実行1 を bash バックグラウンドジョブ経由で起動した際、ジョブ終了時に bash が "Segmentation fault" を表示したが、(a) フォアグラウンド再実行で exit code 0、(b) Windows イベントログに `build\` バイナリの WER 記録（Event 1000）なし、の両方により **bash ジョブ制御のアーティファクトと判定**（アプリ crash ではない）。判定根拠を含め記録済み。
- 音質観測（click/pop 等）は実耳確認が必要 → **HOLD 項目**（D116-5、未実施）

### Scenario B — Repeated Publication

**B-1（同一IR reload ×6）**: 同一内容のため rebuild intent 不発（hash/fingerprint 同一 → 1 publication のみ）。**設計通りの重複排除動作**を確認（latest-wins ではなく fingerprint 一致時は意図自体が発生しない）。

**B-2（rebuild intent burst ×12, interval 2s）**:

| 観測項目 | 結果 |
| --- | --- |
| publication 数 | **11 回**（seq=5..15, worldId=5..15, gen 一致） |
| publishDurationUs | 900–2597 µs（RT コールバック外、異常遅延なし） |
| current/fading world 遷移 | gen/worldId が連番で正常交替 |
| AUTH_CONTRACT 違反 | **0**（`[AUTH_CONTRACT] FAIL` 行なし） |
| stall / XRUN / underrun / quarantine / EMERGENCY / overflow | **0 件** |
| メモリ | 初期立上げ後 ~2177.5 MB で**反復publication中フラット**（単調増加なし）、shutdown で 2089.1 MB へ減少開始 |
| 自動終了 | **exit code 0** |

- ログ: `evidence/D116_scenarioB.log`, `evidence/D116_scenarioB2.log`
- 状態収束（pendingRetireCount → 0 等）の内部カウンタは診断ビルド限定出力のため、外部観測では「メモリフラット + 異常テレメトリ 0 件 + CTest 収束計測テスト PASS」で担保（下記 D116-4）

---

## D116-4 — Memory / Retire Observation

取得済み時系列（2s 間隔、Private Memory MB）:

| Run | 初期 | 定常 | 終了時 | 単調増加 |
| --- | --- | --- | --- | --- |
| Scenario A（12s, publish 1回） | 26 → 516 | 520 一定 | 520 | **なし** |
| Scenario B-1（IR reload ×6） | 76 → 690 | 673 一定 | 585（shutdown 解放） | **なし** |
| Scenario B-2（intent burst ×12 → 11 publish） | 218 → 2177 | 2177.5 一定 | 2089（shutdown 解放開始） | **なし** |

収支判定: publish rate（B-2: 11回/約25s）に対しメモリが定常フラット → **retire → reclaim 収支が均衡**（retire 蓄積による liveCount 単調増加なし）。HOLD 発動条件（retire ↑ / reclaim ≈ 0 / liveCount ↑ の継続）は不成立。

**限界（正直な記録）**: `pendingRetireCount`, `quarantineResident`, `reclaimCount` の時系列そのものは `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF` の Release バイナリでは出力されない（diagLog flush がマクロゲート）。内部カウンタ収束は以下で代替担保:
- Release CTest の実行時計測テスト: ISRSoakTests（256-stress + 7 endurance）, WorldRetirementMeasurementTests, ShutdownRetireIntentDrainTests, RetireGraceSemanticsTests — **すべて PASS**（D116-2）
- 診断ビルドによる直接観測は **D117 起点の推奨事項**（別ビルド構成のため D116 baseline バイナリは保存）

---

## D116 PASS 条件表（現時点の判定）

| Gate | PASS 条件 | 判定 |
| --- | --- | --- |
| Build | Release build PASS | **PASS** |
| Tests | 40/40 PASS | **PASS** |
| Startup | crash/hang なし | **PASS**（2 回 cold start, exit 0） |
| Publication | repeated publish 安定 | **PASS**（11 回連続, seq/gen/worldId 整合） |
| Retire | backlog 異常増加なし | **PASS**（外部観測・メモリフラット）※診断ビルド直接観測は未実施 |
| Reclaim | reclaim が進行 | **PASS**（メモリ定常・shutdown 解放確認）※同上 |
| Memory | 長時間単調増加なし | **PASS**（最大 42s 観測範囲）※3min+ は未達 |
| Audio | click/pop/dropout なし | **HOLD**（実耳確認未実施・D116-5） |
| Overflow | silent loss なし | **PASS**（quarantine/EMERGENCY/overflow 0 件） |
| Shutdown | 完全 Drain / Reclaim | **PASS**（Auto-exit flush → exit 0, メモリ解放）※多重 restart は D116-7 で未実施 |
| Restart | 残留状態なし | **未実施**（D116-7） |
| Phase-II isolation | Phase-II code change = 0 | **PASS**（本検証は観測のみ・コード変更なし） |

## 総合判定: **CONDITIONAL PASS（自動化可能範囲）** — HOLD 項目 3 件

> **【2026-08-29 続報】** D116-5/6/7 を実施（`evidence/D116-5_6_7_AUDIO_LONGRUN_RESTART.md`）。
> D116-7 Restart は **PASS**（6/6 cycles、残留なし）。D116-6 Long-run で **新規 HOLD 該当事象を検出**:
> IR reload + rebuild の組合せ継続時に **約145MB/pair のメモリが実行中に回収されず単調増加**
> （6分で +8.8GB、+1.45GB/min。burst単独・reload単独・restart cycles はフラット。shutdown で解放）。
> 処理時間・異常テレメトリ・AUTH_CONTRACT は全期間正常。**原因監査（D117 相当）まで commit 凍結を継続。**

1. **D116-5 音質確認**（impulse/sine/music/IR switching での実耳確認）— 実機オーディオでの人間聴覚判断が必要（客観代理指標は PASS）
2. **D116-6 長時間試験**（3min 以上の continuous audio + repeated publication）— 本検証は最長 42s
3. **D116-7 Shutdown/Restart 反復**（複数回 start→operate→shutdown→start の残留状態確認）

加えて **別件観測事項（D116 外・要調査）**: Windows イベントログに本日 0:05:58 / 0:27:48 / 0:52:48 の 3 件の `ConvoPeq.exe` クラッシュ（例外コード 0xc0000005、フォールトオフセット 0x1f39140 で同一、**`build-icx\ConvoPeq_artefacts\Release\ConvoPeq.exe`**）を確認。D116 baseline バイナリ（`build\`）とは別の icx ビルドであり D116 判定には影響しないが、**shutdown 時 access violation の可能性**があるため icx ビルドの root-cause 監査を推奨（D116 では修正に入らない）。

## 生成物

- `evidence/D116_release_build.log` / `evidence/D116_ctest_release.log`
- `evidence/D116_scenarioA.log`, `D116_scenarioA2.log`, `D116_scenarioA_memory.csv`
- `evidence/D116_scenarioB.log`, `D116_scenarioB2.log`, `D116_scenarioB_memory.csv`
- `evidence/D116_memory_sampler.ps1`, `D116_memory_sampler_poll.ps1`（観測用具）
- 本ファイル

**commit は未実施**（D116 完了・監査まで凍結継続）。
