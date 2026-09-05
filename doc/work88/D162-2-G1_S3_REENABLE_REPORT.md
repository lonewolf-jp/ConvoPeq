# D162-2-G1 Work Report — S3 Staged Re-enable

- Work item: D162-2-G1（D162-2-G ステージ 1/4・S3=ON / V-D=OFF）
- Date: 2026-09-04
- 基準ソース: ConvoPeq.md `Generated: 2026-09-04 18:29:10` + G1 変更
- 判定: **PASS**（詳細: evidence/D162-2G1_S3_REENABLE_EVIDENCE.md）

## 0. Executive Summary

`RuntimePublicationOrchestrator::clearDeferredForShutdown()` に S3 terminal disposition
（`retireRegisteredDSP(req, "shutdown-clear")`）を挿入し、S3 を EBR authority 経由で
再有効化した。**Debug 6-gen / RWDI 6-gen / RWDI 60-gen すべて exit 0x00000000・crash dump 0・
residual 0・S3 disposition closure 完全達成**。D0 crash signature（Signature A）は全 run で
0 件 — F 修正の維持を回帰確認済み。V-D は未接触。

## 1. 変更ファイル数と diff

| 項目 | 内容 |
| --- | --- |
| 変更 production ファイル数 | **1**（`src/audioengine/RuntimePublicationOrchestrator.cpp`） |
| 変更関数 | `clearDeferredForShutdown()` のみ（他関数は E-4c 計装由来の既存 diff を含むため git diff 上の行数は 83+/21- だが、G1 の新規変更は本関数内 1 block + 注記更新） |
| 実質 diff | (a) `deferredSlot_.reset()` 直前に `CLEAR_SHUTDOWN_DISPOSITION` DIAG + `retireRegisteredDSP(req, "shutdown-clear")` を挿入、(b) 既存 `event=CLEAR` ログから誤記 `(no disposition — pre-existing mismatch)` を除去、(c) D162-2-B 残課題コメントを G1 状態に更新 |
| 触れていないもの | ReleaseResources.cpp / V-D / destroyRolledBackDSP / tryShutdownQuiescentReclaim / EBR / AudioSegmentBuffer.h / AlignedAllocation.h / DSPLifetimeManager / E-3 / S1/S2/S4 / テスト |

ownership semantics・E-4 accounting の新規追加はなし（指示どおり既存 authority の
呼び出し追加のみ）。

## 2. Gate ladder 結果

| Gate | 結果 |
| --- | --- |
| G1-1 Build（Debug / Release / RWDI） | 全 **EXIT=0** |
| G1-2 CTest | Debug **40/40 PASS** / Release **39/40**（AudioEngineHarness 0xC0000374 = D162-1P/B/E/F と同一 pre-existing・従来どおり除外 → 39/39 相当 PASS） |
| G1-3 Debug 6-gen | **exit 0x00000000**・dump 0 |
| G1-4 RWDI 6-gen | **exit 0x00000000**・dump 0 |
| G1-5 RWDI 60-gen | **exit 0x00000000**・dump 0 |

## 3. 主要観測値（60-gen）

| 指標 | 値 |
| --- | --- |
| `CLEAR_SHUTDOWN_DISPOSITION` 件数 | **6**（> 0 条件を充足） |
| `shutdown-clear` retire 件数 | 6（全て E-4d 済み DSP への二重呼び → `retired=0` no-op） |
| E-4d retire（`timer-clear-midrun`） | 6（全て `retired=1`・`enqueue=0` Success） |
| destroy 件数 | 61 = F baseline 60 + 1（**S3 無効時に破壊されなかった 1 件の解消**） |
| residual | **0** |
| EBR pend / ovf | 最終 0 / 0（pend 最大 2 は運転中一時・運転内消化） |
| E-3（INV-D162-8）DIAG | **0 件** |
| E-4 会計 | CREATE 28,456 = CONSUME 28,409 + OVERWRITE 40 + CLEAR 6 + DISCARD 1（完全収支） |
| Signature A / B / C | **0 / 0 / 0** |
| XRUN | 9 件（Callback ≤1.21ms 程度・F baseline 15 / E 17 を下回る・新規クラスなし） |
| DC live / Priv | 1-3 収束 / 382→593MB（F=445MB・E=461MB と同オーダー） |

Debug 6-gen / RWDI 6-gen も同様に S3 発火（1 件ずつ）→ EBR Success → destroy closure。

## 4. disposition 閉包の実測（6/6 完全閉包）

```text
CREATE → CLEAR_MIDRUN_DISPOSITION → retire(timer-clear-midrun, retired=1)
       → EBR enqueue Success → CLEAR → CLEAR_SHUTDOWN_DISPOSITION
       → retire(shutdown-clear, retired=0 no-op: INV-D162-3 実行時検証)
       → D117_DESTROY（EBR digest・dtor body 内）→ FOOTPRINT_RELEASED remaining=0
```

60-gen の 6 件（gen 42/46/48/54/59/64）全てで `remaining=0` を確認。

## 5. crash dump 分類

| dump | 由来 | 分類 |
| --- | --- | --- |
| ConvoPeq.exe.3668.dmp（19:16） | Release CTest HeadlessAudioPathVerification が起動した**旧 build-icx Release binary**（8/18 ビルド）の static teardown 0xC0000005 — cli-smoke-test.ps1 が toleration する既知クラス | pre-existing・G1 無関係 |
| AudioEngineHarness.exe.30912.dmp（19:17） | Release AudioEngineHarness 0xC0000374 | pre-existing（D162-1P/B/E/F 文書済み） |
| G1 soak 3 run | — | dump 0 件 |

## 6. 観測事項（G2/G3 への引き継ぎ）

1. **S3 が『唯一の disposition 実行者』になるケースは未観測**: 今回の全 S3 発火は
   「E-4d midrun retire 直後の二重呼び出し（no-op 帰還）」経路だった。G1 gate 条件
   （S3 block の実行時到達 + 安全帰還）は充足したが、EmergencyDrain/C1 直接経由で
   slot 保持 DSP が S3 の retired=1 になる変異は G3 前に追加観測する（G2 には影響なし）。
2. `[FAULT] coordinator Faulted` / `drain timeout` / `RECOVERY action=3` は過去全ログに
   存在する pre-existing（B/C/D0/E/F と同数）— G1 由来ではない。
3. Release CTest の HeadlessAudioPathVerification は build-icx 旧 binary を自動選択する
   既知挙動（cli-smoke-test.ps1 候補リスト）。dump 混入防止のため将来の候補リスト更新を
   別途検討（今回の Gate 判定には影響せず）。

## 7. 判定と次工程

**D162-2-G1 = PASS。** ユーザー推奨どおり **G2 は V-D-b（`destroyRolledBackDSP` →
`DSPLifetimeManager::retire` authority 化 + `if (false &&` 解除）** で進める。
G1 変更は維持したまま V-D のみを有効化する（G2 ladder は G1 と同一）。
