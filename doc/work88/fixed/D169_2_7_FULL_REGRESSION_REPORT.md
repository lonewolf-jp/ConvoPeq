# D169-2 — Full Regression & Track Closure（Work Report）

```text
Task:    D169-2-7 — Full Regression（+ D169-2 close）
Date:    2026-09-07
Type:    full regression（production 変更 0）
Verdict: **D169-2-7 PASS → D169-2 CLOSED**
```

---

## 1. D169-2-7 結果（P1〜P10 全 PASS）

| Gate | 結果 |
| --- | --- |
| P1 source/diff boundary | PASS — production 変更は `PrepareToPlay.cpp` +17 のみ（D169-2-4）・**ISRLifecycle.cpp/.h 0 行**・test deviation は文書化済み 2 件のみ・unexpected diff 0 |
| P2 binary freshness | PASS — 3 config 全 rebuild・`[OK]` に依存せず failure pattern 直接走査 0 件・**exe mtime > obj mtime ×2**（D169-2-6 の LNK1285 取りこぼし教訓 gate） |
| P3 targeted consistency | PASS — D169-2-5（×4 collapse no-op）・D169-2-6（×50 restart cycles）を fresh binary ×3 config で再現 |
| P4 full CTest | PASS — **Debug 40/40・Release 40/40・RWDI 40/40** |
| P5 lifecycle regression | PASS — positive（same SR/BS → collapse → Prepared 維持）/ negative（SR・BS 変更 → 通常 prepare）/ other paths（Uninitialized/Released → Preparing → Prepared）に branch 侵入なし |
| P6 side-effect regression | PASS — **25/25 NOT EXECUTED**（gen/pub/world/placeholder/latency/analyzer/crossfade/uiConvolver/rebuildIntent/leavePrepare 全不変） |
| P7 device restart | PASS — 150 cycles baseline（D169-2-6）+ full suite 同一 binary PASS |
| P8 shutdown integrity | PASS — admission Closed + ShutdownComplete・collapse は state を残さない（RC-2） |
| P9 known hazard | PASS — MEM_SNAP dangling 参照は pre-existing / separate track として分類・Full Regression 中の新規 AV 0 → STOP 条件不発 |
| P10 最終判定 | **PASS** |

## 2. D169-2 Track 総括

```text
D169-2-1  Contract audit     Case A 確定（collapse×body×leavePrepare 契約不整合・0xC0000409）
D169-2-2  Contract audit     RC-D169-2-1〜7 + RC-1〜10 固定（候補 a 採用）
D169-2-3  Preflight          P1〜P5 全 GO（判別子一意性・挿入位置・JUCE 契約・25 side effect 列挙）
D169-2-4  Implementation     PrepareToPlay.cpp:20 直後 collapse no-op 分岐 1 箇所（+17 行）
D169-2-5  Targeted           PASS（collapse ×4・side effect 全不変を runtime 観測）
D169-2-6  Stress             PASS（restart chain 相当 50 cycles ×3 config・150 cycles）
D169-2-7  Full Regression    PASS（40/40 ×3 config・freshness gate・25/25 NOT EXECUTED）
────────────────────────────────
D169-2 Overall             **CLOSED**
```

- production change: `AudioEngine.Processing.PrepareToPlay.cpp` の collapse detection
  1 箇所のみ（new state / sync primitive / authority = 0）
- test infra deviation（文書化済み 2 件）: collapse regression + restart stress の
  2 テスト追加 + `startAudioOnly` seam（coverage gap R11 の恒久解消を含む）
- defect 系統の closure: D169-1（physical lifetime / D170 で修復）と D169-2
  （prepare transaction / 本 track で修復）が D167 §8 の 2 新規課題として閉包

## 3. 残存記録（D169-2 scope 外）

1. **MEM_SNAP sampler dangling 参照**（`Timer.cpp:1079-1088` — `getActiveRuntimeDSP()` の
   pointer slot 値を `collectTrackedMemoryStatistics()` に渡す経路）: pre-existing hazard /
   separate track 起票候補。timing 依存 flaky AV の潜在（D169-1 family の別 consumer）。
2. `~AudioEngine` dtor の `[FAULT] coordinator in Faulted state` ログ 1 行: 既知残差
   （D165/D168 から存在・実害なし）。
3. build script の errorlevel 取りこぼし（LNK1285 系）: 手順で binary freshness gate を
   適用済み。恒久対応（bat 改修）は tool 変更となるため別判断。

## 4. 成果物

- 正本: [evidence/D169/D169_2_7_FULL_REGRESSION.md](C:\VSC_Project\ConvoPeq\evidence\D169\D169_2_7_FULL_REGRESSION.md)
- 本報告: doc/work88/D169_2_7_FULL_REGRESSION_REPORT.md
- logs: evidence/D169/d169_2_7_build_*.log / d169_2_7_harness_*.log / d169_2_7_ctest_*.log

## 5. 後続候補

- **MEM_SNAP sampler 見直し**（pre-existing hazard・別 track）
- D169-1/D169-2 evidence 一式を含む commit（ユーザー判断）
