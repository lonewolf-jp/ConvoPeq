# P1-5-IR-P2-F — SIGSEGV crash attribution report（2nd attempt）

- **作成**: 2026-09-22 / work113 P1-5-IR Phase 2-F
- **性質**: read-only crash attribution 再試行（production/test 変更 0）
- **基準**: HEAD `1e9e63e3` / binary `F6C49980920C1D8E`（flag cache=OFF）/ P2 matrix = NOT STARTED

---

## 1. 実施内容

- x96dbg(x64dbg) に AudioEngineHarness（`--buzz-os=1` smoke 条件そのまま）をロードして起動。
- x64dbg の初期 state は ntdll で pause（system break）。`run` を繰り返して benign first-chance exception（gdi32full / audioengineharness の thread start / ntdll）を resume 継続。
- crash まで resume 継続（合計 約 11 回 resume・約 8 分の実時間）。
- **最終状態: debuggee が SIGSEGV で terminate し、debugger 側に `state=stopped` のみが残存。**
- crash 停止の状態保持（exception address / RIP / call stack）を取得しようとしたが、**debugger はすでに process termination 後の stopped 状態であり、実行コンテキスト（register/stack/thread）は消滅していた**（`Plugin error (409): Debugger must be paused`）。

## 2. 取得できなかった項目

| 必須取得項目 | 結果 |
| --- | --- |
| exception address / faulting address | **取得できず** |
| RIP | **取得できず** |
| RSP | **取得できず** |
| faulting thread ID | **取得できず** |
| full call stack | **取得できず** |
| module + symbol | **取得できず** |
| exception type / access type | **0xC0000005（SIGSEGV 相当）と crash location は os=1 smoke log から判明済みだが、RIP/stack の詳細はなし** |
| access address | **取得できず** |

- x64dbg の exception stop 構成（`SetExceptionFilter 0xC0000005`）は試したが、**debugger は crash 時点の state を保持しないまま terminate まで進んだ**（手動 terminate 前に state capture の機会がなかった）。
- また crash のタイミングは x64dbg の GUI 側で異なる（debugger は crash を最初に検出せず、debuggee が先に terminate した）。

## 3. 分類（3-F-3）

| RIP 所属 | 分類 |
| --- | --- |
| （未取得 — RIP 不明） | **E: JUCE / CRT / system / symbol unavailable** |

## 4. 判定

```text
Step 3-F = UNKNOWN
SIGSEGV root cause = UNKNOWN
P2 matrix = STOP
```

- 3-F の PASS 条件（exception type / faulting thread / RIP / module / call stack / faulting address）を **1 つも確定できなかった**。
- x64dbg 単独では crash 時の state capture ができず（crash 後の terminate が先に来る）、crash dump を保存する追加手段（LocalDumps 設定・cdb・attach 方式の変更）が必要。

## 5. STOP / 次の承認事項

1. **Windows Error Reporting LocalDumps の有効化**（`HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps` に `DumpType=2` / `DumpFolder` を設定 → 次回 crash 時に `%LOCALAPPDATA%\CrashDumps` への full dump 保存）→ WinDbg/cdb による offline 解析（RIP/stack/module が取得可能）。
2. または cdb（Windows SDK Debuggers）のインストール完了を待っての CLI debugger 再試行。
3. または SIGSEGV crash を production 欠陥の候補として別チケット化し、P2 の OS 条件を暫定 2x に変更（要 承認）。

本 audit は production/test 変更 0 のまま。P2 matrix = STOP。
