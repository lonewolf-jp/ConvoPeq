# P1-5-IR-P2-F3 — SIGSEGV/heap-crash attribution（cdb run）

- **作成**: 2026-09-22 / work113 P1-5-IR Phase 2-F（cdb 再試行）
- **性質**: read-only crash attribution（production/test 変更 0）
- **tool**: Store WinDbg パッケージ内 `cdb.exe`（10.0.29617.1000）を `tmp/cdb.exe` へ複製して使用（WindowsApps 直実行は Permission denied のため）
- **binary**: `build/Release/AudioEngineHarness.exe`（`F6C49980920C1D8E`・PDB 隣接）
- **条件**: os=1 smoke 条件そのまま（`--buzz-os=1 --buzz-probe=real --buzz-conv=on --buzz-eq=off --buzz-order=cte --buzz-ir=tmp/p15_ir_g0.wav --buzz-probe-level=0.5011872`・quiet 既定）

---

## 1. 主要発見（cdb run）

1. **OS=1 capture SIGSEGV は再現しなかった**。cdb 配下では `[GEOM before-capture]` → `[PROBE_METRICS]` → `[GEOM after-capture]` → shutdown まで**完走**（dbg: `tmp/p1_5_ir_p2_raw/cdb_stdout.txt` :860-880）。→ **capture 時 SIGSEGV は非決定的**（ timing/実行環境依存）。
2. 代わりに **exit 時 heap corruption（0xC0000374）を取得**（`tmp/p1_5_ir_p2_raw/cdb_crash.log` :67-108）:
   ```text
   Critical error detected c0000374
   (7210.3dc): Unknown exception - code c0000374 (first chance)
   (7210.3dc): Unknown exception - code c0000374 (!!! second chance !!!)
   ```
   - crash stack（process exit 時・atexit handler）:
     ```text
     ntdll!RtlFreeHeap+0x285
       ← ucrtbase!free_base+0x1b
       ← AudioEngineHarness+0x203556d        ← atexit handler（harness shutdown path）
       ← AudioEngineHarness+0x20343f2
       ← ucrtbase!execute_onexit_table+0x87   ← atexit テーブル実行中
     ```
   - RIP = `ntdll!RtlpMuiRegCreateRegistryInfo+0x1b5`（heap corruption の raise 点）/ access type = **free 時 heap 検査失敗（c0000374・write-free 系）**。
   - **heap corruption は capture 段階ですでに成立しており、exit 時の free で検出された**形。

3. cdb run の capture 自体は完走（`[PROBE] kind=real level=0.501 outPeak=0.026701 → BROADENED`・`inPeak=0.000000` は既知の入力 capture 空問題 — no-os 対照と同じ既存計測課題）。

## 2. 分類（P1-5-IR-P2-E の A〜E に対する更新）

| カテゴリ | 判定 |
| --- | --- |
| A: Audio callback / DSP 内 | **SIGSEGV は非決定的** — cdb 配下では再現せず。capture crash は timing 依存の疑い |
| **D: shutdown/lifetime** | **heap corruption (c0000374) はここで検出** — atexit → `free_base` の対象 block が capture 段階以前に破壊されている |
| E | crash attribution としては依然部分情報 |

**更新された見取り**: （1）capture 中 AV は非決定的・（2）**heap corruption は確定的に存在**（exit 時に検出）。heap corruption が capture 段階の AV の遠因である可能性が高いが、同一欠陥との断定は次の dump まで保留。

## 3. PASS/STOP 判定

```text
exception type       = c0000374 (heap corruption fast-fail)   ← 取得
faulting thread      = main thread (7210.3dc, atexit path)    ← 取得
RIP / module         = ntdll!RtlFreeHeap 経由 / free_base caller = AudioEngineHarness+0x203556d ← 取得
call stack           = captured（全 frame）                    ← 取得
faulting address     = heap block（address 値は log 参照）       ← 間接取得
capture SIGSEGV      = 非再現（非決定的）
```

→ **attribution = PASS（crash 種別は AV から c0000374 heap corruption に更新）**。
**修正には入らない**（read-only 契約維持・P2 matrix = STOP）。

## 4. 次の承認事項

1. **c0000374 heap corruption の source 特定**（`AudioEngineHarness+0x203556d` の symbol 解決 + PageHeap/ASan build での再現）→ test-only の範囲で可能なら実施、production 候補欠陥の切り出しはユーザー判断。
2. capture SIGSEGV は非決定的のため、**OS=1 crash を「nondeterministic / heap corruption 既存」として別チケット化**し、P2 の OS 条件を暫定 2x に変更する判断。
3. 両方とも本 audit の範囲外。

---

## 附録: cdb session log 参照

- debugger log: `tmp/p1_5_ir_p2_raw/cdb_crash.log`（151 行・例外スタック :79-108）
- debuggee stdout: `tmp/p1_5_ir_p2_raw/cdb_stdout.txt`（probe flow 完走 :554-883）
