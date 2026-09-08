# D162-2-I3-1 Crash Root-Cause Audit Report

```text
Date:     2026-09-05
Type:     read-only crash audit（production/test/CMake 変更 0）
Baseline: ConvoPeq.md 2026-09-05 12:02:18 / D162-2-I2 PASS
Evidence: evidence/D162-2I2/I3_1_CRASH_AUDIT.md + i3_1_*.json/log + isrcoordinator_obj_*.txt
判定:     **P2 (Release harness crash) = GO — root cause 位置確定（ISRRuntimePublicationCoordinator.cpp
          admission CAS path の null base write・RVA 0x1F7F89D）/
          P1 (Debug double-release) = NO-GO → scope 更新（生産パスは crash しないことを
          debugger evidence で確定・harness 課題として分離）**
```

---

## P1 — Debug double-release SIGSEGV

- 生産パス（`ConvoPeq.exe --cli-run --cli-device-type WindowsAudio --cli-exit-ms 4000` の
  Debug binary）を live debugger 下で実行 → **double-release（releaseResources 2 回）を
  crash なしで完遂**（2 個体の DSP が remaining=0 で閉包・benign first-chance のみ）。
  free run も同様に clean。I1-D / I2 の D profile（RWDI 6/6）と整合。
- I2-2 開発時に観測した harness SIGSEGV は、実 JUCE device を開かない harness 環境での
  teardown 固有の事象で、当該 interim コードは現行ツリーに存在せず、WER dump も未生成 —
  **faulting instruction は遡及取得不可能**。
- 判定: **P1 = NO-GO → scope 更新**。「double-release は生産パスで crash しない」ことを
  debugger evidence で確定。production 影響ゼロ。以後は harness test-infrastructure 課題。

## P2 — Release AudioEngineHarness crash

### Faulting instruction（2 系統の独立 capture で一致）

```text
live capture（Python mini-debugger）:
  exception 0xC0000005 WRITE @ 0x65（= null + 0x65）
  RIP = image_base + RVA 0x1F7F89D（audioengineharness.exe・memory info で module 帰属確認）
  bytes: C6 42 65 02   = mov byte ptr [rdx+0x65], 2   （rdx = 0）
         C6 42 64 01   = mov byte ptr [rdx+0x64], 1
         4D 89 99 98 01 00 00 = mov qword ptr [r9+0x198], r11
  regs: rcx=0 rdx=0 r8=0 r9=有効ヒープ rbx=1

WER dump AudioEngineHarness.exe.35620.dmp（minidump 解析）:
  ThreadId 0x2e1c | EXCEPTION_ACCESS_VIOLATION
  ExceptionAddress = exe_base(0x7ff60f150000) + RVA 0x1F7F89D（同一命令）
  ExceptionInformation [1, 101] = WRITE @ address 101 (0x65) — 完全一致
```

### Root cause の位置

- faulting instruction のバイトパターンを Release 全 obj から検索 → **1 個体のみ一致**:
  `src/audioengine/ISRRuntimePublicationCoordinator.cpp.obj`
  （evidence isrcoordinator_obj_disasm.txt :4130 — `mov byte ptr [rdx+65h],2`）。
- 含まれる関数（~0x500 バイト）: inline spinlock（MSVC STL 16 バイト atomic lock-pool の
  インライン展開）+ 320 バイト stride テーブル走査 + `cmp r8b,1`（ObligationState==Live）→
  **RecoveryAdmissionTable（D152 系・RecoveryLifecycleWord identity CAS）系コード**。
- fault 実体: base ポインタ rdx=0 のまま slot payload byte field（+0x64/0x65）に書き込み —
  **null base からの member write（UB）**。呼び出し側が null base を渡している。
- Release 固有: 同一ソースの Debug/RWDI は PASS（RWDI 13:22 build 全テスト PASS 実測）—
  link order/CRT ではなく **Release コンパイルの inlined CAS path に現れる null deref UB** に帰属。

### Debug / Release の同一性

fault chain が異なる（Debug: 生産パス crash なし・harness teardown 固有 / Release:
coordinator admission CAS の null write）→ **独立 defect 確定**。

### I2 関係

I2 の変更対象（PrepareToPlay.cpp）とは無関係。G2 世代から存在する Release harness 失敗の
系譜（当時 0xc0000374・コードは別版本）。**I2 closure は変更禁止のまま遵守**。

## I3-2（repair design）への引き継ぎ

- 修正境界: (a) Release crash — 呼び出し側が null base を渡す箇所の特定（crash 関数の
  caller chain 解析は Release PDB が無いため要 1 追加調査: RWDI config で同一経路を
  ブレークするか、obj の reloc/call-site から逆引き）。(b) P1 — production 影響ゼロのため
  対象外とするか、harness 環境の teardown 差分を I3-2 で検証するかの選択。
- 禁止事項（release idempotent 化・admission reopen・I2 closure 変更・production への
  `_exit`/`abandon` 導入等）は遵守。本監査で一切実施していない。
