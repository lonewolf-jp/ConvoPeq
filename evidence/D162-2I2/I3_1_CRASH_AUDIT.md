# D162-2-I3-1 — Crash Root-Cause Audit

```text
Type:                read-only audit
Production changes:  0
Test changes:        0
CMake changes:       0
Baseline:            ConvoPeq.md 2026-09-05 12:02:18 / D162-2-I2 PASS
Targets:             (P1) Debug double-release SIGSEGV
                     (P2) Release AudioEngineHarness crash
Tools:               Python Win32 mini-debugger（自作・evidence/D162-2I2/i3_1_dbg.py）
                     + x64dbg v2.3.0 MCP + WER LocalDumps 実績 dump
判定:                 **GO（P2 は root cause 位置まで確定）/ P1 は capture 前提が消滅したため scope 更新**
```

---

## P1 — Debug double-release SIGSEGV

### Observed（I2-2 開発時点の記録）

```text
2nd releaseResources()
  └ ABOUT_TO_EXIT_SCOPE（ReleaseResources.cpp:710・teardown_diag.log 357 行の最終行）
  X ← SIGSEGV（exit 139）
  └ ~AudioEngine: enter に到達しない
```

### 今回の debugger 実測（新規 evidence）

| 実験 | 結果 |
| --- | --- |
| **生産 CLI の double-release を Debug で実行**（`ConvoPeq_artefacts/Debug/ConvoPeq.exe --cli-run --cli-device-type WindowsAudio --cli-exit-ms 4000`） | **crash なし**。exit_process 正常終了。`releaseResources: enter` 2 回（double-release 成立）・2 個体の DSP が remaining=0 で閉包（i3_1_convopeq_dbg_run.log 56KB）。debugger 下でも同様（2 回目の releaseResources を通過・benign first-chance のみ） |
| **現行 Debug harness を debugger 下で実行**（mini-debugger） | **crash なし**。全テスト完走・exit_process・EXIT=0。second-chance 例外 0 件 |
| x64dbg live session | harness が gdi32full/shlwapi 待ちで長時間停止（debug heap 遅延）— crash まで到達せず。上記 2 実験で代替 evidence を取得 |

### 判定

- **生産パスの double-release（prepare→release→prepare→shutdown-release）は Debug binary でも
  crash しない**（live debugger + free run の両方で本日実証）。I1-D/I2 の D profile
  （RWDI 6/6 exit 0x0）と整合。
- I2-2 開発時に観測した SIGSEGV は **harness コンテキスト固有**（実 JUCE device を開かない
  harness の teardown 環境）であり、(a) 当該 interim コードは現行ツリーに存在せず
  （abandon seam が double-release を迂回）、(b) WER dump も生成されていないため、
  **faulting instruction / call stack は遡及取得不可能**。
- I2 非依存性: BISECT（修復 destroy 無効化でも crash）+ 本日の生産パス clean 実測で
  **I2-independent 確定**。I2 の ownership closure への影響なし。

**P1 = NO-GO（root cause 確定に必要な capture 対象が消滅）→ scope 更新**:
「double-release は生産パスで crash しない」ことを Debugger evidence として確定し、
harness 環境の teardown 差分（実 device なし・手動 audio thread）は I3-2 以降の
test-infrastructure 課題として分離。production への影響は現状ゼロ。

## P2 — Release AudioEngineHarness crash

### Observed

- Release harness free run: 空の出力で即 SIGSEGV（139・0.07s）。
- Release CTest（I2-3 時点）: `AudioEngineHarness (SEGFAULT)`、10:01 の WER dump
  `AudioEngineHarness.exe.35620.dmp` が残存。
- RWDI harness（同一ソース・13:22 build）: **全テスト PASS（EXIT=0）**。
- Debug harness: PASS（上記）。

### 捕獲（mini-debugger・2 回再現）

```text
exception: 0xC0000005 (WRITE) first → second chance
exception address: 0x7ff6a613f89d  (= image_base 0x7ff6a41c0000 + RVA 0x1F7F89D)
module:            audioengineharness.exe（memory info で module 帰属確認済み）
faulting instruction bytes: C6 42 65 02  C6 42 64 01  4D 89 99 98 01 00 00 ...
  = mov byte ptr [rdx+0x65], 0x02      ← faulting instruction（rdx = 0 → 番地 0x65 への WRITE）
  = mov byte ptr [rdx+0x64], 0x01
  = mov qword ptr [r9+0x198], r11 ...
registers: rax=0 rcx=0 rdx=0 rbx=1 r8=0 r9=0x215ba208cc0(有効ヒープ) r10=1 r11=1
stack: rsp 起点の 0x4000 走査で exe module への return address 候補なし（新規 thread or 深い frame）
```

### WER dump との照合（I3-1-B 必須比較）

`AudioEngineHarness.exe.35620.dmp`（minidump 解析）:

```text
ThreadId 0x2e1c | EXCEPTION_ACCESS_VIOLATION
ExceptionAddress 0x7ff60f0cf89d  （exe base 0x7ff60f150000 + RVA 0x1F7F89D — 同一命令）
ExceptionInformation [1, 101]    = WRITE アクセス・ faulting address 101 (0x65)
```

→ **live capture と完全一致**（同一 RVA・同一 faulting address 0x65 = null+0x65 の
member write）。P2 の fault chain は 2 系統の独立 capture で確定。

### root cause の位置

- faulting instruction のバイトパターン `C6 42 65 02 | C6 42 64 01 | 4D 89 99 ...` を
  Release の全 obj から検索 → **1 個体のみ一致**:
  `build-diag/CMakeFiles/AudioEngineHarness.dir/Release/src/audioengine/ISRRuntimePublicationCoordinator.cpp.obj`
  （evidence: isrcoordinator_obj_disasm.txt :4130）
- 命令を含む関数（start 0x...F410・約 0x500 バイト）の構造:
  inline spinlock（xchg/pause/指数バックオフ — MSVC STL の 16 バイト atomic lock-pool の
  インライン展開）+ 320 バイト stride のテーブル走査（`rax = r14*5 << 6`）+
  `cmp r8b,1`（ObligationState == Live の byte 比較）。
  → **`ISRRuntimePublicationCoordinator.cpp` の RecoveryAdmissionTable（D152 系・
  RecoveryLifecycleWord 16 バイト atomic + 320 バイト slot の identity CAS）系コード**と特定。
- fault: slot/base ポインタ `rdx = 0` のまま payload byte field（+0x64/0x65）へ書き込み —
  **null テーブル/スロット base からの member write（UB）**。rcx（引数 1 = base）も 0 で
  到達しており、呼び出し側が null base を渡している。
- Release のみ crash する理由: null からの member write は UB — Debug/RWDI は
  最適化・レイアウト差で実害が出ていない（RWDI PASS 実測）。link order / CRT / destruction
  order ではなく **Release コンパイル固有のコードパス（inlined CAS path の null base）** に帰属。

### Debug との同一性比較

Debug SIGSEGV（生産パスでは crash なし・harness 固有）と Release AV（生産ではない harness
startup の coordinator admission CAS で null base write）は **fault chain が異なる →
独立 defect として扱う**。ただし両方とも harness という test 車両上の事象であり、
生産 CLI（Debug/RWDI/Release いずれも）では D-profile/I1 の実績どおり crash していない。

## I3-1-C — I2 との関係

- P1: 生産 double-release が Debug でも clean であることを live debugger で追加実証 →
  I2-independent の再確認。
- P2: crash 関数は `ISRRuntimePublicationCoordinator.cpp` の admission table 系 —
  **I2 の変更対象（PrepareToPlay.cpp の ownership repair）とは無関係**。G2 世代から存在する
  Release harness 失敗の系譜上にある（当時は 0xc0000374・コードは現 tree とは別版本のため
  症状が変化）。
- **I2 の CallerDestroy → destroyRolledBackDSP closure は変更禁止のまま遵守**（本監査では
  一切触れていない）。

## I3-1 GO / NO-GO 判定

| GO 条件 | P1 | P2 |
| --- | --- | --- |
| faulting instruction 特定 | ✗（capture 対象消滅・生産パスは clean を実証） | **✓**（mov byte [rdx+0x65],2 / rdx=0） |
| root cause 特定 | —（scope 更新: 生産影響ゼロ・harness 固有） | **部分的 ✓**（ISRRuntimePublicationCoordinator.cpp の CAS path・null base 由来。呼び出し元 null 渡しの一次原因は I3-2 で特定） |
| ownership/lifetime chain | — | 部分 ✓（null base = admission table/slot の lifetime 未確定） |
| Debug/Release 同一性 | — | **独立 defect 確定** |
| I2 非依存 | ✓ | ✓ |
| 修正境界の定義 | — | 可（null 渡し箇所の特定 → I3-2） |

**判定 = P2 は GO（I3-2 repair design へ進められる）／P1 は NO-GO→scope 更新
（生産影響ゼロを evidence で確定・以後 harness 課題として分離）。**

## 禁止事項の遵守確認

- `_exit()` / `abandonEngine()` を production へ導入していない（harness/test 専用 seam のまま）。
- release の idempotent 化・shutdown admission reopen・I2 closure 変更・destructor への
  安易な guard 追加 — いずれも未実施。
- 生成物: evidence/D162-2I2/{i3_1_dbg.py, i3_1_debug_crash.json, i3_1_release_crash.json,
  i3_1_release_crash2.json, i3_1_convopeq_dbg_run.log, teardown_diag.log,
  isrcoordinator_obj_disasm.txt, isrcoordinator_obj_symbols.txt}（いずれも audit 成果物）。
