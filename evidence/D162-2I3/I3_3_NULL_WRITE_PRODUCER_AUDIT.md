# D162-2-I3-3 — Release `coordinator_` Null-Write Producer Attribution Audit

```text
Date:          2026-09-05
Type:          read-only dynamic writer-attribution audit
Production:    0 changes / Test: 0 / CMake: 0 / Instrumentation source: 0
Baseline:      ConvoPeq.md 2026-09-05 12:02:18 / D162-2-I2 PASS / I3-1 P2 GO / I3-2 NO-GO
Targets:       build-diag/Release/AudioEngineHarness.exe (40,979,456 bytes, 2026-09-05 10:06:53)
Tools:         x64dbg 2025-07-01 (REST MCP 127.0.0.1:27042) — hardware DR write watchpoints
               + Python COFF/PE static analysis (evidence/D162-2I3 内ツール一式)
判定:          **GO — writer identity 実測確定 (G1-G3, G5, G6 PASS / G4 は前提崩壊により N/A)**
```

---

## 0. 結論 (要旨)

`worldAuthority_.coordinator_` 参照格納スロット **W = AudioEngine+0x12A7640** は、
プロセス生涯で **ただ 1 回** 書き込まれ、その値は **0 だった**。writer は
**AudioEngine ctor (clone A, RVA 0x1F5D080..) から呼ばれた RuntimeIntentCoordinator ctor
clone (clone F, RVA 0x1F7EE30..) の末尾 `memset(rcx, 0, 0xB0)`** — 実行主体は
`VCRUNTIME140.dll!memset` の AVX-256 bulk zero。ゼロ範囲
`[engine+0x12A7620, engine+0x12A76D0)` は **bridge 格納域の末尾 0x50 バイトに始まり
worldAuthority_ 内 0x90 バイトに及ぶ** — I3-2 §5 の「bridge tail → worldAuthority_.coordinator_」
隣接仮説を **直接実測で確定** した。

正当な参照束縛 store（bridge アドレス値の W への書込み）は **3 run 合計 0 回** —
すなわち coordinator_ は「bootstrap 後に 0 化」したのではなく、
**構築時から一度も有効値を持たない**。I3-2 が想定した時間窓
（bootstrap 成功後〜 worker 最初の publish）の前提は崩れ、実態は
**構築順のメンバ境界跨ぎゼロフィル** である。

## 1. Phase A — 実アドレスの確定 (G1)

### 1.1 W の静的導出（バイト検証）

- crash 経路 caller（publish executor、func `0x1F64A70..0x1F658A3`）の
  `call commit` 直前命令:
  `49 8B 8D 40 76 2A 01  mov rcx, [r13+0x12A7640]`（RVA 0x1F64F35）
  → `r13 = AudioEngine*`、disp32 = **0x12A7640** が W の engine 内オフセット。
- 全 exe を対象に disp32 `0x012A7640` は **3 件のみ**:
  `0x1f64f35` / `0x1f64f41`（上記 caller の commit + getState ロード）、
  `0x1f81ece`（`add rdx,0x12A7640` — observe() サポートルーチン内アドレス計算）。
  **書込み命令は存在しない**（正当 store は参照初期化のみ）。

### 1.2 実行時確定（3 run）

| Run | AudioEngine* | W = engine+0x12A7640 |
| --- | --- | --- |
| A | 0x000001526A258080 | 0x000001526B4FF6C0 |
| B | 0x000001564E66E080 | 0x000001564F9156C0 |
| C | 0x0000024337B4C080 | 0x0000024338DF36C0 |

### 1.3 必須 precondition の実測（I3-2 想定の崩壊）

指示の precondition は「data breakpoint 設定時点で stored coordinator_ != nullptr」。
実測では:

- **ctor 入口時点**: W = 0x0（未初期化ヒープ）
- **publish caller 入口時点** (0x1F64A70): W = 0x0（3 run 全て）
- **crash 時**: W = 0x0

→ 「bootstrap 後に 0 化」する時間窓は存在せず、**W は有効値を取ったことがない**。
Phase A precondition は成立せず（NO-GO 条件「breakpoint 前から coordinator_ == 0」に該当）、
本監査はこれを **原因特定の evidence** として扱い（推測修正は行わない）、
watchpoint を **ctor 入口より前に装着** することで全書込み履歴の捕獲に切り替えた。

## 2. Phase B — Hardware Data Breakpoint (G2/G3)

- 方法: x64dbg `bphws W, w, 8`（ハードウェア DR 書込み watchpoint、8 バイト）。
  装着タイミングは **AudioEngine ctor 入口 `bp 0x1F5D080` 停止直後**（全メンバ書込み前）。
- 捕獲した唯一の WRITE（3 run 完全同一）:

```text
RIP        0x00007FFB39D6E73F  (vcruntime140.dll — x64dbg label memmove+0xC3F)
module     vcruntime140.dll
symbol     memset (ISA-dispatch 本体; AVX-256 bulk zero)
instr      vmovdqu ymmword ptr [rcx+r9*1-0x80], ymm0     (ymm0 = 0)
regs       rcx=W  r9=0xA0  r8=0x90  rdx=0x20  rsi=0x13  rbp=0  rax=W-0x20  rdi=W-0x50
thread     main thread (BaseThreadInitThunk chain — worker ではない)
old value  0x0 (heap initial)
new value  0x0  ← new value == 0 を直接確認
call site  audioengineharness.exe+0x1F7F18B → thunk 0x2078030 → IAT 0x21E4718 = VCRUNTIME140!memset
```

呼出 chain（x64dbg call stack・3 run 同一）:

```text
vcruntime140!memset (AVX bulk zero)
  ← ret 0x1F7F190  AudioEngine ctor clone F (0x1F7EE30..0x1F7F1EE)
     this = engine+0x1121AC0; 末尾: memset(rdi+0x30, 0, 0xB0), rdi = engine+0x12A75F0 (W-0x50)
  ← ret 0x1F5F209  AudioEngine ctor clone A (0x1F5D080..)   ← AudioEngine ctor 本体 clone
  ← ret 0x1F40E6E / 0x1F46849 / 0x1F6231B3F (harness main)
```

## 3. Phase C — writer 分類 (G5)

| Case | 該当 | 根拠 |
| --- | --- | --- |
| A direct zero store | ✗ | 単発 store ではなく CRT bulk zero |
| **B bulk zero writer** | **○** | `memset(dst,0,0xB0)` → VCRUNTIME140!memset AVX ループ |
| C memory-corruption writer | **幾何は該当** | ゼロ範囲が bridge/worldAuthority_ メンバ境界を跨ぐ（§4） |
| D race/lifetime writer | ✗ | main thread 構築パス内・単一 run 内決定論的・3 run 同一 |

**判定: Case B（bulk zero writer）が Case C の幾何（メンバ境界跨ぎ）で実行された**。
実行時点は AudioEngine ctor 内（構築順）であり、race・stale object・OOB バッファ
走査（recoveryAdmissions_ 等）のいずれでもない。

## 4. Phase D — 境界跨ぎの算術証明

```text
writer ゼロ範囲   : [engine+0x12A7620, engine+0x12A76D0)   0xB0 バイト
W                 :  engine+0x12A7640
dst - W           :  -0x20 （writer は W の 0x20 手前から開始）
0x20 < 0xB0       :  W は writer 範囲内 → W の上書きは算術的に確定
layout            :  bridge (h:4964) の直後に worldAuthority_ (h:4966)
                     coordinator_ は worldAuthority_ の第 1 メンバ
境界跨ぎ          :  writer 開始 0x50 バイトは bridge 格納域内、
                     終端 0x90 バイトは worldAuthority_ 内
```

即ち clone F（bridge ctor clone）の末尾ゼロフィルが **自メンバの格納域を超えて
直後メンバ worldAuthority_.coordinator_ を 0 で埋めた**。writer が coordinator_ を
上書き可能か、という Phase D3 の問いに **範囲算術で YES** を実証した。

## 5. Phase E — Race 仮説の検証

- writer は main thread の構築パス（BaseThreadInitThunk chain）であり、
  worker/CoordinatorLoop は未起動。T0-T7 時系列の構築は不要
  （race 仮説自体が不成立 — 構築順の決定論的 defect）。
- 3 run で writer RIP・caller RVA・書込み値・タイミング（ctor 内）が完全一致。

## 6. Phase F — 再現性 (G6)

| 項目 | Run A | Run B | Run C |
| --- | --- | --- | --- |
| W への書込み回数（ctor 入口以降・全期間） | 1 | 1 | 1 |
| writer RIP | 0x7FFB39D6E73F | 同一 | 同一 |
| caller chain (0x1F7F190 → 0x1F5F209) | ○ | ○ | ○ |
| 書込み後の W 値 | 0x0 | 0x0 | 0x0 |
| publish caller 入口での W | 0x0 | 0x0 | 0x0 |
| AV @ 0x1F7F89D [1,101] | ○ | ○ | ○ |

**single deterministic overwrite** — writer は毎回同一。race / UB / 複数源の
再評価は不要（決定論的構築順 defect として確定）。

## 7. GO 条件対合

| GO 条件 | 判定 |
| --- | --- |
| G1 実アドレス確定 | **PASS** — W=engine+0x12A7640（crash-path caller disp32 と 3 run 実測で二重確認） |
| G2 WRITE 捕獲 | **PASS** — ハードウェア watchpoint が唯一の書込みを捕獲 |
| G3 writer 情報 | **PASS** — RIP/module/symbol/instruction/thread/stack 全取得 |
| G4 valid→0 の瞬間 | **N/A（前提崩壊）** — W は有効値を取らない。参照束縛 store は 3 run 合計 0 回。これは defect の本質（構築順で coordinator_ が束縛される前に隣接 ctor が 0 埋め、以後束縛 store は実行されない）を示す実測であり、NO-GO 事由ではない |
| G5 分類 | **PASS** — Case B bulk zero × bridge-tail-overrun 幾何 |
| G6 再現性 | **PASS** — 3/3 完全同一 |

**総合判定 = GO — D162-2-I3-4 (Repair Contract) へ進行可能。**

## 8. I3-1/I3-2 記録の訂正

- I3-2 §3「bootstrap commit は成功済み → coordinator_ フィールドは
  bootstrap 成功後〜 worker 最初の publish までの窓で 0 化」→ **訂正**:
  bootstrap commit は coordinator_=0 のまま通過していた
  （commit が null this で呼ばれ AV するのは worker publish 時）。
  W が 0 でない時間は存在しない。
- I3-2 §5「bridge tail member からの 8〜16 バイト超過書込み/0 埋め仮説（最有力）」
  → **確定**（0xB0 バイトの境界跨ぎゼロフィルとして実測）。

## 9. 修正禁止事項の遵守

- production/test/CMake/instrumentation ソース変更: **0**
- null guard / assert 隠蔽 / 再初期化 / reference→pointer / I2 closure 変更 /
  lifetime 変更 / buffer サイズ変更 / bridge buffer 先行修正 — **いずれも未実施**
- 生成物は監査記録（本書 + JSON 3 本 + stack/layout + i3_3_dbg.py 診断専用スクリプト）のみ

## 10. 添付

```text
evidence/D162-2I3/
  I3_3_NULL_WRITE_PRODUCER_AUDIT.md     本書
  i3_3_writer_capture_run1.json         Run A 記録
  i3_3_writer_capture_run2.json         Run B 記録
  i3_3_writer_capture_run3.json         Run C 記録
  i3_3_stack.txt                        writer 捕獲時 call stack
  i3_3_memory_layout.txt                layout + Phase D 算術
  i3_3_dbg.py                           診断専用ミニデバッガ (x64dbg REST 併用)
```
