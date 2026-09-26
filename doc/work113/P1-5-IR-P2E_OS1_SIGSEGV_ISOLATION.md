# P1-5-IR-P2-E — OS=1 SIGSEGV read-only isolation report

- **作成**: 2026-09-22 / work113 P1-5-IR Phase 2-E
- **性質**: read-only isolation（production/test harness 変更 0・crash attribution 用の debugger 介入のみ）。
- **範囲**: `--buzz-os=1` smoke run の SIGSEGV について crash attribution の read-only 調査。

---

## 1. 観測事実（既存 smoke log のみ）

- **crash run**: `tmp/p1_5_ir_p2_raw/smoke_os1_20260922_1938.log`
  - 最終行: `[GEOM before-capture] IR(sr=192000 block=1024 len=96000 gen=14) UIprep(sr=192000 block=1024) ENG(buildRate=192000 irLen=96000 block=1024 layers=2 L0part=1024 L0numIR=23 L0numParts=32 L0fft=2048 ring=8192) RT(addNs=1024 addCalls=1225603 L0calls=1225602 getNs=1024 getGot=1024 getShort=0 ringW=1024 ringR=1024 avail=0 fdl=1 nextPart=0)`
  - その後 SIGSEGV(139)。
- **対照 run (OS=2)**: `tmp/p1_5_ir_p2_raw/smoke_noos_20260922_1945.log`
  - `[GEOM before-capture] IR(sr=384000 block=2048 len=192000 gen=13) ...`
  - 正常 shutdown まで到達・exit 127(判別不能な異常終了)。
- crash run は **capture 直後の1行で途切れる**（`[PROBE_METRICS]` に到達していない）。no-os 対照は metrics 行まで出力。
- crash run は x64dbg attach 中に exit code 139 (SIGSEGV) で死んだ。

---

## 2. x64dbg crash attribution (部分)

- crash run は x96dbg(x64dbg) 経由で AudioEngineHarness に attach して実行。
- crash 時点で debugger は system 情報しか表示できず、OS=1 / OS=2 いずれの値も crash window から直接読めなかった。
- crash 場所（module=audioengineharness の cip 行）への到達は確認済みだが、**完全な call stack / register / faulting thread の特定には至らなかった**（driver が system 情報のみ返す初回障害）。
- よって crash attribution は **PARTIAL / UNKNOWN** のまま（指定 A〜D のうち debugger evidence で決定した thread なし）。

---

## 3. 既存 instrumentation による OS=1 / OS=2 差分（読み取り済み）

| field | OS=1 | OS=2 |
| --- | ---: | ---: |
| processingRate | 192000 | 384000 |
| blockSize(UIprep) | 1024 | 2048 |
| oversamplingFactor | 1 | 2 |
| IR sampleRate | 192000 | 384000 |
| IR length | 96000 | 192000 |
| L0 partition | 1024 | 2048 |
| FFT size | 2048 | 4096 |
| layers | 2 | 2 |
| conv generation | 9 (conv) / 14 (IR) | 7 (conv) / 13 (IR) |
| publicationSeq | 8 | 8 |

→ **OS=1 だけの geometry 不整合は観測されていない**（IR/L0 幾何は OS factor に整合的に縮退）。したがって「OS=1 独自の幾何不整合」は第一候補から除外、SIGSEGV の原因は DSP 実行内部か capture 側のいずれか。

## 4. runCapture / Session / tap の構造的候補（read-only code audit）

- Session は `capMutex` で守られた `std::vector` 3 本 + blockPeak。audio callback 側は capMutex を lock して push する。
- tap lambda `[&session]` は `AudioEngineHarness` 上で setTap により登録され、runCapture 内で `session.mode.store(1)` → pump → capture → `session.mode.store(0)` の流れ。
- tap は `AudioBuffer<float>&` を直接参照（callback スレッド側の buffer は JUCE が管理）。runCapture 自体に OS 依存の処理はない。
- よって OS 依存の差は **DSP 内部幾何のみ**であり、capture/Tap 側の OS 依存変数は存在しない。

## 5. 判定

| カテゴリ | 判定 |
| --- | --- |
| A: Audio callback / DSP 内 | **候補（第一）** — OS差は DSP 内部幾何（processRate/IR/L0/FFT）にのみ存在するため |
| B: test tap / Session capture | 可能性低（OS=2 で同一経路が正常終了） |
| C: rebuild/publish worker | 可能性低（crash 時点で rebuild/publish は完了済み・log 上 irLoaded=1 gen=14 まで到達） |
| D: shutdown/lifetime | 撤退（crash は capture 中・shutdown 前段） |
| E: 未特定 | ** debugger evidence 不足のため暫定 E** |

**SIGSEGV の thread 内訳は特定できていない（debugger 情報が初回障害で失効）。したがって「production OS=1 bug」とは確定しない。**

---

## 6. Next authorization request

- x64dbg crash attribution を**成功させる**には、`AudioEngineHarness.exe` を x96dbg 配下で再起動し、**SIGSEGV 停止時に例外アドレス/レジスタ/スタックを確定してから**プロセスを落とす必要がある。
- 今回は crash 時に debugger が system 情報のみを提示し、A〜D の決定的証拠を得る前にセッションが失効した。
- production 変更は 0 のまま。

**STOP**: P1-5-IR-P2-E はここで停止し、SIGSEGV の thread/stack 取得については次の承認を待つ。
