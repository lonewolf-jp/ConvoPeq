# P1-5-IR-P2 — Step 5-A4: Provenance / Vehicle-State Audit（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（A-4）
- **性質**: **完全 read-only**。source / CMake / production / settle / sleep = **すべて 0 変更**。
  追加実行（run 2・3 回目、probe 再実行、`--buzz-os`、settle 変更）も **0**。
- **目的**: A3 で再現した 1.4759 dB 現象について、historical reference と A3 の
  「実行条件・ソース状態・生成経路」の同一性を source-first で確定する（原因は確定しない）。

---

## A4-0 State Freeze（PASS）

| 項目 | 値 | 判定 |
| --- | --- | --- |
| HEAD | `1e9e63e3` | ✅ |
| `kP15FullMatrix` | `true`（A3 の 1 token・**revert しない**） | ✅ |
| A3 binary SHA-256[:16] | `b2c39a9a0e3b1fe3` | ✅ |
| `CONVOPEQ_CORRECT_POLYPHASE_GAIN` | `BOOL=OFF` | ✅ |
| production source diff | 0 | ✅ |
| CMake diff | 0 | ✅ |
| JUCE diff | 0 | ✅ |
| default / calibration diff | 0 | ✅ |
| settle / sleep diff | 0 | ✅ |
| measurement semantics diff | 0 | ✅ |

```text
A3 = REPRODUCED / P3 = NOT STARTED
```

---

## A4-1 vehicle 差分の確定（source 照合）

### ① `--p1-char` entry — `runP1PolyphaseGainCharacterization()`

| 項目 | source 実読値 | historical / A3 |
| --- | --- | --- |
| `argc` / `argv` | `(void)argc; (void)argv;`（:470）— **一切使用しない** | **MATCH**（引数非依存） |
| hidden / default parameter | **なし**。全条件はソース定数とループで固定（argv 経路が存在しない） | **MATCH** |
| 実行 invocation | `AudioEngineHarness[_off].exe --p1-char`（`tmp/p15_sequence.bat` で確定） | **MATCH** |
| base sample rate | `kSr = 48000.0`（:34）→ `h.start(kSr, kBlock)`（:476） | **MATCH**（両ログの `[IR_RATE_GEN] sourceSr=48000` と整合） |
| block size | `kBlock = 512`（:35） | **MATCH** |
| channel 数 | `Capture` は L 単一（`inL` / `outL` のみ・:121）・`configureCapture(..., captureInput=false, ...)`（:421） | **MATCH** |
| signal generation | `Signal::Sine`・`freq = 1000.0`（`runPair` :666 が固定） | **MATCH** |
| capture length | `kWarmBlocks(4) + kCapBlocks(12)` blocks（:425）・解析窓 `kAnalysisN = 4096`（capture 末尾・:439） | **MATCH** |
| amplitude definition | `amp = std::pow(10.0, ampDb / 20.0)`（:420） | **MATCH** |
| `gainDb` の定義 | `dftDb = 20*log10(dftMag(...))`（:459）／`gainDb = dftDb − ampDb`（:460） | **MATCH** |
| `dftMag` | Hann 窓（`0.5*(1-cos(2πi/(n-1)))`）単一ビン DFT・`×2/wsum` 正規化・周波数基準は **`kSr`（=48000）固定**（:200-214） | **MATCH** |

- **`--p1-char` 以外の条件を実行時に変更する経路は存在しない**（argv 不使用・環境変数依存なし・
  ソース定数のみ）。historical の `AudioEngineHarness_off.exe --p1-char` が**唯一の必要条件**であったことを
  source 上で確定。

### ② `runCase()` の vehicle（変更禁止・source 再確認）

```text
seqBefore = getLastCommittedPublicationSequence()      (:411)
configureChain(...)                                    (:412)
waitBacklogZero(e, 30000)                              (:414)  ← 戻り値は WARN のみ
waitWorldPublished(e, seqBefore, 30000)                (:417)
sleepPump(800)                                         (:418)
configureCapture(...) → h.setTap(...)                   (:421-422)
capture: kWarmBlocks + kCapBlocks blocks               (:424-432)
sleepPump(30) → h.clearTap()                            (:433-434)
解析: maxAbsWindow / countAtOrAboveWindow / dftMag       (:440-461)
```

| 確認項目 | 結果 |
| --- | --- |
| **`waitIrFinalized` の有無** | **`runCase` 内に存在しない。TU 全体にも `waitIrFinalized` は定義されていない**（`ensureTestIr` が `isIRFinalized()` を直接 poll する） |
| `waitWorldPublished` の対象 | `getLastCommittedPublicationSequence()` が **`seqBefore`（`configureChain` の前に取得した値）から前進**し、かつ `getPublicationBacklogCount()==0`。`sleepPump(250)` 後に両者を再確認（:86-103） |
| `publicationSeq` の意味 | `getLastCommittedPublicationSequence()` = **最後にコミットされた publication のシーケンス番号** |
| **`waitWorldPublished` の戻り値** | **`runCase` は無視する**（:417 は `if` を伴わない）→ timeout でも capture へ進む |
| `sleepPump(800)` の位置 | `waitWorldPublished` の**後**・`configureCapture`/capture の**前** |
| IR reload の位置 | **`runCase` の外**。`runPair` が `ensureTestIr` を **sc0 の runCase 前と sc1 の runCase 前の計 2 回**呼ぶ（:664・:682） |

### ③ `ensureTestIr()` の IR lifecycle

```cpp
loadImpulseResponse(irFile, false)                        (:394)
for (i < 3000) { if (isIRFinalized()) return true;
                 pumpMessages(); sleep(100 ms); }         (:395-400)
return isIRFinalized()                                    (:401)
```

| 確認項目 | 結果 |
| --- | --- |
| lifecycle | `loadImpulseResponse` → `isIRFinalized()` poll（100 ms 間隔・最大 3000 回 = 最大 300 s） |
| 呼出位置 | `runPair` 内・**sc0 の runCase 前**、**sc1 の runCase 前**（計 2 回/ケース） |
| 戻り値の扱い | `false` なら `runPair` が `failures++` して当該ケースを打ち切る（:665・:683） |

### 参照行（g0 / os=1 / amp=−20・−6）の geometry 照合

A3 で実測済みの項目を provenance として整理（**再測定は行っていない**）:

| 項目 | historical | A3 | 判定 |
| --- | --- | --- | --- |
| source SR | 48000 | 48000 | MATCH |
| target SR | 48000 | 48000 | MATCH |
| source length | 1 | 1 | MATCH |
| target IR length | 48000 | 48000 | MATCH |
| L0 part | 512 | 512 | MATCH |
| L0 FFT | 1024 | 1024 | MATCH |
| slot | 11 | 11 | MATCH |
| F_scale | 0.50013092 / 0.50026187 | 0.50013092 / 0.50026187 | MATCH |
| IR generation | gen=0 | gen=0 | MATCH |
| `ratio` / `resampled` | 1.0000 / no | 1.0000 / no | MATCH |

---

## A4-2 `kP15FullMatrix` の意味の限定

| 項目 | A4 での扱い | 根拠 |
| --- | --- | --- |
| `kP15FullMatrix=true` で `p15ir` が実行可能 | **PROVEN** | A3 実行（`cases=220`・`p15ir` 18 行）+ binary 内 `p15ir` 文字列 1 |
| `p15ir` vehicle が historical と一致 | **PROVEN** | A4-1 ①②③ の source 照合・A3 の geometry 照合 |
| 1.4759 dB が再現 | **PROVEN** | A3: 22/22 フィールド一致・delta-of-deltas = 0.000000 dB |
| `kP15FullMatrix` 自体が DSP 原因 | **UNKNOWN** | 本 flag は `if` ガードのみ（DSP 経路に引数を渡さない） |
| IR scale が原因 | **UNKNOWN** | F_scale は両者一致（差の説明変数にならない） |
| polyphase gain が原因 | **UNKNOWN** | `CONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF` 固定・flag は A3 で未変更 |
| 1-run-lag が原因 | **UNKNOWN** | A4-4 は topology 一致のみを確定 |
| 3-J が原因 | **UNKNOWN** | 3-J は test-only lifetime 1 token（IR DSP 非変更） |

- **`kP15FullMatrix` は「実行経路を復元する gate」としてのみ扱う**。
  A3 は「vehicle の復元」と「現象の再現」を証明しただけで、原因は証明していない。

---

## A4-3 historical `p15staging` 差異（provenance observation のみ）

A3 で観測された以下を、**原因解析せず** provenance として記録するに留める:

| # | observation |
| --- | --- |
| 1 | historical log（`tmp/p15char_off_partial.log`）は **途中 kill** されており `[P1CHAR] summary` 行を持たない（未完了 run） |
| 2 | `p15staging` は historical 側が **未完**（30 行・期待 36 行）。A3 は 36 行完走 |
| 3 | A3 は **220 cases 完走**（`summary flag_macro=0 cases=220 failures=0`） |
| 4 | **`p15ir` は 18/18 で `gainDb` 完全一致**（位置対応比較・id 順序も完全一致） |
| 5 | 位置対応比較で `gainDb` 不一致は **`p15staging` のみ 18 件**（`p15preset` 7・`p15eq` 18・`p15ir` 18 は不一致 0） |
| 6 | 全セクションで **id 順序の不一致は 0**（ループ構造は同一） |

- **`p15staging` の原因解析・修正は A4 のスコープ外**（A4 では記録のみ）。

---

## A4-4 1-run-lag の位置づけ（topology の一致のみ）

両ログで `p15eq → p15ir` 境界を実測:

| ログ | 直前のケース | 直後のケース | 遷移 | 行番号 |
| --- | --- | --- | --- | --- |
| historical | `p15eq id=b9_am0 os=8` | `p15ir id=g0_os1_am-20 os=1` | **os 8 → 1** | L132 → **L235** |
| A3 | `p15eq id=b9_am0 os=8` | `p15ir id=g0_os1_am-20 os=1` | **os 8 → 1** | L132 → **L235** |

- **両 vehicle は同一の `os8 → os1` transition topology を持つ**（直前ケースの id・os・行番号まで一致）。
- **「os8 → os1 の遷移が 1.4759 dB を発生させた」とは結論しない。** A4 で確定したのは topology の一致まで。

---

## A4-5 追加実行なし（確認）

```text
2 回目の A3 run         : 未実施
3 回目の A3 run         : 未実施
--p1-char 再実行         : 未実施
P2 probe 再実行          : 未実施
--buzz-os / settle 変更  : 未実施
sleepPump / IR geometry 変更 : 未実施
```

- A3 が 1 回で参照値を完全再現しているため、追加反復の合理性は低いと判断（指示どおり）。

---

## A4 終了条件（MATCH 表）

```text
Historical vehicle
        │
        ├─ entry        = MATCH   (argv 不使用・kSr=48000・kBlock=512・Sine/1kHz・4+12 blocks・4096 窓)
        ├─ signal       = MATCH   (Signal::Sine / freq=1000.0 / amp=10^(ampDb/20))
        ├─ SR           = MATCH   (base 48000 / source 48000 / target 48000)
        ├─ OS           = MATCH   (os=1・IIR・n=0)
        ├─ IR lifecycle = MATCH   (loadImpulseResponse → isIRFinalized poll・sc0/sc1 前の計 2 回)
        ├─ settle       = MATCH   (waitBacklogZero(30000) → waitWorldPublished(seqBefore,30000) → sleepPump(800))
        ├─ geometry     = MATCH   (irLen 48000 / part 512 / fft 1024 / slot 11 / peak 0.5 / F_scale 2 値)
        ├─ p15ir rows   = 18/18 MATCH
        └─ Δgain        = +1.4759 dB MATCH   (delta-of-deltas = 0.000000 dB)
                         ↓
                 VEHICLE PROVEN
                         ↓
              mechanism = UNKNOWN
```

## A4 判定

```text
Step 5-A4 = PASS（provenance / vehicle consistency audit 完了）
```

- historical と A3 の vehicle は、entry / signal / SR / OS / IR lifecycle / settle / geometry /
  全 18 `p15ir` 行 / Δgain のすべてで一致し、**VEHICLE PROVEN** として確定。
- **mechanism = UNKNOWN を維持**（polyphase / IR scale / 1-run-lag / 3-J のいずれにも帰属しない）。
- `p15staging` の 18 件不一致は provenance observation として記録のみ（原因解析はスコープ外）。

## 現在の状態（次段階用）

```text
HEAD               = 1e9e63e3
kP15FullMatrix     = true（revert せず）
A3 binary          = b2c39a9a0e3b1fe3
production/CMake/JUCE diff = 0
A4                 = PASS
次段階              = Step 5-B / P3 attribution（A4 PASS 後に初めて着手）
```
