# P1-5-IR-P2 — Step 5-D / P3-1-C: Reverse-Order Experiment — STOP（Forward後段crash）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-1-C）
- **判定**: **STOP** — P3-1-C §8 STOP条件「crash / heap corruption / ASan failure」に該当。
  Reverse の build/run は実施していない。P3-1-D には進んでいない。
- **変更**: source / CMake / production / settle / sleep = **0 変更**（test-only変更も未適用）。
  実行は既存 A3 binary の as-is 実行（Forward 1 run）のみ。
- **raw evidence**: `tmp/p3-1-c_forward.log`（5359行・summaryなし・異常終了）、
  `tmp/p3-1-c_A3_forward_baseline.exe`（A3 binaryバックアップ）。
- **性質**: P3-1-B source audit（必要条件PASS）を受けた P3-1-C の実施記録。H-B は判定対象外。

---

## 0. State Freeze（PASS）

| 項目 | 値 |
| --- | --- |
| HEAD | `1e9e63e34bed7adb9342ebc81259ded9689fc48a`（prefix `1e9e63e3` 一致） |
| `kP15FullMatrix` | `true`（`P1PolyphaseGainCharacterization.cpp:43`・一意） |
| A3 binary SHA-256[:16] | `b2c39a9a0e3b1fe3` |
| `CONVOPEQ_CORRECT_POLYPHASE_GAIN` | `BOOL=OFF`（`build/CMakeCache.txt`） |
| production source / CMake / JUCE / default / calibration diff | **0** |
| settle / sleep diff | **0** |
| measurement semantics diff | **0** |

```text
A3 = REPRODUCED / A4 = PASS / P3-0 = PASS / P3-1-A = PASS / P3-1-B = PASS
H-A = CLOSED / H-B = OPEN（今回の判定対象外）
```

補足：`git diff` 上の4ファイル（`BassBuzzMeasurement.cpp` /
`P1PolyphaseGainCharacterization.cpp` / `PublishPipelineIntegrationTests.cpp` /
`check_layout_offsets.py`）はいずれも A3 ビルド以前からの持ち越しであり、
A3 binary は現 worktree から build されたもの（P1 mtime 07:59:04 → binary 07:59:39）。
production / CMake の差分は 0。`runCase` は
`configureChain:412` → `waitBacklogZero:414` → `waitWorldPublished:417`（戻り値無視）→
`sleepPump(800):418` で P3-1-B audit と一致。

---

## 1. Binary identity（A3 = Forward baseline）

```text
binary path  build/Release/AudioEngineHarness.exe
SHA-256      b2c39a9a0e3b1fe3c8254eb7425f9efc3f49d743c9de7bc9a86a83fde671383c
git HEAD     1e9e63e3
git diff -- production source  0
kP15FullMatrix  true
CONVOPEQ_CORRECT_POLYPHASE_GAIN  OFF
```

- IR資産：`tmp/p15_ir_g0.wav` 8236 bytes / SHA-256[:16]=`2e526f4de5a97afb`（P2既報と一致）、
  `tmp/p15_ir_g3.wav` `988966b2ee03ab6f` / `tmp/p15_ir_g6.wav` `402ac39580e5852d`。
- A3 binary のバックアップを `tmp/p3-1-c_A3_forward_baseline.exe` に保存（hash一致確認済み）。
- クラッシュ発生後に on-disk binary の hash を再確認し、不変（`b2c39a9a…`）であることを確認。

---

## 2. 実験F: Forward（A3と同順序 `{ -20.0, -6.0 }`、1 run）

invocation：`build\Release\AudioEngineHarness.exe --p1-char`（他旗なし、cwd=リポジトリルート）。

### 2-1. 必須値（仮説対象ペア）

| order | case id | ampDb | gainDb_sc0 | limitingEng_sc0 | hardClamp_sc0 | clipEngagement | clipEngMax | os | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1st（L235） | g0_os1_am-20 | -20.0 | **-14.5035** | 0 | 0 | 4096 | 0.024784 | 1 | 0 |
| 2nd（L356） | g0_os1_am-6 | -6.0 | **-13.0276** | 0 | 0 | 4096 | 0.163382 | 1 | 0 |

```text
-20 : -14.5035 dB（A3既知値と一致）
 -6 : -13.0276 dB（A3既知値と一致）
Δ    : +1.4759 dB（既知基準どおり再現）
```

- 行番号は A3/historical と同一（L235/L356）。
- 直前ケースは `p15eq id=b9_am0 os=8`（L132、A3と同一topology）。
- `gainDb_sc1` も A3 と一致（-15.5265 / -15.5220）。

### 2-2. IR geometry / F_scale（forward vs A3：一致）

```text
[IR_RATE_GEN] gen=0 sourceSr=48000 targetSr=48000 actualSr=48000 sourceLen=1 convertedLen=1 ratio=1.0000 resampled=no
[IR_TAIL_GEOM] gen=0 loadedSr=48000 loadedLen=1 targetLength=48000 copySamples=1 fadeSamples=0 fadeStart=1 fadeEnd=1
[L0_WRITE] geom part=512 numIR=12 numParts=16 fft=1024 imm=1 irLen=48000
[L0_WRITE] slot=11 peak=0.500000 bin=0
[IR_CHAIN] F_scale scaleFactor=0.50013092 / 0.50026187（sc0/sc1両パス、A3と同一）
```

### 2-3. 付随観測（記録のみ・解釈しない）

- `p15ir` 18行の `gainDb_sc0/sc1` は A3 と 18/18 一致。ただし非対象6行（os4/os8 の am-6）で
  `clipEngagement` の count のみ微差（例：g0_os4_am-6 Forward 2150 vs A3 2089。
  gainDb・clipEngMax は同一）。g0_os1 ペアは全field一致。
- §4 の既知値（g0_os1 の gainDb/Δ）は再現しているため、STOP条件
  「Forward自体がA3の既知値を再現しない」には**非該当**。

---

## 3. STOP条件の発火（crash）

Forward は `p15ir`（18/18）を含む前半を完走したが、後段で異常終了した：

```text
最終P1CHAR  L5338 [P1CHAR] p15hrnl sat=1.00 os=8 n=3 ampDb=-20.0 …（次ケースIRロード中に終了）
時刻        2026-09-23 13:17:08（開始から約60分）
事象        Faulting application: AudioEngineHarness.exe / 例外コード 0xc0000005（access violation）
            Faulting module: VCRUNTIME140.dll / offset 0x1cca7
証跡        Windows Application Log ID 1000。CrashDumpなし。summary行なし。
```

これは P3-1-C §8 の「crash / heap corruption / ASan failure」に該当する。
よって**その時点で停止**し、**Reverse の追加実行は行わない**（§8遵守）。
publication sequence / world generation / limiter envelope の取得も行っていない（指示どおり）。

---

## 4. Reverse（R）：未実施

```text
P1PolyphaseGainCharacterization.cpp:746 は Forward のまま（{ -20.0, -6.0 }）。
test-only変更・Reverse用build・binary identity取得・Reverse run = すべて未実施。
production source・CMake・JUCE・settle/sleep・measurement semantics = 0 diff維持。
```

変更禁止項目（`runCase` / `runPair` 構造 / `ensureTestIr` / `configureChain` /
`waitBacklogZero` / `waitWorldPublished` / `sleepPump(800)` / `dftMag` / `gainDb` /
limiter / `DSPCore` / `RuntimeBuilder` / IR loading・新規CLI option・logger追加・
limiter envelope accessor/log・production source）はすべて 0 変更。

---

## 5. Pattern分類

```text
Pattern 1 / 2 / 3 の判定には Forward vs Reverse の比較が必須だが、
Reverse が欠測のため分類不能 → INCONCLUSIVE
```

（P3-1-B §3-3 の規則「3値に明確に分類できない場合は INCONCLUSIVE、勝手に repeat しない」に従う。
Forward単独の事実としては「g0_os1ペアの Δ=+1.4759 dB が A3 binary 上で再現」まで。原因帰属は行わない。）

---

## 6. 最重要の解釈制約（P3-1-C §7、結果の如何に関わらず明記）

```text
RuntimeBuilder は successful build ごとに新 DSPCore を生成する。
したがって rebuild 完了時には SimplePeakLimiter の envelope は
新規状態に戻る。

runCase の waitWorldPublished() 戻り値は無視されるため、
ケース境界で rebuild が完了したかは今回の測定ログだけでは確定できない。

したがって reverse-order の結果だけから
「limiter envelope のケース間持続」を肯定/否定してはならない。
```

今回は Reverse 未実施のため、上記に加えて
「Forward の再現だけから limiter が原因と断定しない」ことも明記する。
`p15ir` 18/18 の gainDb 一致および g0_os1 の Δ 再現は、機構の肯定/否定の根拠にはならない。

---

## 7. P3-1-C判定への申し送り

1. Forward baseline（A3順序）は仮説対象部について bit-exact に再現：`Δ=+1.4759 dB` 確定。
2. クラッシュは仮説対象部（`p15ir`）の完了**後**の `p15hrnl os=8` で発生。
   H-B=OPEN の既知不安定領域と整合するが、帰属はしない（H-B は対象外）。
3. Reverse 未達のため Pattern 分類は INCONCLUSIVE。
   次の判断（再試行の要否、P3-1-D X-1/X-2/X-3 の選択）は P3-1-C 判定に委ねる。
   本 Step では P3-1-D の logger/accessor 追加に進んでいない。

## 8. 禁止事項の遵守

```text
production source 0 / DSPCore 0 / SimplePeakLimiter 0 / RuntimeBuilder 0 / ConvolverProcessor 0
AudioEngine 0 / CMake 0 / JUCE 0 / runCase settle 0 / waitBacklogZero 0 / waitWorldPublished 0
sleepPump 0 / dftMag 0 / gainDb formula 0 / IR compensation 0 / gain compensation 0
normalization 0 / new CLI flag 0 / kP15FullMatrix 変更 0 / CONVOPEQ_CORRECT_POLYPHASE_GAIN 0
limiter envelope ログ追加 0 / reverse-order 変更 0（未実施）/ P3-1-D 計装 0
```
