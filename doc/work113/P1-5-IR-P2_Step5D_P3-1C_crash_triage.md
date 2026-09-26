# P1-5-IR-P2 — Step 5-D / P3-1C-R1: Forward Crash Triage（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-1C-R1）
- **性質**: **完全 read-only**。再実行・source変更・build・logger追加は一切なし。
  調査対象は現行 source の読みと既存 `tmp/p3-1-c_forward.log` の読みのみ。
- **目的**: P3-1-C Forward（A3順序）で `p15ir` 完走・`Δ=+1.4759 dB` 再現の後に
  `p15hrnl os=8` で発生した `0xc0000005` を triage し、C1/C2/C3/C4 に分類する。
  Reverse の再投入可否判断のための材料整理であり、原因の断定はしない。
- **前提文書**: P3-1-B source audit（Step5C）、P3-1-A census（Step5C）、P3-0 freeze（Step5B）、
  P3-1-C STOP 報告（Step5D `..._reverse_order_experiment_stop.md`）、P2-J post-fix audit、P2-I teardown audit。

---

## 1. State Freeze（R1開始時・PASS）

```text
HEAD                         1e9e63e34bed7adb9342ebc81259ded9689fc48a
kP15FullMatrix               true（変更禁止・遵守）
CONVOPEQ_CORRECT_POLYPHASE_GAIN OFF（build/CMakeCache.txt）

A3 binary                    build/Release/AudioEngineHarness.exe b2c39a9a0e3b1fe3
A3 backup                    tmp/p3-1-c_A3_forward_baseline.exe b2c39a9a0e3b1fe3（一致再確認）

production source            0 diff
CMake                        0 diff
JUCE                         0 diff
settle/sleep                 0 diff
measurement semantics        0 diff

P3-1-C Forward               STOP（§8 crash該当）
Reverse                      未実施（P1PolyphaseGainCharacterization.cpp:746 は { -20.0, -6.0 } のまま）
P3-1-D                       未実施
H-B                          OPEN / 今回の判定対象外
```

`git diff` 上の4ファイルは A3 以前からの持ち越しのみ（production/CMake 差分 0 を再確認）。

---

## 2. Crash evidence（確定事実のみ）

| 項目 | 値 |
| --- | --- |
| 発生時刻 | 2026-09-23 13:17:08（Forward開始から約60分） |
| Windows診断 | Application Log ID 1000 / Faulting application `AudioEngineHarness.exe` / 例外コード `0xc0000005` / Faulting module `VCRUNTIME140.dll` / offset `0x1cca7` |
| CrashDump | なし（WERダンプ未生成） |
| 直前ログ | `tmp/p3-1-c_forward.log` L5338 `[P1CHAR] p15hrnl sat=1.00 os=8 n=3 ampDb=-20.0 …`（完走） |
| 直後ログ | IR再準備パイプライン（L5339〜L5359）後に断絶。次ケースの `[P1CHAR]`・`summary` なし |
| on-disk binary | hash不変（`b2c39a9a…` を事後に再確認） |

---

## 3. p15hrnl source path（A.1/A.2）

### 3-1. セクション投入順（source確定）

`runP1PolyphaseGainCharacterization` 内の出現順（`kP15FullMatrix` gate付きを含む）：

```text
(G) p15preset   :706
(H) p15eq       :725
(I) p15ir       :738
(J) p15staging  :757
(K) p15sat      :779
(K2) p15hrnl    :804
(L) p15lim      :858
(M) p15listen   :870
```

K2 ループ順序（:812-816）：`sat{0.1,1.0}` → `os{1,8}` → `adb{-20.0,-6.0,0.0}`（計12条件）。
crash は11番目の条件 `(sat=1.0, os=8, am=-6.0)` の sc0 ウィンドウ内で発生
（10番目 `(1.0,8,-20.0)` は L5338 で完走済み）。

### 3-2. K2 の1ケース経路（source確定）

K2 は `runPair` を使わず `runCase` を sc0/sc1 の2回直接呼ぶ（:820-829）。
`ensureTestIr` の呼び出しは K2 ブロック内（:804-856）に**存在しない**。
`ensureTestIr` を呼ぶのは `runPair`（:664 sc0前・:682 sc1前、ただし `irPath != nullptr` の場合のみ）と
`(M)` :887 のみ。したがって K2 実行中の IR 状態は section I 以来の持ち越しであり、
K2 自身による IR ロードは source 上存在しない。

1ケースの実順序（`runCase` :410-433）：

```text
previous case
→ configureChain()            :412（bulk restore内・下記A.4のintent投入）
→ waitBacklogZero(30000)      :414（戻り値はWARNのみ）
→ waitWorldPublished(seqBefore, 30000) :417（戻り値無視）
→ sleepPump(800)              :418
→ configureCapture/capture（warm 4 + cap 12 blocks）→ dftMag/gainDb
```

K2 の `runCase` 引数 mapping（:820-822 と signature :404-408 より）:
`sig=Sine, freq=1000.0, os=osF, type=IIR, sat=sat, headroom/makeup/trim/eqBoost=0.0f,
convBypass=true, capBlocks=kCapBlocks, eqBypass=false`。
すなわち convolver bypass・EQ identity（全band disabled）・staging neutral である。

### 3-3. `p15hrnl` の IR（A.2 結論）

- K2 による新規 IR ロードはなし（§3-2）。
- ログ上の IR 系行（`[IR_RATE_GEN]`/`[IR_TAIL_GEOM]`/`[IR_CHAIN]`/`[L0_WRITE]`）は
  structural rebuild に伴う再準備パイプラインの出力である。
  crash 境界では `sourceSr=384000 sourceLen=7 → targetLength=384000`、
  `F_scale=0.27427088`、`geom part=4096 numIR=12 numParts=16 fft=8192 imm=1 irLen=384000`、
  `slot=11 peak=1.082658`（L5339〜L5359）。
- 同一パターンは A3 ログの同一箇所（A3 L5721/5742/5763、後述§8）にも存在し、
  A3 はこれを通過して完走している。よって当該パイプライン自体は異常ではなく通常動作である。

---

## 4. Preceding-case topology（A.1 詳細）

crash 条件 `(1.0, 8, -6.0)` の直前条件 `(1.0, 8, -20.0)`（L5338完走）との差は `ampDb` のみ。
両条件とも `os=8` のため、境界での OS 変化はない。
K2 内の直近の OS 遷移は `(1.0, 1, 0.0)` → `(1.0, 8, -20.0)`（os 1→8）であり、
当該遷移先の初ケース（am-20）は正常完走している。

---

## 5. OS transition（A.3 結論）

```text
previous OS = 8 ／ crash-case OS = 8
→ setOversamplingFactor(8) は `!= newFactor` ガード（Parameters.cpp:547）により no-op
→ reprepareUiConvolverForProcessingGeometry() の実行なし
→ OS-change 由来の追加 Structural intent なし
```

**crash は OS transition 点では発生していない。**
（P3-0 §P3-0-6 の geometry-mismatch hazard 経路は本境界では動作していない。）

---

## 6. Structural intent topology（A.4 結論）

P3-1-B §2-1/2-2 の merge 規則を crash ケース（sc0: `softClip=false, sat=1.0, os=8`）に適用：

| 呼び出し | ガード | 判定 |
| --- | --- | --- |
| `setEqBypassRequested(false)` | なし | **intent** |
| `setConvolverBypassRequested(true)` | なし（前ケースと同値でも投入） | **intent** |
| `setAutoGainStagingEnabled(false)` 等の静的setter | ガードあり・値不変 | no-op |
| `setSoftClipEnabled(false)` | なし | **intent** |
| `setSaturationAmount(1.0)` | 前ケースと同値（`abs(diff)>1e-6` 不成立） | no-op |
| `setOversamplingFactor(8)` | 同値（§5） | no-op |
| `endBulkParameterRestore(true)` | クラス相違のため非merge | **intent** |

実効 dispatch ≈ 2 本（Snapshot class merge 1 本＋Structural class 1 本）。
これは P3-1-B で確定した通常ケースと同一であり、crash ケース固有の intent 異常はない。
（sc1 側も `softClip=true` の値変化は intent 本数・順序を変えない。同 §2-1。）

---

## 7. IR/lifetime transition（A.5・H-B照合のみ）

1ケースごとに発生する lifetime 関連遷移（共有経路）:

```text
configureChain の Structural intent
→ RuntimeBuilder が successful build ごとに新 DSPCore を生成（RuntimeBuilder.cpp:469）
→ world publication（runCase は waitWorldPublished の戻り値を無視：417）
→ capture（harness tap は limiter 下流：P3-1-A §1-3）
→ DSPCore 破棄に伴う旧 core/旧 world の retirement（deferred retirement worker）
```

H-B（P2-J 確定：OPEN・別チケット・`routerPendingRetire=2` / `coordinator Faulted`）との照合：

- Forward ログに `[FAULT]` / retirement / quarantine 系マーカーは **0 件**（§8 の audit 表参照）。
- H-B の既知 signature（上記2マーカー）は present しない。
- 以上は照合結果の記録であり、新しい因果の推測は行わない（指示どおり）。

---

## 8. Raw evidence audit（Triage B・`tmp/p3-1-c_forward.log` のみ）

| マーカー | count | 最終行 | 備考 |
| --- | --- | --- | --- |
| last completed P1CHAR | 131（p15系） | L5338 `p15hrnl sat=1.00 os=8 am-20.0` | 次条件 am-6.0 の P1CHAR なし |
| last IR load（ensureTestIr由来の特定） | — | 特定不能 | K2はIRロードなし（§3-2）。`[IR_RATE_GEN]`は再準備パイプラインとして全域に出現（計246） |
| last IR_FINALIZED | ABSENT | — | vehicleは当該prefixを出さない |
| last IR_RATE_GEN | 246件中 | L5339 `sourceSr=384000 … sourceLen=7` | crash境界のパイプライン先頭 |
| last IR_TAIL_GEOM | 246件中 | L5344 | 同上 |
| last L0_WRITE | 2570件中 | L5359 `slot=11 peak=1.082658` | geometry適用2回分（両pass相当）の後に断絶 |
| last CONV_IR / CONV_STATUS | ABSENT | — | P3-1-A確定どおり本vehicleは出さない（stderr logger未設置） |
| last PUBLISH | ABSENT | — | 同上 |
| last REBUILD_TELEMETRY | ABSENT | — | 同上 |
| last FAULT | ABSENT | — | H-Bマーカーなし |
| last retirement/quarantine marker | ABSENT | — | 同上 |
| summary | ABSENT | — | 異常終了のためなし |
| WARN / heap / corrupt / SIGSEGV / exception | ABSENT | — | ログ上の異常文言なし |

### A3 との同一箇所比較（read-only・既存ログ間比較）

- A3 の同一境界（am-20 L5720 → am-6 L5784）には同型パイプラインが **3 回**
 （L5721/5742/5763）現れ、A3 は am-6（L5784）・am-0 を完走して `cases=220 failures=0 / exit 0`。
- Forward の同一境界ではパイプライン1回目の途中で断絶。
- 別境界（Forward sat=0.10 os=8 の am-20 L5056 → am-6 L5099）ではパイプライン **2 回**で正常通過。
- すなわち case 境界あたりのパイプライン回数は run 間で非決定的（A3:3 / Forward別境界:2 / 致命境界:1回目で断絶）であり、
  決定的な section ロジックではなく非同期 loader/rebuild タイミング側のばらつきを示す。
  （行数カウントによる観測事実であり、新規計測ではない。）

---

## 9. Classification（C1/C2/C3/C4）

| 分類 | 判定 | 根拠 |
| --- | --- | --- |
| **C1**（`p15hrnl` 固有 path の明確な異常候補） | **非該当** | crash条件のtopology（§3〜§6）は130件以上の先行成功条件と同一。OS遷移なし（§5）、intent異常なし（§6）、IRロードなし（§3-2）。固有の異常候補は source 上見当たらない |
| **C2**（H-B と同一の既知 failure signature） | **不成立** | H-B の既知 signature（`routerPendingRetire` / `coordinator Faulted`）は Forward ログに不在（§7・§8）。「同時系列だった」だけでは成立させない（指示どおり）。`VCRUNTIME140.dll` の `0xc0000005` 単独は H-B の定義 signature ではない |
| **C3**（共通 rebuild/IR/retirement path の failure candidate） | **locus のみ記録・candidate 特定なし** | crash 点は共有 `runCase` 機構（rebuild＋IR再準備＋capture の重なり窓）上にあるが、当該ログからは特定の candidate を分離できない。`PUBLISH`/`REBUILD_TELEMETRY`/limiter診断は本 vehicle で取得不能（P3-1-A確定） |
| **C4**（原因不明） | **採用** | 上記により、現時点の read-only 証拠では原因を特定できない |

付帯観測（分類根拠ではなく flakiness  datum として記録）:
同一 binary＋同一 source で A3 は完走（exit 0）・本 Forward は後段で crash。
`p15ir` section は両 run で `gainDb` 18/18 一致と決定的である一方、
後段のパイプライン回数は非決定的（§8）。長時間 vehicle 後段の非決定性は確定事実とする。

---

## 10. P3-1-C final status（維持）

```text
P3-1-C = STOP
Reverse = NOT EXECUTED
Pattern = INCONCLUSIVE
```

Forward について確定してよい事実（A3 の再現確認として扱う）:

```text
g0/os1/-20  = -14.5035 dB
g0/os1/-6   = -13.0276 dB
Δ            = +1.4759 dB
```

以下には分類しない（維持）:

```text
limiter原因 / state/history依存 / order-dependent のいずれにも分類しない
```

---

## 11. Reverse execution decision

```text
この triage の結果が出るまで Reverse-order experiment も P3-1-D 計装も実施しない（指示どおり）。
本 triage の結論（C4）を受けての次手順選択は P3-1-C 判定に委ねる。
```

- C4 のため、Reverse を同一長時間 vehicle に再投入しても crash の再発により
  比較不能となる交絡リスクが残る（本 triage の記録事項）。
- P3-1-A の X-1/X-2/X-3 はいずれも選択しない（P3-1-C 指示 §7 遵守）。
  特に X-3（limiter envelope accessor/log）は不要を維持する。
  理由：`reverse-order evidence = 0` のまま limiter を計測しても、
  問い「`-20 → -6` の順序が 1.4759 dB anomaly に関係するか」を直接解決しない。

---

## 12. Prohibited actions（遵守記録）

```text
Reverse {-6,-20} 再実行 0 / Forward repeat 0 / p15ir 切り出し改変 0
p15hrnl 削除/skip 0 / settle timeout 延長 0 / sleepPump 変更 0
limiter reset 0 / logger/accessor追加 0 / production source変更 0
test-only source変更 0 / crash dump用設定変更 0 / 新規CLI 0
H-B 再実行 0 / kP15FullMatrix 変更 0
```

STOP 条件に触れる事項は発生しなかった（triage 自体が read-only で完結した）。

---

## 最重要の解釈制約（P3-1-C §7・再掲）

```text
RuntimeBuilder は successful build ごとに新 DSPCore を生成する。
したがって rebuild 完了時には SimplePeakLimiter の envelope は
新規状態に戻る。

runCase の waitWorldPublished() 戻り値は無視されるため、
ケース境界で rebuild が完了したかは今回の測定ログだけでは確定できない。

したがって reverse-order の結果だけから
「limiter envelope のケース間持続」を肯定/否定してはならない。
```
