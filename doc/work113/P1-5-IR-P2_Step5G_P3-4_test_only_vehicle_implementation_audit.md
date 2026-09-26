# P1-5-IR-P2 — Step 5-G / P3-4: Test-Only Vehicle Implementation Audit（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-4）
- **種別**: 完全 read-only。実装・build・実行なし。
- **目的**: P3-3 §10 の4項目が本当に最小の test-only diff で実現できるかを確定する。
  特に `waitWorldPublished()` を成功条件化した F/R vehicle が
  「意図した world を捕捉した」とどこまで保証できるかを詰め、G-A/B/C/D の1つに判定する。
- **前提**: P3-1-C=STOP / R1=COMPLETE(C4) / P3-2=COMPLETE(M3) / P3-3=COMPLETE(V-B) /
  Reverse=NOT EXECUTED / P3-1-D=NOT STARTED / H-B=OPEN・対象外。

---

## 1. State Freeze

```text
HEAD                         1e9e63e34bed7adb9342ebc81259ded9689fc48a
kP15FullMatrix               true
CONVOPEQ_CORRECT_POLYPHASE_GAIN OFF
A3 binary                    b2c39a9a0e3b1fe3
production/CMake/JUCE        0 diff
```

P3-3 後の source/test/build 変更なしを確認（同一4ファイルの持ち越しのみ）。
IR資産 `tmp/p15_ir_g0.wav` は P1 TU 外部の既存資産であり、
P1 TU 内に生成コードは存在しない（`writeWav16` は listen 出力専用 :288）。
よって g0 限定 vehicle に IR 生成依存はない。

## 2. P3-3 V-B の再確認

P3-3 の結論（V-B：test-only変更で分離可能・production 0・計装不要）を維持する。
本 P3-4 はその4項目（§10）の実装前監査であり、仕様を固定してSTOPする。
P3-3 §10 に対する本監査の唯一の修正は §4 の B-scope refinement である。

## 3. adb Reverse変更点（候補A・確定）

- 現行 `:746`：`for (double adb : { -20.0, -6.0 })`。
- Reverse 用：`for (double adb : { -6.0, -20.0 })` の1 token 順序入替のみで足りる。
- 他の `adb` loop（:729 p15eq・:767 staging・:786 sat・:816 K2 等）には触れない。
  g0/os1 限定後は section I の当該1行のみが測定対象となる。
- 新CLIは不要・禁止どおり追加しない。production diff 0 を汚さない。

## 4. `waitWorldPublished` return-honor変更点（候補B・範囲限定付き確定）

### 4-1. 無視箇所の特定（source）

- `:414` `waitBacklogZero` の戻り値 → WARN のみで継続。
- `:417` `waitWorldPublished(e, seqBefore, 30000);` → **戻り値完全無視**（文として呼ぶのみ）。
- capture（:420-463）は戻り値によらず実行される。

最小 diff：`:417` を `const bool published = waitWorldPublished(...);` とし、
`!published` 時に capture せず `return false` する（〜3行）。
`runPair` は `!a/!b → ++failures; return` の既存経路（:670/:688）で吸収できるため、
`runPair` 構造の変更は不要である。

### 4-2. 成功の意味の限定（維持・再確認）

成功条件はあくまで以下であり、これを拡大解釈しない：

```text
publication sequence changed（seqBefore比）
AND
backlog == 0（ポーリング時点）
```

成功は「今回の configureChain が新worldを publish した」ことを意味しない
（merge-no-dispatch・他intent源の publish と区別不能。P3-3 §3-2 維持）。
よって「publish-honored」の意味を以下に限定する：

```text
今回の capture は publish 未確認状態では取得しない
```

### 4-3. B-scope refinement（本監査の修正・重要）

P3-3 §10 の無条件 honor には paradox がある。P3-3 §3-3 の既定動作より：

- 第1ケース（遷移を伴う：Fでは−20、Rでは−6）は dispatch 期待 → **strict-honor**。
  timeout時は当該 run を無効化し capture 比較に進まない。
- 第2ケース（同一パラメータ：Fでは−6、Rでは−20）は merge-no-dispatch が既定動作であり、
  publish は発生しないため wait は **30 s timeout 確定**である。
  無条件 honor では測定すべき第2ケースが必ず無効化される。
  よって第2ケースは「sequence 不変＋backlog 0」の確認（既存2カウンタのみ）に切替え、
  timeout-with-unchanged／success（再dispatch分岐）の両分岐を受理して記録する。
  いずれも world は第1ケースと同一（継承）であり、R0継承が成立する。

この scoping が P3-3 §10 への唯一の修正である。これにより B は G-A 適合となる。

## 5. post-settle必要量の再計算（候補C・確定・非証明宣言付き）

source由来の数値（CrossfadeRuntime.h:218／:216・kSr/kBlock :34-35・tap :192-194・:439）：

```text
fadeTimeSec      0.030（既定。testはfadeを設定しないため既定経路。world別上書きの余地あり）
sample rate      48000 Hz
block size       512（block周期 10.667 ms）
ramp duration    ≒ fadeTimeSec ≒ 0.030 s ≒ 1440 samples ≒ 2.81 blocks
startDelayBlocks 既定 0（structural publishで変更なし・仮定として記録）
capture warm-up  4 blocks（解析窓外）
analysis window  先頭12 blocks中の blocks 4–11（末尾4096 samples）
```

時間関係の要求：ramp完了 ≺ 解析窓開始（captured block 4 開始）。
wait-success後の必要カバー：poll粒度（10 ms）＋ arm遅延（≦1 block）＋
startDelay（仮定0）＋ ramp（≦3 blocks）＋ 安全率。
既存 `sleepPump(800)`（≒75 blocks）に加え、wait-success後の bounded 追加 margin
（例：500 ms ≒ 47 blocks ≈ 必要量の15倍）を置くことで、
既定 fade に対する時間的余裕は source 計算上確保される。

ただし固定 `sleep(XX)` を ramp完了の証明とは扱わない（指示どおり）。
これは観測を伴わない **bounded timing margin** であり、
ramp完了そのものの観測には accessor が要る（＝実行要件ではなく証明要件のため E3 不要・P3-3 §11維持）。
startDelay の非ゼロ・fade 上書きは残余リスクとして記録し、次gateの承認事項とする。

## 6. g0/os1限定方法（候補D・確定）

- 新CLIなし。`P1PolyphaseGainCharacterization.cpp` の hardcode のみ：
  section G/H/J/K/K2/L/M の抑止＋ section I の `irs[]` 単一化（g0）・`osF{1}` 化。
  `kP15FullMatrix` は true のまま（変更禁止）。
  `runPair("p15ir", …)` の使用を維持し、ensureTestIr×2・sc0/sc1・diffStats の
  topology を保全する。概算15–25行。
- 目的適合：70分級全行程・`p15hrnl`・H-B既知不安定領域を vehicle から排除する。
- IR生成依存なし（§1）。

## 7. F vehicle定義（仕様固定・未実行）

```text
fresh process（engine default world から開始）
→ g0 / os1 / -20 dB（第1ケース：strict-honor＋bounded post-settle＋capture）
→ g0 / os1 / -6 dB（第2ケース：sequence-unchanged確認＋capture）
```

- 第1ケースの configureChain は F/R で同一（ampDbはcapture側のためパラメータ外）。
  よって初期遷移は両vehicleで対称である。
- A3 の os8 前提は再現しない（意図的。問いは安定条件下の order 依存性である。§9理由）。

## 8. R vehicle定義（仕様固定・未実行）

```text
fresh process（engine default world から開始）
→ g0 / os1 / -6 dB（第1ケース：strict-honor＋bounded post-settle＋capture）
→ g0 / os1 / -20 dB（第2ケース：sequence-unchanged確認＋capture）
```

- F と同一 binary 系統の別 build（adb順序のみ相違）。実行は承認・build・identity確認の後。
- 比較対象は `Forward Δ vs Reverse Δ`（Pattern 1/2/3 分類は実行後の判定gate）。

## 9. F/R交絡表

| 軸 | F | R | 制御方法 |
| --- | --- | --- | --- |
| IR | g0 | g0 | 固定（§6） |
| OS | 1 | 1 | 固定（§6） |
| amp順序 | −20→−6 | −6→−20 | 反転（§3・別build） |
| process | fresh | fresh | 固定（§7・§8） |
| initial world | 同一条件（default） | 同一条件（default） | 固定（§7・§8） |
| publish確認 | 必須（第1厳格・第2継承確認） | 必須（同左） | `waitWorldPublished`（§4） |
| crossfade時間 | bounded（§5 margin） | bounded（§5 margin） | post-settle |
| production code | 0 diff | 0 diff | test-only |
| H-B | 除外 | 除外 | g0/os1限定（§6） |

同じもの：第1ケースの初期遷移・world・settle・topology・sc構成。
違うもの：amp順序のみ（およびそれに随伴する第1/第2の履歴位置＝測定対象そのもの）。
第2ケースの history（第1のtail）は両vehicleで対称に存在し、order効果の分離を妨げない。

## 10. production diff = 0 の確認条件

- 変更許可：`src/tests/AudioEngineHarness/P1PolyphaseGainCharacterization.cpp` のみ
  （§3・§4・§5・§6の範囲。概算20–30行）。
- 確認：`git diff --name-only` が当該1ファイルのみ＋
  `src/audioengine・src/convolver・src/dsp・src/core・CMake・JUCE` の差分0。
- `kP15FullMatrix=true` 維持・`CONVOPEQ_CORRECT=OFF` 維持・settle/sleep の意図外変更なし。

## 11. test-only diff の正確なファイル・関数・変更範囲

| # | 場所 | 変更 | 行数目安 |
| --- | --- | --- | --- |
| 1 | `:746` adb loop | Forward／Reverse 順序（build別） | 1 |
| 2 | `runCase :417` 付近 | 戻り値honor＋無効化 early-return（§4-3 scope付き） | 〜5 |
| 3 | `runCase :418` 付近 | wait-success後 bounded post-settle（§5） | 〜5 |
| 4 | `runP1… :706-902` 領域 | section抑止＋I の g0/os1 限定（§6） | 〜15–25 |

新規関数・新規CLI・logger・accessor なし。`runPair` 構造不変。

## 12. build/identity gate

binary別（F用・R用）に A3-2 手順を踏襲：Release/OFF・当該1 TU再compile期待・
`p15ir` 文字列含有・SHA-256全値＋prefix記録・HEAD・production差分0・
`kP15FullMatrix`・`CONVOPEQ_CORRECT` の確認。identity不一致時は実行に進まない。

## 13. 実行前STOP条件

```text
binary identity不一致／production差分／kP15変化／意図外settle/sleep変化
crash・heap corruption・ASan failure／p15ir行欠落／IR geometry変化
第1ケース publish-honor timeout（当該run無効・比較に進まない）
compensation／normalization の必要化／新規logger・accessor の必要化
```

## 14. 次段階への承認要求＋最終Gate

承認要求：§11の test-only diff 範囲・§4-3 の scope・§5 の margin（非証明宣言付き）・
§6 の限定・§12 の gate・§13 の条件。
承認後に実装→build→identity→F/R実行の順（本 P3-4 では実行しない）。

```text
G-A（test-only最小diff確定→実装承認へ）: ADOPTED
  A:1 token・CLIなし（§3）／B:scope付きhonorでparadox解消（§4-3）／
  C:margin計算閉鎖＋非証明宣言（§5）／D:production変更なし（§6・§10）
G-B（waitだけでは不十分→再設計）: REJECTED（§4-3で充足）
G-C（post-settle証明不能→計装再検討）: REJECTED（実行に証明は不要・§5・P3-3 §11）
G-D（production変更が必要→STOP）: REJECTED（§10・§11）
```

G-A は仮定ではなく上記再確認の帰結である。§7解釈制約
（新core／publish戻り値無視／envelope持続の肯定・否定禁止）を維持する。
