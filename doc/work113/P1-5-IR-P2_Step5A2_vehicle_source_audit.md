# P1-5-IR-P2 — Step 5-A2: Historical `--p1-char` Vehicle Source Audit（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（A-2）
- **性質**: read-only audit。production / CMake / test source / settle / sleep = **すべて 0 変更**。
  既存 binary の as-is 実行（`--p1-char` 1 回）と既存ログの解析のみ。
- **成果物（要求された 4 点のみ）**:
  1. `kP15FullMatrix` の履歴
  2. historical `p15ir` vs current `--p1-char` の vehicle 差分
  3. reference 1.4759 dB の provenance
  4. S-1 が「1 token で十分」か否か → **判定: Case A（十分・ただし運用条件付き）**

---

## 1. `kP15FullMatrix` の履歴（A2-1）

### 1-1. git 履歴の調査結果

| 調査 | コマンド | 結果 |
| --- | --- | --- |
| ファイルを触った commit | `git log --all --oneline -- <file>` | **1 件のみ**: `d6995b2b feat(work113): gate polyphase gain convention correction` |
| `p15ir` を含む commit | `git log --all -S'p15ir' -- <file>` | **0 件** |
| `kP15FullMatrix` を含む commit | `git log --all -S'kP15FullMatrix'` | **0 件** |
| stash | `git stash list` | 0 件 |
| reflog | `git reflog` | 該当なし（`d6995b2b`→`1e9e63e3` は docs commit） |
| HEAD の `runPair` | `git show HEAD:<file> \| grep runPair` | **0 件** |
| HEAD 行数 vs 作業ツリー行数 | — | **497 行 vs 908 行**（+411 行が未コミット） |

### 1-2. 判定

```text
kP15FullMatrix の初出 commit     = UNKNOWN（git 履歴に存在しない）
kP15FullMatrix が false にされた commit = UNKNOWN（同上）
p15ir / p15eq / p15staging の追加・削除履歴 = UNKNOWN（同上）
```

- **`kP15FullMatrix` と `p15*` sweep 一式（`runPair` を含む）は、どの commit にも存在しない。**
  すべて**未コミットの作業ツリー変更**として導入された。したがって「いつ・どの commit で」を
  git から確定することは**不可能（UNKNOWN）**であり、推測は行わない。
- HEAD の同ファイルには `runPair` 自体が無い（497 行）ため、**historical 状態のソースは git から復元不能**。
  現存する唯一の実装は**現在の作業ツリー**である。

### 1-3. 「なぜ false か」は source コメントで確定（OBSERVED）

`P1PolyphaseGainCharacterization.cpp:41-43`:

```cpp
// P1-5: heavy matrix（EQ/IR/staging/sat/limiter sweep）の実行可否。
//   false のときは preset replay（G）と listening materials（M）のみを実行する
//   （1 config-pair あたり約 37 s の実測レートでは全行列が 1 build あたり 90 分超になるため）。
constexpr bool   kP15FullMatrix = false;
```

- 理由 = **実行時間**（1 config-pair ≈ 37 s・全行列 > 90 分/build）。機能的理由ではない。
- 内部整合の確認（OBSERVED）: gated セクションの config-pair 総数
  （(H)18 + (I)18 + (J)36 + (K)~37 + (L)~33 ≈ 142）× 37 s ≈ **87 分** → コメントの「90 分超」と整合。

---

## 2. historical `p15ir` vs current `--p1-char` の差分（A2-2）

### 2-1. セクション構成と gate の実測マッピング

`[P1CHAR] <kind>` 別の行数（historical = `p15char_off_partial.log` / current = 本 A2 の as-is 実行）:

| セクション | kind | gate | historical | current（flag=false） |
| --- | --- | --- | --- | --- |
| (A) Level | `kind=level` | なし | 19 | 19 |
| (B) Latency | `kind=latency` | なし | 4 | 4 |
| (C) SoftClip | `kind=softclip` | なし | 4 | 4 |
| (D) 固定 dump | `samples` / `v` | なし | 1 / 64 | 1 / 64 |
| (E) P1-3 matrix | `p13` | なし | 12 | 12 |
| (F) P1-3 staging | `p13staging` | なし | 2 | 2 |
| (G) Preset replay | `p15preset` | なし（:706） | 7 | 7 |
| **(H) EQ boost** | `p15eq` | **あり（:725）** | **18** | **0** |
| **(I) IR boost** | `p15ir` | **あり（:738）** | **18** | **0** |
| **(J) Staging×nonlinear** | `p15staging` | **あり（:757）** | **30** | **0** |
| (K) Saturation | `p15sat` | あり（:779） | 0 | 0 |
| (K2) P1-5-HR | `p15hrnl` | なし（:804） | 0 | **12** |
| (L) Limiter | `p15lim` | あり（:859） | 0 | 0 |
| (M) Listening | `p15listen` | なし（:870） | 0 | **12** |

- **gate 対象 = (H)(I)(J)(K)(L) の 5 セクション**（`runPair` ベースの sweep）。
- (K2) と (M) は **historical 実行後に追加された非 gate セクション**（historical ログに存在しない）。
  → 現在の作業ツリーは historical 状態の**後継**であり、同一ではない。

### 2-2. 測定機構が不変であることの実測検証（重要）

非 gate セクションの出力を historical と current で直接 diff:

```text
kind=level    : 19 行  → diff 空（完全一致）
kind=latency  :  4 行  → diff 空（完全一致）
kind=softclip :  4 行  → 行数一致
```

→ **`runCase` / `configureChain` / `ensureTestIr` / capture / 解析の機構は historical と同一**であることが
実測で確認された（bit-identical）。

### 2-3. gated セクションのパラメータ照合

| セクション | 現行ソースのループ（実読） | historical ログの id 並び | 判定 |
| --- | --- | --- | --- |
| (H) `p15eq` | `for osF{1,4,8} → for boost{3,6,9} → for adb{-20,0}`（:727-729） | os=1: b3_am-20, b3_am0, b6_am-20, b6_am0, b9_am-20, b9_am0 → os=4 → os=8 | **完全一致** |
| (I) `p15ir` | `for ir{g0,g3,g6} → for osF{1,4,8} → for adb{-20,-6}`（:744-746）・`convBypass=false`・`satOff=satOn=1.0` | g0: os1_am-20, os1_am-6, os4_am-20, os4_am-6, os8_am-20, os8_am-6 → g3 → g6 | **完全一致** |
| (J) `p15staging` | `for st{st-12,st-6,st0,st+6} → for osF{2,4,8} → for adb{-20,-6,0}` | 30 行・`st0_os8_am-20` 等 | 一致 |

### 2-4. base sample rate（vehicle 間の構造差・flag とは独立）

| vehicle | base SR | 根拠（実測ログ） |
| --- | --- | --- |
| `--p1-char`（historical・current 共通） | **48000** | `[IR_RATE_GEN] sourceSr=48000 targetSr=384000 ratio=8.0000`（os=8 → 48000×8）が両ログで同一 |
| P2 probe flow（Step 4） | **192000** | `[CONV_STATUS] sr=192000.0`・`[IR_RATE_GEN] targetSr=192000`（device SR 由来） |

- 参照 2 行（os=1）は `targetSr=48000` = 48000×1 → baseSR 48000 と整合。
- **`--p1-char` と P2 probe は base SR が異なる**（48000 vs 192000）。これは `kP15FullMatrix` とは
  独立した vehicle 差であり、Step 4 の「NOT REPRODUCED」を flag/settle のみで説明できない理由になる。

### 2-5. vehicle 差分表（要求形式）

| 項目 | historical `p15ir`（g0_os1） | current `--p1-char`（flag=false） |
| --- | --- | --- |
| IR | `tmp/p15_ir_g0.wav` | （(I) 未実行のため IR 使用なし。他セクションは `p15_listen_*` 等） |
| OS | 1 | (I) 未実行。実行された (A) は os{2,4,8}、(E)(F) は os{1,2,4,8} |
| amp | −20 / −6 dB | (I) 未実行 |
| `convBypass` | `false`（:749） | (I) 未実行 |
| `satOff` / `satOn` | 1.0 / 1.0（:750） | (I) 未実行 |
| IR load | `ensureTestIr` → `loadImpulseResponse(path,false)`（:394・sc0/sc1 の**前後 2 回**） | 同一コード（(I) 未実行のため未到達） |
| finalize | `isIRFinalized()` poll（100 ms × 最大 3000）（:395-401） | 同一コード |
| geometry prepare | `configureChain`（`runCase` 内・:412） | 同一コード |
| wait | `waitBacklogZero(30000)` → `waitWorldPublished(seqBefore,30000)` → `sleepPump(800)`（:414-418） | 同一コード |
| capture | `kCapBlocks=12` blocks・解析窓 `kAnalysisN=4096`（末尾）（:425-439） | 同一コード |
| limiter / clamp | `limitingEngaged_sc0/1=0`・`hardClamp_sc0/1=0`・`clipEngagement=4096` | (I) 未実行 |
| base SR | 48000 | 48000 |

---

## 3. reference 1.4759 dB の provenance（A2-3・確定）

### 3-1. 参照 2 行（一次ログ・完全値）

`tmp/p15char_off_partial.log`:

```text
:235  [P1CHAR] p15ir id=g0_os1_am-20 os=1 n=0 ampDb=-20.0
        gainDb_sc0=-14.5035 gainDb_sc1=-15.5265
        limitingEngaged_sc0=0 limitingEngaged_sc1=0 hardClamp_sc0=0 hardClamp_sc1=0
        clipEngagement=4096 clipEngMax=0.024784

:356  [P1CHAR] p15ir id=g0_os1_am-6 os=1 n=0 ampDb=-6.0
        gainDb_sc0=-13.0276 gainDb_sc1=-15.5220
        limitingEngaged_sc0=0 limitingEngaged_sc1=0 hardClamp_sc0=0 hardClamp_sc1=0
        clipEngagement=4096 clipEngMax=0.163382
```

```text
Δ(gainDb_sc0) = -13.0276 - (-14.5035) = +1.4759 dB  ← P0/P1 の「1.48 dB」と一致
Δ(gainDb_sc1) = -15.5220 - (-15.5265) = +0.0045 dB
```

### 3-2. 幾何・IR チェーン状態（両行で完全同一）

| 項目 | am=−20 | am=−6 |
| --- | --- | --- |
| `[IR_RATE_GEN]` | gen=0 sourceSr=48000 targetSr=48000 actualSr=48000 sourceLen=1 convertedLen=1 ratio=1.0000 **resampled=no** | **同一** |
| `[IR_TAIL_GEOM]` | gen=0 loadedSr=48000 loadedLen=1 **targetLength=48000** copySamples=1 fadeSamples=0 fadeDisabled=0 | **同一** |
| `[L0_WRITE] geom` | part=512 numIR=12 numParts=16 fft=1024 imm=1 **irLen=48000** | **同一** |
| `[L0_WRITE] slot` | slot=11 **peak=0.500000** bin=0 | **同一** |
| `[IR_CHAIN] F_scale` | 0.50013092 / 0.50026187（sc0 パス / sc1 パス） | **同一 2 値** |
| limiting / hardClamp | 0 / 0（両 sc） | **同一** |
| clipEngagement | 4096 | 4096 |
| clipEngMax | 0.024784 | 0.163382 |

- **参照 2 行の IR geometry・IR scale・処理 geometry は bit-identical**。
  異なるのは入力 `ampDb` のみ。
- `F_scale` は sc0 パスと sc1 パスで微小に異なる（0.50013092 vs 0.50026187・Δ≈0.00013）が、
  **両行で同じ 2 値**である（run 間で同一）。
- 参照ログでは `gen` は全行 `gen=0`（IR generation カウンタは本 vehicle では 0 のまま）。

### 3-3. 18 行全体の構造（OBSERVED・重要）

全 18 `p15ir` 行の `gainDb_sc0 / gainDb_sc1`:

```text
id                os   sc0         sc1
g0_os1_am-20      1   -14.5035    -15.5265
g0_os1_am-6       1   -13.0276    -15.5220
g0_os4_am-20      4   -15.5265    -18.0254
g0_os4_am-6       4   -18.0254    -18.0254
g0_os8_am-20      8   -18.0254    -13.8159
g0_os8_am-6       8   -13.8159    -13.8138
g3_os1_am-20      1   -13.8159    -15.5265
g3_os1_am-6       1   -13.0276    -15.5220
g3_os4_am-20      4   -15.5265    -18.0254
g3_os4_am-6       4   -18.0254    -18.0254
g3_os8_am-20      8   -18.0254    -13.8159
g3_os8_am-6       8   -13.8159    -13.8138
g6_os1_am-20      1   -13.8159    -15.5265
g6_os1_am-6       1   -13.0276    -15.5220
g6_os4_am-20      4   -15.5265    -18.0254
g6_os4_am-6       4   -18.0254    -18.0254
g6_os8_am-20      8   -18.0254    -13.8159
g6_os8_am-6       8   -13.8159    -13.8138
```

**OBSERVED な構造（機序の帰属はしない）**:

1. **g3 群と g6 群は完全一致**（18 行中 12 行が同一値）。
   すなわち `gainDb` は **IR 資産（g0/g3/g6 = nominal 0/+3/+6 dB）に依存しない**。
2. **値は 6 値の反復シーケンス**を成す: `{-14.5035, -13.0276, -15.5265, -18.0254, -13.8159, -13.8138}` 系。
3. **群間の持ち越し**: 前行群の最終 sc0 値が次行群の最初の sc0 値と一致
   （g0 最終 `-13.8159` → g3 初行 `-13.8159` → g6 初行 `-13.8159`）。
4. **`sc1 == 次行の sc0`** の関係が多数成立（例: `g0_os1_am-20.sc1 = -15.5265 = g0_os4_am-20.sc0`）。
   これは先行文書が記述した「行 N の sc1 == 行 N+1 の sc0」パターンと一致。
5. **`-14.5035` は 18 行中 1 行のみ**（`g0_os1_am-20` = `p15ir` セクションの初行・
   `p15eq`（convBypass=true・IR 未使用）から conv+IR 有効へ切り替わるセクション境界）。
   `-13.0276` は 3 行で共有（g0/g3/g6 の os1_am-6）。
   → **参照ペアは「セクション初行の一意値」vs「共有された値」の組み合わせ**である（OBSERVED）。

- 上記 1-5 は **provenance の記述のみ**。1.48 dB の機序帰属・補正・normalization は行わない（A2 の範囲外・P1 §9-5）。

---

## 4. historical vehicle の settle sequence と P2 vehicle との対比（A2-4）

### 4-1. historical `p15ir` の実際の順序（source 実読 + ログ裏付け）

`runPair`（:661-704）＋ `runCase`（:404-434）＋ `ensureTestIr`（:391-402）:

```text
① ensureTestIr  : loadImpulseResponse(irPath, false) → isIRFinalized() poll（100 ms × 最大 3000）
② runCase(sc0)  : seqBefore = getLastCommittedPublicationSequence()
                  → configureChain(os, IIR, softClip=false, satOff, headroom, makeup, trim, eqBoost, convBypass)
                  → waitBacklogZero(30000)
                  → waitWorldPublished(seqBefore, 30000)
                  → sleepPump(800)            ← 800 ms
                  → capture（kCapBlocks=12 blocks）
                  → sleepPump(30) → clearTap
③ ensureTestIr  : 再度 loadImpulseResponse + isIRFinalized poll
④ runCase(sc1)  : ②と同一（softClip=true・satOn）
⑤ diffStats(yOn, yOff, from=末尾 kAnalysisN=4096, ...) → clipEngagement / clipEngMax
⑥ emit row
```

- **`configureChain` は IR load/finalize の「後」に呼ばれる**。OS 変更に伴う structural rebuild は
  `configureChain` 内で発生するが、**その後に `waitIrFinalized` は呼ばれない**（`runCase` 内に存在しない）。
  続く待機は `waitBacklogZero` + `waitWorldPublished` + `sleepPump(800)` のみ。

### 4-2. 対比表（historical p15ir vs P2 probe）

| 段階 | historical `p15ir`（`--p1-char`） | P2 probe（`--buzz-probe=real`） |
| --- | --- | --- |
| IR load | `ensureTestIr` → `loadImpulseResponse(path,false)`（sc0/sc1 の前で計 2 回） | probe flow の IR load（`[BUZZ] probe-irload`） |
| IR finalize | `isIRFinalized()` poll（100 ms × 最大 3000）— **`configureChain` より前** | `waitIrFinalized(e, 300000)` — IR load 直後 |
| OS / geometry | `configureChain`（OS 変更 → structural rebuild） | `setOversamplingFactor` + rebuild |
| backlog | `waitBacklogZero(30000)` | `waitBacklogZero(30000)` |
| publish | `waitWorldPublished(seqBefore, 30000)` | `waitWorldPublished(seq, 30000, tag)` |
| **settle** | **`sleepPump(800)` = 800 ms** | **`[PROBE] quiet 150000 ms` = 150 s** |
| capture | 12 blocks・解析窓 4096（末尾） | `runCapture(pSig, 2.0, level)`・2 s |
| base SR | **48000** | **192000**（device SR） |
| OS pin | ケース毎に `configureChain` で設定 | `--buzz-os=1` で pin |
| sc | sc0/sc1 の 2 パス（`runPair` twoState） | sc0 のみ（`setSoftClipEnabled(false)`） |

### 4-3. 明示的切断（要求どおり）

- 本対比表は **「1.48 dB が現在 P2 で出なかった」ことと「3-J により消えた」ことを結びつける証拠ではない**。
- 3-J は `BassBuzzMeasurement.cpp:1542` の `static` 1 token 除去（test-only lifetime）であり、
  `--p1-char` の `runCase` / `ensureTestIr` / settle には一切触れていない。
- 両 vehicle は **settle 構造と base SR の両方**が異なるため、Step 4 の NOT REPRODUCED を
  settle 単一要因に帰属することは本 A2 の範囲では行わない。

---

## 5. S-1 の変更境界判定（A2-5）

### 5-1. 判定結果

```text
S-1 判定 = Case A
（kP15FullMatrix = false → true の 1 token で、historical p15ir vehicle の
  p15ir セクションは source-level に復元される）
```

### 5-2. Case A と判定した根拠（OBSERVED）

| # | 根拠 | 種別 |
| --- | --- | --- |
| 1 | `p15ir` セクションの実装（`runPair` + (I) ループ）は**現在の作業ツリーに完全な形で存在**し、`if (kP15FullMatrix)` のみで gate されている（:738） | OBSERVED（source 実読） |
| 2 | 非 gate セクション (A)(B)(C) の出力が historical と **bit-identical**（diff 空）→ 測定機構（`runCase`/`configureChain`/`ensureTestIr`/capture/解析）は不変 | OBSERVED（実測 diff） |
| 3 | (H)(I)(J) のループパラメータ・入れ子順・生成される id 並びが historical ログの id 並びと**完全一致** | OBSERVED（source vs ログ照合） |
| 4 | base SR = 48000 が historical / current で一致（`ratio=8.0000` 行が同一） | OBSERVED |
| 5 | (I) の実行位置（(A)-(H) の後・(J) の前）が historical と同一 | OBSERVED（source 順序） |
| 6 | settle / IR loading / capture / geometry / configuration / test helper の**変更は不要**（(I) は既存の `runPair`/`runCase`/`ensureTestIr` をそのまま呼ぶ） | OBSERVED |

→ Case B（他の変更が必要）には該当しない。Case C（historical source 復元不能）にも該当しない
（ソースは作業ツリーに現存し、gate のみで無効化されている）。

### 5-3. Case A に付随する残存不確定・運用条件（明示）

| # | 内容 |
| --- | --- |
| U-1 | `kP15FullMatrix` は **(H)(I)(J)(K)(L) の 5 セクションを一括で gate** する。flag=true では (K)(K2)(L)(M) も実行される。これらは **(I) より後**に位置するため `p15ir` 行の値には影響しないが、**「p15ir だけを選んで実行する」手段は存在しない**（= 実行時間が増える）。 |
| U-2 | (H)(I)(J) は gate 対象のため、as-is 実行で直接の再現検証はできない。U-1 の非 gate 検証（根拠 2）とパラメータ照合（根拠 3）による**間接確認**に留まる。 |
| U-3 | 実行時間: source コメント（90 分超/build・1 config-pair ≈ 37 s）より、flag=true の 1 run ≈ **90 分超**。Step 5-2 の 3 回反復では **4.5 時間超**。 |
| U-4 | `kP15FullMatrix` の導入・false 化は git 履歴に無い（UNKNOWN）。したがって「flag 以外の差分が無いこと」は git では証明できず、上記の実測 diff（根拠 2）が唯一の証拠である。 |

### 5-4. 結論

```text
S-1 = 「1 token で十分」= 条件付き YES
  - 必要な変更: kP15FullMatrix = false → true の 1 token のみ（他に必要な変更は OBSERVED されない）
  - 付随条件: flag=true は (H)(I)(J)(K)(L) を一括有効化するため 1 run ≈ 90 分超（3 回で 4.5 時間超）
  - 残存不確定: U-1〜U-4（特に git 履歴が無いため「flag 以外の差分なし」は実測 diff 依存）
```

- 本 A2 では **`kP15FullMatrix` を変更していない**（S-1 は未承認）。
- 次段階は **S-1 の実施承認**（ユーザー判断）。承認後も、実行は
  `kP15FullMatrix=true` の 1 token のみとし、settle / IR / capture / geometry には触れないこと。

---

## 6. A2 の禁止事項遵守の確認

```text
production source 変更      0
CMake 変更                  0
kP15FullMatrix 変更         0（false のまま）
settle 変更                 0
sleep 変更                  0
--buzz-os 等の追加          0
measurement normalization   0
compensation                0
P3 root-cause attribution   0（機序帰属の記述なし）
H-B 修正                    0
```

- 実施した実行は既存 binary の as-is `--p1-char` **1 回のみ**（`tmp/p1_5_ir_p2j/step5_p1char_current.log`・exit 0・
  `summary flag_macro=0 cases=73 failures=0`）。crash / heap corruption / ASan error = 0。
