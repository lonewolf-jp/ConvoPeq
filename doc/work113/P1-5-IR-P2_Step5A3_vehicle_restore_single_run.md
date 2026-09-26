# P1-5-IR-P2 — Step 5-A3: S-1 最小復元・単回実行

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（A-3）
- **判定**: **A3 = REPRODUCED**（historical reference が bit-exact に再現）
- **変更**: `P1PolyphaseGainCharacterization.cpp:43` の **1 token のみ**
  （`kP15FullMatrix = false` → `true`）。他は一切変更なし。
- **raw evidence**: `tmp/p1_5_ir_p2j/step5a3_p1char_flagON.log`（6799+ 行・220 cases）

---

## A3-0 State Freeze（変更前・PASS）

| 項目 | 値 | 判定 |
| --- | --- | --- |
| HEAD | `1e9e63e3` | ✅ |
| `CONVOPEQ_CORRECT_POLYPHASE_GAIN` | `BOOL=OFF` | ✅ |
| 変更前 binary SHA-256[:16] | `2054d970da4ec91c`（3-J 版） | ✅ |
| production source diff | 0 | ✅ |
| CMake diff | 0 | ✅ |
| JUCE diff | 0 | ✅ |
| default / calibration diff | 0 | ✅ |
| settle / sleep diff | 0 | ✅ |
| measurement semantics diff | 0 | ✅ |
| `kP15FullMatrix`（変更前） | `false` | ✅ |

```text
H-A = CLOSED / H-B = OPEN / P2 matrix = COMPLETE / P3 = NOT STARTED
```

## A3-1 変更（1 token のみ）

```diff
-constexpr bool   kP15FullMatrix = false;
+constexpr bool   kP15FullMatrix = true;
```

自己検査:

| 検査 | 結果 |
| --- | --- |
| `kP15FullMatrix = true` の出現数 | **1**（一意） |
| `kP15FullMatrix = false` の出現数 | 0 |
| ファイル行数 | **908**（変更前と同一・行の増減なし） |
| `runPair` / `runCase` / `ensureTestIr` / `sleepPump` / SR / OS / timeout / capture window / 新 CLI flag | **変更なし** |
| production source / CMake / JUCE | **差分 0** |

## A3-2 ビルド（OFF 構成・harness のみ）

| 項目 | 値 |
| --- | --- |
| build exit code | **0** |
| 再コンパイルされた TU | `P1PolyphaseGainCharacterization.cpp` の **1 件のみ**（他は未再ビルド） |
| binary | `build/Release/AudioEngineHarness.exe` |
| **新 SHA-256[:16]** | **`b2c39a9a0e3b1fe3`** |
| `CONVOPEQ_CORRECT_POLYPHASE_GAIN` | `BOOL=OFF` ✅ |
| `p15ir` 文字列（binary 内） | **1**（変更前は 0 → セクションが実際にリンクされた） |
| production diff 再確認 | 0 |

## A3-3 実行（1 回のみ）

```text
build\Release\AudioEngineHarness.exe --p1-char
```

| 項目 | 値 |
| --- | --- |
| 開始 / 終了 | 2026-09-23 08:00:00 → 09:10:16 |
| 実行時間 | **70 分 16 秒**（A2 見積り「90 分超」と整合） |
| **exit code** | **0** |
| summary | `[P1CHAR] summary flag_macro=0 cases=220 failures=0` |
| crash / heap corruption / ASan / double-free / UAF | **0 件** |
| `[FAULT]` / `routerPendingRetire` | **0 件**（本 vehicle では H-B マーカーは出現せず） |

---

## A3-4 最優先確認項目（参照 2 行）

`[P1CHAR] p15ir id=g0_os1_am-20` = **L235** / `[P1CHAR] p15ir id=g0_os1_am-6` = **L356**
（historical ログと**同一行番号**）

### 全フィールド照合（11 項目 × 2 行 = 22/22 OK）

| field | g0_os1_am-20 hist | A3 | g0_os1_am-6 hist | A3 |
| --- | --- | --- | --- | --- |
| `os` | 1 | **1** | 1 | **1** |
| `n` | 0 | **0** | 0 | **0** |
| `ampDb` | −20.0 | **−20.0** | −6.0 | **−6.0** |
| **`gainDb_sc0`** | **−14.5035** | **−14.5035** | **−13.0276** | **−13.0276** |
| **`gainDb_sc1`** | **−15.5265** | **−15.5265** | **−15.5220** | **−15.5220** |
| `limitingEngaged_sc0` | 0 | **0** | 0 | **0** |
| `limitingEngaged_sc1` | 0 | **0** | 0 | **0** |
| `hardClamp_sc0` | 0 | **0** | 0 | **0** |
| `hardClamp_sc1` | 0 | **0** | 0 | **0** |
| `clipEngagement` | 4096 | **4096** | 4096 | **4096** |
| `clipEngMax` | 0.024784 | **0.024784** | 0.163382 | **0.163382** |

### 1.48 dB の再現

```text
Δ(gainDb_sc0)  historical = +1.4759 dB
Δ(gainDb_sc0)  A3         = +1.4759 dB
delta of deltas           = +0.000000 dB
```

### IR / geometry 照合（要求 9 項目すべて一致）

| 項目 | historical | A3 |
| --- | --- | --- |
| base SR | 48000 | **48000** |
| os | 1 | **1** |
| target SR | 48000 | **48000** |
| IR length | 48000 | **48000** |
| L0 part | 512 | **512** |
| L0 fft | 1024 | **1024** |
| slot peak | 0.5 | **0.500000** |
| F_scale | 0.50013092 / 0.50026187 | **0.50013092 / 0.50026187** |
| `[IR_RATE_GEN]` | gen=0 sourceSr=48000 targetSr=48000 actualSr=48000 sourceLen=1 convertedLen=1 ratio=1.0000 resampled=no | **同一** |
| `[IR_TAIL_GEOM]` | gen=0 loadedSr=48000 loadedLen=1 targetLength=48000 copySamples=1 fadeSamples=0 | **同一** |
| `[L0_WRITE]` | part=512 numIR=12 numParts=16 fft=1024 imm=1 irLen=48000 / slot=11 peak=0.500000 bin=0 | **同一** |

---

## A3-5 1-run-lag（別項目・記録のみ）

実際のログ順（source 実読と一致することを確認）:

```text
[IR_RATE_GEN/IR_TAIL_GEOM/F_scale]  ← ensureTestIr（IR load + finalize poll）  sc0 パス
[IR_RATE_GEN/IR_TAIL_GEOM/F_scale]  ← ensureTestIr（IR load + finalize poll）  sc1 パス
[IR_RATE_GEN/IR_TAIL_GEOM/F_scale]  ← 次ケースの ensureTestIr
[P1CHAR] p15ir id=g0_os1_am-20 ...
```

- 直前のケースは `p15eq` の **os=8**（`irLen=384000`・`part=4096`・`fft=8192`・`F_scale=0.27427088`）。
  参照 2 行は **os 8→1 遷移の直後**（historical と同一の遷移）。
- `configureChain` → `waitBacklogZero` → `waitWorldPublished` → `sleepPump(800)` → capture の順は
  source（`runCase` :404-434）どおりで、historical と同一。
- **本 Step の目的は lag の機序説明ではない**ため、これ以上の解釈は行わない。

---

## A3-6 判定

```text
A3 = REPRODUCED（Case A）
```

- historical reference 2 行が、**現在の 3-J 修正版 binary（OFF・`b2c39a9a0e3b1fe3`）上で
  bit-exact に再現**した（22/22 フィールド一致・Δ の差 0.000000 dB・同一行番号）。
- したがって A2 で立てた **S-1 = Case A（`kP15FullMatrix` の 1 token で p15ir vehicle が復元される）**
  が実測で確認された。
- **帰属は行わない**（指示どおり）: polyphase gain / IR scale / 1-run-lag / 3-J のいずれが
  原因であるかの断定はしない。3-J は test-only lifetime 1 token であり、本再現はそれを示すものではない。

### 参照行以外の付随観測（記録のみ・解釈しない）

全 6 セクションの**位置対応比較**（id 順序は全セクションで完全一致・id 不一致 0）:

| section | historical 行数 | A3 行数 | 比較可能 | `gainDb` 不一致 |
| --- | --- | --- | --- | --- |
| `p15preset` (G) | 7 | 7 | 7 | **0** |
| `p15eq` (H) | 18 | 18 | 18 | **0** |
| **`p15ir` (I)** | **18** | **18** | **18** | **0** |
| `p15staging` (J) | 30（kill により未完） | 36 | 30 | **18** |
| `p15sat` (K) | 0（未実装） | 42 | — | — |
| `p15lim` (L) | 0（未実装） | 33 | — | — |

- **`p15ir` は 18 行すべてが `gainDb_sc0/sc1` 完全一致**（A3 の対象セクション）。
- `p15staging` のみ 18 件の不一致（例: `st-6_os2_am-20` hist −7.0070 → A3 −9.5063）。
  historical 側は stage ごとに os 非依存の「平坦な」値（st-6 の全行 −7.0070）を示す。
- **本項は記録のみ**であり、機序帰属・補正・normalization は行わない（A4 / P3 の範囲）。
- なお historical ログは `p15staging` の途中で kill されており、`summary` 行を持たない（未完了 run）。

---

## 現時点の状態（次判断用）

```text
HEAD                 = 1e9e63e3
kP15FullMatrix       = true（A3 の 1 token・未 revert）
OFF binary           = b2c39a9a0e3b1fe3（flag=true 版）
production/CMake/JUCE diff = 0
A3                   = REPRODUCED
```

- 追加 run（2 回目・3 回目）を行う場合、本 flag=true の binary をそのまま使用できる
  （1 run ≈ 70 分）。
- revert する場合は 1 token を戻すのみ（他に影響なし）。

## 禁止事項の遵守

```text
settle 延長/短縮 0 / waitIrFinalized 追加 0 / --buzz-os 追加 0 / sleepPump 変更 0
IR geometry 変更 0 / gain compensation 0 / normalization 0 / P2 harness 変更 0
production fix 0 / H-B 修正 0 / P3 attribution 0
```

- 変更は `kP15FullMatrix` の 1 token のみ。実行は `--p1-char` 1 回のみ。
