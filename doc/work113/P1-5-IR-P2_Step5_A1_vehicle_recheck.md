# P1-5-IR-P2 — Step 5（A-1）`--p1-char` vehicle 再現確認 — STOP（実行不能）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（A-1）
- **判定**: **STOP — A-1 は実行不能**。Step 5-0 の STOP 条件
  「`--p1-char` vehicle 自体の source modification が必要」に該当。
  A-1 の目的（同一 vehicle の再現性確認）を成立させるには test source 変更が必要であり、
  Step 5-0 の禁止に触れるため**実行せず停止**した。
- **変更**: production / CMake / JUCE / default / calibration / settle / sleep / source = **すべて 0**
  （本 Step は read-only と既存 binary の as-is 実行のみ）。

---

## Step 5-0 — State Freeze（PASS）

| 項目 | 値 | 判定 |
| --- | --- | --- |
| HEAD | `1e9e63e3` | ✅ |
| OFF binary SHA-256[:16] | `2054d970da4ec91c` | ✅ |
| `CONVOPEQ_CORRECT_POLYPHASE_GAIN` | `BOOL=OFF` | ✅ |
| production source diff | 0 | ✅ |
| CMake diff | 0 | ✅ |
| JUCE source diff | 0 | ✅ |
| default / calibration diff | 0 | ✅ |
| settle / sleep diff | 0 | ✅ |
| measurement semantics diff | 0 | ✅ |
| 新規 test-source modification | 0（既存 M 3 ファイルは持ち越しのみ） | ✅ |

```text
H-A = CLOSED / H-B = OPEN / P2 matrix = COMPLETE / P3 = NOT STARTED
```

---

## Step 5-1 — 元 invocation の確定（ログ上で確定済み）

**元 invocation は推測ではなく一次資料から確定できた**:

`tmp/p15_sequence.bat`（参照値生成時の実行スクリプト）:

```bat
copy /Y build\Release\AudioEngineHarness.exe build\Release\AudioEngineHarness_off.exe >nul
build\Release\AudioEngineHarness_off.exe --p1-char > C:\VSC_Project\ConvoPeq\tmp\p15char_off.log 2>&1
```

- 元 invocation = **`AudioEngineHarness.exe --p1-char`（他旗なし）**。
- ソース側でも裏付け: `P1PolyphaseGainCharacterization.cpp:468-470`

  ```cpp
  int runP1PolyphaseGainCharacterization(int argc, char* argv[])
  {
      (void)argc; (void)argv;   // 旗解析なし
  ```

  → `--p1-char` に追加引数を与えても挙動は同一。invocation は一意に確定。

### 参照値の同定（一次ログで確定）

`tmp/p15char_off_partial.log`:

```text
:235  [P1CHAR] p15ir id=g0_os1_am-20 os=1 n=0 ampDb=-20.0 gainDb_sc0=-14.5035 gainDb_sc1=-15.5265 limitingEngaged_sc0=0 limitingEngaged_sc1=0 hardClamp_sc0=0 hardClamp_sc1=0 clipEngagement=4096 clipEngMax=0.024784
:356  [P1CHAR] p15ir id=g0_os1_am-6  os=1 n=0 ampDb=-6.0  gainDb_sc0=-13.0276 gainDb_sc1=-15.5220 limitingEngaged_sc0=0 limitingEngaged_sc1=0 hardClamp_sc0=0 hardClamp_sc1=0 clipEngagement=4096 clipEngMax=0.163382
```

- **Δ(gainDb_sc0) = −13.0276 − (−14.5035) = +1.4759 dB ≈ 1.48 dB**（P0/P1 の記録値と一致）。
- P0/P1 文書の記述「同一 IR g0・os=1・sc0」は**正しい**（本行が `id=g0_os1_am-*`・`gainDb_sc0`）。
  ※ 注: `−14.5035` は `p15eq` セクションの `os=8/1kHz/amp=-6` 行にも `dftDb` として現れる（偶発的一致）。
  混同を避けるため、参照は必ず `p15ir id=g0_os1_*` 行を指すこと。

### 参照条件の定義（ソースで確定）

`P1PolyphaseGainCharacterization.cpp:738-753`:

```cpp
if (kP15FullMatrix)
{
    const IrCase irs[] = { {"tmp/p15_ir_g0.wav","g0"}, {"tmp/p15_ir_g3.wav","g3"}, {"tmp/p15_ir_g6.wav","g6"} };
    for (const IrCase& ic : irs)
        for (int osF : { 1, 4, 8 })
            for (double adb : { -20.0, -6.0 })
            {
                PairCfg c;
                c.ampDb = adb; c.os = osF; c.convBypass = false; c.irPath = ic.path;
                c.satOff = 1.0f; c.satOn = 1.0f;
                runPair("p15ir", ...);
            }
}
```

- 参照条件 = `p15ir` セクションの **g0 / os=1 / amp=−20 → −6**（conv ON・sat 1.0）。
- native 順序は `g0 → os{1,4,8} → amp{−20,−6}`。1 プロセス = 1 回の全 sweep であり、
  **1 プロセス内で am−20 → am−6 の順に 1 回ずつ**得られる（組み込みの反復方式はなし）。

---

## Step 5-1 実行不能の根拠（STOP）

### 決定的事実: `p15ir` セクションは現在の source で**無効化**されている

| 事実 | 値 | 出所 |
| --- | --- | --- |
| `kP15FullMatrix` は **HEAD に存在しない** | `git show HEAD:…/P1PolyphaseGainCharacterization.cpp \| grep -c 'p15ir'` = **0** | git |
| `kP15FullMatrix` は**作業ツリーの未コミット変更で追加**された | diff に `+constexpr bool kP15FullMatrix = false;` | git diff |
| 現在値 | **`false`**（`P1PolyphaseGainCharacterization.cpp:43`） | source |
| CMake による上書き | **なし**（純粋な `constexpr`・`grep P15FullMatrix CMakeLists.txt` = 0） | source |

→ `p15ir` セクションは `if (kP15FullMatrix)` で gate されており、`false` のため**実行されない**。

### 実測による確認（既存 binary を as-is 実行・source 変更なし）

```text
./build/Release/AudioEngineHarness.exe --p1-char
  → exit 0 / [P1CHAR] summary flag_macro=0 cases=73 failures=0
  → p15ir 行 = 0
  → g0_os1_am-20 / g0_os1_am-6 = ABSENT
```

実行されたセクション（現 binary）: `p15preset` / `p15hrnl` / `p15listen`
参照ログのセクション: `p15preset` / `p15eq` / `p15ir` / `p15staging`
→ **セクション集合が異なる**（= 現 binary の `--p1-char` は参照時とは別構成の vehicle）。

### 保存 binary にも参照条件は存在しない

文字列検索（control = `p15preset`）:

| binary | p15preset | p15ir | p15staging | p15eq |
| --- | --- | --- | --- | --- |
| `AudioEngineHarness.exe`（現行） | 1 | **0** | 0 | 0 |
| `AudioEngineHarness_off.exe`（2026-09-22 11:29） | 1 | **0** | 0 | 0 |
| `AudioEngineHarness_on.exe`（2026-09-22 14:42） | 1 | **0** | 0 | 0 |

- control が 1 であるため検索は有効。**現存する全 binary が `p15ir` を持たない**。
- `p15char_off_partial.log`（11:28）を生成した binary は**既に上書きされて現存しない**
  （11:29 のスクリプト再ビルドで置換された）。

### 結論

参照条件 `g0_os1_am-20` / `g0_os1_am-6` を実行するには
**`kP15FullMatrix` を `false` → `true` に変更する必要がある**。これは
Step 5-0 の STOP 条件「`--p1-char` vehicle 自体の source modification が必要」に該当する。

また Step 5-1 の指示「以前の invocation / 条件をそのまま再実行」は、
**条件（`p15ir` セクション）自体が現行 source に存在しない**ため成立しない。
したがって:

```text
A-1 = STOP（実行不能）— 判定 REPRODUCED / NOT REPRODUCED のいずれにも到達しない
```

- Step 5-2（3 回反復）・Step 5-3（値取得）・Step 5-4（分類）は**未実施**。
- `--p1-char` の as-is 実行（1 回）のみ実施し、上記の不在を実測で確認した（raw: `tmp/p1_5_ir_p2j/step5_p1char_current.log`）。
- crash / heap corruption / ASan error は**なし**（exit 0・cases=73 failures=0）。
  `[FAULT] coordinator Faulted` 系は本 Step でも記録のみ（H-B = OPEN のまま）。

---

## 副次的に得られた OBSERVED（参照行から・P3 の帰属はしない）

参照 2 行（同一ログ・同一セクション・隣接条件）の比較:

| 量 | am=−20 | am=−6 | Δ |
| --- | --- | --- | --- |
| `gainDb_sc0`（SoftClip off） | −14.5035 | −13.0276 | **+1.4759 dB** |
| `gainDb_sc1`（SoftClip on） | −15.5265 | −15.5220 | **+0.0045 dB** |
| `clipEngMax` | 0.024784 | 0.163382 | （level 依存） |
| `clipEngagement` | 4096 | 4096 | 0 |
| `limitingEngaged` | 0 | 0 | 0 |
| `hardClamp` | 0 | 0 | 0 |

- **OBSERVED**: 参照された level 依存差は **sc0 側にのみ現れ、sc1 側ではほぼ消える**（0.0045 dB）。
- 本項は**記録のみ**であり、機序の帰属（P3）・補正・normalization は行わない（P1 §9-5 遵守）。
- limiter / hardClamp は両条件で非接触のため、安全鎖は本差の説明変数ではない（OBSERVED）。

---

## 次判断のための材料（ユーザー判断待ち）

A-1 を実行するには、以下のいずれかの明示的承認が必要:

| 選択肢 | 内容 | 影響 |
| --- | --- | --- |
| **S-1** | `kP15FullMatrix` を `false` → `true` に変更し、`--p1-char` を 3 回実行 | **test source 変更**（1 token）。現行 binary で参照条件を再実行できる。3-J 修正版 binary 上での再現性確認という A-1 の目的に合致 |
| **S-2** | 変更せず、参照条件を「現行 source で実行不能」として **historical OBSERVED のまま凍結**し、A-1 を中止 | source 変更 0。ただし A-1 の問い（現行 binary での再現性）は未解決のまま |
| **S-3** | 別 vehicle（probe flow・Step 4 実測）の結果を参照値の代替比較対象として採用 | source 変更 0。ただし vehicle が異なるため「同一 vehicle 再現性」は依然不明 |

- 推奨は **S-1**（A-1 の目的を満たす唯一の選択肢）。ただし Step 5-0 の STOP 条件に該当するため、
  本 Step では**実施していない**。
- 併せて A-2（source read-only audit）で、`kP15FullMatrix` がいつ・なぜ `false` に設定されたか、
  および参照測定の実行順序（IR load → finalize → OS geometry → waitBacklogZero →
  waitWorldPublished → sleepPump(800) → capture）を確定する作業が必要。

## 禁止の継続

```text
P3 root-cause attribution / settle A/B 比較 / production fix / harness 修正 / H-B 修正 = 未着手・禁止
```
