# Phase 1 Pre-Implementation Audit（P1-0）— 2026-09-22・read-only

- **工程**: C2 `de95f335` 確定 → **P1-0（本 doc・read-only）** → P1-1 implementation + OFF/ON characterization（未着手）
- **前提 BUILD-ID**: HEAD `de95f335` / production `src/**`（`src/CustomInputOversampler.{cpp,h}`・`src/audioengine/**`）の HEAD 差分 **0** / `CMakeLists.txt` HEAD 同一 / flag **未定義** / `shadow_candidate` **0**（Phase 0 baseline）/ snapshot `ConvoPeq.md`（header 2026-09-22 06:20:11・`--check` FRESH・NEWER_SRC_COUNT=0）
- **本 doc は production 変更を一切含まない**（`centerValue *= 2.0` は未挿入・CMake flag 未定義）
- **表記規約**: 各項を **SOURCE**（読み取ったコード/保存された実測の位置）・**OBSERVATION**（観測された数値・構造）・**INFERENCE**（解釈・未検証）・**CONTRACT IMPACT**（判断への影響）に分離する。

---

## P1-0-A 案E の実装位置と変更 scope

### SOURCE

production `src/CustomInputOversampler.cpp` — `interpolateStage()`（:492 開始）の該当域を逐語引用:

```cpp
543:        double centerValue = 0.0;
544:        if (idx >= stage.centerDelayInput)
545:            centerValue = stage.centerCoeff * history[idx - stage.centerDelayInput];
546:        else
547:            bad = true;
548:
549:        if (bad || isBadSample(centerValue))
550:        {
551:            markCorruptionDetected();
552:            output[n * 2 + 0] = 0.0;
553:            output[n * 2 + 1] = 0.0;
554:            continue;
555:        }
556:
557:        convValue *= 2.0;
558:        if (fastAbs(convValue) < kDenormThreshold) convValue = 0.0;
559:        if (fastAbs(centerValue) < kDenormThreshold) centerValue = 0.0;
560:
561:        const int outBase = n << 1;
562:
563:        output[outBase + stage.convParity] = convValue;
564:        output[outBase + stage.centerParity] = centerValue;
```

Shadow 参照実装 `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h`（candidate 側の意味論の正）:

```cpp
449:        convValue *= 2.0;
450:        if (candidate)
451:            centerValue *= 2.0;   // ★ 案 E（candidate hypothesis・runtime 切替）
452:        if (fastAbsRef(convValue) < kDenormThresholdRef) convValue = 0.0;
453:        if (fastAbsRef(centerValue) < kDenormThresholdRef) centerValue = 0.0;
```

**候補位置が 1 箇所であることの機械的証明**（本監査で実測）:

| 検査 | 結果 |
|------|------|
| production ファイル内の `2.0` 出現 | **:557 の 1 箇所のみ**（`:313` は窓関数の `kPi * 0.5 * t`） |
| center 位相の出力書き込み | **:564 の 1 箇所のみ**（`:293-294` は prepare の parity 決定・`:321` は halfband 係数 zeroing・`:368` は centerDelayInput 算出） |
| up 経路の関数 | `processUp()` → 各 stage で `interpolateStage()`（:771）。`prepareSingleStage` も同一関数を使う 1-stage 構成（別実装なし） |
| down 経路 | `decimateStage()` に `*= 2.0` は存在しない（案E は up 側のみの 1 行） |

### OBSERVATION

- 挿入位置は **:557 の直後・:558（denorm clear）の前**に一意に決まる（Shadow :449-451 と同一順序）。
- 期待形（**未実装**）:

```cpp
        convValue *= 2.0;
#if CONVOPEQ_CORRECT_POLYPHASE_GAIN
        centerValue *= 2.0;
#endif
        if (fastAbs(convValue) < kDenormThreshold) convValue = 0.0;
        if (fastAbs(centerValue) < kDenormThreshold) centerValue = 0.0;
```

- production の意味変更は **1 文のみ**・追加行は 3 行（`#if`/`#endif` 含む）。
- P1-0-A の scope 固定: **production = `src/CustomInputOversampler.cpp` のみ**・**build = `CMakeLists.txt`（flag 定義と伝播）のみ**。`AudioEngine.*` / `RuntimeBuilder.*` / `AutoGainPlanner.*` / SoftClip / Limiter / Convolver / F-3 / F-4 / O-20 / calibration / default ON は scope 外。

### INFERENCE

- denorm clear の**前**に挿入する必然性: Shadow は scaling 後の値に対して denorm 判定する。順序を入れ替えると candidate 側の bitwise 等価（REF-FIDELITY の対称性）が崩れる。
- flag OFF では `#if` ブロックが除去されるため、**OFF build は Phase 0 baseline と同一のコード生成**になる（＝ OFF build で Phase 0 gate 54/54 の再現が sanity gate として使える）。

### CONTRACT IMPACT

- HOLD トリガ「案Eの適用位置に複数候補」→ **非成立**（候補は 1 箇所・機械的に証明済み）。
- 変更規模: production **+3 行** / build **flag 定義・伝播のみ**（P1-0-B）。

---

## P1-0-B flag contract

### SOURCE

- 既存規約（プロジェクト内の前例）: `option(NAME "desc" OFF/ON)`（`:40` `:71` `:128` `:129` `:1440` `:1441` `:1702` `:1739`）＋ 伝播は **generator expression** `$<$<BOOL:${VAR}>:MACRO=1>`（`ConvoPeq` :1303-1318 の :1315・`AudioEngineHarness` :1925-1938 の :1938）。
- `src/CustomInputOversampler.cpp` を compile する target は **3 つ**:

| target | 該当箇所 | 経路 |
|--------|----------|------|
| `ConvoPeq`（本体） | :1282 `target_sources(ConvoPeq PRIVATE ${CONVOPEQ_ALL_SOURCES})`（:1196 に oversampler） | `set(CONVOPEQ_ALL_SOURCES` :1133 |
| `AudioEngineHarness` | :1891-1898 の foreach で `CONVOPEQ_ALL_SOURCES` から `MainApplication.cpp`/`MainWindow.cpp` を除外して compile | :1899 `add_executable` |
| `PolyphaseGainFidelityTests` | :2006-2009（`src/tests/.../PolyphaseGainFidelityTests.cpp` + `src/CustomInputOversampler.cpp` を直接 compile） | C1 で新設（R18-1） |

- `build.bat` は `-DVAR[=VALUE]` を CMake に素通し（:39、例 :14-16）→ **build スクリプト変更は不要**。

### OBSERVATION

提案契約（**未実装**）:

```cmake
option(CONVOPEQ_CORRECT_POLYPHASE_GAIN "Correct polyphase gain convention (B-1)" OFF)   # default OFF

# 上記 3 target それぞれの target_compile_definitions に 1 行追加（PRIVATE）
$<$<BOOL:${CONVOPEQ_CORRECT_POLYPHASE_GAIN}>:CONVOPEQ_CORRECT_POLYPHASE_GAIN=1>
```

- ON → `CONVOPEQ_CORRECT_POLYPHASE_GAIN=1`、OFF → **macro 未定義** → `#if` が除去される（既存 `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` と同一の意味論）。
- ビルドコマンド（P1-0-D で使用）: `build.bat Release nopause "-DCONVOPEQ_CORRECT_POLYPHASE_GAIN=ON"` / `…=OFF`。

### INFERENCE

- 3 target すべてに伝播しないと、測定 host（`AudioEngineHarness`）と REF-FIDELITY 基盤（`PolyphaseGainFidelityTests`）が base のまま残り、OFF/ON 比較が成立しない → **3 target が最小完全集合**。
- OFF build が Phase 0 と同一コードになるため、「OFF build で Phase 0 54/54 を再現」が flag 実装そのものの無害性ゲートになる。

### CONTRACT IMPACT

- **default OFF は 2 重で保証可能**（`option()` 既定値 + Phase 1 では ON を明示指定したときのみ ON）→ HOLD トリガ「default ON の既存経路」非成立。
- **Phase 1 の境界**: 「flag ON を測定するための build」であって「default ON 化」ではない。`default ON` / `calibration` は HOLD 継続。
- **要 scope 明示（test-only・3 件）** — production は変更しないが、Phase 1 遂行に必要:
  1. **測定用 harness TU の追加**と `AudioEngineHarness` への test-only 登録（R12-8 の実測 host。C1 の R18-1 と同じ「test-only 登録」枠）。
  2. **`src/tools/build_identity_gate.py` の Phase 1 表示対応**: 現行は CMakeLists.txt に token があると BUILD ブロックに「DEFINED — Phase 0 契約違反 (R15-2): 測定無効」と表示する（:443-449）。Phase 1 では `CMakeCache.txt` の実値（ON/OFF）を表示する必要がある。なお `--check` の fail-closed 判定に token 検査は含まれない（:548-560 は identity/SDK/zero-deps のみ）ため、**ビルド自体は拒否されない**。
  3. **「CMakeLists.txt に token 0 件」不変条件の Phase 1 解除**: R15-2 の期限（「Phase 1 まで」）到来により、Phase 1 commit では token が CMakeLists.txt に**意図的に**出現する。C2 までの 0 件検査は Phase 1 以降は適用対象外であることを記録に残す。

---

## P1-0-C freeze invariant 再確認（C2-P attestation 基準）

| 項目 | Phase 1 開始前の要求 | 実測（本監査・2026-09-22） | 判定 |
|------|---------------------|---------------------------|------|
| Phase 0 evidence | immutable | `tmp/phase0_characterization_20260922.txt` sha256 `284394400bdf63cf…`（report §6.2 に固定・PASS=54 FAIL=0 exit 0） | **OK** |
| `CustomInputOversampler.cpp` | C2 baseline と一致 | worktree sha256 `abca338da828d605…`（report §6.2 の attestation と一致） | **OK** |
| CMake | C2 baseline と一致 | worktree sha256 `528a68d98a470640…`・`git diff HEAD -- CMakeLists.txt` 空 | **OK** |
| default | OFF | flag 自体が未定義（`option()` 不在・CMakeLists/build.bat に token 0 件） | **OK** |
| shadow candidate | Phase 0 baseline を維持 | `CONVOPEQ_POLYPHASE_REF_CANDIDATE` 既定 = 0（Phase 0 は runtime 切替で 0/1 を使用） | **OK** |
| Phase 0 54/54 | freeze 済み | report §6.3（exe 06:12:03 > source 06:11:40・log 06:13・`[FAIL]` 0 件） | **OK** |
| C2 evidence | `de95f335` | `git log -1` = `de95f335`・3 ファイル +975/−0・staged leftovers なし | **OK** |
| production flag | 未定義 | production src 0 件・CMakeLists 0 件・build.bat 0 件（test/tooling の 3 件は検出器/コメントのみ＝定義ではない） | **OK** |

- **Phase 0 の測定値は書き換えない**。**Phase 1 の測定値は Phase 0 report に追記せず、別 evidence ファイル**（例: `doc/work113/phase1_*`・`tmp/p1_*`）に記録する。

### 契約矛盾の明示的解決（freeze と Phase 1 の関係）

- Phase 0 report §6.4 は「production `src/**` 差分 0・`CMakeLists.txt` は HEAD と同一」を **freeze の不変条件**として明記している。Phase 1 はこれを意図的に変更するため、字面上は衝突する。
- 解決: freeze の不変条件は **「Phase 0 の測定基準（evidence）の同一性」を守る条件**であり、Phase 1 の production 変更は **新 BUILD-ID での再測定**（R17-2 の C2 evidence policy・R12-8）として freeze を**置換**する。Phase 0 evidence は不変のまま保持し、Phase 1 evidence は別 BUILD-ID・別ファイルで記録する。Phase 0 の数値の再解釈・上書きはしない。
- **CONTRACT IMPACT**: HOLD トリガ「Phase 0 freeze invariant の破壊」→ **非成立**（破壊ではなく手続き的置換であり、本節でその手続きを固定）。

---

## P1-0-D OFF / ON build matrix

| Build | Flag | 目的 | 起動コマンド |
|-------|-----:|------|-------------|
| OFF | 0 | Phase 0 baseline 再現（54/54 + bitwise 再現確認） | `build.bat Release nopause "-DCONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF"` |
| ON | 1 | 案E production characterization（R12-8） | `build.bat Release nopause "-DCONVOPEQ_CORRECT_POLYPHASE_GAIN=ON"` |

手順（両 build 共通・固定）:

1. build 後に `build/CMakeCache.txt` の `CONVOPEQ_CORRECT_POLYPHASE_GAIN:BOOL=` を記録（同一 build dir 再利用のため cache 実値を必ず添える）。
2. 両 build について **BUILD-ID 6 要素**を取得（`src/tools/build_identity_gate.py --emit-build-id --shadow-candidate 0`・P1-0-B の gate 表示修正後）:
   `snapshot_identity` / `git_head` / `working_tree` / `production_flag` / `shadow_candidate` / `build_config`。
3. **OFF build**: `PolyphaseGainFidelityTests.exe` を実行し **Phase 0 gate 54/54 PASS** を再確認（＝ flag 実装の無害性ゲート）。
4. **ON build**: P1-0-E の測定 matrix を、OFF と**同一入力・同一パラメータ**で実行。
5. **交互再現**: OFF → ON → OFF の順で 2 回目の OFF が同一 BUILD-ID・同一測定値になることを確認（cache 切替汚染の検出）。
6. 各測定ログの先頭に BUILD-ID 6 要素と cache 実値を付与（測定値の帰属保証）。

- **CONTRACT IMPACT**: build dir を共有するため、「cache 実値 + BUILD-ID 6 要素 + binary mtime」の 3 点を各測定に添付しない限り、測定の帰属は保証されない（本 matrix の必須条件として固定）。

---

## P1-0-E R12-8 production characterization 測定 matrix

### SOURCE — 観測可能性（production の既存 tap / API のみを使用）

| 観測点 | 位置 | 内容 |
|--------|------|------|
| chain 出力 tap（**pre-dither / pre-limiter**） | `AudioEngine.Processing.DSPCoreDouble.cpp:565-566`（`if (state.analyzerEnabled && analyzerSource == Output) pushToFifo(processBlock, analyzerFifo)`） | down 後・bypass blend 後の chain 出力。公開 API: `readFromFifo()` / `skipFifo()`（`AudioEngine.Fifo.cpp:8-19`）・`getFifoNumReady()`（`AudioEngine.h:1360`）・`setAnalyzerEnabled()`/`setAnalyzerSource()`（`:1396`/`:1394`） |
| レベル（peak 定義） | `measureLevel()` = **max abs**（`DSPCoreIO.cpp:177-190`）→ `outputLevelLinear`（`DSPCoreDouble.cpp:568-570`） | 公開 getter `getOutputLevel()`（`AudioEngine.h:1350`）・`getInputLevel()`（`:1343`）。**peak であり RMS ではない** |
| latency | `getTotalLatencySamples()`（`AudioEngine.h:1335`・`Latency.cpp:75`）・`estimateOversamplingLatencySamplesImpl`（`Latency.cpp:10-48`） | 報告 latency（PDC） |
| DC blocker（OS ドメイン） | `AudioEngine.h:643-651`（`UltraHighRateDCBlocker`・`init(processingRate, 1.0)`）・`UltraHighRateDCBlocker.h:33-95`（2 段カスケード ±10%） | **カットオフ 1.0 Hz** |
| limiter | `DSPCoreDouble.cpp:715-749` | θ=0.8413951287507587（kOutputHeadroom −0.5 dB）・knee=0.108748・hard clamp ±0.8912509381337456（−1.0 dBFS） |
| 実測 host | `AudioEngineHarness`（CMakeLists :1891-1898・:1899） | production 全ソース（MainApplication/MainWindow を除く）を compile する唯一の測定 host |

### OBSERVATION — 測定 matrix（observable 定義で固定）

| 項目 | 測定方法（production chain） | OFF 期待 | ON 期待 | 判定基準 |
|------|------------------------------|----------|---------|----------|
| **Level: DC** | **chain 出力では測定しない**（1.0 Hz DC blocker が DC を除去・OFF/ON 共通） | — | — | Phase 0 P0-A（oversampler 単体）の値を継承 |
| **Level: 50 Hz / 1 kHz / passband** | 正弦を chain に通し、chain 出力（`readFromFifo`・定常区間）の DFT 振幅比を測る | 基準 | OFF + **+2.4988·N dB** | OFF/ON の**差分**で確認（N = stage 数 = 1/2/3）。既存 staging（headroom/makeup/trim）は両者同一なので差分が flag 効果 |
| **Level 補助** | `getOutputLevel()`（peak dB）と `outputLevelLinear` の OFF/ON 差 | 基準 | 同上 | 参考記録（peak 定義） |
| **Latency** | (a) `getTotalLatencySamples()`、(b) impulse を chain に通し argmax を測る（P0-F と同一 4 経路: 31/90・511/140・IIR3・LP3） | 基準 | **OFF == ON** | (a) 一致・(b) argmax ∈ [floor(D), floor(D)+1] かつ OFF==ON |
| **SoftClip (i)** | OFF/ON 出力の sample 差イベント数（`\|y_ON − y_OFF\| > 0`）と `R_s = count_ON / count_OFF` | 基準 | 差は非ゼロ・`R_s` 記録 | Phase 0 record（119,768→121,267・Δ+1,499・R_s 1.013）との整合を確認（**Phase 0 値は書き換えない**） |
| **SoftClip (ii)** | `max\|y\|`（chain 出力 tap・最終出力の両方） | 基準 | 記録 | Phase 0 の max\|y\|=0.9 と同程度であることを確認 |
| **SoftClip (iii)** | hard-clamp count = `\|y\| ≥ 0.8912509381337456 − 1e−12` の件数（最終出力） | 基準 | 記録 | Phase 0 は 0 件（θ+κ=0.9<1.0） |
| **SoftClip (iv)** | **clip engagement count**: flag 固定で `softClipEnabled=true/false` の 2 run を比較し、差が出た sample 数（＝ clip が作用したサンプル数） | 基準 | 増加（駆動 +2.4988N dB） | 代替指標（下記 INFERENCE 参照） |
| **Limiter (a)** | limiter 入力 peak = chain 出力 tap の max\|·\|（`measureLevel` と同定義）＋ `getOutputLevel()` | 基準 | 記録 | OFF/ON 差を dB で記録 |
| **Limiter (b)** | threshold crossing count / rate = chain 出力 tap の `\|x\| > 0.8413951287507587` の件数と全サンプル比 | 基準 | 増加 | **pre-dither 近似**（limiter 入力は post-dither） |
| **Limiter (c)** | limiter 出力 peak = 最終出力の max\|·\| | 基準 | 記録 | θ 近傍で飽和することを確認 |
| **Limiter (d)** | hard-clamp count = (iii) と同一定義 | 基準 | 記録 | D-5 の「limiter 到達量増加」を chain で閉じる |

- 測定入力は **複数レベル**（例: −20 / −6 / 0 dBFS 相当）・複数周波数で実施する（非線形段があるため単一レベルで結論しない）。
- 全測定で OFF/ON の入力・パラメータ・block size・sample rate を完全一致させる。

### INFERENCE

- (i)(ii)(iii)(iv) と (a)〜(d) はすべて**既存の公開 API / tap** で取得可能 → **追加の production 計装は不要**（Phase 1 の production 変更を 1 行に保てる）。
- ON の level 差分は staging 非依存（D-2/D-4 で確認済み）なので、staging を固定すれば差分測定がそのまま flag 効果になる。ただし最終段に limiter / hard clamp があるため、**「最終出力が必ず +2.4988N dB」とは単純化しない**（θ 超の material では差が圧縮される）。
- **up ドメイン pre-clip 内部量（clip 入力 drive・Padé arg clamp 数 = Phase 0 の tanhClamp 18,200 件）は chain から観測不能** → chain では (i) と (iv) を代替指標とし、Phase 0 record を継承する（**要承認の設計選択 1 件**）。

### CONTRACT IMPACT

- HOLD トリガ「AudioEngine signal chain と測定設計の不一致」→ **非成立**（level / latency / SoftClip / limiter の全項目を observable 定義で固定）。
- 測定用の追加コードは **test-only**（P1-0-B の scope 明示 1 に含める）。production 側の計装追加は不要。

---

## P1-0-F 禁止事項（Phase 1 audit 中に変更しない）

**production / 設定（Phase 1 scope 外）**:

```text
default ON / calibration / release-note の実装 / F-3 / F-4 / O-20
AutoGainPlanner / SoftClip threshold / Limiter threshold / headroom / makeup / Convolver trim
```

**証跡（freeze 契約）**:

- Phase 0 evidence（`tmp/phase0_characterization_20260922.txt`・report §6 の数値/hash）を**書き換えない**。
- C2 commit（`de95f335`）を amend しない。
- `ConvoPeq.md` の再生成は **production source 変更後の build identity 更新フロー**として扱い、Phase 0 freeze の証跡（report §6）を書き換えない。再生成後は header の `Generated` 値と `--check` 結果を **Phase 1 evidence 側**に記録する。
- Phase 1 の測定値・ログは新規ファイル（`tmp/p1_*`・`doc/work113/phase1_*`）に分離する。

**Phase 1 で変更してよいもの（本監査で明示）**:

| 種別 | 対象 | 根拠 |
|------|------|------|
| production | `src/CustomInputOversampler.cpp`（+3 行・案E 1 文） | P1-0-A |
| build | `CMakeLists.txt`（`option()` + 3 target への genex 伝播） | P1-0-B |
| test-only | 測定用 harness TU の追加と `AudioEngineHarness` への登録 | P1-0-B scope 明示 1 |
| test-only tooling | `src/tools/build_identity_gate.py` の Phase 1 表示対応 | P1-0-B scope 明示 2（R18-1(a) の前例） |

---

## 判定

```text
P1-0 = PASS
```

**PASS 条件の充足（実測に基づく）**:

| PASS 条件 | 実測 |
|-----------|------|
| 案Eの実装位置が source と一致 | `interpolateStage()` :557 直後・候補 1 箇所（`2.0` 出現 1・center 書込 1）・Shadow :449-451 と同一順序 |
| flag の定義位置・伝播方法が確定 | `option()`（既定 OFF）+ genex を oversampler を compile する 3 target へ（既存 `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` と同一規約・build.bat 変更不要） |
| default OFF が保証可能 | option 既定 OFF + ON は明示指定時のみ（2 重保証） |
| Phase 0 freeze invariant が維持可能 | P1-0-C の 8 項目すべて OK・置換手続きを明文化 |
| OFF/ON build matrix が固定 | P1-0-D（6 手順・BUILD-ID 6 要素・cache 実値・交互再現） |
| R12-8 の level / SoftClip / limiter / latency 測定が固定 | P1-0-E（全項目 observable 定義・既存 tap のみ） |
| scope 外変更が明確 | P1-0-F（production 1 ファイル・build 1 ファイル・test-only 2 件） |
| 未解決の契約矛盾なし | freeze/Phase 1 の字面衝突を P1-0-C で解決（置換として固定） |

**HOLD トリガ 6 項目 → すべて非成立**:

| トリガ | 実測 |
|--------|------|
| production flag の既存定義を発見 | なし（production 0 件・CMakeLists 0 件・build.bat 0 件。test/tooling の 3 件は検出器/コメント） |
| default ON の既存経路を発見 | なし（`option()` 不在・`#define` 不在） |
| 案Eの適用位置に複数候補 | なし（1 箇所・機械的証明） |
| AudioEngine signal chain と測定設計の不一致 | なし（observable 定義で固定・production 計装不要） |
| Phase 0 freeze invariant の破壊 | なし（8 項目 OK・置換手続きを固定） |
| 既存補償の新たな発見 | なし（D-2/D-4 の census を再確認・factor 依存補償 0 件） |

**P1-1 着手前にユーザー確認が必要な項目（4 件）**:

1. **test-only scope 明示（3 件）**: 測定用 harness TU の追加 / `build_identity_gate.py` の Phase 1 表示対応 / 「CMakeLists token 0 件」不変条件の Phase 1 解除。
2. **測定代替の承認（1 件）**: up ドメイン pre-clip 内部量（clip 入力 drive・Padé arg clamp 数）は chain から観測不能 → 代替指標（SoftClip (i)(iv)）+ Phase 0 record 継承とする設計選択。

**本 doc は read-only 監査の結果であり、案E の実装承認ではない。** `centerValue *= 2.0` は未挿入・CMake flag は未定義・default ON / calibration は HOLD 継続。

---

## 付録: 本監査の限界（未検証項目）

- 実測は行っていない（コード読解と既存証跡の照合のみ）。ON build の測定値は P1-1 で取得する。
- `getOutputLevel()` / `outputLevelLinear` は **peak 定義**（RMS ではない）。
- chain 出力 tap は **pre-dither**（limiter 入力は post-dither）→ threshold crossing count は近似値。
- 1.0 Hz DC blocker の実測周波数応答は未取得（カットオフは `init(processingRate, 1.0)` のコード読解による）。
- SoftClip の clip engagement count（(iv)）は「clip が作用したサンプル数」の代理指標であり、Phase 0 の per-phase attribution（c1〜c4 契約）そのものではない。
- `AudioEngineHarness` の実走（ビルド・起動）は本監査では行っていない（既存 exe `build/Release/AudioEngineHarness.exe` の存在のみ確認）。
