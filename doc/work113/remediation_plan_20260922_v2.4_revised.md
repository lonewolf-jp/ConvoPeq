# ConvoPeq 残件 改修計画書（2026-09-22 時点・再改訂 v2.4）

- **版**: v2.4（v2.3 の全主張をソース実測で再検証し、**R10-1〜R10-13** を追加・**E-1〜E-9** で v2.3 を是正）
- **前版**: `doc/work113/remediation_plan_20260922_v2.3_revised.md`（R9-1〜R9-6）
- **監査実施日**: 2026-09-21（Qoder デスクトップ環境 / OS クラッシュ再起動後の継続セッションを含む）
- **基準ソース**: HEAD = `8f127bfe`（+ 未 push `c4a08171`）。production `src/*.cpp/h` の未 commit 差分 **0 件を再確認**（`git diff --numstat HEAD -- src/CustomInputOversampler.cpp` = 空）
- **authoritative 計測定義**: `doc/work113/model_polyphase_20260920.py` + `_results.txt`
  - v2.4 で **§2.7.1 の全 dev 値がモデルから完全再現できることを確認**（R10-7）
  - 再実行環境: WSL python3 **3.14.4 / numpy 2.5.3 / scipy 1.18.1**。Windows 側は python 3.14 / numpy 2.5.2 / scipy 1.18.0 / pandas 3.0.5
- **本書は read-only 監査と方針案の提示**。production / CMake への適用は本書承認後
- **案 E は有力仮説（candidate hypothesis）**。defect（DC = 0.75^N）は **5 系統独立で再現済み**。ただし是の設計判断は G-0 承認まで「確定」と表記しない（R5-4 継承）
- **Phase 0 着手 GO・Phase 1 以降 HOLD**（v2.2 から変更なし）
- **compile-time flag**: CMake option **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`** / C++ は **`#if`（`#ifdef` 禁止）**
- **パッチ案は文書化済み・未適用**（`remediation_v22_prep_patches_20260922.md`）

---

## v2.3 → v2.4 改訂サマリ

v2.3 の GO/HOLD・案 E の位置付け・Phase 0 の 5 構造・O-* 推奨は**原則として維持**する。本改訂は (a) v2.3 が未棚卸しだった資産の確定、(b) 測定の「定義依存性」の分離、(c) v2.3 本文の数値・ラベル・行番号の是正、(d) 残存する技術判断の明示、を扱う。

### 新規確定（R10-1 〜 R10-13）

| ID | 区分 | 内容 | v2.4 での確定 |
|----|------|------|----------------|
| **R10-1** | defect 再現性 | DC 0.75^N / cand 1.0 を **5 系統独立**で再現（production バイナリ / authoritative モデル / NumPy 転写 / GNU Octave / Maxima）。image/tone 構造定数 −9.542425 dB も 5 系統一致 | defect の**存在・局在・数値**は確定扱い可。「案 E が是」は依然 G-0 待ち（表記禁止は維持） |
| **R10-2** | ライブ帰因 | `AudioEngineHarness.exe` 無引数実行で `[OS_DIRECT]` / `[EQ_DIRECT]` / `[OF_DIRECT]` を実測取得。**round-trip 損失の唯一の支配的要因は Oversampler**（EQ ratio=1.000000・OF ratio≈0.9874=−0.110 dB） | §2.4 の表が authoritative。v2.3 はこの実測表を本文に持たなかった |
| **R10-3** | **measure 定義の分離** | harness の up/down 分割は **peak ベース**で、**段数に依存して値が変わる**（r=2: up 1.000000/down 0.750000 → r=4: 0.750001/0.749998 → r=8: 0.650885/0.648156）。 invariant は round-trip 積のみ | 「up で失う/down で失う」の記述は**因果の根拠にできない**。因果は係数レベル DC 和（conv 1.0 / center 0.5）で確定。gate は round-trip に一本化（§2.7） |
| **R10-4** | **D1 gate 再定義** | base D1 を **81 測定**（3 design × 3 f̂ × 3 N × 3 窓/trim）で **−9.542〜−9.546 dB（構造定数との差 ≤0.0036 dB）** と確定＝**手法不変**。cand D1 は同一条件で **−72.11〜−164.50 dB（最大 92 dB 振幅）**＝**手法依存** | **gate してよいのは base 側のみ**。cand 列は spec/閾値として引用禁止。**R9-5（63/120@0.2 の 0.94 dB 差）は数値的に無意味と確定**（解消済み） |
| **R10-5** | CMake 適用範囲 | `compile_commands.json` 上 `src/CustomInputOversampler.cpp` は **6 エントリ = 2 論理ターゲットのみ**（ConvoPeq ×3 config / AudioEngineHarness ×3 config）。両者とも `add_subdirectory(JUCE)`（**:1043**）以降に定義 | v2.3 §2.4 の配置案（option :40 群 + :1043 後 / :1062 前 `add_compile_definitions`）で**両消費側に届く**。ODR / 配置リスクなし（案 E は .cpp 本体のみ・ヘッダ-layout 不変） |
| **R10-6** | **Shadow の実態** | `PolyphaseGainCandidateRef.h`（102 行・untracked）は **DSP 本体を一切持たない API 骨格**。`PolyphaseGainShadow::prepare()` は `stageCount_` を保存するだけ、`reset()` はフラグ 2 個をクリアするだけ、`centerPhaseGain()` は定数返却のみ | **REF-FIDELITY gate（0-1b）はこの骨格に対しては原理的に実行不能**（比較すべき出力系列が存在しない）。R9-3 の「受け皿完成」評価は過大。§2.9 に実装契約を確定 |
| **R10-7** | §2.7.1 再現 | dev 定義を特定: `dev = cand_dB − base_dB − 20·log10((4/3)^N)`、**f は Fs_IN 規格化の絶対 Hz**（0.45→86400 Hz）、インパルス応答の exact DTFT。再実行で v2.3 の **6 値すべてを全桁再現** | v2.3 §2.7.1 は**正しい**（本セッションで一時疑義したが撤回）。S1 @0.45/@0.49 の「適用外」は 31/90 遷移帯（edge 0.1823 / t_end 0.3452 own-rate、f̂_own=0.225/0.245）内であることの帰結として確定 |
| **R10-8** | P0-C 基線 | §2.7.1 の ripple 行は**ラベルは [0.005,0.30] だがモデルは (0.01,0.30) で算出**していた。再実測で **[0.005,0.30] == [0.01,0.30]（3 桁一致・309,330 bin）** を確認 → gate 値の実質は不変 | ラベルのみ是正。加えて **[0.005,0.40] の基線を新規取得**（S1 1.261/0.928・IIR3 0.059/0.044・LP3 0.029/0.022）→ §2.8.1 が authoritative 基線 |
| **R10-9** | F-2 ライブ化 | 誤順条件で `staging=0 eqAGC=1` を実測（v2.3 は静的読み取りのみ）。コード根拠: `AudioEngine.h:1416-1425`（ON→OFF 遷移時 `setAGCEnabled(!enabled)`）/ 既定 true `:2626` / 誤順 `BassBuzz:1799→:1803` / 正順 `:1833→:1834` | 修正案 A の検証条件 `staging=0 eqAGC=0` は有効。B-1 との無関係性も再確認（AGC 強制でも round-trip 0.75^N） |
| **R10-10** | F-1 棚卸し | PVIT **519 行 / TEST 38**（`TEST(|TEST_F(` 実測 count 38）、`checkNoConflictingTransitions` は `RuntimePublicationValidator.h:101`（`:92 private:` の配下）、`CMakeLists.txt` 未登録（`add_executable` 39）、gtest 利用は本ファイルのみ、外部呼出 9 箇所は**現状コンパイル不可** | v2.3 §4.1 の数値はすべて再現。移行方針 A→B→B'→C（failureReason 主判定）を維持 |
| **R10-11** | 静的解析 | cppcheck 2.21.0（`--std=c++20`, warning/performance/portability）: `CustomInputOversampler.cpp` / `TruePeakDetector.cpp` とも**指摘 0**。clang-tidy は単独実行不可（JUCE ヘッダ未解決）→ `build/compile_commands.json` 経由が必須 | v2.3 §7.1「指摘 0」を再現。Dr. Memory は本マシンで計測不能（rc=127）→ §14 に実態 |
| **R10-12** | **P0-F  latency 確定** | `AudioEngine.Processing.Latency.cpp:119` の「**要実測確認**」マーカーをモデル側で解消: 単位インパルス応答の peak 位置は **S1=15 / 511単段=255（dev 0.0000）**、**IIR3=291 / LP3=583（dev +0.75）**。数式 `D=Σ(taps−1)/2^(s+1)` = 290.25 / 582.25 | **P0-F の ±1 許容は必要かつ十分**（±0.5 は 3 段で FAIL）。非整数群遅延 .25 の離散化が +0.75 の原因。base/cand で peak 位置は同一 → 案 E は latency に非影響 |
| **R10-13** | 環境実態の是正 | AGENTS.md 注記 8「**Maxima は未インストール**」は**誤り**（`C:\maxima-5.50.0\bin\maxima.bat` で稼働・クロスチェック実施済）。`graphify`/`tgrep` は `~/.local/bin` の exe ではない（各々 `Python314\Scripts\graphify.exe` 0.9.64 / `C:\Windows\System32\tgrep` 1.0.4＝trigram grep）。headroom proxy は Qoder トラフィックを圧縮しない（`savings` Today 0/0） | §14 にツール別実使用記録。§14 の表にない「使用できたかのような」記録は残さない |

### v2.3 の是正（E-1 〜 E-9）

| ID | v2.3 の記述 | 実測・正 | 影響 |
|----|-------------|---------|------|
| **E-1** | §1 O-7/O-10「`AGENTS.md` **+5/−3**」 | `git diff --numstat HEAD` = **+38/−3** | 棚卸し数値の誤り。§1 で修正 |
| **E-2** | §1 O-9「`docs/tool-inventory-2026-09-20.md`」 | 実パスは **`doc/tool-inventory-2026-09-20.md`**（`docs/` は存在しない: `ls docs` → No such file） | 路径タイポ。3,102 bytes・untracked は正しい |
| **E-3** | §10.1「`DSPCoreDouble.cpp`」 | 正式パス **`src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`**（:505/:513 は正しい） | 参照可能性の改善 |
| **E-4** | §1 O-15「下準備で追加された untracked は `PolyphaseGainCandidateRef.h` と `remediation_v22_prep_patches_20260922.md`」 | untracked は**総計 20 件**（doc 直下 2・`.opencode/` 1・`doc/work113/` 15・`src/` 1）。`model_polyphase_20260920.py` / `_results.txt` / 計画書 v1.6〜v2.3 / `renew_plan*.md` / `residual_tasks_20260920.md` / `remediation_v22_intermediate_20260921.md` が未列挙 | O-13/O-15 の commit 判断対象が過小。§1 で完全棚卸し |
| **E-5** | §1 に **`evidence/epoch_reclaim_audit.json`** の記載なし | セッション開始時点ですでに ` M`。**git 上 Bin 84 → 84 bytes**（同一長・内容差分）、`file` 判定は **`data`**（.json 拡張子だが非テキスト） | **O-16 として新規計上**。追跡対象 evidence の実体が不明確なまま Phase 0 を迎える構造的欠陥 |
| **E-6** | §1 に **`doc/WorkBuddy_AI_Desktop_Toolchain.md`** の記載なし | untracked（`doc/` 直下） | **O-17 として新規計上** |
| **E-7** | §13 末尾「**技術的未確定 0**」 | v2.4 で **残存技術判断 2 件（T-1・T-2）を明示**（§11.2）。T-1 は R9-3 のshadow実装契約、T-2 は cand D1 の扱い。両者とも v2.4 で**確定案を提示**済みなのでユーザー承認のみ | 「0」は撤回し「**未確定は判断 2 件＋意思決定 N 件**」に是正。0 を主張したことで Phase 0 の前提が検出されなかったことが今回の主たる発見 |
| **E-8** | §2.1「1 stage: DC round-trip 0.750000（**up 平均 0.75 / down 単体 DC 1.0**）」 | 意味は正しいが **peak 系の実測（r=2: upGain=1.000000 / downGain=0.750000）と並置すると矛盾して見える**。両者は測定定義の違い（R10-3） | §2.1 を「peak/mean 双方の定義を明記」する形に書き換え |
| **E-9** | §2.5「D1 期待値 cand **−84〜−116 dB**」 | 実測は **−72.11〜−164.50 dB**（R10-4）。窓・trim・N で 92 dB 振幅 | cand の数値レンジ記載を削除し「手法依存・記録のみ」に置換 |

---

## 0. 凡例と全体戦略

| 区分 | 着手優先度 | 影響範囲 | ロールバック |
|------|------------|----------|--------------|
| **P0 即時**（破壊的・不可逆） | なし | — | — |
| **P1 高** | B-1 | 全オーディオ経路（RT: `DSPCoreFloat.cpp:263/:419`・`DSPCoreDouble.cpp:361/:541`／SoftClip 局所: `Float:405/:413`・`Double:505/:513`） | flag OFF rebuild（compile-time rollback） |
| **P2 中** | F-1〜F-4 | test-only（F-3 は production 潜在欠陥の記録） | ファイル revert |
| **P3 低** | O-1〜O-17, B-3, D-1, D-2, R-1, R-2 | 環境 or CLI or 記録 | 設定 revert |

**全体戦略**: §6 の実行順序（Step 0 現状固定 → Step 1 Phase 0 → Step 2 review → Step 3 F-2 → Step 4 F-1 → Step 5 O-* → Step 6 Phase 1+ → Step 7 別課題）。

**v2.4 での位置**: Step 0 の技術側は完了。Step 1 着手前に残っているのは **G-0 承認・Phase 0 GO（ユーザー）＋ T-1/T-2 の承認（本書 §11.2）**。

**B-1 触達点の完全列挙**（v2.4 新規。Phase 1 の影響評価母数）:
```
CustomInputOversampler.cpp :392  prepareSingleStage   （定義）
                              :452/:469  reset / clearAllStages
                              :725/:785  processUp / processDown
CustomInputOversampler.h   :35/:36/:41                （宣言）
AudioEngine.h              :971  CustomInputOversampler softClipOS
DSPCoreLifecycle.cpp       :188/:261  prepareSingleStage(31, 90.0, internalMaxBlock)
DSPCoreFloat.cpp           :263/:419  oversampling up/down   :405/:413  softClipOS up/down
DSPCoreDouble.cpp          :361/:541  oversampling up/down   :505/:513  softClipOS up/down
AudioEngine.Processing.Latency.cpp :6-8 static_assert 群 / :22-23 tap表 / :29-32 群遅延 / :119 要実測マーカー
```

---

## 1. §1 未 commit / 未 push 対応（O-1〜O-17・完全棚卸し）

### 1.1 実測 diffstat（`git diff --numstat HEAD` / `git ls-files --others --exclude-standard`）

| ID | パス | 実測 | v2.3 との差分 | 推奨 |
|----|------|------|----------------|------|
| **O-1** | `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | **+634/−2** | 一致 | 案 A（最小実行可能資産）で commit |
| **O-1b** | `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | **+8/−0** | 一致 | O-1と同 commit |
| **O-2** | 台帳更新 | 独立 commit | — | 独立 |
| **O-3 / O-12** | `ConvoPeq.md` | **+73/−2** | 一致 | **commit しない**（再生成: `python output_sourcecode_markdown.py`） |
| **O-4** | `Testing/Temporary/CTestCostData.txt` | **D**（削除） | 一致 | 現状維持 |
| **O-5** | `.opencode/opencode.json` | untracked | 一致 | 触らない |
| **O-6** | push | ahead 2 | 一致 | **ユーザー手動** |
| **O-7 / O-10** | `AGENTS.md` | **+38/−3** | **E-1（v2.3 は +5/−3）** | 環境記録として commit 可。※§14 の Maxima 誤記を含むため**是正コミットと同時**が望ましい |
| **O-8** | `tools/__pycache__/apply-solidlsp-bash-ls-patch.cpython-314.pyc` | M（追跡済みバイナリ） | 一致 | commit しない → 将来 `.gitignore` + `git rm --cached` |
| **O-9** | `doc/tool-inventory-2026-09-20.md` | untracked 3,102 B（**doc/** 配下） | **E-2（v2.3 は docs/）** | 触らない（ユーザー判断） |
| **O-11** | `doc/work113/residual_tasks_20260919.md` | **+71/−1** | 一致 | 台帳として commit 可 |
| **O-13** | `doc/work113/*.md` untracked **15 件**（v1.6〜v2.4 計画書系列・intermediate・prep_patches・renew_plan・renew_plan_verification・residual_tasks_20260920・**model_polyphase_20260920.py / _results.txt**） | **E-4（件数と範囲を確定）** | 承認版+台帳を 1 commit。**py/results も同一 work item の evidence なので同梱を推奨** |
| **O-14** | `.mcp.json` | **+33/−23** | 一致 | commit（環境記録・O-10 と同種） |
| **O-15** | `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h` | untracked（102 行） | 一致（ただし評価は **R10-6** で是正） | O-13 に同梱可。**ただし commit 前に §2.9 の実装契約を反映した版へ** |
| **O-16（新）** | `evidence/epoch_reclaim_audit.json` | **セッション開始時点で M・Bin 84→84・`file`=data** | **v2.3 未計上（E-5）** | **ユーザー判断**: ①追跡外化（`.gitignore` + `git rm --cached`）②内容特定後に台帳化 ③破棄。放置は不可（下記） |
| **O-17（新）** | `doc/WorkBuddy_AI_Desktop_Toolchain.md` | untracked | **v2.3 未計上（E-6）** | 触らない or O-9 と同種の環境文書として扱いを統一 |

### 1.2 O-16 と Phase 0 の衝突（新規指摘）

`[OS_DIRECT]` 等の計測を実行すると harness が **`evidence/*.json` を書き換える**（本セッションで 8 ファイルが変化。うち `shutdown_trace.json` は schema `v1`→`v4`）。HEAD に commit 済みの evidence は **v1**、現行バイナリは **v4** を出力する。

- **帰結**: commit 済み evidence が stale であり、「監査証憑として git 上の evidence を参照する」運用は成立しない。
- **帰結**: Phase 0 は read-only を標榜するため、**Phase 0 の測定実行自体が作業ツリーを汚す**。v2.4 はこれを欠陥として扱う（gate ではない）。
- **推奨（Phase 0 前に確定させる）**: 測定は `--buzz-out=` 等の明示パスで `tmp/`（`.gitignore` 済み）へ退避するか、evidence 群を追跡外化。**どちらもユーザー判断**。
- 注: 本セッションでは session-start で clean だった 8 ファイルを `git checkout -- evidence/` で復元した。`epoch_reclaim_audit.json` のみが開始時点から変更済み（=O-16）で、これは復元していない。

---

## 2. §2 B-1: CustomInputOversampler up/down round-trip 利得欠陥

### 2.1 確定している事実（v2.4・測定定義を明記）

```
主因        CustomInputOversampler の up 段 center 位相に ×2 が無い（polyphase 利得規約の不対称）
欠陥局在    interpolateStage の centerValue 出力のみ（:543-564）。decimateStage は正規（DC 利得 1.0）
round-trip  DC: base 0.75^N / cand 1.0（12 桁）  ← invariant（gate はこれのみ）
            ratio 1/2/4/8 = 1.000000 / 0.750000 / 0.562500 / 0.421875（preset 非依存）
段ごとの和  conv 分岐: Σcoeffs 0.5 → ×2 → DC 利得 1.0（正しい）
            center 分岐: centerCoeff 0.5 → ×2 なし → DC 利得 0.5（不対称＝欠陥）
            decimate: 0.5 + 0.5 = 1.0（正しい）
利益        (4/3)^N = +2.498775 / +4.997549 / +7.496324 dB（1/2/3 段）
画像        up 出力の像/トーン = 0.25/0.75 = 1/3 = −9.542425 dB（構造定数・手法不変）
係数        FIRsum 1.000000000000000 / center 0.500000000000000 / convSum 0.500000000000000（全 6 design）
契約不整合  isLinearPhaseFIR / isSymmetricUpDown（h:21-22）・Latency.cpp:6-8 static_assert と矛盾
production  変更 0 / commit 0（v2.4 で再確認）
```

**peak / mean の帰属（E-8・R10-3 の確定表現）**: DC 入力を up すると出力は位相交互に `1.0, 0.5, 1.0, 0.5, …` となる。**平均は 0.75、ピークは 1.0**。down 段は対称 tap によりナイキスト変調（±0.25 成分）を除去するため、round-trip は平均 0.75 に収束する。したがって
- peak ベース計測 → 「up 1.0 / down 0.75」（失って見えるのは down）
- mean ベース解析 → 「up 0.75 / down 1.0」（失って見えるのは up）
の**両方が同じ階段状信号の別々の記述**であり、**因果の根拠にはできない**。因果は上記「段ごとの和」＝係数レベルの分岐 DC 利得（1.0 と 0.5 の不対称）で確定する。

### 2.2 根本原因（行番号・v2.4 で再検証）

| 箇所 | 内容 |
|------|------|
`prepareStage` :287-390 | `taps=jmax(3,taps|1)`・`centerTap=(taps-1)/2`・`centerParity=centerTap&1`・`convParity=1-centerParity`・Kaiser β 三分岐 `:301-304`・sinc `:308-317`（center で 0.5）・halfband ゼロ化 `:319-323`・`sum→1.0 :325-333`・`rawCoeffs[centerTap]=0.5 :335`（`:348` でも再設定）・非 center を 0.5 へ `:336-347`・`convCount=(taps-convParity+1)/2 :350`・`convCoeffs`/`convCoeffsReversed` 構築 `:360-365`・`centerCoeff :367`・`centerDelayInput=(centerTap-centerParity)/2 :368`・`historyUpKeep :369`・`historyDownKeep :372`・size `:374-375`
`interpolateStage` :492-568 | `:543 centerValue = 0.0` / `:545 centerValue = stage.centerCoeff * history[idx - stage.centerDelayInput]` / `:549 bad 判定` / **`:557 convValue *= 2.0`（conv 側のみに ×2）** / `:559 centerValue denorm クリア` / **`:563 output[outBase+convParity]=convValue`・`:564 output[outBase+centerParity]=centerValue`（center に ×2 なし）**
`decimateStage` :570-723 | silence fast path `:583-613` / `copy(history+keep,input) :615` / `coeffs = convCoeffs.get() :618` / 境界 guard `:620-649` / **`:658 double acc = stage.centerCoeff * centerSample;`** → conv tap 累加（**全体で ×2 なし＝DC 利得 1.0 で正しい**）
`processUp` :725-783 / `processDown` :785- | hardFallback 透過・corruption クリア
`reset()` :452-467 vs `clearAllStages()` :469-484 | アトミック **3** vs **1** → Phase 0 は `reset()` 経由必須（R4-3 継承）
`prepareSingleStage` :392- | SoftClip 局所 2× OS の構築経路（`Lifecycle :188/:261` で `(31, 90.0, internalMaxBlock)`）
SIMD | `dotProductAvx2 :159-216` は **convValue 専用**。center 位相への ×2 は SIMD 非依存（スカラー追加 1 行で済む）。`:175-178` prefetch 境界 guard・`:194-199` vextractf128+hadd 削減・`:202-204` `_mm256_zeroupper()`（MS Learn で VHADDPD/`_mm256_hadd_pd` 意味論と denormal 規定を裏取り）

**×2 の不在は 5 手段で独立確認**: `rtk rg`（0 hit）/ **tgrep 1.0.4**（`centerValue *= 2` → 0 hit、`convValue \*= 2` → 1 hit）/ **ast-grep 0.44.0**（同じく 1 hit / 0 hit）/ **AiDex**（`centerValue` 5 hit = :543/:545/:549/:559/:564 のみ）/ **serena**（17 メソッド overview）。

### 2.3 5 系統独立再現（R10-1）

| # | 系統 | 実行 | base r=2 | base r=8 | cand | Δ dB | image/tone |
|---|------|------|----------|----------|------|------|------------|
| 1 | **production バイナリ** | `AudioEngineHarness.exe`（無引数）→ `[OS_DIRECT]` | 0.750000 | 0.421875 | 1.0* | +2.4988 | 未測（peak 系） |
| 2 | authoritative モデル | `model_polyphase_20260920.py` | 0.750000000000 | 0.421875000000 | 1.000000000000 | +2.4988 | −9.5424 |
| 3 | NumPy 逐語転写 | `tmp/cio_literal.py`（C++ ループを 1:1 写し） | 0.750000 | 0.421875 | 1.000000 | +2.4988 | −9.5424 |
| 4 | GNU Octave 11.3.0 | `tmp/cio_octave.m` | 0.75 | 0.421875 | 1.0 | +2.4988 | −9.5424 |
| 5 | Maxima 5.50.0 | `tmp/maxima_check.mac` / `--batch-string` | 0.75 | 0.421875 | 1.0 | +2.4988 | 記号 |

\* cand 列は production 実測では得られない（flag 未適用）。モデル・Octave・Maxima で 1.0 を確認。

**設計点（6 design）もソース確定**: `tapsForStage :84-94` → LinearPhase `{1023,255,63}`・IIRLike `{511,127,31}`。`attenuationForStage :96-106` → LinearPhase `{160,140,120}`・IIRLike `{140,110,90}`。よってモデルの DES 6 点（31/90・127/110・511/140・63/120・255/140・1023/160）は production と 1:1 対応。

**文献準拠の単独テスト（R10-1 補強）**: dsp-eleted（DSPRelated）で Rick Lyons / Tim Wescott / David G.（dgshaw6）が述べる規約 —
- 「each polyphase branch の係数和を別々に足して DC 利得が 1.0 か確認せよ」（dgshaw6）→ 本件は **1.0 と 0.5 で不対称** → 欠陥独立判定成立。
- 「ピーク振幅 1 の低周波正弦を入れ、出力のピークが 1/2 か 1 かを見よ」（Lyons）→ Phase 0 の 0-3 実測手順はこの指示と同一。
- 「間隙にゼロを挿入するとゲインが 1/L になる。実務者は FIR 係数に L を乗じて補償する」（Lyons）→ 半帯域 2× 補間では**両分岐に ×2**が正規。conv 側のみ ×2 は「半分だけ補償」の状態。
出典: `https://www.dsprelated.com/thread/1990/interpolation-and-filter-gain`（Lyons/Wescott/dgshaw6 応答 2017-02-23）。補助: MathWorks `dsp.FIRHalfbandInterpolator` / `dsp.FIRInterpolator`（polyphase・noble identity の一次説明）。**Vaidyanathan / Crochiere-Rabiner の原典本文は未到達**（この点のみ二次資料依存として記録）。

### 2.4 ライブ帰因テーブル（R10-2・authoritative）

`AudioEngineHarness.exe`（**引数なし**で実行。`--buzz*` を付けると `PublishPipelineIntegrationTests.cpp:1123` の分岐で測定 entry に派遣され `:1223` の無条件 attribution に到達しない — v2.4 で確定制御）。

```
[OS_DIRECT]  preset=IIRLike     r=1  in 0.5  out 0.500000 rt=1.000000 up=1.000000 dn=1.000000 thd=-153.987dB
[OS_DIRECT]  preset=IIRLike     r=2  in 0.5  out 0.375000 rt=0.750000 up=1.000000 dn=0.750000 thd=-152.094dB
[OS_DIRECT]  preset=IIRLike     r=4  in 0.5  out 0.281250 rt=0.562500 up=0.750001 dn=0.749998 thd=-151.010dB
[OS_DIRECT]  preset=IIRLike     r=8  in 0.5  out 0.210937 rt=0.421875 up=0.650885 dn=0.648156 thd=-152.452dB
[OS_DIRECT]  preset=LinearPhase r=2  ...      rt=0.750000（IIRLike と一致）
[OS_DIRECT]  preset=LinearPhase r=4  ...      rt=0.562500 up=0.750002 dn=0.749995
[OS_DIRECT]  preset=LinearPhase r=8  in 0.5  out 0.210937 rt=0.421875 up=0.650890 dn=0.648151
[OS_DIRECT]  singleStage taps=31 atten=90.0 r=2  rt=0.750000 up=1.000000 dn=0.750000
[EQ_DIRECT]  base192k_2048_agc0  in 0.5 out 0.500000 ratio=1.000000 thd=-153.990dB activeBands=0 paramsEnabled=0 totalGain=0.000000dB
[EQ_DIRECT]  rt768k_4096_agc1    ratio=1.000000 thd=-153.820dB   [EQ_DIRECT] rt768k_4096_agc0 ratio=1.000000 thd=-153.820dB
[OF_DIRECT]  sr=192k lpMode=0/1/2 ratio=0.987446 / 0.987442 / 0.987434   sr=768k lpMode=0/1/2 ratio=0.987446 / 0.987442 / 0.987433
```

**結論（確定）**:
1. 測定対象 3 経路のうち**唯一の桁違い損失は Oversampler**（0.75^N）。EQ は 3 条件とも ratio=1.000000（完全無損失・activeBands=0 の flat 構成）、OutputFilter は ratio≈0.9874 = **−0.110 dB**（lpMode/sr 非依存の一定値）。
2. `rt = up × dn` が全 row で成立（1.0×1.0 / 1.0×0.75 / 0.750001×0.749998 / 0.650885×0.648156 = 0.421875）。
3. **preset（IIRLike / LinearPhase）非依存**。tap 構成が違っても 0.75^N が同一 → 利得規約の問題であり tap 設計の問題ではない（案 E の妥当性を支持）。
4. up/dn 分割値は段数に依存して変わる（R10-3）。**P0 gate にしてよいのは rt のみ**。

### 2.5 修正案と DESIGN-CONTRACT-A

案 A〜D 不採用・**案 E 有力仮説**（E1〜E5 は v2.1/v2.2 のまま。E3 は R7-1 補正、E5 は JUCE 補助証拠）。

```cpp
// src/CustomInputOversampler.cpp interpolateStage() 内（:557 の直後）
        convValue *= 2.0;                 // 既存（:557）
#if CONVOPEQ_CORRECT_POLYPHASE_GAIN
        centerValue *= 2.0;               // ★ 案 E（両 polyphase 位相への ×2 対称適用・candidate）
#endif
        if (fastAbs(convValue) < kDenormThreshold) convValue = 0.0;
        if (fastAbs(centerValue) < kDenormThreshold) centerValue = 0.0;   // 既存 :559
```

**構造的恒等式（R5-9 継承・v2.4 でも有効）**: `h_rt = 2·(convCoeffs ⋆ convCoeffs) + c·δ[n−15]`（c: base 0.25 / cand 0.5・残差 ≤1.4e-17）。passband で cand/base = (4/3)^N。

**Phase 1 影響評価の母数**: §0 の触達点リスト（RT 2 経路 × float/double + SoftClip 局所 OS × float/double）を**全 4 経路で**P0-B/C/D/E を確認する（v2.3 は经路分解を明記せず）。

### 2.6 CMake / flag（R10-5 で確定）

```cmake
# 1) option 群（:40 近傍・既存 8 個の option と並置）
option(CONVOPEQ_CORRECT_POLYPHASE_GAIN
       "Correct polyphase gain convention (B-1 案E)" OFF)

# 2) add_subdirectory(JUCE)（:1043）の後・juce_add_gui_app(ConvoPeq（:1062）の前
add_compile_definitions(
    CONVOPEQ_CORRECT_POLYPHASE_GAIN=$<BOOL:${CONVOPEQ_CORRECT_POLYPHASE_GAIN}>)
```

- **配置の十分条件を実測確定**: CIO.cpp をコンパイルするのは **ConvoPeq（`target_sources(... ${CONVOPEQ_ALL_SOURCES})` :1282、リスト内 :1196、閉じ :1280）** と **AudioEngineHarness（`add_executable` :1899）** の 2 論理ターゲット（×3 config = compile_commands 6 エントリ）。両者の定義は :1043 より後 → **1 か所の `add_compile_definitions` で両者に届く**。39 の `add_executable` のうち残り 37 は CIO.cpp を参照しない。
- **ODR 安全性**: 案 E は `.cpp` 関数本体のみの変更（ヘッダ・`Stage` layout・アトミック・public シグネチャ不変）→ 混在ビルドでも layout 不一致は発生しない。ただし**部分的な flag 混在は意味的に危険**（一方が 1.0・他方が 0.75^N）なので、ビルドは always global（`build.bat` `CMAKE_EXTRA_FLAGS` :179/:181 経由で `-D...=ON`）で行う。
- 既存前例: `NUC_DEBUG_GUARDS`（:52-58 の `add_compile_definitions`）。
- C++ は **`#if CONVOPEQ_CORRECT_POLYPHASE_GAIN`**（`#ifdef` 禁止＝値 0 でも有効に評価させるため）。
- 既定 OFF / runtime flag 不採用。**現状 未適用**。
- 注: `CONVOPEQ_CORRECT_POLYPHASE_GAIN` は **tgrep 全リポジトリ検索で production に 1 件も存在しない**（ヒットするのは `doc/work113/*.md` の計画文のみ）。test 側には別名 `CONVOPEQ_POLYPHASE_REF_CANDIDATE` が存在（→ §2.9・T-1）。

### 2.7 Phase 0 characterization（測定仕様・v2.4 改定）

```
0-0  G-0: DESIGN-CONTRACT-A（E1〜E5）+ T-1/T-2 をユーザーが明示承認
0-1  Baseline record-only: round-trip DC = 0.75^N ±1e-6（invariant のみ gate・up/dn 分割は診断ログ）
0-1b REF-FIDELITY: Shadow == production（bitwise・reset() 経由・§2.9 の実装契約に基づく）
0-2  Shadow Candidate E: round-trip DC = 1.0 ± 1e-6
0-3  周波数（D1/D2/D3）— D1 は base 側のみ gate（§2.8.2）・D2 gate [0.005,0.40]
0-4  block/reset bitwise（partition 4 種 + reset contract）
0-5  SoftClip local OS（float + double・R8-10）
0-6  float/double equivalence（maxAbsErr ≤5e-7 / RMS ≤5e-8）
```

**D1 定義（v2.4 改定・R10-4）**:
- 軸: **D1 = |Y_up(image bin)| / |Y_up(tone bin)|**、up 出力長 2N、**tone bin = f̂·N**、**image bin = N−f̂·N**（R8-1/R9-1 継承）
- **gate する量**: `D1_base = −9.542425 ± 0.01 dB`（1 段あたり。手法不変を 81 測定で確認）
- **gate しない量**: `D1_cand`（同条件で −72.11〜−164.50 dB と 92 dB 振幅）。**記録のみ**。矩形窓・Hann+trim8 の**2 条件を必ず併記**（手法差を報告可能な形に残す）
- tone/image bin は expected を**主報告**、±8 bin argmax は**検証ログ**。argmax 漂移を FAIL と判定しない（R9-2 継承）
- final alias rejection は引き続き P0-D（stopband）が担当

**Shadow Reference**: §2.9（T-1）に実装契約を確定。骨格のみでは gate 実行不能。

**D2 / D3**: v2.2/v2.3 のまま。D2 gate [0.005, **0.40**] base==cand ≤0.01 dB、0.45 は記録のみ（遷移帯）。D3 worst spur −98〜−101 dB（窓 sidelobe）・base==cand 差 0.00 dB。

### 2.8 B-1-P0 GATE（判定表・v2.4）

| ID | 判定対象 | 条件 | v2.3 からの変更 | 合否 |
|----|----------|------|----------------|------|
| G-0 | 契約 | DESIGN-CONTRACT-A（E1〜E5）+ **T-1・T-2** 明示承認 | T-1/T-2 を追加 | □ |
| G-BL | Baseline sanity | **round-trip DC = 0.75^N ±1e-6 のみ**。up/down 分割値は診断ログ（gate 外） | 分割を gate から除外（R10-3） | □ |
| REF-FIDELITY | Shadow==production | bitwise（tolerance 0）・**§2.9 契約の 3 対象**（係数配列 / 出力系列 / reset 前後状態）・`reset()` 経由 | 対象を 3 項目に具体化（R10-6） | □ |
| P0-A | Candidate DC | round-trip 1.0 ±1e-6（必要条件・十分条件ではない） | — | □ |
| P0-B | 低域 | 50 Hz / 1 kHz unity ±0.1 dB。**4 経路**（RT float/double・SoftClip float/double）で | 经路分解を明記 | □ |
| P0-C | passband ripple | **[0.005,0.30] max−min ≤0.05 dB（dense）**・基線は §2.8.1 | 基線を v2.4 実測で確定（R10-8） | □ |
| P0-C' | differential | (4/3)^N ±0.05 dB（**IIR3/LP3 f≤0.45**・S1 f≤0.30） | §2.7.1 再現確認済み（R10-7）→ 条件は妥当なまま | □ |
| P0-D | stopband | alias ≤ −(A−10) dB @[t_end, 0.5] | §2.8.3 の FIR 絶対基準を使用 | □ |
| P0-E | **image invariance** | **E-1: base D1 = −9.5424 ±0.01 dB（gate）** / **E-1b: cand D1 2 条件記録（gate 外）** / E-2: D2 [0.005,0.40] base==cand ≤0.01 dB / E-3: 0.45 記録のみ | **gate を base 側に移動**（R10-4） | □ |
| P0-F | latency | peak ∈ [floor(D), floor(D)+1]、D=Σ(taps−1)/2^(s+1)。**基線: S1 15（dev 0）・511単段 255（dev 0）・IIR3 291（dev +0.75）・LP3 583（dev +0.75）** | ±1 が必須であることを実測確定（R10-12）。base==cand も追加 | □ |
| P0-G | float/double | maxAbsErr ≤5e-7 / RMS ≤5e-8 | — | □ |
| P0-H | block/reset | partition bitwise・reset contract（atomic 3 / atomic 1 の区別） | — | □ |
| P0-I | SoftClip | I-a DC=1.0±1e-6 / I-b 安全性 gate / I-c 動作点 record（`prepareSingleStage(31,90.0)`・`Lifecycle:188/:261`） | 触達点を実測で確定 | □ |

**全 PASS → Phase 1 eligibility**。FAIL → ①案 D 統合 → ②tap 再設計 → ③TruePeakDetector 型（参照・第三順位）。

#### 2.8.1 passband ripple 基線（v2.4 新規実測・exact DTFT・dense）

| config | 帯域 | base | cand | bins |
|--------|------|------|------|------|
| S1 (31/90) N=1 | **[0.005,0.30]** | **0.001** | **0.001** | 309,330 |
| | [0.005,0.40] | 1.261 | 0.928 | 414,188 |
| | [0.005,0.45] | 5.242 | 3.607 | 466,617 |
| IIR3 (511/127/31) | **[0.005,0.30]** | **0.034** | **0.025** | 309,330 |
| | [0.005,0.40] | 0.059 | 0.044 | 414,188 |
| | [0.005,0.45] | 0.111 | 0.084 | 466,617 |
| LP3 (1023/255/63) | **[0.005,0.30]** | **0.016** | **0.012** | 309,330 |
| | [0.005,0.40] | 0.029 | 0.022 | 414,188 |
| | [0.005,0.45] | 0.053 | 0.039 | 466,617 |
| 511/140 single | [0.005,0.30] | 0.000 | 0.000 | 309,330 |

`|H| cand @50Hz`: S1 −0.0000 / IIR3 −0.0078 / LP3 −0.0035 / 511単段 +0.0000 dB。`base @50Hz`: −2.4988 / −7.5067 / −7.5010 / −2.4988 dB。`cand 0.1 dB edge`: S1 0.3526 / IIR3 0.4897 / LP3 0.4945 / 511単段 0.4886 Fs_in。
**重要**: [0.005,0.30] と [0.01,0.30] は全 config で 3 桁一致 → v2.3 §2.7.1 の値はそのまま gate に使える（ラベルのみ v2.4 で是正）。

#### 2.8.2 D1 手法不変性（v2.4 新規実測・R10-4 の根拠）

| design | f̂ | 条件 | base | cand |
|--------|----|----|------|------|
| 31/90 | 0.05 | rect N=8192 | −9.544 | −80.25 |
| | | rect N=32768 / N=131072 | −9.543 | −91.31 / −90.01 |
| | | trim8(N=taps×8) | −9.543 | −75.86 / −92.23 / −90.18 |
| | | trim8+Hann | −9.543 | −90.84 |
| | 0.20 / 0.30 | 同上 9 条件 | −9.541〜−9.545 | −74.86〜−104.51 |
| 63/120 | 0.05 | 同上 | −9.542（全条件同一） | −82.50〜−135.01 |
| | 0.20 / 0.30 | 同上 | −9.542〜−9.543 | −80.88〜−130.08 |
| 511/140 | 0.05 | 同上 | −9.542〜−9.543 | −73.76〜−163.41 |
| | 0.20 / 0.30 | 同上 | −9.541〜−9.546 | −72.11〜−164.50 |

- **base: 81 測定すべて [−9.546, −9.541]。構造定数 −9.542425 との最大差 0.0036 dB → gate 可**
- **cand: [−72.11, −164.50]。最大 92.4 dB の手法依存 → gate 不可（記録のみ）**
- **R9-5 の確定解消**: 表 §2.7.3 の 63/120@0.2 = −103.3 dB とモデル −102.36 dB の Δ0.94 dB は、上記 92 dB の感度帯に対する窓・trim・N の差であり、**物理量でも設計差異でもない**。Phase 0 の C++ 実測で「一致/不一致」を論じる対象ではない。

#### 2.8.3 FIR 絶対基準（P0-D floor・v2.3 から引き継ぎ・再確認）

| design | −0.1 dB edge | transition_end | floor（A−10） |
|--------|--------------|----------------|---------------|
| 511/140 | 0.2448 | 0.2590 | −130 dB |
| 127/110 | 0.2317 | 0.2782 | −100 dB |
| 31/90 | 0.1823 | 0.3452 | −80 dB |
| 1023/160 | 0.2472 | 0.2552 | −150 dB |
| 255/140 | 0.2396 | 0.2681 | −130 dB |
| 63/120 | 0.2109 | 0.3130 | −110 dB |

### 2.9 Shadow 実装契約（T-1 の確定案・R10-6）

`src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h` の実態（v2.4 全文読了）:

| 要素 | 実態 | REF-FIDELITY への影響 |
|------|------|------------------------|
| `namespace convo::polyphase_ref` | test-only・production 変更 0 | ✓ 問題はなし |
| `expectedDcRoundTrip(stages, candidate)` | base 0.75^N / cand 1.0 | ✓ R10-1 と一致（正しい） |
| `computeD1Bins(fhat, N)` | tone=round(f̂N) / image=round((1−f̂)N) | ✓ R9-1/§2.7 の補正軸と一致（正しい） |
| `class PolyphaseGainShadow::prepare(taps,atten,stageInputMax,stages)` | **`stageCount_ = stages` を保存するだけ**（コメント `// prepare は production と同一パラメータ契約` だが実装なし） | ✗ **係数生成が存在しない** |
| `::reset()` | `prepared_=false; stageCount_=0` のみ（production は atomic 3 個＋履歴 clear） | ✦ **production のリセット契約を模倣していない** |
| `centerPhaseGain()` | `#if CONVOPEQ_POLYPHASE_REF_CANDIDATE` で 2.0 / 1.0 を返す**のみ** | ✗ 信号経路がない |
| processUp/processDown 相当 | **存在しない** | ✗ **bitwise 比較の対象が無い** |

**帰結（確定）**: v2.3 R9-3 の「Phase 0-1b/0-2 の受け皿」は**受け皿として機能していない**。`0-1b REF-FIDELITY` は現状の骨格では実行不能であり、v2.3 の「技術的未確定 0」は誤りだった（E-7）。

**T-1 確定案（Phase 0-1b の実装契約）**: production を触れない制約下では shadow は複製実装が唯一。以下を gate の前提条件として固定する。

1. **係数生成の一致**: `taps=jmax(3,taps|1)` → `centerTap`/`centerParity`/`convParity` → Kaiser β 三分岐（`:301-304`）→ sinc（`:308-317`）→ halfband ゼロ化（`:319-323`）→ `sum→1.0`（`:325-333`）→ `rawCoeffs[centerTap]=0.5` → 非 center 0.5 化（`:336-347`）→ `convCoeffs` / `convCoeffsReversed`（`:360-365`）。**浮動小数の演算順序まで一致させる**（bitwise 判定のため）。
2. **履歴・分岐順序の一致**: `historyUpKeep`/`historyDownKeep`（`:369/:372`）、silence fast path（`:583-613`）、境界 guard（`:620-649`）、denorm クリア（`:558-559`）、`isBadSample` の適用位置。
3. **比較対象（3 項目・tolerance 0 = bitwise）**:
   - (a) prepare 後および prepareSingleStage 後の**係数配列全要素**（`rawCoeffs`/`convCoeffs`/`convCoeffsReversed`/`centerCoeff`/各 keep 長）
   - (b) 決定的疑似乱数入力（シード固定）**3 ブロック × 2 preset × ratio {2,4,8}** の processUp / processDown 出力系列
   - (c) `reset()` 前後および `clearAllStages()` 前後の状態差分（**production の atomic 3 vs 1 の区別を再現**していること）
4. **candidate 切替のマクロ系統を統一**（新規発見・要承認）: production 側は `CONVOPEQ_CORRECT_POLYPHASE_GAIN`、shadow 側は `CONVOPEQ_POLYPHASE_REF_CANDIDATE`（tgrep 確認: shadow 側は shadow の中にのみ存在）。**2 マクロは独立に ON/OFF でき、位相のズレ（shadow は cand・production は base 等）を黙認する構造**。
   - **推奨**: Phase 0 は `CONVOPEQ_POLYPHASE_REF_CANDIDATE` を保持してよいが、**「shadow の candidate 状態 == production の flag 状態」を asserts する検査を P0-H に追加**。または shadow を同一 CMake option で駆動し、別マクロを削除する。
5. **0-2 Candidate の位置付け**: shadow の cand は「production をフラグ ON でビルドした結果の予測」にすぎない。**最終判定は flag ON ビルドの AudioEngineHarness 実測**（`[OS_DIRECT] rt`）とし、shadow 一致は十分条件ではない（P0-A の注記に統合）。

### 2.10 教訓（v2.3 §2.8 の 1〜28 継承 + 29〜33）

29. **【R10-3】「段ごとのゲイン」を因果の記述に使うな**: peak/mean・単段/多段で符号が反転する。不変量（round-trip）と構造量（分岐係数和）で述べよ。
30. **【R10-4】gate は手法不変な量にのみ置く**: cand image は 92 dB 振幅で動く。不変な base（−9.5424）を gate し、可変なものは 2 条件併記で記録する。
31. **【R10-6】「骨格を作った」を「受け皿が完成した」と呼ぶと、gate が実行不能なまま承認される**: 比較対象が存在しないのに bitwise gate を置いていた。実装契約はコードの形ではなく**比較可能な出力の存在**で検証する。
32. **【R10-11/§14】ツールの有無と稼働形は実測で書く**: AGENTS.md の「Maxima 未インストール」は誤りで、Maxima はクロスチェックに実際には使えた。誤記は調査の射程を狭める。
33. **【E-4/E-5】棚卸しは git の生出力で行う**: 列挙ベースの棚卸し（v2.3 O-1〜O-15）は 20 件の untracked 中 2 件しか覆っておらず、開始時点で変更済みの追跡バイナリ（O-16）を検出できなかった。`git diff --numstat HEAD` + `git ls-files --others --exclude-standard` を棚卸しの一次手段に固定する。

### 2.11 Phase 0 適用要件

production src/ modified=0 / staged=0 / measurement のみ test-only / commit 禁止 / calibration 禁止 / engine-fit 0.75^N unchanged。**加えて（v2.4）**: 測定実行で tracked な `evidence/` が汚れないことを確認してから 0-1 に進む（§1.2）。

---

## 3. §3 harness / production 潜在欠陥（F-2 / F-3 / F-4）

### 3.1 F-2: `--buzz-rigcheck=eq` の実効 AGC

- 根本原因: `setAutoGainStagingEnabled`（`AudioEngine.h:1416-1432`）の **ON→OFF 遷移時のみ** `getEQProcessor().setAGCEnabled(!enabled)`（:1425）。既定 `std::atomic<bool> autoGainStagingEnabled { true }`（:2626）
- eq モード誤順: **:1799 `configureProbeFlatEQ(e)` → :1803 `setAutoGainStagingEnabled(false)`**（結果として AGC が再 ON）
- 正順（eqdiag 等）: **:1833 staging false → :1834 configureProbeFlatEQ**
- `configureProbeFlatEQ`（:1072）内 `setEQAGCEnabled(false)`（:1081）
- **実測で確定（R10-9）**: 誤順条件で `staging=0 eqAGC=1`
- **修正案 A（呼出順を eqdiag パターンへ揃える）**・パッチ案 `remediation_v22_prep_patches_20260922.md` §2（未適用）
- 検証: `AudioEngineHarness.exe --buzz-rigcheck=eq` で `staging=0 eqAGC=0`
- **B-1 の原因ではない**（AGC を強制しても round-trip 0.75^N は不変。§2.4 の [EQ_DIRECT] ratio=1.000000 が独立証拠）

### 3.2 F-3: EQ dry/wet 混合（記録のみ・案 D）

`EQProcessor.Processing.cpp:982-996` の `wetPtr[i] = wetPtr[i]*wetGainState + dryValue*dryGain` 型。遷移時のみ。定常 B-1 の原因ではない。修正は別 work item（scope discipline）。

### 3.3 F-4: `ir` / `irwet` 用語（R5-8 継承）

- **`ir`**: IR ロード後も `setConvolverBypassRequested(true)`（:1745）維持 = **dry ベースライン測定モード**
- **`irwet<digit>`**: :1763 分岐・**:1778 `setConvolverBypassRequested(false)`** = wet 畳み込み対照
- 未知モード fail-closed :1708-1710
- **修正案 B（記録のみ）**

### 3.4 測定 entry 派遣の罠（v2.4 新規・R10-2 の再現条件）

`PublishPipelineIntegrationTests.cpp:1123`: `else if (a == "--buzz" || a.rfind("--buzz-", 0) == 0)` → 任意の `--buzz*` 引数がある時点で測定 entry へ派遣し **return する**ため、**:1223 の無条件 `(void)runEqDirectDriveAttribution();` に到達しない**。`:1217` の `runBuzzArgParserTests()` も同様に到達しない。
→ **§2.4 の attribution を再現するには引数なしで実行する**。スクリプト化する場合この制約を明記すること（v2.3 は無記載）。

---

## 4. §4 F-1: PublicationValidatorIsolationTests 移行・退役

### 4.1 現状（v2.4 で再確認・R10-10）

- 経路: **`src/audioengine/RuntimePublicationValidator.h`（105 行）/ `.cpp`（211 行）**（`src/core/` には存在しない）
- 構造化判定: h:13 `ValidationFailureReason` / h:24 `failureReason` / h:64 `validatePublication`
- `private:` h:**92** / `bool checkNoConflictingTransitions(` h:**101**（本体 cpp:169）
- 検査順序 cpp:8-41: Semantic → Topology → Resources → ConflictingTransitions（first-fail）
- PVIT: **519 行 / TEST 38**（rtk rg `TEST(|TEST_F(` = 38）/ 呼出 **9 箇所**（:135/:246/:255/:265/:273/:281/:313/:324/:334）→ private 呼出のため**現状コンパイル不可**
- CMake 未登録（`add_executable` 総数 39）/ gtest を使うのは本ファイルのみ / `tools/build-debug.bat:29` に stale target 参照
- 分類 25/9 は手動分類の目安（R6-4）

### 4.2 移行方針

Phase 0 分類 → A（CrossfadeAuthority 4）→ B（validator 系 + transition）→ **B' Semantic Equivalence Gate** → C（退役 + build-debug.bat 修正）。
**B' 主判定**: 移植先が `result.failureReason`（enum）で旧 private 検査と一致すること。`errorMessage` は補助 assert のみ。**MIGRATE CASES THEN RETIRE**。

---

## 5. §5 別課題

| ID | 内容 | 優先度 |
|----|------|--------|
| **R-2** | 数値 parse 未捕捉例外 → fail-closed 化。対象: BassBuzz 直接 `stod/stoi/stof`（**:1582 / :1583 / :1590 / :1591 / :1592 / :1614**）+ `parseHcIdx`/`parseLcIdx` 本体（**:1562 / :1567** の `return std::stoi(s);`）+ 呼出（:1606-:1611）+ PPIT `stoi`（**:1136 / :1146 / :1149 / :1157 / :1166**）。仕様: try/catch + `[BUZZ] FAIL: invalid numeric` + 非ゼロ終了 + regression test。**前例あり**: `parseOnOff`（:1572-1578）と `--buzz-order` は既に fail-closed 化済み（commit `c4a08171`）→ 同一パターンで統一可能。スケッチ: prep_patches §3（未適用） | **最優先（極小・前例あり）** |
| **R-1** | `--buzz-flip-eqgain=` silent-ignore（設計上ステップ固定） | 次（極小） |
| **B-3** | timestamp-based capture（現行目的には十分） | 中 |
| **D-2** | headroom ランタイム統合（→ §14: Qoder では proxy 無効。統合先の再定義が必要） | 中 |
| **D-1** | build identity gate M1/M2 | 高 |
| **O-16 派生** | evidence/ 追跡方針（§1.2）。測定が tracked ファイルを書く構造的欠陥の解消 | 中（Phase 0 前） |

---

## 6. 推奨する実行順序（v2.4）

```
[Step 0] 現状固定 — 技術側完了（v2.4 で更に確定）
  - HEAD 8f127bfe / production src/ 未 commit 0（再確認）
  - §2.7.1 dev 値がモデルから全桁再現（R10-7）/ ripple 基線を [0.005,0.30] で確定（R10-8）
  - D1 の手法不変性を 81 測定で確定（R10-4）/ latency 式を実測裏取り（R10-12）
  - 残: ユーザー判断（O-1〜O-17 / G-0 / Phase 0 GO / T-1 / T-2）
  - ★ v2.4 追加: evidence/ が汚れない実行形態を先に決める（§1.2）

[Step 1] §2 B-1 Phase 0（production 変更 0）— G-0 + T-1 承認が前提
  - 0-1 Baseline（round-trip のみ gate）
  - 0-1b REF-FIDELITY: ★ Shadow を §2.9 契約で実装してから比較（骨格のままでは実行不能）
  - 0-2 Candidate → 0-3 周波数（D1 は base gate / cand 2 条件記録）→ 0-4 block/reset → 0-5 SoftClip → 0-6 float/double
  - candidate マクロ 2 系統の一致性 assert を P0-H に追加（§2.9-4）

[Step 2] Phase 0 review（ユーザー GO gate）
  - P0-E は image invariance（base 側）として提示
  - GO → Phase 1 資格 / FAIL → 案D → tap再設計 → TPD型

[Step 3] §3 F-2（test-only・案 A・prep_patches §2）
  - :1799/:1803 を :1833/:1834 パターンへ / rigcheck=eq で staging=0 eqAGC=0 を確認
  - F-3/F-4 は記録のみ

[Step 4] §4 F-1（A→B→B'→C・failureReason 主判定）

[Step 5] §1 O-1〜O-17（ユーザー判断）
  - O-16（evidence）を先に決定（Step 0 の前提）
  - O-13 は 15 件、O-15 は §2.9 反映版、O-17 は O-9 と扱いを統一

[Step 6] B-1 Phase 1 以降（Step 2 GO 前提）
  - CMake option 適用（:40 群 + :1043 後/:1062 前・R10-5）→ flag ON 実測（引数なし実行・§3.4）→ Phase 3-A…4
  - 4 経路（RT/SoftClip × float/double）で P0-B/C/D/E を再確認

[Step 7] 別課題: R-2（parseOnOff 前例に揃える）→ R-1 → B-3, D-2, D-1
```

---

## 7. 検証計画（v2.4）

### 7.0 authoritative モデル再現

```bash
wsl.exe bash -lc 'cd /mnt/c/VSC_Project/ConvoPeq && python3 doc/work113/model_polyphase_20260920.py'
# 期待（v2.4 で実測確定）:
#   DC base 0.75^N / cand 1.0（12 桁）
#   full-chain dev: S1 0.45Fs +1.6350 / 0.49Fs +3.3927
#                   IIR3 +0.0121 / +0.0062   LP3 -0.0025 / +0.0025   （全桁再現）
#   ripple[0.005,0.30]: S1 0.001/0.001  IIR3 0.034/0.025  LP3 0.016/0.012
#   D1 base passband -9.5424 / D1 OLD AXIS 行が残存（解釈禁止）
```

### 7.1 v2.4 追加分（本書で新規作成した再現手順）

| ファイル | 目的 | 実行 |
|----------|------|------|
| `tmp/v24_ripple_band.py` | dev 全桁再現 + ripple バンド 4 条件（§2.8.1） | `wsl.exe bash -lc "cd /mnt/c/VSC_Project/ConvoPeq && python3 tmp/v24_ripple_band.py"` |
| `tmp/v24_d1_invariance.py` | D1 手法不変性 81 測定（§2.8.2） | 同上 |
| `tmp/v24_latency_probe.py` | 群遅延 peak 位置（R10-12 / P0-F 基線） | 同上 |
| `tmp/cio_literal.py` | C++ ループ逐語転写（R10-1 系統 3） | 同上 |
| `tmp/cio_octave.m` | Octave クロスチェック（系統 4） | `"C:/Program Files/GNU Octave/Octave-11.3.0/mingw64/bin/octave-cli" tmp/cio_octave.m` |
| `tmp/maxima_check.mac` | Maxima クロスチェック（系統 5） | `C:/maxima-5.50.0/bin/maxima.bat --very-quiet --batch-string '...'"` |

注: `tmp/` は `.gitignore` 済み（`.cline/`・`.workbuddy-ai/` と並びに O-8 の将来候補）。上記は evidence ではない。evidence 化するか废弃かは Phase 0 判断。

### 7.2 単体 / 統合 / 静的解析

| 項目 | コマンド | 期待 |
|------|----------|------|
| **B-1 attribution** | `build\Release\AudioEngineHarness.exe`（**引数なし**） | §2.4 の全 row（rt=0.75^N・EQ=1.000000・OF≈0.9874） |
| F-2 | `build\Release\AudioEngineHarness.exe --buzz-rigcheck=eq --buzz-dur=2.0` | `staging=0 eqAGC=0`（現状 `eqAGC=1`） |
| F-1 | `ctest --test-dir build --output-on-failure` | 移行先 PASS・PVIT 削除 |
| B-1 Phase 0/1 | 同上（flag ON/OFF rebuild） | OFF ≈ 0.75^N / ON ≈ 1.0 |
| Shadow/REF | Phase 0-1b/0-2 test-only | §2.9 の 3 対象で bitwise → Cand rt=1.0 |
| 静的解析 | `C:\Program Files\Cppcheck\cppcheck.exe --std=c++20 --enable=warning,performance,portability src/CustomInputOversampler.cpp` | 指摘 0（**v2.4 で再確認**・`TruePeakDetector.cpp` も 0） |
| clang-tidy | `compile_commands.json` から 1 変換単位ずつ（LLVM 23.1.1） | 単独 `--std=c++20` 実行は不可（JUCE 未解決・exit 1） |
| レイアウト/AST | serena `get_symbols_overview`（CIO 17 メソッド）/ graphify `query_graph` | graph は 2026-09-17 時点（要再生成） |

注: `PublishPipelineIntegrationTests.exe` は存在しない。ターゲットは **AudioEngineHarness**（41,137,664 bytes・`CMakeLists.txt:1899`）。

---

## 8. ロールバック

| 対象 | 方法 |
|------|------|
| §1 commit | `git revert <sha>` |
| §2 B-1 behavioral | **compile-time rollback**: `CONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF` で rebuild（runtime 即時復旧ではない） |
| §2 B-1 source | `#if` ブロック + CMake option を削除 + rebuild |
| §2.9 Shadow | test-only ファイル revert（production 影響なし） |
| §3 F-2 | 1 行 revert |
| §4 F-1 | 移行先を残す場合のみファイル復元 |
| §5 | 実装しないため不要 |
| 下準備成果 | doc + test-only のため production ロールバック不要（commit 済なら revert） |

---

## 9. 影響度まとめ

| 区分 | 影響 | 影響度 | 段階リリース |
|------|------|--------|--------------|
| O-1〜O-17 | リポジトリ/環境記録 | 極小（O-16 は証憑の信頼性に中） | 任意 |
| B-1 案 E | 全オーディオ経路（最大 +7.4963 dB・4 経路） | **極大** | 必須 |
| T-1 Shadow 実装 | test-only（ただし Phase 0 の成立条件） | 中（工数は複製分に比例） | 不要 |
| F-2 | test-only | 小 | 不要 |
| F-3 / F-4 | 記録のみ | なし | 不要 |
| F-1 | test-only | 中 | 不要 |
| R-2 等 | CLI/harness | 極小（前例 `parseOnOff` で均質化） | 不要 |

---

## 10. 監査ログ・参照

### 10.1 継承（v2.2 からの主要ソース確認）

| ファイル | 確認 | 結果 |
|----------|------|------|
| CustomInputOversampler.h | :21-22 / :32 reset / :41 prepareSingleStage / :71 clearAll / :87 centerCoeff=0.5 | ✓ |
| CustomInputOversampler.cpp | :84-106 tap/atten 表 / :543-564 / :557 conv×2・center ×2 無し / :658 / :392/:725/:785 / reset :452 / clearAll :469 | ✓ defect |
| AudioEngine.h | :971 softClipOS / staging setter :1416-1425 / default true :2626 | ✓ |
| BassBuzzMeasurement.cpp | :1293/:1300-1303/:1312-1313/:1341/:1352/:1361/:1367/:1376-1384/:1413/:1423/:1431-1435 / :1526-1534 / F-2 :1799/:1803 vs :1833/:1834 / :2042-2044 | ✓ |
| PublishPipelineIntegrationTests.cpp | :1116/:1123/:1217/:1223 / stoi :1136/:1146/**:1149**/:1157/:1166 | ✓ R8-9・§3.4 |
| CMakeLists.txt | option 群 :40 / NUC :52-58 / `add_subdirectory(JUCE)` :1043 / `juce_add_gui_app` :1062 / `CONVOPEQ_ALL_SOURCES` :1196（〜:1280）/ `target_sources` :1282 / harness :1899 / `add_executable` 39 | ✓ R10-5 |
| AudioEngine.Processing.DSPCoreDouble.cpp | :361/:505/:513/:541 | ✓（E-3 で正式パス） |
| AudioEngine.Processing.DSPCoreFloat.cpp | :263/:405/:413/:419 | ✓ |
| AudioEngine.Processing.DSPCoreLifecycle.cpp | :188/:261 `prepareSingleStage(31, 90.0, internalMaxBlock)` | ✓ |
| AudioEngine.Processing.Latency.cpp | :6-8 static_assert / :22-23 tap 表 / :29-32 stageRate・groupDelay / :119 要実測マーカー | ✓ → R10-12 で解消 |
| TruePeakDetector.cpp | prepareStage :192-281（sum→1.0・center 0.5）/ interpolateStage :284-311（**×2 なし**） | ✓ 型は別物（unity 側） |

### 10.2 v2.4 追加

| ファイル / 成果物 | 確認 | 結果 |
|----------|------|------|
| model_polyphase_20260920.py :189-190 | dev 定義 `cand − base − 20log10((4/3)^N)`・`freq_pts` は絶対 Hz・`spectrum()` は impulse DTFT | ✓ R10-7 確定 |
| model_polyphase_20260920_results.txt :17-53 | §2.7.1 の全 24 行 | ✓ 全桁再現 |
| 〃 :191（`for lo, hi in ((0.005,0.45),(0.01,0.30))`） | ripple バンドは (0.01,0.30) で算出 | ✗ E-8（ラベル是正・実質差なし） |
| PolyphaseGainCandidateRef.h 全文 102 行 | prepare/reset/centerPhaseGain の実体 | ✗ R10-6（DSP 本体なし） |
| tmp/osdirect_run.txt（248,346 B）/ osdirect2/3 | `[OS_DIRECT]/[EQ_DIRECT]/[OF_DIRECT]` 実 log | ✓ R10-2 |
| git `diff --numstat HEAD` / `ls-files --others` | 10 tracked M/D + 20 untracked | ✗ E-1/E-4/E-5/E-6 |
| `file evidence/epoch_reclaim_audit.json` | `data`（非テキスト） | ✗ O-16 |
| compile_commands.json | CIO.cpp = 6 エントリ / 2 論理ターゲット | ✓ R10-5 |

### 10.3 v2.4 保存後セルフ検証（同一セッション・機械実施）

| 手段 | 検証範囲 | 結果 |
|------|------|------|
| `tmp/vcheck_v24.py`（full-path 引用 + 意味検査 23 件） | 12 引用 / 23 assertion | 22 pass・範囲外引用 0・ファイル行数（CIO 872 / REF 102 / CMake 2044 / PVIT 519）および option 8 / `add_executable` 39 / TEST 38 を再現 |
| `tmp/vcheck_v24_ctx.py`（文脈推定で `:NNN` 全件バウンズ） | 73 行中の全行番号参照 | out-of-range 報告 8 件は**すべて同一行で CMakeLists 参照を CIO.cpp と誤適用したスクリプト側の所産**（文書側の誤り 0）。エイリアス解決不能 28 行は目視確認 |
| `tmp/vcheck_v24_rows.py`（CIO/BBM/PVIT/model/Latency の 47 位置を内容照合） | 引用範囲の端点・識別子 | CIO 全 span（:492/:549/:568/:570/:618/:620/:649/:725/:783/:785/:452/:467/:469/:484/:392）、BBM fail-closed :1708-1710・:1745・:1763・:1778、PVIT 呼出 9 箇所、model :188-191（`(0.005,0.45),(0.01,0.30)` まで一致）、Latency :22-23/:29-32 を確認 |

**発見 2 件（いずれも本節で是正済）**:
1. **引用ドリフト**: `dotProductAvx2` の開始行は `:159`（`:160` は第 2 引数）。§2.2 を `:159-216` に修正。関数内の下位引用 `:175-178`（prefetch guard）・`:194-199`（vextractf128+hadd 削減）・`:202-204`（`_mm256_zeroupper`）は端点まで一致。
2. **環境主張の過剰**: 「`ctx_batch_execute` では `/c/...` と `&&` が壊れる」→ **再測で `&&` は PS7 で動作**と確認。失敗するのは `/c/...` 形式のみ（`C:\c\...` に解決）。`wsl.exe bash -c` 経由の rtk はサンドボックス内から成功（HEAD SHA `8f127bfe` 返却）。§12.2・§14 を実測通り修正。

**検証していないもの**: 本セルフ検証は**引用の正確性**を対象とする。§2.7.1 dev 値・D1 不変性 81 測定・latency 4 条件はそれぞれ `tmp/v24_ripple_band.py` / `tmp/v24_d1_invariance.py` / `tmp/v24_latency_probe.py` の実行結果から再現した値であり、production 側の未実行（Phase 1+ HOLD）のため**ビルド・実機での再確認は未実施**。

---

## 11. 判断待ち項目（v2.4）

### 11.1 ユーザー意思決定（技術調査の対象外・v2.3 から継続）

| 項目 | 選択肢 | 推奨 |
|------|--------|------|
| O-1 計装 | A 最小 / B 全 / C 退役 | **A** |
| O-2 台帳 | 同梱 / 独立 | **独立** |
| O-6 push | 即時 / 別 / 手動 | **ユーザー手動** |
| O-8 pyc | 復元 / 触らない / 別 work item | **触らない→将来 work item** |
| O-9 doc/tool-inventory | 触らない / commit | **触らない** |
| O-10 AGENTS.md | 触らない / commit | **commit（ただし §14 の Maxima 誤記を同時に是正）** |
| O-11 台帳 | 触らない / commit | **commit** |
| O-12 ConvoPeq.md | 触らない | **触らない（再生成資産）** |
| O-13 work113 md/py 15 件 | 承認版+台帳 / 全 commit / 触らない | **承認版+台帳+モデル（py/results）を 1 commit** |
| O-14 .mcp.json | 触らない / commit | **commit（環境記録）** |
| O-15 Shadow | 同梱 / 別 / 触らない | **§2.9 反映後に O-13 同梱** |
| **O-16 evidence/epoch_reclaim_audit.json（新）** | 追跡外化 / 内容特定後台帳化 / 破棄 | **追跡外化 + 測定出力は `tmp/` へ**（§1.2） |
| **O-17 doc/WorkBuddy_AI_Desktop_Toolchain.md（新）** | 触らない / O-9 と同種扱い | **O-9 と同一方針に揃える** |
| G-0 契約 | 承認 / 修正 / 却下 | **承認（v2.1 版 E1〜E5）** |
| B-1 修正案 | E（candidate） | **E（仮説）** |
| Phase 0 | GO / HOLD | **GO（T-1・T-2 の承認が条件）** |
| F-2 | A 呼出順逆 | **A** |
| F-1 | 全実施 | **全実施** |
| Phase 3 commit | 分割 / 単一 | **分割** |

### 11.2 残存技術判断（v2.4 で確定案提示・要承認）— 2 件

v2.3 は「技術的未確定 0」と主張したが、以下 2 件は未確定だった（E-7）。**本書が確定案を提示したので、ユーザーは承認するだけでよい。**

| ID | 論点 | v2.4 確定案 | 根拠 |
|----|------|-------------|------|
| **T-1** | REF-FIDELITY gate の実行条件 | §2.9 の 5 項目契約（係数生成一致 / 履歴・分岐順序一致 / 3 対象 bitwise / candidate マクロ 2 系統の一致性 assert / shadow cand は予測にすぎない）。骨格のままでは実行不能 | R10-6・§2.9 実態表 |
| **T-2** | cand D1 の gate 可否 | **gate しない**。gate は base `−9.542425 ± 0.01 dB`（手法不変）に移動し、cand は矩形窓と Hann+trim8 の 2 条件を記録。alias rejection は P0-D に一本化 | R10-4・§2.8.2（81 測定で base ≤0.0036 dB / cand 92 dB 振幅） |

### 11.3 B-1 採用案の前提条件（v2.4）

1. DESIGN-CONTRACT-A + **T-1・T-2** が明示承認（G-0）
2. REF-FIDELITY bitwise（§2.9 契約・`reset()` 経由）
3. round-trip DC = 1.0 ± 1e-6（必要条件）
4. P0-B/C passband 維持（4 経路）
5. P0-D stopband 維持
6. **P0-E-1 base D1 = −9.5424 ± 0.01 dB**（E-2 D2 [0.005,0.40] ≤0.01 dB）
7. P0-F latency 不変（base peak == cand peak・§2.8 表の基線）
8. P0-I SoftClip 安全性 gate
9. P0-C' differential が center-phase ×2 に帰属
10. D1 測定は補正軸 + expected bin 主報告 + argmax 検証ログ（R9-1/R9-2/R10-4）

### 11.4 v2.4 最終判定表

| 項目 | 判定 | v2.3 から |
|------|------|-----------|
| §1 O-1〜O-17 | **GO 候補（O-16/O-17 を計上・完全棚卸し）** | 拡充 |
| §3 F-2 | **GO 候補（案 A・実測で原因確定）** | 強化 |
| §3 F-3 / F-4 | **記録継続** | 変更なし |
| §4 F-1 | **GO 候補（failureReason 主判定・数値再確認）** | 変更なし |
| §5 R-2 | **仕様確定 + 前例 `parseOnOff` で均質化可能** | 強化 |
| B-1 原因分析 | **GO（5 系統独立再現・R10-1）** | 強化 |
| B-1 Phase 0 | **GO（ただし T-1・T-2 承認が前提）** | 条件追加 |
| B-1 Phase 1+ | **HOLD** | 変更なし |
| B-1 案 E | **有力仮説（確定表記禁止）** | 変更なし |
| §2.7.1 authoritative 値 | **完全再現（R10-7）** | 新規確定 |
| D1 測定仕様 | **base gate / cand record に改定（R10-4）** | 改訂 |
| Shadow | **骨格のみ・gate 実行不能（R10-6）** | **是正** |
| 静的解析 | cppcheck 指摘 0 再現（R10-11） | 確認 |
| 技術的未確定 | **2 件（T-1・T-2）・確定案提示済み** | **是正（E-7）** |
| TruePeakDetector | 後続候補第三順位 | 変更なし |
| compile-time flag | `CONVOPEQ_CORRECT_POLYPHASE_GAIN` + `#if`（配置は R10-5 で妥当） | 確認 |

---

## 12. エビデンス（v2.3 からの追加分）

### 12.1 5 系統クロスチェック（R10-1）

production バイナリ / authoritative モデル / `tmp/cio_literal.py`（NumPy 逐語転写）/ `tmp/cio_octave.m`（Octave 11.3.0）/ `tmp/maxima_check.mac`（Maxima 5.50.0）。全系統が base 0.75 / 0.5625 / **0.421875**、cand **1.0**、Δ **+2.498775 dB**、image/tone **1/3 = −9.542425 dB**、FIRsum 1.0 / center 0.5 / convSum 0.5（6 design すべてで）に一致。

### 12.2 ツール実測（R10-11/R10-13）

| ツール | 形 | 結果 |
|--------|----|------|
| cppcheck 2.21.0 | `C:\Program Files\Cppcheck\cppcheck.exe`（PATH 外） | CIO.cpp / TPD.cpp 指摘 0（c++20） |
| clang-tidy 23.1.1 | `C:\Program Files\LLVM\bin\clang-tidy.exe`（PATH 外） | 単独実行 exit 1（JUCE 未解決）→ compile_commands 必須 |
| ast-grep 0.44.0 | WSL `/usr/local/bin/ast-grep` | `convValue *= 2.0;` 1 hit / `centerValue *= 2.0;` 0 hit |
| ag 2.2.0 / fdfind 10.3.0 / rg 15.1.0 / sed / awk / fzf | WSL（`fd` は bare 名で存在せず `fdfind`） | 稼働 |
| tgrep 1.0.4 | `C:\Windows\System32\tgrep`（**trigram grep**・AST ではない） | 同上（regex は PCRE 風・`(` 単体はエラー） |
| graphify 0.9.64 | `%APPDATA%\Python\Python314\Scripts\graphify.exe` | graph.json 74,421 nodes・**2026-09-17 時点で stale** |
| ccc（cocoindex） | `C:\Users\user\.local\bin\ccc.exe` | search 稼働（意味検索 hit は doc 側） |
| semble | `C:\Users\user\.local\bin\semble.exe`（MCP 未登録・CLI のみ） | 稼働。`evidence/**/impl-*.ninja` 6 ファイルを size 上限で skip と警告 |
| AiDex / serena MCP | `aidex_query` / `get_symbols_overview` | centerValue 5 hit（:543/:545/:549/:559/:564）/ CIO 17 メソッド |
| NumPy 2.5.2・SciPy 1.18.0・pandas 3.0.5 / trafilatura 2.2.0 / crawl4ai | `C:\Python314\python.exe` | 稼働（`import crawl4ai` OK・`__version__` は submodule 扱い） |
| Octave 11.3.0 | PATH 上 | 稼働（`idivide` 非対応 → `floor(a/b)` に置換して実行） |
| **Maxima 5.50.0** | `C:\maxima-5.50.0\bin\maxima.bat --very-quiet --batch-string` | **稼働（AGENTS.md 注記 8「未インストール」は誤り）** |
| obscura 0.2.2 | `C:\Users\user\.local\bin\obscura\obscura.exe` | 稼働確認（本次元の取得は firecrawl/ddgs/brave で充足） |
| Dr. Memory | — | **計測不能**（rc=127・侵入的セキュリティ干渉。AGENTS.md 注記 7 と一致） |
| context7 MCP | `query-docs` | `libraryId` 必須（resolve 無しでは使えない）。本次元はライブラリ API ではなく数学規約 → 不使用 |
| MS Learn MCP | `microsoft_docs_search` | AVX2 `_mm256_hadd_pd` / IEEE denormal 規定を取得（§2.2 SIMD 注記の裏取り） |
| firecrawl / ddgs / brave MCP | scrape / search_text / web_search | DSPRelated（Lyons/Wescott/dgshaw6 原引用）・MathWorks・音響関連日本語資料 |
| **context-mode `ctx_batch_execute`** | Qoder では**サンドボックスが PowerShell 7.6.6** | `/c/...` パスが `C:\c\...` に解決され失敗（`&&` は PS7 で動作・当初記録は是正済）。**AGENTS.md の「MSYS2」記述は Qoder に適用不可**（詳細 → §14） |
| headroom proxy | `headroom savings` | Today 0/0・Last 30d 985/5,808（17%）。**Qoder トラフィックは非圧縮**（`ANTHROPIC_BASE_URL` 非経由） |

### 12.3 未確定事項の棚卸し（v2.4 最終）

| 種別 | 状態 |
|------|------|
| R8-1 / R9-1 モデル D1 | 技術側完了（v2.3）+ **v2.4 で gate 再定義（R10-4）** |
| §2.7.1 dev 値の疑義 | **解消（R10-7・v2.3 が正しい・前回疑義を撤回）** |
| R9-5 cand D1 差 0.94 dB | **解消（手法感度帯内のノイズ。物理量でない）— R10-4** |
| Latency.cpp:119「要実測確認」 | **解消（R10-12・モデルで dev 0.0000/+0.75 を実測。±1 許容が必須と確定）** |
| production 追加欠陥 | **検出なし**（cppcheck 0 / 5 手段で ×2 不在確認 / TPD は別型） |
| Shadow gate 実行条件 | **T-1 として確定案提示（要承認）** |
| evidence/ 証憑の信頼性 | **O-16 として計上（要判断）** |
| ユーザー意思決定 | O-1〜O-17 / G-0 / Phase 0 GO / F 方針 / T-1 / T-2 |

---

## 13. v2.4 で追加した確定事項（R10 一覧）

- **R10-1** defect を 5 系統独立で再現（0.75^N / 1.0 / +2.4988 dB / −9.5424 dB）。設計点 6 構成も `tapsForStage`/`attenuationForStage`（:84-106）で確定
- **R10-2** ライブ帰因表（`[OS_DIRECT]`/`[EQ_DIRECT]`/`[OF_DIRECT]`）。EQ 無損失・OF −0.110 dB・OS のみ 0.75^N
- **R10-3** up/down 分割は peak ベースかつ段数依存 → 不変量は round-trip、因果は分岐係数和。gate を round-trip に一本化
- **R10-4** D1 base は 81 測定で手法不変（≤0.0036 dB）→ gate 可。cand は 92 dB 振幅 → 記録のみ。R9-5 を解消
- **R10-5** CMake 適用対象は 2 論理ターゲット（6 エントリ）。:1043 後配置で両者に届く。ODR リスクなし
- **R10-6** Shadow は DSP 本体を持たない骨格 → REF-FIDELITY は現状実行不能。§2.9 で実装契約を確定
- **R10-7** v2.3 §2.7.1 の dev 6 値を定義特定のうえ全桁再現（疑義を撤回）
- **R10-8** ripple 基線を [0.005,0.30] で実測確定（[0.01,0.30] と 3 桁一致）+ [0.005,0.40] を新規取得
- **R10-9** F-2 をライブ実測（`staging=0 eqAGC=1`）に格上げ
- **R10-10** F-1 棚卸し数値（519 行 / TEST 38 / 9 呼出 / 未登録 / 39 targets）を再確認
- **R10-11** cppcheck 指摘 0 を再現。clang-tidy は compile_commands 必須
- **R10-12** 群遅延を実測（S1 15 / 単段 255 / IIR3 291 / LP3 583）→ P0-F ±1 が必須。base==cand
- **R10-13** 環境実態を是正（Maxima 稼働 / graphify・tgrep の実体 / headroom proxy 無効 / context-mode PowerShell）

**是正**: E-1 AGENTS.md diffstat / E-2 doc パス / E-3 正式パス / E-4 untracked 範囲 / E-5 O-16 新規 / E-6 O-17 新規 / E-7「技術的未確定 0」撤回 / E-8 up/down 記述 / E-9 cand D1 レンジ記載削除。
**教訓 29〜33** を §2.10 に追加。

---

## 14. 付録 — トークン削減 3 層パイプラインの本次元実態（AGENTS.md との差異）

AGENTS.md は「headroom proxy + context-mode MCP + rtk(WSL) の 3 層を常時」を要求するが、**Qoder デスクトップ環境では成立しない**（実測で報告する。使えたかのように書かない）。

| 層 | 本次元での実態 |
|----|----------------|
| headroom **proxy** | **無効**（Qoder は `ANTHROPIC_BASE_URL` 非経由。`savings` Today 0/0 を実測） |
| headroom **MCP** | 接続済（compress/retrieve/stats）。本作業では large-output 回避を context-mode で行ったため未使用 |
| **context-mode MCP** | **主軸**。`ctx_batch_execute` のサンドボックスは **PowerShell 7.6.6**（一時 `.ctx-mode-XXXX\script.ps1` を実行・既定 cwd は Qoder アプリ配下なので `Set-Location` 必須）。**`/c/...` 形式のみ失敗**（`C:\c\...` に解決され「パス…存在しない」）。`&&` / `;` / `$(...)` は**動作する**（PS7 のため。当初記録の「`&&` 不可」は誤り・本次元再測で是正）。`C:/...` と `'C:\...'` は両方正解。`wsl.exe bash -c '...'` 経由の WSL＋rtk もサンドボックス内から動作を実測（HEAD SHA 返却）→ AGENTS.md の MSYS2 記述は ZCode/commandcode 向けで Qoder に適用不可 |
| **rtk (WSL)** | **CLI 圧縮に使用**（`rtk rg` / `rtk ag` / `rtk fdfind`）。ただし `wsl bash -c '...'` 内の変数展開が外側 MSYS で二重展開される事例あり → 変数を使わないリテラル記述で回避 |

**実効構成**: context-mode MCP（主）→ rtk(WSL)（CLI）→ 素の `wsl.exe bash -lc` / Git Bash（フォールバック）。
**改善提案（D-2 に関連）**: Qoder では proxy 前提の統合設計は成立しない。`ctx_batch_execute` の PowerShell 前提を AGENTS.md に追記し、WSL 呼び出しはリテラル表記を規約化する。

---

*本書は v2.3（`remediation_plan_20260922_v2.3_revised.md`）の再改訂版 v2.4 である。*
*v2.3 本文は参照文書として維持し、**R10-1〜R10-13 および E-1〜E-9 が v2.3 と矛盾する箇所では本書を優先する**。*
*R5〜R9 は本書でも有効（上記是正点を除く）。中間エビデンスは `remediation_v22_intermediate_20260921.md`、パッチ案は `remediation_v22_prep_patches_20260922.md`（production 未適用）。*
**v2.3 継承の監査・確定事項**: R9-1〜R9-6 / R8-1〜R8-12 / R7-1〜R7-4 / R6-1〜R6-7 / R5-1〜R5-12。
