# Part 8 修正の実行スコープ案（承認待ち）

- 作成: 2026-11-09 11:30
- 基準 HEAD: `2aecd73e823067e8cf26c2d4a8fa2e4e6c6d3166`（main、working tree クリーン）
- 位置づけ: Owner 決定（2026-11-09）の「Implementation, commit, PR, and merge remain unauthorized until this decision is explicitly accepted and the applicable execution scope is confirmed.」に対し、**実行スコープを確定するための提案**である
- コード変更: **未実施**。本報告書の作成のみ

## §1 承認状況の整理

| 項目 | Owner 決定 | 本報告での扱い |
|---|---|---|
| A stderr | APPROVE FIX | 実装設計と受入基準を提示 |
| B 最小位相 | MAINTAIN | 変更しない |
| C 群遅延クランプ | APPROVE CURRENT BEHAVIOR | 変更しない |
| D README | MAINTAIN | 変更しない |
| E ポリフェーズ | APPROVE（回帰試験と互換性の明示が条件） | 実装設計・テスト更新・互換性記載を提示 |
| F Bypass | APPROVE（文書化のみの受容は不可） | 実装設計・受入基準を提示 |
| G CI | APPROVE | ワークフロー設計と受入基準を提示 |

実装・commit・PR・merge は、このスコープが承認されるまで行わない。

## §2 変更対象ファイル一覧

### A — stderr バックプレッシャの解消

| ファイル | 変更 |
|---|---|
| `src/convolver/ConvolverProcessor.LoaderThread.cpp` | `fprintf(stderr, ...)` 7 箇所（505/540/691/771/791/813/894 行）と `printChainCheckpoint` 本体（501 行〜）を診断マクロで囲む |
| `CMakeLists.txt` | 新オプション `CONVOPEQ_ENABLE_LOAD_CHAIN_DIAGNOSTICS`（既定 OFF）を追加し、ConvoPeq / AudioEngineHarness / 既存テストターゲットへ define を渡す（既存 `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` の定義・define 群と同じ記法。CMakeLists 129 行の option と 1676 行以降の define 群を参照） |

- 既存の `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` に相乗りさせない理由: 同マクロは MKL アロケータ計測など RT 側の計測も有効化する（`src/MKLNonUniformConvolver.cpp` で 29 箇所）。読み込み調査のために RT 計測条件を変えたくない
- 文字列自体は残すため、マクロ ON のビルドでは従来と同じ 16.0 行/ロードが出る
- テスト・スクリプトからの文字列参照は無い（`IR_CHAIN` / `IR_TAIL_ENV` / `IR_RATE_GEN` を検索して該当なし）

### E — ポリフェーズ利得補正の既定化

| ファイル | 変更 |
|---|---|
| `CMakeLists.txt:134` | `option(CONVOPEQ_CORRECT_POLYPHASE_GAIN ... OFF)` → `ON`。無効化したい場合の逃げ道として OFF 指定は維持 |
| `src/tests/AudioEngineHarness/PolyphaseGainFidelityTests.cpp` | production 期待値 0.75 のゲート 7 箇所（496 / 759–763 / 975 / 1052 / 1122–1129 / 1159 行）を 1.0 基準へ更新。`emitBuildIdBlock` の `production_flag` 表示（107–112 行）を「ON が正」に変更 |
| `src/tools/build_identity_gate.py:446–456` | CMakeCache の値を表示する文言（`OFF (Phase 1: default OFF)` / `ON (Phase 1: 明示指定 — characterization 用)`）を新しい既定に合わせる |
| `ARCHITECTURE.md`、`BUILD_GUIDE_WINDOWS.md` | 既定値と意図の記述を更新 |
| `manual/`（該当節）とリリースノート | 互換性の明示（§6）。Owner 条件「explicit documentation of preset compatibility changes」に対応 |
| `src/tests/AudioEngineHarness/P1PolyphaseGainCharacterization.cpp` | 原則変更なし。`failures` は `runCase`（publish 確認）の失敗で、利得期待値のゲートは無い。ON 既定での `gainDb` を記録し直す |

### F — Bypass の IR 完全性

| ファイル | 変更 |
|---|---|
| `src/MKLNonUniformConvolver.cpp` | 667 行（`tailEnabled` 導出）、680–700 行（層ゲイン）、718–719 行（`m_tailEnabled` / `m_tailStrength`）、777–823 行（`l1Part` / 層長 / 幾何の不変条件）、1899 行（Get のゲイン適用） |
| `src/MKLNonUniformConvolver.h` | 変更不要（試作で確認: `m_tailEnabled` を常に true とし、Bypass は層ゲイン 1.0 で表現できる。新規フラグは不要） |
| `src/tests/MT-NUPC-Measurement.cpp` | `computeLayerCfg` の tail 無効時の理論（l1Len/l2Len=0、L0=32 分割）を新幾何へ更新。T8（irLen=5000・tail 無効）と M4 の bypass 行の期待値、および block 依存の期待値を更新 |
| 新規テスト | Bypass の IR 完全性（block 256〜4096）と、既定モードのテール（H）の回帰を ctest に追加（§4） |

### G — CI での compile DB 生成

| ファイル | 変更 |
|---|---|
| `.github/workflows/isr-verification.yml` | oneAPI（MKL+IPP）の復元、`cmake -S . -B build -G "Ninja Multi-Config"` の configure、生成ヘッダ `JuceHeader.h` の用意、`build/compile_commands.json` の存在検査、監査の実行 |
| `CMakeLists.txt`（任意） | 生成ヘッダ専用のカスタムターゲット（`GenerateJuceHeader`）を追加し、CI から安定した名前で呼べるようにする |
| 任意: `.github/actions/oneapi-setup/action.yml` | `h2r1-release-measurement.yml` の winget 導入＋キャッシュを共通化（重複解消。今回はコピーでも可） |

- 監査スクリプト本体は変更不要（`build/compile_commands.json` を既に見ている）
- 失敗時の扱いは現状どおり fail-closed（DB が無ければ失敗）を維持する

## §3 実装設計

### A. stderr のブロック解消

```
#if CONVOPEQ_ENABLE_LOAD_CHAIN_DIAGNOSTICS
static void printChainCheckpoint(...) { ... }
#else
static void printChainCheckpoint(...) { juce::ignoreUnused(tag, buf, sr); }
#endif
```

各 `fprintf(stderr, ...)` を同じマクロで囲む。`printChainCheckpoint` の呼び出し 5 箇所は残し、本体を空実装にすることで呼び出し側の条件分岐を増やさない。

- 受入: 既定ビルドで `AudioEngineHarness --n1-analysis` の stderr が 0 行、`[N1-analysis] ALL PASS`、exit 0
- 受入: 未ドレインのパイプ（`stderr_block_test2.py`）で 90 秒停止が解消し、10 秒以内に終了
- 受入: マクロ ON ビルドで従来どおり 16.0 行/ロード（証拠の再取得手段を残す）

### E. 補正の既定化

CMake の既定を ON にし、テストの期待値を「補正後 = 1.0」へ移す。Phase 0 の shadow candidate 比較は残し、production と candidate が一致することを確認する二重系に組み替える。

- 受入: `PolyphaseGainMetrics`（ON 既定）で全構成 DC 1.000000000、帯域 0.9993〜1.0009
- 受入: `PolyphaseGainFidelityTests` が全 PASS、`ctest` が 54/54
- リスク: 出力レベルが 2×/+2.50 dB、4×/+5.00 dB、8×/+7.50 dB 上昇する。ON ビルドの実測では peakRatio が 1.03〜1.67 に上がっており、リミッタへの接触が早まる可能性がある。回帰ではピークとリミッタ作動の変化を記録する

### F. Bypass の IR 完全性

案 1（採用案・試作で検証済み）: Bypass でも L1/L2 を確保し、ゲイン 1.0・整形なしで IR 全体を畳み込む。あわせて層の幾何に不変条件を入れる。

```
// (1) 被覆: tail 無効でも L1/L2 を確保する
const bool fullCoverage = tailEnabled || (tailMode == 2);
l0Len = std::min(irLen, fullCoverage ? l0LenTarget : l0MaxLen);
l1Len = fullCoverage ? std::max(0, std::min(irLen - l0Len, kL1MaxParts * l1Part)) : 0;
l2Len = fullCoverage ? std::max(0, irLen - l0Len - l1Len) : 0;

// (2) ゲイン: Bypass は等倍（整形しない）
if (tailMode == 2) { layer1Gain = 1.0; layer2Gain = 1.0; }

// (3) 幾何の不変条件: L1 のパート長が L0 保持長を超えないこと
//     超える場合は倍率を下げ、必要な場合は L0 保持長を l1Part まで広げる
mult = largestPow2AtMost(min(tailL1L2Mult, l0CoverageAligned / l0Part));  // 下限 2
l1Part = l0Part * mult;
l0Len = std::min(irLen, std::max(l0LenTarget, l1Part));
```

Get 側は `layerGain = m_tailEnabled ? m_tailLayerGain[li] : 0.0` のままでよい（層を常に確保するため `m_tailEnabled` は常に true）。新規フラグは不要で、変更は SetImpulse の被覆・ゲイン・幾何に閉じる。Air Absorption の減衰は `tailEnabled && tailMode == 0` で既に除外されるため、Bypass には掛からない。

**試作による実測**（リポジトリ外のコピーに上記を適用。production は無変更）

| block | tailMode=2 修正前（現行） | tailMode=2 試作 v5 | tailMode=1 修正前 | tailMode=1 試作 v5 |
|---|---|---|---|---|
| 256 | maxErrFull 2.499e-01（0.1707 秒で切断） | **3.331e-16** | 1.093e-01 | 1.093e-01 |
| 512 | maxErrFull 2.499e-01（0.3413 秒で切断） | **3.886e-16** | 1.093e-01 | 1.093e-01 |
| 1024 | maxErrFull 2.499e-01（0.6827 秒で切断） | **3.331e-16** | 2.499e-01（末尾 896 サンプル欠落） | 1.093e-01 |
| 2048 | — | **3.053e-16** | — | 1.093e-01 |
| 4096 | — | **2.776e-16** | — | 1.093e-01 |

- 「修正前」は現行コードの実測（切断長は §F の表）。「試作 v5」は (1)(2)(3) を適用したコピーの実測
- tailMode=2 の誤差 3e-16 は倍精度の丸め誤差であり、IR 全体が厳密に再現されている
- tailMode=1 の 1.093e-01 はテール整形（意図した音作り）による差で、ブロック長に依存しない

**追加発見（H）: 既定モードのブロック長依存**

幾何の不変条件（`l1Part ≤ L0 保持長`）が満たされない block 1024 では、**既定の tailMode=1 でもテールが壊れる**。実測では末尾 896 サンプルが無音になり、テールのエネルギーが 22.28 dB → 11.71 dB に落ちた（ブロック長 512 との比較）。48 kHz では `l1Part = 8 × block` が 0.12 秒の L0 保持長を超える block ≥ 1024 で発生する。F の幾何修正はこの不変条件を満たすため、既定モードのこの欠落も同時に解消する（試作で 1024 のテールが 22.28 dB に戻ることを確認）。

案 2（不採用）: L0 のパーティション数を IR 全体まで増やす。`kL0PartsHardCap = 256` の範囲でも 512 サンプル区間 × 256 = 131,072 サンプル（48 kHz で 2.7 秒）までしか届かず、IR 全体（最大 2^21）を L0 単独で覆うと計算量が IR 長に比例して増える。

- 受入: 測定アプリで tailMode=2 の `maxErrFull` が block 256/512/1024/2048/4096 のすべてで 1e-10 未満
- 受入: 保持長以降の無音が解消し、IR 末尾 100 ms（−11.9 dB）のエネルギーが出力に現れる
- 受入: block 1024 の tailMode=1 でテールエネルギーが 22 dB 台に戻る（H の回帰）
- 受入: MT-NUPC の T8 / M4 bypass 期待値を新幾何へ更新し、`[NG]=0`
- 受入: N1 解析（`--n1-analysis`）が ALL PASS、`ctest` 54/54
- 影響: Bypass のリソースは tailMode=1 と同程度になる（従来は 1 層）。既定の tailMode=1 は block ≥ 1024 でテール欠落が解消する（音が変わるのは不具合が直る方向）
- 未確定: block 8192 以上（`l1Part` の下限 2 倍でも L0 保持長を超える領域）の扱い。L0 保持長を広げるか、対応ブロック長の上限を定めるかは §8 で確認する

### G. CI の compile DB 生成

1. `h2r1-release-measurement.yml` と同じ手順で oneAPI（MKL+IPP）を復元（キャッシュキーは同形式、`mkl.h` / `ipp.h` / `mkl_core.lib` / `ippcore.lib` の実在で判定）
2. `CONVO_CI_BUILD=1` と `CMAKE_PREFIX_PATH`（mkl / ipp）を設定し、`cmake -S . -B build -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCONVOPEQ_REQUIRE_MKL=ON` を実行（アプリのビルドはしない）
3. **生成ヘッダを用意する**。`ninja -C build -f build-Release.ninja ConvoPeq_artefacts/JuceLibraryCode/JuceHeader.h`（実測で数秒）。CMake にカスタムターゲットを足す場合は `cmake --build build --config Release --target GenerateJuceHeader`
4. `build/compile_commands.json` の存在を検査し、無ければ fail
5. `isr-run-tiered-verification.ps1` に `-RequireClangTidy` を渡す既存経路で監査を実行

- ローカル実測（2026-11-09）: クリーン configure だけでは `[ERROR] clang-tidy invocation failed (exit=1)`（`JuceHeader.h` 未生成、`AudioEngine.h:41`）で失敗し、ヘッダ生成後に `[PASS] clang-tidy audit gate verified`（exit 0）となった。configure だけでは不足する
- 受入: 週次 exhaustive が success、ログに `[PASS] clang-tidy audit gate verified`
- 受入: DB を消した状態では従来どおり `[ERROR] compile_commands.json missing` で fail（fail-closed の維持）
- 受入: クリーンな `build` ディレクトリからの実行で PASS（キャッシュされた DB に依存しない）
- リスク: oneAPI の導入に数分かかる（キャッシュで 2 回目以降は短縮）。導入失敗時は監査を skip せず fail させる

## §4 回帰テスト計画

| 対象 | 実行 | 合否基準 |
|---|---|---|
| A 既定 | `AudioEngineHarness.exe --n1-analysis <dir> post` | stderr 0 行、ALL PASS、exit 0 |
| A 未ドレイン | `stderr_block_test2.py` | 10 秒以内に終了（停止しない） |
| A 証拠再取得 | マクロ ON ビルドで同上 | 16.0 行/ロードを再現 |
| E | `PolyphaseGainMetrics.exe`（OFF/ON 双方） | ON 既定で DC 1.000000000、帯域 0.9993〜1.0009 |
| E | `PolyphaseGainFidelityTests.exe` | 全 PASS（更新後の期待値） |
| E 互換 | 既存プロジェクトの代表プリセットを読み込み、出力ピークとリミッタ作動を記録 | レベル上昇が想定（+2.5〜+7.5 dB）内で、異常な clip が無い |
| F 完全性 | 測定アプリの ctest 化（新規ターゲット） | tailMode=2 の maxErrFull < 1e-10（block 256/512/1024/2048/4096） |
| H 回帰 | 同上（tailMode=1 で測定） | block 1024 のテールエネルギーが 22 dB 台（現行は 11.71 dB） |
| F 幾何 | `MTNUPCMeasurement.exe` | T8 / M4 更新後 `[NG]=0`、`structural failures = 0` |
| F | `AudioEngineHarness.exe --n1-analysis` | ALL PASS |
| G | `pwsh .github/scripts/isr-verify-clang-tidy-audit.ps1 -Tier exhaustive -RequireClangTidy`（DB＋ヘッダあり／DB なし／DB のみ） | あり PASS、なし fail、DB のみ fail（`JuceHeader.h` 不足） |
| 全体 | `ctest` | 54/54 を維持（新規追加分を除く） |

F の完全性テストは、リポジトリ外で使った測定アプリのロジックを ctest ターゲットへ移植する（production の `MKLNonUniformConvolver` をリンクし、IR 1.0 秒・block 256/512/1024/2048/4096 で `maxErrFull` を判定）。移植しない場合は測定アプリの実行結果を証拠として残す。

## §5 互換性の記載（E）

記載が必要な内容:

- オーバーサンプリング時のラウンドトリップ利得が 1.0 になる（従来は 2× = −2.50 dB、4× = −5.00 dB、8× = −7.50 dB）
- 既存プロジェクトの音は同じ設定でも 2.5〜7.5 dB 大きくなる。プリセットの数値は変わらない
- 自動ゲインステージングやリミッタの作動点が変わるため、必要に応じて出力メイクアップの再調整を案内する
- 旧挙動が必要な場合は `-DCONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF` でビルドできる（サポート外の逃げ道として明記するかは要判断）

## §6 実装順序と検証ゲート

1. **A**（独立・低リスク）→ 既定ビルドで stderr 0 行と ALL PASS を確認
2. **G**（CI）→ ローカルで DB あり／なしの両方を再確認してからワークフローへ反映
3. **E**（音声影響・互換性あり）→ テスト期待値の更新と `PolyphaseGainMetrics` の ON 実測を同時に実施
4. **F**（音声影響・幾何変更）→ 測定アプリで完全性を確認してからテスト契約を更新
5. 最終確認: `ctest` 54/54、`--n1-analysis` ALL PASS、`MTNUPCMeasurement` `[NG]=0`、監査 PASS

各段階で commit を分けるかは Owner の判断。推奨は「A+G（診断・CI）」と「E+F（音声）」の 2 本に分ける構成（切り分けとロールバックが容易）。

## §7 リスクとロールバック

| 項目 | リスク | ロールバック |
|---|---|---|
| A | 調査時に診断が見えない | マクロ ON で再ビルド（コード変更不要） |
| E | 既存プロジェクトの音量増、リミッタ接触 | CMake オプション OFF で旧挙動へ戻せる（既定のみ変更） |
| F | Bypass の CPU・メモリ増 | 変更した 5 箇所を戻す。幾何の期待値も同時に戻す |
| G | CI 時間の増加、oneAPI 導入失敗、ヘッダ生成のターゲット名依存 | 復元ステップを外し、監査を monitor へ戻す（fail-closed の意図は失われる）。ヘッダはカスタムターゲット化して名前依存を解消する |

## §8 Owner に確認いただきたい事項

1. **A**: 診断マクロを新設（`CONVOPEQ_ENABLE_LOAD_CHAIN_DIAGNOSTICS`、既定 OFF）とするか、既存 `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` に統合するか。推奨は新設
2. **E**: 互換性の告知範囲（リリースノートのみ／アプリ内通知／README 追記）。推奨はリリースノート＋README。旧挙動のビルドオプションを「サポート対象」とするかは別途判断
3. **F**: Bypass の層維持に伴う CPU・メモリ増の容認。MT-NUPC の T8 / M4 期待値（32 分割 → 新幾何）の変更承認
4. **F / H**: 幾何の不変条件（`l1Part ≤ L0 保持長`）の修正は、既定の tailMode=1 が block ≥ 1024 でテールを落とす既存不具合（H）も同時に直す。F の PR に H の修正を含めるか、別 PR に分けるか。推奨は同一 PR（同じ不変条件であり、分けると F の受入「ブロック長非依存」を満たせない）
5. **F**: block 8192 以上（倍率の下限 2 でも L0 保持長を超える領域）の扱い。L0 保持長を広げるか、対応ブロック長の上限を定めるか
6. **G**: CI への oneAPI 導入（winget＋キャッシュ、数分）の容認。共通アクション化まで行うか、コピーで済ませるか。生成ヘッダは ninja のファイル指定か、カスタムターゲット（`GenerateJuceHeader`）か
7. **実装単位**: 1 PR か 2 PR（A+G / E+F+H）か 4 PR か

## §9 変更規模（概算）

| 項目 | 変更行数目安 | テスト更新 | 文書 |
|---|---|---|---|
| A | 30〜40（ガード追加） | 無し（測定で確認） | 必要なら BUILD_GUIDE に 1 節 |
| E | 5（CMake）＋ テスト 120 前後 | 7 ゲート＋BUILD ブロック | 互換性の節、リリースノート |
| F | 80〜120（被覆・ゲイン・幾何の不変条件） | MT-NUPC の理論式と 2 ケース、新規完全性テスト（5 ブロック長）、H 回帰 | なし（動作変更のみ） |
| G | 60〜80（ワークフロー） | 無し | なし |

## §10 結論

- 承認済み 4 件（A・E・F・G）は、いずれも変更対象・受入基準・ロールバックが特定できた。設計はリポジトリ外の試作で検証済み（F は block 256〜4096 で IR 全体を厳密再構成、G はクリーン configure＋生成ヘッダで監査 PASS）。**追加の調査は不要**で、承認後すぐに着手できる
- F の検証中に、既定モードのブロック長依存（H: block ≥ 1024 でテール欠落）を新たに確認した。同じ幾何の不変条件で直るため、F と同一 PR とするのが妥当（§8 の 4）
- 未承認の B・C・D には触れない
- 実装・commit・PR・merge は、本スコープの承認まで行わない




