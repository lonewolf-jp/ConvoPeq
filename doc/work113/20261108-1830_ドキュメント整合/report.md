# 4 ドキュメントの実装整合（README / ARCHITECTURE / SOUND_PROCESSING / BUILD_GUIDE）

- 実施: 2026-11-08 17:30〜
- 対象: `main` = `4d65cf39`
- 目的: ルートの 4 ドキュメントを現在のソースコードの実装に合わせ、可能な限り漏れなく修正する。英語で記述する。`README.md` の "New in v0.6.11" 節は対象外。

## 1. 方法

1. 4 ドキュメントを読み取り専用の調査エージェント 4 本で並行監査し、各記述をコードと突き合わせた（`README.md`、`ARCHITECTURE.md`、`SOUND_PROCESSING.md`、`BUILD_GUIDE_WINDOWS.md`）。
2. 監査結果のうち影響の大きい指摘は自分でコードを読み直して裏取りした（クラス名、API シグネチャ、DSP 段順、遮断周波数、定数、CMake のオプション、build.bat の分岐）。
3. 数値（ファイル数、行数、サイズ、テスト数）は `git ls-files`、`stat`、`CMakeLists.txt` の解析で再計算した。
4. 修正は決定的な置換スクリプトで適用し、適用漏れ（MISS）を 0 件にしてから次へ進んだ。

## 2. README.md（33 件）

| 項目 | 修正前 | 修正後（実測） |
|---|---|---|
| `src/` の規模 | 336 files / ~4.62 MB | 385 files / ~6.11 MB |
| テスト | 36 executables / 55 files / 40 `add_test` | 53 executables / 97 files / 54 `add_test` |
| `src/audioengine/` | 126 files | 128 files |
| `src/convolver/` | 10 files | 12 files（`Internal.h`、`Runtime.cpp` を含む一覧に修正） |
| `src/core/` | 40 files | 41 files |
| `EpochDomain.h` | 32.2 KB | 35.6 KB |
| `tools/` | 66 .py + 54 .bat | 81 .py + 54 .bat + 2 .cmd |
| `.github/` | 184 files | 268 files |
| `CMakeLists.txt` | 2,030 行 / v0.6.10 | 2,583 行 / v0.6.11 |
| `build.bat` | 347 行 / icx は `-j 1` | 358 行 / icx は `-j 2` |
| MSVC Debug の AVX2 | baseline ISA | `/arch:AVX2`（全構成） |
| コールバック入口 | `ScopedNoDenormals` | `ensureThreadFloatingPointEnvironment()`（スレッドごとに 1 回） |
| DSP 段順 | Soft Clip → Makeup Gain / Dither → Downsampling | Makeup Gain → Soft Clip / Downsampling → DC block → Dither |
| OutputFilter | 最終プロセッサで条件分岐 | 段ごとに適用（conv 出力に HC/LC、EQ 出力に HPF/LPF） |
| `LoaderThread` | `std::thread` | `juce::Thread` |
| Dither の GUI 表記 | 「9th-order」 | 「12th-order」（9th は adaptive 側） |
| DC ブロック | 2 段 | 入力・OS 後・出力の 3 箇所 |
| IR 畳み込みの FFT | MKL DFTI | 実時間は Intel IPP（DFTI はアナライザと IR 前処理） |

## 3. ARCHITECTURE.md（44 件の意味修正 + 193 のサイズ欄 + テスト 14 行）

- 学習モードを実装どおり 6 種（Shortest / Short / Middle / Long / Ultra / Continuous）に修正し、目標時間（10/30/60/120/300 秒、Continuous は無期限）を記載した。従来の「Short/Medium/Long + スペクトル 3 種」は存在しない。
- Layer 4 を `RuntimeIntentCoordinator` に修正し、API を `precheckPublish()` → `commit()` → `enqueueRetire()`/`retire()` に直した。`publishWorld()` は存在しない。
- `Admission.evaluate(PublishRequest, AudioEngine, RuntimeReaderContext)`、`buildRuntimePublishWorld(sealedSnapshot, spec)`（旧オーバーロードも併記）へシグネチャを修正した。
- `worldId` を Authoritative に移し、`config/authority_inventory.json` のスナップショットが Diagnostic のままである点を注記した。
- RuntimeGraph のパスを `src/audioengine/RuntimeGraph.h` に修正した。
- シャットダウン順を実装（`requestShutdownClearNonRt` → `drainDeferredRetireQueues` → `drainPendingRetireIntentsForShutdown` → `drainAll` → `markShutdownComplete`）に修正した。
- `EQCoeffCache` の寿命は `DSPHandleRuntime` と明記した（`RefCountedDeferred` ではない）。
- EpochDomain の「per-slot `alignas(64)`」を削除し、実装（ホットなカウンタのみパディング）に合わせた。
- アナライザタップは固定遅延より前、XRUN は `max(1.5×expected, 3.0 ms)`、OutputFilter は 2 段適用（`convIsLast` は存在しない）に修正した。
- 未記載だった `CONVOPEQ_ENABLE_TEST_HOOKS` と `CONVOPEQ_CORRECT_POLYPHASE_GAIN` を追加した。
- 見出しの版を 0.6.11 に更新し、末尾の日付を 2026-11-08 にした。
- 全表のサイズ欄を実ファイルから再計算して置換した（193 セル）。テスト表は 53 実行ファイルを網羅し、不足していた 14 件を追加した。

## 4. SOUND_PROCESSING.md（約 54 件 + 信号フロー図の再構成）

- `DSPCore::process()` を `processDouble()` に修正した。
- 入力 DC ブロックの係数は `std::expm1()` による `alpha = 1 − exp(−ω)` と明記した。
- アナライザ FIFO（`LockFreeAudioRingBuffer`、1,048,576 サンプル）と適応キャプチャキュー（`LockFreeRingBuffer<AudioBlock, 4096>`）を明確に区別した。
- Mixed 位相は「低域が最小位相、高域が線形位相」に修正した（従来は逆）。
- 畳み込みの FFT を Intel IPP に修正し、AVX2 はホットループであることを明記した。
- OutputFilter を段別適用に修正し、遮断周波数が内部処理レート基準であることを明記した。
- ソフトクリップのパラメータを `saturationAmount` からの導出（0.95−0.45s / 0.05+0.35s / 0.10s）に修正し、`prevSampleInOut` が未使用であること、`_mm256_round_pd` を使っていないことを反映した。
- Dither の GUI 表記を "12th-order" に修正し、RNG リングが 8 チャンネルであること、最大整形誤差が約 −60 dBFS であることを修正した。
- `EQCoeffCache` の寿命、`SMOOTHING_TIME_SEC` が線形ランプであること、makeup gain がソフトクリップの前であることを修正した。
- 信号フロー図を実装順（makeup → clip → downsample → アナライザタップ → DC ブロック → dither → メータ → リミッタ → 固定遅延 → 出力）に再構成し、局所 2× ソフトクリップ OS と出力 DC ブロックを追加した。
- 参照表のサイズを実測値に更新した。

## 5. BUILD_GUIDE_WINDOWS.md（39 件 + テスト 14 行 + 手順 1 件）

- `icpx` は環境設定を行わない別名であることを明記した。
- configure は最大 3 試行（2 リトライ）、identity gate は初回に stamp を作成し以後 `--check`、Phase 8 は icx のみ、RC1109 の検出は無く非 MSVC 失敗時に 1 回リトライ、と実装どおりに修正した。
- `/MT` は icx の Debug にも適用されることを修正した。
- clangd 用 `compile_commands.json` のコピー・変換・BOM 除去という手順を追加した。
- `CONVO_CI_BUILD` 定義時に clang-tidy が強制 ON になることを追記した。
- icx の PGO は `/Qprof-gen` があるが未統合であることを明記した。
- VS Code の icx reconfigure タスクが `-B build`（MSVC のビルドディレクトリ）を使う危険を注記した。
- `launch.json` の 2 構成が `build\Release\ConvoPeq.exe` を指しており実体と異なることを注記した。
- テスト一覧に 14 件を追加し、実行ファイル数・`add_test` 数を 53 / 54 に修正した。
- 未記載オプション 2 件と `CONVOPEQ_PGO_PGD` を追加した。

## 6. 検証

- 4 ドキュメントの見出し数が変更前と一致すること（README 34、ARCHITECTURE 46、SOUND 54、BUILD_GUIDE 76）を確認した。
- コードフェンスが偶数であること、表の列数が 3〜5 であることを確認した。
- 修正したサイズ欄を実ファイルと突き合わせた（`AudioEngine.h` 315.6 KB、`EpochDomain.h` 35.6 KB、`ConvolverProcessor.h` 72.7 KB、`.Timer.cpp` 122.1 KB など）。
- `find-nise-kanji` で 4 ドキュメントに偽漢字がないことを確認した。
- README の "New in v0.6.11" 節に差分がないことを `git diff` で確認した。
- `0.6.10` の残存が 0 件であることを確認した。

## 7. 作業中の事故と復旧

ARCHITECTURE.md のテスト行を統合するスクリプトで、削除範囲の終端を「文書内で最後の表の行」と誤って指定し、約 460 行を削除してしまった。`git checkout` で原文に戻し、保存済みの決定的な修正スクリプト（意味修正 → テスト表 → サイズ → 表の修復 → 行の統合）を順に再適用して復旧した。復旧後は見出し 14 節、テスト表 53 行、末尾の版表記まで確認している。

## 8. 確認できなかった事項

- AMD CPU での icx バイナリの実行可否（実行環境がない）。
- icx Debug が実際にコンパイル不能であること（同ビルドを実行していない。構成ゲートの存在のみ確認）。
- 学習モードの旧「収束時間」記述はコードに対応する定数がないため、実装の目標時間に置き換えた（推測値は残していない）。

## 9. 変更ファイル

| 種別 | ファイル |
|---|---|
| 文書 | `README.md`、`ARCHITECTURE.md`、`SOUND_PROCESSING.md`、`BUILD_GUIDE_WINDOWS.md` |
| 報告 | 本ファイル |
