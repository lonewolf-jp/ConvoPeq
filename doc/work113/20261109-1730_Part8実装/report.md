# Part 8 修正の実装報告（PR-1 / PR-2）

- 完了: 2026-11-09 17:30
- 基準: `main` = `2aecd73e823067e8cf26c2d4a8fa2e4e6c6d3166`
- 実行範囲: Owner 実行許可（PR-1: A+G / PR-2: E+F+H+I）。merge・リリース・配布は未実施
- PR: [#205](https://github.com/lonewolf-jp/ConvoPeq/pull/205)（A+G）、[#206](https://github.com/lonewolf-jp/ConvoPeq/pull/206)（E/F/H/I）

## §1 コミットと変更ファイル

### PR-1（#205）

| 順 | コミット | 内容 |
|---|---|---|
| 1 | `43bf433f` fix(loader): gate IR load-chain diagnostics behind a build option | A |
| 2 | `f36d0560` ci(isr): prepare compile database and JUCE header before the clang-tidy audit | G |

変更ファイル: `src/convolver/ConvolverProcessor.LoaderThread.cpp`、`CMakeLists.txt`、`.github/workflows/isr-verification.yml`

### PR-2（#206）

| 順 | コミット | 内容 |
|---|---|---|
| 1 | `5a871d3b` feat(oversampling): enable the corrected polyphase gain convention by default | E |
| 2 | `42a04333` fix(convolver): keep the full IR in Bypass mode | F |
| 3 | `2fd80a18` fix(convolver): keep the tail layers aligned above 1024-sample blocks | H |
| 4 | `1a97ab14` fix(convolver): cap the L1 multiplier at 4 to keep the first tail partition | I |

変更ファイル: `CMakeLists.txt`、`src/MKLNonUniformConvolver.cpp` / `.h`、`src/tests/TailLayerIntegrityTests.cpp`（新規）、`src/tests/MT-NUPC-Measurement.cpp`、`src/tests/AudioEngineHarness/PolyphaseGainFidelityTests.cpp`、`src/tools/build_identity_gate.py`、`ARCHITECTURE.md`、`BUILD_GUIDE_WINDOWS.md`、`README.md`、`manual/AUDIO_SETTINGS_WINDOW_EN.md`、`manual/AUDIO_SETTINGS_WINDOW_JP.md`、`RELEASE_NOTES.md`（新規）

## §2 ビルド・テスト・静的検査

| 項目 | 結果 |
|---|---|
| 全ターゲットビルド（Release、MSVC 14.51） | 489 ステップ、成功 |
| `ctest` | **56/56 PASS**（350.96 秒） |
| 新規 `TailLayerIntegrityTests` | failures=0（F/H/I 全 ID） |
| `MTNUPCMeasurement` | structural failures=0（T8 79/79、M4 bypass 180/180、B13 grid `[NG]=0`） |
| `PolyphaseGainMetrics` | 全構成 DC 1.000000000（IIR3/LP3 × 2×/4×/8×） |
| `PolyphaseGainFidelityTests` | 54 PASS / 0 FAIL |
| clang-tidy 監査ゲート（exhaustive, enforce） | `[PASS] clang-tidy audit gate verified` |
| 変更 TU の clang-tidy（registry の required checks） | `MKLNonUniformConvolver.cpp`・`TailLayerIntegrityTests.cpp` とも警告・エラーなし |
| 既存警告の改善 | `bugprone-branch-clone`（main の Bypass 防御枝の重複）が F の変更で解消 |

## §3 受入条件の判定

| ID | 条件 | 判定 | 根拠（Production テスト） |
|---|---|---|---|
| F-1 | Bypass の完全再構成（≤1e-10） | **PASS** | block 64〜32768 × IR 0.5/1/3 秒で 3.3e-16〜3.9e-16 |
| F-2 | 末尾 100 ms が ±0.1 dB | **PASS** | 全組み合わせ |
| F-3 | 無音窓なし（各 8192 窓 ≥ 0.5×IR） | **PASS** | 全組み合わせ |
| F-4 | 幾何の健全性 | **PASS** | 層数 1〜3、partSize・numPartsIR 整合 |
| F-5 | 524288 の幾何健全性 | **PASS** | 1 層、partSize 524288、1 パート |
| H-1 | 既定モードのテール水準 1.176 dB ± 0.1 | **PASS** | block 256〜16384 で同一 |
| H-2 | block 64/128 の許容（±2.5 dB・無音なし） | **PASS** | 0.967 / 1.176 dB |
| H-4 | 3 秒 IR のテール保持 | **PASS** | −11.21 dB（64〜512）、−8.89 dB（1024〜16384） |
| I-1 | block 512 / 3 秒で `firstErr = -1` | **PASS** | maxErr 3.886e-16 |
| I-2 | `l1Part ≤ 4 × l0Part` かつ `≤ held` | **PASS** | l0 512/12、l1 2048/64 |

- 修正前の再現（H-3）: 既定モードは block 1024 で −8.06 dB、2048 で −13.99 dB、4096 で −10.31 dB、8192 以上で無音。Bypass は 32×blockSize で切断（0.171〜0.683 秒）。同じケースが修正前コミットで失敗し、4 コミット目で合格する
- 試作（リポジトリ外）の数値は設計選択の証拠であり、本表は Production テストの結果のみ

## §4 CPU・メモリ（Production テスト、48 kHz、IR 3.0 秒、5 回中央値）

| block | CPU（ms/音声 1 秒） | メモリ増分 |
|---|---|---|
| 64 | 7.33 | 7.61 MB |
| 128 | 5.55 | 8.38 MB |
| 256 | 4.01 | 11.86 MB |
| 512 | 3.34 | 11.58 MB |
| 1024 | 1.95 | 13.54 MB |
| 2048 | 1.77 | 13.50 MB |
| 4096 | 1.72 | 14.14 MB |
| 8192 | 1.42 | 15.54 MB |
| 16384 | 1.39 | 12.70 MB |
| 32768 | 1.44 | 19.99 MB |

- 最大 7.33 ms/音声 1 秒、19.99 MB。許容（10 ms、32 MB）内
- 測定条件・集計方法は実装準備仕様 §3.3 のとおり。環境非依存の保証値ではない

## §5 RT 安全性（I）

- 上限警告は `SetImpulse`（NonRT）にのみ存在（`MKLNonUniformConvolver.cpp` 660〜664 行）
- `Add` / `Get` に警告・追加分岐・アロケーションはない（ソース確認）
- `std::atomic<bool> m_blockSizeWarned` を `exchange` し、インスタンスあたり 1 回に制限
- `kMaxSupportedBlockSize = 32768`。超過時も `l1Part ≤ L0 被覆` を満たす健全な幾何を返す

## §6 互換性（E）

- 既存セッションは同一設定で 2.5〜7.5 dB 大きく再生される（2×/4×/8× の従来損失 2.50/5.00/7.50 dB）
- プリセット値は不変。`-DCONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF` で旧挙動を再現可能
- 告知: `README.md`、`manual/AUDIO_SETTINGS_WINDOW_EN.md`、`manual/AUDIO_SETTINGS_WINDOW_JP.md`、`RELEASE_NOTES.md`

## §7 PR 間の整合

- `main` から PR-1 → PR-2 の順にローカルの一時ブランチでマージし、**競合なし**を確認（一時ブランチは削除済み）
- `CMakeLists.txt` は A のオプションと E の既定 ON が共存することを確認

## §8 未解決・制約・残存リスク

- I の根本原因（block 512 で L1 が 13 分割以上になると先頭が落ちる理由）は未特定。倍率上限 4 は回避策であり、原因解明とは分離して記録する
- block 32768 超は性能保証外（幾何の健全性のみ担保）
- CPU・メモリは本機の測定値であり、他環境の保証値ではない
- 3.0 秒 IR の block 64/128 における層分割差（H-2 の 0.967 dB）は聴感評価を未実施
- CI の oneAPI 復元所要時間は未測定（ローカル configure 33.9 秒、ヘッダ生成 2 秒）
- merge・リリース・配布は Owner の別途許可が必要

## §9 実行した検証コマンド（要旨）

```
cmake -S . -B build -G "Ninja Multi-Config" -DCONVOPEQ_REQUIRE_MKL=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure          # 56/56
build/TailLayerIntegrityTests_artefacts/Release/TailLayerIntegrityTests.exe
build/MTNUPCMeasurement_artefacts/Release/MTNUPCMeasurement.exe
build/Release/PolyphaseGainMetrics.exe
build/Release/PolyphaseGainFidelityTests.exe
pwsh .github/scripts/isr-verify-clang-tidy-audit.ps1 -Tier exhaustive -RequireClangTidy
build/Release/AudioEngineHarness.exe --n1-analysis <dir> post  # PR-1 ブランチで stderr 0 行
```
