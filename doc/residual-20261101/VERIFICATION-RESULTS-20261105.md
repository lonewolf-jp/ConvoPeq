# 検証結果一覧（2026-11-05）

- 基準: `main` = `f90f56be`
- 実行環境: Windows 11 x64、icx 2026.2.0（oneAPI）、`build-icx`（Release, AVX2）、`build-icx-on`（同＋`CONVOPEQ_CORRECT_POLYPHASE_GAIN=ON`）

## 1. 実行した検証

| # | 対象 | コマンド | 結果 |
|---|---|---|---|
| 1 | ポリフェーズ利得・位相（OFF） | `build-icx\Release\PolyphaseGainFidelityTests.exe`（oneAPI 環境） | PASS=54 FAIL=0 |
| 2 | ポリフェーズ利得・位相（ON） | `build-icx-on\Release\PolyphaseGainFidelityTests.exe` | §2 に追記 |
| 3 | D8-1（RES2-17） | `ctest -R "^D8_1_WrapperCacheTests$"` | Passed 0.14 s |
| 4 | FFT（RES2-14） | `ctest -R "^FFTBackendTests$"` | Passed 0.12 s |
| 5 | 破損判定（RES2-02） | `ctest -R "^CorruptSampleGuardTests$"` | Passed 0.17 s |
| 6 | ソース整合 | `output_sourcecode_markdown.py --check` | FRESH（`NEWER_SRC_COUNT: 0`） |

## 2. ON ビルドの結果

PASS=19 FAIL=35（OFF 期待値に対する正の対照）。DC は全 6 構成で 1.0、SoftClip OS DC は 0.5、float/double は 3.31e-08 / 2.96e-08。既存記録の ON 19/54 を再現。

## 3. 実行できなかった検証

| 検証 | 理由 | 未検証範囲 |
|---|---|---|
| U5 の RT オーバーヘッド測定 | 本セッションでは独立測定を実施していない | On/Off のブロック時間比較 |
| preset replay 比較（案 E） | 対象プリセットの選定が必要 | 出力レベルの差 |
| EQ/IR boost 併用（案 E） | 測定条件の設計が必要 | 併用時の挙動 |
| 聴感評価 | 人手が必要 | 体感差 |
| B2 Stage 2 のオーバーヘッド | 実装していない | 本番計測の負荷 |
| TX-02/TX-10 実機 | 実機環境が必要 | 実機動作 |

## 4. 追加の数値解析（2026-11-05）

| 検証 | コマンド | 結果 |
|---|---|---|
| 案 E OFF/ON 6 指標 | `PolyphaseGainMetrics.exe`（build-icx / build-icx-on） | §4.1 の表 |
| U5 RT オーバーヘッド | `RtMeasureOverheadTests.exe`（env unset / 2 / 1） | Off 1.34、TimeOnly 26.84、Full 26.60 ns/block |
| Worker アサーション | `ConvoPeq` / `AudioEngineHarness` 再ビルド | ビルド成功（結果は §5） |

### 4.1 案 E の数値（OFF / ON）

| 構成 | DC | RMS 比 | ピーク比 | 0.05 応答 | レイテンシ |
|---|---|---|---|---|---|
| IIR3 r2 | 0.750000 / 1.000000 | 0.74005 / 0.98745 | 0.78582 / 1.03477 | 0.750663 / 1.000884 | 255 / 255 |
| IIR3 r4 | 0.562500 / 1.000000 | 0.55598 / 0.98840 | 0.93757 / 1.66678 | 0.562829 / 1.000585 | 286 / 286 |
| IIR3 r8 | 0.421875 / 1.000000 | 0.41692 / 0.98860 | 0.60689 / 1.43734 | 0.421883 / 1.000021 | 290 / 290 |
| LP3 r2 | 0.750000 / 1.000000 | 0.74022 / 0.98738 | 0.78265 / 1.03041 | 0.750126 / 1.000169 | 501 / 501 |
| LP3 r8 | 0.421875 / 1.000000 | 0.41792 / 0.99081 | 0.61216 / 1.45073 | 0.422092 / 1.000514 | 461 / 475 |

## 5. ビルド確認

icx Release で `ConvoPeq` と `AudioEngineHarness` を再ビルド（149 ステップ、error 0）。D8_1・CorruptSampleGuardTests・PolyphaseGainMetrics・RtMeasureOverheadTests の 4 テストが Passed（0.68 s）。
