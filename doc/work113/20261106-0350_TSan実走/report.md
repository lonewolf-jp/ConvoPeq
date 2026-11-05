# RES2 検証遮断要因の解消（TSan 限定実走と soak 手動 1 回）

- 実施: 2026-11-06 03:00–04:10（Windows 11 x64、WSL 2 Ubuntu 26.04）
- 基準: `main` = `ce386431`
- 結論: TSan は計装と環境を整えて実走できたが、本命テストは MKL・IPP 依存でビルドできなかった。soak は手動 1 回を実施し、ALL PASS だった。TX 実機は指定待ちで未実施。

## 1. 実施した作業

| 対象 | 作業 | 結果 |
|---|---|---|
| TSan | 計装範囲の変更、Linux 依存の導入、configure・build・test | 計装 3 本が PASS・報告 0 件。本命テストはビルド不能 |
| soak | 現行 commit からハーネスを再ビルドし、1 回実行 | ALL PASS、744.8 秒 |
| TX 実機 | 実機・シナリオ・実施者の確認 | 未指定のため未実施 |

## 2. 変更ファイルと commit

| commit | 内容 |
|---|---|
| `6ea370db` | `CMakeLists.txt`: テストターゲットへの TSan 計装、非 MSVC での MSVC 専用フラグ除去 |
| `dc89a334` | 記録 2 本の新規作成と、レジスタ・台帳・判断材料の更新 |
| `4fc5d1fe` | `src/DiagnosticsConfig.h` の変更を撤回（ASan の bad-free による） |
| `57fd2c14` | 撤回の反映と、実走の再実行記録 |

RT 経路、Publish、Retire、Epoch、Crossfade、DSP の実装、既定値、CI workflow、schedule は変更していない。

`src/DiagnosticsConfig.h` の判定を値ベースに変える案は、ASan で `bad-free ... system_aligned_free` を出したため撤回した。確保と解放で別のアロケータになる翻訳単位が生じるためで、組み合わせを揃える修正は本作業の範囲外である。

この失敗は、`defined()` による判定が、MKL ビルドで全翻訳単位を MKL アロケータに寄せていたことを示す。値判定にすると、その暗黙の統一が崩れる。

## 3. 実行環境

| 項目 | 値 |
|---|---|
| TSan | WSL 2、Ubuntu 26.04.1、clang 21.1.8、CMake 4.4.4、2 vCPU、3,919 MB |
| soak | Windows 11 Home 26300、Core i7-11700K、31.8 GB、oneAPI あり |
| ハーネス | `build\Release\AudioEngineHarness.exe`（2026-11-06 03:09 ビルド） |

## 4. 実測結果

### 4.1 TSan

| テスト | 終了コード | 実行時間 | 検査 | 報告 |
|---|---|---|---|---|
| `MpscBoundedRingTests` | 0 | 0.66 秒 | 10 件 PASS | 0 |
| `ScopedEpochReaderStructuralTests` | 0 | 0.48〜0.51 秒 | 11 件 PASS | 0 |
| `ISRRuntimeIdentityTests` | 0 | 0.14 秒 | 3 件 | 0 |

- configure は 86.0 秒（初回）、56.3 秒（再実行）で成功した。
- TSan ランタイムは `Running under ThreadSanitizer v3` で初期化を確認した。
- `ISRSemanticValidationTests` は 3/156 オブジェクトで停止した。原因は `MKLNonUniformConvolver.h:47-48` の `mkl.h`・`ipp.h` である。

### 4.2 soak

| 項目 | 値 |
|---|---|
| コマンド | `--soak --scenario=all` |
| 実行時間 | 744.8 秒（12.4 分） |
| 結果 | ALL PASS、issued 180,300 / accepted 180,300 / rejected 0 |
| メモリ | 284.7 MB で収束 |
| 前回（2026-11-05） | 799.9 秒 |

## 5. 判定根拠

- TSan の実走は、終了コード 0 と TSan 報告 0 件で判定した。聴感や主観は使っていない。
- soak は `ALL PASS` の出力と、issued・accepted・rejected・backlog の一致で判定した。
- 本命テストを実走できていないため、TSan の検出力は評価していない。3 本の PASS を完全なスレッド安全性の根拠にはしない。

## 6. 遮断要因の解消状況

| 要因 | 状態 |
|---|---|
| テストが TSan で計装されない | 解消（計装対象を追加） |
| WSL の pkg-config・JUCE 依存不足 | 解消（依存を導入し configure 成功） |
| テストに付く `/EHsc`・`/utf-8` でビルド不能 | 解消（非 MSVC では除去） |
| MKL の無い環境で `mkl.h` を要求 | 未解消（判定の変更は ASan の bad-free で撤回） |
| 音声処理系が MKL・IPP を必須にしている | **未解消**。Linux では探索もリンクも MSVC 専用 |

## 7. 未確認事項

- `ISRSemanticValidationTests` の TSan 実走。MKL・IPP の Linux 対応が前提になる。
- TSan の検出力。3 本はいずれも小規模で、ISR の publish・retire 経路を実行していない。
- soak の連続実行と、セルフホストランナーとしての運用。
- 実機での 1.0 ms 達成、XRUN の有無、最終値一致。

## 8. 次の判断が必要な事項

| 事項 | 内容 |
|---|---|
| TSan の継続 | MKL・IPP の Linux 対応に着手するか、Windows で代替するか、現状で打ち切るか |
| soak の週次化 | セルフホストランナーの登録、費用、運用責任の確定 |
| TX 実機 | ASIO 実機、シナリオ、実施者、実施時期の指定 |
| waiver | 測定結果が出た後の個別判断 |

## 9. 検証

| 確認 | 結果 |
|---|---|
| Windows configure | 成功（MKL・IPP 検出） |
| Windows Release ビルド | `AudioEngineHarness` 成功（990.5 秒） |
| Linux configure | 成功 |
| 実走 | 3 本 PASS、soak ALL PASS |
| 文書 | 表の列数不一致 0、偽漢字 0、日本語 lint 0 |
| CI（PR #149） | Windows Release 通過（20 分 13 秒）、ASan 通過、isr-verify 通過。初回の ASan は `bad-free` で失敗し、ヘッダ変更の撤回後に通過 |
| 統合 | PR #149（`e1641405`）、スナップショット PR #150（`1163dcdf`） |
| `ConvoPeq.md` | 再生成して FRESH。分割 4 件（最大 2.00 MB）も生成 |
| Dropbox | 記録 9 件とスナップショット 5 件を納品し、再取得照合すべて一致 |
