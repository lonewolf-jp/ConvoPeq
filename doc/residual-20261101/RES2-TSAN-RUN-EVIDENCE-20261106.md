# TSan 診断実走の記録（2026-11-06）

- 目的: RES2-11 の判断材料として、Linux 専用環境で TSan を有効にした configure・build・test を実行し、対象テストの採否と実走結果を記録する。
- 基準: `main` = `ce386431`（作業開始時）。作業ツリーには本作業の変更（CMakeLists.txt、src/DiagnosticsConfig.h）がある。
- 結論: TSan ランタイムは動作した。計装した 3 本は PASS、報告は 0 件である。ただし本命の `ISRSemanticValidationTests` はビルドできず、実走していない。これは TSan の失敗ではない。音声処理系が MKL と IPP を必須にしているためで、原因はビルド構成にある。

## 1. 環境

| 項目 | 値 |
|---|---|
| ホスト | Windows 11 Home 26300、Core i7-11700K、31.8 GB |
| 仮想環境 | WSL 2、Ubuntu 26.04.1 LTS、カーネル 6.18.54.1-microsoft-standard-WSL2 |
| 割当 | 2 vCPU、3,919 MB（`.wslconfig` の既存設定） |
| コンパイラ | clang 21.1.8 |
| ビルド | CMake 4.4.4、Ninja 1.13.2（Multi-Config） |
| 導入した依存 | `pkg-config`、`libasound2-dev`、X11 系、freetype、fontconfig、GL、curl |
| ビルド先 | `~/tsan-build`（ext4） |

WSL の `/` は 16 GB しかない。空きは 3.9 GB だった。ビルド先は `/mnt/c` ではなく、ディストリビューション内の ext4 に置いた。

## 2. 変更内容

| ファイル | 変更 | 目的 |
|---|---|---|
| `CMakeLists.txt` | `ENABLE_TSAN` ブロックで、テストターゲットにも `-fsanitize=thread` を付ける | `ConvoPeq` のフラグはテストへ伝播せず、テストが未計装だった |
| `CMakeLists.txt` | 非 MSVC のときに限り、対象テストの `COMPILE_OPTIONS` を空にしてから計装フラグを付ける | テスト定義に付く `/EHsc`・`/utf-8` が clang では入力ファイル扱いになり、ビルドできない |

### 2.1 撤回した変更

`src/DiagnosticsConfig.h` の判定を `defined(JUCE_DSP_USE_INTEL_MKL)` から値判定へ変える案を試したが、撤回した。理由は ASan ジョブの失敗である。

- 値判定にすると、`-D` で 1 を渡す翻訳単位と、`juce_dsp.h` が 0 を定義する翻訳単位が混在する。
- その結果、確保は `mkl_malloc`、解放は `system_aligned_free` という組み合わせが生じ、`bad-free` になった（`SUMMARY: AddressSanitizer: bad-free ... DiagnosticsConfig.h:38 in convo::system_aligned_free`）。
- 確保と解放の組み合わせを揃える修正は本作業の範囲外なので、元の判定に戻した。

TSan の計装には影響しない。計装した 3 本は `DiagnosticsConfig.h` を経由しない。

## 3. configure

```
cmake -S . -B ~/tsan-build -G "Ninja Multi-Config" \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DENABLE_TSAN=ON -DCONVOPEQ_REQUIRE_MKL=OFF
```

| 項目 | 結果 |
|---|---|
| 成否 | 成功 |
| 所要 | 86.0 秒（初回）、56.3 秒（再実行） |
| 出力 | `ENABLE_TSAN=ON: ThreadSanitizer enabled (Clang only)` |
| MKL | `MKL: not found, CONVOPEQ_REQUIRE_MKL=OFF → system aligned allocator (work72)` |
| IPP | `Intel IPP not found - skipping` |

## 4. 対象テストの採否

| テスト | 採否 | 根拠 |
|---|---|---|
| `ISRSemanticValidationTests`（現行） | 計装するが**実走不可** | `MKLNonUniformConvolver.h:47` が `mkl.h` を要求し、ビルドが停止（3/156 オブジェクト、20.5 秒） |
| `ISRRuntimeIdentityTests` | 採用・実走 | ビルド成功。TSan シンボル 817 |
| `MpscBoundedRingTests` | 採用・実走 | ビルド成功。TSan シンボル 817 |
| `ScopedEpochReaderStructuralTests` | 採用・実走 | ビルド成功。TSan シンボル 817 |
| `ScopedEpochReaderTests` | 不採用 | リンクで `curl_*` 未定義。JUCE の curl 連携が Linux で解決しない（TSan とは無関係） |
| `RetireGraceSemanticsTests` | 不採用 | `add_dependencies(... ConvoPeq)` があり、MKL 依存でビルドできない |

`ISRRuntimeIdentityGeneratorsTests` という名前のターゲットは存在せず、実体は `ISRRuntimeIdentityTests` だった。資料に書かれた名前と実際のターゲット名が違うと、configure の結果だけを見て採否を決めると取り違えるため、名前は実際のターゲットで記録し、資料側の記述は次回更新する。

## 5. 実走結果

実行条件は 3 本とも同じ。`TSAN_OPTIONS="halt_on_error=0:exitcode=66:second_deadlock_stack=1"`。この設定は競合を検出しても停止せず、終了コード 66 を返す。停止させると最初の報告で実行が止まり、残りの検査結果が失われるため、診断では停止しない設定を選んだ。

| テスト | 終了コード | 実行時間 | 検査 | TSan 報告 |
|---|---|---|---|---|
| `MpscBoundedRingTests` | 0 | 0.66 秒 | 10 件 PASS | 0 |
| `ScopedEpochReaderStructuralTests` | 0 | 0.48〜0.51 秒 | 11 件 PASS | 0 |
| `ISRRuntimeIdentityTests` | 0 | 0.14 秒 | 3 件（成功時は無出力） | 0 |

- 実走は各 1 回、ヘッダ撤回後の再実行が 1 回である。いずれも終了コード 0 で、検査は通った。回数が少なく、再現性は未確認である。
- TSan ランタイムの初期化は `TSAN_OPTIONS=verbosity=1` で確認した。`***** Running under ThreadSanitizer v3 *****` を出力している。
- 同じ確認で `WARNING: ThreadSanitizer: memory layout is incompatible, possibly due to high-entropy ASLR` が出て、固定アドレスで再実行されていた。WSL カーネルの ASLR 設定によるもので、テスト結果には影響していない。

## 6. 遮断要因

`ISRSemanticValidationTests` を実走できない理由は、次の 3 点である。

1. `src/MKLNonUniformConvolver.h:47-48` が `mkl.h` と `ipp.h` を無条件に include する。これは音声処理の主要ヘッダで、`AudioEngine.h` 経由で多くのテストがこれを辿る。
2. MKL と IPP の探索は、CMake で MSVC のときだけ実行される（`CMakeLists.txt:79-92`、`118`）。Linux では `CONVOPEQ_HAS_MKL` が OFF のままになる。
3. MKL のリンクも MSVC のときだけ設定される（`CMakeLists.txt:1747-1748` ほか）。Linux に MKL を入れても、include パスとリンクの指定が無い。

つまり、TSan のためだけに Linux 環境を整えても本命テストは動かない。include は見つからず、リンクも通らない。要するに、環境の問題ではない。構造の問題だ。動かすには、MKL・IPP 依存を Linux でも解決する変更が要る。あるいは、音声処理系を切り離したテスト用ターゲットを新設する。どちらもビルド構成の変更であり、本作業の承認範囲を超える。

## 7. 未検証・未達

- `ISRSemanticValidationTests` の TSan 実走（ビルド不能）。
- 検出能力の評価。報告 0 件は「競合が無い」ことの証明ではない。3 本はいずれも小規模で、ISR の publish・retire 経路を実行していない。対象は MPSC リング、reader-safety の構造検査、ID 生成の単調性だけであり、音声スレッドと公開スレッドが実際に競合する場面を再現していないため、検出できたとしても範囲は限られる。
- 繰り返し実行、負荷を変えた実行、失敗時の分類。
- Windows での TSan。clang の TSan は Windows を対象にしていない。
- 同じ構成での再実行。WSL の割当を変えた場合の時間と成否も見ていない。

## 8. 判断材料

| 論点 | 現時点の材料 |
|---|---|
| 専用環境は用意できるか | 用意できる。configure とテストのビルド・実行は成功した |
| 本命テストは動くか | 動かない。MKL・IPP の Linux 対応が前提になる |
| TSan の検出力は確認できたか | ランタイムの動作は確認した。検出力は未評価 |
| 必須ゲート化の条件 | 未充足。対象テストが実走できていない |

範囲は狭い。これは記録であって、達成の証明ではない。

## 9. 実行時間の実測

| 工程 | 時間 |
|---|---|
| 依存パッケージの導入 | 約 3 分 |
| configure（初回） | 86.0 秒 |
| configure（再実行） | 56.3 秒 |
| `ISRSemanticValidationTests` のビルド（失敗まで） | 20.5 秒 |
| 採用 3 本のビルド | 約 3.5 分（configure の再実行を含む） |
| 採用 3 本の実走 | 合計 1.31 秒（初回 1.28 秒） |
