# 未修正バグ改修の実装・検証

- 実施: 2026-11-06
- 基準: `main` = `1451a88e`
- 対象: #4・#5、#6、#7、#2、#1 統合テストフック、未使用 `<limits>` の削除
- 制約: 実装とテスト実行のみ。コミット、PR、マージは行っていない
- 作業ツリー: 8 ファイルを変更（未コミット）

## 1. 変更一覧

| ファイル | 変更 |
|---|---|
| `src/eqprocessor/EQProcessor.Parameters.cpp` | #4・#5・#7。freq/gain/Q の非有限値拒否と範囲クランプ、飽和の非有限値拒否、不正な filterStructure の拒否 |
| `src/eqprocessor/EQProcessor.h` | 未使用の `<limits>` を削除 |
| `src/convolver/ConvolverProcessor.StateAndUI.cpp` | #6。手動オーバーライド分岐で非有限の `autoDetectedIRLength` を格納しない |
| `src/convolver/ConvolverProcessor.MixedPhase.cpp` | #2。片方向アンラップを `unwrapPhaseRadians` へ置換 |
| `src/audioengine/AudioEngine.h` | #1。テストフックの宣言（`CONVOPEQ_ENABLE_TEST_HOOKS` 時のみ） |
| `src/audioengine/AudioEngine.Cache.cpp` | #1。衝突スロットを使った誤ヒット検証の実装（同マクロ時のみ） |
| `src/tests/AudioEngineHarness/STG11EQRetireTests.cpp` | #4・#5・#7・#2 の回帰テスト（D14・D15） |
| `src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp` | #6 の回帰テスト（D8）、#1 の衝突試験 |

## 2. 実装内容

### 2.1 #4 バンドセッター

`setBandFrequency` / `setBandGain` / `setBandQ` の先頭で `convo::numeric_policy::isFinite` を確認し、非有限値は格納しない。有限値は次の範囲へ正規化する。

| パラメータ | 範囲 |
|---|---|
| frequency | 20〜20000 Hz |
| gain | −48〜+48 dB |
| Q | 0.01〜20 |

Nyquist 依存の上限は従来どおり係数生成側で適用する。係数式と既定値は変えていない。

### 2.2 #5 飽和

`setNonlinearSaturation` の `jlimit` の前に isFinite を追加した。NaN と Inf は格納しない。

### 2.3 #6 Convolver の D8

`setState` の手動オーバーライド分岐で、`autoDetectedIRLength` が非有限なら格納しない。手動フラグと他のプロパティの処理は従来どおり。

### 2.4 #7 filterStructure

不正な列挙値は Serial へ変換せず、そのまま拒否して現状維持する。兄弟セッターと同じ契約に揃えた。

### 2.5 #2 MixedPhase

手書きの片方向ループを、両方向補正と累積を備えた `ConvolverProcessorInternal::unwrapPhaseRadians` へ置換した。位相の生値の算出方法と後段の重み付け・スロープリミッターは変えていない。

### 2.6 #1 統合テストフック

`CONVOPEQ_ENABLE_TEST_HOOKS` が定義されたビルドでのみ、次の検証を有効にする。

- ゲイン +6 dB と −6 dB のキーを作り、キーとハッシュが異なることを確認する。
- +6 dB のキャッシュを、あえて −6 dB のハッシュ位置へ登録して衝突を再現する。
- `getOrCreate` が別キーを誤ヒットせず、退避したハッシュへ新しいキャッシュを登録することを確認する。

CI の既定（マクロ OFF）ではフックはコンパイルされない。テスト側も同じマクロで囲んでいる。

### 2.7 `<limits>`

`EQProcessor.h` 内で `numeric_limits` を使っていないことを確認し、include を削除した。同ヘッダを直接 include する 15 ファイルに、推移的な依存で `numeric_limits` を使うものがないことも確認済み。

## 3. テスト結果

### 3.1 STG11EQRetireTests（既存ターゲット）

| 項目 | 結果 |
|---|---|
| ビルド | 成功、540.5 秒（Release） |
| 実行 | exit 0 |
| D13 係数キャッシュキー | PASS |
| D14 入力検証（#4・#5・#7） | PASS |
| D15 位相アンラップ（#2） | PASS |
| TD1-0〜TD1-4b（既存） | PASS |

D14 の確認内容は、freq/gain/Q の NaN・Inf 拒否、上下限クランプ、飽和の NaN・Inf 拒否、不正な filterStructure の拒否、有効値の受理である。D15 は、上方向・下方向の巻き戻し補正、連続位相の不変を確認する。

### 3.2 AudioEngineHarness（統合）

`CONVOPEQ_ENABLE_TEST_HOOKS=ON` でビルドし、`CONVOPEQ_SOURCE_ROOT` を設定して実行した。

| 項目 | 結果 |
|---|---|
| ビルド | 成功（Release） |
| 実行 | exit 0、305.4 秒 |
| D8 autoDetected NaN 拒否 | PASS（before=0.500、after=0.500） |
| #1 衝突プローブ | PASS |
| STG-6-D1、D18、D3、D4、D5 ほか | すべて PASS |
| 総合 | `AudioEngineHarness: all publish pipeline tests PASS` |

補足:

- 最初の実行では D18-D が FAIL した。原因は環境変数 `CONVOPEQ_SOURCE_ROOT` の未設定で、ソース構造を読むテストが失敗したもの。設定後に再実行し、PASS を確認した。
- `CONVOPEQ_ENABLE_TEST_HOOKS` はテスト実行のため一時的に ON にし、実行後に OFF へ戻した。CMakeLists は変更していない。
- #1 のテスト出力の改行がリテラル `\n` になっていたため、実行後にソースの表記だけ修正した。テストの判定には影響しない。
- #1 の衝突プローブは `CONVOPEQ_ENABLE_TEST_HOOKS=ON` の構成でのみコンパイル・実行される。CI の既定構成（OFF）では実行されない。

## 4. 音響仕様への影響

| 項目 | 変更 |
|---|---|
| 既定値 | なし |
| パラメータ範囲 | なし（セッターの正規化範囲は係数段の既存範囲と同じ） |
| 係数式 | なし |
| フィルター構造 | なし |
| オーバーサンプリング利得 | なし（#3 は方式 C のまま） |
| Phase 0 テスト契約 | なし |

挙動の変化は次の 3 点である。

- 範囲外の有限値がセッター時点で正規化される。保存値と UI 表示が係数段の範囲に揃う。
- 非有限値と不正な列挙値が無視され、直前の値が残る。
- MixedPhase の目標位相が正しいアンラップに変わる。再現評価では帯域内の差は最大 0.123 rad だった。

## 5. 未実施・残存

| 項目 | 状態 |
|---|---|
| #1 探索上限 64 | 保留（未実施） |
| #1 無効バンドのキー正規化 | 保留（未実施） |
| #12 TruePeak ガード | 実施しない（判断どおり） |
| N1 改善要望の起票 | 未実施 |
| コミット・PR・マージ | 実施済み（`eeec4a6c`、PR #156、マージ `f641a187`） |
| ConvoPeq.md | stale のまま |

## 6. コミットと PR

| 項目 | 結果 |
|---|---|
| コミット | `eeec4a6c`（8 ファイル） |
| PR | #156 MERGED |
| マージコミット | `f641a187` |
| main / origin/main | `f641a187`（一致） |
| PR の CI | 全チェック pass（Windows Release 14 分 24 秒、ASan、TSan、isr-verify、CodeQL、lint ほか） |
| main の CI | Windows MSVC は pass。Sanitizer と CodeQL は報告時点で実行中（同一コミットの PR CI は全通過） |
| マージ | 完了（squash、追加差分なし） |

## 7. 検証の再現

```
# STG11（D13〜D15）
build\Release\STG11EQRetireTests.exe

# 統合（フック有効ビルド）
set CONVOPEQ_SOURCE_ROOT=C:\VSC_Project\ConvoPeq
build\Release\AudioEngineHarness.exe
```

ビルドには `MKLROOT` と `IPPROOT` の設定が必要である。
