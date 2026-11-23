# バグ精査 4 件の個別修正と検証

- 実施日時: 2026-11-15 17:00〜18:20 JST
- 対象リビジョン: `daf45501020a7471d03445e51eee2061291413ec`（`main`、PR #214 のマージコミット）
- 作業ツリー: `C:\VSC_Project\ConvoPeq-fix`（`daf45501` からの worktree、ブランチ `fix/bugtriage-20261115`）
- 本番ソースを変更していないツリー: `C:\VSC_Project\ConvoPeq`（他タスクの未コミット変更 189 件を含む。今回は一切触れていない）

## 1. 結論

| 優先 | ID | 判定 | 実装 | 検証 |
| ---: | --- | --- | --- | --- |
| 1 | BUG-013 | **バグ（音響誤出力）** | 修正済み（commit `1e0d54b7`） | 3 テストすべて PASS。旧コードでは 17 件 FAIL することを確認 |
| 2 | B-1 | **バグ（OOM 時のリーク）** | 修正済み（commit `0c494037`） | 旧所有順で 61,571,072 B 漏れることを実測。修正後 0 B |
| 3 | B-2 | **バグ（死んだ早期終了条件）** | 修正済み（commit `eed93ae2`） | 設計結果が旧実装と完全一致することを確認 |
| 4 | BUG-028 | **非バグ（判定を変更）** | **変更なし** | MSVC と icx の双方で `vzeroupper` が自動生成されることを逆アセンブルで確認 |

3 件を修正し、1 件は「バグではない」と判定を改めた。BUG-028 は前回の精査レポートで「バグ（低重大度）」と書いたが、これは**ソース上の記述だけを見てコンパイラの生成コードを確認していなかったための誤り**である。今回の実測で取り下げる。

## 2. 着手前に整えた前提（最新 ConvoPeq.md）

Owner 指示どおり、`daf45501` をクリーンにチェックアウトした一時 worktree から `ConvoPeq.md` を再生成し、2 MiB 以下に 4 分割した。

- 生成: `output_sourcecode_markdown.py`（`TARGET_ITEMS = {src, build.bat, CMakeLists.txt}`）、397 セクション、7,102,462 B
- 分割: `### 📄` のセクション境界で 4 分割。全パートが 2 MiB 以下、連結すると元ファイルとバイト一致
- 配置: `C:\VSC_Project\ConvoPeq-md-daf45501\`（リポジトリ外。本体ツリーの `ConvoPeq.md` は他タスクの未コミット変更なので上書きしていない）
- Dropbox: `/ConvoPeq-md-daf45501-20261115/`（`ConvoPeq.md`、`ConvoPeq_part1〜4.md`、`MANIFEST.md`）
- PR #213（Live Runtime Evidence）と PR #214（icx 対応）の変更が入っていることを文字列の出現数で確認済み

## 3. BUG-013（音響誤出力）— commit `1e0d54b7`

### 3.1 原因

`ProductionFft::createPlan()` は IPP の次数を

```cpp
int order = 0; int tmp = fftSize;
while (tmp > 1) { tmp >>= 1; ++order; }   // 非 2 冪では floor(log2)
```

で求めていた。IPP の実数 FFT は 2 の冪長しか扱えないため、非 2 冪の `fftSize` を渡すと実際のプランは `2^order` 点になる。一方 `Plan::complexSize` は `fftSize / 2 + 1` のままで、`isValid()` は `spec != nullptr` しか見ない。呼出側は「要求した長さのプラン」を得たと誤認する。

到達経路は MKLNonUniformConvolver の L2 テール層:

```
l0Part = nextPowerOfTwo(max(blockSize, 64))   // 2 の冪
l1Part = l0Part * l1Mult                      // l1Mult は 1/2/4
l2Part = l1Part * tailL1L2Multiplier          // 2〜16 を 1 刻みで UI から選べる
fftSize = l2Part * 2
```

既定の倍率 8 は 2 の冪なので既定構成では発火しない。3、5、6、7、9〜15 を選ぶと L2 テールの FFT 結果が壊れる。Air Absorption モードは下限が 6 で、これは 2 の冪ではない。

### 3.2 修正（2 層に分けた）

1. **`src/FFTBackend.cpp`**: `createPlan()` の先頭で `fftSize < 2` と非 2 冪を拒否する（fail-closed）。不正長は `isValid() == false` の Plan を返すので、既存の失敗経路（`releaseAllLayers()` して `false`）にそのまま乗る。
2. **`src/TailLayerMultiplier.h`（新規）**: 倍率の契約を 1 箇所に閉じ込めた。`sanitizeTailL1L2Multiplier()` は要求値を [2, 16] の 2 の冪へ丸める（同距離は大きい方＝テール分解能を削らない）。`MKLNonUniformConvolver` は丸めた値を 1 回だけ適用し、値が変わったときは NonRT ログに残す。Air Absorption の下限は 6 → 8 に変更。層の設定では `partSize` が 2 の冪でなければ fail-closed で `false` を返す（防御層）。

丸め結果の例: 2→2、3→4、5→4、6→8、7→8、9〜11→8、12〜16→16。

### 3.3 検証

| テスト | 内容 | 結果 |
| --- | --- | --- |
| `FFTBackendTests`（既存 + 追加） | 2 の冪 8 種を受理、非 2 冪と範囲外 12 種を拒否、`fftSize`/`complexSize` の整合 | PASS |
| `TailLayerMultiplierTests`（新規ターゲット） | 丸め表 20 ケース、[-64, 256] 全域の不変条件、冪等性、`l0Part × l1Mult × 倍率` の合成 FFT 長 | PASS |
| `TailLayerIntegrityTests`（既存 + 追加） | 実コンバータで tailMode 0/1、倍率 12 種。全層 `partSize` が 2 の冪、かつ丸め後の値で作った参照と出力がビット一致 | PASS（failures=0） |

**旧コードでの再現**（`src/FFTBackend.cpp` と `src/MKLNonUniformConvolver.cpp` を `daf45501` に戻して再ビルド）:

- `FFTBackendTests`: `FAIL testProductionFftPlanPowerOfTwoContract: non-power-of-two -1024 was accepted`
- `TailLayerIntegrityTests`: **failures=17**。`l2` が 12288 / 14336 / 18432 / 20480 / 22528 / 24576 / 26624 / 30720 となり、倍率 6・7・9〜15 で `outMatch=0`（出力が実際に変わる）

つまり新しいテストは旧実装の不具合を確実に検出する。

## 4. B-1（OOM 時のメモリリーク）— commit `0c494037`

### 4.1 原因

`AudioSegmentBuffer::create()` は次の順で確保していた。

```cpp
auto* left  = aligned_malloc(...);      // 失敗時は std::bad_alloc を送出（nullptr を返さない）
auto* right = aligned_malloc(...);
if (!left || !right) { ... }            // 到達しないデッドコード
auto buf = std::unique_ptr<AudioSegmentBuffer>(new AudioSegmentBuffer());
buf->leftSamples_.reset(left);          // ここまで left/right は生ポインタ
```

- `convo::aligned_malloc` は失敗時に `std::bad_alloc` を送出するため `if (!left || !right)` は成立しない
- 2 本目で例外が出ると `left`（5 秒 × 768 kHz × 8 B = 30,720,000 B ≈ 29.3 MiB）が漏れる
- `new AudioSegmentBuffer()` で例外が出ると両方（≈ 58.6 MiB）が漏れる

### 4.2 修正

`AudioSegmentBuffer` を先に確保し、そこから先の所有者を常に `unique_ptr` にする。各バッファは確保直後に `reset()` で流し込む。

```cpp
auto buf = std::unique_ptr<AudioSegmentBuffer>(new AudioSegmentBuffer());
buf->leftSamples_.reset(static_cast<double*>(allocate(kCapacity * sizeof(double), kAlignment)));
buf->rightSamples_.reset(static_cast<double*>(allocate(kCapacity * sizeof(double), kAlignment)));
return buf;
```

これで失敗点は確保関数の 1 箇所だけになり、例外が出た時点の所有者は必ず `buf` になる（ヘッダが掲げていた strong exception guarantee を実際に満たす）。確保関数は `createWithAllocator()` として注入可能にし、テストから失敗を起こせるようにした。本番の呼出し元は `create()` のみ。

Owner レビュー指摘（allocator が nullptr を返すと不完全なバッファが返り得る）を受けて、契約をコード側でも強制している。`allocateChecked()` が nullptr を `std::bad_alloc` に変換するため、契約違反でも「確保済みに見えるが中身が空」のバッファは返らない。契約は「失敗時は必ず例外を送出し、nullptr を返さない」で、`testNullptrAllocatorFailsClosed` で固定した（`threw=1, calls=1, growth=0 B`）。

### 4.3 検証（新規ターゲット `AudioSegmentBufferTests`）

| ケース | 期待 | 実測 |
| --- | --- | --- |
| 成功経路（create → pushBlock → copyLatest） | 値が往復する | PASS |
| 検出器の感度（意図的に 1 本リーク） | 8 MiB 閾値を超える | 30,785,536 B を検出 |
| 1 本目の確保で例外 | 増加なし | 0 B |
| 2 本目の確保で例外（B-1 の本命） | 増加なし | **0 B** |
| 旧所有順に戻した変異版で 2 本目失敗 | 漏れる | **61,571,072 B**（2 回反復 × 30.7 MB） |
| 確保関数が nullptr を返す（契約違反） | `std::bad_alloc` で fail-closed | `threw=1, calls=1`、増加 0 B |

本番の唯一の呼出し元 `src/NoiseShaperLearner.cpp` が新しいヘッダでコンパイルできることも確認した。

## 5. B-2（CMA-ES の早期終了）— commit `eed93ae2`

### 5.1 原因と、精査時点の見立ての訂正

`CmaEsOptimizerDynamic::update()` は sigma を `[sigmaMin, sigmaMax]` にクランプする。旧条件

```cpp
if (currentSigma < std::min(1e-4, config.cmaesParams.sigmaMin * 0.1)) break;
```

は既定構成（`sigmaMin = 1e-6`、初期 sigma 0.3）で閾値 1e-7 となり、`sigma >= sigmaMin` なので**決して成立しない**。ここまでは精査時の指摘どおり。

ただし「だから常に最大世代まで回る」は誤りだった。実測すると、既定構成の実行は**停滞判定（gen > 20 かつ 6 世代連続で改善なし）で gen=32 に終了**しており、100 世代まで回っていなかった。つまり死んだ条件は計算時間を浪費していない。

到達可能にした場合の影響も測った。

| 案 | 既定構成 | `sigmaMin = 0.05` 構成 |
| --- | --- | --- |
| 旧実装（死んだ条件） | gen=32、誤差 20.873414 | gen=100、誤差 14.846581 |
| `sigma <= 1e-4` | gen=18、誤差 **20.874867** | gen=100、誤差 14.846581 |
| `sigma <= sigmaMin`（下限到達） | 誤差 20.873464 | 誤差 **21.564231** |

「下限到達で終了」は `sigmaMin = 0.05` の設計品質を 14.85 → 21.56 サンプルへ悪化させた。この実装の sigma は収束指標ではなく、クランプされた固定ステップ幅として働くため、下限到達は「探索の終わり」を意味しない。

### 5.2 修正

死んだ条件を削除し、終了判定を到達可能な 2 つ（重み付き RMSE 1.0 サンプル未満、停滞 6 世代）に一本化した。あわせて `setParams()` の二重呼出しを 1 回にまとめた（初期 sigma の範囲調整と後段の `setParams(p)` は同じ値だった）。

### 5.3 検証（新規ターゲット `AllpassDesignerCmaesTests`）

- 48 kHz、8 セクション、一定群遅延 10 サンプル、`shouldExit` の呼出し回数で実行世代数を観測
- 既定構成: gen=32/100、最大誤差 20.873414 サンプル、8 セクションすべて安定、同一シードで完全再現
- `sigmaMin = 0.05` 構成: gen=100/100、最大誤差 14.846581 サンプル
- **旧実装との比較**: 全セクションの `rho`/`theta` を `%.17g` で出力した `[SIG]` 行が 2 構成とも**完全一致**。世代数・誤差も一致

Owner の「修正後も最適化結果が変化しない」は、この一致で満たしている。

## 6. BUG-028（decimate 経路の zeroupper）— 判定を「非バグ」に変更

### 6.1 精査時の判定と、その根拠の弱点

前回は「`dotProductDecimateAvx2`（211〜276 行）と `decimateStage` の convCount 4〜7 経路（672〜694 行）に `_mm256_zeroupper()` が無い」ことを根拠に「バグ（低重大度）」とした。ソース上の記述だけを見ており、**コンパイラが何を生成するかを確認していなかった**。

### 6.2 実測

`CustomInputOversampler.cpp` から `_mm256_zeroupper()` を**すべて削除**した状態でコンパイルし、`dumpbin /disasm` で 256 ビット（ymm）を使う関数ごとに `vzeroupper` の有無を数えた。

MSVC 19.52（`/O2 /arch:AVX2`、`/GL` を外した構成）:

| 関数 | ymm 命令数 | vzeroupper |
| --- | ---: | ---: |
| `dotProductAvx2` | 14 | 2 |
| `dotProductDecimateAvx2` | 26 | 1 |
| `decimateStage` | 3 | 3 |

icx 2026.2（`/QxCORE-AVX2`）:

| 関数 | ymm 命令数 | vzeroupper |
| --- | ---: | ---: |
| `dotProductAvx2` | 65 | 1 |
| `dotProductDecimateAvx2` | 91 | 1 |
| `decimateStage` | 12 | 3 |
| `prepareStage` | 76 | 1 |

明示的な `vzeroupper` が 1 つも無いソースから、両コンパイラが AVX/SSE 境界に `vzeroupper` を自動生成している。MSVC は AVX を使う関数から外部関数を呼ぶ地点と関数出口に挿入し、icx（LLVM の `X86VZeroUpper`）も同じ。したがって**ソースに無いことは欠陥ではない**。

`dotProductAvx2` に既にある明示的な `_mm256_zeroupper()` も冗長（無害）である。削除は今回の範囲外として触っていない。

### 6.3 追加で分かったこと（別課題として記録）

MSVC は `__FMA__` マクロを定義しない。`src/CustomInputOversampler.cpp` の decimate SIMD 経路は

```cpp
#if defined(__AVX2__) && defined(__FMA__)
```

で囲まれているため、**MSVC ビルドではこの最適化経路がコンパイルされず、decimate はスカラーループに落ちる**（逆アセンブルで ymm 命令が 0 件であることを確認）。icx では `__FMA__` が定義されるため有効になる。性能差の是正は本件とは別課題とする。

## 7. 作業環境と、検証で踏んだ落とし穴

- ビルド: MSVC 14.52.37025（19.52）/ Ninja Multi-Config / Release。構成時に `MKLROOT` と `IPPROOT` が必要
- **ヘッダのみの変更は再ビルドされない**。このビルドツリーのコンパイル規則に depfile が無く、`AudioSegmentBuffer.h` を書き換えても `ninja: no work to do` になった。ヘッダを変えた検証では、依存する `.cpp` の mtime を更新して強制再コンパイルする必要がある（`Copy-Item` は mtime を保持するので、戻したあとの `os.utime` も必須）
- **`/GL`（LTCG）オブジェクトには機械語が無い**。逆アセンブルには `RelWithDebInfo` など IPO 無効の構成のオブジェクトを使う
- `dumpbin /disasm:no_rawbytes` はこのバージョンでは構文エラーになる。`/disasm` を使う
- `dumpbin` の出力を PowerShell のパイプで受けると stderr が握り潰される。ファイルへリダイレクトしてから読む

## 8. 変更ファイルとコミット

| commit | 内容 | 変更 |
| --- | --- | --- |
| `1e0d54b7` | BUG-013 | `src/FFTBackend.cpp`、`src/MKLNonUniformConvolver.cpp`、`src/TailLayerMultiplier.h`（新規）、`src/tests/FFTBackendTests.cpp`、`src/tests/TailLayerMultiplierTests.cpp`（新規）、`src/tests/TailLayerIntegrityTests.cpp`、`CMakeLists.txt` |
| `99886a2a` | B-1 | `src/AudioSegmentBuffer.h`、`src/tests/AudioSegmentBufferTests.cpp`（新規）、`CMakeLists.txt` |
| `bc854b6b` | B-2 | `src/AllpassDesigner.cpp`、`src/tests/AllpassDesignerCmaesTests.cpp`（新規）、`CMakeLists.txt` |

ブランチ `fix/bugtriage-20261115` は `daf45501` から 3 コミット（11 ファイル、+1,155 / −29）。

> Owner レビュー後の変更: B-1 のコミットに、`createWithAllocator()` の契約（失敗時は必ず例外）をコードで強制する `allocateChecked()` と、その検証 `testNullptrAllocatorFailsClosed` を追加した。コミットを amend したため `0c494037` → `99886a2a`、`eed93ae2` → `bc854b6b` に変わっている（コミット数は 3 のまま）。

## 8.1 納品（push / PR / Issue）

| 項目 | 内容 |
| --- | --- |
| push | `origin/fix/bugtriage-20261115` = `bc854b6b` |
| Pull Request | [#216](https://github.com/lonewolf-jp/ConvoPeq/pull/216)（base `main`、3 コミット、11 ファイル） |
| マージ | **完了**。`gh pr merge 216 --merge`（2026-11-15 20:27 JST、merge commit `296ecfebd7ac0a646a509cd0076ebbdd91310ff5`）。CI 16 pass / 1 skipping / 0 fail、CodeRabbit 指摘 0、`mergeStateStatus=CLEAN` を確認してから実行した |
| Issue（新規発見） | [#215](https://github.com/lonewolf-jp/ConvoPeq/issues/215) — MSVC の `__FMA__` 未定義で decimate SIMD 経路が無効になる件（**OPEN のまま**。本 PR では解決しない独立課題） |
| PR 本文の確認事項 | `tailL1L2Multiplier` の丸めは意図した仕様（UI 正規化はフォローアップ）／`createWithAllocator()` の契約はコードで強制 |

PR 本文には、B-2 が「処理時間の改善ではなく到達不能条件の削除」であること、BUG-028 を非バグと判定した根拠（逆アセンブル）、MSVC/FMA 問題を別 Issue にしたことを明記した。

## 8.2 CI とレビュー（PR #216）

| チェック | 状態 |
| --- | --- |
| Windows MSVC / Release | pass（24 分 33 秒） |
| isr-verify | pass |
| authority-runtime-tests / authority-static-checks | pass |
| lint-audioengine / compliance-list-md / audit-report-verification | pass |
| CodeQL（actions / c-cpp / csharp / javascript-typescript） | pass |
| Debug TSan（Clang, best-effort） | pass |
| CodeRabbit | pass（「No actionable comments were generated in the recent review.」） |
| Windows MSVC / probe（dispatch only） | skipping（意図的） |
| RelWithDebInfo ASan (MSVC, /MD) | pass（10:03 → 11:13、Build と Run tests with ASan を含む全ステップ success） |

**16 pass / 1 skipping / 0 fail。** CodeRabbit の指摘は無く、追加のソース修正は不要だった。PR の `mergeStateStatus` は `CLEAN`、`mergeable` は `MERGEABLE`。

### マージ後の main（`296ecfeb`）

`main` への push でも CI が走るため、マージ結果が main で壊れていないことを確認した。

| ワークフロー | 状態 |
| --- | --- |
| Windows MSVC Build | pass |
| Sanitizer CI（Debug TSan + RelWithDebInfo ASan、Build と `Run tests with ASan` を含む） | pass（11:27Z → 12:41Z） |
| audioengine-lint / list-compliance / ISR Authority Compliance v6.4 | pass |
| Push on main (dynamic) | pass |

**main の CI は 6 件すべて success、failure 0、実行中 0。** マージ後の `main` は `296ecfeb` のまま。

## 8.3 今回の PR に含めなかったもの（フォローアップ）

- 新規テスト 3 ターゲット（`TailLayerMultiplierTests` / `AudioSegmentBufferTests` / `AllpassDesignerCmaesTests`）を `CONVOPEQ_ASAN_TEST_TARGETS` に追加していない。ASan でのリーク検査を効かせる価値があるので、別コミットで追加するのが望ましい
- `tailL1L2Multiplier` の UI 表示正規化（スライダーを 2/4/8/16 に離散化）
- Issue #215（MSVC の `__FMA__` 未定義で decimate SIMD 経路が無効）の是正

## 9. テスト結果（ctest、Release）

```
1/5 Test #20: FFTBackendTests ..................   Passed    0.22 sec
2/5 Test #21: TailLayerMultiplierTests .........   Passed    0.07 sec
3/5 Test #50: TailLayerIntegrityTests ..........   Passed    3.46 sec
4/5 Test #51: AudioSegmentBufferTests ..........   Passed    0.45 sec
5/5 Test #52: AllpassDesignerCmaesTests ........   Passed    0.38 sec

100% tests passed out of 5
```

`ConvoPeq` 本体ターゲットも全ビルドに成功した（136 ステップ、エラー 0、`ConvoPeq_artefacts\Release\ConvoPeq.exe` 47,908,864 B を生成）。3 件の修正を含むアプリ全体がリンクまで通ることを確認済み。

## 10. 残課題

- Issue #215（MSVC で `__FMA__` 未定義のため decimate SIMD 経路が無効）の是正
- 新規テスト 3 ターゲット（`TailLayerMultiplierTests` / `AudioSegmentBufferTests` / `AllpassDesignerCmaesTests`）の `CONVOPEQ_ASAN_TEST_TARGETS` 追加
- `tailL1L2Multiplier` の UI 表示正規化（スライダーを 2/4/8/16 に離散化して実効値と一致させる。今回は 3 コミットに限定するため見送り）
- `dotProductAvx2` の冗長な `_mm256_zeroupper()`（無害。削除するかは方針次第）
- 前回レポートで「未実施」とした NUPC 幾何（BUG-010〜012, 014）とオーバーサンプリング（025, 026）の数値網羅検証

## 11. マージ後の最新 ConvoPeq.md

Owner から「ローカルの `ConvoPeq.md` が 2026-11-08 生成で今回の PR を反映していない」との指摘があったため、マージ後の `main`（`296ecfeb`）から再生成した。

- 基点: `296ecfebd7ac0a646a509cd0076ebbdd91310ff5`（PR #216 のマージコミット）
- 生成: 2026-11-15 20:28:25、401 セクション、7,149,199 B
- 分割: 4 パート、すべて 2 MiB（2,097,152 B）以下、連結すると元ファイルとバイト一致
- 収録確認: `sanitizeTailL1L2Multiplier`（10）、`kMinFftSize`（2）、`allocateChecked`（3）、`testNullptrAllocatorFailsClosed`（5）、`testProductionFftPlanPowerOfTwoContract`（6）、`checkTailMultiplierPowerOfTwo`（2）
- ローカル: `C:\VSC_Project\ConvoPeq-md-296ecfeb\`（6 ファイル）
- Dropbox: `/ConvoPeq-md-296ecfeb-20261115/`（`ConvoPeq.md`、`ConvoPeq_part1〜4.md`、`MANIFEST.md`）

前回のスナップショット（`daf45501`、7,102,462 B / 397 セクション）との差は、PR #216 の 11 ファイル +1,155 / −29 を含む main 側の更新。
