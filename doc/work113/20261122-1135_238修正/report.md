# SimplePeakLimiter のソフトニー修正（#238）

対象は `SimplePeakLimiter` のソフトニーである。旧実装はニー区間でゲインが 1 を下回らず、リミッティングが始まらなかった。数式を境界条件から導き直し、実装とテストを入れ替えた。基準 main は `a85d6cac`。

## 1. 結論

- 旧ニーは終端が閾値そのものだったため、`threshold / peak = 1` となりゲイン削減が起きなかった。区間幅も 0.58 dB しかなく、コメントの「1.0 dB」と一致していなかった。
- ニーを閾値中心の対数振幅対称（全幅 1.0 dB、-2.0〜-1.0 dBFS）に置き、両端の値と傾きを満たす 3 次エルミートで結んだ。開始点はゲイン 1・傾き 0、終端点は `T/peak` とその傾き `-T/peak^2` に一致する。
- 期待値は実装とは別の方法（p 空間の 3 次多項式を 4 条件から解く）で作り、実装値との差は最大 2.3e-15 だった。
- 単体テストを追加し、13 種類のケースを確認した。MSVC Release でビルド・実行し、CTest でも通る。Float 経路と Double 経路は同じ定数と同じ呼び出しになった。

## 2. 仕様とパラメーター（Phase 1）

### 2.1 呼び出し側

| 経路 | ファイル | 行 |
| --- | --- | --- |
| Float | `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` | 508-514 |
| Double | `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` | 724-733 |

どちらも `SimplePeakLimiter::processBlock` を呼び、その直後に `±kOutputHeadroom` のハードクランプを置く。リミッターはクランプの前段である。

### 2.2 定数の意味

| 定数 | 値 | 意味 |
| --- | --- | --- |
| `kPLThreshold` | 0.8413951287507587 | `10^(-1.5/20)`。**-1.5 dBFS** |
| `kOutputHeadroom` | 0.8912509381337456 | `10^(-1.0/20)`。**-1.0 dBFS**（ハードクランプ） |
| 旧 `kPLKnee` | 0.108748 | `kOutputHeadroom * (10^(1/20) - 1)`。**-1 dBFS 基準の 1 dB の線形差** |

旧 `kPLKnee` は dB 幅ではない。線形振幅の差を「1 dB」と呼んでいた値で、しかも基準は閾値ではなく -1 dBFS である。これを `threshold - knee * 0.5` として足し引きすると、ニーは **-2.08〜-1.50 dBFS（0.58 dB 幅）** になり、コメントの 1.0 dB とも合わない。

### 2.3 旧ニーが動かない理由

- 区間は `clipStart < peak <= threshold`。この範囲では `threshold / peak >= 1` である。
- ゲインは `1 - (1 - threshold/peak) * kneeShape`。`1 - threshold/peak <= 0`、`kneeShape > 0` なので、**ゲインは 1 以上**になる。
- envelope は `desiredGain < envelope` のときだけ下がる。ゲインが 1 以上なので下がらない。実測でも、ニー区間の実効ゲインは 1.000000 のままだった。
- 終端 `peak = threshold` では `t = 0.5` で、smoothstep の後半を使っていない。しかもその点のハードニーゲインは `T/T = 1` なので、ここで削減を終える設計自体が成立しない。

### 2.4 設計の確定

- 閾値は -1.5 dBFS、ハードクランプは -1.0 dBFS。**閾値中心で全幅 1.0 dB のニーは、上端がちょうど -1.0 dBFS になる**。クランプとの接続が自然である。
- ソフトニーは「閾値の上下 ±0.5 dB」に置く。上端でハードニーの `T/peak` に接続し、それ以上は従来どおり `T/peak` とする。
- 幅の定義は dB（対数振幅）にする。リミッターの閾値もクランプも dBFS で決まっており、線形振幅で対称にすると上下の dB 幅がずれる。
- `ConvoPeq.md`（`a85d6cac` 時点の生成物）の `SimplePeakLimiter.h` 節と `kPLKnee` の記述を確認し、作業対象のソースと一致していることを確かめた。

## 3. 数式の導出（Phase 2）

### 3.1 境界

`T` を閾値、`K` をニー全幅（dB）とすると、

```
a = T * 10^(-K/40)      （ニー開始、-2.0 dBFS）
b = T * 10^(+K/40)      （ニー終了、-1.0 dBFS）
```

`T = 0.8413951287507587`、`K = 1.0` のとき `a = 0.7943282225511487`、`b = 0.8912509244752658` である。`b` は `kOutputHeadroom` と 1e-9 以内で一致する。

### 3.2 境界条件

求めるゲイン `g(p)` は次を満たす。

```
g(p) = 1                      (p <= a)
g(a) = 1,  g'(a) = 0
g(b) = T/b, g'(b) = -T/b^2
g(p) = T/p                    (p >= b)
```

上端の値と傾きはハードニー `T/p` のものなので、`b` でゲインも傾きも連続する。出力 `p*g(p)` で見ると、下端の傾きは 1、上端の傾きは `T/b - T/b = 0` となり、一定出力の領域へ滑らかに移る。

### 3.3 補間式

`x = (p - a) / (b - a)` として、3 次エルミート基底で

```
g = h00(x) * 1 + h01(x) * (T/b) + h11(x) * (b - a) * (-T/b^2)
h00 = 2x^3 - 3x^2 + 1
h01 = -2x^3 + 3x^2
h11 = x^3 - x^2
```

`h10` の項は `g'(a) = 0` なので落ちる。

### 3.4 線形振幅で補間する理由

- dB 領域で補間すると、サンプルごとに `log10` と `pow` が要る。RT 経路では避けたい。
- ここでは幅の定義だけを dB にし、補間は線形ゲインで行う。境界の値と傾きを線形ゲインで指定するので、`p*g` の連続性と傾きの連続性は線形領域で保証される。
- 追加コストは 1 ブロックあたり `pow` 2 回だけ。サンプルループには除算も新しい関数呼び出しも増やしていない（ニー区間は多項式のみ）。

### 3.5 端の場合

- `K <= 0`、`K` が NaN、`K` が極端に大きく境界が非有限: ハードニー（`p > T` で `T/p`）へ戻す。
- `T <= 0`、`T` が NaN、`T` が無限: リミッティングしない（ゲイン 1 のまま）。

### 3.6 単調性と上限

- ゲインは `[1, T/b]` の範囲で単調減少、出力 `p*g` は単調増加であることを 2,000 点の掃引で確認した。
- 出力の上限は `T`（-1.5 dBFS）。ハードクランプの -1.0 dBFS には届かない。

## 4. 数値検証（Phase 3）

期待値は p 空間の 3 次多項式 `c0 + c1 p + c2 p^2 + c3 p^3` を、上の 4 条件から連立で解いて求めた。実装（エルミート基底）とは別の式である。

`T = 0.8413951287507587`、`K = 1.0`、`a = 0.7943282225511487`、`b = 0.8912509244752658`。

| 点 | 入力ピーク | 正規化位置 x | 期待ゲイン | 実装ゲイン（後） | 旧実装の実効ゲイン | 差（期待−実装） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 開始点の直前 | 0.794248790 | -0.001 | 1.000000000 | 1.000000000 | 1.000000 | 0.0e+00 |
| 開始点 | 0.794328223 | 0.000 | 1.000000000 | 1.000000000 | 1.000000 | 0.0e+00 |
| 25% | 0.818558898 | 0.250 | 0.996071968 | 0.996071968 | 1.000000 | 2.2e-15 |
| 50% | 0.842789574 | 0.500 | 0.984863655 | 0.984863655 | 0.999980 | 2.3e-15 |
| 75% | 0.867020249 | 0.750 | 0.967238733 | 0.967238733 | 0.996860 | 2.2e-15 |
| 終了点 | 0.891250924 | 1.000 | 0.944060876 | 0.944060876 | 0.993760 | 0.0e+00 |
| 終了点の直後 | 0.891340050 | 1.001 | 0.943966480 | 0.943966480 | 0.993660 | 0.0e+00 |
| 閾値 | 0.841395129 | 0.486 | 0.985690944 | 0.985690944 | 1.000000 | 1.9e-15 |
| 強い制限（1.4） | 1.400000000 | — | 0.600996521 | 0.600996521 | 0.600997 | 0.0e+00 |
| 強い制限（2.0） | 2.000000000 | — | 0.420697564 | 0.420697564 | 0.420698 | 0.0e+00 |

- 旧実装の実効ゲインは、ニー区間で 1.0 のままである（`desiredGain > 1` は envelope を下げない）。表の「旧実装」列がその実測である。
- 旧実装の閾値直前は 1.000000、新実装は 0.985691 で、0.5 dB のニー内で 0.125 dB の削減が始まっている。
- 開始点・終了点の前後 1e-4 相対での差は 0 で、値の連続性を確認した。傾きは開始点で 0、終了点で -1.059254（理論値 `-T/b^2 = -1.059254`）。

## 5. テスト（Phase 4）

`src/tests/SimplePeakLimiterKneeTests.cpp` を追加し、`CMakeLists.txt` に `SimplePeakLimiterKneeTests` として登録した（CTest）。

- 期待値はすべてリテラルで、実装の式を評価して作っていない。導出は p 空間の連立解で、テストのヘッダにその手順を書いた。
- 各ケースは新しいインスタンスと 1 サンプルのブロックで測る。envelope の時定数と静的カーブを混同しない。
- 13 ケースの対応は次のとおり。

| 要求されたケース | テスト |
| --- | --- |
| Knee 幅 0 | `checkHardKneeWhenDisabled`（knee=0） |
| Knee 幅が小さい | 0.1 dB の 3 点（`kGainCases`） |
| 通常の Knee 幅 | 1.0 dB の 8 点 |
| 開始点前後 | 開始点の直前・開始点・連続性 |
| Knee 中央 | 25%・50%・75% |
| 終了点前後 | 終了点・直後・連続性・傾き |
| 閾値直前 | `threshold` の行（50% と同一点） |
| 閾値直後 | 終了点以降のハードニーと 1.4 |
| 強い制限 | 1.4、2.0 |
| 左右で異なるピーク | `checkChannels`（L=0.5、R=0.95） |
| 右が nullptr | `checkMonoNullRight` |
| 境界の浮動小数点 | `std::nextafter` での前後比較 |
| 不正パラメーター | `checkInvalidThreshold`、`checkExtremeInputsAndKnee` |

実行結果（MSVC Release）:

```
[PASS] static soft-knee curve (11 cases)
[PASS] knee <= 0 or NaN falls back to the hard knee
[PASS] knee start/end are continuous (value)
[PASS] slope is continuous at both knee boundaries (-1.059254)
[PASS] curve is bounded by 1 and monotone (2000 points)
[PASS] output ceiling is T (-1.5 dBFS) and the hard clamp is not reached
[PASS] stereo uses the louder channel and scales both (0.88567908289553554)
[PASS] right=nullptr uses the left peak
[PASS] invalid threshold passes through without NaN
[PASS] extreme inputs and huge knee stay finite
SimplePeakLimiterKneeTests: PASS
```

CTest: `ctest -C Release -R SimplePeakLimiterKneeTests` で 1/1 Passed（0.16 秒）。既存の `CorruptSampleGuardTests` と `GainStagingContractTests` を加えた 3 本でも 3/3 Passed だった。

## 6. 修正（Phase 5）

変更は `SimplePeakLimiter.h` と 2 つの呼び出し側、テスト、CMake 登録に限った。

- `processBlock` の引数を `kneeDb`（dB 幅）に変え、境界 `a`、`b`、終端ゲイン、終端傾きをブロック先頭で求める。
- サンプルループは、`p <= a` で 1、`p >= b` で `T/p`、その間でエルミート多項式を評価する。
- RT 制約: 動的確保なし、ロックなし、ブロック処理なし、`noexcept` 維持。追加はブロックあたり `pow` 2 回と、ニー区間の多項式（乗算 5 回程度）。
- 呼び出し側は `kPLKneeDb = 1.0` を渡す。Float と Double で同一の値・同一の呼び出しである。

Hard Clamp、Safety Net、Release 時定数、LookAhead、他のリミッター段には触れていない。

## 7. 統合検証（Phase 6）

| 項目 | 結果 |
| --- | --- |
| 単体テスト | 10 チェックすべて PASS（CTest 1/1） |
| 既存 DSP テスト | `CorruptSampleGuardTests`、`GainStagingContractTests` と合わせて 3/3 Passed |
| Float 経路 | `DSPCoreIO.cpp` で `kPLThreshold` と `kPLKneeDb = 1.0` を渡す |
| Double 経路 | `DSPCoreDouble.cpp` で同じ値・同じ呼び出し |
| Hard Clamp との接続 | ニー終端 = -1.0 dBFS = `kOutputHeadroom`。出力上限は T（-1.5 dBFS）でクランプに達しない |
| Release ビルド（MSVC） | ConvoPeq とテストを Release でビルド、成功 |
| アプリ起動 | `--cli-run --cli-exit-ms 1500` で終了コード 0、`SHUTDOWN_END` まで到達 |
| Release ビルド（icx） | `build.bat Release icx app nopause` で成功（警告 158、エラー 0） |
| 静的ゲート | `check-audioengine-lint.ps1` 合格、`check-list-compliance.ps1` 合格（警告のみ） |
| 修正前後のカーブ | `before_curve.csv` と `after_curve.csv`（400 点、-6〜+6 dBFS） |

icx のビルドは `build-icx` を新規に構成して行った。所要は約 9 分、生成物は `build-icx\ConvoPeq_artefacts\Release\ConvoPeq.exe`（48,721,920 バイト）である。同じ exe を `--cli-run --cli-exit-ms 1500` で動かし、終了コード 0 と `SHUTDOWN_END` までを確認した。

修正前後のカーブは同じフォルダに置いた。`before_table.csv` と `after_table.csv` は上の表の生データである。

## 8. 範囲外として残すこと

- `DSPCore::reset()` が `peakLimiter.reset()` を呼ばない。envelope が DSPCore の寿命中に持続する。C-49 とは別の項目で、今回の修正には含めていない。
- `limitingEngaged` の metric は `>= kPLThreshold` で数える。新しいニーは -2.0 dBFS から削減を始めるため、メトリクスの盲点は残る。これも別項目である。
- ハードクランプと Safety Net の設計は変更していない。

## 9. 参照

- Issue #238
- 監査: `doc/work113/20261121-1201_詳細検証/report.md` の §6 と §9.9（C-49 の判定訂正）
- 旧判定の訂正: `doc/work113/20261121-2233_PRI01照合/report.md` §10
- 対象: `src/audioengine/SimplePeakLimiter.h`、`AudioEngine.Processing.DSPCoreIO.cpp`、`AudioEngine.Processing.DSPCoreDouble.cpp`
- テスト: `src/tests/SimplePeakLimiterKneeTests.cpp`、`CMakeLists.txt`
- 生データ: 同フォルダの `before_table.csv`、`after_table.csv`、`before_curve.csv`、`after_curve.csv`、`probe.cpp`
