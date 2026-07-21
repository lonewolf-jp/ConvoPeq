# ConvoPeq ソースコード調査報告書（Part 10）

`ConvolverProcessor.MixedPhase.cpp`（Mixed Phase変換: Allpass設計によるIRの位相補正、874行）を確認しました。Finding #2（`IRAnalyzer.cpp`）と同型の規約違反が、想定より広い範囲で再発していることを確認しました。

---

## 0. 今回のサマリ

| No | 重大度 | ファイル | 概要 |
|----|--------|----------|------|
| 10 | **Medium**（規約違反が複数箇所で系統的に再発） | `ConvolverProcessor.MixedPhase.cpp`（他にも波及の可能性） | MKL DFTIを直接呼ぶ関数内で、`std::vector<double>`/`std::make_unique`が7箇所使用されている。Finding #2と同型の規約違反。 |

---

## 1. 【Medium】MKL関連バッファへのstd::vector/std::make_unique使用が複数箇所で再発

### 該当ファイル

`src/convolver/ConvolverProcessor.MixedPhase.cpp`（主に`convertToMixedPhaseAllpass`関数内、ローカル68行目〜）

### 発見の経緯

Finding #2（`IRAnalyzer.cpp`）と同じ規約違反パターン（「Audio thread以外で、かつMKL使用箇所ではnew/std::vector/std::make_uniqueを使用せず、mkl_malloc等を使用する」）が他のファイルにも波及していないか、`grep`で横断確認しました。

### 具体例（最も明確なもの）

```cpp
if (DftiComputeForward(dfti.handle, linearSpec.get()) != DFTI_NO_ERROR) return {};
if (DftiComputeForward(dfti.handle, minimumSpec.get()) != DFTI_NO_ERROR) return {};

std::vector<double> phiMinUnwrapped(static_cast<size_t>(complexSize));   // ← 規約違反
phiMinUnwrapped[0] = std::atan2(minimumSpec.get()[0].imag, minimumSpec.get()[0].real);
```

`linearSpec`/`minimumSpec`自体は`convo::makeAlignedArray`系のアライン済みバッファを正しく使用しているのに、その直後でFFT結果から位相を計算するための作業バッファ`phiMinUnwrapped`は`std::vector<double>`を使っています。同一関数内で「アライン済み確保」と「規約違反の確保」が混在している状態です。

### 確認した違反箇所一覧（同一関数`convertToMixedPhaseAllpass`内、他にも波及の可能性大）

| 変数 | 種別 | 用途（推定） |
|------|------|--------------|
| `cachedRho`, `cachedTheta` | `std::vector<double>` | ディスクキャッシュからの読込先 |
| `entry.ir` | `std::make_unique<juce::AudioBuffer<double>>` | キャッシュエントリ |
| `phiMinUnwrapped` | `std::vector<double>` | 最小位相スペクトルの位相アンラップ |
| `targetGroupDelayStd`, `optim_freq_hz`, `optim_target_gd` | `std::vector<double>` | Allpass設計の最適化ターゲット |
| `freq_hz` | `std::vector<double>` | 周波数軸配列 |
| `entry.ir`（2箇所目） | `std::make_unique<juce::AudioBuffer<double>>` | 変換結果のキャッシュ登録 |
| `rho`, `theta` | `std::vector<double>` | Allpass係数 |

いずれも同一関数（またはその中の入れ子スコープ）内にあり、この関数自体が冒頭から`DftiCreateDescriptor`/`DftiComputeForward`等のMKL DFTI APIを多用しているため、Finding #2と全く同じ理屈で規約違反に該当します。

### 重大度・実害の評価

- 本関数は非RTスレッド（IRロード/変換パイプライン、`ConvolverProcessor.LoaderThread.cpp`から呼ばれるバックグラウンド処理）専用であり、**Audio Thread規約そのものへの抵触ではありません**。
- `juce::AudioBuffer<double>`を返り値とし`std::function`を引数に取るシグネチャのため、この関数自体はおそらく`noexcept`ではなく（Finding #2のような`noexcept`+throw の組み合わせクラッシュリスクは低いと推測されます。ただし本报告では`noexcept`修飾子の有無を1文字単位までは再確認できておらず、"要確認"扱いとします）。
- 実害としては「規約からの逸脱」自体であり、Finding #2で述べた通り一貫性の欠如が将来のメモリ管理判断を誤らせるリスクにつながります。

### 推奨対応

個々の変数を1つずつ手動でパッチ化するよりも、**プロジェクト全体を対象にした機械的な監査**を推奨します。具体的には:

1. `#include <mkl.h>` または `#include <mkl_dfti.h>` を含む全ファイルを列挙する。
2. それらのファイル内で`std::vector`、`std::make_unique`、裸の`new`の出現箇所を洗い出す。
3. 各出現箇所について、実際にMKL API呼び出しと同一関数スコープ内にあるかを確認する（本報告の`IRAnalyzer.cpp`・`MixedPhase.cpp`の調査で用いた方法と同じ）。
4. Audio Threadから到達しないことを確認した上で、`convo::makeAlignedArray<T>`等の既存ラッパーへ一括置換する。

このアプローチの方が、本報告で1関数ずつ潰していくより、同種の見落としを構造的に無くせると考えます。今回は時間の制約上、上記の一覧を提示するに留め、個別のunified diffパッチは作成していません。

---

## 2. 調査範囲の更新

- `ConvolverProcessor.MixedPhase.cpp`（869行）: `convertToMixedPhase`/`convertToMixedPhaseAllpass`前半〜中盤を精読。`convertToMixedPhaseFallback`以降、および869行中の残り約300行は未読。

---

## 3. 次のステップ（提案）

1. `ConvolverProcessor.MixedPhase.cpp`の残り、`ConvolverProcessor.Rebuild.cpp`、`ConvolverProcessor.ResampleAndFallback.cpp`、`ConvolverProcessor.StateAndUI.cpp`
2. 上記「機械的監査」の実施（`#include <mkl.h>`ファイル × `std::vector`/`make_unique`出現箇所の全数チェック）
3. ISR系残り約12ファイル
4. `EQProcessor.h`（1225行、まだ断片的にしか見ていない主要ヘッダ）
