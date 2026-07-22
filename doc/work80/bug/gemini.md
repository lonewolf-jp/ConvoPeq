ご提示いただいた `ConvoPeq` のソースコード（`CMakeLists.txt`、`build.bat`、`AlignedAllocation.h`、`AllpassDesigner.cpp` 等）および関連モジュールの調査を行いました。

分析の結果、**音声処理アルゴリズム（DSP）の計算ロジック**、**メモリ管理（ポインタ操作・診断機能）**、ならびに**コンパイル・ビルド設定**において発見されたバグおよび潜在的な不具合について詳細を報告いたします。

※なお、以前ご指摘のあった「IRファイルの位相最適化学習後にアウトプットレベルメーターが-100〜-120dBに低下する不具合」は修正済みであることを確認の上、それ以外の未解決の不具合に絞って分析しています。

---

## 1. 音声処理アルゴリズム（DSP）におけるバグ

### 🐛 Bug 1: オールパスフィルタ最適化における極半径 ($\rho$) の上限値不一致

* **対象ファイル**: `src/AllpassDesigner.cpp`
* **該当箇所**: `unconstrainedToRho` 関数および `sectionGroupDelay` 関数
* **問題の説明**:
`unconstrainedToRho` 関数では、無制約変数を極半径 $\rho$ に変換する際、上限を `0.98` に固定しています。
```cpp
inline double unconstrainedToRho(double x) {
    return 0.98 * stableSigmoid01(x); // 上限が 0.98
}

```


一方で、`sectionGroupDelay` 関数では極半径の上限を `0.995` にクランプしています。
```cpp
const double rho = std::clamp(std::abs(gain), 0.0, 0.995);

```


* **音声処理への影響**:
混合位相（Mixed Phase）補正や高解像度FIR/IIR位相整合において、急峻な群遅延特性を形成するためには $\rho = 0.99 \sim 0.995$ 程度の深い極が必要です。CMA-ES最適化ルーチン（`unconstrainedToRho`）の上限が `0.98` に制限されているため、アルゴリズムが目標とする群遅延ピークに収束できず、**高域・鋭い位相補正時の誤差が残存する**原因となります。
* **修正案**:
極半径の上限定数を統一（例: `0.995`）します。
```cpp
inline double unconstrainedToRho(double x) {
    constexpr double kMaxRho = 0.995;
    return kMaxRho * stableSigmoid01(x);
}

```



---

### 🐛 Bug 2: CMA-ES 目的関数におけるサンプル数未正規化による収束誤差

* **対象ファイル**: `src/AllpassDesigner.cpp`
* **該当箇所**: `designWithCMAES` 内の `costFunc`
* **問題の説明**:
`costFunc` は各周波数点での二乗誤差の総和の平方根を返しています。
```cpp
double weightedSquaredError = 0.0;
for (size_t i = 0; i < freq_hz.size(); ++i) {
    // ...
    const double diff = tau_sum - target_group_delay_samples[i];
    weightedSquaredError += weight[i] * diff * diff;
}
return std::sqrt(weightedSquaredError);

```


* **音声処理への影響**:
評価周波数点数 $N$ (`freq_hz.size()`) に依存してコスト関数のスケールが $\sqrt{N}$ 倍に増大します。評価グリッドの密度を変更するとコストの絶対値が変わるため、CMA-ESの終了判定閾値（`targetCost` や `tolFun`）が正常に機能しなくなり、**周波数分解能を変更した際に最適化が早期終了したり過剰ループを起こすバグ**を引き起こします。
* **修正案**:
重み付き平均二乗誤差（RMSE）となるよう、重みの総和または点数 $N$ で正規化します。
```cpp
double totalWeight = 0.0;
double weightedSquaredError = 0.0;
for (size_t i = 0; i < freq_hz.size(); ++i) {
    // ...
    weightedSquaredError += weight[i] * diff * diff;
    totalWeight += weight[i];
}
return std::sqrt(weightedSquaredError / (totalWeight > 0.0 ? totalWeight : 1.0));

```



---

### 🐛 Bug 3: `initialMean` における周波数パラメータの境界チェック漏れ（NaN発生リスク）

* **対象ファイル**: `src/AllpassDesigner.cpp`
* **該当箇所**: `designWithCMAES` 内の初期値設定処理
* **問題の説明**:
`config.minFreqHz` や `config.maxFreqHz` の値を対数変換（`std::log`）して初期中心周波数を決定しています。
```cpp
const double logMin = std::log(config.minFreqHz);
const double logMax = std::log(config.maxFreqHz);

```


* **音声処理への影響**:
UIや設定ファイルから `minFreqHz <= 0.0`（0Hz指定など）が渡された場合、`std::log(0)` により `logMin` が `-inf` となり、以降のパラメータベクトル全体が `NaN` / `-inf` に汚染されます。これにより**CMA-ES演算が即座にクラッシュするか、無効なオールパス係数を出力する**バグが発生します。
* **修正案**:
対数計算の前に可聴帯域・サンプリング周波数に基づいたガードを追加します。
```cpp
const double safeMinHz = std::max(1.0, config.minFreqHz);
const double safeMaxHz = std::min(0.48 * sampleRate, std::max(safeMinHz + 10.0, config.maxFreqHz));
const double logMin = std::log(safeMinHz);
const double logMax = std::log(safeMaxHz);

```



---

## 2. メモリ管理およびスレッド安全性におけるバグ

### 🐛 Bug 4: `ScopedAlignedPtr::reset` における自己代入保護の欠如 (Use-After-Free)

* **対象ファイル**: `src/AlignedAllocation.h`
* **該当箇所**: `ScopedAlignedPtr::reset`
* **問題の説明**:
`reset` メソッド内で、保持しているポインタと引数 `p` が同一である場合のチェック（`ptr != p`）が存在しません。
```cpp
void reset(T* p = nullptr) noexcept
{
    static_assert(std::is_trivially_destructible_v<T>, "...");
    if (ptr)
    {
        aligned_free(ptr); // p == ptr の場合、ここでメモリが解放される
    }
    ptr = p; // 解放済みのポインタが再代入される
}

```


* **影響**:
既存の `ScopedAlignedPtr` が管理する生ポインタを誤って再セット（`ptr.reset(ptr.get())` など）した場合、即座にメモリが解放されダングリングポインタとなります。その後の音声処理ループでアクセスするとランタイムクラッシュやメモリ破壊（Use-After-Free）を引き起こします。
* **修正案**:
`ptr != p` ガードを追加します。
```cpp
void reset(T* p = nullptr) noexcept
{
    static_assert(std::is_trivially_destructible_v<T>, "...");
    if (ptr && ptr != p)
    {
        aligned_free(ptr);
    }
    ptr = p;
}

```



---

### 🐛 Bug 5: 診断ビルド時（`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`）におけるメモリ解放非対称性

* **対象ファイル**: `src/AlignedAllocation.h`
* **該当箇所**: `aligned_malloc` および `aligned_free`
* **問題の説明**:
`aligned_malloc` 側ではマクロ `DIAG_MKL_MALLOC` を使用して診断モジュールへメモリ確保量を記録していますが、解放側の `aligned_free` では `mkl_free` を直接呼び出しています。
```cpp
inline void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr = DIAG_MKL_MALLOC(size, (int)alignment); // 確保を記録
    // ...
}

inline void aligned_free(void* ptr) noexcept {
    if (ptr != nullptr) {
        mkl_free(ptr); // 解放が記録されない！
    }
}

```


* **影響**:
診断有効化ビルド（`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1`）実行時に、メモリ確保カウンターのみが増加し解放カウンターが更新されません。これにより、**実際のメモリリークの有無にかかわらず `RuntimeHealthMonitor` が偽陽性のメモリリーク警告を発出し続ける**原因となります。
* **修正案**:
`aligned_free` 側も `DIAG_MKL_FREE` マクロを経由するように修正します。
```cpp
inline void aligned_free(void* ptr) noexcept {
    if (ptr != nullptr) {
        DIAG_MKL_FREE(ptr);
    }
}

```



---

## 3. コンパイル・ビルド設定の不具合

### 🐛 Bug 6: Intel ICX ビルド時におけるテストターゲットの MKL リンク定義漏れ

* **対象ファイル**: `CMakeLists.txt`
* **該当箇所**: `GainStagingContractTests` および `EQProcessorMaxGainTests` ターゲット設定
* **問題の説明**:
`CMakeLists.txt` 内で Intel ICX コンパイラ使用時（`CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"`）、メインターゲット `ConvoPeq` や一部テストには `/Qmkl:sequential` フラグが付与されていますが、`GainStagingContractTests` や `EQProcessorMaxGainTests` には同オプションが設定されていません。
* **影響**:
Intel ICX でビルドを実行した際（`build.bat Release icx` 等）、これらのテストターゲットにおいて MKL 関連シンボルの未解決リンクエラー（`LNK2019` / `LNK1120`）が発生し、**ビルドプロセスが失敗**します。
* **修正案**:
`CMakeLists.txt` 内の IntelLLVM 条件分岐ブロックに対象ターゲットの MKL フラグを追加します。
```cmake
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_compile_options(GainStagingContractTests PRIVATE /Qmkl:sequential)
    target_compile_options(EQProcessorMaxGainTests PRIVATE /Qmkl:sequential)
endif()

```



---

## 概要まとめ

| 番号 | 分類 | 不具合概要 | 主な影響 |
| --- | --- | --- | --- |
| **1** | DSP/アルゴリズム | CMA-ESにおける極半径 $\rho$ の上限値限制 (0.98 vs 0.995) | 高鋭度オールパス補正時の誤差残存 |
| **2** | DSP/数学 | CMA-ESコスト関数（RMSE）の点数未正規化 | 周波数グリッド変更時の最適化異常終了 |
| **3** | DSP/数学 | `initialMean` での `minFreqHz <= 0` ガード漏れ | NaN発生による最適化処理のクラッシュ |
| **4** | メモリ安全 | `ScopedAlignedPtr::reset` の自己再代入チェック漏れ | Use-After-Free によるランタイムクラッシュ |
| **5** | メモリ診断 | `aligned_free` での `DIAG_MKL_FREE` 未使用 | 診断ツールでの偽陽性メモリリーク検出 |
| **6** | ビルド環境 | Intel ICX 用テストターゲットでの `/Qmkl` オプション漏れ | ICXビルド時のリンカーエラー |


前回の6つの項目に加え、`ConvoPeq` のオーディオエンジン、同期機構、周辺DSPモジュール、およびCI/CD環境における最新のバグレポートと技術監査のログに基づき、さらに修正が必要な**8つの重大な不具合・潜在的リスク**を解析しました。

---

## 4. オーディオエンジンおよび同期・ライフサイクルにおけるバグ

### 🐛 Bug 7: RCU遅延削除キューにおけるメモリリーク

* **該当モジュール**: `AudioThread` 制御および EpochベースRCUモデル


* **問題の説明**:
リアルタイムスレッドの安全性を担保するために導入されている遅延削除キュー（deferred deletion queue）において、特定の状態遷移やスレッド間同期のタイミングにより、メモリの解放処理が正常にキックされないバグが確認されています。


* **音声処理への影響**:
EQパラメータの頻繁な変更やIR（インパルス応答）の再読み込みを繰り返すうちに、古いメモリブロックが解放されずに蓄積し、アプリケーションのメモリ使用量が肥大化（メモリリーク）します。



### 🐛 Bug 8: コンヴォルヴァー（Convolver）破棄時におけるUse-After-Free (UAF) エラー

* **該当モジュール**: 畳み込み演算エンジン（Convolver）のライフサイクル管理


* **問題の説明**:
UIや制御スレッド側からコンヴォルヴァーオブジェクトを破棄（Destruction）するタイミングと、リアルタイム音声処理スレッド（AudioThread）側での参照クリアのタイミングの間にレースコンディション（競合状態）が存在します。


* **音声処理への影響**:
IRファイルの切り替え時などに、すでに解放された古いコンヴォルヴァーインスタンスのポインタに対してAudioThreadがアクセスを試みるため、非決定的なランタイムクラッシュ（Use-After-Free）を引き起こします。



### 🐛 Bug 9: 状態遷移パスおよび遷移完了通知のハンドリング不備

* **該当モジュール**: 状態遷移マシン（State Machine）および通知ルーチン


* **問題の説明**:
`ConvoPeq` 内部のソースコードにおいて、処理状態の遷移パス（Transition Paths）の定義、および遷移が完了した際の「完了通知（Transition Completion Notifications）」のルーティングとハンドリングに不備が見られます。


* **音声処理への影響**:
バイパスの切り替えやフィルター形状のモーフィング中に処理が途中でスタックしたり、完了通知が正しく配信されないことで、UIと内部音声エンジン側の状態が乖離（不一致）する原因となります。



---

## 5. 信号処理（DSP）拡張モジュールにおけるバグ

### 🐛 Bug 10: ソフトクリッパー（Soft Clipper）におけるSIMD状態の破損

* **該当モジュール**: `SoftClipper` クラス（AVX2/SIMD最適化コード）


* **問題の説明**:
ソフトクリッパー内のSIMD並列演算ループにおいて、レジスタの退避・復元、あるいはメモリのアライメント境界のハンドリングにミスがあり、SIMD状態（SIMD State）の汚染・破損が発生しています。


* **音声処理への影響**:
クリッピング閾値付近での演算が異常値となり、出力音声に**突発的なプチノイズやバーストノイズ、あるいは激しい歪み**を混入させます。



### 🐛 Bug 11: ソフトニー・リミッター（Soft-Knee Limiter）における数学的不連続性

* **該当モジュール**: リミッターのダイナミクス処理アルゴリズム


* **問題の説明**:
ソフトニー（Soft-Knee）特性を計算する補間数式において、数式上の不連続点（数学的不連続性）が存在しています。


* **音声処理への影響**:
入力信号の振幅がニーステージの境界線を通過する際、ゲイン減少のカーブが滑らかにつながらず急峻に変化するため、**高次の高調波歪みや可聴帯域内のポップノイズ**を誘発します。



### 🐛 Bug 12: AVX2 デシメーション（間引き処理）におけるメモリ安全性問題

* **該当モジュール**: `AVX2 Decimation`（ダウンサンプリング/帯域分割処理）


* **問題の説明**:
AVX2を用いたデシメーション処理において、バッファ境界チェックの不足、またはSIMDポインタのインクリメント計算の誤りによるメモリ安全性の問題が指摘されています。


* **音声処理への影響**:
音声ブロックの末尾で領域外アクセス（境界外の読み書き）が発生し、**隣接するヒープ領域のデータを破壊するか、最悪の場合はアプリケーションが即座にクラッシュ**します。



### 🐛 Bug 13: ノイズシェーパ（Noise Shaper）のゲイン補正誤り

* **該当モジュール**: ディザリングおよび `NoiseShaper` モジュール


* **問題の説明**:
量子化ノイズを可聴帯域外へ追いやる際のフィルター処理において、適用されるゲイン補正係数（Gain Compensation）の計算ロジックに誤りがあります。


* **音声処理への影響**:
特定のサンプリングレートやビット深度においてノイズシェーピングが過剰または過少に働き、**想定よりもノイズフロアが上昇する、あるいは全体のエネルギーバランスが崩れる**不具合が生じます。



---

## 6. CI/CDおよびシステム同期のリスク

### 🐛 Bug 14: "ISR Verification" ワークフローの継続的な失敗

* **該当モジュール**: CI/CD環境（GitHub Actions等）および Practical Stable ISR Bridge Runtime


* **問題の説明**:
リアルタイム性と自己修復機能を検証するための自動検証ワークフロー（"ISR Verification"）が失敗し続けています。


* **影響**:
オーディオエンジン内の様々な同期リスク（Synchronization Risks）や前述のバグがCI側で検知されているものの、統合テストをパスできない状態になっており、**製品クオリティの安定性を自動担保できない状態**になっています。



---

### 🛠 次のステップへの推奨事項

1. **メモリ安全性（Bug 7, 8, 12）の優先解決**: リアルタイム処理において最も致命的なUAFおよび領域外アクセスを、ポインタのライフサイクルとRCUの解放トリガーを見直すことで最優先で修正してください。


2. **DSPの数式検証（Bug 10, 11, 13）**: ソフトクリッパーのSIMDレジスタ退避処理の修正、およびリミッターの補間式の連続性確保（C2連続な3次スプライン等への置き換え）を行ってください。