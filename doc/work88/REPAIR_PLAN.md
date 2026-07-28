# ConvoPeq 改修設計書 — BUG-011〜BUG-046 修正計画 (v20)

**凡例**: ✅ 実装完了 → Appendix 参照。📋 設計確定 → 「設計」セクション参照。
**ステータス**: **v20 ACK応答型・Shutdown優先度・Handle Fairness・QSVC通知・EpochWaiting注記を追加**。EnqueueResult enum 定義、QUEUE-9/10 追加、QSVC-1 拡充、DSPState設計注記。全15項目中6実装済み・9設計確定。

---

# 設計（確定版：残タスク9項目の設計）

## P1-1: FFT Backend Concept化 [🟡P1] — 詳細設計

### 1. 設計目的
`MKLNonUniformConvolver` の FFT 呼び出し（`ippsFFTFwd_RToCCS_64f` / `ippsFFTInv_CCSToR_64f`）を C++20 Concept ベースの抽象化によりテスト可能にする。virtual dispatch ゼロを維持し、RT パスへの影響ゼロを保証。Intel IPP 直呼び出しのままではエラー注入が不可能であり、FFT エラー時の異常系テスト（`clearFFTOutputOnError` の動作検証）を実装できない。

### 2. 現状分析
| 観点 | 現在 | 課題 |
|------|------|------|
| FFT API | `ippsFFTFwd_RToCCS_64f` / `ippsFFTInv_CCSToR_64f` 直呼び出し | テスト時にエラー注入不可 |
| Layer 構造体 | `IppsFFTSpec_R_64f* fftSpec` を直接保持 | IPP 型に依存 |
| エラー処理 | `clearFFTOutputOnError()` 実装済み（6箇所） | 異常系テストが未実装 |
| explicit instantiation | 未対応 | テンプレート導入後のバイナリ肥大リスク |
| テスト基盤 | GoogleTest/GMock なし（カスタムフレームワーク） | GMock 非依存の方針維持 |

### 3. FFT Backend Concept 定義

```cpp
template <typename FftBackend>
concept FftBackendConcept = requires(FftBackend& b, const double* in, double* out) {
    { b.forward(in, out) } -> std::same_as<IppStatus>;
    { b.inverse(in, out) } -> std::same_as<IppStatus>;
};
```

**Concept 採用の根拠**:
| 方式 | virtual | RT-safe | エラー注入 | 採用 |
|------|---------|---------|-----------|------|
| **Concept（静的ポリモーフィズム）** | ゼロ | ✅ | ✅ | **採用** |
| CRTP | ゼロ | ✅ | ✅ | 不採用（Concept で十分） |
| virtual + Mock | あり | ❌（RT不可） | ✅ | 不採用（GMock 不使用ポリシー違反） |
| 現状維持（直呼び出し） | ゼロ | ✅ | ❌ | 不採用（テスト不可能） |

### 4. ProductionFft — Intel IPP ラッパー

```cpp
class ProductionFft {
public:
    explicit ProductionFft(IppsFFTSpec_R_64f* spec) noexcept : fftSpec_(spec) {}

    IppStatus forward(const double* in, double* out) noexcept {
        return ippsFFTFwd_RToCCS_64f(in, out, fftSpec_, workBuf_);
    }

    IppStatus inverse(const double* in, double* out) noexcept {
        return ippsFFTInv_CCSToR_64f(in, out, fftSpec_, workBuf_);
    }

    void setWorkBuffer(Ipp8u* buf) noexcept { workBuf_ = buf; }

private:
    IppsFFTSpec_R_64f* fftSpec_;
    Ipp8u* workBuf_ = nullptr;
};

static_assert(FftBackendConcept<ProductionFft>);
```

**ProductionFft 契約**:
| ID | 契約 |
|----|------|
| FFT-PROD-1 | ProductionFft は `IppsFFTSpec_R_64f*` を保持してよい（所有はしない） |
| FFT-PROD-2 | spec の生成/破棄は NonRT（Builder/MessageThread）専用 |
| FFT-PROD-3 | `forward()` / `inverse()` は RT から呼び出し可能 |
| FFT-PROD-4 | `forward()` / `inverse()` は `noexcept` |
| FFT-PROD-5 | 失敗時は `IppStatus` を返す（エラーコード伝搬） |
| FFT-PROD-6 | RT 内で allocation / free / exception / log を発生させない |

### 5. TestFft — エラー注入可能なテスト用実装

```cpp
class TestFft {
public:
    IppStatus forward(const double*, double*) noexcept { return result_; }
    IppStatus inverse(const double*, double*) noexcept { return result_; }

    void setResult(IppStatus s) noexcept { result_ = s; }
    void setResultOnCall(IppStatus fwd, IppStatus inv) noexcept {
        resultForward_ = fwd; resultInverse_ = inv;
    }

private:
    IppStatus result_{ippStsNoErr};
    IppStatus resultForward_{ippStsNoErr};
    IppStatus resultInverse_{ippStsNoErr};
};

static_assert(FftBackendConcept<TestFft>);
```

**TestFft テストパターン**:
| パターン | setResult | 期待動作 |
|----------|-----------|---------|
| 正常系 | `ippStsNoErr` | 出力バッファ変更なし（通常処理継続） |
| 異常系 | `ippStsErr` | `clearFFTOutputOnError` が出力をゼロクリア |
| 部分異常 | forward→err, inverse→ok | Forward エラーのみ fail-closed |
| 復帰 | err→次の呼び出しで ok | エラー後もシステム継続 |

### 6. Layer 構造体のテンプレート化

**第一推奨: Layer 構造体単位のみテンプレート化**（クラス全体はコンパイル爆発のため非推奨）。

#### ISR改善提案: FFTExecutionContext 分離（将来課題）

ISR原則「RTは判断しない・所有しない」の観点では、理想的には `Layer` が FFT Backend を直接保持しない設計が望ましい。

```cpp
// 理想（将来設計）:
// Layer は FFT の存在すら知らない
struct Layer {
    // FFT 関連フィールドなし
    // ... オーディオ処理以外の責務のみ
};

// FFT Execution Context が Layer の FFT 処理を委譲される
class FFTExecutionContext {
    ProductionFft fft;
    void processLayer(Layer& layer, const double* input, double* output);
};
```

これにより `Layer` はオーディオデータのコンテナに専念し、FFT 演算は `FFTExecutionContext` が責務を持つ。ただし現状案（Layerテンプレート化）でも virtual ゼロ・RT-safe は保証されており、本フェーズでは現状案で十分妥当である。

```cpp
// ── 現行（IPP 直依存）──
struct Layer {
    IppsFFTSpec_R_64f* fftSpec    = nullptr;
    Ipp8u*             fftWorkBuf = nullptr;
};

// ── 変更後（テンプレート化）──
template <FftBackendConcept FftBackend>
struct Layer {
    FftBackend fft;              // ★ FFT backend インスタンス
    // fftSpec/fftWorkBuf は FftBackend 内部で管理
    // ... 他のメンバ（変更なし）
};
```

**変更影響範囲**:
| 構成要素 | 変更 | 備考 |
|----------|------|------|
| `Layer::fftSpec` | ❌ 削除 → `Layer::fft` に置換 | `FftBackend` 内部で管理 |
| `Layer::fftWorkBuf` | ❌ 削除 → `Layer::fft` に内包 | `ProductionFft::setWorkBuffer()` |
| Layer 内 FFT 呼び出し | ✅ FFT 直呼び → `l.fft.forward()` / `l.fft.inverse()` | 6箇所すべて変更 |
| `complexSize` 等の非FFTメンバ | 変更なし | |
| `fftPlanOwner` | 変更なし（FFT backend 生成時に使用） | |
| FFT spec 初期化（`SetImpulse()`） | ✅ `ProductionFft` を Layer に保存 | 初期化時のみ変更 |
| `areFftDescriptorsCommitted()` | ✅ backend 経由で確認 | |

**Layer テンプレート宣言**:
```cpp
// MKLNonUniformConvolverLayer.h
template <FftBackendConcept FftBackend = ProductionFft>
class MKLNonUniformConvolverLayer {
    Layer<FftBackend> m_layers[3];
};
```

### 7. Explicit Instantiation 戦略

```cpp
// MKLNonUniformConvolverLayer.h（header）
extern template class MKLNonUniformConvolverLayer<ProductionFft>;

// MKLNonUniformConvolverLayer.cpp（production TU）
template class MKLNonUniformConvolverLayer<ProductionFft>;

// テストファイル（test TU）— TestFft は暗黙の instantiation のみ
// extern template 宣言は test TU では #ifdef ガード等で除外
```

**Production 型のみ explicit instantiation**:
| Instantiation | 方法 | 含まれるビルド |
|-------------|------|--------------|
| `MKLNonUniformConvolverLayer<ProductionFft>` | explicit instantiation | Release + Debug |
| `MKLNonUniformConvolverLayer<TestFft>` | 暗黙（`extern template` 対象外） | Test TU のみ |

**コンパイル時間対策**:
| 対策 | 方法 |
|------|------|
| `extern template` | Header に宣言し複数 TU での重複インスタンス生成を防止 |
| Layer 単位のみ | クラス全体のテンプレート化を避ける |
| TEST ビルド分離 | Test TU は Release ビルドに含めない |

### 8. Fail-Closed 契約

| ID | 契約 |
|----|------|
| FFT-FAIL-1 | FFT が non-success を返したら出力をゼロクリアする |
| FFT-FAIL-2 | stage を ready にしない |
| FFT-FAIL-3 | stale な結果を publish しない |
| FFT-FAIL-4 | RT 内で retry しない |
| FFT-FAIL-5 | error flag/counter は atomic relaxed でよい |
| FFT-FAIL-6 | log は NonRT へ委譲する |

### 9. テスト計画

| テスト名 | 内容 |
|----------|------|
| `FftProductionInstantiation` | `ProductionFft` のみ Release binary に含まれる |
| `FftTestBackendInjection` | `TestFft` でエラー注入可能 |
| `FftForwardErrorFailClosed` | forward エラー時 fail-closed |
| `FftInverseErrorFailClosed` | inverse エラー時 fail-closed |
| `FftNoPublishOnError` | エラー時 publish なし |
| `FftNullSpecHandling` | null spec ハンドリング |
| `FftSizeMismatch` | サイズ不一致 |
| `FftAllSixSitesCovered` | 6箇所すべての FFT 呼び出しをカバー |

### 10. マイグレーションパス（段階的移行）

| Phase | 内容 | 成果物 | CI 通過 |
|-------|------|--------|---------|
| **Phase 1** | `FftBackendConcept` / `ProductionFft` / `TestFft` 定義 | 新規ヘッダ/ソース | ✅ 既存コード非変更 |
| **Phase 2** | `Layer` 構造体テンプレート化（クラス全体は非テンプレート） | `MKLNonUniformConvolverLayer.h` | ✅ コンパイル通過 |
| **Phase 3** | 既存 `Layer` → `Layer<ProductionFft>` 置換 + FFT 呼び出し6箇所変更 | コンパイル通過 | ✅ 全テスト通過 |
| **Phase 4** | Explicit instantiation + `extern template` | Production binary | ✅ バイナリ検証 |
| **Phase 5** | テスト追加（TestFft 注入） | 全テスト追加 | ✅ 異常系カバレッジ |

各 Phase は独立して CI 通過可能。Phase 1〜2 は既存コードに影響を与えない。

### 11. リスクと対策

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| テンプレートコード膨張 | LOW | MEDIUM | Layer 単位テンプレート化＋extern template |
| コンパイル時間増加 | LOW | LOW | extern template で最小化 |
| FFT パフォーマンス劣化 | LOW | HIGH | Concept はコンパイル時解決、runtime オーバーヘッドゼロ |
| RT パスへの意図しない影響 | LOW | CRITICAL | Phase 3 で RT パスのベンチマーク確認 |
| IPP バージョン非互換 | LOW | MEDIUM | `ProductionFft` が IPP 呼び出しをラップ、変更箇所を1箇所に閉じ込め |

### 12. 見積工数
設計0.5日＋実装1日＋テスト0.5日 = **2日**（既存設計から変更なし）

---

## ADD-4: ASan/TSan CI job分離 [🔷INFO] — 詳細設計

### 1. 設計目的
現在の `CMakeLists.txt` は `ENABLE_ASAN` オプションを持つが、Debug ビルド（`/MTd` 静的CRT）と ASan（`/MDd` 動的CRT必須）の CRT 非互換により同一 job で両立できない。ASan/TSan を Debug とは別 job に分離し、各 sanitizer の特性に応じた CRT 設定でビルド・テストを実行可能にする。

### 2. 現状分析

**現行の CMakeLists.txt ASan 設定**（L1037-1056）:
```cmake
option(ENABLE_ASAN "Enable AddressSanitizer (Debug only)" OFF)
if(ENABLE_ASAN)
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(ConvoPeq PRIVATE /fsanitize=address)
        target_link_options(ConvoPeq PRIVATE /fsanitize=address)
        set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
            "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(ConvoPeq PRIVATE -fsanitize=address)
        target_link_options(ConvoPeq PRIVATE -fsanitize=address)
        set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
            "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
endif()
```

**課題**:
| 課題 | 詳細 |
|------|------|
| CRT 非互換 | Debug は `/MTd`（静的CRT）、ASan は `/MDd`（動的CRT）必須 → 同一ターゲットで両立不可 |
| TSan 未対応 | `ENABLE_TSAN` オプションが存在しない |
| icx 非互換 | Intel oneAPI icx は ASan 対応するが、MSVC とは別のフラグが必要 |
| ビルド構成混在 | 同一 CMake 設定で複数の CRT を使い分ける仕組みがない |

### 3. CI Job Matrix

| Config | CRT | Sanitizer | Compiler | 目的 |
|--------|-----|-----------|----------|------|
| Debug | `/MTd` | なし | MSVC cl.exe | 既存 Debug ビルド（静的CRT、ASan なし） |
| Debug-ASan | `/MDd` | AddressSanitizer | MSVC cl.exe | メモリ安全性検証（新規） |
| Debug-TSan | dynamic CRT | ThreadSanitizer | Clang (WSL/Linux) | データ競合検出（新規） |
| Release | `/MT` | なし | MSVC cl.exe | 既存 Release ビルド |
| Release-PGO | `/MT` | なし | MSVC cl.exe | PGO 最適化ビルド |

### 4. CMake 変更計画

**4.1 新オプション追加**:
```cmake
option(ENABLE_ASAN "Enable AddressSanitizer (Debug ASan job)" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer (Debug TSan job, Clang only)" OFF)
```

**4.2 前提条件チェック**:
```cmake
if(ENABLE_ASAN AND ENABLE_TSAN)
    message(FATAL_ERROR "ASan and TSan are mutually exclusive. Enable only one.")
endif()

if(ENABLE_TSAN AND MSVC)
    message(FATAL_ERROR "TSan requires Clang (MSVC not supported). Use Clang or WSL Clang.")
endif()
```

**4.3 CRT 自動切替ロジック**:
```cmake
if(ENABLE_ASAN)
    # ASan 必須: 動的 CRT（/MDd for Debug, /MD for Release）
    # 静的 CRT（/MT /MTd）は MSVC ASan と非互換（LNK2038）
    set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

if(ENABLE_TSAN)
    # TSan は Clang の -fsanitize=thread で有効化
    target_compile_options(ConvoPeq PRIVATE -fsanitize=thread)
    target_link_options(ConvoPeq PRIVATE -fsanitize=thread)
    set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
```

**4.4 互換性マトリックス**:
| 設定 | MSVC cl | Intel icx | Clang |
|------|---------|-----------|-------|
| ENABLE_ASAN=ON | ✅ `/fsanitize=address` + `/MDd` | ✅ `-fsanitize=address` + `/MD` | ✅ `-fsanitize=address` |
| ENABLE_TSAN=ON | ❌ 未対応 | ⚠️ 実験的 | ✅ `-fsanitize=thread` |
| 両方 OFF（通常） | ✅ `/MT` `/MTd` | ✅ `/MT` | ✅ |

### 5. GitHub Actions Workflow 設計

```yaml
jobs:
  debug:
    runs-on: windows-latest
    steps:
      - run: cmake -B build -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
      - run: cmake --build build --config Debug
      - run: ctest -C Debug --output-on-failure

  debug-asan:
    runs-on: windows-latest
    steps:
      - run: cmake -B build -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DENABLE_ASAN=ON
      - run: cmake --build build --config Debug
      - run: ctest -C Debug --output-on-failure

  debug-tsan:
    runs-on: ubuntu-latest  # WSL Clang または Linux Clang
    steps:
      - run: cmake -B build -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DENABLE_TSAN=ON
      - run: cmake --build build --config Debug
      - run: ctest -C Debug --output-on-failure
```

### 6. TSan 対応方針

| 項目 | 方針 |
|------|------|
| **コンパイラ** | Clang 14+（MSVC 非対応）。WSL/Linux 上で実行 |
| **CRT** | 動的 CRT が必要（静的 CRT は TSan 非対応） |
| **CMake 設定** | `-fsanitize=thread` を compile/link flags に追加 |
| **除外リスト** | TSan の false positive 抑制用 `.tsanignore` ファイルを用意 |
| **実行環境** | WSL Ubuntu または CI Linux runner |

### 7. テスト戦略

| Job | 実行テスト | 期待結果 |
|-----|-----------|---------|
| Debug | 全テスト（通常） | 既存カバレッジ維持 |
| Debug-ASan | メモリ関連テスト全般 | use-after-free / buffer overflow / leak ゼロ |
| Debug-TSan | マルチスレッドテスト中心 | data race ゼロ |

**ASan 期待検出項目**:
- use-after-free（Retired DSP へのアクセス）
- heap-buffer-overflow（FFT バッファ境界）
- stack-buffer-overflow（AudioSegmentBuffer 等）
- memory leak（FFT spec / plan リーク）

**TSan 期待検出項目**:
- Data race on `retiredBuffer`（SafeStateSwapper）
- Data race on `pendingReceipt_`（AudioEngine）
- Lock inversion（fallbackQueue mutex + 他ロック）

### 8. リスクと対策

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| ASan の CRT 切替で他の設定と競合 | LOW | HIGH | `ENABLE_ASAN` 時のみ CRT 切替、通常時は変更なし |
| TSan の false positive | MEDIUM | LOW | 除外リスト `.tsanignore` で抑制 |
| CI ランナー増加によるコスト | MEDIUM | LOW | 対象テストを絞り込む（全テスト不要） |
| icx の ASan 非互換 | LOW | MEDIUM | icx では `ENABLE_ASAN` を OFF にする構成を推奨 |
| Clang/MSVC の挙動差 | LOW | LOW | TSan job は Linux Clang で実行、結果の解釈は別評価 |

### 9. 見積工数
CMake 変更0.5日 + CI workflow 設定0.5日 + 動作検証0.5日 = **1.5日**（設計詳細化による精度向上）

---

## P0-2: RefCountedDeferred→DSPHandleRuntime移行 [🔴P0] — 設計確定

### 現状
| 観点 | 状態 |
|------|------|
| DSPHandleRuntime 基盤 | ✅ 実装済み（create/resolve/retire/quarantine/reclaim full lifecycle） |
| PublishReceipt handle 保持 | ✅ 実装済み（DSPCore* と過渡的に併存） |
| storeReceipt() handle 受取 | ✅ 実装済み |
| EQCoeffCache 移行 | ❌ 未着手（RefCountedDeferred 継承が残存） |
| tryAddRef() 呼び出し元 | ✅ Dead Code 確認済み |

### 完了条件
1. `EQCoeffCache` が `RefCountedDeferred<EQCoeffCache>` を継承しない
2. `EQCoeffCache` が `DSPHandleRuntime::create()` で生成される
3. `EQCoeffCache` への全アクセスが `DSPHandleRuntime::resolve()` 経由
4. `tryAddRef()` が削除されている（呼び出し元ゼロ確認済み）
5. `RefCountedDeferred` テンプレート全体が Dead Code として削除/非推奨化
6. `PublishReceipt` から `DSPCore* dsp` が削除され、`DSPHandle` のみになる

### 移行手順
**Phase 1: EQCoeffCache の Handle 化（Coordinator 経由）**
- `EQCoeffCache` の基底を `RefCountedDeferred<EQCoeffCache>` から通常の struct に変更
- コンストラクタを private にし、Builder が生成を担当
- **ISR Authority**: `EQCoeffCache` 自身は `DSPHandleRuntime` を知らない
- Builder → Coordinator → `DSPHandleRuntime::create()` の流れで Handle 登録:
  ```cpp
  // Builder が生成
  auto cache = std::make_unique<EQCoeffCache>(...);
  // Coordinator が Handle 登録（唯一の Authority）
  DSPHandle handle = coordinator.registerDSPHandle(cache.get());
  // Builder は Handle を保持し、以降は resolve() 経由でアクセス
  ```
- 呼び出し側は `DSPHandle` 経由で `resolve()` → 使用 → `retire()` のライフサイクルに変更

**Phase 2: 既存参照箇所の置換**
- `EQProcessor` 内で `EQCoeffCache*` 直接保持 → `DSPHandle` を保持
- アクセス時に `handleRuntime_.resolve(handle)` でポインタ取得
- epoch protection scope 内で使用後、即座に参照を手放す

**Phase 3: Dead Code 削除**
- `tryAddRef()` の削除（呼び出し元ゼロ確認済み）
- `RefCountedDeferred` テンプレートの deprecated 化
- `PublishReceipt::dsp` フィールドの削除（`DSPHandle` に一本化）

### テスト計画
| テスト | 内容 |
|--------|------|
| EQCoeffCacheCreateResolve | `create()` + `resolve()` で正しくポインタ取得 |
| EQCoeffCacheRetireReclaim | `retire()` 後 `resolve()` が nullptr を返す |
| EQCoeffCacheLifecycle | 生成→使用→退役の完全ライフサイクル |
| PublishReceiptHandleOnly | `DSPCore* dsp` 削除後も機能する |
| RefCountedDeferredLegacy | `RefCountedDeferred` が使用されていない |

### リスクと対策
| リスク | 対策 |
|--------|------|
| EQCoeffCache 生成頻度高で Handle Table 溢れ | `kMaxSlots=256` 上限確認。溢れる場合は拡張 |
| resolve() 失敗時の fallback なし | 呼び出し元が nullptr チェック + デフォルト係数使用 |

### 見積工数
実装0.5日 + テスト0.5日 = **1日**

**契約**: REFCOUNT-1〜4（RefCountedDeferred は Legacy、新規禁止）
**HANDLE-5/6/7**: 本セクション完了条件に集約。詳細は Appendix FIX-P1-2 参照。

---

## P1-2: Stale receipt quarantine [🟡P1] — 設計確定

### 現状
| 観点 | 状態 |
|------|------|
| PublishReceipt DSPHandle 保持 | ✅ 実装済み（`AudioEngine.h:4338`） |
| storeReceipt() パラメータ | ✅ DSPHandle 受取済み（`AudioEngine.h:1142`） |
| DSPHandleRuntime::quarantine() | ✅ 実装済み（`ISRDSPHandle.cpp:120-125`） |
| Emergency quarantine 呼び出し | ✅ AudioEngine.Timer.cpp:1788-1793 |
| Receipt 状態機械 | ❌ 未実装 |
| resetReceipt() 関数 | ❌ 未実装（コード内に存在せず） |

### Receipt 状態機械（確定）

```
Empty ──storeReceipt()──→ Ready ──normal retire──→ Consumed
                            │
                            ├──stale/emergency/mismatch──→ StaleExported
                            │                                 │
                            │                          quarantine 実行
                            │                                 │
                            │                                 ▼
                            │                     Quarantined（DSPState と一致）
                            │                     （自動 free しない）
                            │                                 │
                            │                          DestroyPending
                            │                                 │
                            │                          Reclaimed
                            │
                            └──resetReceipt()──→ Empty（通常のリセット）
```

**状態定義**（DSPState enum と整合）:
| Receipt状態 | 対応 DSPState | 意味 | 遷移条件 |
|------------|--------------|------|---------|
| `Empty` | — | receipt なし | `storeReceipt()` 成功 |
| `Ready` | `Active` | receipt 有効、retire 待機 | Normal Retire 成功、または異常検出 |
| `StaleExported` | `Retired` | evidence 出力完了 | quarantine 実行 → `Quarantined` |
| `Quarantined` | `Quarantined` | DSP 隔離済み、自動 free しない | shutdown drain → `DestroyPending` |
| `DestroyPending` | `DestroyPending` | 解放予約状態 | reclaim → `Reclaimed` |
| `Reclaimed` | `Reclaimed` | メモリ解放済み | — |
| `Consumed` | `Retired` | 正常 retire 完了（Quarantine不要） | — |

### ISR Authority: Quarantine Intent パターン

ISR原則「Coordinatorのみが唯一のAuthority」に従い、`resetReceipt()` は直接 quarantine を実行せず、
Coordinator に「隔離要求 (QuarantineIntent)」を発行する。Coordinator が ACK を返してから
`pendingReceipt_` を解放する。

```
resetReceipt()
  │
  ├──emitQuarantineIntent(handle, reason)──→ Coordinator
  │                                              │
  │                                        quarantine() 実行
  │                                        QuarantineManager 記録
  │                                              │
  │                                        Coordinator ACK
  │                                              │
  │←─────────── ACK ─────────────────────────────┘
  │
  └──pendingReceipt_.reset()
      receiptReady_ = false
```

#### resetReceipt() 関数設計（修正版）
```cpp
void resetReceipt() noexcept {
    if (pendingReceipt_.has_value() && !pendingReceipt_->handle.isNull()) {
        // ★ ISR: Receipt は quarantine を直接実行せず、Intent を発行する
        coordinator_.emitQuarantineIntent(
            pendingReceipt_->handle,
            convo::isr::QuarantineReason::ReceiptReset);
        // Coordinator ACK を待つ（NonRT パスなので同期的に待機可能）
        // Coordinator が quarantine + audit を完了後、ACK が返る
    }
    // Coordinator ACK 後に pendingReceipt_ を解放
    pendingReceipt_.reset();
    receiptReady_.store(false, std::memory_order_relaxed);
}
```

### 呼び出し箇所
- `onPublishCompleted()` Emergency パス → quarantine 呼び出し済み（`AudioEngine.Timer.cpp:1788-1793`）
  - ★ ISR改善: 直接 quarantine → `emitQuarantineIntent()` 経由に変更
- `retirePublishedDSP()` Normal Retire 完了後 → `pendingReceipt_.reset()` 追加
- Shutdown 時 → `DSPQuarantineManager::destroyForShutdown()` で drain

### 完了条件
1. `resetReceipt()` 関数が実装されている
2. Receipt 状態機械の全遷移がコード上で表現されている
3. Emergency quarantine + DSPHandleRuntime の協調が完了している
4. 全既存テスト通過＋新規テスト追加

### テスト計画
| テスト | 内容 |
|--------|------|
| ReceiptStaleExport | stale receipt → evidence export 遷移 |
| ReceiptQuarantineTransition | Ready → StaleExported → Quarantined |
| ReceiptResetDoesNotDropRetireObligation | reset → quarantine 呼び出し確認 |
| ReceiptEmergencyOverride | Emergency → quarantine 呼び出し確認 |
| ReceiptShutdownDrain | shutdown drain 確認 |

### 見積工数
実装0.5日 + テスト0.5日 = **1日**

---

## ADD-2: MMCSS AvRevert例外登録 [🔷INFO] — 設計確定

### 現状
`AudioEngine.Mmcss.cpp:201-208` に `revertMmcssOnAudioThread()` 実装済み。
Coding rule では Audio Thread 内の MMCSS API 呼び出しを禁止しているが、ASIO thread entry を
フックできない場合のみ例外として許可する設計。

### 対応内容
以下の例外登録簿を作成し、`coding_rule_jp.txt` または `doc/` 配下に配置する。

**例外登録簿**:
| # | 機能 | ファイル | 行 | 理由 | 承認日 |
|---|------|---------|----|------|-------|
| 1 | `AvRevertMmThreadCharacteristics` | `AudioEngine.Mmcss.cpp` | 201-208 | ASIO thread entry 非フック可能時のみ例外許可。`thread_local` ガードにより一度だけ実行 | 2026-07-28 |

**MMCSS-EX 契約**:
```text
MMCSS-EX-1: Audio Thread 内での MMCSS API 呼び出しは、ASIO thread entry を
            フックできない場合のみ例外として許可する。
MMCSS-EX-2: 呼び出しは thread_local guard により一度だけとする。
MMCSS-EX-3: RT 内で log しない（#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS でガード済み）。
MMCSS-EX-4: 失敗しても音声を止めない。
MMCSS-EX-5: 例外登録簿に記載する。
```

### 完了条件
1. 例外登録簿が作成されている
2. MMCSS-EX-1〜5 が記載されている
3. コード変更は不要（既存実装で完了）

### 見積工数
30分（文書更新のみ）

---

## ISR Authority 整理 — Coordinator 唯一 Authority 原則の徹底

本設計書の全5項目を ISR「Coordinator が唯一の Authority」原則で評価し、整理する。

### 現状の Authority マップ

| 操作 | 現在の Authority | ISR 理想 | 状態 |
|------|-----------------|----------|------|
| FFT Backend 選択 | `MKLNonUniformConvolver`（Layer テンプレート） | 同左（Concept は Authority ではない） | ✅ |
| Handle 生成 | `DSPHandleRuntime::create()`（誰でも呼べる） | Coordinator 経由 | ⚠️ P0-2 で修正 |
| Handle 解決 | `DSPHandleRuntime::resolve()`（誰でも呼べる） | 同左。ただし返されたポインタは **Epoch保護下でのみ有効**。保護なしでの保持はISR違反 | ⚠️ 契約明示が必要 |
| Handle 退役 | `DSPHandleRuntime::retire()` → Coordinator | Coordinator が唯一 | ✅ 既存 |
| Quarantine | `resetReceipt()` が直接 `quarantine()` | Coordinator が Intent 受信→実行 | ⚠️ P1-2 で修正 |
| Receipt 解放 | `resetReceipt()` が即時 `pendingReceipt_.reset()` | Coordinator ACK 後 | ⚠️ P1-2 で修正 |
| MMCSS 例外 | `revertMmcssOnAudioThread()`（thread_local 制御） | 同左（技術的制約、文書管理） | ✅ |

### 改善反映状況

| # | 改善項目 | 反映先 | ステータス |
|---|---------|--------|-----------|
| 1 | `EQCoeffCache` → Coordinator 経由 Handle 登録 | P0-2 Phase 1 | ✅ 本設計書で修正 |
| 2 | `resetReceipt()` → Quarantine Intent パターン | P1-2 resetReceipt() | ✅ 本設計書で修正 |
| 3 | Coordinator ACK → `pendingReceipt_.reset()` | P1-2 resetReceipt() | ✅ 本設計書で修正 |
| 4 | Coordinator を Intent Dispatcher 化（Queue + Loop） | P0-4 (§A) | ✅ 本設計書で修正 |
| 5 | ACK 定義（queued / retire complete / reclaim complete） | P0-4 (§A, §B) | ✅ 本設計書で修正 |
| 6 | Retire→Epoch→Reclaim 契約（waitReaders 必須化） | P0-4 (§B) | ✅ 本設計書で修正 |
| 7 | 状態機械を DSPState enum と統一 | P1-2 | ✅ 本設計書で修正 |
| 8 | `resolve()` Epoch 保護契約の明文化 | ISR Authority マップ | ✅ 本設計書で修正 |
| 9 | FFTExecutionContext 分離（Layer が FFT を知らない） | P1-1 改善提案 | 📋 将来課題 |

### resolve() の Epoch 保護契約

`DSPHandleRuntime::resolve(handle)` は Read-only であり Coordinator を介す必要はないが、
返されたポインタは **Epoch 保護下でのみ有効**である。以下の契約を追加する。

| ID | 契約 |
|----|------|
| RESOLVE-1 | `resolve()` は Read-only、Coordinator 不要。いつでも呼び出し可能 |
| RESOLVE-2 | `resolve()` が返したポインタは Epoch 保護 scope（`enterReader()`/`exitReader()` または RAII Guard）内でのみ有効 |
| RESOLVE-3 | Epoch 保護を解除した後はポインタを逆参照してはならない |
| RESOLVE-4 | `resolve()` 成功 ≠ オブジェクトの有効性保証。`resolve()` 時点の generation 一致のみ確認 |

### 未対応の Authority 課題（長期）

| 課題 | 理由 | 対応時期 |
|------|------|---------|
| `EQCoeffCache` の完全な Coordinator 管理化 | Builder 層の設計が未着手 | P0-2 実装フェーズで対応 |
| `DSPHandleRuntime` の Coordinator ラッパー統一 | 現状は直接利用可能（既存コードとの整合性） | 別タスク |
| Receipt 状態機械の Coordinator 統合 | 状態遷移の責務は Receipt が保持 | 現状維持 |

### ISR 寿命管理パイプラインとの比較

Practical ISR Bridge Runtime が定義する寿命管理パイプライン:

```
Publish → Observe → Retire Authority → Epoch → Delete
```

現在の設計における各ステップの対応を評価する。

| ISR パイプライン | 現在の設計での実現 | Coordinator 経由 | 評価 |
|-----------------|-------------------|-----------------|------|
| **Publish** | `DSPTransition::storeReceipt()` + `publishAtomic()` | ✅ DSPTransition 内で完結 | ✅ |
| **Observe** | `retirePublishedDSP()` Timer 定期観測 | ⚠️ Timer 直接呼び出し | △ Coordinator 経由が望ましい |
| **Retire Authority** | `DSPHandleRuntime::retire()` + `DSPLifetimeManager::retire()` | ⚠️ 一部 Coordinator 経由 | △ 統一されていない |
| **Epoch** | `publicationEpochDistance()` epoch 差分検出 | ✅ 状態機械内で完結 | ✅ |
| **Delete** | `DSPHandleRuntime::reclaim()` | ⚠️ 直接呼び出し可 | △ Coordinator ラッパー不足 |

**残課題**: ISR パイプラインの `Observe`・`Retire Authority`・`Delete` の3ステップは、
現在 Coordinator を経由しない経路が存在する。本設計書の P1-2 (Quarantine Intent) は
`Retire Authority` の一部を Coordinator 経由に変更したが、完全な統合には以下が必要:

1. `retirePublishedDSP()` の Timer→Coordinator 委譲（Observe Authority）
2. `DSPHandleRuntime::reclaim()` の Coordinator 専用化（Delete Authority）
3. Coordinator に `emitRetireIntent()` / `emitReclaimIntent()` インターフェース追加

**→ 上記3項目は本設計書の P0-4 として新規追加。以下のセクションで設計確定。**

---

## P0-4: ISR Coordinator 経由寿命管理（Observe/Delete Authority）[🔴P0] — 設計確定

### 概要
Practical ISR Bridge Runtime の `Publish → Observe → Retire Authority → Epoch → Delete` パイプラインのうち、
Observe・Delete の2 Authority が Coordinator を経由していない問題を修正する。

### 現状の Authority 分析

| パイプライン | 現在 | 問題 |
|-------------|------|------|
| **Observe** | `retirePublishedDSP()` が Timer から直接呼び出される | Coordinator が観測タイミングを制御できない |
| **Delete** | `DSPHandleRuntime::reclaim()` が直接呼び出し可能 | Coordinator が解放タイミングを保証できない |
| **Interface** | Coordinator に emitRetire/emitReclaim がない | 委譲先のインターフェースが存在しない |

---

### A. Observe Authority 整備

#### 設計
```
Timer callback
  │
  ├── emitObserveIntent() ──→ Coordinator
  │                              │
  │                        Intent Queue に追加
  │                              │
  │←────── ACK (queued) ────────┘
  │
  └──（Timer は即時復帰）

Coordinator Loop (別スレッド):
  Intent Queue から取り出し
       │
  processIntent()
       │
  retirePublishedDSP() 実行
  （Normal/Fallback/Emergency 判定）
       │
  DSPHandleRuntime::retire()
       │
  Epoch 安全確認（waitReaders）
       │
  executeReclaim()
       │
  ACK (reclaim complete)
       │
  pendingReceipt_.reset()
```

Timer は「観測要求 (ObserveIntent)」をキューに追加するのみ。Coordinator は独立したループで
Intent を処理する。これにより Coordinator は「Facade」ではなく「Intent Dispatcher」として機能する。

**ACK 定義**:
| ACK種別 | 意味 | 発行タイミング |
|---------|------|--------------|
| `ACK (queued)` | Intent がキューに追加された | emitObserveIntent() 直後 |
| `ACK (reclaim complete)` | Retire + Epoch待機 + Reclaim 完了 | executeReclaim() 完了後 |

#### 変更内容
| ファイル | 変更 |
|---------|------|
| `AudioEngine.Timer.cpp` | `retirePublishedDSP()` 直接呼び出し → `coordinator_.emitObserveIntent()` |
| `ISRRuntimePublicationCoordinator.h` | `emitObserveIntent()` 宣言追加（`audioengine/` 配下。`core/` 版はテンプレート版のため変更しない） |
| `ISRRuntimePublicationCoordinator.cpp` | `emitObserveIntent()` 実装（内部で `retirePublishedDSP()` を Intent 経由で起動） |

#### 契約
| ID | 契約 |
|----|------|
| OBSERVE-1 | Timer は ObserveIntent のみ発行し、Retire Authority を直接実行しない |
| OBSERVE-2 | Coordinator は ObserveIntent を Intent Queue に追加し、即時復帰する（Timer をブロックしない） |
| OBSERVE-3 | Coordinator Loop は Intent Queue から取り出した Intent を `processIntent()` で処理する |
| OBSERVE-4 | Coordinator は Normal/Fallback/Emergency の3分類を判定する |
| OBSERVE-5 | `Coordinator::ACK(queued)` = Intent がキューに追加されたことのみ保証。処理未完了でも発行される |
| OBSERVE-6 | `Coordinator::ACK(reclaim complete)` = Retire + Epoch待機 + Reclaim の全完了を保証 |
| OBSERVE-7 | Timer は `ACK(reclaim complete)` 受信後に pendingReceipt_ を安全に解放する |
| OBSERVE-8 | ObserveIntent は NonRT パス（Timer Thread）からのみ発行可能 |

---

### B. Delete Authority 整備

#### 設計
```
（現状）
DSPHandleRuntime::reclaim(handle)  ← 誰でも呼べる

（変更後）
Coordinator 完全ライフサイクル:
requestReclaim(handle)  ← 外部からの要求受付
  │
  ├── Intent Queue に追加
  └── ACK (queued)

Coordinator Loop:
processIntent()
  │
  ├── executeRetire(handle)
  │     │
  │     ├── DSPHandleRuntime::retire()
  │     ├── State → Retired
  │     └── (Epoch 待機開始)
  │
  ├── waitReaders(handle)  ★ Epoch 安全確認（必須）
  │     │
  │     ├── 全 Reader が epoch 保護を解放するまで待機
  │     └── State → DestroyPending（ISR: Retire→Epoch間の分離）
  │
  ├── executeReclaim(handle)  ← Coordinator のみ
  │     │
  │     ├── DSPHandleRuntime::reclaim()
  │     └── State → Reclaimed
  │
  └── ACK (reclaim complete)

【ISR不変条件】executeRetire() と executeReclaim() の間には
必ず waitReaders()（Epoch 安全確認）が入る。直接 executeReclaim()
を呼んではならない。

`DSPHandleRuntime::reclaim()` を Coordinator 専用の内部メソッドに変更し、
外部からは `Coordinator::requestReclaim()` 経由でのみ間接的に呼び出せるようにする。
```

`DSPHandleRuntime::reclaim()` を Coordinator 専用の内部メソッドに変更し、
外部からは `Coordinator::requestReclaim()` 経由でのみ呼び出せるようにする。

#### 変更内容
| ファイル | 変更 |
|---------|------|
| `ISRDSPHandle.h` | `reclaim()` を private または Coordinator フレンドに変更 |
| `ISRDSPHandle.cpp` | `reclaim()` のアクセス制御変更 |
| `ISRRuntimePublicationCoordinator.h` | `requestReclaim(handle)` 宣言追加（`audioengine/` 配下。`core/` 版は非対象） |
| `ISRRuntimePublicationCoordinator.cpp` | `requestReclaim()` 実装（epoch確認→waitReaders→reclaim→ACK） |

#### 契約
| ID | 契約 |
|----|------|
| DELETE-1 | `DSPHandleRuntime::reclaim()` は Coordinator 専用。外部から直接呼び出し禁止 |
| DELETE-2 | `executeRetire()` と `executeReclaim()` の間には必ず `waitReaders()` を挿入する（ISR不変条件） |
| DELETE-3 | `requestReclaim()` は epoch 安全確認後にのみ reclaim を実行する |
| DELETE-4 | 全 reader が epoch 保護を解放したことを確認してから reclaim する |
| DELETE-5 | reclaim 完了後 Coordinator は `ACK(reclaim complete)` を返す |
| DELETE-6 | `executeRetire()` 完了時の `ACK(retire committed)` は Epoch 待機完了を保証しない |
| DELETE-7 | shutdown 時のみ Coordinator をバイパスした強制 reclaim を許可（`DSPQuarantineManager::destroyForShutdown()`） |

---

### C. Coordinator Interface 拡充

#### 設計
以下のインターフェースを `ISRRuntimePublicationCoordinator`（`audioengine/ISRRuntimePublicationCoordinator.h`）に追加する。
`core/RuntimePublicationCoordinator`（テンプレート版）は対象外。

```cpp
class RuntimePublicationCoordinator {    // audioengine/ISRRuntimePublicationCoordinator.h
public:
    // ★ ISR: Intent 発行（外部 → Coordinator）
    //     Intent はキューに追加され、Coordinator Loop が非同期に処理する
    void emitObserveIntent() noexcept;        // Timer → Coordinator
    void emitRetireIntent(DSPHandle handle,
        RetireReason reason) noexcept;         // 汎用 Retire Intent
    void emitQuarantineIntent(DSPHandle handle,
        QuarantineReason reason) noexcept;     // Quarantine Intent
    void requestReclaim(DSPHandle handle) noexcept;  // Reclaim Request

private:
    // ★ Intent Dispatcher（Coordinator Loop から呼ばれる）
    void processIntent() noexcept;

    // Coordinator 専用メソッド（外部から直接呼び出し禁止）
    void executeRetire(DSPHandle handle);      // → Retired state
    void executeReclaim(DSPHandle handle);     // → Reclaimed state
    void executeQuarantine(DSPHandle handle);  // → Quarantined state

    // ★ Epoch 安全確認（ISR: Retire→Reclaim 間の必須ステップ）
    bool waitReaders(DSPHandle handle) noexcept;

    // Intent Queue
    std::queue<Intent> intentQueue_;
    static constexpr size_t kMaxIntentQueueSize = 256;  // ★ 容量上限
};
```

**注: Intent 種類の整理**:
現状は `emitObserveIntent` / `emitRetireIntent` / `emitQuarantineIntent` / `requestReclaim`
の4種類がある。ISR的には以下の単一 Intent で表現可能:
```
LifecycleIntent { type: Observe | Retire | Quarantine | Shutdown, handle, reason }
```
現状でも問題ないが、API数を削減したい場合は将来 `emitIntent(LifecycleIntent)` に統一してもよい。

### Intent Queue 契約

| ID | 契約 |
|----|------|
| QUEUE-1 | Intent Queue は **FIFO**（追加順に処理）。Publish→Observe→Retire→Epoch→Delete の全順序を保証する |
| QUEUE-2 | Queue は **Epoch 順** も保証する。同一 DSP に対する複数 Intent は Epoch の昇順に処理される |
| QUEUE-3 | **Handle 単位の直列化**: 同一 Handle に対する Intent は必ず逐次処理される。Handle A の Retire と Handle B の Observe は並行処理可能。Handle A の Retire 完了前に Handle A の Reclaim が開始されることはない |
| QUEUE-4 | `kMaxIntentQueueSize = 256` を超えた場合、最も古い Intent から破棄し HealthEvent を発火する（Backpressure） |
| QUEUE-5 | Queue 溢れが発生した場合、Coordinator は新規 Intent の受付を停止してよい。このとき emitter には `EnqueueResult::QueueFull` を返す。ACK は `EnqueueResult` enum（Accepted, QueueFull, ShuttingDown の3値）とする |
| QUEUE-6 | **Intent 冪等性**: 同一 DSP が既に `Retired` 状態にある場合、それに対する ObserveIntent / RetireIntent は無視する（2重 Retire 防止） |
| QUEUE-7 | 同一 Handle に対する ObserveIntent が重複して投入された場合、Coordinator は最新の 1 件のみを処理する（重複排除） |
| QUEUE-8 | Shutdown 時は Queue を **Drain** する（未処理 Intent を破棄せず、全て処理してから停止する） |
| QUEUE-9 | **Shutdown Intent 優先**: ShutdownIntent は Queue 内で最も優先度が高い。FIFO 順序を無視して即座に処理され、Coordinator Loop の停止シーケンスに移行する |
| QUEUE-10 | **Handle 間 Fairness**: Queue はラウンドロビン方式で Handle を選択する。同一 Handle の連続処理は最大 `kMaxConsecutivePerHandle = 4` 件まで。以降は別 Handle の Intent を処理してから再開する |

### Coordinator Loop 停止シーケンス

```
Shutdown 要求
  │
  ├── Intent Queue への新規追加を禁止
  │     └── 以降の emitObserveIntent/emitRetireIntent は ACK(shutdown) を返す
  │
  ├── Queue Drain: 残存 Intent を全て処理
  │     │
  │     ├── ObserveIntent → retire + reclaim（通常通り）
  │     ├── RetireIntent  → retire（epoch待機なし、即時 reclaim）
  │     ├── ReclaimIntent → reclaim（epoch確認省略）
  │     ├── QuarantineIntent → setState(Quarantined) + audit（即時）
  │     └── ShutdownIntent → 即時 shutdown（Drain中断）
  │
  ├── Coordinator Loop 停止
  └── 未処理の強制 reclaim（DSPQuarantineManager::destroyForShutdown）
```

### Quarantine 統一サービス

→ **P0-5 として本改修計画に追加。以下を参照。**

---

### D. 完了条件

1. `retirePublishedDSP()` が Timer から直接呼ばれず、Coordinator 経由になった
2. `DSPHandleRuntime::reclaim()` が Coordinator 専用になった
3. Coordinator に `emitObserveIntent()` / `emitRetireIntent()` / `requestReclaim()` / `emitQuarantineIntent()` が追加された
4. 既存の全テストが通過する
5. 新規テストが追加されている

### E. テスト計画

| テスト | 内容 |
|--------|------|
| ObserveIntentTimerFlow | Timer → emitObserveIntent → Coordinator → retire → ACK |
| DeleteAuthorityRestricted | reclaim() が Coordinator 以外から呼べない |
| DeleteAuthoritySafeFlow | requestReclaim → epoch確認 → reclaim → ACK |
| CoordinatorInterfaceContract | 4インターフェースの正常系 |
| ShutdownBypassReclaim | shutdown 時の強制 reclaim が動作する |

### F. リスクと対策

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| Coordinator への集中でレイテンシ増加 | LOW | MEDIUM | emitObserveIntent は非同期設計。ACK 待ちは NonRT パスのみ |
| reclaim() のアクセス制限で既存コード破損 | MEDIUM | HIGH | 全呼び出し元を洗い出し（AiDex/grep）、Coordinator 経由に置換 |
| Timer パスの変更でタイミング問題 | LOW | MEDIUM | 契約 OBSERVE-1〜5 で保護。Timer は Intent 発行のみに責務限定 |

### G. 見積工数

| 項目 | 工数 |
|------|------|
| Observe Authority 整備 | 実装0.5日 + テスト0.5日 |
| Delete Authority 整備 | 実装0.3日 + テスト0.3日 |
| Coordinator Interface 拡充 | 実装0.3日 + テスト0.3日 |
| 統合テスト・検証 | 0.5日 |
| **合計** | **2.0日** |

---

## P0-5: QuarantineService 統一 — 二重 Authority 解消 [🔴P0] — 設計確定

### 問題
`DSPHandleRuntime::quarantine()` と `DSPQuarantineManager::quarantineHandle()` は
独立した2機構であり、Coordinator が「State変更」と「Audit登録」の両方を直接呼ぶ必要がある。
これは Authority Singularization に反する。

### 設計
`QuarantineService` を導入し、2機構を単一 Authority に統合する。

```cpp
class QuarantineService {
public:
    void quarantine(DSPHandle handle, QuarantineReason reason) noexcept
    {
        dspHandleRuntime_.setSlotState(handle.slot, DSPState::Quarantined);
        dspQuarantineManager_.quarantineHandle(
            handle.slot, handle.generation, reason);
    }
    void unquarantine(DSPHandle handle) noexcept;
    bool isQuarantined(DSPHandle handle) const noexcept;
private:
    DSPHandleRuntime& dspHandleRuntime_;
    DSPQuarantineManager& dspQuarantineManager_;
};
```

### 変更内容
| ファイル | 変更 |
|---------|------|
| `新規: QuarantineService.h` | クラス定義 |
| `新規: QuarantineService.cpp` | 実装 |
| `AudioEngine.Timer.cpp:1788-1793` | `QuarantineService::quarantine()` に置換 |
| `AudioEngine.Threading.cpp:36-61` | `QuarantineService::quarantine()` に置換 |

### 契約
| ID | 契約 |
|----|------|
| QSVC-1 | State変更 + Audit を単一トランザクションとして実行。失敗時は State をロールバック。ロールバックも失敗した場合は `HealthEvent::QuarantineServiceFailure` を発火し、Coordinator は該当 Handle を Fatal 状態に遷移する |
| QSVC-2 | Coordinator は `QuarantineService` を介さずに直接 `DSPHandleRuntime::quarantine()` 等を呼んではならない |
| QSVC-3 | `unquarantine()` は State + Audit の両方を整合性をもって戻す |
| QSVC-4 | ライフタイムは Coordinator が管理する |

### 完了条件
1. `QuarantineService` クラスが実装されている
2. P1-2 `resetReceipt()` 内の2重呼び出しが置換されている
3. `AudioEngine.Timer.cpp` および `AudioEngine.Threading.cpp` の直接呼び出しが置換されている
4. 既存テスト全件通過
5. 新規テスト追加

### テスト計画
| テスト | 内容 |
|--------|------|
| QuarantineServiceStateAndAudit | State変更 + Audit の両方実行確認 |
| QuarantineServiceRollback | 失敗時 State ロールバック確認 |
| QuarantineServiceDirectCallBlocked | QSVC-2 違反がコンパイルエラーになる |
| QuarantineServiceUnquarantine | unquarantine の整合性確認 |

### 見積工数
実装0.3日 + テスト0.3日 = **0.6日**

---

## 推奨実装順序（依存関係順）

### Phase 1: 基盤整備（既存5項目）

1. **P0-2 DSPHandleRuntime移行** — EQCoeffCache の Handle Table 移行（DSPHandleRuntime 基盤は既存）[🔴P0]
2. **P1-2 PublishReceipt** — Receipt 状態機械の完成（Handle保持 + quarantine 基盤は既存）[🟡P1]
3. **P1-1 FFT Backend Concept化** — 詳細設計完了（実装フェーズ別タスク）[🟡P1]
4. **ADD-2 MMCSS例外登録** — 文書のみ [🔷INFO]
5. **ADD-4 ASan/TSan CI分離** — 詳細設計完了（CI設定フェーズ別タスク）[🔷INFO]

### Phase 2: Coordinator Authority 整備（新規4項目）

6. **P0-4A Observe Authority** — Timer→Coordinator 委譲（P1-2 完了後）[🔴P0]
7. **P0-4B Delete Authority** — reclaim() Coordinator 専用化（P0-4A と並行可）[🔴P0]
8. **P0-4C Coordinator Interface** — 上記2項目のインターフェース追加（先行して実施）[🔴P0]
9. **P0-5 QuarantineService** — DSPHandleRuntime + DSPQuarantineManager 統合（P1-2 完了後）[🔴P0]

**既に実装済み（コード確認完了）**:
- ✅ P0-1: SafeStateSwapper tail 2-writer 解消（head 専用化）
- ✅ P0-3: AudioSegmentBuffer 61MB ヒープ化
- ✅ P2: updateAudioThreadSnapshotFade 削除
- ✅ FIX-D1: kMaxMismatch epoch ベース化
- ✅ ADD-1: fallbackQueue bounded化
- ✅ ADD-3: DeferredFreeThread Logger rate limit

## 設計上の注意点

| # | 項目 | 重要度 | 状態 |
|---|------|--------|------|
| 1 | kMaxMismatch Timer周期依存 | ✅ 解決済み | FIX-D1 対応済み（kMaxEpochDrift 移行完了） |
| 2 | Emergency Override後の stale receipt | 🟡 LOW | P1-2 で対応（基盤実装済み、状態機械は本設計書で確定） |
| 3 | onTransitionComplete/notifyTransitionComplete | 🔷 INFO | 現状維持（設計上の統合フック） |
| 4 | release/acquire + External Serialization二層依存 | 🔷 INFO | 設計上の既知制約 |
| 5 | Fatal時の pendingReceipt_ 診断用保持 | 🔷 INFO | 設計上の既知制約 |
| 6 | MMCSS AvRevertのRT性 | 🔷 INFO | ADD-2 で対応（文書のみ、本設計書で設計確定） |
| 7 | ASan/TSan CI job分離 | 🔷 INFO | ADD-4 で対応（詳細設計完了） |
| 8 | Coordinator 唯一 Authority 原則 | 🔷 INFO | P0-2/P1-2 で対応（ISR Authority整理セクション参照） |
| 9 | FFTExecutionContext 分離（Layer が FFT を知らない） | 📋 将来課題 | P1-1 改善提案として記載 |
| 10 | ISR Coordinator 経由寿命管理（Observe/Delete） | 🔴 P0 | P0-4 で対応（本設計書で設計確定） |

## 未確定・未決定事項

**v20 時点で全9項目の設計は確定済みです。ACK応答型・Shutdown優先度・Handle Fairness・QSVC通知・EpochWaiting注記を追加完了。**

以下は「未着手」ですが設計は確定しています:
- P0-2 EQCoeffCache DSPHandleRuntime移行 — 設計確定（本設計書「設計」セクション参照）
- P1-2 Receipt 状態機械 — 設計確定（本設計書「設計」セクション参照）
- P1-1 FFT Backend Concept化 — 詳細設計完了（本設計書「設計」セクション参照）、実装フェーズは別タスク
- ADD-2 MMCSS例外登録 — 設計確定（本設計書「設計」セクション参照）、文書化のみ
- ADD-4 ASan/TSan CI job分離 — 詳細設計完了（本設計書「設計」セクション参照）、CI設定フェーズは別タスク
- P0-4A Observe Authority — 設計確定（本設計書「設計」セクション参照）、Timer委譲
- P0-4B Delete Authority — 設計確定（本設計書「設計」セクション参照）、reclaim Coordinator専用化
- P0-4C Coordinator Interface — 設計確定（本設計書「設計」セクション参照）、4インターフェース追加
- P0-5 QuarantineService — 設計確定（本設計書「設計」セクション参照）、二重Authority統合

以下はコード実装済み:
- FIX-D1 kMaxMismatch epochベース化 — ✅ **コード実装済み**

以下は設計方針確定（未完全実装）:
- MemoryPool 化（P0-3長期目標）— 設計方針確定、v15では暫定対応（ScopedAlignedPtr + unique_ptr）
- Handle Table 完全移行（P0-2二次案）— 設計方針確定、一次案優先

<!-- ========== Appendix 継続 ========== -->

## B. 修正案詳細 (FIX)

### FIX-P0-1: SafeStateSwapper — Option A（head 専用化）✅ 実装済み

### 目標
`tryReclaim()` から `tail` 書き込みを完全に削除し、head 専用の reclaim に変更する。
tail writer を `swap()` のみに単一化する。

**v12 検証**: コード実装完了。`SafeStateSwapper.h:293` に `// ★ Option A: tail に書き込まない。head 専用化` 確認。
`publishAtomic(tail)` は `swap()` 内1箇所のみ。以下の実装手順は参考情報として維持。

### 決定根拠（2026-07-28 調査完了）
**ソースコード確認（WSL grep/rg/ast-grep）**:
- `publishAtomic(tail, ...)` が2箇所（`swap()` L131 + `tryReclaim()` L266）— **tail 2-writer 確認**
- `swap()` caller: `StateAndUI.cpp:986` の1箇所のみ — ✅ Single Producer
- `tryReclaim()` caller: `DeferredFreeThread.h:143,158` のみ — ✅ Single Consumer

δ案（現状維持）は、`swap()` caller 単一性だけを証明していた。
必要な証明「tail を書く主体が単一である」には `tryReclaim()` からの tail 書き込み削除が必須。

### 実装手順（SafeStateSwapper.h tryReclaim 修正）

#### 変更: tryReclaim() の head 専用化
```cpp
ConvolverState* tryReclaim(uint64_t minReaderEpoch) noexcept
{
    // [Single Consumer debug assert — 変更なし]

    // 1. fallbackQueue を先に確認
    { std::lock_guard<std::mutex> lock(fallbackMutex);
      if (!fallbackQueue.empty()) {
          const auto entry = fallbackQueue.top();
          if (entry.epoch < minReaderEpoch) {
              if (entry.state != nullptr) {
                  fallbackQueue.pop();
                  return entry.state;
              }
          }
      }
    }

    // 2. ring head を確認
    // ★ head はローカル変数 h で追跡。head atomic と h は以下のルールで同期:
    //    next = increment(h)  →  publishAtomic(head, next)  →  h = next
    //    この3ステップは必ず一組で扱う（単独で h だけ更新しない）。
    size_t h = convo::consumeAtomic(head, std::memory_order_acquire);
    if (h == convo::consumeAtomic(tail, std::memory_order_acquire))
        return nullptr;

    // 3. null slot skip (bounded loop)
    for (size_t i = 0; i < kMaxRetired; ++i)
    {
        const uint64_t entryEpoch = convo::consumeAtomic(
            retiredBuffer[h].epoch, std::memory_order_acquire);
        ConvolverState* ptr = convo::consumeAtomic(
            retiredBuffer[h].state, std::memory_order_acquire);

        if (ptr == nullptr || entryEpoch == 0) {
            // null slot: head を進めて次の slot へ
            // ★ 同期ルール: next → publishAtomic(head, next) → h = next
            // ★ 実装推奨: advanceHead(h) helper に集約することで更新規則を1箇所に閉じ込める
            //    例: h = advanceHead(h);  // 内部で next→publish→h=next を実行
            const size_t nextH = (h + 1) % kMaxRetired;
            convo::publishAtomic(head, nextH,
                std::memory_order_release);
            h = nextH;  // ローカル追跡も同期
            if (h == convo::consumeAtomic(tail, std::memory_order_acquire))
                return nullptr;
            continue;
        }

        if (isOlder(entryEpoch, minReaderEpoch)) {
            // reclaim 可能
            convo::publishAtomic(retiredBuffer[h].state, nullptr,
                std::memory_order_release);
            // ★ 同期ルール: next → publishAtomic(head, next)
            const size_t nextH = (h + 1) % kMaxRetired;
            convo::publishAtomic(head, nextH,
                std::memory_order_release);
            // (return のため h=nextH は不要)
            return ptr;
        }

        // reclaim 不可 — ★ tail へ回転しない
        break;
    }
    return nullptr;
}
```

#### 削除するコード
`tryReclaim()` 内の以下のブロックを完全に削除:
```cpp
// ★ 削除: head を進めて tail 側へ回転する
const size_t t = convo::consumeAtomic(tail, std::memory_order_acquire);
...
convo::publishAtomic(tail, nextTail, std::memory_order_release);
```

#### null slot skip ポリシー
| 条件 | 動作 |
|------|------|
| `state == nullptr` | head を進めて skip（bounded loop） |
| `epoch == 0` | head を進めて skip（bounded loop） |
| `epoch < minReaderEpoch` | reclaim（state を返す） |
| `epoch >= minReaderEpoch` | nullptr を返す（tail へ回転しない） |

### CI 3層化

| Layer | コマンド | 成功条件 |
|-------|---------|---------|
| L1: rg | `rg -n "publishAtomic\(tail" src/SafeStateSwapper.h` | swap() 内のみ |
| L2: ast-grep | `tryReclaim` 内の `publishAtomic.*tail` 禁止 | 0 matches |
| L3: contract | `SafeStateSwapperTailWriterSingleTests` 他 | all green |

### テスト追加

```text
SafeStateSwapperTailWriterSingleTests     — publishAtomic(tail) が swap() にのみ存在
SafeStateSwapperHeadOnlyReclaimTests      — tryReclaim() が head のみ更新
SafeStateSwapperNullSlotSkipTests         — null slot を安全に skip
SafeStateSwapperEpochOrderTests           — epoch < minReaderEpoch のみ reclaim
SafeStateSwapperHeadBlockingTests         — head non-reclaimable で後続を触らない
SafeStateSwapperFullFallbackTests         — ring full 時に fallbackQueue へ退避
SafeStateSwapperFallbackOverflowTests     — fallback overflow で quarantine / health
SafeStateSwapperReaderStuckTests          — reader stuck 時に reclaim 停止、UAF なし
```

### リスクと対策
| リスク | 対策 | 重要度 |
|--------|------|--------|
| null slot 連続で ring が詰まる | bounded loop で最大 kMaxRetired まで skip。上限到達時は fallback | LOW |
| fallbackQueue 溢れ | ADD-1 で kMaxFallback 導入 | MEDIUM |
| head 専用化で epoch 逆転 | epoch 単調増加により発生しない（INV-EPOCH-MONOTONIC） | LOW |

### 見積工数
実装1日＋テスト1日＋CI追加0.5日 = **2.5日**

---

## FIX-P2: updateAudioThreadSnapshotFade 削除（旧FIX-HW-3）✅ 実装済み

### 目標
Dead Code 確定に伴い、`updateAudioThreadSnapshotFade()` と `updateFade()` を削除する。

**v12 検証**: コード実装完了。`AudioEngine.h:3738` に DELETED コメント確認。`src/core/SnapshotCoordinator.h:111` も同様。

### 決定根拠
- ✅ 全ツール（grep/ast-grep/rg/cocoindex/semble/graphify）で呼び出し元ゼロを確認
- ✅ `advanceFade()` は `AudioBlock.cpp:475` から LIVE 呼び出しあり → 維持
- ✅ 将来復元は Git 履歴から容易

### 変更内容
1. `AudioEngine.h:3731-3740` — `updateAudioThreadSnapshotFade()` 関数ブロック削除
2. `src/core/SnapshotCoordinator.h:111-138` — `updateFade()` 削除（他からの呼び出しがないことを確認済み）
3. `AudioBlock.cpp:475` — `advanceFade()` 呼び出しは維持。コメントに `[LIVE]` と追記

### 見積工数
30分

---

## FIX-P1-1: FFT Backend Concept 化 + explicit instantiation（旧FIX-U-6）

### 目標
FFT エラー時の異常系テストを実装する。RT パスへのオーバーヘッドはゼロとする。

### 決定根拠（2026-07-28 調査）
- **テスト基盤**: カスタムテストフレームワーク（GoogleTest/GMock/GMock なし）
- **FFT API**: Intel IPP 直接呼び出し（`ippsFFTFwd_RToCCS_64f` / `ippsFFTInv_CCSToR_64f`）
- GMock 非依存のため、virtual + MockFft 方式は不適切
- → **Concept 方式を採用**（virtual dispatch ゼロ、RT-safe 確定）

### 実装手順

#### Step 1: FFT Backend Concept
```cpp
template <typename FftBackend>
concept FftBackendConcept = requires(FftBackend& b, const double* in, double* out) {
    { b.forward(in, out) } -> std::same_as<IppStatus>;
    { b.inverse(in, out) } -> std::same_as<IppStatus>;
};
```

#### Step 2: ProductionFft
```cpp
class ProductionFft {
public:
    explicit ProductionFft(IppsFFTSpec_R_64f* spec) noexcept : fftSpec_(spec) {}

    IppStatus forward(const double* in, double* out) noexcept {
        return ippsFFTFwd_RToCCS_64f(in, out, fftSpec_, workBuf_);
    }

    IppStatus inverse(const double* in, double* out) noexcept {
        return ippsFFTInv_CCSToR_64f(in, out, fftSpec_, workBuf_);
    }

    void setWorkBuffer(Ipp8u* buf) noexcept { workBuf_ = buf; }

private:
    IppsFFTSpec_R_64f* fftSpec_;
    Ipp8u* workBuf_ = nullptr;
};

static_assert(FftBackendConcept<ProductionFft>);
```

#### Step 3: TestFft（エラー注入可能なテスト用）
```cpp
class TestFft {
public:
    IppStatus forward(const double*, double*) noexcept { return result_; }
    IppStatus inverse(const double*, double*) noexcept { return result_; }

    void setResult(IppStatus s) noexcept { result_ = s; }
    void setResultOnCall(IppStatus fwd, IppStatus inv) noexcept {
        resultForward_ = fwd; resultInverse_ = inv;
    }

private:
    IppStatus result_{ippStsNoErr};
    IppStatus resultForward_{ippStsNoErr};
    IppStatus resultInverse_{ippStsNoErr};
};

static_assert(FftBackendConcept<TestFft>);
```

#### Step 4: MKLNonUniformConvolver の修正 — ★ Layer 単位テンプレート化推奨

`IppsFFTSpec_R_64f*` を直接保持する代わりに、テンプレートパラメータとして FFT 実装を受け取る。
ProductionFft をデフォルトテンプレート引数に指定。

**ISR 観点での注意**: **`Layer` 構造体単位のみのテンプレート化を第一推奨とする。**
クラス全体を `template<class FFT>` にするとコンパイル依存が増大し、インスタンス爆発のリスクがある。
可能なら `Layer` 構造体単位のみテンプレート化し、クラス全体のテンプレート化は避ける:

```cpp
// 推奨: Layer 単位のみテンプレート化
template <FftBackendConcept FftBackend>
struct Layer { /* ... */ };

// 非推奨（コンパイル爆発）:
template <typename FftBackend>
class MKLNonUniformConvolver { /* ... */ };
```

#### Step 5: テスト追加
- TestFft が `ippStsNoErr` を返す正常系
- TestFft が `ippStsErr` を返す異常系（`clearFFTOutputOnError` の動作確認）
- 6箇所全ての FFT 呼び出しをカバー

#### Step 6: explicit instantiation

バイナリ肥大とコンパイル時間対策:

```cpp
// MKLNonUniformConvolverLayer.cpp — Production 型のみ explicit instantiation
template class MKLNonUniformConvolverLayer<ProductionFft>;

// テストファイル — TestFft での instantiation
// (テスト target でのみコンパイル)
```

### ProductionFft 契約

```text
FFT-PROD-1: ProductionFft は `IppsFFTSpec_R_64f*` を保持してよい（所有はしない）。
FFT-PROD-2: spec の生成 / 破棄は NonRT のみ。
FFT-PROD-3: forward / inverse は RT から呼び出し可能。
FFT-PROD-4: forward / inverse は noexcept。
FFT-PROD-5: 失敗時は IppStatus を返す。
FFT-PROD-6: RT 内で allocation / free / exception / log を発生させない。
```

### Fail-Closed 契約

```text
FFT-FAIL-1: FFT が non-success を返したら出力をゼロクリアする。
FFT-FAIL-2: stage を ready にしない。
FFT-FAIL-3: stale な結果を publish しない。
FFT-FAIL-4: RT 内で retry しない。
FFT-FAIL-5: error flag / counter は atomic relaxed でよい。
FFT-FAIL-6: log は NonRT へ委譲する。
```

### 注意点
- Concept 方式は静的ポリモーフィズムのため、virtual dispatch は完全にゼロ
- `FftBackendConcept` 経由の呼び出しはコンパイル時に解決される
- MKLNonUniformConvolver のテンプレート化により、テスト時のみ TestFft を注入可能
- `unique_ptr<FftPolicy>` は使用禁止（virtual dispatch 発生のため）
- **Layer 単位のみテンプレート化** を推奨。クラス全体のテンプレート化はコンパイル依存増大のため避ける

### RT パス影響評価
Concept 方式（virtual ゼロ）のため、RT パスへの影響はゼロ。

### テスト追加

```text
FftProductionInstantiationTests   — ProductionFft のみ release binary に含まれる
FftTestBackendInjectionTests      — TestFft でエラー注入可能
FftForwardErrorFailClosedTests    — forward エラー時 fail-closed
FftInverseErrorFailClosedTests    — inverse エラー時 fail-closed
FftNullSpecTests                  — null spec ハンドリング
FftSizeMismatchTests              — サイズ不一致
FftNoPublishOnErrorTests          — エラー時 publish なし
FftNoMklLeakTests                 — MKL リソースリークなし
```

### 見積工数
設計0.5日 + 実装1日 + テスト0.5日 = **2日**

---

## FIX-D1: kMaxMismatch Epoch ベース検出への移行 ✅ 実装済み

### 目標
Timer 呼び出し回数ベースの `kMaxMismatch = 5` を epoch 差分ベースに変更する。

**v12 検証**: コード実装完了。
- `AudioEngine.Timer.cpp:1800`: `publicationEpochDistance(currentEpoch, receiptEpoch) > kMaxEpochDrift` 確認
- `AudioEngine.h:4354-4355`: `kMaxEpochDrift = 10` + `kMaxMismatch` deprecated 確認

### 変更内容
`AudioEngine.Timer.cpp` の `retirePublishedDSP()` 内、不一致検出ロジック：
```cpp
// 現在（Timer 呼び出し回数ベース）:
uint32_t cnt = mismatchCount_.fetch_add(1, std::memory_order_relaxed) + 1;
if (cnt >= kMaxMismatch) { fatal_ = true; }

// 修正後（epoch 差分ベース）:
// pendingReceipt_->publicationEpoch と router_->currentEpoch() の差で判定
// ★ publicationEpochDistance() helper 経由で将来の epoch policy 変更に備える
const auto currentEpoch = engine_.currentPublicationEpoch();  // ISR Coordinator の最新 epoch
const auto receiptEpoch = pendingReceipt_->publicationEpoch;
if (publicationEpochDistance(currentEpoch, receiptEpoch) > kMaxEpochDrift) { fatal_ = true; }
```

### 定数定義
```cpp
// AudioEngine.h に追加
static constexpr uint64_t kMaxEpochDrift = 10;  // 最大許容 epoch 差
// kMaxMismatch は deprecated として残す（後方互換性のため。外部参照がある場合）
```

### 注意点
- `publicationEpochDistance()` helper: 将来の多次元 Epoch を考慮し、対象を明示。
  `(a >= b) ? (a - b) : 0` のような安全な差分計算をラップ
- epoch 差が `uint64_t` の wraparound を起こさない前提が必要
- ISR Runtime の epoch は実質的に wraparound しない（64bit、単調増加）
- `kMaxMismatch` は削除せず deprecated として残す（外部参照がある場合）

### 見積工数
1時間

---

## FIX-P1-2: Stale receipt quarantine 状態機械（旧FIX-D2 拡張）

### 問題
`DSPTransition::onPublishCompleted()` の Emergency Override パス（HealthState Critical）は `storeReceipt()` を呼ばないため、以前の receipt が `pendingReceipt_` に残留する。

また、`resetReceipt()` だけでは retire 義務が消失するリスクがある。
stale receipt の oldDSP は quarantine へ移し、retire 義務を確実に履行する。

### 既存の Handle Table + Quarantine Infrastructure

`src/audioengine/ISRDSPHandle.h` に **`DSPHandleRuntime`**（完全な Handle Table）が既に実装済み:

```cpp
class DSPHandleRuntime {
    DSPHandle create(void*);         // 登録
    ResolvedDSP resolve(DSPHandle);  // 検証
    void retire(DSPHandle);          // retire 遷移
    void quarantine(DSPHandle);      // ★ quarantine 遷移（既存）
    void reclaim(DSPHandle);         // 解放
};
```

`src/audioengine/ISRDSPQuarantine.h` に **`DSPQuarantineManager`** も実装済み:
- `quarantineHandle(slot, generation, reason)` — slot + generation 単位の隔離
- `reclaimSlot(slot, generation)` — 隔離解除
- `AudioEngine.Threading.cpp:42` で実際に使用中（未使用ではない）
- `kMaxSlots = 256` — 上限固定

**ISR 原則**: Quarantine Authority は **1個** に統一する。
既存の `DSPHandleRuntime::quarantine(DSPHandle)` と `DSPQuarantineManager::quarantineHandle(slot, gen, reason)` は
**独立した2つの機構**である（ソース確認済み）:
- `DSPHandleRuntime::quarantine()` → slot の `DSPState` を `Quarantined` に遷移（atomic, generation不問）
- `DSPQuarantineManager::quarantineHandle()` → slot+generation 一致確認 + audit log 記録

両者は complementary であり、両方を呼び出すことで slot 状態遷移 + audit の両方を実現する。

`PendingReceiptQuarantine` の新設は **二重 Authority** となるため、**廃止**。

### 推奨状態機械（設計セクション v16 確定版と同一。DSPState enum に統合）

```
Empty ──storeReceipt()──→ Ready ──normal retire──→ Consumed
                            │
                            ├──stale/emergency/mismatch──→ StaleExported
                            │                                 │
                            │                          quarantine 実行
                            │                                 │
                            │                                 ▼
                            │                     Quarantined ← DSPState::Quarantined
                            │                                 │
                            │                          DestroyPending
                            │                                 │
                            │                          Reclaimed
                            │
                            └──resetReceipt()──→ Empty（通常のリセット）
```

**DSPState 対応**: `StaleExported`=`Retired`, `Quarantined`=`DSPState::Quarantined`,
`DestroyPending`=`DSPState::DestroyPending`, `Reclaimed`=`DSPState::Reclaimed`

重要: **Quarantined は自動 free しない。** reader / fade / epoch の安全確認なしに free してはならない。
Quarantined→DestroyPending→Reclaimed の遷移は Coordinator の `waitReaders()` 通過後にのみ実行される。

**設計注記**: `DSPState` 列挙には明示的な `EpochWaiting` 状態は存在しないが、`Retired→DestroyPending` 間の Epoch 待機期間を Coordinator の `waitReaders()` が表現する。実装上は `Retired` 状態かつ `activeReaderCount() > 0` が暗黙の EpochWaiting 相当である。

### 実装

#### Step 1: Quarantine エントリ型
```cpp
struct alignas(64) DSPQuarantineEntry {
    DSPCore* dsp{nullptr};
    convo::isr::PublicationEpoch epoch{0};
    uint64_t quarantinedAtTick{0};
};
```

#### Step 2: resetReceipt → DSPHandleRuntime 経由の quarantine

`PendingReceiptQuarantine` の新設は **二重 Authority** となるため**廃止**。
代わりに既存の `DSPHandleRuntime` を経由する:

```cpp
// ★ ISR 推奨: PublicationReceipt 自体が DSPHandle を保持する
//    lookupDSPHandleForRuntime() による逆引きは Authority 分散のため非推奨。

// 変更前:
struct PublishReceipt {
    DSPCore* dsp{nullptr};                              // raw pointer
    convo::isr::PublicationEpoch publicationEpoch{0};
    convo::isr::PublicationGeneration generation{0};
};

// 変更後:
struct PublishReceipt {
    convo::isr::DSPHandle handle{};                      // ★ Handle 保持
    convo::isr::PublicationEpoch publicationEpoch{0};
    convo::isr::PublicationGeneration generation{0};
};

// resetReceipt では Handle 経由で直接 quarantine:
void resetReceipt() noexcept {
    if (pendingReceipt_.has_value()) {
        // retire 義務を DSPHandleRuntime 経由で quarantine へ移転
        // ★ Handle を直接保持しているため逆引き不要
        handleRuntime_.quarantine(pendingReceipt_->handle);
    }
    pendingReceipt_.reset();
    receiptReady_.store(false, std::memory_order_relaxed);
}
```

**DSPHandleRuntime + DSPQuarantineManager 協調**:
両者は独立した機構であり、quarantine 時には両方を呼び出す:
1. `DSPHandleRuntime::quarantine(handle)` — slot state を `Quarantined` に遷移
2. `DSPQuarantineManager::quarantineHandle(slot, gen, reason)` — audit log 記録

※ ソース確認: `DSPHandleRuntime::quarantine()` は `DSPQuarantineManager` を内部的に呼ばない。

#### Step 3: Quarantine Lifecycle 全体像

**既存の `DSPHandleRuntime` の `DSPState` 列挙**:

```text
Constructing → Active → Retired → Quarantined → DestroyPending → Reclaimed
                    ↘ CrossfadingIn/Out ↗
```

これが ISR の完全な Lifecycle であり、receipt quarantine もこの中に含まれる。

```text
Published (storeReceipt)
    ↓
Retired (retirePublishedDSP - Normal Retire)
    ↓ (stale/emergency)
Quarantined (DSPHandleRuntime::quarantine)
    ↓ (shutdown / safe drain)
DestroyPending → Reclaimed
```

#### Step 3: Retire 義務移転ルール

```text
RECEIPT-1: pendingReceipt_ を reset する前に、oldDSP を quarantine へ移す。
RECEIPT-2: quarantine された DSP は、reader / fade / epoch の安全確認なしに free しない。
RECEIPT-3: evidence export は NonRT で行う。
RECEIPT-4: RT 内で file I/O / logger / exception を発生させない。
RECEIPT-5: quarantine 増加は diagnostic counter と health event に記録する。
RECEIPT-6: shutdown 時は drain を試み、不可能なら leak-safe に quarantine する。
```

### テスト追加

```text
ReceiptStaleExportTests                  — stale receipt evidence export
ReceiptQuarantineTransitionTests         — quarantine 遷移確認
ReceiptResetDoesNotDropRetireObligationTests — retire 義務消失防止
ReceiptEmergencyOverrideTests            — Emergency Override 時 quarantine
ReceiptShutdownDrainTests                — shutdown drain
```

### 見積工数
設計0.5日 + 実装0.5日 + テスト0.5日 = **1.5日**

---

## FIX-ADD-1: fallbackQueue bounded 化 ✅ 実装済み

### 問題
`SafeStateSwapper.h` の `std::priority_queue<FallbackEntry> fallbackQueue` は unbounded。
reader stuck や retire stall 時に無限に成長する可能性がある。

**v12 実装確認**: 実際のコードは `kMaxFallback=1024` 上限 + `fallbackOverflowCount_` atomic increment。
overflow 時の Coordinator通知は `getPendingRetiredCount()` の外部ポーリングに委譲（quarantine までは未実装）。

### 実装

```cpp
// SafeStateSwapper.h (v12 実装確認: 実コード SafeStateSwapper.h:119-135)
// overflow 時は quarantine ではなく fallbackOverflowCount_ を atomic increment。
// Coordinator への通知は getPendingRetiredCount() の外部ポーリングに委譲。
static constexpr size_t kMaxFallback = 1024;

// overflow 時の処理:
std::lock_guard<std::mutex> lock(fallbackMutex);
if (fallbackQueue.size() >= kMaxFallback) {
    // ★ overflow counter（relaxed atomic、diagnostic only）
    fallbackOverflowCount_.fetch_add(1, std::memory_order_relaxed);
    // Coordinator への通知は外部ポーリング（getPendingRetiredCount()）に委譲
} else {
    fallbackQueue.push({oldState, epoch2});
}
```

### ルール

```text
FALLBACK-1: fallbackQueue は NonRT でのみ使用する。
FALLBACK-2: 上限 kMaxFallback = 1024。
FALLBACK-3: 上限到達時は新規 push を拒否（drop）。fallbackOverflowCount_ を atomic increment して記録。
FALLBACK-4: fallback overflow 通知は SafeStateSwapper::getPendingRetiredCount() の外部ポーリングに委譲。
FALLBACK-5: overflow 時の quarantine は未実装（将来課題）。現状は leak-safe に counter 記録のみ。
```

### 見積工数
0.5日

---

## FIX-ADD-2: MMCSS AvRevert 例外登録

### 問題
`coding_rule_jp.txt` では Audio Thread 内の MMCSS 設定を禁止しているが、
`revertMmcssOnAudioThread()` が Audio callback 内で `AvRevertMmThreadCharacteristics` を呼ぶ。

### 調査結果
- `AudioEngine.Mmcss.cpp:204` — `revertMmcssOnAudioThread()` が `::AvRevertMmThreadCharacteristics(t_mmcssHandle)` を呼ぶ
- `AudioEngine.h:2303-2305` — MMCSS shutdown は flag 経由で Audio Thread に委譲
- 設計コメントに「ASIO thread entry をフックできない場合のみ例外として許可」と記載あり

### 対応

```text
MMCSS-EX-1: Audio Thread 内での MMCSS API 呼び出しは、
            ASIO thread entry をフックできない場合のみ例外として許可する。
MMCSS-EX-2: 呼び出しは thread_local guard により一度だけとする。
MMCSS-EX-3: RT 内で log しない（#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS でガード済み）。
MMCSS-EX-4: 失敗しても音声を止めない。
MMCSS-EX-5: 例外登録簿に記載する。
```

### 例外登録簿への記載例

```text
| # | 機能 | ファイル | 行 | 理由 | 承認日 |
|---|------|---------|----|------|-------|
| 1 | AvRevertMmThreadCharacteristics | AudioEngine.Mmcss.cpp | 201-204 | ASIO thread entry 非フック可能時のみ | 2026-07-28 |
```

### 見積工数
30分（文書更新のみ）

---

## FIX-ADD-3: DeferredFreeThread Logger rate limit ✅ 実装済み

### 問題
`DeferredFreeThread.h:168` で backlog 警告を毎ループ出力している。
`kPendingRetiredWarnThreshold` 到達時は毎 iteration でログ出力が発生する。

### 対応

```cpp
// DeferredFreeThread.h に rate limit 追加
std::chrono::steady_clock::time_point lastLogTime_;
static constexpr auto kLogInterval = std::chrono::seconds(5);

// ログ出力部分
if (pendingRetired >= kPendingRetiredWarnThreshold) {
    const auto now = std::chrono::steady_clock::now();
    if (now - lastLogTime_ >= kLogInterval) {
        juce::Logger::writeToLog("[DIAG] DeferredFreeThread backlog pending="
                                 + juce::String(static_cast<juce::int64>(pendingRetired)));
        lastLogTime_ = now;
    }
}
```

### ルール

```text
LOG-1: DeferredFreeThread の log は rate limit する（5秒間隔以上）。
LOG-2: 同一条件の連続 log は間引く。
LOG-3: critical な場合のみ error log（通常は DIAG level）。
```

### 見積工数
15分

---

## FIX-ADD-4: ASan / TSan CI job 分離

### 問題
現在の CMakeLists.txt には ASan 設定が含まれるが、Debug ビルド（/MTd）とは非互換。
ASan と TSan は同時に使えないため、別 job に分離する必要がある。

### 調査結果
`CMakeLists.txt:1037-1056` — ASan 設定存在（`/fsanitize=address`）。実際の endif() は L1056。
ただし Debug は `/MTd`（静的CRT）で ASan 非対応。

### 推奨 CI 構成

| Config | CRT | Sanitizer | 備考 |
|--------|-----|-----------|------|
| Debug | /MTd | なし | 既存の Debug タスク |
| Debug-ASan | /MDd | AddressSanitizer | 新規 CI job |
| Debug-TSan | dynamic CRT | ThreadSanitizer | 新規 CI job（要 Clang） |
| Release | /MT | なし | 既存の Release タスク |
| Release-PGO | /MT | なし | 既存の PGO タスク |

### CMakeLists.txt 変更案

```cmake
# ASan / TSan は専用ターゲットでのみ有効化
option(ENABLE_ASAN "Enable AddressSanitizer (Debug ASan job)" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer (Debug TSan job, Clang only)" OFF)

if(ENABLE_ASAN AND ENABLE_TSAN)
    message(FATAL_ERROR "ASan and TSan are mutually exclusive. Enable only one.")
endif()

if(ENABLE_TSAN AND MSVC)
    message(FATAL_ERROR "TSan requires Clang (MSVC not supported). Use Clang or WSL Clang.")
endif()

if(ENABLE_ASAN)
    # ASan 必須: 動的 CRT（/MDd for Debug, /MD for Release）
    # 静的 CRT（/MT /MTd）は MSVC ASan と非互換（LNK2038）
    set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

if(ENABLE_TSAN)
    # TSan は Clang の -fsanitize=thread で有効化
    target_compile_options(ConvoPeq PRIVATE -fsanitize=thread)
    target_link_options(ConvoPeq PRIVATE -fsanitize=thread)
    set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
```

### 見積工数
CI 設定1日

---

## FIX-D3: onTransitionComplete / notifyTransitionComplete デッドコード処理（現状維持）

### 選択肢

#### ISR 観点: 削除が原則だが保留中

ISR では「Authority が存在しないコードは削除」が原則。
`onTransitionComplete()` と `notifyTransitionComplete()` は以下の状態:

- `notifyTransitionComplete()`: **メソッド本体は実装済み**（`RuntimePublicationOrchestrator.cpp:392`）、4責務の処理ロジックを持つ
- **ただし外部呼び出し元はゼロ**（AiDex 検証済み）。`notifyTransitionComplete` 自体はどこからも呼ばれていない
- `onTransitionComplete()`: `DSPTransition.h:132` に**定義済み**（宣言のみではない）。`notifyTransitionComplete`（`RuntimePublicationOrchestrator.cpp:398`）から呼び出されているが、`notifyTransitionComplete` 自体が呼ばれていないため間接的に到達不能

#### Option A: 完全削除（推奨）※ただし本体実装済みのため注意
関数本体が実装済みであるため、単なる宣言削除ではない。削除する場合は実装コードも削除する。

#### Option B: Reserved Hook（現状維持）
コメントに「将来の統合フック」として維持。本設計書では現状維持とする。

### 推奨: Option B（現状維持）
呼び出し元不在だが、設計上の統合ポイントとして責務定義を保持する（コードコメント L383-391 に明記）。

### 見積工数
15分

---

## A. 実装済み事項一覧（全37件）

### HW-1: Publication Metadata Propagation ✅ 完了

| 項目 | 内容 |
|------|------|
| **対応バグ** | BUG-030（拡張） |
| **重要度** | 🔴 HIGH |
| **関連ファイル** | 7ファイル |
| **ステータス** | ✅ **実装完了・テスト通過（19/19）** |

**設計の核**: `DSPTransition` が `oldDSP` を `PublishReceipt` として保存し、Timer の retire パスで `publicationEpoch` を伝搬する。Retire は Normal/Fallback/Emergency の3分類。

**実装ファイル**:

| ファイル | 変更内容 |
|---------|---------|
| `ISRRuntimeSemanticSchema.h` | `PublicationGeneration` 型エイリアス追加 |
| `ISRRuntimePublicationCoordinator.h` | `currentPublicationEpoch()` getter |
| `AudioEngine.h` | `PublishReceipt` struct + receipt管理メンバ + `storeReceipt()`/`retirePublishedDSP()`/診断カウンタ |
| `AudioEngine.Timer.cpp` | `retirePublishedDSP()` 定義（3分類＋診断カウンタ）＋3 CAS パス更新 |
| `DSPLifetimeManager.h` | `retire(DSPCore*, uint64_t epoch)` overload |
| `DSPTransition.h` | `storeReceipt(oldDSP, epoch)` + CAS パス更新 |
| `RuntimePublicationOrchestrator.cpp` | （初回実装後に削除 → storeReceipt は DSPTransition に移動） |

**設計上の重要な決定**:

| 判断 | 根拠 |
|------|------|
| Retire の3分類: Normal / Fallback / Emergency | Normal のみ publicationEpoch 伝搬。Fallback/Emergency は runtimeEpoch |
| Publication Metadata Propagation は Normal Retire のみ対象 | Invariant として明文化 |
| 診断カウンタ: normal/fallback/emergencyRetireCount_ | `publishCount ≈ normal + emergency`。fallback は startup/shutdown で少数発生 |
| release/acquire 二層保護 | External Serialization（設計層）＋ atomic 操作（言語層） |
| Fatal 時も current を retire | リーク防止。pendingReceipt_ のみ診断用保持 |

### C-4: TruePeakDetector int→size_t ✅ 完了

| 項目 | 内容 |
|------|------|
| **対応バグ** | BUG-019 |
| **重要度** | 🟢 LOW |
| **ファイル** | `src/TruePeakDetector.cpp`, `src/TruePeakDetector.h` |
| **ステータス** | ✅ **実装完了** |

**修正内容**:
- `kStage0LOffset`/`kStage0ROffset`/`kStage1LOffset`/`kStage1ROffset`: `int` → `size_t`
- `interpolateStage()` 第3引数: `int inputSamples` → `size_t inputSamples`
- ループ変数: `int n` → `size_t n`（`ptrdiff_t` で安全演算）
- `scanPeak()` 呼び出し: `static_cast<int>(up2Samples)` で警告抑制

### グループA: 即時実施可能（13件完了）

| ID | バグ | ファイル | 修正内容 |
|----|------|----------|----------|
| A-1 | BUG-038 | `SpectrumAnalyzerComponent.h:74` | `FFT_MAGNITUDE_SCALE = 2.0f / NUM_FFT_POINTS` |
| A-2 | BUG-035 | `ConvolverProcessor.LoadPipeline.cpp` | RAII `ApplyComputedIRLoadingGuard` 導入 |
| A-3 | BUG-036 | `ConvolverProcessor.LoadPipeline.cpp` | `irL.release()`/`irR.release()` を init 成功時に移動 |
| A-4 | BUG-034 | `MKLNonUniformConvolver.cpp`（6箇所） | `clearFFTOutputOnError()` ヘルパー導入 |
| A-5 | BUG-011/012/013 | `CmaEsOptimizer.h/Dynamic.h/cpp` | `sigma = std::clamp(s, sigmaMin, sigmaMax)` 5箇所 |
| A-6 | BUG-029 | `DSPTransition.h` | Emergency Override で `exchangeFadingRuntimeDSP` を使用 |
| A-7 | BUG-028 | `CrossfadeRuntime.h` | `complete()` で全フラグリセット（pending/useDryAsOld/等） |
| A-8 | BUG-015 | `ISRRetireRouter.cpp` | `n` でリトライロジック内蔵＋戻り値確認 |
| A-9 | BUG-016 | `CmaEsOptimizer.h/Dynamic.h` | `sanitize()` で NaN/Inf→0.0 クランプ |
| A-10 | BUG-042/044/046 | 各クラス | Rule of Five（`=delete`/`=default`） |
| A-11 | BUG-045 | `IRConverter.cpp` | resample 失敗時に `actualSampleRate = sourceRate` |
| A-12 | BUG-039 | `CustomInputOversampler.cpp` | `std::min(targetSamples, static_cast<int>(upsampledBlock.getNumSamples()))` |
| A-13 | BUG-040 | `NoiseShaperLearner.cpp` | `sampleRateHz > 0 ? ... : 48000` フォールバック |

### グループB: 設計確定済み（4件完了）

| ID | バグ | ファイル | 修正内容 | ステータス |
|----|------|----------|----------|-----------|
| B-1 | BUG-030 | `AudioEngine.h`, `DSPTransition.h`, `AudioEngine.Timer.cpp` | `claimFadingRuntimeDSP` CAS-only 実装 | ✅ 完了 |
| B-4 | BUG-032 | `SnapshotCoordinator.h:122` | `getCurrentSnapshot()` インターフェース追加 | ✅ 完了 |
| B-5 | BUG-024 | `SnapshotFadeState.h` | `fadeGeneration_` ABA 対策（generation比較） | ✅ 完了 |
| B-6 | BUG-037 | `ConvolverProcessor.h:883`, `ConvolverProcessor.Lifecycle.cpp:107` | `loaderGeneration_` UAF 防止（デストラクタ先頭 fetch_add） | ✅ 完了 |

### グループC: 計画的対応（7件完了）

| ID | バグ | ファイル | 修正内容 | ステータス |
|----|------|----------|----------|-----------|
| C-1 | BUG-033 | `AudioEngine.Processing.BlockDouble.cpp:421` | `dryScale` ラムダキャプチャ追加 | ✅ 完了 |
| C-2 | BUG-025 | `SnapshotCoordinator.cpp:38` | `n` 化 | ✅ 完了 |
| C-3 | BUG-018 | 3ファイル | `!=1.0` → `std::abs(x-1.0)>1e-5f` | ✅ 完了 |
| C-4 | BUG-019 | `TruePeakDetector.cpp:102-111` | `int` → `size_t` | ✅ 完了（HW-1 関連で本編C-4も同時完了） |
| C-5 | BUG-020 | `ConvolverProcessor.LoaderThread.cpp:151-152` | `if(targetLength<=0)return 0;` | ✅ 完了 |
| C-6 | BUG-021/022 | `ConvolverProcessor.Lifecycle.cpp:147-150` | RCU `GlobalGuard` 追加（2箇所） | ✅ 完了 |
| C-7 | BUG-026 | `ObservedRuntime.h:49` | `rootEnterSucceeded()`確認 | ✅ 完了 |

### グループD: 余裕時（4件完了）

| ID | バグ | ファイル | 修正内容 | ステータス |
|----|------|----------|----------|-----------|
| D-1 | BUG-041 | `NoiseShaperLearner.cpp:649` | VLA→`makeAlignedArray` ヒープ割当 | ✅ 完了 |
| D-2 | BUG-043 | `IRConverter` | パラメータ名修正 | ✅ 完了 |
| D-3 | BUG-027 | `SnapshotCoordinator.cpp:15` | `target==null` 時 state 再確認 | ✅ 完了 |
| D-4 | BUG-046 | `PsychoacousticDither.h` | A-10 に含む（Rule of Five） | ✅ 完了 |

### 解決済み未確定事項

| ID | 内容 | 解決日 |
|----|------|--------|
| U-1 | `getCurrentSnapshot()` インターフェース確認（`SnapshotCoordinator.h:122`） | ✅ 2026-07-27 |
| U-4 | Publication Metadata Propagation to Retire Path | ✅ 2026-07-28（→ HW-1 実装完了） |
| U-5 | B-6 Generation インクリメントタイミング（デストラクタ先頭） | ✅ 2026-07-27 |

### v12 新規追加実装済み事項（6件）

以下の6項目は v11 計画書では「⚠️ 未実装」と記載されていたが、v12 ソースコード検証（AiDex/AST-grep/grep/serena 他）により実装完了を確認。

| ID | 内容 | ファイル | 確認内容 |
|----|------|----------|----------|
| ✅ **P0-1** | SafeStateSwapper tail 2-writer 解消（head 専用化） | `SafeStateSwapper.h` | `tryReclaimSlot()` + `advanceHead()` + `ReclaimResult` enum 実装済み。`publishAtomic(tail)` は `swap()` のみ |
| ✅ **P0-3** | AudioSegmentBuffer 61MB ヒープ化 | `AudioSegmentBuffer.h`, `NoiseShaperLearner.h` | `ScopedAlignedPtr` heap + factory + Rule of Five + `static_assert(sizeof<1024)` |
| ✅ **P2** | updateAudioThreadSnapshotFade 削除 | `AudioEngine.h:3738`, `src/core/SnapshotCoordinator.h:111` | DELETED コメント確認。`advanceFade()` は維持 |
| ✅ **ADD-1** | fallbackQueue bounded化 | `SafeStateSwapper.h:448` | `kMaxFallback=1024` + overflow counter 実装済み |
| ✅ **ADD-3** | DeferredFreeThread Logger rate limit | `DeferredFreeThread.h:169,184-185` | `kLogInterval=5s` + `lastLogTime_` 実装済み |
| ✅ **FIX-D1** | kMaxMismatch epochベース化 | `AudioEngine.Timer.cpp:1800`, `AudioEngine.h:4355` | `kMaxEpochDrift=10` + `publicationEpochDistance()`

## C. レビュー履歴

| 版 | レビュー | 主な変更 |
|----|---------|----------|
| v1 | — | 初版。Phase 1〜4 分類 |
| v2 | 1次 | グループA/B/C/D 再分類。W-6/W-8/W-10 設計変更 |
| v3 | 1次 | 全調査確定。3部構成。B全6件設計確定 |
| v4 | 2次 | B-1: acquiringFadingSlot→CAS+exchange。B-2: publish順序。他 |
| v5 | 3次 | B-1: CAS+exchange→CAS-only。A-4: CCS/FFT サイズ差異明記。B-2: 安全性証明追記 |
| v6 | 4次 | A-4: 「7箇所」→「6箇所」修正。異常系テスト方法変更。A-2 isLoading競合確認。B-2「未確定」に格下げ |
| v7 | 2026-07-27 | 全実装状況をコードベース調査により確認。実装済み29件をAppendixに移動。未実装5件の設計を新設 |
| **v8** | **2026-07-28** | **HW-1/C-4 実装完了に伴い全未解決事項を「残課題」に集約。実装済み37件をAppendix Aに統合。設計上の注意点5項目を追加。全7回のレビューサイクル完了** |
| **v9** | **2026-07-28** | **全残課題を最終調査・確定。HW-2(Single Producer確認→δ案)、HW-3(Dead Code確定→削除)、U-6(CRTP方式決定)、設計上の注意点5項目全件調査完了。全ツール(WSL grep/ast-grep/rg/cocoindex/semble/graphify/serena/AiDex)使用。** |
| **v10** | **2026-07-28** | **レビュー指摘を全面的に反映。P0-1: δ案→Option A(head専用化)＋HEAD-4/5/6＋INV-AUTHORITY。P0-2: retired flag案廃止→DSPHandleRuntime移行＋HANDLE-5/6/7。P0-3: 61MBヒープ化＋Rule of Five。P1-1: FFT Backend Concept化＋Layer単位テンプレート。P1-2: quarantine状態機械＋RECEIPT-6。ADD-1〜4: fallbackQueue bounded/MMCSS/Logger/ASan。全7項目の設計上の注意点更新。** |
| **v11** | **2026-07-28** | **P1-1/ADD-4 詳細設計を新設・旧設計セクションより分離。全11項目の設計確定。** |
| **v12** | **2026-07-28** | **全コードベース検証（AiDex/AST-grep/grep/serena/semble/cocoindex/graphify使用）。P0-1/P0-3/P2/ADD-1/ADD-3/FIX-D1の6項目がコード実装済みであることを確認、ステータス修正。P0-2/P1-2の一部実装確認。未完了は残5項目（P0-2一部/P1-2一部/P1-1/ADD-2/ADD-4）。P1-1/ADD-4は詳細設計完了、実装フェーズは別タスク。onTransitionCompleteは宣言のみではなく定義済み（DSPTransition.h:132）、notifyTransitionCompleteから呼び出されているが間接到達不能。** |

## D. 調査結果詳細

### C.1 HW-1: Publication Metadata Propagation 調査結果

**調査ツール**: AiDex/grep/semble/cocoindex/serena/ast-grep/rg

✅ **確定した事実**:
- `RetireIntent` 構造体（`ISRRetire.h`）には既に `retireEpoch` フィールドが存在する
- `commit()` 関数（`ISRRuntimePublicationCoordinator.cpp`）は `PublicationEpoch epoch` パラメータを受け取る
- DSPLifetimeManager::retire(DSPCore*, uint64_t) overload 追加により epoch 伝搬が可能に

### C.2 P0-1: SafeStateSwapper tail 2-writer 調査結果

**調査ツール**: AiDex/grep/ast-grep/rg/sed/awk + コード実査

✅ **修正済みの事実（v12 時点）**:
- `swap()` は publish 順序: `publishAtomic(state, release) → publishAtomic(epoch, release) → publishAtomic(tail, release)` — **正しい**
- `tryReclaim()` からの `publishAtomic(tail, ...)` は **削除済み**
- `tryReclaim()` は **Single Consumer 前提**（コードコメント L270 に明記）
- `getState()` は `activeState` のみ読み `retiredBuffer` を直接読まない ✅

履歴: v9=δ案(現状維持) → v10=Option A(head専用化) → v12=実装完了確認

### C.3 HW-3: updateAudioThreadSnapshotFade 調査結果

**調査ツール**: AiDex/grep/ast-grep/rg/sed/cocoindex/semble/graphify

🔴 **確定**: `updateFade()` は未呼び出し、`snapshotAlpha` 等は DSP 処理パスのどこからも未参照。
SnapshotFade の結果は全く使用されていない ≈ Dead Code。

### C.4 P1-1: FFT clearFFTOutputOnError 調査結果（旧U-6）

**調査ツール**: AiDex/grep/rg

- ✅ A-4（`clearFFTOutputOnError`）実装済み（`MKLNonUniformConvolver.cpp` 内6箇所）
- ✅ `unique_ptr<FftPolicy>` は存在しない（virtual dispatch なしの状態維持）
- ❌ FFT エラー時の異常系テストが未実装
- ❌ explicit instantiation 未対応

### C.5 P0-2: RefCountedDeferred tryAddRef 調査結果

**調査ツール**: AiDex/grep/rg/WSL grep

✅ **確定した事実**:
- `tryAddRef()` は既に **CAS loop** を実装（`RefCountedDeferred.h:48-56`）
- `compareExchangeAtomic` で count 0 への increment は atomic に防止される
- ❌ `retired_` flag がない — retire 済みオブジェクトへの tryAddRef が成功し得る（resurrection）
- ❌ RCU 保護契約が文書化されていない

### C.6 P0-3: AudioSegmentBuffer 61MB 調査結果

**調査ツール**: AiDex/grep/rg/WSL grep

✅ **修正済みの事実（v12 時点）**:
- `AudioSegmentBuffer.h` — ヒープ化完了済み（`ScopedAlignedPtr` + factory `create()`）
- 合計約 **61.44 MB** はスタック→ヒープに移行済み
- `NoiseShaperLearner.h:278` で `std::unique_ptr<AudioSegmentBuffer> segmentBuffer` として保持
- `static_assert(sizeof(AudioSegmentBuffer) < 1024)` でスタック禁止を保証

### C.7 ADD-1〜4 調査結果（v12 更新）

| ID | 項目 | v11 状態 | v12 状態 | 詳細 |
|----|------|----------|----------|------|
| ADD-1 | fallbackQueue bounded | ❌ unbounded | ✅ **実装済み** | `kMaxFallback=1024` + overflow counter |
| ADD-2 | MMCSS例外登録 | ❌ 未登録 | ❌ 未登録 | コードは存在、例外登録簿への記載のみ未完了 |
| ADD-3 | Logger rate limit | ❌ 未実装 | ✅ **実装済み** | `kLogInterval=5s` + `lastLogTime_` |
| ADD-4 | ASan/TSan CI | ❌ 未分離 | ❌ 未分離 | CI設定フェーズ未着手（詳細設計完了） |

### C.8 追加調査で確定した事実

**P0-2: tryAddRef 呼び出し元ゼロ判定**:
- `src/RefCountedDeferred.h:48` で定義されているが、`.cpp` / `.h` の**いずれからも呼び出しなし**
- `MKLNonUniformConvolver.h:284` の `refCount` は別の軽量参照カウンタ（RefCountedDeferred非使用）
- **結論: `tryAddRef()` は Dead Code。修正は予防措置**

**P0-1: SafeStateSwapper tryReclaim Single Consumer 検証**:
- `SafeStateSwapper::tryReclaim()` の真の呼び出し元: **`DeferredFreeThread.h:143` のみ**
- `ISRRetireRouter::tryReclaim()` → `provider_->tryReclaim()` は `IEpochProvider*` 経由で **`EpochDomain`（別RCU実装）** を呼ぶ
- `EQProcessor.Core.cpp` の `m_epochDomain.tryReclaim()` も **EpochDomain 独自**
- ✅ **SafeStateSwapper は真に Single Consumer**

**P1-2: DSPQuarantineManager / DSPHandleRuntime 既存確認**:
- `src/audioengine/ISRDSPQuarantine.h` に `DSPQuarantineManager` が **既に実装済み**
- API: `quarantineHandle(slot, generation, reason)` / `reclaimSlot(slot, generation)` / `isActive(slot)` / `destroyForShutdown(slot)`
- `kMaxSlots = 256`（上限固定）
- `AudioEngine.Threading.cpp:42` で **実際に使用中**（未使用ではない）
- `src/audioengine/ISRDSPHandle.h` に **`DSPHandleRuntime`**（完全な Handle Table）が実装済み
  - API: `create/resolve/retire/quarantine/reclaim` — 全ライフサイクル管理
  - `DSPHandle{slot, generation}` — ABA 防止
  - `DSPState` 列挙: `Constructing→Active→Retired→Quarantined→DestroyPending→Reclaimed`
  - これが ISR の理想的 Handle Table パターン
- ❌ `DSPQuarantineManager` は (slot, generation) ペアで動作。receipt の (DSPCore*, epoch) とは型が合わない
- ⚠️ `DSPHandleRuntime::quarantine(DSPHandle)` と `DSPQuarantineManager` は**独立した別機構**:
  - `DSPHandleRuntime::quarantine()`: slot state を `Quarantined` に遷移（generation不問）
  - `DSPQuarantineManager::quarantineHandle()`: generation一致確認後 audit log 記録
  - 両者は相互に呼び出さない。必要に応じて両方を呼ぶ設計が必要

## E. 調査で使用したツール

| ツール | 用途 |
|--------|------|
| WSL grep | 全テキスト検索・全実装項目のコードベース確認 |
| ast-grep | 構造パターン検索（`engine_.storeReceipt`, `retirePublishedDSP` 等） |
| rg (ripgrep) | 高速フィルタリング検索 |
| cocoindex (ccc.exe) | 構造的grep（receiptReady_, fatal_, mismatchCount_ 等の全参照網羅） |
| semble | セマンティックコード検索 |
| graphify | ナレッジグラフ解析（RuntimePublicationCoordinator ノード確認） |
| serena MCP | プロジェクト構成確認 |
| AiDex MCP | プロジェクトインデックス管理・ステータス確認 |

---

*本設計書は ISR Runtime OS 設計原則に基づく。v10-18: ISR段階的改善。v19: QUEUE-3〜8。**v20: EnqueueResult enum・QUEUE-9(Shutdown優先)・QUEUE-10(Handle Fairness)・QSVC-1通知拡充・DSPState EpochWaiting注記。全15項目中6実装済み・9設計確定。***
