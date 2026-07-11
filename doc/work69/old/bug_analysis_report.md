# ConvoPeq bug.md 検証報告書

**調査日**: 2026-07-07
**対象**: `doc/work69/bug.md`（全14のバグ主張）
**調査者**: GitHub Copilot (DeepSeek V4 Flash)
**使用ツール**: WSL grep/rg/ast-grep, ccc (cocoindex-code), semble, graphify v0.9.8, AiDex

---

## 検証結果サマリ

| バグ# | カテゴリ | 結果 | 深刻度 |
|-------|----------|------|---------|
| #1 | バイパスブレンド欠落 | **VALID（方向が逆）** | 🟡 中 |
| #2 | リングバッファ負index | **LOW**（C++20以降は問題なし） | 🟢 低 |
| #3 | NoiseShaperLearner | **INVALID（ハルシネーション）** | ❌ |
| #4 | EQCacheManager data race | **INVALID（std::atomic使用済み）** | ❌ |
| #5 | ConvolverProcessor dtor | **INVALID（使用方法は正当）** | ❌ |
| #6 | dotProductAvx2 cleanup | **INVALID（3段階ループ実装済み）** | ❌ |
| #7 | besselI0無限ループ | **INVALID（上限100で有限）** | ❌ |
| #8 | CacheMap dtor crash | **INVALID（RAII削除）** | ❌ |
| #9 | IRState+mkl_free | **INVALID（正規パターン）** | ❌ |
| #10 | mixSmoothingSmall | **INVALID（存在しない）** | ❌ |
| #11 | calcLowShelfBiquad | **INVALID（存在しない）** | ❌ |
| #12 | BuildInputSemanticContract | **INVALID（存在しない）** | ❌ |
| #13 | NUPC delay alignment | **INVALID（存在しない）** | ❌ |
| #14 | Retire queue MPSC | **INVALID（存在しない）** | ❌ |

---

## 詳細検証結果

### バグ#1: DSPCoreFloat.cpp のバイパス・ブレンド欠落 [VALID/方向逆]

**主張**: DSPCoreDouble.cpp にバイパスブレンドロジックがない

**実コード検証**:

| ファイル | `bypassBlendRequested` | ドライブレンド | `bypass`文字列出力 |
|----------|----------------------|--------------|------------------|
| `DSPCoreDouble.cpp` | ✅ 548行目に宣言 | ✅ oversampling==1と>1両方に実装 | ✅ |
| `DSPCoreFloat.cpp` | ❌ 存在せず | ❌ 完全欠落 | ❌ ゼロ件 |

**結論**: バグ報告は**方向が完全に逆**。Double版は正しく実装済み。Float版にバイパスブレンドが欠落している。


**該当コード**:
- `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` — 548行目以降に完全な実装
- `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` — 378行目の `oversampling.processDown` 後にブレンドなし



---

### バグ#2: リングバッファ負のindex [LOW/C++20対応]

**主張**: `(idx - 1) & DELAY_BUFFER_MASK` で負値のビット演算が未定義動作

**実コード検証**: `src/convolver/ConvolverProcessor.Runtime.cpp` 488行目

```cpp
// fallback path (iRead < 1 or near-end)
int idx = iRead + i;
double p0 = srcBuf[(idx - 1) & DELAY_BUFFER_MASK];
// fast path (iRead >= 1 and enough space):
// ガードによりs+i-1 >= srcBufが保証
```

- C++20以降: 2の補数が保証され、`(-1) & MASK` = `MASK` = 最終要素で正しいラップアラウンド
- x86/x64の全実装で正しく動作
- fast path は `if (iRead >= 1 ...)` のガードあり

**推奨**: `(idx - 1 + DELAY_BUFFER_SIZE) & DELAY_BUFFER_MASK` とすればC++17以前でも完全ポータブル

---

### バグ#3〜#14: INVALID（ハルシネーション）

**全12件が実コードに存在しない関数名・ファイル名・変数名を使用。**

#### バグ#5/#9: IRState + mkl_free の実際

実コードは正しいパターン:
```
IRState::IRState():
  - std::unique_ptr<juce::AudioBuffer<double>> irOwner  // non-POD
  - const juce::AudioBuffer<double>* ir
  - double sampleRate
  - uint64_t generation

割当: convo::aligned_make_unique<IRState>()
  → mkl_malloc(sizeof(IRState), 64) + placement new

解放: deleter:
  → state->~IRState()    // unique_ptrのdtorを呼ぶ
  → mkl_free(state)      // MKLメモリ解放
```

これは `AlignedObjectDeleter` (AlignedAllocation.h) と同一パターンで**設計上正しい。**

#### バグ#6: dotProductAvx2 の実際

`src/TruePeakDetector.cpp` 128行目:
```cpp
for (; i <= n - 16; i += 16)  // ← 16要素SIMD × 4アキュムレータ
for (; i <= n - 4;  i += 4)   // ← 残余4要素SIMD
for (; i < n; ++i)             // ← スカラー後処理（0〜3要素）
return sum;                    // ← 正しい
```

**3段階のループで残余を完全カバー。** バグ報告の「残余サンプル無視」は誤り。

#### バグ#7: besselI0 の実際

`src/TruePeakDetector.cpp` 113行目:
```cpp
for (int n = 1; n < 100; ++n)  // ← ハードリミット100
```

**上限100で必ず終了。** 関数自体が無限ループになることはありえない。

---

## バグ報告の特徴分析

本バグ報告は以下の明確なハルシネーション特徴を示す：

1. **ファイル名が不正確**: dotProductAvx2 → CustomInputOversampler.cpp（実際はTruePeakDetector.cpp）
2. **関数名が存在しない**: calcLowShelfBiquad, emitRetireIntent, tailOutputBuf
3. **CMake設定が存在しない**: BuildInputSemanticContractTests, /STACK:8388608
4. **変数名が存在しない**: delayFadeRampBuffer, candidatePopulationMatrix
5. **ロジックが実コードと逆**: #1のバイパス方向逆転
6. **存在するコードも不正確な文脈**: besselI0のファイルを間違えている

**原因**: AI（言語モデル）が `ConvoPeq.md` の高レベル要約のみを解析し、実際のソースコードと一致しない内容を生成した典型的な幻覚パターン。

---

## 総合評価と推奨事項

### 真に価値のある発見（1件）
1. **DSPCoreFloat.cpp のバイパスブレンド欠落** — 実害の可能性あり。Double版からの移植が必要。

### 軽微な懸念（1件）
2. `(idx - 1) & DELAY_BUFFER_MASK` のC++17移植性 — 実害極めて低いが、安全に修正可能。

### 推奨
- **このバグ報告を鵜呑みにしないこと。** 12/14の主張がハルシネーション。
- 本当のコードレビューには、**実コードをgrep/rg/semble等で直接検証した結果**のみを信頼すること。
- bug.md の内容をソースコード管理に取り込む前に、全主張を個別検証すること。
- 逆に、**Double版に存在してFloat版に存在しない**バイパスロジックは実際のコード差異であり、本報告の唯一の真正な発見である。
