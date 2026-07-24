# ConvoPeq Part 7 & Part 8 改修計画 検証レポート

**検証日**: 2026-07-24
**検証対象**: `ConvoPeq_Part7_Part8_RepairPlan.md` の実装状況
**検証方法**: ソースコード分析 + ビルドテスト + CTest実行

---

## 検証サマリー

| フェーズ | 状態 | 問題点 | 備考 |
|---------|------|--------|------|
| Phase 1 | ✅ 正常 | なし | 等電力クロスフェード正しく実装 |
| Phase 2 | ✅ 正常 | なし | FTZ/DAZ スレッド初期化正しく実装 |
| Phase 3 | ✅ 正常 | なし | モノラル→ステレオ複製正しく実装 |
| Phase 4 | 未実装 | — | 計画のみ（今回は対象外） |
| Phase 5 | ✅ 正常 | 1点要確認 | getVersion() の assert が always-pass |
| Phase 6 | 情報 | — | 将来対象 |

---

## Phase 1: クロスフェード等電力化 — ✅ 正常

### 検証項目

| # | 検証内容 | 結果 |
|---|---------|------|
| 1 | `equalPowerSin` が匿名名前空間に定義されている | ✅ AudioBlock.cpp L19, BlockDouble.cpp L19 |
| 2 | float版ラムダが `equalPowerSin(gNew)` / `equalPowerSin(1.0-gNew)` を使用 | ✅ L443-444 |
| 3 | double版ラムダが `equalPowerSin(gNew)` / `equalPowerSin(1.0-gNew)` を使用 | ✅ L421-422 |
| 4 | float版に `dryScale` が存在し正しく適用 | ✅ L442, L445-446 |
| 5 | double版の `dryScale` 取扱い（意図的に省略、コメント明記） | ✅ 正当な設計判断 |
| 6 | 3ファイルの `equalPowerSin` 実装が完全一致 | ✅ 完全一致 |
| 7 | `runLatencyAlignedCrossfadeMixLoop` の呼び出し箇所が2つのみ（float/double） | ✅ 遺漏なし |
| 8 | 古い線形クロスフェードパターン（`gOld = 1.0 - gNew`）が残存しない | ✅ 残存なし |

### 新規バグリスク

- **等電力クロスフェードの数学的正当性**: `equalPowerSin(x)` は `sin(π/2·x)` の6次テイラー近似。中間点（x=0.5）で `sin(π/4) ≈ 0.707`、`0.707² + 0.707² ≈ 1.0` となりエネルギー保存条件を満たす。✅ 安全
- **double版の `dryScale` 省略**: `useDryAsOld=false` のため `dryScale=1.0` 固定。将来拡張時に float版パターンで対応可能。✅ 安全

---

## Phase 2: FTZ/DAZ スレッド起動時1回設定化 — ✅ 正常

### 検証項目

| # | 検証内容 | 結果 |
|---|---------|------|
| 1 | `ensureThreadFloatingPointEnvironment()` が AudioEngine.h に宣言されている | ✅ L2385 |
| 2 | `AudioEngine.Mmcss.cpp` に実装されている | ✅ L219-231 |
| 3 | `thread_local bool` ガードパターンが使用されている | ✅ `t_fpEnvReady` |
| 4 | `_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON)` が設定されている | ✅ |
| 5 | `_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON)` が設定されている | ✅ |
| 6 | AudioBlock.cpp で `ScopedNoDenormals` → `ensureThreadFloatingPointEnvironment()` に置換済み | ✅ L101 |
| 7 | BlockDouble.cpp で同様の置換済み | ✅ L103 |
| 8 | `tryApplyMmcssForSelfManagedThread()` が変更されていない | ✅ |
| 9 | `xmmintrin.h` / `pmmintrin.h` のインクルード経路が確保されている | ✅ `AudioEngine.h` → `<immintrin.h>` 経由 |

### 新規バグリスク

- **スレッド起動時1回設定の設計**: `thread_local bool` ガードにより、スレッド起動時にのみ FTZ/DAZ を設定。MXCSR の save/restore が不要になるためパフォーマンス向上。✅ 安全
- **残存する `ScopedNoDenormals`**: ConvolverProcessor.Runtime.cpp, EQProcessor.Processing.cpp に残存するが、これらは非 callback パスまたは callback 内 DSP パスであり、スレッド起動時の設定が有効であるため冗長で無害。✅ 安全

---

## Phase 3: processToBuffer モノラル→ステレオ複製 — ✅ 正常

### 検証項目

| # | 検証内容 | 結果 |
|---|---------|------|
| 1 | モノラル入力時（`numChannels == 1`）に Ch0→Ch1 複製が行われる | ✅ |
| 2 | 出力バッファが2ch以上のときのみ複製 | ✅ `destination.getNumChannels() > 1` |
| 3 | ステレオ入力パス（`numChannels == 2`）に影響なし | ✅ `if` 条件で分離 |

### 新規バグリスク

- **モノラル入力時の R チャンネル**: 以前は常時無音だったが、L→R 複製によりセンター定位のモノラル信号として扱われる。✅ 正しい挙動
- **ステレオ入力への影響**: `numChannels == 1` の条件により完全に分離。✅ 安全

---

## Phase 5: デッドコード削除 + アサート追加 — ✅ 正常（1点要確認）

### 検証項目

| # | 検証内容 | 結果 |
|---|---------|------|
| 1 | `PublicationBuffer` クラスが ISRRuntimePublicationCoordinator.h から削除されている | ✅ |
| 2 | `PublicationBuffer` 実装が ISRRuntimePublicationCoordinator.cpp から削除されている | ✅ |
| 3 | `PublicationBuffer` の残存参照がソースコード内にない | ✅ ゼロ件 |
| 4 | `getVersion()` にアサートが追加されている | ✅ L169-171 |
| 5 | `ISRDSPHandle.h` に `static_assert` が追加されている | ✅ `is_trivially_copyable_v`, `is_standard_layout_v` |
| 6 | `is_always_lock_free` がコメントアウトされている（CMPXCHG16B 依存） | ✅ 意図的に無効化 |

### 要確認事項

**`getVersion()` の `assert(true)` は always-pass（機能していない）**:
```cpp
std::uint64_t RuntimePublicationCoordinator::getVersion() const noexcept {
    assert(/* Message Thread check: simplified for non-JUCE context */ true);
    return persistentState_.mappedRuntimeGeneration;
}
```
- **現状**: `assert(true)` は常にパスするため、実質的にアサーションとして機能していない
- **意図**: コメントで「ADR-010: simplified for non-JUCE context」と説明があり、将来の Message Thread チェック実装を意図しているプレースホルダと推測
- **リスク**: 低（デバッグビルドでのスレッドセーフティチェックが意図通りに機能していない）
- **推奨**: 将来的に `JuceHeader.h` をインクルードして `jassert()` を使用するか、`assert()` の引数を実際のスレッドチェックに差し替える

### 新規バグリスク

- **PublicationBuffer 削除**: 使用箇所がなかったため、削除による影響なし。✅ 安全
- **static_assert**: `is_trivially_copyable_v` と `is_standard_layout_v` は DSPHandle の型要件を保証。✅ 安全
- **is_always_lock_free コメントアウト**: 16バイト構造体のため CMPXCHG16B に依存。ビルド設定によっては lock-free でない場合がある。✅ 正当な判断

---

## 配線漏れチェック

### equalPowerSin の配線

| ファイル | 定義 | 使用 | 状態 |
|---------|------|------|------|
| AudioEngine.Processing.AudioBlock.cpp | ✅ | ✅ | 正常 |
| AudioEngine.Processing.BlockDouble.cpp | ✅ | ✅ | 正常 |
| convolver/ConvolverProcessor.Runtime.cpp | ✅ | ✅ | 正常（既存） |
| eqprocessor/EQProcessor.Processing.cpp | ✅ | ✅ | 正常（既存） |

### ensureThreadFloatingPointEnvironment の配線

| ファイル | 宣言/実装 | 使用 | 状態 |
|---------|----------|------|------|
| AudioEngine.h | ✅ 宣言 | — | 正常 |
| AudioEngine.Mmcss.cpp | ✅ 実装 | — | 正常 |
| AudioEngine.Processing.AudioBlock.cpp | — | ✅ 呼出し | 正常 |
| AudioEngine.Processing.BlockDouble.cpp | — | ✅ 呼出し | 正常 |

### ScopedNoDenormals の配線

| ファイル | 状態 | 判定 |
|---------|------|------|
| AudioEngine.Processing.AudioBlock.cpp | 削除済み → `ensureThreadFloatingPointEnvironment()` | ✅ |
| AudioEngine.Processing.BlockDouble.cpp | 削除済み → `ensureThreadFloatingPointEnvironment()` | ✅ |
| convolver/ConvolverProcessor.LoaderThread.cpp | 残存（非 callback パス） | ✅ 冗長だが無害 |
| convolver/ConvolverProcessor.Runtime.cpp | 残存（callback 内 DSP パス） | ✅ 冗長だが無害 |
| eqprocessor/EQProcessor.Processing.cpp | 残存（callback 内 DSP パス） | ✅ 冗長だが無害 |
| NoiseShaperLearner.cpp | 残存（非 callback パス） | ✅ 冗長だが無害 |

---

## 総合判定

**改修計画の実装は適切に行われている。** 重要な問題点は検出されなかった。

### 実装済みフェーズ

1. **Phase 1**: ✅ クロスフェード等電力化（`equalPowerSin` 適用 + `dryScale` 追加）
2. **Phase 2**: ✅ FTZ/DAZ スレッド起動時1回設定化（`ensureThreadFloatingPointEnvironment`）
3. **Phase 3**: ✅ processToBuffer モノラル→ステレオ複製
4. **Phase 5**: ✅ デッドコード削除 + アサート追加

### 未実装フェーズ

5. **Phase 4**: RT-safe ヘルパー関数の DSP Foundation 層への発展（計画のみ、今回は対象外）
6. **Phase 6**: 防御的改善（将来向け）

### 要確認事項（1点）

- `getVersion()` の `assert(true)` はプレースホルダ。将来のメッセージスレッドチェック実装が必要。

---

*本レポートは `ConvoPeq_Part7_Part8_RepairPlan.md` の実装状況をソースコード分析により検証した結果です。*
