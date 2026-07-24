# ConvoPeq Part 7 & Part 8 改修計画書（改訂版）

**策定日**: 2026-07-24
**改訂日**: 2026-07-24（ユーザーレビュー反映）
**ベース**: `ConvoPeq_Part7_Part8_Verification_BugList.md`
**対象**: BUG-001〜BUG-010（10件）
**原則**: 実害の大きいものから段階的に改修。各フェーズでビルド＋テスト＋静的解析を実行。

---

## フェーズ一覧（ADR対応表付き）

| フェーズ | 対象 | 内容 | ADR | 予估工数 |
|---------|------|------|-----|---------|
| Phase 1 | BUG-001 | クロスフェード等電力化 + double版 dryScale 追加 | ADR-003, ADR-006 | 中 |
| Phase 2 | BUG-002 | FTZ/DAZ スレッド起動時1回設定化 | ADR-006 | 小 |
| Phase 3 | BUG-006 | processToBuffer モノラル→ステレオ複製 | なし | 小 |
| Phase 4 | BUG-005 | RT-safe ヘルパー関数の DSP Foundation 層への発展 | ADR-002 | 中 |
| Phase 5 | BUG-003, 004, 008 | デッドコード削除 + アサート追加 | ADR-001, ADR-005, ADR-010 | 小 |
| Phase 6 | BUG-009, 010, 019 | 防御的改善（将来向け） | なし | 情報 |

---

## Phase 1: BUG-001 — クロスフェード等電力化（最優先）

### ADR 対応

- **ADR-003**（Runtime責務境界）: DSPCore は `equalPowerSin` を直接呼び出すが、Crossfade 時間・DryMix・Policy の決定は CrossfadeAuthority の責務
- **ADR-006**（RuntimeInvariant）: クロスフェード中のエネルギー保存は不変条件として保証する必要がある

### 問題

`runLatencyAlignedCrossfadeMixLoop` に渡すラムダで `gOld = 1.0 - gNew`（線形和）を使用しており、中間点で `-3dB dip` が発生する。さらに float 版には `dryScale` があるが double 版にないという不整合がある。

### 改修方針

**責務境界の明確化**:

```cpp
// DSPCore は CrossfadePolicy を決定しない。
// Crossfade 時間・DryMix・Policy は CrossfadeAuthority が担当。
// DSPCore は gOld/gNew の計算のみ。
// CrossfadeMixContext として取得する方が Runtime OS に整合（Future Architecture Note）。
```

1. **float 版**（`DSPCoreFloat.cpp` ラムダ内、ConvoPeq.md:31447-31464）:
   - `gOld = 1.0 - gNew` を `gOld = equalPowerSin(1.0 - gNew)` に変更
   - `gNew` も `equalPowerSin(gNew)` に変更
   - `dryScale` は既にあるためそのまま維持

2. **double 版**（`DSPCoreDouble.cpp` ラムダ内、ConvoPeq.md:32181-32193）:
   - `gOld = 1.0 - gNew` を `gOld = equalPowerSin(1.0 - gNew)` に変更
   - `gNew` も `equalPowerSin(gNew)` に変更
   - **dryScale の追加**: float 版と Crossfade Semantics を一致させるため、`useDryAsOld` フラグと `crossfadeRuntime_.getDryScaleGain()` の呼び出しを追加

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | ラムダ内 2 行変更 |
| `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` | ラムダ内 2〜5 行変更 |

### 検証方法

- **Test A: Energy Invariance**: `gainOld² + gainNew² ≈ 1.0`（エネルギー保存の不変条件）を自動検証。判定条件: `Energy Error (dB) = 10*log10(gOld² + gNew²)` が `±0.01 dB` 以内（equalPowerSin の6次テイラー近似誤差を考慮）
- **Test B: Crossfade Semantic**: `gainOld`/`gainNew`/`dryScale` の一致確認。dryScale 含む旧実装と新実装の意味論が一致することを確認
- **Test C: Equal Power Curve Verification**: 等電力曲線 `gOld² + gNew² ≈ 1.0` が中間点でも成立することを検証。旧 Crossfade（線形）と新 Crossfade（等電力）のゲインカーブを比較
- **Shadow Compare**: Peak/RMS/Latency を比較。旧 Crossfade（線形）と新 Crossfade（等電力）の差分を確認し、アルゴリズム変更による変化量を把握
- **実装前チェック**: float 版と double 版のクロスフェード処理が `dryScale` 以外も完全に一致していることを確認（`useDryAsOld`・`crossfadeRuntime`・`gain` 取得順・ラムダ引数の全比較）
- `equalPowerSin` の既存使用箇所（`ConvolverProcessor.Runtime.cpp:62948`）と同一アルゴリズムであることを確認
- ビルド成功 + CTest 実行

### リスク

- `equalPowerSin` は libm 不使用の RT-safe 実装のため、オーディオスレッドでの使用に問題なし
- sin 近似の精度は 6 次テイラー展開で十分（既存使用実績あり）

---

## Phase 2: BUG-002 — FTZ/DAZ スレッド起動時1回設定化

### ADR 対応

- **ADR-006**（スレッド環境管理）: 浮動小数点環境の初期化をスレッド起動時に1回だけ行い、以降は永続的に維持

### 問題

`juce::ScopedNoDenormals` が毎回コールバック時に MXCSR を save/restore する。スレッド起動時に1回だけ設定する方が効率的。

### 改修方針

`tryApplyMmcssForSelfManagedThread()` と同じ設計パターン（`thread_local bool` ガード＋初回のみ実行）を採用。ただし、将来の拡張を見据えて汎用的なインターフェース名を採用:

1. `AudioEngine.h` にメンバ関数宣言追加:
   ```cpp
   void ensureThreadFloatingPointEnvironment();
   ```

2. `AudioEngine.Mmcss.cpp` に実装追加:
   ```cpp
   void AudioEngine::ensureThreadFloatingPointEnvironment()
   {
       // ★ Floating-point execution environment の初期化（スレッド起動時1回のみ）
       // 現在: FTZ + DAZ（デノーマル演算の高性能化）
       // 将来拡張: MXCSR RoundMode, ExceptionMask, FENV 全体をここで管理
       static thread_local bool t_fpEnvReady = false;
       if (t_fpEnvReady)
           return;
       t_fpEnvReady = true;
       _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
       _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
       // vmlSetMode は MainApplication で設定済み（スレッドローカルのため別スレッドでは効果なし）
   }
   ```

3. `DSPCoreFloat.cpp` / `DSPCoreDouble.cpp` の `getNextAudioBlock()` 内:
   - `const juce::ScopedNoDenormals noDenormals;` を削除
   - 代わりに `ensureThreadFloatingPointEnvironment()` を呼び出し

**設計条件（前提）**:
- オーディオスレッドでは外部ライブラリが MXCSR を書き換えないこと
- 将来外部ライブラリ（MKL VML 等）が MXCSR を変更する場合のみ、スレッド起動時に `ensureThreadFloatingPointEnvironment()` を再呼び出しする設計に変更が必要

**AudioThreadCapability との整合メモ**:
- 将来的には `AudioThreadCapability` 構造体に `FloatingPointEnvironmentReady` フィールドを追加し、スレッドごとの初期化状態を管理
- 今回は `thread_local bool` のみで対応し、Future Architecture Note として残す

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `src/audioengine/AudioEngine.h` | メンバ関数宣言追加 |
| `src/audioengine/AudioEngine.Mmcss.cpp` | 実装追加 |
| `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | `ScopedNoDenormals` → `ensureThreadFloatingPointEnvironment()` |
| `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` | 同上 |

### 検証方法

- ビルド成功 + CTest 実行
- MXCSR レジスタ値をデバッガで確認（FTZ+DAZ ビットがセットされていること）
- **Denormal 入力 → CPU 負荷測証**: Denormal を含む信号を処理した際の CPU 使用率が通常演算と同等であることを確認

### リスク

- `ScopedNoDenormals` の save/restore が不要になるため、パフォーマンス向上が期待できる
- MXCSR の復元が行われなくなるが、オーディオスレッドは専用スレッドのため問題なし

---

## Phase 3: BUG-006 — processToBuffer モノラル→ステレオ複製

### ADR 対応

- なし（DSP レイヤーの局部的修正）

### 問題

`processToBuffer()` でモノラル入力時、R チャンネルが常時無音になる。

### 改修方針

モノラル入力時に Ch0 を Ch1 へ複製:

```cpp
// 既存: for (int ch = 0; ch < numChannels; ++ch) { copy... }
// 変更後:
for (int ch = 0; ch < numChannels; ++ch)
{
    const float* src = source.buffer->getReadPointer(ch, source.startSample);
    float* dst = destination.getWritePointer(ch, 0);
    juce::FloatVectorOperations::copy(dst, src, numSamples);
}

// モノラル入力時: L→R 複製
if (numChannels == 1 && destination.getNumChannels() > 1)
{
    float* dstR = destination.getWritePointer(1, 0);
    const float* dstL = destination.getReadPointer(0, 0);
    juce::FloatVectorOperations::copy(dstR, dstL, numSamples);
}
```

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `src/audioengine/AudioEngine.Processing.DSPCoreToBuffer.cpp` | 3〜5 行追加 |

### 検証方法

- モノラル入力デバイス設定で L/R 両チャンネルに信号がることを確認
- ビルド成功 + CTest 実行

### リスク

- 既存のステレオ入力パスに影響なし（`numChannels == 1` のときのみ適用）
- モノラル入力が許容されるデバイス設定がある場合のみ影響

---

## Phase 4: BUG-005 — RT-safe ヘルパー関数の DSP Foundation 層への発展

### ADR 対応

- **ADR-002**（RT Math ユーティリティ層）: DSP ユーティリティ関数を `RTMath/` へ集約し、Single Source of Truth を実現

### 問題

`isFiniteNoLibm`・`fastTanh`・`musicalSoftClipScalar` が 3 ファイルにわたり独立定義されている。

### 改修方針

**中間目標**: `FastTanhApprox.h` への集約
**最終目標**: `DSP Foundation`（または `DSP Utilities`）レイヤーへの発展
段階的に共有化:

1. **Phase 4a**: `DSPCoreDouble.cpp` のスカラーフォールバックを `FastTanhApprox.h` のスカラー版に統一
2. **Phase 4b**: `DSPCoreFloat.cpp` に `FastTanhApprox.h` を `#include` し、独自 `fastTanh` を置換
3. **Phase 4c**: `DSPCoreIO.cpp` を同様に統合

**DSP Foundation 層の Future Architecture Note**:
- 将来 `DSP Foundation`（または `DSP Utilities`）レイヤーに以下を集約:
  - `FastTanh` (Tanh 近似)
  - `SoftClip` (サチュレーション)
  - `Finite` (有限値判定)
  - `Denormal` (デノーマル対策)
  - `FastSqrt` (平方根近似)
  - `EqualPowerSin` (等電力クロスフェード)
  - `SIMD helpers` (AVX2/SSE ユーティリティ)
  - `Interpolation` (補間)
  - `Window` (窓関数)
  - `Filter` (フィルタープリミティブ)
- 今回は `FastTanhApprox.h` への集約に留め、`DSP Foundation` への発展は Future Architecture Note として残す

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` | スカラーパス統合 |
| `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | `#include` 追加 + 置換 |
| `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` | 同上 |
| `src/dsp/math/FastTanhApprox.h` | 必要に応じてスカラー API 確認 |

### 検証方法

- 各ファイルの `fastTanh` 呼び出し箇所を grep で確認
- ビルド成功 + CTest 実行
- **THD/IMD 測定**: サチュレーション特性が変化しないことを確認

### リスク

- `FastTanhApprox.h` の Policy テンプレートとのシグネチャ差異に注意
- テストカバレッジが不十分な場合、サチュレーション特性の微妙な差異を見落とす可能性

---

## Phase 5: デッドコード削除 + アサート追加

### ADR 対応

- **ADR-001**（SPSC前提）: `PublicationBuffer` の `std::mutex` は SPSC 前提に違反
- **ADR-005**（設計保証）: `static_assert` でコンパイル時に設計条件を保証
- **ADR-010**（ISR Runtime 安全性）: `getVersion()` のスレッドセーフティ保証

### BUG-003: PublicationBuffer デッドコード削除

**対象**: `ISRRuntimePublicationCoordinator.h` 内の `PublicationBuffer` クラス
**方針**: クラス定義を削除（使用箇所なし）

### BUG-004: getVersion() アサート追加

**対象**: `ISRRuntimePublicationCoordinator.cpp` の `getVersion()`
**方針**: デバッグビルドで呼び出しスレッド検証アサートを追加:
```cpp
std::uint64_t RuntimePublicationCoordinator::getVersion() const noexcept {
    // ★ ADR-010: デバッグビルド: Message Thread からの呼び出しのみ許可
    // getInstanceWithoutCreating() はヘッドレス環境（Headless Test/CLI/Batch Render）でも安全
    if (auto* mm = MessageManager::getInstanceWithoutCreating())
        jassert(mm->isThisTheMessageThread());
    return persistentState_.mappedRuntimeGeneration;
}
```

### BUG-008: std::atomic\<DSPHandle\> static_assert 追加

**対象**: `ISRDSPHandle.h` の `activeRuntimeDSPHandle_` 宣言部
**方針**: `static_assert` を追加（ADR-005: 設計をコンパイル時に保証）:
```cpp
static_assert(std::is_trivially_copyable_v<DSPHandle>,
    "DSPHandle must be trivially copyable for ISR Runtime");
static_assert(std::is_standard_layout_v<DSPHandle>,
    "DSPHandle must be standard layout for ISR Runtime");
static_assert(std::atomic<DSPHandle>::is_always_lock_free,
    "atomic<DSPHandle> must be lock-free on x64 for ISR Runtime");
// ★ 将来的に memcmp/atomic compare/バイナリ比較を前提にする場合:
// static_assert(std::has_unique_object_representations_v<DSPHandle>,
//     "DSPHandle must have unique object representations for binary comparison");
```

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `src/core/ISRRuntimePublicationCoordinator.h` | `PublicationBuffer` 削除 |
| `src/core/ISRRuntimePublicationCoordinator.cpp` | `getVersion()` にアサート追加 |
| `src/core/ISRDSPHandle.h` | `static_assert` 追加 |

---

## Phase 6: 防御的改善（将来向け・情報）

### BUG-009: RCUReader::enter() 検証順序

**現状**: 1スレッド専用のため実害なし
**将来対応**: マルチスレッド共有時に `ownerThreadToken` CAS を `nestingDepth` イクリメント前に移動

### BUG-010: 命名混乱

**現状**: 3つの異なる実体が「Coordinator」「Bridge」を交差使用
**将来対応**: Facade 移行後にリネーム。今やる必要なし。

### BUG-019: RTTraceRelay 未結線

**現状**: 準備済み未使用
**将来対応**: トレース機能が必要になったときに結線

---

## テスト計画

### 各フェーズ共通

1. **ビルドテスト**: Debug + Release の両方でビルド成功
2. **CTest**: `ctest -C Debug --output-on-failure` で全テスト合格
3. **静的解析**: CodeQL 実行（既存のクエリスイート使用）

### Phase 1 特有

- **Test A: Energy Invariance**: `gainOld² + gainNew² ≈ 1.0`（エネルギー保存の不変条件）を自動検証。判定条件: `Energy Error (dB) = 10*log10(gOld² + gNew²)` が `±0.01 dB` 以内
- **Test B: Crossfade Semantic**: `gainOld`/`gainNew`/`dryScale` の一致確認
- **Test C: Equal Power Curve Verification**: 等電力曲線 `gOld² + gNew² ≈ 1.0` が中間点でも成立することを検証
- **Shadow Compare**: Peak/RMS/Latency を比較 + 変化量確認
- float 版と double 版の出力が一致することを確認

### Phase 2 特有

- デバッガで MXCSR レジスタ値を確認（FTZ+DAZ ビットがセットされていること）
- **Denormal 入力 → CPU 負荷測証**: Denormal を含む信号を処理した際の CPU 使用率が通常演算と同等

### Phase 3 特有

- モノラル入力デバイスで L/R 両チャンネルに信号があることを確認

### Phase 4 特有

- **THD/IMD 測定**: サチュレーション特性が変化しないことを確認
- FFT 比較ではなく時波形での直接比較

---

## 進捗管理

| フェーズ | 状態 | ビルド | CTest | ADR | 備考 |
|---------|------|--------|-------|-----|------|
| Phase 1 | 未着手 | — | — | ADR-003, ADR-006 | 最優先 |
| Phase 2 | 未着手 | — | — | ADR-006 | |
| Phase 3 | 未着手 | — | — | なし | |
| Phase 4 | 未着手 | — | — | ADR-002 | DSP Foundation |
| Phase 5 | 未着手 | — | — | ADR-001, 005, 010 | |
| Phase 6 | 情報 | — | — | なし | 将来対応 |

---

## Runtime OS v1.0 整合チェックリスト

| 項目 | 状態 | 備考 |
|------|------|------|
| 各フェーズに ADR を明記 | ✅ | Phase 1-5 に ADR 対応表を記載 |
| Phase 1 に Crossfade Energy 自動検証 | ✅ | Test A/B/C の3段階テスト + Shadow Compare (変化量確認) |
| Phase 1 の dryScale は float/double 意味論一致 | ✅ | double 版にも dryScale 追加 |
| Phase 2 の FTZ/DAZ は汎用的なインターフェース名 | ✅ | `ensureThreadFloatingPointEnvironment()` |
| Phase 4 は DSP Foundation 層への発展を Future Architecture Note として残す | ✅ | `FastTanhApprox.h` → `DSP Foundation/` |

---

## 注意事項

1. **ConvoPeq.md の行番号**: 本計画書の行番号は検証時の `ConvoPeq.md`（2026-07-23 版）に基づく。実際のソースファイルでは行番号が異なる場合がある。コードパターンで照合すること。

2. **dryScale の double 版への追加**: float 版の `dryScale` が double 版にないことは新規発見である。修正する場合は、`useDryAsOld` フラグの有無と `crossfadeRuntime_.getDryScaleGain()` の呼び出しを double 版にも追加する必要がある。

3. **FTZ/DAZ の最適化**: `ScopedNoDenormals` は現在も正常に動作している。最適化はパフォーマンス上の利益のみで、機能的なバグ修正ではない。

4. **テストカバレッジ**: Phase 4（ヘルパー共有化）はサチュレーション特性に影響する可能性があるため、THD/IMD 測定を含む十分なテストカバレッジを確保すること。

5. **ADR の正式化**: 本計画書の ADR 対応表は現時点のコード分析に基づく暫定的なものである。各 ADR の正式な文書化は別途実施すること。

---

## Part 8 §K 残り2項目の調査結果

### Auto Gain Staging UI 表示ギャップ

**結論**: 問題の前提が変化している。

- `computeAndApplyAutoGain()` という関数名は現在のソースに存在しない
- `AutoGainPlanner.cpp`（ConvoPeq.md:44328）に `AutoGainPlanner::plan()` が実装済み（V2）
- `updateGainStagingDisplay()`（DeviceSettings.cpp:9660）は表示ラベルの更新のみ
- 実際の自動ゲイン値計算は `AutoGainPlanner::plan()` → `AudioEngine` への単一代入（53289行）で行われる
- **UI 表示との同期**: `DeviceSettings::updateGainStagingDisplay()` が `plan()` の出力とどう同期しているかは `DeviceSettings.cpp` の詳細確認が必要だが、v14.0 で `AutoGainPlanner` が導入されており、旧問題の前提とは大きく異なっている

### M/S 最大ゲイン計算の位相関係

**結論**: v14.30〜v14.47 で全面リファクタリング済み。

- 旧 `computeEstimatedMaxGainDb()` は `computeEstimatedMaxGainComplex()`（ConvoPeq.md:69129）に置き換え
- 新設ヘルパー群: `BandHelper::collectActiveBands`（68461）→ `EQResponseSampler::evaluate`（72622）→ `PeakEstimator::estimate`（72914）→ `UpperBoundEstimator::estimateMax`（73071）
- M/S 位相関係の扱いはこれらのヘルパー内に分散。`BandHelper` が `BandCollection` を生成し、`EQResponseSampler` が周波数応答をサンプリングする設計
- **要再調査**: M/S 位相の相殺効果が `PeakEstimator` / `UpperBoundEstimator` で正しく考慮されているかは、これらのクラスの内部ロジックを詳細に確認する必要がある

---

## 棚卸し完了確認

| 項目 | 状態 | 備考 |
|------|------|------|
| Part 7 全 finding 検証 | ✅ 完了 | 10件確認 |
| Part 8 全 finding 検証 | ✅ 完了 | 3件確認 |
| Part 8 §K 継続4項目 | ✅ 完了 | 2解決 + 2要再調査（上記） |
| ADR 対応表 | ✅ 完了 | Phase 1-5 に ADR 対応 |
| テスト計画 | ✅ 完了 | Energy Test / Shadow Compare / THD/IMD |
| 将来設計メモ | ✅ 完了 | DSP Foundation / AudioThreadCapability / CrossfadeMixContext (Future Architecture Note) |

---

*本計画書は `ConvoPeq_Part7_Part8_Verification_BugList.md` の検証結果とユーザーレビューに基づき改訂しました。2026-07-24 最終更新。*
