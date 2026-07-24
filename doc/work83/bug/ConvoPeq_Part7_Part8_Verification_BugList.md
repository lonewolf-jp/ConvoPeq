# ConvoPeq Part 7 & Part 8 検証結果 バグリスト

**検証日**: 2026-07-24
**検証対象**: `ConvoPeq_Part7_findings_2026-07-23.md`, `ConvoPeq_Part8_findings_2026-07-23.md`
**検証方法**: `ConvoPeq.md`（81,616行）を対象にパターン検索＋前後コード確認
**注意**: Part 7/8 の行番号は生成時の古い版に基づくため、現在の `ConvoPeq.md` では行番号が異なる。コードパターンで照合。

---

## 検証結果サマリー

| Finding | Part 7/8 判定 | 検証結果 | 優先度 | 実害 |
|---------|--------------|---------|--------|------|
| **No.9** | 確認済み | ✅ **確認（＋追加不整合発見）** | 中 | クロスフェード中間点で-3dB dip |
| **No.10/11** | 確認済み（最重要） | ⚠️ **部分的に誤り** — `ScopedNoDenormals` が存在 | 中→低 | RAII毎のオーバーヘッド（実害は限定的） |
| **No.12** | 確認済み | ✅ **確認** | 低 | デッドコード（未使用） |
| **No.13** | 確認済み | ✅ **確認** | 低 | テストのみ使用、現状実害なし |
| **No.14** | 情報共有 | ✅ **確認** | 情報 | 命名混乱 |
| **No.15** | 要検証 | ✅ **確認** | 低 | 防御ロジックの穴（現状1スレッド専用） |
| **No.16** | 確認済み | ✅ **確認** | 中 | 関数分散による保守リスク |
| **No.18** | 要検証 | ✅ **確認** | 中 | モノラル入力時Rch無音 |
| **No.19** | 情報 | ✅ **確認** | 情報 | 未結線インフラ |
| **No.20** | 低・防御的 | ✅ **確認** | 低 | static_assert追加推奨 |
| **Part8 §J** | 解消 | ✅ **確認** | — | FTZ/DAZ設定スレッド解消 |
| **Part8 §K** | 2/4解決 | ✅ **確認** | — | 残2項目は再調査必要 |

---

## 詳細バグリスト

### BUG-001: No.9 — クロスフェード線形/等電力不整合（確認・追加不整合あり）

**严重度**: 中
**状態**: 確認済み（修正パッチ推奨）
**影響箇所**:
- `DSPCoreFloat.cpp`（float版）: `runLatencyAlignedCrossfadeMixLoop` ラムダ内（ConvoPeq.md:31457）
- `DSPCoreDouble.cpp`（double版）: 同上（ConvoPeq.md:32190）

**問題内容**:
1. **線形ゲイン**: `gOld = 1.0 - gNew` により中間点で `-3dB dip` 発生
2. **追加不整合**: float版には `dryScale`（31456行）があるが、double版（32190行）にはない
   - float版: `alignedOldL * dryScale * gOld`（dryScale適用あり）
   - double版: `alignedOldL * gOld`（dryScaleなし）
   - これは float版と double版の挙動不一致

**既存の正しい実装**:
- `ConvolverProcessor.Runtime.cpp:62374` — `equalPowerSin()` 定義済み
- 同ファイル 62948-62949 行 — `wg[i] = equalPowerSin(mix); dg[i] = equalPowerSin(1.0 - mix);`
- `EQProcessor.Processing.cpp:71440-71441` — `wNew = equalPowerSin(t); wOld = equalPowerSin(1.0 - t);`

**検証結果**: Part 7 の記述は正確。float版の `dryScale` 不整合は Part 7 で言及されていなかった**新規発見**。

---

### BUG-002: No.10 — FTZ/DAZ 設定（Part 7 記述に誤りあり）

**严重度**: 低〜中（オーバーヘッド問題に変更）
**状態**: Part 7 の記述は**部分的に誤り**
**影響箇所**: `AudioEngine.Processing.DSPCoreFloat.cpp:31874`, `AudioEngine.Processing.DSPCoreDouble.cpp:31116`

**Part 7 の記述（誤り）**:
> 「実際のオーディオコールバックスレッドは、アプリケーション実行中一度も自スレッド用のFTZ/DAZを設定していません」

**実際のコード（31116行・31874行）**:
```cpp
const juce::ScopedNoDenormals noDenormals;
```
JUCE の `ScopedNoDenormals` が `getNextAudioBlock()` の先頭で使用されており、**FTZ/DAZ は設定されている**。

**残る問題**:
1. **RAII オーバーヘッド**: `ScopedNoDenormals` は毎回コールバック時に MXCSR を save/restore する。スレッド起動時に1回だけ設定する方が効率的
2. **`vmlSetMode` 未呼び出し**: オーディオスレッドで Intel MKL VML を使う場合に備えて `vmlSetMode` の呼び出しがない（現状 MKL 未使用なら影響なし）
3. **`ScopedMXCSR` のコメント矛盾**: 「専用スレッドや Realtime Audio Thread では使用しない」と記載されているが、`ScopedNoDenormals` は同一スコープで使用中

**検証結果**: Part 7 は `_MM_SET_FLUSH_ZERO_MODE` の直接呼び出しのみを検索し、`ScopedNoDenormals` を見落としていた。FTZ/DAZ 自体は設定されているが、スレッド起動時1回設定への最適化は推奨。

---

### BUG-003: No.12 — PublicationBuffer デッドコード（確認）

**严重度**: 低
**状態**: 確認済み
**影響箇所**: `ISRRuntimePublicationCoordinator.h` 内（ConvoPeq.md:50911-50921）

**問題内容**:
```cpp
class PublicationBuffer {
    std::vector<const void*> queued_;  // 動的確保
    std::mutex guard_;                  // ミューテックス
};
```
- `std::mutex` + `std::vector::push_back`（動的確保を伴う）を内包
- RT-safe ファイル内に存在するが、**使用箇所が一切ない**
- 将来誤ってオーディオスレッドから結線されるリスク

**検証結果**: Part 7 の記述は正確。削除または「未使用・結線禁止」コメントの追加を推奨。

---

### BUG-004: No.13 — `getVersion()` の非 atomic アクセス（確認）

**严重度**: 低
**状態**: 確認済み
**影響箇所**: `ISRRuntimePublicationCoordinator.cpp`（ConvoPeq.md:50305-50307）

**問題内容**:
```cpp
std::uint64_t RuntimePublicationCoordinator::getVersion() const noexcept {
    return persistentState_.mappedRuntimeGeneration;  // plain struct 読み取り
}
```
- `persistentState_` は `MessageThread-only` とコメント但書あり
- 呼び出し元はテストコードのみ（77635, 77649行）
- 本番コードからの呼び出しは現時点でない

**検証結果**: Part 7 の記述は正確。現状実害なし。将来的なスレッドセーフティ向上のためアサート追加を推奨。

---

### BUG-005: No.16 — RT-safe ヘルパー関数の分散複製（確認）

**严重度**: 中
**状態**: 確認済み
**影響箇所**: 3ファイルにわたり独立定義

| 関数 | 定義箇所（ConvoPeq.md行番号） |
|------|------------------------------|
| `isFiniteNoLibm` | 32555, 33259, 33733, 70589 |
| `fastTanh` | 33386, 33800, 68297 |
| `musicalSoftClipScalar` | 32602, 33405, 33819 |

**問題内容**:
- `FastTanhApprox.h`（68263行で `#include` 確認済み）に共有実装が存在
- `DSPCoreDouble.cpp` は AVX2 パスで共有ヘッダを使用するが、**スカラーフォールバックでは独自実装を使用**
- `DSPCoreFloat.cpp`・`DSPCoreIO.cpp` は `FastTanhApprox.h` を `#include` していない可能性あり

**検証結果**: Part 7 の記述は正確。保守時の一貫性リスクあり。

---

### BUG-006: No.18 — `processToBuffer()` モノラル入力時 Rch 無音（確認）

**严重度**: 中
**状態**: 確認済み
**影響箇所**: `AudioEngine.Processing.DSPCoreToBuffer.cpp`（ConvoPeq.md:34614-34642）

**問題内容**:
```cpp
const int numChannels = std::min(2, source.buffer->getNumChannels());  // モノラル→1
for (int ch = 0; ch < numChannels; ++ch) { ... }  // Ch0 のみコピー
for (int ch = numChannels; ch < destination.getNumChannels(); ++ch)
    destination.clear(ch, 0, numSamples);  // Ch1 を無音クリア
```
- モノラル入力時: L チャンネルに信号、R チャンネルは常時無音
- 一般的にはモノラル入力は L/R 両チャンネルへ複製すべき

**検証結果**: Part 7 の記述は正確。実害の有無はデバイス設定に依存。

---

### BUG-007: No.19 — RTTraceRelay 未結線（確認）

**严重度**: 情報（低優先度）
**状態**: 確認済み
**影響箇所**: `AudioEngine.h:44024`（`convo::isr::RTTraceRelay rtTraceRelay_`）

**問題内容**:
- `enqueue()`・`drain()` の呼び出し箇所が自身の実装以外にない
- ロックフリーで RT-safe だが、準備されたが未使用のインフラ

**検証結果**: Part 8 の記述は正確。将来の結線を見据えた防御的報告。

---

### BUG-008: No.20 — `std::atomic<DSPHandle>` ロックフリー未検証（確認）

**严重度**: 低
**状態**: 確認済み
**影響箇所**: `ISRDSPHandle.h`（ConvoPeq.md:46050-46051）

**問題内容**:
```cpp
std::atomic<DSPHandle> activeRuntimeDSPHandle_{ DSPHandle::null() };
std::atomic<DSPHandle> fadingRuntimeDSPHandle_{ DSPHandle::null() };
```
- `DSPHandle` は `uint32_t slot + uint64_t generation` の16バイト構造体
- `is_always_lock_free` の `static_assert` が **3箇所のみ**（8613, 8615, 45982行）
- `std::atomic<DSPHandle>` には検証なし

**検証結果**: Part 8 の記述は正確。CMPXCHG16B で通常ロックフリーだが、`static_assert` 追加を推奨。

---

### BUG-009: No.15 — RCUReader::enter() オーナートークン検証順序（確認）

**严重度**: 低（防御的）
**状態**: 確認済み
**影響箇所**: `RCUReader.h`（ConvoPeq.md:66093-66117）

**問題内容**:
- `nestingDepth` イクリメント → `previousDepth > 0` 判定 → `ownerThreadToken` CAS の順序
- 別スレッドがネストと誤認して早期 return する可能性
- 現状は全 `RCUReader` インスタンスが1スレッド専用のため実害なし

**検証結果**: Part 7 の記述は正確。将来のマルチスレッド共有時に問題化する可能性。

---

### BUG-010: No.14 — 命名混乱（確認）

**严重度**: 情報
**状態**: 確認済み
**影響箇所**: `AudioEngine.h`（41783, 43004, 43067, 43145行付近）

**問題内容**:
- `runtimePublicationBridge_` の型は `RuntimePublicationCoordinator`
- `RuntimePublicationBridge` クラスが別途存在
- `using RuntimePublicationCoordinator = convo::RuntimePublicationCoordinator<...>` も存在
- 3つの異なる実体が「Coordinator」「Bridge」を交差使用

**検証結果**: Part 7 の記述は正確。リネームは影響範囲が広いため情報共有のみ。

---

## Part 8 §K: gain_design_spec.md 継続4項目の検証

| 項目 | Part 8 判定 | 検証結果 |
|------|------------|---------|
| `setAutoGainStagingEnabled()` 実装仕様 | 解決済み | ✅ 確認（AudioEngine.h で実装確認） |
| `buildRuntimePublishWorld()` レガシーオーバーロード | 解決済み | ✅ 確認（★v9.5 fallback で明示的補完） |
| Auto Gain Staging UI表示ギャップ | 要再調査 | ✅ 確認（`computeAndApplyAutoGain()` は存在せず、`AutoGainPlanner.cpp` で再確認必要） |
| M/S最大ゲイン計算位相関係 | 要再調査 | ✅ 確認（`computeEstimatedMaxGainComplex()` へ置き換え済み、新ヘルパー群で再確認必要） |

---

## 推奨修正優先順位

| 優先度 | Finding | 修正内容 |
|--------|---------|---------|
| **1（高）** | No.9 | `equalPowerSin` をクロスフェードラムダに適用 + double版に `dryScale` 追加 |
| **2（中）** | No.10 | `ScopedNoDenormals` をスレッド起動時1回設定に変更（RAIIオーバーヘッド削減） |
| **3（中）** | No.18 | `processToBuffer()` でモノラル入力時に Ch1 へ Ch0 を複製 |
| **4（中）** | No.16 | `DSPCoreFloat.cpp`・`DSPCoreIO.cpp` を `FastTanhApprox.h` に統一 |
| **5（低）** | No.12 | `PublicationBuffer` を削除または「未使用」コメント追加 |
| **6（低）** | No.13 | `getVersion()` にデバッグアサート追加 |
| **7（低）** | No.20 | `std::atomic<DSPHandle>` に `static_assert` 追加 |
| **情報** | No.14, 15, 19 | 命名整理、将来の拡張時に検討 |

---

*本レポートは Part 7/8 の各 finding を `ConvoPeq.md`（2026-07-23 版、81,616行）と照合して検証した結果です。行番号はバージョン間で変動するため、コードパターンで照合しました。*
