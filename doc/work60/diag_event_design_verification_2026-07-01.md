# Numeric-Only DiagEvent 設計検証レポート（付録）

**作成日**: 2026-07-01
**方法**: ソースコード調査による全実装前提の検証
**対象ファイル**: `LockFreeRingBuffer.h`, `AudioEngine.h`, `AudioEngine.Timer.cpp`

---

## 検証結果一覧

| # | 調査項目 | ステータス | ソースコード根拠 |
|---|---------|-----------|-----------------|
| 1 | `convo::exchangeAtomic` | ✅ 確認済み | `Timer.cpp:807-809`, `Timer.cpp:864`（xRunDropCount） |
| 2 | `diagSequenceCounter()` | ✅ 確認済み | `Timer.cpp:66` |
| 3 | `diagPrefix(gen)` | ✅ 確認済み | `Timer.cpp:20-28` |
| 4 | `RTAuxMutable` 構造体の位置 | ✅ 確認済み | `AudioEngine.h:1313-1403`。xRunDropCount:1402行 |
| 5 | **`LockFreeRingBuffer::push()` シグネチャ** | ✅ **`const T&`** | `LockFreeRingBuffer.h:32`。**余分なコピーは発生しない** |
| 6 | `runtimeWorld` (Timer Thread) | ✅ 確認済み | `Timer.cpp:867` 他 |
| 7 | `std::to_string` in static_assert | ✅ 問題なし | 既に除去済み、constexpr-safe |
| 8 | `enum class : uint8_t` パターン | ✅ 既存コードと同一 | `RebuildTelemetryEvent : uint8_t` 等 |
| 9 | `fetch_add(relaxed)` for stats | ✅ 既存コードと同一 | `cliTelemetryCallbackCount`（1515行） |
| 10 | **`default: jassertfalse;`** | ✅ **追加済み** | Timer側 switch に追加 |

---

## 最重要：LockFreeRingBuffer::push() のコピー動作

```cpp
// LockFreeRingBuffer.h:32 — 現行実装
bool push(const T& item) noexcept {
    size_t w = convo::consumeAtomic(writeIndex, std::memory_order_acquire);
    size_t r = convo::consumeAtomic(readIndex, std::memory_order_acquire);
    if ((w - r) >= Capacity) return false; // full
    buffer[w & MASK] = item;  // ← この1回だけのコピー代入
    convo::publishAtomic(writeIndex, w + 1, std::memory_order_release);
    return true;
}
```

- シグネチャ: **`const T&`**（const reference）→ 余分な一時オブジェクト生成なし
- 呼び出し: `diagBuffer.push(event)` — 参照渡し
- コピー回数: **1回のみ**（`buffer[slot] = item` の代入）
- **「push(T value) なら2回コピー」という懸念は該当しない。** 同一シグネチャが xRunBuffer でも使用済み。

---

## 未確定事項（本設計の範囲外）

| 事項 | 理由 | 対応 |
|------|------|------|
| ホスト callback 間隔 8-12ms の原因 | ConvoPeq 内部では計測不可 | MMCSS/ETW で外部計測 |
| DspTimingData ObserveExtra 分離 | 現状 ~80B で実用上問題なし | 将来の肥大化時に再検討 |
| バッファ容量 512 の適正 | 机上計算のみ | [DIAG_STAT] の dropRate で実測評価 |
