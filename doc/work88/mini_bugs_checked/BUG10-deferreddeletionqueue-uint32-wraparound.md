# BUG 10: DeferredDeletionQueue uint32_t ラップアラウンドによる誤った満杯/空検出

- **発見日**: 2026-07-25
- **重要度**: High
- **影響**: 連続運用〜25時間後に DeferredDeletionQueue が誤って満杯/空を検出し、
  リタイア要求がフォールバックキューへの流出またはドロップされる

## 場所

| ファイル | 行 | コンテキスト |
|---------|-----|-------------|
| `src/DeferredDeletionQueue.h` | 80 | `enqueue()` — スロット可用性チェック |
| `src/DeferredDeletionQueue.h` | 120 | `reclaim()` — エントリ準備完了チェック |
| `src/DeferredDeletionQueue.h` | 172 | `drainAllUnsafe()` — エントリ準備完了チェック |

## 現象

`DeferredDeletionQueue` は Dmitry Vyukov の bounded MPMC アルゴリズムを採用している。
`enqueuePos` / `dequeuePos` は `std::atomic<uint32_t>`（32ビット）、
`sequences[]` も `std::atomic<uint32_t>`。

シーケンス番号は以下のように進行する：
- 初期化: `sequences[i] = i`
- enqueue 成功後: `sequences[idx] = pos + 1`
- dequeue 成功後: `sequences[idx] = scanPos + kQueueSize` (= scanPos + 4096)
- dequeuePos の後退を防ぐため、次のサイクルでは sequences[i] の値が enqueuePos より小さくなる

正しい比較は **32ビットモジュラ減算 + int32_t 再解釈** だが、
現在のコードは **intptr_t（符号付き64ビット）にゼロ拡張してから減算** している。

```cpp
// 現在（バグ）:
intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
```

### 具合的な障害シナリオ

キューが完全にドレインされた後、32ビットカウンタがラップした状況：

```
dequeuePos = enqueuePos = 0xFFFFFF00  (4,294,967,040)
sequences[i] = dequeuePos + kQueueSize
             = 0xFFFFFF00 + 4096
             = 0x00000F00  (uint32_t でラップ → 3,840)
```

プロデューサが `pos = 0xFFFFFF00` で可用性チェック：

| 方式 | seq | pos | diff | 判定 |
|------|-----|-----|------|------|
| **現在** `(intptr_t)seq - (intptr_t)pos` | 3,840 | 4,294,967,040 | **−4,294,963,200** (負) | → **満杯** (誤り！) |
| **正解** `(int32_t)(seq - pos)` | 3,840 | 4,294,967,040 | 256 (正) | → **空きあり** (正しい) |

キューは**空**でスロットは**利用可能**だが、コードは誤って**満杯**と判定し、
`enqueue()` が `false` を返してエントリをフォールバックまたはドロップする。

### 影響を受ける全3サイト

```cpp
// Line 80 — enqueue: スロット可用性
intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
if (diff == 0)     → CAS （正しく動作）
else if (diff < 0) → return false（満杯） ← 誤判定！
else               → retry （誤判定）

// Line 120 — reclaim: エントリ準備完了
const intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(scanPos + 1);
if (diff != 0) → break（空） ← エントリが存在するのに空と誤判定！

// Line 172 — drainAllUnsafe: エントリ準備完了（同上）
intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
if (diff == 0) → dequeue （正しく動作）
else           → break（空） ← 誤判定！
```

## 発現条件

- 32ビット `enqueuePos` / `dequeuePos` が約 4.3×10⁹ に達したとき
- サンプルレート 48kHz、オーディオブロックサイズ 512 サンプルの場合：
  - `enqueuePos` 増加速度 ≈ 48,000/512 ≈ 94 ops/sec（通常運用）
  - 32ビットラップ ≈ 約 4.3×10⁹ / 94 ≈ **約 18ヶ月**
  - ただし、IR ローディングやパラメータ変更が多い場合のリタイア頻度によってはより早く発生
- 約 24〜48 時間の高負荷継続テストで顕在化する可能性がある

## 修正方針

全3サイトで `intptr_t` の代わりに `int32_t` を使用し、32ビットモジュラ減算を正しく行う：

```cpp
// 修正前:
intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

// 修正後:
int32_t diff = static_cast<int32_t>(static_cast<uint32_t>(seq - pos));
```

この式は：
1. `seq - pos` を `uint32_t` で計算（モジュラ減算）
2. `int32_t` にキャストして符号付き解釈（範囲 [-2³¹, 2³¹)）
3. `kQueueSize = 4096` であるため、真の差は常にこの範囲内

また、念のため `static_assert` で `kQueueSize <= INT32_MAX` を確認することを推奨。
