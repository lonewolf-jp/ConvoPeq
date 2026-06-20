# Architecture Debt 詳細調査レポート

**作成日**: 2026-06-20
**対象**: No.3 (ConvolverState 経路), No.7 (partitionData 経路)
**使用ツール**: grep, Serena MCP, CodeGraph MCP, AiDex MCP, semble, 直接ファイル読取

---

## 1. ConvolverState 経路 — 全体像

### 1.1 構成要素

| 要素 | ファイル | 役割 |
|------|---------|------|
| `ConvolverState` (struct) | `src/ConvolverState.h` L43-221 | FFT周波数領域パーティション + 作業バッファを保持 |
| `ConvolverState.cpp` | `src/ConvolverState.cpp` | `stateIdCounter` + `generateNewStateId()` のみ |
| `ConvolverRuntime` (struct) | `src/ConvolverRuntime.h` L8-65 | オーバーラップ/入出力バッファのラッパー |
| `SafeStateSwapper` | `src/SafeStateSwapper.h` | Epoch-based RCU 管理 |
| `DeferredFreeThread` | `src/DeferredFreeThread.h` | retired ConvolverState の非同期解放 |

### 1.2 ConvolverState のメンバ

| メンバ | 型 | 用途（設計上） | 現状 |
|--------|-----|--------------|------|
| `partitionData` | `double*` | IR周波数領域パーティション | **読まれていない** |
| `overlapBuffer` | `std::atomic<double*>` | OLAオーバーラップ | **読まれていない** |
| `inputBuffer` | `std::atomic<double*>` | FFT入力作業領域 | **読まれていない** |
| `outputBuffer` | `std::atomic<double*>` | FFT出力作業領域 | **読まれていない** |
| `partitionSizeBytes` | `size_t` | パーティションデータサイズ | 保存/復元のみ |
| `numPartitions` | `int` | パーティション数 | **コピー専用** |
| `fftSize` | `int` | FFTサイズ | **コピー専用** |
| `generationId` | `uint64_t` | 世代管理 | 書き込みのみ |
| `fftHandle` | `ScopedDftiDescriptor` | MKL DFTI descriptor | **未使用** |
| `stateId` | `uint64_t` | 状態識別子 | ✅ Snapshotで使用 |

### 1.3 参照関係の完全トレース

```
ConvolverState の全参照箇所 (grep: 80 matches, Serena: 全確認済み)
```

#### Write Path （書き込み）— 全4箇所

| # | 関数 | ファイル | 行 | 内容 | スレッド |
|---|------|---------|----|------|---------|
| 1 | `ConvolverState()` コンストラクタ | `ConvolverState.h:92-112` | 全メンバ初期化 | Message |
| 2 | `ConvolverState` ムーブコンストラクタ | `ConvolverState.h:169-185` | 所有権移転 | Message |
| 3 | `ConvolverState` ムーブ代入 | `ConvolverState.h:192-216` | 所有権移転 | Message |
| 4 | `applyComputedIR()` → `updateConvolverState()` | `LoadPipeline.cpp:488-502` | `newState` 生成 + `rcuSwapper.swap()` | Message |

#### Read Path （読み取り）— 書き込み以外の全参照

| # | 関数 | ファイル | 行 | 読むメンバ | スレッド |
|---|------|---------|----|-----------|---------|
| 1 | `isCacheEntrySafeToDelete()` | `LoadPipeline.cpp:219` | `stateId` (via `rcuSwapper.getState()`) | **Message** |
| 2 | `createSnapshotFromCurrentState()` | `AudioEngine.Snapshot.cpp:28` | `stateId` (via `getConvolverState()`) | **UI (非Audio)** |
| 3 | `cleanup()` | `ConvolverState.h:155-162` | `partitionData`, `overlapBuffer`, etc. 解放 | DeferredFree |
| 4 | `getActiveCoeffSet()` | `AudioEngine.h` (コメントのみ) | — | — |

**→ `ConvolverState` の全メンバのうち、Audio Thread から参照されているものは存在しない。**

### 1.4 `SafeStateSwapper` / `rcuSwapper` の使用状況

```
rcuSwapper の全参照箇所 (grep: 4 matches)
```

| 関数 | ファイル | 行 | 内容 |
|------|---------|----|------|
| `rcuSwapper.swap(newState)` | `StateAndUI.cpp:1028` | ConvolverState 書込み |
| `rcuSwapper.getState()` | `LoadPipeline.cpp:219` | `isCacheEntrySafeToDelete()` 内で読取り |
| `rcuSwapper.tryReclaim(...)` | `Lifecycle.cpp:417` | シャットダウン時の強制回収 |
| `rcuSwapper` → `DeferredFreeThread` | `Lifecycle.cpp:354` | バックグラウンド回収スレッド |

**→ Audio Thread から `rcuSwapper.getState()` の呼び出しはない。**

---

## 2. ConvolverRuntime 経路 — 全体像

### 2.1 ConvolverRuntime のメンバ

| メンバ | 型 | 用途 | 現状 |
|--------|-----|------|------|
| `overlapBuffer` | `ScopedAlignedPtr<double>` | OLAオーバーラップ | **未使用** |
| `inputBuffer` | `ScopedAlignedPtr<double>` | FFT入力 | **未使用** |
| `outputBuffer` | `ScopedAlignedPtr<double>` | FFT出力 | **未使用** |
| `currentFFTSize` | `int` | FFTサイズ追跡 | **比較専用** |
| `currentNumPartitions` | `int` | パーティション数追跡 | **比較専用** |

### 2.2 全参照箇所

| 関数 | ファイル | 行 | 内容 |
|------|---------|----|------|
| `runtime.reallocate()` | `LoadPipeline.cpp:500` | 書込み (Message Thread) |
| `runtime.clear()` | `Lifecycle.cpp:420` | 解放 (Message Thread) |
| `runtime.reset()` | — | **誰も呼んでいない** |

**→ `ConvolverProcessor::process()` (Audio Thread) 内での `runtime` 参照は0件。**

---

## 3. 設計遺産の定量評価

### 3.1 メモリリークリスク

`ConvolverState::cleanup()` は以下のリソースを解放する:

| リソース | 確保方法 | 解放確認 |
|---------|---------|---------|
| `partitionData` | `IRConverter::convertFile()` → `mkl_malloc(bytes, 64)` | ✅ `cleanup()` → `convo::aligned_free()` |
| `overlapBuffer` | `ConvolverState()` コンストラクタ → `makeAlignedArray` | ✅ `cleanup()` → `exchangeAtomic` + `aligned_free` |
| `inputBuffer` | 同上 | ✅ 同上 |
| `outputBuffer` | 同上 | ✅ 同上 |
| `fftHandle` | `DftiCreateDescriptor` | ✅ `ScopedDftiDescriptor::~ScopedDftiDescriptor()` |

**→ メモリリークはない。解放パスは正常に動作する。**

### 3.2 CPU オーバーヘッド

- `ConvolverState` のコンストラクタで 3 x `makeAlignedArray<double>(fftSize)` + 1 x `DftiCreateDescriptor` が毎回実行される
- `applyComputedIR()` のパス（cache hit + `loadIR()`）で毎回これらの確保が発生
- **無駄なメモリ確保 + MKL descriptor 生成**が発生している

### 3.3 コード複雑性への影響

| メトリクス | 値 |
|-----------|-----|
| `ConvolverState.h` 行数 | 221行 |
| `ConvolverRuntime.h` 行数 | 65行 |
| `SafeStateSwapper.h` 行数 | 350行 |
| `DeferredFreeThread.h` 行数 | 180行 |
| `ConvolverState.cpp` 行数 | 15行 |
| **設計遺産コード合計** | **約831行** |
| 関与するファイル数 | 5ファイル |
| 維持すべきテスト | ISRSemanticValidationTests, PartialPublicationRejectTests の一部 |

---

## 4. 整理オプションと影響評価

### オプションA: 現状維持（最小変更）

**内容**: 何もしない。現在の状態を維持。
**リスク**:

- 開発者の混乱（デッドコードの存在）
- 無駄なメモリ確保・解放が継続
- 保守コストが残る

### オプションB: 軽量化（推奨）

**内容**:

- `ConvolverState` から `partitionData`, `overlapBuffer`, `inputBuffer`, `outputBuffer`, `fftHandle` を削除
- `ConvolverRuntime` から `overlapBuffer`, `inputBuffer`, `outputBuffer`, `currentNumPartitions` を削除
- `stateId` のみ保持（スナップショット用）
- `SafeStateSwapper` + `DeferredFreeThread` は維持（`stateId` の RCU 管理に必要）

**影響**:

| 項目 | 影響 |
|------|------|
| `CacheManager` フォーマット | 要変更: `header.numPartitions` は保存継続、`partitionData` は save/load 不要に |
| `ConvolverState` コンストラクタ | シグニチャ変更: `(uint64_t genId, double sr)` に縮小 |
| `applyComputedIR()` | `runtime.reallocate()` 呼び出し削除 + `ConvolverState` 構築を簡略化 |
| `cleanup()` | `partitionData`/`overlapBuffer`/`inputBuffer`/`outputBuffer` 解放削除 |
| `PreparedIRState` | `partitionData`/`partitionSizeBytes` フィールド削除 |
| `IRConverter::convertFile()` | `partitionData` 確保 + コピーを削除 |
| **工数見積** | **2〜3人日** |

### オプションC: 全削除（高リスク）

**内容**: `ConvolverState`, `ConvolverRuntime`, `SafeStateSwapper`, `DeferredFreeThread` をすべて削除。
**リスク**:

- `SafeStateSwapper` は現在も `rcuSwapper` として使用中（write path は生きている）
- 削除すると `updateConvolverState()` の経路全体を再設計が必要
- `CacheManager` のキャッシュフォーマット変更が必要（後方互換性喪失）
- テストの大規模修正
- **推奨しない**

---

## 5. 推奨アクション

```
優先度: オプションB（軽量化） > オプションC（全削除）は非推奨
```

### 即時対応不要（継続監視）

- メモリリークなし
- Audio Thread のパフォーマンスに影響なし
- 既存の動作に影響なし

### 次回アーキテクチャ整理時の TODO

1. `ConvolverState` の軽量化（`stateId` のみ保持）
2. `ConvolverRuntime` の削除（`runtime` メンバごと）
3. `PreparedIRState::partitionData` の削除
4. `IRConverter::convertFile()` の partitionData 確保処理の削除
5. `CacheManager` のキャッシュフォーマット更新（version 3）
6. 後方互換性: 旧キャッシュ（version 2）からの読み取り対応を維持

---

## 6. 付録: 全ツール確認結果

### AiDex MCP 確認

| 検索対象 | 結果 |
|---------|------|
| `ConvolverState` 80 matches | 全ファイル網羅確認 |
| `partitionData` の `->` 参照 | **0件** |

### Serena MCP 確認

| シンボル | 外部参照 |
|---------|---------|
| `ConvolverState::partitionData` | コンストラクタ/ムーブ/デストラクタのみ |
| `ConvolverState::numPartitions` | コンストラクタ/ムーブのみ |
| `ConvolverRuntime` | 宣言のみ (`ConvolverProcessor.h:1159`) |
| `SafeStateSwapper::getState()` | 内部経路 (`rcuSwapper.getState()`) |

### CodeGraph MCP

| エンティティ | コミュニティ | 次数 |
|-------------|------------|------|
| `ConvolverRuntime` | 8 | 6 (全6件が宣言/定義のみ) |

### grep 確認

| パターン | 件数 |
|---------|------|
| `runtime.reallocate` | **1** (LoadPipeline.cpp:500, write only) |
| `runtime.clear` | **1** (Lifecycle.cpp:420, cleanup only) |
| `runtime.reset` | **0** (誰も呼んでいない) |
| `rcuSwapper.swap` | 1 (StateAndUI.cpp:1028, write) |
| `rcuSwapper.getState` | 1 (LoadPipeline.cpp:219, Message Thread) |

### semble 確認

- セマンティック検索の全結果が上記数値と一致
- Audio Thread からの読み取り経路は存在しないことを確認
