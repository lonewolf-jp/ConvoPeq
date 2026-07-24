# ConvoPeq Part 7 & Part 8 実装チェックリスト

**作成日**: 2026-07-24
**ベース**: `ConvoPeq_Part7_Part8_RepairPlan.md`
**原則**: 各フェーズでビルド＋CTestを実行し、段階的に改修を進める。

---

## Phase 1: クロスフェード等電力化（最優先）

### 実装前チェック

- [ ] float版とdouble版のクロスフェード処理が `dryScale` 以外も完全に一致していることを確認
  - `useDryAsOld` フラグの有無
  - `crossfadeRuntime_.getDryScaleGain()` の呼び出し順
  - ラムダ引数のシグネチャ
- [ ] `equalPowerSin` の既存使用箇所（`ConvolverProcessor.Runtime.cpp:26`）と同一アルゴリズムであることを確認

### 実装項目

- [ ] **float版**: `AudioEngine.Processing.AudioBlock.cpp:414-435` のラムダ内
  - `gOld = 1.0 - gNew` を `gOld = equalPowerSin(1.0 - gNew)` に変更
  - `gNew` も `equalPowerSin(gNew)` に変更
  - `dryScale` は既にあるためそのまま維持
- [ ] **double版**: `AudioEngine.Processing.BlockDouble.cpp:390-407` のラムダ内
  - `gOld = 1.0 - gNew` を `gOld = equalPowerSin(1.0 - gNew)` に変更
  - `gNew` も `equalPowerSin(gNew)` に変更
  - **dryScale の追加**: `useDryAsOld` フラグと `crossfadeRuntime_.getDryScaleGain()` の呼び出しを追加
- [ ] `equalPowerSin` の `#include` を確認（既に存在する場合は不要）

### 検証

- [ ] ビルド成功（Debug + Release）
- [ ] CTest 実行
- [ ] Test A: Energy Invariance (`Energy Error (dB) = 10*log10(gOld² + gNew²)` が `±0.01 dB` 以内)
- [ ] Test B: Crossfade Semantic (`gainOld`/`gainNew`/`dryScale` の一致確認)
- [ ] Test C: Equal Power Curve Verification (等電力曲線が中間点でも成立)

---

## Phase 2: FTZ/DAZ スレッド起動時1回設定化

### 実装前チェック

- [ ] `tryApplyMmcssForSelfManagedThread()` の設計パターン（`thread_local bool` ガード）を確認
- [ ] `ScopedNoDenormals` の現在の使用箇所を確認

### 実装項目

- [ ] `AudioEngine.h` にメンバ関数宣言追加: `void ensureThreadFloatingPointEnvironment();`
- [ ] `AudioEngine.Mmcss.cpp` に実装追加
- [ ] `DSPCoreFloat.cpp` の `getNextAudioBlock()` 内: `ScopedNoDenormals` → `ensureThreadFloatingPointEnvironment()`
- [ ] `DSPCoreDouble.cpp` の `getNextAudioBlock()` 内: 同上

### 検証

- [ ] ビルド成功（Debug + Release）
- [ ] CTest 実行
- [ ] MXCSR レジスタ値をデバッガで確認（FTZ+DAZ ビットがセットされていること）

---

## Phase 3: processToBuffer モノラル→ステレオ複製

### 実装前チェック

- [ ] `processToBuffer()` の現在の実装を確認
- [ ] モノラル入力デバイスの設定を確認

### 実装項目

- [ ] `AudioEngine.Processing.DSPCoreToBuffer.cpp` にモノラル入力時の L→R 複製ロジックを追加

### 検証

- [ ] ビルド成功（Debug + Release）
- [ ] CTest 実行

---

## Phase 4: RT-safe ヘルパー関数の DSP Foundation 層への発展

### 実装前チェック

- [ ] `FastTanhApprox.h` の現在の API を確認
- [ ] 各ファイルの `fastTanh` 呼び出し箇所を grep で確認

### 実装項目

- [ ] **Phase 4a**: `DSPCoreDouble.cpp` のスカラーフォールバックを `FastTanhApprox.h` のスカラー版に統一
- [ ] **Phase 4b**: `DSPCoreFloat.cpp` に `FastTanhApprox.h` を `#include` し、独自 `fastTanh` を置換
- [ ] **Phase 4c**: `DSPCoreIO.cpp` を同様に統合

### 検証

- [ ] ビルド成功（Debug + Release）
- [ ] CTest 実行
- [ ] THD/IMD 測定（サチュレーション特性が変化しないこと）

---

## Phase 5: デッドコード削除 + アサート追加

### 実装前チェック

- [ ] `PublicationBuffer` の使用箇所がないことを確認
- [ ] `getVersion()` の呼び出し元を確認
- [ ] `DSPHandle` の構造を確認

### 実装項目

- [x] **BUG-003**: `ISRRuntimePublicationCoordinator.h` の `PublicationBuffer` クラスを削除
- [x] **BUG-004**: `ISRRuntimePublicationCoordinator.cpp` の `getVersion()` にプレースホルダアサート追加（TODO コメント付き）
- [x] **BUG-008**: `ISRDSPHandle.h` に `static_assert` 3点セット追加

### 検証

- [ ] ビルド成功（Debug + Release）
- [ ] CTest 実行

---

## 進捗管理

| フェーズ | 状態 | ビルド | CTest | 備考 |
|---------|------|--------|-------|------|
| Phase 1 | 未着手 | — | — | 最優先 |
| Phase 2 | 未着手 | — | — | |
| Phase 3 | 未着手 | — | — | |
| Phase 4 | 未着手 | — | — | DSP Foundation |
| Phase 5 | 実装済み | ✅ | ✅ | プレースホルダ実装（TODO付き） |

---

*本チェックリストは `ConvoPeq_Part7_Part8_RepairPlan.md` の改修計画に基づき策定しました。*
