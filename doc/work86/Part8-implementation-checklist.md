# Part8 findings 是正 実装チェックリスト

作成日: 2026-07-25
ベース文書: `doc/work86/Part8_remediation-plan.md`

凡例: ✅ 完了 | 🔄 作業中 | ⬜ 未着手

---

## Phase 1: 小規模即時対応

### T1. No.20: `atomic<DSPHandle>` 型特性3点セット検証
**優先度: P1-High** | **種別: 🛠** | **工数: 小（1行）**

- [ ] 1. `ISRDSPHandle.h` の `// static_assert(std::atomic<DSPHandle>::is_always_lock_free` コメント解除
- [ ] 2. `is_trivially_copyable` + `is_standard_layout` + `is_always_lock_free` の3点セットを確認
- [ ] 3. icx ビルド確認

### T2. No.24: JSON バリデーションエラー文字列のエスケープ
**優先度: P3-Low** | **種別: 🛠** | **工数: 小（1ファイル）**

- [ ] 1. `ISRClosureGraphWalker.cpp` の `validationError` JSON出力部を特定
- [ ] 2. エスケープ処理（`\b`, `\f`, `\n`, `\r`, `\t`, `\"`, `\\`, 0x00-0x1F→`\u00XX`）を実装
- [ ] 3. icx ビルド確認

### T3. O-1: CriticalExitCondition.blocker 設定補完
**優先度: P3-Low** | **種別: 🛠** | **工数: 小（1行）**

- [ ] 1. `RuntimeHealthMonitor.h` の `CriticalExitBlocker` enum に `ActiveReaderRemaining` 追加
- [ ] 2. `RuntimeHealthMonitor.cpp` の readerHealthy 評価部に blocker 設定追加
- [ ] 3. icx ビルド確認

---

## Phase 2: データ喪失リスク対応

### T6. No.19: `RTTraceRelay::drain()` の結線
**優先度: P2-Medium** | **種別: 🛠** | **工数: 小（1ファイル）**

- [ ] 1. `AudioEngine.Timer.cpp` の `timerCallback` 内で drain() 呼び出し追加
- [ ] 2. icx ビルド確認

---

## Phase 3: 軽微安全対策

### T5. No.23: `ConvolverProcessor::cleanup()` コメント修正
**優先度: P2-Medium** | **種別: 📋** | **工数: コメントのみ**

- [ ] 1. `ConvolverProcessor.LoadPipeline.cpp` の `cleanup()` 第2ループコメントを実態に合わせる
- [ ] 2. icx ビルド確認（コード変更なしのため不要だが念のため）

### T7. O-2: `RuntimePolicyEngine::canExecute()` unsigned underflow ガード
**優先度: P3-Low** | **種別: 🛠** | **工数: 小（1行）**

- [ ] 1. `RuntimePolicyEngine.cpp` の `canExecute()` に early-return ガード追加
- [ ] 2. icx ビルド確認

---

## Phase 4: デッドコード整理

### T4. No.21: `advancePhase()` の削除
**優先度: P3-Low** | **種別: 🛠** | **工数: 小（関数削除）**

- [ ] 1. `ISRShutdown.h` から `advancePhase()` 宣言を削除
- [ ] 2. `ISRShutdown.cpp` から `advancePhase()` 実装を削除
- [ ] 3. 呼出元が他にないことを確認（grepで確認済み）
- [ ] 4. icx ビルド確認

---

## 統合テスト

- [ ] 1. MSVC Release/Debug ビルド
- [ ] 2. icx Release ビルド（`build.bat Release icx nopause`）
- [ ] 3. 全ワークフロースクリプト通過確認
