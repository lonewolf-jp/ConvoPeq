# 遅延コールバック安全性チェックリスト（ConvoPeq）

作成日: 2026-05-24
対象: `onClose` / `Timer` / `ThreadPool` を中心とした遅延実行クロージャ全般

## 目的

破棄済みオブジェクト参照（Use-After-Free）を防ぐため、
UI/非同期処理の遅延コールバックで満たすべき安全条件を明文化する。

---

## 0. 先に結論（今回監査サマリ）

- `launchAsync` / `callAsync` の `this` 直接捕捉は、今回修正後スキャンで未検出。
- `onClose` の直接 `this` 捕捉 1件を SafePointer 化済み。
- `ThreadPool.addJob` は全件で `SafePointer` または `WeakReference` による生存確認あり。
- `Timer::callAfterDelay` は全件で `SafePointer` ガードあり。
- `Timer` 利用クラスは大半で明示 `stopTimer()` 実施。
  - 一部（`ConvolverSettingsComponent`, `MixedPhaseOptimizationComponent`, `NoiseShaperLearningComponent`）は明示停止なしだが、JUCEの `Timer` デストラクタ停止に依存しており、現時点で直ちに不具合確定ではない。

---

## 1. 共通チェック（遅延クロージャ全般）

### 1.1 捕捉ポリシー

- [ ] 非同期/遅延実行クロージャで raw `this` を直接捕捉しない。
- [ ] `juce::Component` 派生は `juce::Component::SafePointer<T>` を優先。
- [ ] 非 `Component` は `juce::WeakReference<T>` を優先。
- [ ] 実行先で `nullptr` チェック後にのみメンバアクセスする。

### 1.2 所有権ポリシー

- [ ] self-delete（`delete this`, `unique_ptr(this)`）を禁止。
- [ ] 破棄責務は外部オーナー（`unique_ptr` 等）に一本化する。
- [ ] close 時は「非表示」と「オーナー破棄」を分離する。

### 1.3 シャットダウン整合

- [ ] デストラクタ開始時に新規遅延処理の受付フラグを閉じる。
- [ ] ChangeListener/コールバック登録を先に解除する。
- [ ] 破棄順序で参照先が先に死なないようにする。

---

## 2. `onClose` チェック

### onClose チェック項目

- [ ] `onClose` ラムダが `this` を直接捕捉していない。
- [ ] UI更新は `SafePointer` 生存確認後に実施。
- [ ] `onClose` からの再入（多重 close）で破綻しない。

### onClose 監査結果（2026-05-24）

- 修正済み: `src/MainWindow.cpp`
  - `newSettingsWindow->onClose = [this] { ... }`
  - → `SafePointer<MainWindow>` 捕捉＋ `nullptr` ガードへ変更。

判定: ✅

---

## 3. `Timer` チェック

### チェック項目

- [ ] `timerCallback()` が重い処理/ブロッキングI/Oを行わない。
- [ ] デストラクタで `stopTimer()` を明示する（推奨）。
- [ ] `timerCallback()` 内で外部参照時の前提（有効サンプルレート等）を検証する。
- [ ] timer から間接起動する遅延処理も生存チェックする。

### 監査結果（2026-05-24）

明示 `stopTimer()` あり（代表）:

- `src/MainWindow.cpp`
- `src/DeviceSettings.cpp`
- `src/SpectrumAnalyzerComponent.cpp`
- `src/audioengine/AudioEngine.CtorDtor.cpp`
- `src/convolver/ConvolverProcessor.Lifecycle.cpp`

明示 `stopTimer()` なし（要観察・改善余地）:

- `src/ConvolverSettingsComponent.cpp`
- `src/MixedPhaseOptimizationComponent.cpp`
- `src/NoiseShaperLearningComponent.cpp`

備考:

- JUCE `Timer` のデストラクタで停止されるため即不具合確定ではない。
- ただし保守性・可読性の観点では明示停止を推奨。

判定: ⚠️（運用上は許容、規律強化余地あり）

---

## 4. `ThreadPool` チェック

### ThreadPool チェック項目

- [ ] `addJob` ラムダで raw `this` を直接捕捉しない。
- [ ] `SafePointer` / `WeakReference` で生存確認する。
- [ ] UIスレッドへ戻す際も同じ生存確認を維持する。

### ThreadPool 監査結果（2026-05-24）

- `src/ConvolverControlPanel.cpp`
  - `irPreviewThreadPool.addJob([safeThis, ...])` + `callAsync` 内 guard
- `src/NoiseShaperLearner.cpp`
  - `saveThreadPool.addJob([weakSelf, ...])` + `weakSelf.get()` guard

判定: ✅

---

## 5. `callAfterDelay` / `callAsync` / `launchAsync` 連携チェック

### 遅延API連携チェック項目

- [ ] `Timer::callAfterDelay` は `SafePointer` を使う。
- [ ] `MessageManager::callAsync` は弱参照経由で実行する。
- [ ] `launchAsync` の完了コールバックは `SafePointer` で UI生存確認する。

### 遅延API連携 監査結果（2026-05-24）

- `callAfterDelay`: `src/MainWindow.cpp` 全5箇所で `SafePointer` ガードあり。
- `callAsync`: 既修正分を含め、raw `this` 直接捕捉は監査スキャンで未検出。
- `launchAsync`: `MainWindow` / `ConvolverControlPanel` / `DeviceSettings` で生存確認設計を確認。

判定: ✅

---

## 6. 実施済み改善（この監査サイクル）

1. `AudioEngine` の `callAsync([this, ...])` を `WeakReference<AudioEngine>` 化
   - 対象: `src/audioengine/AudioEngine.Learning.cpp`
2. `AudioEngineProcessor` の `callAsync([this, state])` を `WeakReference<AudioEngine>` 化
   - 対象: `src/audioengine/AudioEngineProcessor.cpp`
3. `MainWindow` の `onClose` を `SafePointer<MainWindow>` 化
   - 対象: `src/MainWindow.cpp`

---

## 7. 次回監査での推奨アクション

- [ ] `ConvolverSettingsComponent` / `MixedPhaseOptimizationComponent` / `NoiseShaperLearningComponent` のデストラクタに明示 `stopTimer()` を追加し、規律を統一。
- [ ] 新規遅延コールバック追加時に、このチェックリストをPRテンプレの確認項目へ組み込む。
- [ ] `onClose` / `callAsync` / `addJob` で raw `this` を検出する静的スキャンルールをCIに追加。

---

## 8. 監査時の最小確認コマンド（参考）

- 遅延実行ポイント抽出: `launchAsync`, `callAsync`, `callAfterDelay`, `addJob`, `onClose`
- ガード有無確認: `SafePointer`, `WeakReference`, `nullptr` 判定
- 破棄整合確認: デストラクタ内 `stopTimer`, listener解除, callback無効化

（本リポジトリでは最終的に strict scan + Debug build の通過を必須確認とする）
