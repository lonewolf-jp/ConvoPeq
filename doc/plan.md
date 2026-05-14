# ConvoPeq 最終改修計画書

本計画書は、全ソースコードのアーキテクチャ不変条件監査に基づき、ConvoPeq を **「形式的に正しさが保証された並行 Immutable DSP Runtime System」** へと完成させるための最終改修計画を定義します。  
改修の優先度は **Critical（即時）＞ High（短期）＞ Medium（中長期）** とし、すべての項目はアーキテクチャ不変条件の回復または強化を目的とします。

---

## 1. アーキテクチャ不変条件

改修の前提として、以下の不変条件を定義します。すべての修正はこれらの不変条件を回復・強化するために行われます。

| 不変条件 | 内容 |
|----------|------|
| **IR‑1** | レンダーフェーズ不変性 — Audio Thread は不変スナップショットの消費のみを行い、状態遷移・初期化を禁止する |
| **IR‑2** | 単一公開世界 — 全ランタイム状態は単一構造体の原子的な世界切り替えとして公開される |
| **IR‑3** | 所有権固定の状態変更 — 状態変更は特定のスレッド／フェーズにのみ所有され、レンダーフェーズへ漏洩しない |
| **IR‑4** | 隠蔽同期の禁止 — Audio Thread 内でロック・libm・動的メモリ確保・遅延初期化を完全排除する |
| **IR‑5** | エポック一貫観測 — オブザーバは同一エポックに属する完全なスナップショットのみを観測する |
| **IR‑6** | 所有権移転 — 任意の mutable オブジェクトは単一フェーズのみが所有し、publish 後は旧所有者の変更を禁止する |
| **IR‑7** | クロスフェード隔離 — 新旧ランタイムの可変状態は完全分離され、共有可変リソースを禁止する |

---

## 2. 🔴 Critical（即時改修：不変条件違反の回復）

### 2.1 `LinearRamp::reset()` の Audio Thread 呼び出し
**違反する不変条件**: IR‑1, IR‑3, IR‑6  
**問題の本質**: フェーズ遷移（ランプ初期化）がレンダーフェーズに侵入し、API 契約違反と並行アクセス危険を生じている。  
**修正方法**:
- `armCrossfadeIfPending()` から `dspCrossfadeGain.reset()` と `setCurrentAndTargetValue(0.0)` を削除。
- これらの初期化を **`commitNewDSP()`（Message Thread）** へ移動。
- Audio Thread が実行するのは、**事前準備された不変遷移の activate** のみとする。

**推奨する設計パターン**:
```cpp
// Message Thread
void prepareCrossfade(double sampleRate, double fadeTimeSec) {
    // 遷移の準備（不変設定の計算）
}

// Audio Thread
void activatePreparedCrossfade() {
    // 事前計算された遷移をアトミックに発動
}
```

### 2.2 `juce::SmoothedValue` の置換
**違反する不変条件**: IR‑4  
**問題の本質**: `bypassFadeGain` と `smoothTotalGain` で使用されている `juce::SmoothedValue` が `std::pow`/`std::exp`（libm）を呼び出し、決定論的レイテンシを損なう。  
**修正方法**:
- `EQProcessor.h` の該当メンバを `convo::LinearRamp` に置き換え。
- `prepareToPlay()` で `linearRamp.reset()` を呼び出し、初期値を設定。
- Audio Thread では `setTargetValue()` のみを使用。

**知覚品質への配慮**: バイパスフェードや出力トリムで指数曲線が必要な場合は、`value *= precomputedCoeff` 型の RT‑safe 乗算平滑化器を別途実装する。

### 2.3 不変ランタイムの純度維持
**違反する不変条件**: IR‑1, IR‑3, IR‑6  
**問題の本質**: `syncParametersFrom()` 等が、公開済みの `activeDSP` を Message Thread から変更可能にしており、immutable runtime 原則に違反する。  
**修正方法**:
- `activeDSP` に対するすべての `mutable` 操作を**全面的に禁止**し、`syncParametersFrom()` を削除。
- 設定変更は必ず**新規 `DSPCore` の構築と原子差し替え**で行う。
- Audio Thread が触れる `DSPCore` 参照を **`const` 修飾**し、コンパイラによる mutation 検出を保証する。

不変トポロジ（IR 構成）と可変パラメータ（ゲイン等）を分離する**階層的不変構造**の導入を推奨:
```cpp
// トポロジは不変、パラメータだけを安価に差し替え
std::atomic<IRTopology*> topology;
std::atomic<ParameterSnapshot*> parameters;
```

### 2.4 公開境界の形式化
**違反する不変条件**: IR‑2, IR‑5  
**問題の本質**: 複数の atomic 変数に分散した公開が publication domain の分裂を引き起こし、スナップショット不整合のリスクを生じている。  
**修正方法**:
- Publication Edge 専用テンプレート関数を導入し、`memory_order_release`/`acquire` を強制:
  ```cpp
  template<typename T>
  inline void publishAtomic(std::atomic<T*>& dst, T* value) noexcept {
      dst.store(value, std::memory_order_release);
  }
  ```
- 可能な限り複数の atomic 変数を `RuntimeSnapshot` 構造体に集約し、単一の atomic 差し替えで公開する。

---

## 3. 🟠 High（短期改修：所有権とライフサイクルの形式化）

### 3.1 `RCUReader` の所有権とコピー安全性
**違反する不変条件**: IR‑5, IR‑6  
**修正方法**: コピー／ムーブを明示的に禁止:
```cpp
RCUReader(const RCUReader&) = delete;
RCUReader& operator=(const RCUReader&) = delete;
RCUReader(RCUReader&&) = delete;
RCUReader& operator=(RCUReader&&) = delete;
```

### 3.2 例外安全なメモリ確保
**影響範囲**: すべての NUC、コンボリューション、FFT ワークバッファ  
**修正方法**: 戻り値を `aligned_unique_ptr<T>` とする例外安全な確保関数を導入:
```cpp
template<typename T, typename... Args>
aligned_unique_ptr<T> aligned_make_unique(Args&&... args) {
    void* mem = convo::aligned_malloc(sizeof(T), 64);
    if (!mem) throw std::bad_alloc{};
    try {
        return aligned_unique_ptr<T>(new (mem) T(std::forward<Args>(args)...));
    } catch (...) {
        convo::aligned_free(mem);
        throw;
    }
}
```

### 3.3 `commitNewDSP` のデッドロック除去
**修正方法**: `sendChangeMessage()` を `rebuildMutex` スコープ外に移動し、Flood 防止のための合体フラグを導入:
```cpp
if (!pendingChangeNotification.exchange(true))
    triggerAsyncUpdate();
```

---

## 4. 🟡 Medium（中長期改善：ロバスト性と可観測性）

### 4.1 シャットダウン状態の一元管理
**修正方法**: `shutdownInProgress` と `gShuttingDown` を**インスタンスローカルな** `EngineLifecycleState` に統合し、状態遷移を `compare_exchange_strong` でカプセル化する。

### 4.2 再構築理由の形式化
**修正方法**: 再構築要求を `RebuildReason` ビットマスクで伝達し、**再構築依存グラフ**（例: SR 変更 → トポロジ再構築必須、ゲイン変更 → パラメータ更新のみ）を導入する。

### 4.3 `NoiseShaperLearner` の状態機械化
**修正方法**: `Idle → Starting → Running → Stopping` の状態機械を導入し、`Stopping` 中の開始を拒否。`std::jthread` で協調的キャンセルを実現する。

---

## 5. 実装ロードマップ

| フェーズ | 優先度 | 内容 | 期間 |
|----------|--------|------|------|
| 1 | Critical | `LinearRamp` 修正、`SmoothedValue` 置換、`syncParametersFrom` 削除、Publication Edge マクロ導入 | 2 週間 |
| 2 | High | `RCUReader` コピー禁止、`aligned_make_unique` 導入、デッドロック修正 | 1 週間 |
| 3 | Medium | 状態機械の形式化、`memory_order` 統一、ドキュメント化 | 4 週間 |

---

## 6. 検証計画

- **静的解析**: 改修後、`ASSERT_NON_RT_THREAD` 違反が発生しないことを確認。
- **スレッド安全性**: ThreadSanitizer による長時間ストレステスト（RCU 境界は補助扱い）。
- **ARM 環境試験**: Apple Silicon で長時間オーディオ連続再生試験。
- **メモリ安全性**: AddressSanitizer で改修後のメモリリークをチェック。

---

## 7. 最終目標

本計画を完遂することで、ConvoPeq は **「決定論的並行 Immutable DSP Runtime System」** へと完成します。  
現在の課題は DSP アルゴリズムや SIMD 最適化ではなく、**所有権代数、公開意味論、クロスフェード時間隔離、再構築依存オーケストレーション** という、リアルタイムシステムの形式的保証の領域にあります。  
これらの形式化こそが、x86／ARM 両アーキテクチャ上での長期的信頼性を確立する唯一の道です。