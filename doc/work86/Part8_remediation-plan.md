# ConvoPeq Part8 findings 是正改修計画書

作成日: 2026-07-25（最終更新: 2026-07-25）
ベース文書: `doc/work86/ConvoPeq_Part8_findings_2026-07-23.md`
ステータス: 設計判断完了・実装可能

---

## 0a. 調査確定事項（2026-07-25 追記）

本計画書作成後に追加調査を実施し、以下の2項目を確定した。

| ID | 項目 | 調査結果 | 設計判断 |
|----|------|---------|---------|
| **D1** | NoiseShaperLearner RT結線経路 | **解決済み**: `publishGenerationResult()` → `storeLearnedCoeffsToBank()` → `bank.generation++` → RT generation tracking の完全経路を確認。`bestCoefficients` のatomic更新後、`callAsync` 経由でcoefficient bankに書き込まれ、世代番号の変更をRT側が検出する。`adaptiveNoiseShaper.setCoefficients()` はこのbankから読み込まれる。 | 本件はバグではなく、設計通りの正常な動作であった。文書 `§N` を修正済み。 |
| **D2** | M/S最大ゲイン位相関係 | **構造的制約として継続**: `BandHelper::collectActiveBands()` は単一バンド配列のみ走査。M/Sエンコード時の周波数依存の位相関係を考慮しておらず、最大〜3dBの過小評価リスクがある。ただし過小評価側（安全側）のため緊急対応不要。 | 将来の精度向上タスクとして記録。M/S有効時に補正係数（+3dB）を適用するオプションを追加可能。 |

---

## 凡例

| 記号 | 意味 |
|------|------|
| 🛠 | コード修正を伴う |
| ⚡ | ISR Runtime 安全性要確認 |
| 📋 | 文書化・設計判断のみ |

---

## 0. バグ一覧と優先度

| ID | 項目 | 種別 | 優先度 | 現状 |
|----|------|------|--------|------|
| 21 | `advancePhase()` スキップロジックバグ | 🛠 | P2-Medium | `ReclaimComplete→VerifyDrained` が `EmergencyDrain` を非terminalのままスキップ |
| 20 | `atomic<DSPHandle>` static_assert コメントアウト | 🛠 | P2-Medium | コメント解除が必要。CMPXCHG16B依存のためビルド時に検証すべき |
| 23 | `cleanup()` 強制削除未実装 | 🛠 | P2-Medium | 第2ループも `waitForThreadToExit(0)` のみ。コメントと実装に乖離 |
| 24 | JSON未エスケープ | 🛠 | P3-Low | CI診断出力のみ。`validationError` に `"` が含まれると不正JSON |
| 19 | `RTTraceRelay::drain()` 未結線 | 🛠 | P3-Low | RTで `enqueue` 済だが `drain` 未呼出。4096エントリ溢れでデータ喪失 |
| O-1 | `blocker` 設定漏れ | 🛠 | P3-Low | Reader残留単独原因時の診断blocker欠落 |
| O-2 | cooldown unsigned underflow | 🛠 | P3-Low | `uint64_t` 減算のアンダーフローリスク |

---

## 1. Phase 1: 小規模即時対応

### T1. No.20: `atomic<DSPHandle>` 型特性3点セット検証

**優先度: P1-High** | **種別: 🛠** | **工数: 小（1行）**

#### 問題
`ISRDSPHandle.h` で `DSPHandle` の型要件のうち `is_trivially_copyable` / `is_standard_layout` は static_assert されているが、`std::atomic<DSPHandle>` のロックフリー性がコンパイル時検証されていない。16バイト構造体のatomicはCMPXCHG16B命令に依存する。icxでは `is_always_lock_free` がコンパイル時保証されないため、Runtime初期化時に `is_lock_free()` で検証する。

#### 修正内容
```cpp
// DSPHandle 型要件:
//   trivially_copyable → static_assert（コンパイル時保証）
//   standard_layout    → static_assert（コンパイル時保証）
//   lock-free atomic   → Runtime初期化時に is_lock_free() 検証
static_assert(std::is_trivially_copyable_v<DSPHandle>,
    "DSPHandle must be trivially copyable for ISR Runtime");
static_assert(std::is_standard_layout_v<DSPHandle>,
    "DSPHandle must be standard layout for ISR Runtime");

// DSPHandleRuntime コンストラクタ内:
std::atomic<DSPHandle> test{ DSPHandle::null() };
assert(test.is_lock_free() && "atomic<DSPHandle> must be lock-free on x64");
```

#### リスク
- **極低**: プロジェクトは `/arch:AVX2`(MSVC) / `/QxCORE-AVX2`(icx) でビルドされ、CMPXCHG16B対応のHaswell以降が前提。

---

### T2. No.24: JSON バリデーションエラー文字列のエスケープ

**優先度: P3-Low** | **種別: 🛠** | **工数: 小（1ファイル）**

#### 問題
`ISRClosureGraphWalker.cpp:79` で `validationError`（`std::string_view`）をJSON文字列値として書き出す際、ダブルクォート・バックスラッシュ・制御文字のエスケープ処理がない。

#### 修正内容
```cpp
// 変更前:
if (!valid) {
    file << "\"" << std::string(validationError) << "\"";
}

// 変更後:
if (!valid) {
    // JSON仕様(RFC 8259)に従い全制御文字をエスケープ
    file << "\"";
    for (char c : validationError) {
        switch (c) {
            case '"':  file << "\\\""; break;
            case '\\': file << "\\\\"; break;
            case '\b': file << "\\b";  break;  // U+0008
            case '\f': file << "\\f";  break;  // U+000C
            case '\n': file << "\\n";  break;  // U+000A
            case '\r': file << "\\r";  break;  // U+000D
            case '\t': file << "\\t";  break;  // U+0009
            default:
                // その他の制御文字(0x00-0x1F)は \u00XX 形式
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[16];
                    snprintf(buf, sizeof(buf), "\\u%04x",
                        static_cast<unsigned int>(static_cast<unsigned char>(c)));
                    file << buf;
                } else {
                    file << c;
                }
                break;
        }
    }
    file << "\"";
}
```

**補足**: JSON 仕様（RFC 8259 §7）上、UTF-8 のマルチバイト文字（日本語等）はエスケープ不要である。本エスケープ処理はバイト単位で動作するが、制御文字（U+0000-U+001F）のみを対象としており、マルチバイト文字の各バイト（0x80以上）は `else { file << c; }` でそのまま出力されるため問題ない。

#### リスク
- **極低**: CI証跡出力専用（`std::ofstream`・`std::filesystem` 使用）で Runtime 非関与。

---

### T3. O-1: CriticalExitCondition.blocker 設定補完

**優先度: P3-Low** | **種別: 🛠** | **工数: 小（1行）**

#### 問題
`RuntimeHealthMonitor.cpp` のCritical出口評価で、`readerHealthy` のみが原因で `metricsHealthy=false` となるケースに対応する `blocker` 設定がない。

#### 修正内容
```cpp
// 変更前（readerHealthy単独原因でblocker未設定）:
// readerHealthy の評価のみで blocker 設定なし

// 変更後:
const bool readerHealthy = (m_retireRouter == nullptr
    || m_retireRouter->activeReaderCount() == 0);
if (!readerHealthy) {
    exitCond.blocker = CriticalExitBlocker::ActiveReaderRemaining;
}
```

**注意**: `CriticalExitBlocker` enum に `ActiveReaderRemaining` 値を追加する。

#### 影響範囲調査結果
`CriticalExitBlocker` enum は `RuntimeHealthMonitor.h` 内でのみ定義・使用:
- **switch文**: `CriticalExitBlocker` に対するswitch文は存在しない
- **参照元**: 設定箇所は `RuntimeHealthMonitor.h:103-107`（`canExit()`）と `RuntimeHealthMonitor.cpp:255,257`（retire関連）の計3箇所のみ
- **読み取り元**: 診断情報として書き込まれるのみ。他モジュール（Shutdown/Diagnostics/Telemetry/UI）から参照なし
- **結論**: enum値追加の影響は RuntimeHealthMonitor 内部に閉じる。安全。

#### リスク
- **極低**: 診断情報のみ。安全動作（Critical状態からの誤った離脱防止）には影響なし。

---

## 2. Phase 2: データ喪失リスク対応

### T6. No.19: `RTTraceRelay::drain()` の結線

**優先度: P2-Medium** | **種別: 🛠** | **工数: 小（1ファイル）**

#### 問題
`RTTraceRelay::enqueue()` は `AudioBlock.cpp:304` から呼ばれているが、`drain()` が誰からも呼ばれていない。`RELAY_BUFFER_SIZE = 4096` で満杯になると古いイベントが上書きされる。

#### `drain()` 実装監査結果
`ISRRTExecution.cpp` の `drain()` を監査:
- ✅ メモリ確保なし（stack only）
- ✅ ロギングなし（juce::Logger/file I/O 不使用）
- ✅ ブロッキング操作なし（mutex/lock/sleep 不使用）
- ✅ vector/コンテナ操作なし
- ✅ lock-free atomics のみ

**結論**: Timer スレッドから安全に呼び出せる。

#### 修正内容
```cpp
// AudioEngine.Timer.cpp の timerCallback 内に追加:
rtTraceRelay_.drain();  // 読み捨て（lock-free）。空の場合は即return。
```

**補足**: `getCurrentDrainCount()` による事前判定は不要。`drain()` 内部で `readIndex_ == writeIndex_` をチェックし即returnする。現在はイベント読み捨て。将来のトレース分析用に拡張可能。

#### リスク
- **低**: lock-free + 非ブロッキング。Timerスレッドから安全。

---

## Phase 3: 軽微安全対策

### T5. No.23: `ConvolverProcessor::cleanup()` コメント修正
**優先度: P2-Medium** | **種別: 📋** | **工数: コメントのみ**

#### 問題
`ConvolverProcessor.LoadPipeline.cpp` の第2ループのコメント「強制削除」が実態と乖離。

#### 設計判断
コード変更不要。コメントを実態に合わせる。

#### 修正内容
```cpp
// 変更前:
// 【Leak Fix】LoaderThreadの異常蓄積防止
// スレッドが終了しない場合でも、一定数を超えたら強制削除

// 変更後:
// 【Leak Fix】LoaderThreadの異常蓄積防止（安全策）
// 終了済みスレッドの削除を促進する。強制削除は行わない。
```

#### リスク
- **なし**: コメントのみ。

---

### T7. O-2: `RuntimePolicyEngine::canExecute()` unsigned underflow ガード
**優先度: P3-Low** | **種別: 🛠** | **工数: 小（1行）**

#### 問題
```cpp
return (nowUs - entry.lastExecutedUs) >= entry.cooldownUs;
```
`lastExecutedUs > nowUs` の場合、符号なし減算がアンダーフロー。

#### 修正内容
```cpp
// 変更前:
return (nowUs - entry.lastExecutedUs) >= entry.cooldownUs;

// 変更後:
if (nowUs < entry.lastExecutedUs)
    return false;
return (nowUs - entry.lastExecutedUs) >= entry.cooldownUs;
```

#### リスク
- **極低**: 単一監視タイマースレッド前提の防御的ガード。

---

## Phase 4: デッドコード整理

### T4. No.21: `advancePhase()` の削除
**優先度: P3-Low** | **種別: 🛠** | **工数: 小（関数削除＋呼出元確認）**

#### 問題
`ISRShutdown.cpp` の `advancePhase()` は呼出元が存在しないデッドコード。`ReclaimComplete→VerifyDrained` のスキップロジックにバグ。

#### 設計判断
**修正ではなく削除**: 呼出元が存在しないデッドコードは削除する。ISR Runtime設計原則（責務を減らす）に適合。実際のシャットダウンFSMは `transitionTo()` で動作しており不要。

#### 修正内容
`ISRShutdown.h` から宣言、`ISRShutdown.cpp` から実装を削除。

#### リスク
- **極低**: デッドコード削除。シャットダウンシーケンスに影響なし。

---



## 5. 継続検討事項

以下の項目は設計判断または追加調査が必要なため、本計画では確定せず継続検討とする。

### D2. M/S 最大ゲイン位相関係による過小評価

**ステータス: 構造的問題のため設計変更が必要**

`BandHelper::collectActiveBands()` は単一バンド配列のみを走査し、Mid/Side 各チャンネル別のバンド設定や位相関係を考慮しない。M/Sエンコード時の周波数依存の位相関係はゲイン推定に反映されておらず、最大〜3dBの過小評価リスクがある。本件はアルゴリズムレベルの設計変更が必要なため、本改修計画の対象外とする。

---

## 改修実施順序（レビュー反映）

```
Phase 1: 小規模即時対応
  ├── T1 (No.20) — static_assert 3点セット有効化（1行）
  ├── T2 (No.24) — JSON完全エスケープ（RFC 8259準拠）
  └── T3 (O-1) — blocker設定追加（影響範囲調査済み）

Phase 2: データ喪失リスク対応
  └── T6 (No.19) — RTTraceRelay drain() 結線（実装監査済み）

Phase 3: 軽微安全対策
  ├── T5 (No.23) — cleanup コメント修正（コード変更不要）
  └── T7 (O-2) — underflow ガード追加

Phase 4: デッドコード整理
  └── T4 (No.21) — advancePhase() 削除
```

---

## リスク評価マトリクス

| ID | リスク | 確率 | 影響 | 対策 |
|----|--------|------|------|------|
| T1 | 非ロックフリー環境でコンパイルエラー | 極低 | 高（ビルド停止） | x64+AVX2前提プロジェクトのため現実的リスクなし |
| T2 | エスケープ漏れによるCIパース失敗 | 低 | 低（CI診断のみ） | 網羅的エスケープ処理で対応 |
| T3 | blocker設定追加による誤診断 | 極低 | 低 | 診断情報のみの変更 |
| T4 | advancePhaseが将来誤って使われる | 低 | 高（FSM停止） | 削除が最も安全。放置も可（現在デッドコード） |
| T5 | コメント修正のみのため実害なし | — | — | — |
| T6 | drain呼出位置の誤りによる競合 | 低 | 低 | lock-free実装のためTimerスレッドから安全 |
| T7 | 時刻逆進時のcooldown無効化 | 極低 | 低 | 防御的ガードで対応 |
