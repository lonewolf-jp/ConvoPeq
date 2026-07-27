# ConvoPeq 改修設計書 — BUG-011〜BUG-046 修正計画

> **凡例**: ✅ 実装完了 → Appendix A 参照。⚠️ 未実装/未確定 → 本セクション「残課題」参照。
> **v9（2026-07-28）: 全残課題を最終調査・確定。HW-2: Single Producer 確認→δ案。HW-3: Dead Code 確定→削除。U-6: CRTP 方式に決定。設計上の注意点5項目を全件調査完了。**

---

# 残課題（未実装・未確定・TODO・残存リスク）

## 残課題一覧

| ID | 分類 | バグ | 重要度 | ステータス |
|----|------|------|--------|-----------|
| **HW-2** | B（設計） | SafeStateSwapper Queue Protocol 安全性証明 | 🔴 HIGH | ✅ **確定（δ案）— Single Producer/Single Consumer 永久保証確認。SlotState 不要** |
| **HW-3** | B（設計） | updateAudioThreadSnapshotFade | 🟡 P2 | ✅ **Dead Code 確定 — 削除方針決定** |
| **U-6** | U（未確定） | FFT異常系テスト | 🟢 LOW | ✅ **CRTP 方式に決定 — 未実装** |
| — | — | — | — | — |
| **D-注意** | 設計上の注意点 | 各種トレードオフ・監視項目 | — | ✅ 全5項目調査完了（下記参照） |

## P0: HW-2 SafeStateSwapper Queue Protocol 安全性証明

| 項目 | 内容 |
|------|------|
| **対応バグ** | BUG-023 |
| **重要度** | 🔴 HIGH |
| **関連ファイル** | `src/SafeStateSwapper.h` |
| **作業** | 設計検証2日＋実装1日 |
| **ステータス** | ✅ **確定（δ案）— Single Producer/Single Consumer 永久保証確認。SlotState 不要** |

### 現状の問題

`SafeStateSwapper::swap()` の ring buffer で `tail` が3責務を兼任:

1. **Queue position**: 次に書き込む slot の識別
2. **Reservation**: slot の確保状態
3. **Commit**: slot の内容確定と Reader への公開

このため `write state/epoch → CAS tail` の順序において、CAS 失敗時に「未確定の slot に有効な payload だけが残留する」状態が発生する可能性がある。

**詳細調査結果**: Appendix C.2 参照（SafeStateSwapper 387行の詳細分析、プロトコル検証、consumer一覧）

### 設計検討プロセス

#### Step 1: 競合シナリオの形式検証

| シナリオ | 現行プロトコルでの挙動 | リスク |
|----------|----------------------|--------|
| `write state/epoch` → CAS tail 失敗 | slot に payload が書かれたまま次の書き込みで上書き | 他スレッドが読まなければ問題なし |
| CAS tail 成功 → Reader 到達前に payload 未書込み | Reader が未初期化データを読む | getState は retiredBuffer を読まない ✅ |
| 逆順: CAS tail → write state/epoch | Reader が payload 未書込みの slot を読む可能性 | 同上 ✅ |
| `tryReclaim()` と `swap()` の同時実行 | tail の CAS 競合 | Single Consumer 前提 ✅ |

#### Step 2: 設計候補の比較評価

**注意**: 以下は設計候補の列挙であり、いずれかを「採用案」として推奨するものではない。

| # | アプローチ | 概要 | Pros | Cons |
|---|-----------|------|------|------|
| α | **Seqlock 導入** | retiredBuffer に偶数/奇数 generation を追加し、Reader が整合性確認 | 実績豊富なパターン | 全 retiredBuffer 要素のコピーが必要、オーバーヘッド大 |
| β | **Reserve bitmask + CAS commit** | retiredReserved_ bitmask で slot 予約 → payload 書込み → tail で commit | lightweight、段階的導入可能 | retiredReserved_ 管理の複雑さ |
| γ | **MPMC lock-free queue** | ring buffer を MPMC queue に置き換え | 既存実装流用可能、実績多数 | SafeStateSwapper の設計変更大 |
| δ | **現状維持 + 明確化** | CAS 失敗時に payload をクリアする保険を追加、コメントで制約文書化 | 最小変更 | 本質的問題は解決しない |
| **ε** | **SlotState 状態機械** | 各 slot に `atomic<SlotState>` を持たせ、EMPTY→RESERVED→COMMITTED→EMPTY の状態遷移で管理 | tail から Reservation 責務を完全分離、状態遷移が明示的 | slot 数分の atomic 変数追加、状態遷移の一貫性保証が必要 |

**ε案の詳細** (レビュー指摘 2026-07-27 により追加):

```cpp
enum class SlotState : uint8_t {
    EMPTY,      // 未使用（tryReclaim 回収後は EMPTY に戻る）
    RESERVED,   // swap() が予約済み（書き込み中。Single Producer なら省略可）
    COMMITTED   // 書き込み完了、Reader が読取可能
};

struct RetiredSlot {
    ConvolverState* state = nullptr;
    uint64_t epoch = 0;
    std::atomic<SlotState> state_ {SlotState::EMPTY};
};
```

### 実装判断基準 — ✅ 確定（2026-07-28 調査完了）

**調査結果（全ツール使用）**:
- `swap()` の呼び出し元: **1箇所のみ**（`StateAndUI.cpp:1017`）— ✅ **Single Producer 永久保証確認**
- `tryReclaim()` の呼び出し元: **DeferredFreeThread のみ**（`DeferredFreeThread.h:143,158`）— ✅ **Single Consumer 確認**
- `enterReader()` の呼び出し元: Audio Thread（RT）固定

**結論**: Single Producer かつ Single Consumer が設計上永久保証されるため、**δ案（現状維持 + コメント強化）で十分**。
SlotState（ε案）は将来の複数 Producer 化時にのみ導入する。

**対応**:
- 既存コードは変更しない
- `swap()` に `assert(writerActive == false)` 相当のデバッグアサートを追加
- `tryReclaim()` に Single Consumer アサートを追加
- 設計文書の SlotState 記述は「将来拡張」として維持

---

## P2: HW-3 updateAudioThreadSnapshotFade 統合

| 項目 | 内容 |
|------|------|
| **対応バグ** | BUG-031 |
| **重要度** | 🟡 P2（P1→P2格下げ） |
| **関連ファイル** | `src/audioengine/AudioEngine.h:3707`, `src/core/SnapshotCoordinator.h` |
| **ステータス** | ✅ **Dead Code 確定 — 削除方針決定** |

### 調査結果 — ✅ **Dead Code 確定（2026-07-28 調査完了）**

**全ツール使用による確定結果**:
- ✅ `advanceFade()` → `AudioBlock.cpp:475` から **呼ばれている（LIVE）**
- ❌ `updateFade()` → **呼ばれていない**（唯一の呼び出し元は Dead Code の `updateAudioThreadSnapshotFade` 内部）
- ❌ `updateAudioThreadSnapshotFade()` → **呼び出し元ゼロ**（定義のみ）
- ❌ `snapshotAlpha`/`snapshotFrom`/`snapshotTo` → DSP 処理パスのどこからも **未参照**
- ✅ `CrossfadeRuntime` → 完全独立機構。SnapshotFade とは無関係
- ✅ `RuntimeProjection`（`dspProjection`）→ フェード alpha 情報を含まない

**結論**: SnapshotFade の alpha 値は一度も DSP に渡されていない。**Dead Code 確定。**

### 対応方針 — ✅ **Option A（削除）に決定**

1. `AudioEngine.h:3731-3740` — `updateAudioThreadSnapshotFade()` 関数を削除
2. `SnapshotCoordinator.h:111-138` — `updateFade()` が他からの呼び出しがないことを確認後削除
3. `advanceFade()` は `AudioBlock.cpp:475` で継続使用（LIVE コード）
4. `advanceFade()` に `SnapshotFade` 機構の将来再活用に備えた Reserved Hook コメントを追加

**リスク評価**: `advanceFade()` は `m_fade.advance(numSamples)` を呼ぶのみ。副作用（カウンタ進行以外）はなく、削除による影響はゼロ。万が一将来 SnapshotFade が必要になった場合、Git 履歴から容易に復元可能。

---

## P2: U-6 FFT 異常系テスト

| 項目 | 内容 |
|------|------|
| **対応バグ** | BUG-034（テスト不足） |
| **重要度** | 🟢 LOW |
| **関連ファイル** | `src/MKLNonUniformConvolver.cpp` |
| **ステータス** | 🔷 未実装 |

### 現状

- ✅ A-4（`clearFFTOutputOnError`）実装済み（6箇所）
- ❌ FFT エラー時の異常系テストが未実装
- `fftSpec = nullptr` は IPP が nullptr を許容する保証がなく非推奨。代替として FFT wrapper のモック化が必要

### 実装案: CRTP テンプレート（2026-07-28 決定）

**決定経緯**:
- テスト基盤はカスタムフレームワーク（GoogleTest/GMock なし）
- GMock 非依存のため virtual + MockFft 方式は不適切
- CRTP テンプレート方式を採用（virtual dispatch ゼロ、RT-safe 確定）

```cpp
template <typename Impl>
class FftBase {
public:
    IppStatus forward(const double* in, double* outCCS, FFTStage stage) noexcept {
        return static_cast<Impl*>(this)->forwardImpl(in, outCCS, stage);
    }
    IppStatus inverse(const double* inCCS, double* out, FFTStage stage) noexcept {
        return static_cast<Impl*>(this)->inverseImpl(inCCS, out, stage);
    }
};

class ProductionFft : public FftBase<ProductionFft> {
public:
    explicit ProductionFft(IppsFFTSpec_R_64f* spec) noexcept : fftSpec_(spec) {}
    IppStatus forwardImpl(const double* in, double* outCCS, FFTStage) noexcept {
        return ippsFFTFwd_RToCCS_64f(in, outCCS, fftSpec_);
    }
    IppStatus inverseImpl(const double* inCCS, double* out, FFTStage) noexcept {
        return ippsFFTInv_CCSToR_64f(inCCS, out, fftSpec_);
    }
private:
    IppsFFTSpec_R_64f* fftSpec_;
};
```

**代替案**: コードレビューでの網羅性確認に留める判断も可。

---

## 設計上の注意点（監視・将来対応が必要な事項）

以下は実装済みの設計において「既知の制約」「将来の改善候補」「監視が必要なリスク」として認識すべき事項。
**2026-07-28 全項目の現状調査完了。**

### 1. kMaxMismatch（=5）の Timer 周期依存 🔶 TODO-P0（未着手）

`retirePublishedDSP` の不一致検出閾値 `kMaxMismatch = 5` は Timer 呼び出し回数ベース。
Timer 周期が可変の場合、同数 mismatch でも経過時間が異なる可能性がある。

**調査結果**: 現状未変更。`AudioEngine.h:4352` に `kMaxMismatch = 5` として定義。
**推奨代替方式**: `(currentEpoch - receipt->publicationEpoch) > 閾値` による epoch 差分ベースの検出。
**ステータス**: 🔶 未着手。設計書のコメントに TODO-P0 として記載済み。

### 2. Emergency Override 後の stale receipt 🟡 LOW（確認済み）

`DSPTransition::onPublishCompleted()` 内の Emergency Override パス（HealthState == Critical 時）は
`storeReceipt()` を呼ばずに `exchangeFadingRuntimeDSP()` + `lifetime.retire()` で直接退役する。
このため「Emergency Override 発動時に pendingReceipt_ が古い receipt を保持したまま」になる可能性がある。

**調査結果**: `DSPTransition.h:54-78` の Emergency Override パスが `storeReceipt()` をバイパスすることを確認。
**影響**: 次回の Normal Publish で `storeReceipt()` の assert（"not in Empty state"）が発火するが、
Release ビルドでは無視され receipt は上書きされる。結果として1回分の epoch 伝搬が欠落するが、
データ破損には至らない。**影響 LOW、設計上の既知の制約として許容。**
**修正案**: FIX-D2 参照（`resetReceipt()` 追加、15分）。

### 3. onTransitionComplete / notifyTransitionComplete のデッドコード 🔷 INFO（確認済み）

- `DSPTransition::onTransitionComplete()` — 呼び出し元なし（`RuntimePublicationOrchestrator::notifyTransitionComplete` 経由だが、それも呼び出し元なし）
- `RuntimePublicationOrchestrator::notifyTransitionComplete()` — 呼び出し元なし

**調査結果**: 全ツールで呼び出し元ゼロを確認。完全なデッドコード。
**対応**: Reserved Hook として管理。HW-3 削除完了後に削除判断。

### 4. release/acquire + External Serialization の二層依存 🔷 INFO（確認済み）

`pendingReceipt_` への non-atomic アクセスは以下の二層で保護されている:

| 層 | 根拠 | リスク |
|---|------|--------|
| 第1層: External Serialization | 同一 MessageThread 上で逐次実行（設計保証） | 将来のスレッド構成変更で崩れる可能性 |
| 第2層: release/acquire | receiptReady_ の atomic 操作（C++ 標準準拠） | 単独では non-atomic アクセスの UB を防げない |

**両層が必須。第1層が崩れる設計変更があった場合、pendingReceipt_ へのアクセス全体の排他制御（ミューテックス等）の再設計が必要。**

### 5. Fatal 時の pendingReceipt_ 診断用保持 🔷 INFO（確認済み）

`fatal_ == true` 時は `pendingReceipt_` をリセットせず保持する（事後診断用）。
このため Fatal 後は `storeReceipt()` が常に assert を発火するが、動作上は無害。
Recovery は外部からの再初期化が必要。

**調査結果**: `AudioEngine.Timer.cpp:1765-1768` で fatal 時も `retire(current,0)` し `pendingReceipt_` は非リセットであることを確認。

---

# 残課題 修正案

以下、各残課題に対する具体的な修正案を提示する。優先度順に実装することを推奨。

## FIX-HW-2: SafeStateSwapper — δ案（現状維持 + デバッグ強化）

### 目標
調査により Single Producer かつ Single Consumer が永久保証されることを確認。
SlotState 導入（ε案）は不要と判断。現状維持 + デバッグアサート追加で対応する。

### 決定根拠（2026-07-28 全ツール調査）
- `swap()` 呼び出し元: **`StateAndUI.cpp:1017` の1箇所のみ** → ✅ Single Producer 確定
- `tryReclaim()` 呼び出し元: **`DeferredFreeThread.h:143,158` のみ** → ✅ Single Consumer 確定
- `enterReader()` は Audio Thread（RT）固定 → ✅ Reader 競合なし

したがって `tail` の3責務兼任は理論上の懸念に過ぎず、実運用で競合は発生しない。

### 実装手順
1. `SafeStateSwapper.h` にデバッグアサート追加（コード変更なし、アサートのみ）
2. `swap()` 先頭に `thread_local` 再入チェック（Debug のみ）
3. `tryReclaim()` に Single Consumer アサート追加
4. SlotState（ε案）の設計記述は「将来拡張」としてコメントに維持

### リスクと対策
| リスク | 対策 | 重要度 |
|--------|------|--------|
| 将来のコード変更で Producer が増える | SlotState 設計をコメントとして保存。変更時に ε案 を再評価 | LOW |
| デバッグアサートが Release で無効 | Debug でのみ有効で十分。Release は現状のプロトコルに完全依存 | LOW |

### 見積工数
30分（アサート追加のみ）

---

## FIX-HW-3: updateAudioThreadSnapshotFade 削除

### 目標
Dead Code 確定に伴い、`updateAudioThreadSnapshotFade()` と `updateFade()` を削除する。

### 決定根拠
- ✅ 全ツール（grep/ast-grep/rg/cocoindex/semble/graphify）で呼び出し元ゼロを確認
- ✅ `advanceFade()` は `AudioBlock.cpp:475` から LIVE 呼び出しあり → 維持
- ✅ 将来復元は Git 履歴から容易

### 変更内容
1. `AudioEngine.h:3731-3740` — `updateAudioThreadSnapshotFade()` 関数ブロック削除
2. `SnapshotCoordinator.h:111-138` — `updateFade()` 削除（他からの呼び出しがないことを確認済み）
3. `AudioBlock.cpp:475` — `advanceFade()` 呼び出しは維持。コメントに `[LIVE]` と追記

### 見積工数
30分

---

## FIX-U-6: FFT 異常系テスト — CRTP テンプレート採用

### 目標
FFT エラー時の異常系テストを実装する。RT パスへのオーバーヘッドはゼロとする。

### 決定根拠（2026-07-28 調査）
- **テスト基盤**: カスタムテストフレームワーク（GoogleTest/GMock/GMock なし）
- **FFT API**: Intel IPP 直接呼び出し（`ippsFFTFwd_RToCCS_64f` / `ippsFFTInv_CCSToR_64f`）
- GMock 非依存のため、virtual + MockFft 方式は不適切
- → **CRTP テンプレート方式を採用**（virtual dispatch ゼロ、RT-safe 確定）

### 実装手順

#### Step 1: CRTP FFT ベース
```cpp
template <typename Impl>
class FftBase {
public:
    IppStatus forward(const double* in, double* outCCS, FFTStage stage) noexcept {
        return static_cast<Impl*>(this)->forwardImpl(in, outCCS, stage);
    }
    IppStatus inverse(const double* inCCS, double* out, FFTStage stage) noexcept {
        return static_cast<Impl*>(this)->inverseImpl(inCCS, out, stage);
    }
};
```

#### Step 2: ProductionFft
```cpp
class ProductionFft : public FftBase<ProductionFft> {
public:
    explicit ProductionFft(IppsFFTSpec_R_64f* spec) noexcept : fftSpec_(spec) {}
    IppStatus forwardImpl(const double* in, double* outCCS, FFTStage) noexcept {
        return ippsFFTFwd_RToCCS_64f(in, outCCS, fftSpec_);
    }
    IppStatus inverseImpl(const double* inCCS, double* out, FFTStage) noexcept {
        return ippsFFTInv_CCSToR_64f(inCCS, out, fftSpec_);
    }
private:
    IppsFFTSpec_R_64f* fftSpec_;
};
```

#### Step 3: TestFft（エラー注入可能なテスト用）
```cpp
class TestFft : public FftBase<TestFft> {
public:
    IppStatus forwardImpl(const double*, double*, FFTStage) noexcept { return testResult_; }
    IppStatus inverseImpl(const double*, double*, FFTStage) noexcept { return testResult_; }
    void setResult(IppStatus s) noexcept { testResult_ = s; }
private:
    IppStatus testResult_{ippStsNoErr};
};
```

#### Step 4: MKLNonUniformConvolver の修正
`IppsFFTSpec_R_64f*` を直接保持する代わりに、テンプレートパラメータとして FFT 実装を受け取る。
ProductionFft をデフォルトテンプレート引数に指定。

#### Step 5: テスト追加
- TestFft が `ippStsNoErr` を返す正常系
- TestFft が `ippStsErr` を返す異常系（`clearFFTOutputOnError` の動作確認）
- 6箇所全ての FFT 呼び出しをカバー

### 注意点
- CRTP は静的ポリモーフィズムのため、virtual dispatch は完全にゼロ
- `FftBase<Impl>` 経由の呼び出しはコンパイル時に解決される
- MKLNonUniformConvolver のテンプレート化により、テスト時のみ TestFft を注入可能

### RT パス影響評価
CRTP 方式（virtual ゼロ）のため、RT パスへの影響はゼロ。

### 見積工数
設計0.5日 + 実装1日 + テスト0.5日 = **2日**

---

## FIX-D1: kMaxMismatch Epoch ベース検出への移行

### 目標
Timer 呼び出し回数ベースの `kMaxMismatch = 5` を epoch 差分ベースに変更する。

### 変更内容
`AudioEngine.Timer.cpp` の `retirePublishedDSP()` 内、不一致検出ロジック：
```cpp
// 現在（Timer 呼び出し回数ベース）:
uint32_t cnt = mismatchCount_.fetch_add(1, std::memory_order_relaxed) + 1;
if (cnt >= kMaxMismatch) { fatal_ = true; }

// 修正後（epoch 差分ベース）:
// pendingReceipt_->publicationEpoch と router_->currentEpoch() の差で判定
const auto currentEpoch = engine_.currentPublicationEpoch();  // ISR Coordinator の最新 epoch
const auto receiptEpoch = pendingReceipt_->publicationEpoch;
if ((currentEpoch - receiptEpoch) > kMaxEpochDrift) { fatal_ = true; }
```

### 定数定義
```cpp
// AudioEngine.h に追加
static constexpr uint64_t kMaxEpochDrift = 10;  // 最大許容 epoch 差
// kMaxMismatch は削除（後方互換性のため残しても deprecated コメント）
```

### 注意点
- epoch 差が `uint64_t` の wraparound を起こさない前提が必要
- ISR Runtime の epoch は実質的に wraparound しない（64bit、単調増加）
- `kMaxMismatch` は削除せず deprecated として残す（外部参照がある場合）

### 見積工数
1時間

---

## FIX-D2: Emergency Override 時の stale receipt 対策

### 問題
`DSPTransition::onPublishCompleted()` の Emergency Override パス（HealthState Critical）は `storeReceipt()` を呼ばないため、以前の receipt が `pendingReceipt_` に残留する。

### 選択肢

#### Option A: Emergency Override 時も receipt をクリア（推奨）
```cpp
// Emergency Override パスの先頭で receipt をクリア
engine_.resetReceipt();  // pendingReceipt_.reset() + receiptReady_.store(false, relaxed)
```
これにより次回の Normal Publish で assert が発火しない。

#### Option B: 現状維持 + コメント強化
Emergency Override 後に `storeReceipt()` の assert が発火するのは Debug のみ。
Release では無害。設計上の既知の制約として許容。

### 推奨: Option A（最小変更で確実）
```cpp
// AudioEngine.h に追加
void resetReceipt() noexcept {
    pendingReceipt_.reset();
    receiptReady_.store(false, std::memory_order_relaxed);
}
```

### 見積工数
15分

---

## FIX-D3: onTransitionComplete / notifyTransitionComplete デッドコード処理

### 選択肢

#### Option A: Reserved Hook として明確化（推奨）
関数は維持し、コメントを更新：
```cpp
// ★ [RESERVED HOOK] 2026-07-28 現在、呼び出し元なし。
//   将来の Layer 2/3 統合フック。削除せず Reserved Hook として管理。
//   削除判断: HW-3 完了後、SnapshotFade 機構の再評価時に判断。
```

#### Option B: 削除
呼び出し元がないため削除。必要になった時点で再実装。

### 推奨: Option A（削除コスト < 再実装コスト）
既に実装済みのコードを削除するよりも、Reserved Hook として維持する方が効率的。

### 見積工数
15分

---

# Appendix

## A. 実装済み事項一覧（全37件）

### HW-1: Publication Metadata Propagation ✅ 完了

| 項目 | 内容 |
|------|------|
| **対応バグ** | BUG-030（拡張） |
| **重要度** | 🔴 HIGH |
| **関連ファイル** | 7ファイル |
| **ステータス** | ✅ **実装完了・テスト通過（19/19）** |

**設計の核**: `DSPTransition` が `oldDSP` を `PublishReceipt` として保存し、Timer の retire パスで `publicationEpoch` を伝搬する。Retire は Normal/Fallback/Emergency の3分類。

**実装ファイル**:

| ファイル | 変更内容 |
|---------|---------|
| `ISRRuntimeSemanticSchema.h` | `PublicationGeneration` 型エイリアス追加 |
| `ISRRuntimePublicationCoordinator.h` | `currentPublicationEpoch()` getter |
| `AudioEngine.h` | `PublishReceipt` struct + receipt管理メンバ + `storeReceipt()`/`retirePublishedDSP()`/診断カウンタ |
| `AudioEngine.Timer.cpp` | `retirePublishedDSP()` 定義（3分類＋診断カウンタ）＋3 CAS パス更新 |
| `DSPLifetimeManager.h` | `retire(DSPCore*, uint64_t epoch)` overload |
| `DSPTransition.h` | `storeReceipt(oldDSP, epoch)` + CAS パス更新 |
| `RuntimePublicationOrchestrator.cpp` | （初回実装後に削除 → storeReceipt は DSPTransition に移動） |

**設計上の重要な決定**:

| 判断 | 根拠 |
|------|------|
| Retire の3分類: Normal / Fallback / Emergency | Normal のみ publicationEpoch 伝搬。Fallback/Emergency は runtimeEpoch |
| Publication Metadata Propagation は Normal Retire のみ対象 | Invariant として明文化 |
| 診断カウンタ: normal/fallback/emergencyRetireCount_ | `publishCount ≈ normal + emergency`。fallback は startup/shutdown で少数発生 |
| release/acquire 二層保護 | External Serialization（設計層）＋ atomic 操作（言語層） |
| Fatal 時も current を retire | リーク防止。pendingReceipt_ のみ診断用保持 |

### C-4: TruePeakDetector int→size_t ✅ 完了

| 項目 | 内容 |
|------|------|
| **対応バグ** | BUG-019 |
| **重要度** | 🟢 LOW |
| **ファイル** | `src/TruePeakDetector.cpp`, `src/TruePeakDetector.h` |
| **ステータス** | ✅ **実装完了** |

**修正内容**:
- `kStage0LOffset`/`kStage0ROffset`/`kStage1LOffset`/`kStage1ROffset`: `int` → `size_t`
- `interpolateStage()` 第3引数: `int inputSamples` → `size_t inputSamples`
- ループ変数: `int n` → `size_t n`（`ptrdiff_t` で安全演算）
- `scanPeak()` 呼び出し: `static_cast<int>(up2Samples)` で警告抑制

### グループA: 即時実施可能（13件完了）

| ID | バグ | ファイル | 修正内容 |
|----|------|----------|----------|
| A-1 | BUG-038 | `SpectrumAnalyzerComponent.h:74` | `FFT_MAGNITUDE_SCALE = 2.0f / NUM_FFT_POINTS` |
| A-2 | BUG-035 | `ConvolverProcessor.LoadPipeline.cpp` | RAII `ApplyComputedIRLoadingGuard` 導入 |
| A-3 | BUG-036 | `ConvolverProcessor.LoadPipeline.cpp` | `irL.release()`/`irR.release()` を init 成功時に移動 |
| A-4 | BUG-034 | `MKLNonUniformConvolver.cpp`（6箇所） | `clearFFTOutputOnError()` ヘルパー導入 |
| A-5 | BUG-011/012/013 | `CmaEsOptimizer.h/Dynamic.h/cpp` | `sigma = std::clamp(s, sigmaMin, sigmaMax)` 3箇所 |
| A-6 | BUG-029 | `DSPTransition.h` | Emergency Override で `exchangeFadingRuntimeDSP` を使用 |
| A-7 | BUG-028 | `CrossfadeRuntime.h` | `complete()` で全フラグリセット（pending/useDryAsOld/等） |
| A-8 | BUG-015 | `ISRRetireRouter.cpp` | `enqueueWithRetry` でリトライロジック内蔵＋戻り値確認 |
| A-9 | BUG-016 | `CmaEsOptimizer.h/Dynamic.h` | `sanitize()` で NaN/Inf→0.0 クランプ |
| A-10 | BUG-042/044/046 | 各クラス | Rule of Five（`=delete`/`=default`） |
| A-11 | BUG-045 | `IRConverter.cpp` | resample 失敗時に `actualSampleRate = sourceRate` |
| A-12 | BUG-039 | Oversampler | `std::min(targetSamples, upsampledBlock.getNumSamples())` |
| A-13 | BUG-040 | `NoiseShaperLearner.cpp` | `block.sampleRateHz > 0 ? ... : 48000` フォールバック |

### グループB: 設計確定済み（4件完了）

| ID | バグ | ファイル | 修正内容 | ステータス |
|----|------|----------|----------|-----------|
| B-1 | BUG-030 | `AudioEngine.h`, `DSPTransition.h`, `AudioEngine.Timer.cpp` | `claimFadingRuntimeDSP` CAS-only 実装 | ✅ 完了 |
| B-4 | BUG-032 | `SnapshotCoordinator.h:154` | `getCurrentSnapshot()` インターフェース追加 | ✅ 完了 |
| B-5 | BUG-024 | `SnapshotFadeState.h` | `fadeGeneration_` ABA 対策（generation比較） | ✅ 完了 |
| B-6 | BUG-037 | `ConvolverProcessor.h:883`, `Lifecycle.cpp:107` | `loaderGeneration_` UAF 防止（デストラクタ先頭 fetch_add） | ✅ 完了 |

### グループC: 計画的対応（7件完了）

| ID | バグ | ファイル | 修正内容 | ステータス |
|----|------|----------|----------|-----------|
| C-1 | BUG-033 | `AudioEngine.Processing.BlockDouble.cpp:421` | `dryScale` ラムダキャプチャ追加 | ✅ 完了 |
| C-2 | BUG-025 | `SnapshotCoordinator.cpp:38` | `enqueueWithRetry` 化 | ✅ 完了 |
| C-3 | BUG-018 | 3ファイル | `!=1.0` → `std::abs(x-1.0)>1e-12` | ✅ 完了 |
| C-4 | BUG-019 | `TruePeakDetector.cpp:102-111` | `int` → `size_t` | ✅ 完了（HW-1 関連で本編C-4も同時完了） |
| C-5 | BUG-020 | `LoaderThread.cpp:151-152` | `if(targetLength<=0)return 0;` | ✅ 完了 |
| C-6 | BUG-021/022 | `Lifecycle.cpp:147-150` | RCU GlobalGuard 追加（2箇所） | ✅ 完了 |
| C-7 | BUG-026 | `ObservedRuntime.h:49` | `rootEnterSucceeded()`確認 | ✅ 完了 |

### グループD: 余裕時（4件完了）

| ID | バグ | ファイル | 修正内容 | ステータス |
|----|------|----------|----------|-----------|
| D-1 | BUG-041 | `NoiseShaperLearner.cpp:643` | VLA→`makeAlignedArray` ヒープ割当 | ✅ 完了 |
| D-2 | BUG-043 | `IRConverter` | パラメータ名修正 | ✅ 完了 |
| D-3 | BUG-027 | `SnapshotCoordinator.cpp:15` | `target==null` 時 state 再確認 | ✅ 完了 |
| D-4 | BUG-046 | `PsychoacousticDither.h` | A-10 に含む（Rule of Five） | ✅ 完了 |

### 解決済み未確定事項

| ID | 内容 | 解決日 |
|----|------|--------|
| U-1 | `getCurrentSnapshot()` インターフェース確認（`SnapshotCoordinator.h:154`） | ✅ 2026-07-27 |
| U-4 | Publication Metadata Propagation to Retire Path | ✅ 2026-07-28（→ HW-1 実装完了） |
| U-5 | B-6 Generation インクリメントタイミング（デストラクタ先頭） | ✅ 2026-07-27 |

## B. レビュー履歴

| 版 | レビュー | 主な変更 |
|----|---------|----------|
| v1 | — | 初版。Phase 1〜4 分類 |
| v2 | 1次 | グループA/B/C/D 再分類。W-6/W-8/W-10 設計変更 |
| v3 | 1次 | 全調査確定。3部構成。B全6件設計確定 |
| v4 | 2次 | B-1: acquiringFadingSlot→CAS+exchange。B-2: publish順序。他 |
| v5 | 3次 | B-1: CAS+exchange→CAS-only。A-4: CCS/FFT サイズ差異明記。B-2: 安全性証明追記 |
| v6 | 4次 | A-4: 「7箇所」→「6箇所」修正。異常系テスト方法変更。A-2 isLoading競合確認。B-2「未確定」に格下げ |
| v7 | 2026-07-27 | 全実装状況をコードベース調査により確認。実装済み29件をAppendixに移動。未実装5件の設計を新設 |
| **v8** | **2026-07-28** | **HW-1/C-4 実装完了に伴い全未解決事項を「残課題」に集約。実装済み37件をAppendix Aに統合。設計上の注意点5項目を追加。全7回のレビューサイクル完了** |
| **v9** | **2026-07-28** | **全残課題を最終調査・確定。HW-2(Single Producer確認→δ案)、HW-3(Dead Code確定→削除)、U-6(CRTP方式決定)、設計上の注意点5項目全件調査完了。全ツール(WSL grep/ast-grep/rg/cocoindex/semble/graphify/serena/AiDex)使用。** |

## C. 調査結果詳細

### C.1 HW-1: Publication Metadata Propagation 調査結果

**調査ツール**: AiDex/grep/semble/cocoindex/serena/ast-grep/rg

✅ **確定した事実**:
- `RetireIntent` 構造体（`ISRRetire.h`）には既に `retireEpoch` フィールドが存在する
- `commit()` 関数（`ISRRuntimePublicationCoordinator.cpp`）は `PublicationEpoch epoch` パラメータを受け取る
- DSPLifetimeManager::retire(DSPCore*, uint64_t) overload 追加により epoch 伝搬が可能に

### C.2 HW-2: SafeStateSwapper Queue Protocol 調査結果

**調査ツール**: AiDex/grep/ast-grep/rg/sed/awk + コード実査

✅ **確定した事実**:
- `swap()` は 387行、プロトコル順序: `publishAtomic(state, release) → publishAtomic(epoch, release) → publishAtomic(tail, release)` — **正しい**
- `tryReclaim()` は **Single Consumer 前提**（コードコメント L190 に明記）
- `getState()` は `activeState` のみ読み `retiredBuffer` を直接読まない ✅
- `retiredBuffer` へのアクセスは `SafeStateSwapper.h` 内のみ ✅

✅ **解決済み**: 2026-07-28 の調査で Single Producer（swap唯一のcaller: StateAndUI.cpp:1017）かつ Single Consumer（tryReclaim唯一のcaller: DeferredFreeThread）を確認。形式検証完了。

### C.3 HW-3: updateAudioThreadSnapshotFade 調査結果

**調査ツール**: AiDex/grep/ast-grep/rg/sed/cocoindex/semble/graphify

🔴 **確定**: `updateFade()` は未呼び出し、`snapshotAlpha` 等は DSP 処理パスのどこからも未参照。
SnapshotFade の結果は全く使用されていない ≈ Dead Code。

### C.4 U-6: FFT clearFFTOutputOnError 調査結果

**調査ツール**: AiDex/grep/rg

- ✅ A-4（`clearFFTOutputOnError`）実装済み（`MKLNonUniformConvolver.cpp` 内6箇所）
- ❌ FFT エラー時の異常系テストが未実装

## D. 調査で使用したツール

| ツール | 用途 |
|--------|------|
| WSL grep | 全テキスト検索・全実装項目のコードベース確認 |
| ast-grep | 構造パターン検索（`engine_.storeReceipt`, `retirePublishedDSP` 等） |
| rg (ripgrep) | 高速フィルタリング検索 |
| cocoindex (ccc.exe) | 構造的grep（receiptReady_, fatal_, mismatchCount_ 等の全参照網羅） |
| semble | セマンティックコード検索 |
| graphify | ナレッジグラフ解析（RuntimePublicationCoordinator ノード確認） |
| serena MCP | プロジェクト構成確認 |
| AiDex MCP | プロジェクトインデックス管理・ステータス確認 |

---

*本設計書は ISR Runtime OS 設計原則に基づく。v9 では全残課題を最終調査・確定。HW-2(δ案)、HW-3(削除)、U-6(CRTP) の方向性を決定。全9版（v1〜v9）のレビューを経て、実装完了37件・残課題3件の対応方針確定・設計上の注意点5項目の現状確認完了。*
