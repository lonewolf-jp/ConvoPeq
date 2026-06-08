# 改修計画 v2.7 検証＋ユーザー補正版 (2026-06-08)

> **本ドキュメントは v2.7 分析レポートに対し、ユーザーレビューで得られた補正と最終評価を統合した確定版である。**
>
> ---
> **最終判定: 97〜98/100 — 実装フェーズへ進めてよい**
>
> 設計凍結に必要な3条件は全て確定。残る課題は型安全性・将来保守性を高める改善候補であり、着手を妨げない。
>

---

## 0. 総評

v2.7 レポートの分析は概ね妥当。特に以下の3点は採用すべき：

1. `audioThreadRcuReader` 共用問題を最優先で解消
2. Worker 単一チャネル案を撤回（v2.6 の誤り）
3. Timer を Message チャネルへ統合（v2.6 の `ObserveChannel::Timer` は不要）

ただし以下の補正を適用する。

---

## 1. 「enter が実質機能していない」は断定しすぎ（← 修正点）

v2.7 レポートの以下の記述は実測なしの過剰断定であった：

> 異なるスレッドからの enter() は ownerThreadToken CAS に失敗し、epochProvider->enterReader() が呼ばれない

**補正**: 確かに CAS に失敗するケースは存在するが、実際に保護が機能していないかどうかは以下に依存する：

- その Reader がどのスレッドで最初に enter されたか
- nestingDepth の推移
- 呼び出し頻度とタイミング

したがって評価は **「高リスク設計欠陥」** までにとどめ、 **「現在保護が機能していない」** とは断定しない。

**→ 対策**: 実測による確認、または設計として確実に各スレッド専用 RCUReader を持たせる修正が必要。

---

## 2. 現状分析（ソースコード照合結果）

### 2.1 既に完了している項目

| 項目 | 状態 | 根拠 |
|------|------|------|
| `EpochDomainReaderGuard.h` 削除 | ✅ 完了 | `Test-Path` → False |
| `ObservedRuntime` RCU化 | ✅ 完了 | `RCUReaderGuard guard` + `explicit ObservedRuntime(RCUReader&)` |
| `ownerThreadId` NDEBUG保護 | ✅ 完了 | `#ifndef NDEBUG` でラップ済み |
| `observeCurrentRuntime(RCUReader&)` | ✅ 完了 | 引数は既に `RCUReader&` |

**ただし**: `ObservedRuntime::operator=(ObservedRuntime&&)` は現状 `= default`。
計画 v2.7 では `= delete` を主張しているが、この点は設計判断が必要（後述）。

### 2.2 未実施の項目

| 項目 | 状態 | 備考 |
|------|------|------|
| `ObserveChannel` 列挙型導入 | ❌ | 現状は `int readerIndex` + 4 slot |
| `messageThreadRcuReader` | ❌ | 全スレッドが `audioThreadRcuReader` を共用中 |
| `publicationReader` (RuntimePublicationOrchestrator) | ❌ | まだ RCUReader メンバなし |
| `readControlRuntimeHandle()` 廃止 | ❌ | 11箇所で使用中（internal 7 + external 4） |
| `readAudioRuntimeHandle()` 廃止 | ❌ | 3箇所で使用中 |
| NoiseShaperLearner の RCUReader | ❌ | `engine.readControlRuntimeHandle()` 経由 |
| Audio 二重 enter 解消 | ❌ | AudioBlock/BlockDouble の2ファイル |
| `ObservedRuntime` move assignment `= delete` | ❌ | 現状 `= default` |

### 2.3 重大問題: audioThreadRcuReader の全スレッド共用

```cpp
// AudioEngine.h L2296-2362
convo::ObservedRuntime observedSnapshot { audioThreadRcuReader }; // ← 常にこれ！
```

これにより Audio Thread / Message Thread / Worker Thread の区別なく、すべてが `audioThreadRcuReader` を経由している。RCUReader は `ownerThreadToken` の CAS により単一スレッド占有を前提とする設計であり、この状態は**設計上の欠陥**である。

**影響度**: 高（v2.7 レポートでは「中」としていたが訂正）

### 2.4 EQProcessor の rcuReader は別ドメイン（← 追記）

v2.7 レポートでは未解決事項とした EQProcessor の `rcuReader` について、実際の使用箇所を確認した結果：

- `EQProcessor::process()` 内の `RCUReaderGuard guard(rcuReader)` は **EQ内部状態（EQState）の保護用**
- `readControlRuntimeHandle()` / `readAudioRuntimeHandle()` は使用していない
- つまり RuntimePublishWorld 観測用ではなく、EQ 内部の state snapshot 保護が目的

**結論**: EQProcessor の `rcuReader` は ObserveChannel 体系とは独立した Reader ドメインであり、本改修計画の対象外として良い。

---

## 3. 未確定事項の確定

### 3.1 Timer と Message の統合 → ✅ 確定

JUCE Timer 派生クラスは Message Thread 上でコールバックされる。したがって：

- `AudioEngine::timerCallback()` → Message Thread → `messageThreadRcuReader`
- `SpectrumAnalyzerComponent`（Timer 内で `engine.readControlRuntimeHandle()`） → Message Thread → `messageThreadRcuReader`
- `ObserveChannel::Timer` は不要。`ObserveChannel::Message` に統合

v2.6 で提案された独立 Timer チャネルは撤回。

### 3.2 Worker スロットの割当方式 → 未確定（選択肢あり）

**選択肢 A: 固定スロット（Worker0〜Worker3）**

- 単純明快
- 現状 NoiseShaperLearner 1 Worker には過剰
- 将来 Worker 追加時に上限あり

**選択肢 B: 動的 ObserveChannelId 割当**

- 拡張性が高い
- 実装工数が増える（スロットアロケータ / 解放処理 / 上限管理）
- Practical Stable ISR Bridge Runtime の目的には過剰か

**推奨**: まず固定スロット（Worker0〜Worker3 = 4スロット）で実装し、将来必要性が出た時点で動的割当に移行する。

### 3.3 PublicationAdmission の Reader 所有権 → 確定

計画 v2.7 の設計が正しい：

```
RuntimePublicationOrchestrator（Reader 所有者）
  └── publicationReader（メンバ RCUReader）
        └── PublicationAdmission::evaluate(RCUReader&) に渡す
```

`PublicationAdmission` は Reader を所有せず、引数で受け取る。

### 3.4 SpectrumAnalyzerComponent の Reader 注入方式 → 未確定

現状: `engine.readControlRuntimeHandle()` を呼び出し

選択肢:

1. **SpectrumAnalyzerComponent 自身が RCUReader を所有する**（NoiseShaperLearner と同様）
   - コンストラクタで `convo::RCUReader rcuReader(engine.getRetireRouter())`
   - `makeRuntimeReadHandle(rcuReader, ObserveChannel::Message)` を呼ぶ
   - 自己完結型で最もクリーン
2. **AudioEngine から messageThreadRcuReader の参照を取得する**
   - `engine.getMessageThreadRcuReader()` ゲッターを追加
   - 簡易だが、UI コンポーネントが AudioEngine の内部 Reader に依存する

**推奨**: 選択肢1（RCUReader 自己所有）。RCUReader は軽量（数個の atomic + ポインタ 1 つ）であり、UI コンポーネントが持っても問題ない。

### 3.5 ObservedRuntime move assignment → 未確定

現状 `= default` で動作している。削除する場合の影響：

- ✅ `makeRuntimeReadHandle()` 内の `observedSnapshot = m_coordinator.observeCurrentRuntime(...)` を使用不可に → 新設計ではそもそもこのパターンが不要になる
- ❌ `RuntimeReadHandle` の move に影響を与える可能性 → `RuntimeReadHandle` は `ObservedRuntime` をメンバに持つので要確認

**推奨**: `= delete` 方向で設計し、実装時に `RuntimeReadHandle` への影響を検証する。

---

## 4. Reader 所有権マップ（確定版）

| 所有者 | インスタンス名 | ObserveChannel | スレッド | 備考 |
|--------|---------------|----------------|----------|------|
| `AudioEngine` | `audioThreadRcuReader` | `Audio` | Audio | 既存、維持 |
| `AudioEngine` | `messageThreadRcuReader` | `Message` | Message + Timer | 新規追加 |
| `RuntimePublicationOrchestrator` | `publicationReader` | `Publication` | Message | 新規追加 |
| `NoiseShaperLearner` | `rcuReader` | `Worker0` | Worker | 新規追加 |
| 将来のWorker | `rcuReader` | `Worker1`〜`Worker3` | Worker | 将来 |
| `SpectrumAnalyzerComponent` | `rcuReader` | `Message` | Message(Timer) | 新規追加、自己所有 |
| `EQProcessor` | `rcuReader` | —（対象外） | Audio | 既存、**別ドメインのため本計画対象外** |

チャネル数: 7（Audio + Message + Publication + Worker0〜3）= 最小7、Reserved 込みで 10 が妥当。

---

## 5. Practical Stable ISR Bridge Runtime 観点の最終評価

| 評価軸 | スコア | 備考 |
|--------|--------|------|
| 計画の方向性 | 90/100 | 正しいが前提が古い |
| コード適合性 | 94/100 | 現行との乖離は少ない |
| リスク分析 | 88/100 | enter 機能の過剰断定あり |
| 実装開始可否 | **条件付き可** | 下記3条件 |

### 実装開始の条件

1. **Reader 所有権マップを最終確定** → 本ドキュメント §4 で確定済み
2. **EQProcessor の扱い確定** → 本計画対象外で確定（§2.4）
3. **SpectrumAnalyzerComponent の Reader 注入方式確定** → 自己所有で確定（§3.4）

**上記3条件は本ドキュメントで確定した。** これをもって実装フェーズへ進むことができる。

### 採用推奨事項

1. `audioThreadRcuReader` 共用問題を最優先で解消
2. Worker 単一チャネル案を撤回（固定スロット方式で継続）
3. Timer を Message チャネルへ統合
4. `ObservedRuntime::operator=(ObservedRuntime&&)` を `= delete` 方向で検討
5. `PublicationAdmission` より上位の Orchestrator が Reader を所有
6. EQProcessor は本計画の対象外とする

---

## 6. 最終確定設計判断（2026-06-08）

以下、3ラウンドのレビューを経て確定した設計判断を列挙する。

### 確定事項

| # | 項目 | 決定 | 根拠 |
|---|------|------|------|
| D1 | Reader 所有権マップ | §4 の通り | 全所有者・チャネル・スレッド確定 |
| D2 | EQProcessor の扱い | 本計画の対象外 | `rcuReader` は EQ 内部状態保護用であり RuntimePublishWorld 観測用ではない |
| D3 | SpectrumAnalyzerComponent Reader | 自己所有（`RCUReader` メンバ） | AudioEngine の Reader 公開 API 増加を避ける |
| D4 | PublicationAdmission Reader 所有 | 案A: Orchestrator 所有 + 引数注入 | Admission は判定器として stateless 維持 |
| D5 | Worker スロット | 固定 8 スロット（Worker0〜Worker7） | 動的割当の複雑さ不要、配列サイズ増のみで対応可能 |
| D6 | ObserveChannel 数 | 13（Audio/Message/Publication/Worker×8/Reserved×2） | 8 Worker で当面の拡張に十分 |
| D7 | `ObservedRuntime` move assignment | `= delete` | `unique_resource` として自然。現行使用箇所は代替可能 |
| D8 | Timer チャネル | Message に統合（独立 Timer チャネル不要） | JUCE Timer は Message Thread 上でコールバック |
| D9 | Audio 二重 enter 解消 | AudioBlock.cpp・BlockDouble.cpp の explicit `RCUReaderGuard` を削除 | 二重は Snapshot.cpp のみ該当せず |

### 将来の改善候補（着手条件外）

| # | 項目 | 内容 |
|---|------|------|
| F1 | `RuntimeReaderContext` | `RCUReader&` + `ObserveChannel` を束縛する型安全オブジェクト |
| F2 | 動的 ObserveChannelId 割当 | Worker 追加時に固定上限を超えた場合の拡張手段 |

### 最終評価

- **総合スコア**: 97〜98/100
- **実装可否**: 可（設計凍結して実装フェーズへ進めてよい）
- **残課題**: 型安全性・将来保守性を高める改善候補のみ。致命的欠陥なし。

---

## 7. 修正フェーズ計画（最終版）

| Phase | 内容 | 工数 |
|-------|------|------|
| 0 | Phase-E P5 完了確認 + 現状把握 | 0.1日 |
| 1 | `ObservedRuntime` move assignment `= delete` | 0.1日 |
| 2 | `ObserveChannel` 導入 + `makeRuntimeReadHandle` 改修 | 1.0日 |
| 3 | RCUReader 所有権再配分（各クラスに Reader 追加） | 1.0日 |
| 4 | `readControlRuntimeHandle` 廃止（11箇所を順次置換） | 1.0日 |
| 5 | Audio Thread 二重 enter 解消（2ファイル） | 0.3日 |
| 6 | テストと検証 | 1.0日 |
| **合計** | | **3.5人日** |

---

## 付録: RuntimeReaderContext（将来の改善案）

```cpp
struct RuntimeReaderContext {
    RCUReader& reader;
    ObserveChannel channel;
};
```

現在の `makeRuntimeReadHandle(RCUReader&, ObserveChannel)` では、呼び出し側が誤った組み合わせ（例：`messageThreadRcuReader` に `ObserveChannel::Audio`）を渡してもコンパイルエラーにならない。

`RuntimeReaderContext` を各クラスが構築時に保持すれば、その後の全呼び出しでチャネルと Reader が常に正しく束縛される。ただし実装開始後の改善事項であり、v2.7 の着手条件ではない。
