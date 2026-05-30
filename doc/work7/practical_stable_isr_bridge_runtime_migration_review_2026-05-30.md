# Practical Stable ISR Bridge Runtime 移行レビュー（詳細調査結果）

- 作成日: 2026-05-30
- 対象: `src/audioengine` を中心とした ISR Bridge Runtime 遷移実装
- 目的: 「実運用で破綻しにくい Practical Stable ISR Bridge Runtime」への移行状況レビュー妥当性を、ソースコード証跡ベースで整理

---

## 1. 結論サマリ（先に要点）

### 総合判定

- **部分達成（基盤は強いが、運用安定性の要件で未収束点あり）**

### 主要判定

1. **Snapshot Semantic Unification は未達**
   - AudioThread の主要処理経路で、`RuntimeGraph` / `GlobalSnapshot` / crossfade 準備スナップショットが並列参照される構造が継続。

1. **Observe Path Collapse は未達**
   - `getNextAudioBlock` 系における観測経路が単一面へ統合されていない。

1. **Publication Coordinator 自体は有効に機能**
   - ただし publish 呼び出しトリガー入口は複数ファイルに分散（commit/timer/prepare/release）。

1. **Retire/Reclaim Governance は実装済み（未実装ではない）**
   - watermark/saturation/backpressure の管理ロジックあり。

1. **ContractRegistry / VerifierRegistry の実体は未検出**
   - `src/**/*.h,cpp` 範囲で名称一致ヒットなし。

---

## 2. 調査スコープと方法

- 対象ディレクトリ: `src/audioengine`, `src`（registry確認）
- 方法:
  - シンボル/パターン検索による行番号特定
  - 重点ファイルは関数本体を読んで文脈確認
- 観点:
  - semantic plane の分離/混在
  - publish authority と trigger 入口の統治
  - retire/reclaim の飽和制御
  - 契約/検証レイヤー（registry）存在確認

---

## 3. 詳細証跡（ファイル・行）

## 3.1 AudioThread 経路の多重 semantic 参照（未統合）

### A) float path（`getNextAudioBlock`）

- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp:145`
  - `runtimeGraph->activeNode`
- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp:180`
  - `getRuntimeSnapshot(runtimeReadView)`
- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp:231`
  - `consumeCrossfadePreparedSnapshot()`

### B) double path

- `src/audioengine/AudioEngine.Processing.BlockDouble.cpp:104`
  - `runtimeGraph->activeNode`
- `src/audioengine/AudioEngine.Processing.BlockDouble.cpp:120`
  - `getRuntimeSnapshot(runtimeReadView)`
- `src/audioengine/AudioEngine.Processing.BlockDouble.cpp:152`
  - `consumeCrossfadePreparedSnapshot()`

**評価**: Audio callback で複数面を読む構造が残っており、観測一元化の観点で未完。

---

## 3.2 `processWithSnapshot` の意味論混在

- `src/audioengine/AudioEngine.Processing.Snapshot.cpp:13` 以降
  - `runtimeGraphHint` 未指定時に `readAudioRuntimeView()` + `getRuntimeGraph(...)`
  - DSP 解決は runtime graph 依存
  - 一方で引数 `snap`（`GlobalSnapshot`）から parameter snapshot を構築して処理

**評価**: 補助経路でも `RuntimeGraph` と `GlobalSnapshot` の二面参照が継続。

---

## 3.3 publish 入口の分散状況

### Commit 経路

- `src/audioengine/AudioEngine.Commit.cpp`
  - `508/509`, `540/541`, `591/592`, `632/633`, `798/799` 行付近
  - `makeRuntimePublicationCoordinator().publishState(...)`

### Timer 経路

- `src/audioengine/AudioEngine.Timer.cpp:395/396`
  - fade完了系 publish

### PrepareToPlay 経路

- `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp:123/124`, `219/220`
  - 初期化/再初期化系 publish

### ReleaseResources 経路

- `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:121/122`
  - shutdown/release系 publish
- `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:190/191`
  - `clearPublishedRuntimeSnapshotsNonRt()`

**評価**: publish 実体は Coordinator 経由で揃うが、トリガー入口は複数。将来変更時の semantic drift を防ぐ運用規約が重要。

---

## 3.4 Retire/Reclaim Governance 実装確認

- `src/audioengine/AudioEngine.Threading.cpp`
  - `138/139`: `fallbackQueueDepth_`, `retireQueueDepth_` publish
  - `176-178`: `retireHighWatermark_`, `retireLowWatermark_`, `retireSaturationActive_` consume
  - `185/186`, `194`, `221/222`, `224`: watermark/saturation 更新 publish
  - `299/300`: queue depth consume

**評価**: backpressure/飽和回復ロジックは存在し、未実装判定は不適切。むしろ「運用閾値設計・検証の制度化」が次論点。

---

## 3.5 Contract/Verifier registry の存在確認

- 検索対象: `src/**/*.{h,cpp}`
- 検索語: `ContractRegistry|VerifierRegistry`
- 結果: **一致なし**

**評価**: registry層（命名上の明示オブジェクト）としては未確認。別名実装の可能性はあるため、必要なら次段で概念同等の構造（validator table/assert policy manager 等）を追加探索する。

---

## 4. 実運用リスク評価（破綻しにくさ観点）

- **高: Audio callback 側の多重semantic面参照**
  - タイミング差/意図しない組合せ参照が将来的に再混入する余地。

- **中: Publication Invariant の集中管理不足**
  - 入口分散そのものより、経路ごとに precondition/postcondition（freeze・policy・fade整合）が分岐することが本質リスク。

- **中: governance運用基準の明文化不足**
  - watermark値の妥当性、飽和時の期待遅延・復帰条件の品質基準が文書化不足だと、実運用での再発解析が難化。

- **低: registry命名層未検出**
  - 直ちに破綻には繋がらないが、監査容易性（説明責任）で不利。

---

## 5. 推奨アクション（優先順）

- **Priority 1: 観測面の単一化（最優先）**
  - `getNextAudioBlock` / `processWithSnapshot` で、参照ソースを「単一 authority view」へ寄せる。

- **Priority 2: Snapshot Semantic Unification**
  - `RuntimeGraph` / `GlobalSnapshot` / crossfade prepared snapshot の authority を1面へ収束させる。

- **Priority 3: Publication Invariant Centralization**
  - publish呼び出し前後の invariant を共通チェック化（非RTのみ実行）。

- **Priority 4: retire governance の運用指標化**
  - HWM/LWM/saturation 時の目標値と許容レンジをドキュメント化。

- **Priority 5: RuntimeGraph → RuntimeWorld 依存整理（次段階）**
  - 現状の `RuntimeGraph -> DSPCore*` 直結依存を、`RuntimeWorld -> Execution Topology -> DSPCore` の責務分離へ段階移行する。

- **Priority 6: 契約層の可視化（任意だが推奨）**
  - Contract/Verifier 相当を命名付きで配置し、レビュー容易性を上げる。

---

## 6. 最終判定（レビュー妥当性）

今回の詳細調査結果から、既存レビューの中核主張は**概ね妥当**。

- 妥当:
  - semantic unification 未達
  - observe path collapse 未達
  - semantic divergence（RTクラッシュではなく意味論ドリフト）が最大運用リスク
- 補正:
  - retire governance は「未実装」ではなく「実装済み＋運用基準未整備」
  - publish の論点は「入口分散」単体ではなく「Publication Invariant 集中管理」
- 未確認:
  - ContractRegistry / VerifierRegistry 実体（名称一致なし）

---

## 8. 追加レビュー反映（2026-05-30）

外部レビュー（提示テキスト）との突合結果、以下を本レポートの正式補足として採用する。

- **採用判断**
  - 本レポートは `Practical Stable ISR Bridge Runtime` 観点のベースライン評価として妥当。

- **強化された解釈**
  - 最大リスクはクラッシュより **semantic divergence**（意味論ドリフト）。
  - Publish は「入口の数」より **Invariant を1か所で担保できるか** が本質。
  - Retire Governance は存在しており、課題は「有無」ではなく **チューニング/制度化**。

- **将来課題の明示（Bridge次段）**
  - `RuntimeGraph -> DSPCore*` の強結合は、最終ISR化での柔軟性を下げる。
  - 中期的に `RuntimeWorld -> Execution Topology -> DSPCore` への責務分離を進める。

- **成熟度レンジ（運用実感に合わせた暫定）**
  - RT安全化フェーズ: **ほぼ完了**
  - Practical Stable ISR Bridge Runtime: **80～85%**
  - Final ISR Runtime: **未到達**

---

## 7. 付記

本レポートは、該当日付時点の `main` ワークツリー上の静的読解・検索結果を整理したもの。
