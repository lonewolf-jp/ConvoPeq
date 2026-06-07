以下は、**提示された ConvoPeq.md（ISR Bridge Runtime 周辺コード）に対する「実運用で破綻しやすい未達成ポイントの抽出」**です。
評価軸は「理論的正しさ」ではなく、**長時間運用・負荷変動・並列性・障害耐性で破綻するかどうか**です。

---

# 1. 未達成な内容（重要度順）

## ① Epoch / RCU と Snapshot ライフサイクルの「責務分散」

### 問題

Snapshot の寿命管理が以下に分散：

* `SnapshotCoordinator`
* `EpochDomain`
* `ISRRetire`
* `RuntimePublicationCoordinator`
* `DeferredDeletionQueue`
* `ISRDSPQuarantine`

👉 **RCU系システムとしては典型的な「二重以上の回収経路」状態**

---

## 該当コード

```cpp
m_epochDomain->enqueueRetire(oldTarget, snapshotDeleter, retireEpoch);
```

```cpp
m_epochDomain->publish();
m_epochDomain->reclaimRetired();
```

```cpp
SnapshotFactory::destroy(...)
```

さらに：

* `ISRDSPQuarantine`
* `ISRRetireRuntimeEx`
* `DeferredDeletionQueue`

👉 **同一オブジェクトが複数経路で寿命管理可能**

---

## あるべき姿

* Snapshot の lifetime は **単一 authority（RCU/Epoch）に統一**
* destroy 経路は1つのみ
* retire は「登録のみ」で副作用なし

---

## 改修方法

### ✔ 必須リファクタ

```text
Snapshot destruction authority = EpochDomain のみ
```

### 修正：

* `SnapshotFactory::destroy` を直接呼ぶ経路禁止
* Quarantine / DeferredDeletionQueue を EpochDomain に統合
* retire API を「enqueueのみ」に制限

---

# ② SnapshotCoordinator が「状態機械 + メモリ管理 + フェード制御」を兼務

## 問題

責務が過密：

* Snapshot pointer 管理
* fade state 管理
* RCU retire 発行
* atomic slot 操作
* lifecycle transition

---

## 該当コード

```cpp
GlobalSnapshot* oldTarget = m_slots.exchangeTarget(...)
m_fade.start(fadeSamples);
m_epochDomain->enqueueRetire(...)
```

---

## 問題の本質

👉 ISR Bridge の中心が **“God Object化”している**

特に問題なのは：

* fade
* snapshot swap
* retire scheduling

が **同一関数内で直列実行されている点**

---

## あるべき姿

責務分離：

```text
SnapshotCoordinator
  ├── SnapshotSlotManager
  ├── FadeController
  ├── RetireScheduler (RCU Adapter)
```

---

## 改修方法

* `SnapshotSlotStore` を coordinator 外へ完全分離
* fade は `SnapshotFadeEngine` に移動
* coordinator は state transition event のみ生成

---

# ③ atomic ordering の意味が局所的で全体整合性が保証されていない

## 問題

個別に memory_order が書かれているが：

* global consistency model が存在しない
* acquire/release のペアが局所最適
* epoch と atomic が混在

---

## 該当コード

```cpp
exchangeTarget(..., std::memory_order_acq_rel);
loadCurrent(..., std::memory_order_acquire);
```

---

## 問題の本質

👉 「HB（happens-before）」は成立しているが、

**“システム全体の順序モデル”が定義されていない**

結果：

* スナップショット更新順序が理論的に保証されない箇所がある
* クロススレッド可視性が局所依存

---

## あるべき姿

* ISR Bridge 全体に以下を定義：

```text
Phase Ordering Model:
  Publish → Activate → Retire → Reclaim
```

---

## 改修方法

* ordering rules を `ISREpochMemoryModel.h` に統合
* atomic はルールベース化（直接記述禁止）

---

# ④ Snapshot の生成と比較ロジックが二重化されている

## 問題

```cpp
SnapshotFactory::create()
SnapshotFactory::createImpl()
SnapshotFactory::areSnapshotsEquivalent()
SnapshotFactory::computeContentHash()
```

👉 判定ロジックが散在

---

## リスク

* 差分生成の不整合
* キャッシュ破壊
* 無効 snapshot が混入

---

## あるべき姿

```text
SnapshotDecision Engine = 単一入口
```

---

## 改修方法

* `SnapshotFactory` を2層化

```text
SnapshotBuilder（生成専用）
SnapshotEquivalenceEngine（判定専用）
```

---

# ⑤ Retire系処理がリアルタイム境界を跨いでいる

## 問題

```cpp
m_epochDomain->enqueueRetire(...)
m_epochDomain->reclaimRetired()
```

＋

```cpp
DeferredDeletionQueue
ISRDSPQuarantine
```

---

## 問題の本質

👉 RTスレッドとNon-RTスレッドの境界が曖昧

* retire enqueue はRT
* reclaim は不定周期
* quarantine がさらに中間層

---

## あるべき姿

```text
RT → RetireToken発行のみ
Non-RT → 回収専用スレッド
```

---

## 改修方法

* `reclaimRetired()` を RT から完全排除
* 専用 GC thread に統一
* Quarantine 廃止 or GC統合

---

# ⑥ ObservedRuntime の安全性が「スレッドID依存」で脆弱

## 問題

```cpp
if (ownerThreadId != std::this_thread::get_id())
    return nullptr;
```

---

## 問題の本質

* thread id ベースは再利用・転送環境で破綻
* lock-free設計と整合しない
* coroutine / task system と衝突

---

## あるべき姿

* thread id 禁止
* epoch guard ベース

---

## 改修方法

```text
Thread Identity → Epoch Reader Token に置換
```

---

# ⑦ fade state が snapshot lifecycle と密結合

## 問題

```cpp
m_fade.start(fadeSamples);
m_fade.resetToIdle();
```

* snapshot swap と fade が同期依存

---

## 問題の本質

👉 「音響状態」と「構造状態」が混在

---

## あるべき姿

```text
Fade = Signal domain
Snapshot = Structure domain
完全分離
```

---

## 改修方法

* fade を DSP graph 側へ移動
* snapshot は immutable に固定

---

# 総合評価（実運用耐性）

現状は：

| 領域                | 状態         |
| ----------------- | ---------- |
| Snapshot lifetime | ⚠ 分散・競合あり  |
| Thread safety     | ⚠ 局所成立     |
| RCU構造             | ⚠ 二重管理あり   |
| Retire処理          | ❌ 境界不明確    |
| DSP安定性            | ⚠ fade依存あり |

---

# 結論

Practical Stable ISR Bridge Runtime は現状：

> **「理論的に整合した分散RCU設計」ではなく
> 「複数のRCU系パターンを重ねた準ハイブリッド構造」**

そのため実運用では主に以下で破綻しやすい：

* retire二重解放
* snapshot不整合
* fadeとswapの競合
* reclaim遅延によるメモリ圧迫
* atomic順序の設計逸脱

---

必要なら次に：

* 「Phase-2〜Phase-3の崩壊点マップ」
* 「完全単一RCU化アーキテクチャ案」
* 「リアルタイム安全性スコアリング」

まで落とせます。
