# P1-5-IR-P2 — Step 5-AF / P3-5-R24: Coordinator/Admission Boundary Design Audit（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R24）
- **種別**: read-only design audit。実装・build・run なし。
- **目的**: R23 の `cmt>0 / seq=0` 形（main-site commit-enqueue 到達後・publish 未達）を
  commit → Coordinator → Admission → Publish 間で分離する最小観測点を1点選定する。
- **結論**: **R24-A**。Coordinator intent take 点を最小点として確定（§6）。
  次は実装承認ゲートへ。

---

## 1. State Freeze

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a／kP15FullMatrix true／OFF
R16 take＋R19 B2＋R22 commit＋R10 accessor＋T6/T9＋delta観測 保持
ConvoPeq.md 再生成済み（本 Step 開始時）。HEAD と R23 state の一致を確認。
R23 source 差分＝承認範囲のみ（reread 前提を満たす）。
R23 vehicle 保持。R23文書のみを根拠に決めない。
```

## 2. R23 observation baseline

```text
Case A： take=1／bld=0／cmt=0／seq=0 → B2前（build-stage）
Case B： take=1／bld=1／cmt=0／seq=0 → B2後・commit前
Case C： take=1／bld=1／cmt=1／seq=0 → **cmt後・publish前（R24 の対象）**
Case D： take=1／bld=1／cmt=1／seq=1 → publish 到達
```

実装はなく、requested state と active state は分離記録である（R6 規律）。
`cmt>0／seq=0` が gap で実測された。

## 3. Source trace（最新 ConvoPeq.md 基準で再確認）

```text
main-site commit enqueue（R22 counter：:1420）
        ↓
enqueuePublicationIntentForRuntimeCommit（Commit.cpp:800）
  │ currentBuildSnapshot_ 更新（Mutex）
  │ handle 事前登録（registerDSPHandleForRuntime）
  └─> internal queue
        ↓
Coordinator consume（Orchestrator／CoordinatorLoop）
        ↓
  ├─ (prior) checkLatestCommitSequentialEquivalence()（:1668）
  ├─ admission evaluate（PublicationAdmission）
  ├─ defer／retry（Coordinator::submitPublishRequest）
  ├─ publish orchestration
  └─ RuntimeWorld commit → sequence bump（Commit.cpp:402）
        ↓
publication sequence increment（+1）
```

- R23 `cmt` の計測点から `seq` writer までの全分岐を確認した。
- call site 混入： recovery 系にも enqueue call site が2件（RebuildDispatch.cpp:1085／:1165）
  存在するが、R22 counter は **main-site のみ**を数える（recovery 非加算・設計済み）。

## 4. 分離対象（3概念の混同禁止・R24 §4 再掲）

```text
A. Coordinator到達：     commit enqueue → Coordinator が intent 取得
B. Admission受理：       Coordinator → Admission → accepted
C. Publish：             accepted → RuntimeWorld commit → sequence++
```

R23 の C 結果 `seqΔ=0` は保有する。次に要るのは A または B の最小観測点である。

## 5. Candidate observation points（比較・5案）

| 候補 | 意味 | RT／NonRT | 既存観測 | 判定 |
| --- | --- | --- | --- | --- |
| Coordinator intent take | commit後 Coordinator が取得 | NonRT（Orchestrator） | なし | **有力候補（§6 採用）** |
| Coordinator dispatch | Coordinator 処理開始 | 同上 | なし | 有力だが take と同等粒度。本設計では take 側を採用 |
| Admission attempt | Admission 判定開始 | 同上 | なし | 有力だが take で網羅可能（§6）。次段温存 |
| Admission accepted | publish 可能状態到達 | 同上 | なし | 同上 |
| Admission rejected | rejection 確認 | 同上 | lastDroppedGeneration 既存（§3） | take 観測後に drop で裏付け可能。単独抱合はせず take 点と併用 |
| retry／defer | defer 発生 | 同上 | なし | 原因計装になりやすく慎重（本監査では見送り） |
| sequence bump | 既存 seq | 原子既存 | あり | **追加不要** |

- retry／defer の個別計数は原因計装に直結するため本 Step では見送る（R24 §7遵守）。
- drop reason 分類・per-task linkage は禁止（R24規律）。

## 6. Minimal observation selection（1点のみ）

**Coordinator intent take 点を1点選定する**（R24 §6 の①）。

```text
commit=1
coord=0 → seq=0 なら commit後→Coordinator到達前
commit=1
coord=1 → seq=0 なら Coordinator到達後→publish前
```

- queued／taken／bld／cmt と合わせて `commit → coordinator → publish` の最小拡張が成立する。
- R21／R22 と同様に **main-site 系のみ**を数える（recovery 経路を巻き込まない）。
- 退避観測点として `drop`（lastDroppedGeneration Δ）は既に保有しており、
  将来の admission 側深掘り時に併用可能である。

## 7. Instrumentation contract（実装する場合）

```cpp
std::atomic<std::uint64_t> coordinatorTakeCount_ { 0 };
// ...
[[nodiscard]] std::uint64_t getCoordinatorTakeCount() const noexcept
{
    return convo::consumeAtomic(coordinatorTakeCount_, std::memory_order_acquire);
}
```

候補位置：Coordinator が main-site intent を take した直後（厳密位置は次段実装監査で
actual call site を re-trace して確定する。本監査では take 境界そのものは確定し、
実装位置は「Coordinator take 直後」に限定する。）

- getter のみ。writer＝1箇所。既存カウンタ（take／bld／cmt／dispatch）不変。
- const／noexcept／POD／scalar only／ownership 非露出／RT影響なし／
  publish-retire-rebuild 副作用なし（R7–R22 同基準）。
- test-only は差分読出し＋既存 WARN 行への `coord` field 追加のみ。

## 8. STOP＋R24 Gate

```text
R24-A： ADOPTED
  最新ソースで chain 再確認／commit→Coordinator→Admission→seq 境界確定／
  最小1点（Coordinator take）選定／既存 counter と意味重複なし／
  recovery 等別経路混入なしのすべて成立。
  → 次は Coordinator take counter の実装承認ゲートへ（本 Step では実装しない）。
R24-B／R24-C： 非該当。
```

- F／R 実行・P3-1-D・limiter／stale／crossfade 帰属なし。
- R4 境界・保留事項・§7解釈制約・H-B 対象外を維持する。
