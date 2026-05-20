# ConvoPeq Shared EpochDomain Scalability 検証計画書

## 目的

本書は、`GlobalSnapshot` と `RuntimePublication` が共有する EpochDomain について、
**負荷増加時でも retire/reclaim の安全性と進行性を維持できるか** を検証するための計画を定義する。

狙い:

- shared strategy の採用妥当性を定量的に確認する
- bug2 系 UAF 再発防止に必要な証跡を残す
- 失敗時の縮退運用（fallback）を事前確定する

### REV3.2運用優先注記

- 本書の epoch strategy 比較は設計参照表現として扱う。
- 実装運用は `plan5.md` REV3.2 を優先し、
  `runtime exposes evidence / CI validates evidence` を固定方針とする。
- epoch arbitration の最終配置は RetireRuntime 内部責務への統合優先で解釈する。

用語正規化（齟齬回避）:

- 本書では `RuntimePublication` を正規記法として扱う。

---

## 検証対象と前提

検証対象:

- shared EpochDomain
  - producer: Message publish path / background handoff
  - observer: Audio callback read path
  - reclaim: SnapshotRetireManager / RuntimeWorldRetireManager

前提:

- Audio Thread で allocate/free/lock を行わない
- retire authority は単一化済み
- DSPHandle allocator policy が適用済み

---

## 検証シナリオ

### S1: 定常負荷

- 中程度 publish 頻度で長時間実行
- retire backlog が収束するか確認

### S2: バースト負荷

- publish/retire を短時間に集中発生
- grace 待機が増えても reclaim が停滞しないか確認

### S3: 非対称負荷

- producer 高速 / consumer 低速を意図的に作る
- queue 水位と遅延が受入範囲内か確認

### S4: shutdown 境界

- shutdown 開始直前に retire backlog を作成
- drain 手順で取り残しなく終端できるか確認

---

## 測定項目（KPI）

### M1: Retire Queue 水位

- `retire_queue_depth_max`
- `retire_queue_depth_p95`
- 観測点: scenario 全区間

### M2: Reclaim 遅延

- `retire_to_reclaim_latency_ms_p50/p95/max`
- 観測点: object family 別（GlobalSnapshot / RuntimePayload / DSPHandle family）

### M3: Grace 進行性

- `epoch_advance_rate_per_sec`
- `grace_wait_timeout_count`
- 観測点: burst / 非対称区間

### M4: 安全性シグナル

- `stale_handle_reject_count`
- `payload_closure_violation_count`
- `uaf_suspect_count`（0 必須）

### M5: 終端健全性

- `shutdown_drain_remaining_count`
- `deferred_fallback_remaining_count`
- 観測点: shutdown 完了時

---

## 受入基準（Acceptance Criteria）

### A1: 安全性（必須）

- `uaf_suspect_count == 0`
- `payload_closure_violation_count == 0`
- grace 条件未達で destroy が発生しない

### A2: 進行性（必須）

- すべての scenario で reclaim が停滞せず、最終的に backlog が収束
- `grace_wait_timeout_count` が連続増加しない

### A3: 遅延（推奨）

- `retire_to_reclaim_latency_ms_p95` が S1/S2 で設計想定内
- `retire_to_reclaim_latency_ms_max` が shutdown 直前を除き異常スパイクを示さない

### A4: 終端（必須）

- shutdown 完了時に `shutdown_drain_remaining_count == 0`
- fallback queue の残留が 0

注記:

- 「設計想定内」の閾値は実測初回ベースライン確定後、同一文書へ追記して固定する。

R10 固定ルール:

- shared 継続 / split 移行の判定は次の3軸で行う:
  - latency（reclaim latency p95/p99）
  - callback jitter（audio callback duration jitter）
  - reclaim burst（単位時間あたり reclaim 件数ピーク）
- 判定は必ず shared と split の同一シナリオ比較で実施する。

---

## 失敗時フォールバック（事前確定）

### F1: Shared -> Split Epoch 切替

発動条件:

- A2（進行性）不達が再現性ありで確認された場合

内容:

- `GlobalSnapshot` 系と `RuntimePublication` 系の EpochDomain を分離
- retire queue も domain 別に分割

制約:

- authority 単一化原則は維持（domain 内で単一）

移行手順（最小）:

1. split epoch 用 domain を作成
2. RuntimePublication family を新domainへ再割当
3. retire queue を domain 分割
4. grace/reclaim pipeline を段階切替
5. rollback 可能な状態で burn-in 検証

### F2: Reclaim Throttling / Batch Policy 見直し

発動条件:

- A3（遅延）不達、ただし A1/A2 は満たす場合

内容:

- reclaim batch size / tick 間隔を非RTで再調整
- burst 時の backlog 吸収を優先

### F3: Shutdown 強制 Drain モード

発動条件:

- A4（終端）不達

内容:

- shutdown フェーズ限定で drain 優先モードを適用
- fallback queue を先行空にしてから最終 reclaim を実行

---

## 実施手順

1. 計測カウンタを有効化（Debug/Release 両方）
2. S1->S2->S3->S4 の順に同一ビルドで実行
3. KPI を scenario 別に集計
4. A1〜A4 判定
5. 不達時は F1/F2/F3 の該当分岐を適用し、再検証

---

## 成果物

- scenario 別 KPI 集計表
- shared vs split 比較表（latency/jitter/reclaim burst）
- 受入判定（Pass/Fail）
- Fail 時の fallback 適用記録
- 次フェーズ（Phase B/C）への Go/No-Go 判定

Go/No-Go 判定規則（R10）:

- Go(shared継続): shared が split に対して latency/jitter/burst の全軸で劣後しない
- Go(split移行): split が shared より 1軸以上で安定性優位かつ他軸で許容内
- No-Go: いずれの方式も A1/A2/A4 を満たさない

---

## 参照

- `doc/work/plan5.md`
- `doc/work/ISR_HB_Graph_Specification.md`
- `doc/work/ISR_Retire_Authority_Graph.md`
- `doc/work/ISR_Runtime_State_Matrix.md`
- `doc/work/ISR_DSPHandle_Allocator_Policy.md`
