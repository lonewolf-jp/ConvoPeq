# ConvoPeq ISR Phase 0: Runtime Causality Specification

本書は `doc/plan4.md` / `doc/rule4.md` / `doc/rule4-coding.md` / `doc/detailed_design_plan4_rule4_jp.md` に整合する、
**Phase 0（因果関係仕様化）の凍結仕様**である。

本書で規定した happens-before（HB）と memory order は、以降の Phase 1 〜 6 で変更してはならない。

---

## 1. スコープと設計拘束

### 1.1 対象

- `RuntimeStore::current`（`std::atomic<const RuntimeState*>`）
- `PublicationLog`（MPSC append / single-consumer）
- `PublicationCoordinator` の publish / retire 線形化
- `EpochDomain` の reader protocol / safe reclaim

### 1.2 非対象

- UI 機能仕様
- DSP アルゴリズム仕様
- Phase 1 以降の構造分割詳細

### 1.3 最高優先不変条件

1. Runtime 可視状態の同期点は `RuntimeStore::current` のみ。
2. publish 単位は `RuntimeState` 全体のみ（split publication 禁止）。
3. reclaim 安全条件は **`readerEpoch > retiredEpoch`**（`>=` 禁止）。
4. RT は consume only（publish/retire/reclaim 禁止）。

---

## 2. HBチェーン（規範）

以下の HB は必須であり、いずれか1つでも崩れた実装は不合格とする。

### 2.1 RuntimeState publish/observe/retire/reclaim

1. `publish(newState)`
   - `current.store(newState, memory_order_release)`
2. `observe()`
   - `current.load(memory_order_acquire)`
3. `retire(oldState)`
   - retire queue enqueue を release で publish
4. `reclaim(oldState)`
   - reader table を acquire で走査し、安全条件成立時のみ reclaim

HB要件:

`publish(newState)` HB `observe(newState)`

`publish(newState)` HB `retire(oldState)`

`retire(oldState)` HB `reclaim(oldState)`

### 2.2 PublicationLog append/consume/retire/reclaim

1. producer は payload 初期化完了後に link CAS（release）を実行。
2. consumer は `head->next` を acquire load で観測して消費。
3. consumer は消費済み node を retire enqueue（release）。
4. reclaimer は acquire 走査で安全判定し reclaim。

HB要件:

`append(payload write + link CAS release)` HB `consume(next acquire + payload read)`

`consume(node)` HB `retire(node)` HB `reclaim(node)`

### 2.3 Epoch advancement / safe reclaim

1. retire node に `retiredEpoch = E` を記録（release）。
2. reader は `leaveEpoch()` で quiescent を release publish。
3. reclaimer は reader table を acquire load で走査。
4. **全 active reader で `localEpoch > E`** のときのみ reclaim 可能。

---

## 3. memory_order 契約（固定表）

| 操作 | 許可オーダー | 根拠 |
| --- | --- | --- |
| RuntimeState publish (`current.store`) | `release` 固定 | publish HB observe を形成 |
| RuntimeState observe (`current.load`) | `acquire` 固定 | publish後の完全初期化可視化 |
| PublicationLog link CAS (`tail->next`) success | `release` 固定 | payload visibility publish point |
| PublicationLog link CAS failure | `acquire` 固定 | 競合先進行の可視化 |
| PublicationLog tail correction CAS success | `release` 固定 | tail前進の可視化 |
| PublicationLog tail correction CAS failure | `relaxed` 許可 | 補助失敗時の再試行のみ |
| retire enqueue | `release` 固定 | enqueue HB reclaim |
| reclaimer reader scan | `acquire` 固定 | quiescent publish と対応 |
| `leaveEpoch()` quiescent publish | `release` 固定 | reclaim判定側 acquire と対応 |
| RT-local 非共有カウンタ | `relaxed` のみ許可 | 他スレッド同期不要 |

禁止:

- `seq_cst` の便宜導入
- publication path での `relaxed`
- helper 外 atomic dot-call

---

## 4. PublicationLog 線形化仕様（MPSC + single-consumer）

### 4.1 役割

- producer: commit intent 発行側（非RT、複数）
- consumer: `PublicationCoordinator` のみ（単一）

### 4.2 線形化点

1. append linearization point:
   - `tail->next` への CAS success 時点
2. consume linearization point:
   - consumer が `head->next` を acquire load で観測し、`consumedTail` を進めた時点

### 4.3 FIFO保証

- append CAS success 順を consume 順とする。
- consumer は単方向走査順以外で再順序化してはならない。

### 4.4 MPSC append 擬似手順（固定）

```cpp
for (;;) {
    Node* tail = tailPtr.load(std::memory_order_acquire);
    Node* next = tail->next.load(std::memory_order_acquire);

    if (next == nullptr) {
        if (tail->next.compare_exchange_weak(
                next,
                newNode,
                std::memory_order_release,
                std::memory_order_acquire)) {
            tailPtr.compare_exchange_strong(
                tail,
                newNode,
                std::memory_order_release,
                std::memory_order_relaxed);
            break;
        }
    } else {
        tailPtr.compare_exchange_strong(
            tail,
            next,
            std::memory_order_release,
            std::memory_order_relaxed);
    }
}
```

必須制約:

- helping rule（`next != nullptr` 時の tail correction）省略禁止。
- `newNode->next = nullptr` 初期化を append 前に完了。
- append 前 node 再利用禁止（単一割当・単一retire）。
- consumer 以外の consume/truncate/reclaim 禁止。

---

## 5. Epoch reader protocol（規範）

### 5.1 enter/leave

- `enterEpoch()`:
  - `recursionDepth++`
  - 0→1 遷移時のみ `localEpoch` を acquire 取得
- `leaveEpoch()`:
  - `recursionDepth--`
  - 1→0 遷移時のみ quiescent を release publish

### 5.2 例外条件

- underflow は fail-fast（debug assert 必須）
- `kMaxEpochRecursionDepth` 超過は fail-fast

### 5.3 callbackシーケンス（固定）

1. callback entry: `enterEpoch()`
2. `observe()`（1回のみ）+ DSP
3. callback exit: `leaveEpoch()`

禁止:

- callback中 multiple observe
- guard なし pointer 保持
- thread handoff

---

## 6. supersede時の順序契約（固定）

1. `new RuntimeState` fully-constructed 完了
2. `publish(newState)`（release）
3. reader が `observe(acquire)` で `newState` 可視化可能化
4. `retire(oldState)` enqueue（release）
5. `EpochDomain` safe判定後 `reclaim(oldState)`

禁止:

- publish 前 retire
- transition 単体 direct retire/reclaim
- partially initialized state publish

---

## 7. failure atomicity 契約

- build failure 時は publish を実行しない。
- `current` は旧 state のまま維持。
- 失敗オブジェクトは publish 前に局所破棄。
- 失敗オブジェクトの retire 登録禁止。

---

## 8. 実装レビュー観点（Phase 1以降の強制チェック）

1. HB chain が本書 2章と一致するか。
2. memory_order が本書 3章固定表に一致するか。
3. PublicationLog の線形化点が 4.2 と一致するか。
4. safe reclaim 不等号が `>` になっているか。
5. helper外 atomic dot-call が存在しないか。
6. callback内 single observe が維持されているか。
7. publish authority / retire authority の漏洩がないか。

---

## 9. 受入判定（Phase 0完了条件）

Phase 0 は以下を満たした時点で完了とする。

- HB chain（publish/observe/retire/reclaim）が文書化済み
- operation別 memory_order 固定表が存在
- PublicationLog append linearization が定義済み
- epoch advancement 条件が定義済み
- safe reclaim 不等号が `readerEpoch > retiredEpoch` と明記済み

本書は ISR移行の因果仕様の唯一参照とし、未記載の memory order を実装に追加してはならない。

---

## 10. 現行ソース再調査（2026-05-19, 再採番ドラフト）

本章は Phase 0 仕様書に対する「現行追従状況」の監査付録であり、
`doc/detailed_design_plan4_rule4_jp.md` の `15.11` と **同一採番（C/H/M）** で同期する。
`10.1`〜`10.3` の旧A採番は履歴として本更新で置換する。

### 10.1 再採番サマリ（Open項目のみ）

| 区分 | 件数 | 意味 |
| --- | --- | --- |
| Critical | 1 | plan4 最終形（単一 publish unit / 最終責務分割）に未到達 |
| High | 0 | 最終形へ到達するための主要な構造・統制ギャップはクローズ済み |
| Medium | 0 | 追加検証・監査更新で収束可能な未確定事項はクローズ済み |

### 10.2 Critical（再採番）

| ID | 乖離内容 | 主要証跡（現行） | 判定理由 |
| --- | --- | --- | --- |
| C-03 | SnapshotCoordinator 解体未完 | `src/audioengine/AudioEngine.h:73,791,2661`, `src/core/SnapshotCoordinator.h` | Phase 5 要件（責務分割完了）に未到達 |

### 10.3 High（再採番）

| ID | 乖離内容 | 主要証跡（現行） | 判定理由 |
| --- | --- | --- | --- |

### 10.4 Medium（再採番）

| ID | 乖離内容 | 現状 | 次の検証 |
| --- | --- | --- | --- |

### 10.5 解消済み（旧IDクローズ）

以下は現行 `src/**` 実測で解消確認済み（再発監視は継続）。

- 旧 `C-03`（dual epoch）: `EpochManager::instance(` / `EpochCoreReaderGuard` / `g_deletionQueue` ヒット 0
- 旧 `C-08`（shutdown ignoring epoch）: `reclaimAllIgnoringEpoch` ヒット 0
- 旧 `C-04`（completion side-channel）: `m_fadeCompleted` ヒット 0
- 旧 `C-05`（abort direct destroy）: `abortFade` ヒット 0
- 旧 `C-01`（単一 publish unit 未達）: `currentDSPBits` / `fadingOutDSPBits` を削除し、NonRT 側保有へ統合
- 旧 `C-02`（`RuntimeState` 未導入）: `src/audioengine/AudioEngine.h` で `struct RuntimeState` を導入（`RuntimePublishWorld` は互換 alias）
- 旧 `H-01`（`seq_cst`）: `memory_order_seq_cst` ヒット 0
- 旧 `H-02`（observe lifetime 契約未収束）: `observeCurrentRuntime` を主契約へ統一し、`observeCurrentSnapshot` / `getSnapshotCoordinator` 公開面を撤去
- 新 `H-01`（PublicationCoordinator core 独立型未確立）: `src/core/RuntimePublicationCoordinator.h` を導入し、`AudioEngine` の nested 実装を `RuntimePublicationBridge` + 外部 coordinator 型へ置換
- 新 `H-03`（RuntimeStore 書込み権限の最終閉鎖未完）: `RuntimeStore<RuntimePublishWorld, convo::RuntimePublicationCoordinator<...>>` へ固定し、`acquireWriteAccess()` を外部 owner 型の static factory 経由に限定
- 旧 `H-04`（監査章と実装進捗の時差）: 15章/10章/task の C/H/M 再採番を同期反映
- 旧 `H-05`（Stage 運用チェック更新遅延）: `doc/task.md` 未チェック項目を再採番ベースで正規化
- 旧 `M-01`（EQ cache snapshot ownership 最終判定）: `AudioEngine.Cache.cpp` の `CacheMap` RAII + `enqueueDeferredDeleteNonRt` 経路で所有権/寿命一元化を再確認
- 旧 `M-02`（completion通知の SPSC 一本化最終判定）: `SnapshotCoordinator.cpp` の `tryCompleteFade` CAS 一回性 + `m_fadeCompleted` 非依存化を再確認
- 旧 `M-03`（thread handoff 禁止の静的検証）: `check-src-atomic-dotcall.ps1` に ObservedSnapshot handoff 検知ルールを追加し、Strict scan pass
- 旧 `M-04`（PublicationLog 線形化の受入証跡形式）: `doc/task.md` に受入証跡テンプレートを追加

### 10.6 同期運用注記

- 本章は `doc/detailed_design_plan4_rule4_jp.md` `15.11` の再採番結果に同期する。
- 今後の更新は **15章と10章を同一コミットで同時更新**する。

---

## 11. Phase 0 継続運用ルール（15.9準拠）

`doc/detailed_design_plan4_rule4_jp.md` 15.9 の変更を受け、Phase 0 は以下の運用を追加固定する。

### 11.1 方針

- 個別バグ修正ではなく、**Phase 境界強制の移行統制**として進める。
- 「新実装追加」より **旧経路削除**を優先する。
- **未完成 ISR + 旧 mutable runtime の長期共存を禁止**する。

### 11.2 強制順序

1. Phase 0 固定化（本書）
2. Phase 1 Epoch 統合
3. Phase 2 Publication / Transition 線形化
4. 旧 runtime 同期面の強制削除

禁止:

- 旧 epoch 残存のまま RuntimeStore 導入
- `mutex queue` と `PublicationLog` の併存運用

### 11.3 Phase 1 着手ゲート（本書準拠）

Phase 1 に進む前に、最低限以下を確認する。

- 本書 2章 HB と 3章 memory_order の改変がない
- `reclaimAllIgnoringEpoch` 撤去計画が task 化済み
- dual epoch 解消の callsite 移設計画が task 化済み

---

## 12. 監査更新ルール

各フェーズ完了時、以下を必ず更新する。

1. 本書 10章（現行監査）
2. `doc/detailed_design_plan4_rule4_jp.md` 15章
3. `doc/task.md`（Stage B 以降の完了条件）

更新時は「件数」ではなく、必ず file:line 証跡を更新する。
