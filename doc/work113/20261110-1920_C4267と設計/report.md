# C4267 の独立パッチと unmatched 記録の設計案（2026-11-10）

## §0 範囲

ソース・テスト・閾値・設定の変更、ビルド、テスト、Git 操作は行っていない。以下は**パッチと設計案の提出**である。

## §1 ① worldLifetime の経路分離（維持）

現行仕様を維持する。`bootstrap`／`reconfigure`／`deferred` は `publish` に含まれ、`bootstrap` の `unmatched = 1` は既知の記録差として扱う。API 変更は行わない。

## §2 ② C4267 の独立パッチ

### 2.1 パッチ

`patch/c4267_size_guard.patch`（1,001 バイト、2 ハンク）。適用対象は `src/InputInspection.cpp`（現在の 8,406 バイト、SHA-256 `C7259580…`）。

```diff
@@ -1,6 +1,8 @@
 #include "InputInspection.h"
 
+#include <cstddef>
 #include <cstring>
+#include <limits>
 
 namespace convo::safeinput
@@ -15,7 +17,14 @@
     const char* d = text.toRawUTF8();
-    const int n = text.getNumBytesAsUTF8();
+    const std::size_t numBytes = text.getNumBytesAsUTF8();
+
+    // C4267: size_t -> int の変換前に範囲を検査する。8 MiB の上限（kMaxXmlFileBytes）では
+    // 到達しないが、公開 API が limits を差し替え可能なため、契約として明示的に弾く。
+    if (numBytes > static_cast<std::size_t> (std::numeric_limits<int>::max()))
+        return -1;
+
+    const int n = static_cast<int> (numBytes);
     int depth = 0;
```

### 2.2 設計意図

- ご指示どおり、**`size_t` のまま `INT_MAX` と比較**してから符号付きへ変換する。比較を `int` に落としてから行うと、変換時点で既に切り捨てが起きているため無意味になる。
- 上限超過時は `-1` を返す。`computeMaxElementDepth` の既存契約（構造を判定できない場合は `-1`、呼出側は `fail`）と同じ扱いで、新しい分岐を増やさない。
- 現行の 8 MiB 上限ではこの分岐に到達しない（最悪 24 MiB）。到達しないこと自体は変えず、契約として明示する。

### 2.3 確認したこと

| 項目 | 結果 |
|---|---|
| パッチの適用（一時ツリー、`patch -p0`） | 警告・fuzz なしで成功 |
| 適用結果の一致 | 事前に用意した修正版とバイト一致 |
| 変更行数 | 追加 9 行（include 2 行を含む） |
| 影響範囲 | `computeMaxElementDepth` の入口のみ。走査ロジック・判定条件・戻り値は不変 |

### 2.4 未確認

- **コンパイル検証は未実施**（ビルド禁止のため）。`std::numeric_limits`（`<limits>`）と `std::size_t`（`<cstddef>`）の追加で足りることはコード上確認したが、実ビルドは承認後に行う。
- 警告が消えることの確認も未実施（同上）。

### 2.5 挙動の差分

| 入力サイズ | 変更前 | 変更後 |
|---|---|---|
| 〜8 MiB（全経路） | 走査する | 同じ（`numBytes ≤ INT_MAX` のため分岐しない） |
| 約 715 MB 超（現行では到達不能） | `int` へ切り捨て → 未定義の走査 | `-1` を返し、呼出側は `fail` |

## §3 ③ unmatched 記録の設計案（提案 C）

### 3.1 目的

D1 の `unmatched = 3` を 0 にする。対象は次の 3 系統で、**要求時刻の意味を維持**し、所有権移転・retry・reclaim 条件を変えない。

### 3.2 呼出し関係（全経路）

```
[公開]  AudioEngine::enqueueRuntimePublicationFireAndForget
          └─ ownerChannel.enqueue（World。D1 の対象外）

[D1 要求]
  A. ISRRetireRouter::enqueueRetire(ptr, deleter, epoch, type)      ← 現在の記録点
       ← ISRRetireRouter::enqueueWithRetry
            ← AudioEngine::enqueueDeferredDeleteNonRt
            ← ISRRuntimePublicationCoordinator::enqueueRetire   (B)
            ← EQProcessor（自前の router 経由）
       ← ISRRetireRouter::retire（Non-RT）
  B. ISRRuntimePublicationCoordinator::enqueueRetire
       └─ router.enqueueWithRetry → A
  C. SnapshotCoordinator
       C1. startFade → enqueueWithRetry(*m_epochProvider, ...)   （直接 D。A を通らない）
           └ 失敗時 quarantineRetireSink → Q → E → T
       C2. resetFadeStateAndRetireTarget → m_epochProvider->enqueueRetire(...)  （直接 D）
       C3. completeFade → enqueueWithRetry(*m_epochProvider, ...) （直接 D）
           └ 失敗時 quarantineRetireSink → Q → E → T
       C4. quarantineRetireSink 内の m_retireSink->terminalReclaim(...)  （T。C1/C3 の失敗時のみ）
  D. ISRRetireRouter::retireRT
       └─ provider_->enqueueRetire（直接 D）            ← 項目 3 で記録済み
  E. terminalReclaimRetryable                          （本番呼出元なし。テストのみ）

[完了]
  D キュー: reclaim / drainAllUnsafe
  Terminal: drain（legacyPending / 通常）/ terminalReclaim（同期）/ drainAll
  Q: drain / drainAllUnsafe
  Retryable: terminalReclaimRetryable / drainAll の retryable 分岐
```

現在、要求が記録されないのは **C1〜C4** である。C2/C3 は `IEpochProvider` を直接呼び、C1 は `IRetireSink` を直接呼ぶため、A の入口を通らない。

### 3.3 記録位置（提案 C）

| # | 位置 | 記録する内容 | 理由 |
|---|---|---|---|
| 1 | `ISRRetireRouter::enqueueRetire`（既存） | ptr, epoch, 最小 reader | 変更なし |
| 2 | `ISRRuntimePublicationCoordinator::enqueueRetire` の先頭 | ptr, epoch, 最小 reader | B 経由の要求時刻を A より前に取る |
| 3 | `SnapshotCoordinator::startFade`（C1 の直前） | ptr, epoch, 最小 reader | D 直行の要求を拾う |
| 4 | `SnapshotCoordinator::resetFadeStateAndRetireTarget`（C2 の直前） | 同上 | D 直行の要求を拾う |
| 5 | `SnapshotCoordinator::completeFade`（C3 の直前） | 同上 | D 直行の要求を拾う |
| 6 | `ISRRetireRouter::retireRT`（既存・項目 3） | 同上 | 変更なし |

C4（`quarantineRetireSink` 内の T 直行）は C1/C3 の失敗時のみ到達し、その時点で要求は記録済みである。したがって **C4 には記録を追加しない**（二重記録の回避）。`quarantineRetireSink` を呼ぶのは `startFade` と `completeFade` の 2 か所だけであり、他に C4 単独で到達する経路は無い。

### 3.4 重複の防止

- レコーダの `recordRequest` は**ポインタ単位で冪等**である。同じポインタの要求が既に未完了であれば、新しいスロットを作らず既存の ID を返す（`requests` を増やさない）。
- したがって 2 と 1 の両方を通る経路（B → A）でも二重計上しない。**最初の記録時刻が採用**され、要求時刻の意味は「最初に退役を決めた時刻」になる。
- 完了時にスロットが解放されるため、ポインタが別オブジェクトとして再利用された場合は新しい要求になる（正しい）。
- 追加するのは記録呼び出しだけで、所有権・retry・reclaim の条件式には触れない。

### 3.5 失敗時の取り扱い（幽霊 pending の防止）

| 経路 | 失敗時 | 対応 |
|---|---|---|
| B（coordinator） | `router.enqueueWithRetry` が非 Success（QueueFull／Shutdown）で所有権は呼出元に残る | `cancelRequest(ptr)` を呼ぶ（記録を取り消す） |
| C1/C3（D 直行） | 失敗時は `quarantineRetireSink` が Q→E→T で必ず受領 | 取り消し不要 |
| C4（T 直行） | `terminalReclaim` は常に受領（growable） | 取り消し不要 |
| D（retireRT） | 既に `cancelRequest` を実装済み | 変更なし |

### 3.6 期待される効果

- D1 の `unmatched` は 3 → 0 になる（C1〜C4 の要求が記録されるため）。
- `worldLifetime.unmatched = 1`（bootstrap）は本設計の対象外であり、1 のまま残る（Owner 判断どおり）。
- `requests` の総数は増える（C 経路の分）。`N_valid` と判定には影響しない（判定は完了数と最大遅延のみ）。

### 3.7 実装時に追加する検証（承認後）

1. 単体: 同一ポインタを B→A の順で 2 回記録しても `requests` が 1 であること。
2. 単体: C1〜C4 の各経路で要求と完了が 1 対 1 になること。
3. アプリ: `unmatched` が 0（D1）／1（world）であること。
4. 回帰: 既存 60 テスト。
5. 判定不変: `evaluateRetireDelay` に差分がないこと。

### 3.8 リスク

- 記録点が増えるため、将来の経路追加で漏れが再発し得る。**「退役 API を追加したら要求記録も追加する」を AGENTS.md ではなく、レコーダのコメントと本設計に残す**（AGENTS.md の更新は未承認のため）。
- C 経路の要求時刻は「スナップショットの退役決定時」であり、D キューへの格納時刻よりわずかに早い。D1 の定義（要求→解放）に合致する。

## §4 ④ retireRT（受容の記録）

- 単体: 100,000 組で requests と completions が一致（68.4 ns/組、満杯時 283.1 ns/回）。
- 実アプリ: `rtRequests = 0`（`RefCountedDeferred` に実利用が無い）。
- 両者を区別して記録し、呼出元が導入された段階で再検証する。実走例のための経路追加は行わない。

## §5 成果物と保全

- 本報告書: `doc/work113/20261110-1920_C4267と設計/report.md`
- C4267 パッチ: 同 `patch/c4267_size_guard.patch`
- 実装 4 件の差分（未変更）: `doc/work113/20261110-1836_4件実装/patches/`、`tmp/rollback/manifest.txt`
- 今回のターンで `src` は変更していない（パッチは一時ツリーでの適用確認のみ）。
