# W-2 設計・C2 改訂設計・`retire_delay` 経路別分析（2026-11-10）

## §0 結論

1. **W-2 の具体的な終了処理ライフサイクル設計を提出した**（`w2_post_teardown_design.md`、実装なし）。`~AudioEngine` の**最初に宣言したメンバ**を最後の破棄者として使い、メンバ破棄完了時点を `postTeardown` として捕捉する。言語保証（宣言の逆順破棄）に依拠する。その後の取りこぼしは `postProcess`（静的ガード）の差分 0 で検出する。
2. **C2 の改訂設計を提出した**（`c2_nonrt_design_v3.md`、実装なし）。RT 呼出しは「`jassert` + カウンタ + **状態を一切変更せず return**」とし、target はスロットに残す。呼出元は実作業ツリーで再確認し、本番 0・テスト 2・ハーネス 1 を確認した。
3. **`retire_delay` の経路別分析を行った**（`retire_delay_analysis.md`、修正なし）。最大待ち時間は **EBR 待ちが 100%**、実行待ちは 1 ms 未満である。停滞は「`epoch=1` の一群が `minReaderEpoch=1` のまま待つ」形で、対象ドメインの `globalEpoch` が 2 で凍結していた。reader は無名登録のため帰属を特定できていない。
4. **用語を修正した**: `recordMisses` は「記録に失敗した**呼び出し回数**」であり、オブジェクト数ではない。今回の 21 対象は 2 回ずつ試行され 42 回になる。
5. probe を除去した状態で全再ビルドし、ctest とアプリ実行で確認した（§5）。

## §1 W-2（`postTeardown`）の設計

詳細は `w2_post_teardown_design.md`。Owner が求めた 5 点への回答は次のとおり。

| 求められた点 | 設計回答 |
|---|---|
| 実行主体と実行時点 | `AudioEngine` の**最初のメンバ** `RetireLatencyTeardownReporter`。宣言の逆順破棄により、`eqProcessor` などすべてのメンバの破棄後に走る |
| どの所有者が保証するか | C++ の言語保証（メンバは宣言の逆順で破棄、本体の後）。`AudioEngine` に基底クラスが無いことを前提条件として明記 |
| 取りこぼしの排除 | (a) カウンタは単調で後の捕捉は前を含む、(b) 終了処理は単一スレッド、(c) `postProcess` の差分 0 を実測の証明にする |
| 累積か差分か | 3 段（`runtime` / `final` / `postTeardown`）はすべて**累積**。差分は `delta` フィールドに明示して併記 |
| 拒否数と完了数の区別 | `recordMisses`（失敗した呼出し回数）、`postShutdownRequests`（shutdown 相が理由）、**新設 `postShutdownCompletions`**（shutdown 相以降の完了）を分けて表示 |
| 既存判定の不変 | `evaluateRetireDelay` は `runtime` 段のみを読む。`postTeardown` / `postProcess` / `delta` は判定に渡さない |

補足: `m_teardownReporter` の破棄時点では他のメンバは既に無いため、レポータは `ISRDebugRuntime` を使わず、独立した小さな JSON ライタで自前の段を書く。

## §2 C2（Non-RT 専用）の改訂設計

詳細は `c2_nonrt_design_v3.md`。前版からの変更点は次のとおり。

```cpp
if (convo::numeric_policy::isAudioThread())
{
    jassert (false && "resetFadeStateAndRetireTarget is Non-RT only");
    convo::fetchAddAtomic (m_rtViolationCount, std::uint64_t{1}, std::memory_order_release);
    return;                       // 状態を一切変更しない（target はスロットに残す）
}
```

- RT 経路では `exchangeTarget`・記録・移送・`resetToIdle` を行わない。
- 後続の Non-RT 呼出しが同じ前提で通常どおり退役させる。
- 呼出元の棚卸しは実作業ツリーで再確認（本番 0、テスト 2 ファイル、ハーネス 1）。
- 検証計画は 9 件（成功／Q／Q 満杯→E／Q/E 満杯→T／二重移送なし／記録 1 回／**RT 違反**／**RT 違反後の正常退役**／回帰）。
- T の `std::vector` 確保失敗は `std::terminate` になる既存契約であり、**Owner 判断の対象**として残した。

## §3 `retire_delay` の経路別分析

詳細は `retire_delay_analysis.md`。要点だけを示す。

### 3.1 時間成分の分解

| 実行（bypass 40 × 250 ms） | 最大 latency | EBR 待ち | 実行待ちの最大 |
|---|---|---|---|
| run1 | 4,408.2 ms | 4,408.2 ms（100%） | 0.777 ms |
| run2 | 4,415.6 ms | 4,415.6 ms（100%） | 0.740 ms |
| run3 | 4,388.0 ms | 4,388.0 ms（100%） | 0.501 ms |

**実行待ちは 1 ms 未満**であり、reclaim 処理はボトルネックではない。遅延はすべて EBR 待ちである。

### 3.2 個々の request の署名

最大待ちの要求はすべて同じ形である。

| 項目 | 値 |
|---|---|
| path | `queue` |
| 要求時 `epoch` / `minReaderEpoch` / `minReaderSlot` | `1` / `1` / `0` |
| ebr / exec | 4,408 ms / 0.000 ms |
| 完了経路 | `DeferredDeletionQueue::reclaim`（`Path::Queue`） |

60〜800 ms の遅延も同じ署名で、**同一原因の大小**である。

### 3.3 停滞中の reader 状態

- 対象ドメインの `globalEpoch` は **2 のまま 8.5 秒間**動かなかった。
- `minReaderEpoch` は 1 のまま、滞留 20 件が一定。
- 最小 epoch の reader は **slot 0**、depth 1、区画の滞留は 0.1〜0.2 ms（短い区画を繰り返す）。
- `ownerTag` は **"unnamed"**。本番の登録 3 か所（`RCUReader`／`ScopedEpochReader`／`ISRRetireRouter`）はいずれもタグを渡していない。

### 3.4 原因の候補（確定は次段階）

| 仮説 | 内容 |
|---|---|
| H-1 | 退役先ドメインが epoch を公開しないため `minReaderEpoch` が進まない（shutdown の `drainAll` まで待つ） |
| H-2 | reader の入場経路の一部（`reserveReaderThread` / `reacquireReaderSlot`）が slot の epoch を更新しない |

いずれかを確定するには、ドメイン別の `publishEpoch` 回数と経路別の入場回数の計測が必要である（§9 に提案）。

### 3.5 ばらつき（4.4〜13.9 秒）

実行時間は同じ 16 秒で、最大値だけが変わる。待ち時間は「退役した時刻から `minReaderEpoch` が進む時刻まで」なので、シナリオによる相対位置の違いがそのまま出る。負荷やシャットダウン処理への依存は確認できなかった（`drainAll` の完了は `Path::DrainAll` で別集計され、最大値の対象外）。

## §4 用語の修正（Owner 指摘）

| 用語 | 正しい意味 | 今回の値 |
|---|---|---|
| `recordMisses` | 記録に**失敗した呼び出し回数**（オブジェクト数ではない） | 通常運転 0。shutdown 相では 42（21 対象 × 2 試行） |
| `postShutdownRequests` | そのうち shutdown 相が理由の回数 | 同上 |
| `postShutdownCompletions` | shutdown 相以降に観測した完了件数（W-2 で新設） | 21 |
| `unmatchedCompletions` | 対応する要求が無い完了 | 0（C5 適用後） |

以降のレポートはこの表現に統一する。

## §5 検証状態

### 5.1 ctest

```
100% tests passed out of 60
Total Test time (real) = 358.59 sec
ctest_exit=0
```

probe をすべて除去した状態で全 253 ターゲットを再ビルドしたうえでの結果である。`AudioEngineHarness` を含む 60 件が合格した。

### 5.2 アプリ実行（probe 除去後）

| シナリオ | requests | completions | unmatched | pending | recordMisses | 最大 latency |
|---|---|---|---|---|---|---|
| bypass 40 × 250 ms | 1,849 | 1,849 | **0** | 0 | 0 | 4,407.1 ms |
| bypass 60 × 2 ms | 545 | 545 | **0** | 0 | 0 | 11,795.0 ms |
| EQ 変更 | 187 | 187 | **0** | 0 | 0 | 13,932.2 ms |

world は 3 シナリオとも `commits = destroys`、`unmatched = 1`（bootstrap 由来の既知値）。`sync=1` と `drain=1` は C5 の 2 件に対応する。

### 5.3 `evaluateRetireDelay` の不変

| 項目 | 値 |
|---|---|
| バイト数 | 2,756（実装前後で一致） |
| SHA-256 | `20346AD1C2B6597EDE74FAFF90201F42…`（一致） |

## §6 変更ファイルとロールバック

今回の作業で**製品コードの変更はない**。調査用プローブは 2 組入れて除去した。

| プローブ | 対象 | 復元確認 |
|---|---|---|
| 21 件調査（第 1 組） | 11 ファイル（`AudioEngine.h`、`ISRRetireRouter.cpp`、`ISRRuntimePublicationCoordinator.cpp`、`SnapshotCoordinator.*`、`AudioEngine.CtorDtor.cpp`、`DSPLifetimeManager.cpp`、`EQProcessor.Core.cpp`、`AudioEngine.Retire.cpp`、`AudioEngine.Commit.cpp`、`ISRRetireLatency.h`） | 11/11 SHA-256 一致 |
| `retire_delay` 調査（第 2 組） | 3 ファイル（`EpochDomain.h`、`DeferredDeletionQueue.h`、`ISRRetireLatency.h`） | 3/3 SHA-256 一致 |

成果物は文書のみである。commit／merge／release は行っていない。

## §7 未達・残課題

1. **W-2 の実装**は未承認（設計のみ）。
2. **C2 の実装**は未承認（設計のみ）。
3. `retire_delay` の原因は H-1／H-2 のいずれかに絞り込めていない。追加計測（ドメイン別 publish 回数、経路別入場回数、reader タグ付け）が必要。
4. `postTeardown` / `postProcess` の実装は、W-2 の承認後に着手する。
