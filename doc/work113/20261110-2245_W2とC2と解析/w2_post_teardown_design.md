# W-2: `postTeardown` 段の具体設計（実装未承認）

Owner 判断（2026-11-10）で W-2 の方針が採用された。本稿は、実装前に明示を求められた 5 点（実行主体と時点、取りこぼしの排除、累積か差分か、拒否数と完了数の区別、既存判定の不変）に答える確定設計である。**実装はしない。**

## §1 現状の窓と問題

| 段 | 生成位置 | 捕捉できる範囲 |
|---|---|---|
| `runtime` | `ISRRetireRouter::drainAll()` の先頭（`markShutdownPhase` が `captureRuntimeStage` を呼ぶ） | 通常運転の終了直前 |
| `final` | `~AudioEngine` 本体の末尾（`emitFinalViolationReport`、`AudioEngine.CtorDtor.cpp:345`） | `drainAll` の後、**メンバ破棄の前** |

C++ では、`~AudioEngine` の本体が終わった後にメンバが**宣言と逆順**で破棄される。今回の 21 件は `~EQProcessor`（`eqProcessor` メンバの破棄）で起きるため、`final` では捕捉できない。これが「窓の外」の正体である。

## §2 実行主体と実行時点

### 2.1 言語保証の利用

メンバは宣言の逆順で破棄される。したがって **最初に宣言したメンバは最後に破棄される**。この性質を使い、`AudioEngine` の最初のメンバとして RAII レポータを置く。

```cpp
class AudioEngine
{
    // ★ W-2: 最初に宣言する = 最後に破棄される（~AudioEngine のメンバ破棄完了時点を捕捉）
    RetireLatencyTeardownReporter m_teardownReporter;

    // ... 既存メンバ（eqProcessor, m_retireRouter, m_coordinator, debugRuntime_ 等）
};
```

`RetireLatencyTeardownReporter` は次だけを持つ。

| 要素 | 内容 |
|---|---|
| 出力先 | `juce::String path_`（コンストラクタでは空。`AudioEngine` のコンストラクタ本体で `setOutputPath()` を呼ぶ） |
| 状態 | なし（レコーダはグローバルなので参照を保持する必要がない） |
| デストラクタ | レコーダのスナップショットを取り、`postTeardown` 段を**自前で** JSON に書く |

### 2.2 他のメンバに依存できない理由と対策

`m_teardownReporter` のデストラクタが走る時点で、他のメンバ（`debugRuntime_`、`m_retireRouter`、`eqProcessor`）は既に破棄済みである。したがって次を守る。

- レポータは `ISRDebugRuntime` を使わない。**独立した小さな JSON ライタ**を持つ（数十行）。
- 出力先は `setOutputPath()` で受け取った文字列だけを使う。
- レコーダ（関数ローカル static）はプロセス終了まで生きているので参照可能である。

### 2.3 どこまでを保証し、どこからを保証しないか

| 期間 | 破棄されるもの | 捕捉する段 | 保証 |
|---|---|---|---|
| `~AudioEngine` 本体 | `releaseResources` などの明示処理 | `final`（既存） | する |
| `AudioEngine` のメンバ（逆順） | `eqProcessor`（EQState 21 件）、`m_retireRouter`、`m_coordinator` など | **`postTeardown`（新設）** | する（言語保証） |
| `AudioEngine` の基底クラス（6 個） | `juce::AudioSource` / `ChangeBroadcaster` / `ChangeListener` / `ConvolverProcessor::Listener` / `juce::Timer` / `juce::AsyncUpdater` | `postProcess` の差分で検出 | 基底は退役対象を所有しない（`postProcess` が 0 差分であることを実測で確認する） |
| その後のプロセス終了処理 | `MainWindow` の残り、JUCE、静的オブジェクト | `postProcess`（差分があるときだけ記録） | 差分として検出 |

**前提の訂正**: 当初「基底クラスなし」と書いたが誤りである。`AudioEngine` は 6 つの基底クラスを持つ。メンバは基底より先に破棄されるため、`m_teardownReporter`（最初の**データメンバ**）はメンバ破棄の最後に走るが、基底クラスの破棄よりは前である。基底クラスは退役対象を所有しない（JUCE のインタフェースとタイマー）。この前提は `postProcess` の差分 0 で実測的に確認する。

### 2.4 `postProcess`（取りこぼしの検出）

プロセス終了時点の捕捉は、レコーダより後に生成される静的ガードで行う。

```cpp
struct PostProcessReporter
{
    ~PostProcessReporter() noexcept
    {
        const auto s = RetireLatencyRecorder::instance().snapshot();
        if (s.requests != baseline.requests || s.completions != baseline.completions)
            writeStageFile(path, "postProcess", s);   // 差分があるときだけ書く
    }
    Snapshot baseline;   // postTeardown 直後の値
};
```

静的オブジェクトの破棄順は生成の逆順である。レコーダは最初の `recordRequest` で生成され、このガードはそれより後に生成されるため、**ガードの方が先に破棄される**（＝レコーダはまだ生きている）。この順序に依存するため、ガードは「レコーダの `instance()` を呼んだ直後に生成する」ことをコードで保証する。

`postProcess` は通常 0 差分で終わる。差分が出た場合は「`postTeardown` の後にまだ退役があった」ことを意味するため、**観測の取りこぼしを検出する安全網**として機能する。

## §3 取りこぼしの排除（設計上の根拠）

1. レコーダのカウンタは単調増加であり、後の捕捉は必ず前の捕捉を含む（累積）。したがって「捕捉時点が遅いほど取りこぼしは少ない」。
2. 終了処理は単一スレッドである（Audio Thread 停止済み、Coordinator Loop は join 済み）。`postTeardown` の時点でエンジン内の並行作業は無い。
3. `postProcess` が差分 0 を示せば、`postTeardown` の後に退役が無かったことの**実測的な証明**になる。
4. 記録**できなかった**要求（shutdown 相）は、窓を後ろへ動かしても記録されない。これは窓の問題ではなく記録の問題であるため、§4 のとおり拒否数として別掲する。

## §4 累積 / 差分の定義と表示

### 4.1 各段は累積

| 段 | 意味 | 型 |
|---|---|---|
| `runtime` | `markShutdownPhase` 時点の累積 | 累積 |
| `final` | `~AudioEngine` 本体末尾の累積 | 累積 |
| `postTeardown` | メンバ破棄完了時点の累積 | 累積 |
| `postProcess` | プロセス終了時点の累積（差分があるときのみ出力） | 累積 |

### 4.2 差分は明示フィールドで併記

読み手が引き算をしなくて済むよう、差分も書く。

```json
"retireLatencyD1": {
  "thresholdNs": 50000000,
  "required": 59,
  "runtime":      { "capturePoint": "markShutdownPhase (drainAll の先頭・drain 前)", "cumulative": true, "...": "..." },
  "final":        { "capturePoint": "emitFinalViolationReport (~AudioEngine 本体の末尾)", "cumulative": true, "...": "..." },
  "postTeardown": { "capturePoint": "~AudioEngine のメンバ破棄完了 (最初に宣言したメンバのデストラクタ)", "cumulative": true, "...": "..." },
  "postProcess":  { "capturePoint": "プロセス終了時 (静的ガード)", "cumulative": true, "written": false },
  "delta": {
    "finalMinusRuntime":        { "requests": 0, "completions": 0, "unmatched": 0, "recordMisses": 0 },
    "postTeardownMinusFinal":   { "requests": 1, "completions": 1, "unmatched": 21, "recordMisses": 0 },
    "postProcessMinusTeardown": { "requests": 0, "completions": 0, "unmatched": 0, "recordMisses": 0 }
  }
}
```

`delta` の値は例である（実装時に実測で埋まる）。

## §5 shutdown 相の「拒否数」と「その後の完了数」を分けて表示

### 5.1 用語の修正（Owner 指摘）

`recordMisses` は**記録に失敗した呼び出し回数**である。オブジェクト数ではない。今回の 21 対象は B と A2 の 2 回試行するため 42 回になる。レポートでもこの表現に統一する。

### 5.2 追加する計数

| 計数 | 意味 | 現状 |
|---|---|---|
| `recordMisses` | 記録に失敗した**呼び出し回数**（shutdown 相・ID 枯渇・容量不足の合計） | 既存 |
| `postShutdownRequests` | そのうち **shutdown 相が理由**の回数 | 既存 |
| `postShutdownCompletions` | `markShutdownPhase` 以降に観測した**完了**の件数 | **新設** |
| `unmatchedCompletions` | 対応する要求が無い完了 | 既存 |

表示は次の 1 行を追加する。

```
shutdown 相: 記録拒否 <postShutdownRequests> 回 / 記録失敗 <recordMisses> 回 / その後の完了 <postShutdownCompletions> 件 / 未対応 <unmatchedCompletions> 件
```

今回の実行に当てはめると `42 / 42 / 21 / 21` になる（21 対象 × 2 試行、完了 21 件）。

## §6 既存判定を変えないこと

- `evaluateRetireDelay` は **`runtime` 段のみ**を読む（現行どおり）。`postTeardown` / `postProcess` / `delta` は判定に渡さない。
- `retire_delay` の `observations` / `maxLatencyNs` も現行どおり `runtime` 由来とする。
- したがって W-2 の追加で verdict の値は変わらない。実装時の検証項目に「`evaluateRetireDelay` のバイト一致」と「同一シナリオで verdict が変わらないこと」を含める。

## §7 実装対象（承認後）

| ファイル | 変更 |
|---|---|
| `src/audioengine/AudioEngine.h` | `RetireLatencyTeardownReporter` を最初のメンバとして追加。`setOutputPath` をコンストラクタ本体で呼ぶ |
| 新規 `RetireLatencyTeardownReporter.h/.cpp` | 独立 JSON ライタ（数十行）。`postTeardown` を書く |
| `src/audioengine/ISRRetireLatency.h` | `postShutdownCompletions_` の追加、`Snapshot`/`StageStats` への反映 |
| `src/audioengine/ISRHB.cpp` | 3 段＋差分の出力。`postProcess` は別ファイルまたは同ファイルの追加節 |
| 静的ガード（新規 1 ファイル） | `postProcess` の捕捉（差分があるときだけ書く） |

## §8 検証計画

| # | 検証 | 期待 |
|---|---|---|
| 1 | `postTeardown - final` の `unmatched` | 21（EQState 21 件） |
| 2 | `postTeardown - final` の `completions` | 21 |
| 3 | `postShutdownRequests` / `recordMisses` | 42 / 42（21 対象 × 2 試行） |
| 4 | `postShutdownCompletions` | 21 |
| 5 | `postProcess - postTeardown` | 0（取りこぼしなし） |
| 6 | 3 段の累積値 | 単調増加（requests / completions） |
| 7 | `evaluateRetireDelay` | バイト一致・verdict 不変 |
| 8 | 既存 60 テスト | 合格 |

## §9 未確定事項（Owner 判断を仰ぐ）

1. `postProcess` を常時出すか、差分があるときだけ出すか。設計案は「差分があるときだけ」（通常の出力を増やさない）。
2. `postTeardown` を既存の `hb_violation_report.json` に含めるか、別ファイルにするか。設計案は同一ファイル（1 か所で読める）。
3. 静的ガードの生成位置（レコーダ生成直後）をコードで強制できるか。強制できない場合は「生成順序に依存する」旨をコメントで明示する。
