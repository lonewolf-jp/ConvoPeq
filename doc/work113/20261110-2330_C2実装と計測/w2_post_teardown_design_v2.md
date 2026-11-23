# W-2 改訂設計 v2: 基底クラス監査にもとづく `postTeardown` の確定（実装未承認）

Owner 判断（2026-11-10）を受け、案 A（基底クラスの破棄経路のソース監査）を採用した改訂版である。**実装は行っていない。**

## §1 結論

1. `AudioEngine` の 6 つの基底クラスのデストラクタは、**退役・完了を一切発生させない**ことをソースで確認した（§2）。したがって「最初に宣言したデータメンバ」を最終捕捉点とする案は成立する。
2. したがって `postTeardown` は **`AudioEngine` が所有するリソースの破棄完了時点の最終捕捉**として定義する。
3. `postProcess`（静的ガード）は**保証ではなく追加の観測値**に格下げする。Owner 指摘のとおり、静的オブジェクトの破棄順は「構築の逆順」であり、レポータより先に構築された静的オブジェクトはレポータより後に破棄される。その時点の退役は捕捉できない。
4. 3 段の累積値・差分の併記、`recordMisses` / `postShutdownRequests` / `postShutdownCompletions` の区別、`evaluateRetireDelay` が `runtime` 段のみを使う点は維持する。

## §2 基底クラスの破棄経路の監査（案 A）

`AudioEngine` の基底クラスは 6 つである（`src/audioengine/AudioEngine.h:600-605`）。

| # | 基底クラス | デストラクタの実装 | 退役・完了を発生させるか |
|---|---|---|---|
| 1 | `juce::AudioSource` | `virtual ~AudioSource() = default;`（`JUCE/modules/juce_audio_basics/sources/juce_AudioSource.h:130`） | **しない**（空） |
| 2 | `juce::ChangeBroadcaster` | `ChangeBroadcaster::~ChangeBroadcaster() {}`（`JUCE/modules/juce_events/broadcasters/juce_ChangeBroadcaster.cpp:43-45`） | **しない**（空） |
| 3 | `juce::ChangeListener` | `virtual ~ChangeListener() = default;`（`JUCE/modules/juce_events/broadcasters/juce_ChangeListener.h:60`） | **しない**（純粋インタフェース） |
| 4 | `ConvolverProcessor::Listener` | `virtual ~Listener() = default;`（`src/ConvolverProcessor.h:207`） | **しない**（純粋インタフェース） |
| 5 | `juce::Timer` | `jassert(!isTimerRunning() ...); stopTimer();`（`JUCE/modules/juce_events/timers/juce_Timer.cpp:359-368`） | **しない**（タイマー解除のみ。`timerCallback()` は呼ばれない） |
| 6 | `juce::AsyncUpdater` | `activeMessage->shouldDeliver.set(0);`（`JUCE/modules/juce_events/broadcasters/juce_AsyncUpdater.cpp:61-71`） | **しない**（保留中のコールバックを破棄する。`handleAsyncUpdate()` は呼ばれない） |

### 2.1 監査の限界（正直な記載）

- 5 と 6 は**保留中のコールバックを実行しない**ことを確認した。ただし `stopTimer()` と `cancelPendingUpdate()` はメッセージマネージャのロックを取る可能性があり、**ブロックはし得る**（退役は発生しない）。
- 監査は「基底クラスのデストラクタ本体」に限定した。基底クラスが保持するメンバ（`juce::ListenerList` など）の破棄は、登録済みリスナーの**ポインタを捨てるだけ**で、リスナー側のコードを呼ばない。
- したがって「基底の破棄で退役は起きない」は、上記 6 つの実装に依存する。JUCE を更新した場合は再監査が必要である。

## §3 3 段の定義（改訂）

| 段 | 捕捉点 | 位置づけ | 保証 |
|---|---|---|---|
| `runtime` | `markShutdownPhase`（`drainAll` の先頭・drain 前） | 判定用（既存） | 既存のまま |
| `final` | `emitFinalViolationReport`（`~AudioEngine` 本体の末尾） | 判定用（既存） | 既存のまま |
| `postTeardown` | **最初に宣言したデータメンバ**のデストラクタ（メンバ破棄完了・基底破棄の直前） | **最終捕捉**（本設計） | 言語保証＋§2 の監査 |
| `postProcess` | プロセス終了時の静的ガード | **追加の観測値**（保証ではない） | なし（差分が出れば「捕捉外の退役があった」ことの検出のみ） |

`postProcess` の意味を Owner 指摘どおりに限定する。「差分 0 ならプロセス終了まで取りこぼしが無い」とは**主張しない**。主張できるのは「`PostProcessReporter` の破棄時点までに観測できた差分が 0 だった」までである。

## §4 実行主体とライフサイクル（実装案）

```cpp
class AudioEngine : public juce::AudioSource, /* ... 他 5 基底 ... */
{
    // ★ W-2: 最初に宣言するデータメンバ = メンバ破棄の最後に走る
    RetireLatencyTeardownReporter m_teardownReporter;

    // ... 既存メンバ ...
};
```

| 時点 | 起きること | 捕捉 |
|---|---|---|
| `~AudioEngine` 本体 | `releaseResources` などの明示処理 | `final`（既存） |
| メンバ破棄（逆順） | `eqProcessor`（EQState 21 件）、`m_retireRouter`、`m_coordinator` など | **`postTeardown`** |
| `m_teardownReporter` の破棄 | レコーダのスナップショットを自前 JSON に書く | `postTeardown` の出力 |
| 基底 6 つの破棄 | §2 のとおり退役なし | （捕捉不要） |
| その後の静的破棄 | 不明（順序に保証なし） | `postProcess`（観測のみ） |

レポータは破棄時点で他メンバが無いため、`ISRDebugRuntime` を使わず独立した小さな JSON ライタで書く。出力先は `AudioEngine` のコンストラクタ本体で `setOutputPath()` により渡す。

## §5 表示（累積・差分・拒否数と完了数）

```json
"retireLatencyD1": {
  "thresholdNs": 50000000,
  "required": 59,
  "runtime":      { "capturePoint": "markShutdownPhase", "cumulative": true, "...": "..." },
  "final":        { "capturePoint": "~AudioEngine 本体末尾", "cumulative": true, "...": "..." },
  "postTeardown": { "capturePoint": "最初に宣言したデータメンバのデストラクタ", "cumulative": true, "...": "..." },
  "postProcess":  { "capturePoint": "プロセス終了時（静的ガード・観測のみ）", "cumulative": true, "written": false },
  "delta": {
    "finalMinusRuntime":        { "requests": 0, "completions": 0, "unmatched": 0 },
    "postTeardownMinusFinal":   { "requests": 1, "completions": 1, "unmatched": 21 },
    "postProcessMinusTeardown": { "requests": 0, "completions": 0, "unmatched": 0 }
  }
}
```

`delta` の値は例である。`postTeardownMinusFinal` の `unmatched: 21` が、今回特定した `~EQProcessor` の 21 件に対応する。

| 計数 | 意味 | 今回の実測 |
|---|---|---|
| `recordMisses` | 記録に失敗した**呼び出し回数** | 通常 0、shutdown 相で 42 |
| `postShutdownRequests` | そのうち shutdown 相が理由の回数 | 42 |
| `postShutdownCompletions`（新設） | shutdown 相以降に観測した完了件数 | 21 |
| `unmatchedCompletions` | 対応する要求が無い完了 | 0（C5 適用後、レポート時点） |

## §6 既存判定の不変

- `evaluateRetireDelay` は `runtime` 段のみを読む。`postTeardown` / `postProcess` / `delta` は判定に渡さない。
- 実装時の検証項目に「`evaluateRetireDelay` のバイト一致」と「同一シナリオで verdict が変わらないこと」を含める。

## §7 検証計画（承認後）

| # | 検証 | 期待 |
|---|---|---|
| 1 | `postTeardown - final` の `unmatched` | 21（EQState 21 件） |
| 2 | `postTeardown - final` の `completions` | 21 |
| 3 | `postShutdownRequests` / `recordMisses` | 42 / 42（21 対象 × 2 試行） |
| 4 | `postShutdownCompletions` | 21 |
| 5 | `postProcess - postTeardown` | 0（ただし保証ではなく観測値） |
| 6 | 3 段の累積が単調増加 | はい |
| 7 | `evaluateRetireDelay` | バイト一致・verdict 不変 |
| 8 | 既存 60 テスト | 合格 |

## §8 実装対象（承認後）

| ファイル | 変更 |
|---|---|
| `src/audioengine/AudioEngine.h` | `RetireLatencyTeardownReporter` を最初のデータメンバとして追加、ctor 本体で出力先を設定 |
| 新規 `RetireLatencyTeardownReporter.h/.cpp` | 独立 JSON ライタ（`postTeardown` を書く） |
| `src/audioengine/ISRRetireLatency.h` | `postShutdownCompletions_` の追加 |
| `src/audioengine/ISRHB.cpp` | 3 段＋差分＋新設計数の出力（`postProcess` は観測のみ） |
| 新規 1 ファイル | `postProcess` の静的ガード（差分があるときだけ書く） |

## §9 未確定事項

1. `postProcess` を常時出力するか、差分があるときだけ出力するか（設計案は後者）。
2. `postTeardown` を既存の `hb_violation_report.json` に含めるか別ファイルにするか（設計案は同一ファイル）。
3. 静的ガードの生成順序（レコーダ生成直後）をコードで強制できない場合の注記の置き方。
