# BUG-032: createSnapshotFromCurrentState で個別 atomic 読み取り間に torn-read が発生

## 発見日
2026-07-26

## ファイル
`src/audioengine/AudioEngine.Snapshot.cpp:28-53`

## 問題
`createSnapshotFromCurrentState()` は SnapshotParams を組み立てるために
多数の atomic 変数を個別に読み取る:

```cpp
const bool convBypass = convo::consumeAtomic(convBypassRequested, ...);  // line 43
const bool eqBypass = convo::consumeAtomic(eqBypassRequested, ...);      // line 44
const bool softClip = convo::consumeAtomic(softClipEnabled, ...);        // line 45
const float satAmount = convo::consumeAtomic(saturationAmount, ...);     // line 46
... 合計 14 個の独立した atomic 読み取り
const int maxBlockSize = convo::consumeAtomic(maxSamplesPerBlock, ...);  // line 53
```

各 atomic 読み取りは正しいメモリオーダーを使用しているが、読み取りの集合全体はアトミックではない。
UI/Message スレッドがこれらの変数を更新している最中に読み取ると、各変数が異なる時点の
値が混ざった SnapshotParams が組み立てられる。

### 具体例

| 時間 | UI Thread | Timer Thread (createSnapshotFromCurrentState) |
|------|-----------|-----------------------------------------------|
| t0 | — | convBypass = false を読む |
| t1 | convBypassRequested = true | — |
| t2 | eqBypassRequested = true | — |
| t3 | softClipEnabled = true | — |
| t4 | — | eqBypass = true を読む |
| t5 | — | softClip = true を読む |
| t6 | — | satAmount = 0.5 を読む |

結果: `convBypass=false, eqBypass=true, softClip=true` という矛盾したパラメータセット。
UI は「バイパス + EQ + ソフトクリップ」を同時に設定したつもりだが、スナップショットには
convBypass=false が残る。

### 影響
- 矛盾したパラメータセットが GlobalSnapshot として公開される
- Audio Thread が一貫性のない設定を参照する可能性がある
- クリティカルな設定（convBypass）が欠落することがある

### リスク評価
- **重大度**: MEDIUM — 通常は UI 更新が Timer より遅いため問題になりにくいが、
  理論上は矛盾したスナップショットが公開される
- **発生頻度**: 低（個別の atomic 更新とスナップショット生成のタイミング依存）
- **影響範囲**: 1 tick 分のスナップショットのみ、次の tick で修正される

### 修正方針
可能な限り `std::atomic<SnapshotParams>` にまとめる。
または struct 単位の atomic 読み取りに変更して torn-read を防止する。
