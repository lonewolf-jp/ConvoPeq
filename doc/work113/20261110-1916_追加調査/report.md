# 追加調査: C4267 の安全性と unmatched の発生経路（2026-11-10）

## §0 範囲

今回の指示は「追加調査と、実装しないことの確認」である。ソース・テスト・閾値・設定の変更は行っていない。ビルド・テスト・Git 操作も行っていない（既存の実装済みバイナリと成果物の読み取りのみ）。

## §1 ① 経路分離の制約（維持の明記）

`worldLifetime` の経路は **commit 時点の文脈**で決まる。以下は現行記録の仕様であり、**bootstrap／reconfigure／deferred は `publish` に含まれる**。

| 経路 | 現行の記録 | 区別に必要なもの |
|---|---|---|
| 通常公開 | `publish` | — |
| 起動時の置換（bootstrap） | `publish`（`worldAuthority_.publish` を直接呼ぶため、後述のとおり **commit 記録なし**） | Init.cpp からの記録 |
| 再構成（reconfigure） | `publish` | 呼出元から経路を渡す API 変更 |
| deferred 再公開 | `publish` | 同上 |
| 終了処理 | `shutdown`（commit 時に shutdown 中なら） | — |
| 拒否 | `rejected` | — |

この制約は解消していない。API 変更は承認されていないため、実装しない。

## §2 ② C4267 警告の安全性

### 2.1 上限は 8 MiB（10 MB ではない）

`computeMaxElementDepth` の入力上限を決めているのは `src/InputInspection.h:23` の `kMaxXmlFileBytes = 8 * 1024 * 1024`（8 MiB）である。10 MB は `DeviceSettings::saveNoiseShaperState` の保存側ガードであり、読み込み・検査の上限ではない。

### 2.2 呼出し前の適用順序

`inspectXmlFile`（`src/InputInspection.cpp`）の順序は次のとおり。

1. `if (size > limits.maxBytes) return fail(...)`（19 行目付近）
2. `const juce::String text = file.loadFileAsString();`（23 行目付近）
3. `computeMaxElementDepth(text)`

したがって深さ走査は、**8 MiB を超えるファイルでは実行されない**。

### 2.3 全呼出元が既定の上限を使っている

`computeMaxElementDepth` を呼ぶのは `inspectXmlFile` のみで、`inspectXmlFile` は `parseXmlFile` からのみ呼ばれる（本番コード）。`parseXmlFile` の本番呼出元はすべて `limits` を省略しており、既定の 8 MiB が適用される。

| 呼出元 | 上限 |
|---|---|
| `DeviceSettings.cpp:855`（`loadNoiseShaperState`） | 既定 8 MiB |
| `DeviceSettings.cpp:918`（設定の読込） | 既定 8 MiB |
| `DeviceSettings.cpp:1003`（履歴・学習状態） | 既定 8 MiB |
| `MainWindow.cpp:1710` | 既定 8 MiB |
| `NoiseShaperLearner.cpp:1514` | 既定 8 MiB |
| `LearnedStateMigration.cpp:59` | 既定 8 MiB |

### 2.4 切り捨てが起きない理由

`n = text.getNumBytesAsUTF8()` は、`loadFileAsString()` が不正な UTF-8 バイトを U+FFFD（3 バイト）に置換する場合があるため、ファイルサイズより大きくなり得る。上限は **3 × 8 MiB = 24 MiB** で、`INT_MAX`（約 2.1 GB）を大きく下回る。したがって現行の全経路で `const int n` の切り捨ては発生しない。

### 2.5 残るリスク

`inspectXmlFile` / `parseXmlFile` は `XmlLimits` を引数で受け取る公開 API である。将来 **約 715 MB を超える `maxBytes`** を渡す呼出元が現れると、`int` の範囲を超えて破綻する。現時点でそのような呼出元は無いが、API の契約としては未保証である。

### 2.6 提案（未実装・承認待ち）

| 案 | 内容 | 評価 |
|---|---|---|
| **A（推奨）** | `const std::int64_t n64 = text.getNumBytesAsUTF8(); if (n64 > INT_MAX) return -1; const int n = static_cast<int>(n64);` | キャストの安全性を**上限に依存せず**保証する。3 行。挙動は「巨大入力は -1（構造を判定できない）」で既存の失敗経路と同じ |
| B | `const int n = static_cast<int>(text.getNumBytesAsUTF8());` のみ | 1 行で警告は消えるが、上限 8 MiB の維持が前提。将来の limits 変更で破綻する |
| C | `std::size_t n` にしてループ変数も `size_t` 化 | 根本的だが変更範囲が広く、承認済みパッチからの逸脱が大きい |

A を提案する。B は「8 MiB 上限が全経路で維持される」ことを条件に最小修正として成立するが、契約が暗黙になる。

## §3 ③ unmatched の追跡（world 1 件・D1 3 件）

### 3.1 `worldLifetime.unmatched = 1` の発生経路

**bootstrap world である。**

`AudioEngine::Init`（`src/audioengine/AudioEngine.Init.cpp`）は、初期 world を `commitRuntimePublication` ではなく **`worldAuthority_.publish(...)` を直接呼んで**公開する。

```cpp
auto* oldWorld = worldAuthority_.publish(std::move(bootstrapWorld), PublishMetadata{...}, &committed);
...
bootstrapBridge.retirePublishedRuntimeWorldNonRt(oldWorld, false);
```

- 記録されない理由: commit の記録は `enqueueRuntimePublicationFireAndForget` の owner channel enqueue 成功時にある。bootstrap は owner channel を通らないため、**commit が記録されない**。
- 破棄は記録される: 後続の `retirePublishedRuntimeWorldNonRt` が deferred delete を積み、その deleter で破棄が記録される。
- 結果: 破棄だけが記録され、`unmatched` に 1 件計上される。
- 所有権: bootstrap world は `worldAuthority_` が直接所有し、置換時に旧所有者として返る。owner channel を経由しない点が他と異なる。

発生件数が常に 1 なのは、bootstrap が起動時に 1 回だけだからである（バースト回数を変えても 1 のまま）。

### 3.2 D1 `unmatched = 3` の発生経路

**`SnapshotCoordinator` の直接経路である。**

- `AudioEngine` は `convo::SnapshotCoordinator m_coordinator` を保持している（`AudioEngine.h:5445`）。
- `SnapshotCoordinator` は `m_retireSink->terminalReclaim(...)`（`SnapshotCoordinator.cpp:45`）と `m_epochProvider->enqueueRetire(...)`（同 75／110／132。静的ヘルパ `enqueueWithRetry` 経由を含む）を直接呼ぶ。
- D1 の要求フックは `ISRRetireRouter::enqueueRetire` にあり、上記の経路は**ルーターの公開 API を通らない**。
- 一方、完了フックは `TerminalReclaimAuthority::drain` / `terminalReclaim` / D キューにあるため、**完了だけが記録される**。
- 結果: 3 件が `unmatched` に計上される（スナップショットの退役回数と一致）。
- 所有権: いずれの API も所有権を受け取る（terminal は格納、epoch provider は D キューへ）。所有権移転は正常で、記録の入口だけが欠けている。
- `rtRequests = 0` であるため、RT 経路が原因ではない（RT 経路は `RefCountedDeferred` からのみ呼ばれ、現行コードに実利用が無い）。

### 3.3 提案（未実装・承認待ち）

| 案 | 内容 | 長所 | 短所 |
|---|---|---|---|
| **A** | 要求記録を、全経路が通る `TerminalReclaimAuthority::store` / 同期 `terminalReclaim` に移す | 全経路を 1 か所で網羅 | 「要求時刻」が格納時刻になる（要求から格納までの差は µs〜ms） |
| B | `SnapshotCoordinator` の呼出しをルーター経由に変える | 要求時刻を保てる | 呼出し側の変更（所有権契約は不変） |
| C | ルーターの `enqueueRetire` に加えて、`ISRRuntimePublicationCoordinator::enqueueRetire` と `SnapshotCoordinator` の 3 箇所にも記録を足す | 要求時刻を保てる | 記録点が増え、将来の経路追加で漏れやすい |

A が最も漏れにくい。ただし意味が「格納時刻」に変わるため、D1 の定義（要求→解放）と厳密には一致しなくなる。B は意味を保てるが、変更が呼出し側に及ぶ。**どちらも承認後に実装する。**

## §4 ④ `retireRT` の未検証事項（受容の記録）

- 単体: `recordRequestRT` / `recordComplete` の 100,000 組で requests と completions が一致（68.4 ns/組、区画満杯時 283.1 ns/回）。
- 実アプリ: `rtRequests = 0`。`retireRT` は `RefCountedDeferred.h:43` からのみ呼ばれ、その利用箇所は移行済みのコメントだけである。
- したがって **RT 経路の実測は未検証**であり、単体の一致をもって実アプリの保証とはしない。呼出元が導入された段階で再検証する。

## §5 差分の保全と未解決事項

- 実装 4 件の差分は `doc/work113/20261110-1836_4件実装/patches/`（3 ファイル）と `tmp/rollback/manifest.txt`（SHA-256）で再確認できる。今回の調査では変更していない。
- 未解決のまま残す事項: ①の経路分離、②の C4267、③の unmatched 2 件、④の RT 実走。
- 実装 4 件は「完了」であるが、上記 4 件は「未解決」であり、完了扱いにしない。
