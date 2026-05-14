# AudioEngine lint ルール候補（single-world 再発防止）

## 目的

- Runtime publish 読取を単一路に固定する。
- CrossfadePreparedState の読取一本化を維持する。
- 退行を機械検出できる最小ルールを導入する。

## 機械チェック結果（現時点）

- src/audioengine 配下で、single-world 方針違反の主要パターンは検出なし。
- 例外として、以下は設計上の許容箇所（writer/helper 側）。
  - AudioEngine.h の makeEngineRuntimeState 内の crossfade atomic load
  - AudioEngine.h の resolve helper 内 fallback（engineRuntime 未指定時）

## ルール候補

### LINT-AE-001

- ルール: src/audioengine/*.cpp で getRuntimeGraphState() / getEngineRuntimeState() の引数なし呼び出しを禁止。
- 意図: world 固定なしの読取を防止。
- 推奨パターン:
  - const auto* runtimeWorld = getRuntimePublishWorld();
  - const auto* runtimeGraph = getRuntimeGraphState(runtimeWorld);
  - const auto* engineRuntime = getEngineRuntimeState(runtimeWorld);
- 機械検出（PowerShell）:
  - rg -n "getRuntimeGraphState\(\)|getEngineRuntimeState\(\)" src/audioengine --glob "*.cpp"

### LINT-AE-002

- ルール: resolveCurrentDSPFromRuntimePublish / resolveFadingDSPFromRuntimePublish の単引数呼び出しを禁止。
- 意図: resolve helper 内 fallback による別 world 再読取を防止。
- 推奨パターン:
  - resolveCurrentDSPFromRuntimePublish(runtimeGraph, engineRuntime)
  - resolveFadingDSPFromRuntimePublish(runtimeGraph, engineRuntime)
- 機械検出（PowerShell）:
  - rg -n "resolveCurrentDSPFromRuntimePublish\([^,\n\)]*\)|resolveFadingDSPFromRuntimePublish\([^,\n\)]*\)" src/audioengine --glob "*.cpp"

### LINT-AE-003

- ルール: src/audioengine/*.cpp で dspCrossfadePending.load / dspCrossfadeUseDryAsOld.load / dspCrossfadeStartDelayBlocks.load を禁止。
- 意図: CrossfadePreparedState 読取の一本化を維持。
- 推奨パターン:
  - runtimeCrossfadePending(engineRuntime, runtimeGraph)
  - runtimeCrossfadeUseDryAsOld(engineRuntime, runtimeGraph)
  - runtimeCrossfadeStartDelayBlocks(engineRuntime, runtimeGraph)
- 機械検出（PowerShell）:
  - rg -n "dspCrossfadePending\.load\(|dspCrossfadeUseDryAsOld\.load\(|dspCrossfadeStartDelayBlocks\.load\(" src/audioengine --glob "*.cpp"

### LINT-AE-004（推奨・警告レベル）

- ルール: 同一関数内で getRuntimePublishWorld() を 3 回以上呼ぶ場合は警告。
- 意図: 同一処理ブロックで世界が切り替わる余地を減らす。
- 注意: 既存コードの一部は意図的に複数回取得するため、まず警告運用が安全。
- 機械検出（PowerShell）:
  - rg -n "getRuntimePublishWorld\(\)" src/audioengine --glob "*.cpp"

## 運用案

- まず LINT-AE-001/002/003 を CI の fail 条件に設定。
- LINT-AE-004 は warn で運用開始し、誤検出を観察してから fail 化を検討。
- 例外許容は最小化し、必要な場合はレビューコメントで根拠を明記する。

## 参考（実装済みの準拠例）

- src/audioengine/AudioEngine.Processing.AudioBlock.cpp
- src/audioengine/AudioEngine.Processing.BlockDouble.cpp
- src/audioengine/AudioEngine.Timer.cpp
- src/audioengine/AudioEngine.Commit.cpp
- src/audioengine/AudioEngine.Processing.Snapshot.cpp
- src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp
