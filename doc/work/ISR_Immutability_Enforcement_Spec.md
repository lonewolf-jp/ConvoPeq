# ConvoPeq Publish Immutability Enforcement Spec

## 目的

`publish immutability discipline` を人間運用に依存させず、**検証可能な enforcement** として定義する。

用語正規化（齟齬回避）:

- 本書では `RuntimePublication` を正規記法として扱う。

## Enforcement モデル（確定）

### E1. Builder Freeze

- publish対象 object は Builder で構築
- `freeze()` 後は mutating API を無効化
- freeze されていない object の publish を禁止

### E2. Immutable Facade

- observe 側へ渡す型は read-only facade
- mutable メソッドは facade から不可視

### E3. Publish Seal（Debug/Release共通フック）

- publish 時に seal bit を設定
- seal 後の mutating call は `assert + diagnostic log`（Debug）
- Release では no-op ではなく failure counter を増分

### E4. Post-publish Mutation Audit

- Atomic scan + 静的ルールで publish object への write を検出
- CI で fail するルールとして運用

## 適用対象

- RuntimePublication payload
- EngineRuntime publish fields
- RuntimeGraph publish fields
- DSPConfig publish payload

## 非適用

- RTLocalState
- DSPExecutionInstance
- telemetry counters

## 判定ゲート

- [ ] freeze 呼び出し経路が全 publish path で実装済み
- [ ] facade 経由以外の observer path が存在しない
- [ ] seal 違反時の診断が有効
- [ ] post-publish write 検出ルールがCIで有効

---

## R1 Closed 最小検証（必須）

- [ ] Test-1: freeze 未実行 payload publish が拒否される
- [ ] Test-2: publish 後 write が seal violation として検出される（Debug/Release）
- [ ] Test-3: static rule + atomic scan のいずれかが publish object write を検出した場合、CI が fail する

証跡:

- テスト結果ログ（Test-1〜3）
- CI fail/pass 実行記録
