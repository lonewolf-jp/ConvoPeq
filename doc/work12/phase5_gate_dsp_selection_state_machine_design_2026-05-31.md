# Phase5-Gate: DSP Selection State Machine 設計承認

作成日: 2026-05-31
対象: `src/audioengine/**` の `transition.active` 依存置換

## 1. 目的

`transition.active` を実行分岐の唯一トリガーに使う現行方式を廃止し、
状態機械（Stable / Entering / Retiring）で DSP 選択・クロスフェード・retire 判定を統治する。

## 2. 状態図

```mermaid
graph LR
  Stable --> Entering: new runtime admitted
  Entering --> Retiring: crossfade armed
  Retiring --> Stable: grace complete + retire ack
  Entering --> Stable: hard reset path
```

状態定義:

- `Stable`: active runtime のみ有効。fading/runtime-next は null。
- `Entering`: publish 済み・クロスフェード開始前後の遷移準備。
- `Retiring`: old runtime が grace 待ち。retire authority 管理下。

## 3. AudioThread 分岐置換表

| 旧分岐 | 新分岐 | 判定ソース | 備考 |
| --- | --- | --- | --- |
| `transition.active == false` | `state == Stable` | RuntimeWorld.execution/scheduling | activeのみ実行 |
| `transition.active == true && next != nullptr` | `state == Entering` | RuntimeWorld.execution + overlap | crossfade準備・遅延整合 |
| `transition.active == true && fading pending` | `state == Retiring` | Retire queue/intents | old runtime を grace 待ち |

## 4. 互換期間フェイルセーフ

- 互換期間は `TransitionState` を **観測用途のみ**許容（分岐不可）。
- 互換fail-safe:
  1. state 不整合時は `HardReset` へフォールバック
  2. `Retiring` 長時間停滞時は retire escalation (`quarantine -> reclaim`) を強制
  3. publication precheck 不合格時は `Rejected` 終端へ遷移

## 5. 実装契約

1. `transition.active` を条件式で参照する実行分岐を禁止。
2. state 機械は RuntimeWorld semantic (`execution/scheduling/overlap`) を正とする。
3. AudioThread は state を acquire 読み取りのみで処理し、非RT authority を持たない。
4. retire 完了判定は Retire authority 側（Phase7設計）で一元化する。

## 6. 設計承認記録

- 承認フェーズ: `Phase5-Gate`
- 承認判定: **Approved**
- 条件:
  - 状態図あり
  - 置換表あり
  - 互換フェイルセーフあり
- 次アクション: Phase5 実装で `transition.active` 実行依存ゼロ化を実施
