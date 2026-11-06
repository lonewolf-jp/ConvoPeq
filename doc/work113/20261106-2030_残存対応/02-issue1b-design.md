# #1 残存 B — 無効バンドのキー正規化（設計案）

- 基準: `main` = `f7dcdf7a`
- 位置づけ: 設計検討のみ。コード変更・テスト追加は別途承認とする。
- 優先度: #9 より低い。

## 1. 現状

`EQProcessor::makeCacheKey`（`EQProcessor.h:872-912`）は、バンドの有効・無効にかかわらず周波数・ゲイン・Q をクランプしてビット化し、キーへ入れる。そのため、無効バンドの値だけが違う 2 つの設定は別キーになり、別のキャッシュが作られる。

係数生成側（`EQProcessor.ProcessingCache.cpp:33-52`）は、無効バンドに既定の `EQCoeffsSVF()`（バイパス）を入れ、`bandActive[i]=false` にする。処理側は `bandActive` を先に確認するため、無効バンドの係数と `channelModes` は読まれない（`Processing.cpp:1140`、`:1181`、`:1233`。`:1039` も短絡評価で同じ）。

したがって、無効バンドの周波数・ゲイン・Q は係数にも音声出力にも影響しない。影響するのはキャッシュの本数と生成回数だけである。

## 2. 提案

キー生成時、無効バンドの `frequencyBits`・`gainBits`・`qBits` を 0 に固定する。`enabled`・`type`・`channelMode` は現状どおり保持する。

```cpp
// 現行（抜粋）
key.frequencyBits[i] = canonicalFloatBits(frequency);
key.gainBits[i]      = canonicalFloatBits(gain);
key.qBits[i]         = canonicalFloatBits(q);

// 変更案
if (!band.enabled)
{
    key.frequencyBits[i] = 0u;
    key.gainBits[i]      = 0u;
    key.qBits[i]         = 0u;
}
else
{
    key.frequencyBits[i] = canonicalFloatBits(frequency);
    key.gainBits[i]      = canonicalFloatBits(gain);
    key.qBits[i]         = canonicalFloatBits(q);
}
```

クランプは現行どおり先に適用する。無効バンドではクランプ結果を捨てるため、NaN や範囲外の値でもキーは 0 に落ち着く。

## 3. 等価性

キーが一致する 2 設定では、次の項目がすべて一致する。

1. 全バンドの `enabled`・`type`・`channelMode`。
2. 有効バンドの周波数・ゲイン・Q（同一クランプ後）。
3. `totalGain`・`saturation`・`agc`・`filterStructure`・`sampleRate`・`maxBlockSize`。

無効バンドの値は係数にも処理にも使われないため、係数キャッシュの内容と出力は一致する。

正規化はキーを粗くするだけで、出力が異なる設定を同じキーにまとめることはない。逆方向（キーが違えば出力が違う）は現行の性質を保つ。

## 4. 端ケース

| ケース | 挙動 |
|---|---|
| `sampleRate <= 0` | 係数生成は全バンドをバイパスにする。値は出力に影響しないが、本設計は `enabled` だけで正規化するためキーは値ごとに残る。初期化経路のみで頻度は低い。必要なら `enabled && sampleRate > 0` を条件にする拡張を別途検討する |
| 無効バンドに NaN・Inf | 0 固定のためキーは安定する |
| バンドの有効化・無効化 | `enabled` が変わりキーも変わる。誤った再利用は起きない |
| 無効バンドの type・channelMode 変更 | キーは変わる（現行どおり）。値の正規化より粗くしないため安全側 |

## 5. 影響範囲

- 無効バンドの値だけを変えた場合、キャッシュの再生成が起きなくなる。メモリと生成時間が減る。
- 音声出力は変わらない。
- `paramsHash`（= `hashCacheKey(key)`）の値が変わる。`paramsHash` は `eqCoeffHash` としてスナップショット、ISR ランタイム、テレメトリで使われる。用途はキャッシュのピン留めと同一性確認であり、同じ設定からは常に同じ値を計算するため、実行時の整合は保たれる。
- `eqCoeffHash` の値そのものを固定値として比較するテストや記録がある場合は、期待値の更新が必要になる。実装時に洗い出す。

## 6. テスト案

| # | 検証 | 種別 |
|---|---|---|
| 1 | 無効バンドの周波数・ゲイン・Q だけが違う 2 設定で、`makeCacheKey` の `==` と `hashCacheKey` が一致する | 単体 |
| 2 | その 2 設定で `cacheKeyMatches` が真になる | 単体 |
| 3 | バンドを有効化するとキーが変わる | 単体 |
| 4 | 無効バンドに NaN・Inf を入れてもキーが安定する | 単体 |
| 5 | 有効バンドの値を変えるとキーが変わる（既存性質の維持） | 単体 |
| 6 | `createCoeffCache` の係数と `bandActive` が一致する（等価性の直接確認） | 単体 |
| 7 | フック有効時、`getOrCreate` が同一キャッシュを返す | 統合 |

## 7. リスクと対策

| リスク | 対策 |
|---|---|
| 将来、無効バンドの値を音声以外の用途へ使い、同じキーを流用する経路が増える | キーは係数キャッシュ専用であることをコメントで明記する |
| `eqCoeffHash` の期待値を持つテスト・記録が壊れる | 実装時に洗い出し、期待値を更新する |
| 正規化条件の取り違えで、有効バンドの値まで 0 になる | テスト #5 と #6 で検出する |

## 8. 承認状態

設計のみ完了。コード変更、テスト追加、および `eqCoeffHash` 期待値の更新は未承認。
