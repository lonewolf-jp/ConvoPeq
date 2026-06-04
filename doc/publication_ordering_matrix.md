# Publication Ordering Matrix

## 役割分離

`runtimeGeneration`、`publicationSequence`、`activationEpoch` の3つの識別子はそれぞれ異なる役割を持つ。

| 識別子 | 型 | 役割 | 単調性 | 更新タイミング |
|---------|-----|------|--------|---------------|
| `runtimeGeneration` | `uint64_t` | RuntimeWorld の世代。再構築のたびに increment | 単調増加 | `RuntimeBuilder::buildRuntimePublishWorld()` |
| `publicationSequence` | `PublicationSequenceId` (uint64_t) | Publication の順序。Publish のたびに increment | 単調増加 | `reserveRuntimePublicationIdentity()` |
| `activationEpoch` | `uint64_t` |  Activation 時点の epoch。RCU epoch との比較用 | 単調増加 | `GenerationSemantic.activationEpoch` |

## 順序関係

```
runtimeGeneration <= publicationSequence (通常)
activationEpoch <= runtimeGeneration (通常)
```

- `runtimeGeneration` は rebuild ごとに 1 増加
- `publicationSequence` は publish ごとに 1 増加（rebuild なしの publish も含む）
- `activationEpoch` は GenerationSemantic に格納され、generation と同一値で初期化

## 検証方法

- `ISRRuntimePublicationCoordinator::commit()` 内で `sequenceId` の単調増加を検証
- `CoverageVerifier` でフィールドの包含関係を検証
