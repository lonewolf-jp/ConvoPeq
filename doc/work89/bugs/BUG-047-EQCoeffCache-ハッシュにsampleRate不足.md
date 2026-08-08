# BUG-047: EQCoeffCache::computeParamsHash に sampleRate/maxBlockSize が含まれず、サンプルレート変更後に古い EQ 係数が使われる

- **発見日**: 2026-07-26
- **重要度**: **HIGH**
- **カテゴリ**: ロジックエラー / キャッシュ不整合
- **該当ファイル**: `src/eqprocessor/EQProcessor.ProcessingCache.cpp:24-44` (computeParamsHash)
- **関連ファイル**: `src/audioengine/AudioEngine.Cache.cpp:53-65` (getOrCreate のハッシュルックアップ)

## 症状

サンプルレート変更後、EQ パラメータを変更していない限り、古いサンプルレート用に計算された SVF 係数が使い続けられる。これにより、EQ フィルタのカットオフ周波数が Design 値から乖離し、意図しない周波数特性となる。

## 再現手順

1. 48kHz で任意の EQ を設定（例: 1kHz, +6dB, Q=0.707）
2. 正常に EQ がかかっていることを確認
3. オーディオデバイスのサンプルレートを 96kHz に変更（`prepareToPlay` 呼び出し）
4. `createSnapshotFromCurrentState` が Timer から呼ばれる
5. `eqCacheManager.getOrCreate(eqParams, 96000, ...)` は `computeParamsHash(eqParams)` でハッシュを計算
6. **ハッシュに sampleRate が含まれない**ため、48kHz 時に生成したキャッシュのハッシュと一致
7. 48kHz 用の係数を持つ `EQCoeffCache` が返される
8. スナップショットに 48kHz 係数のハッシュが格納される
9. Audio Thread の `buildAudioThreadProcessingState` が `eqCacheManager.get(hash)` で 48kHz キャッシュを取得
10. `eqRt().process(block, eqParams, cache)` が 48kHz 係数で 96kHz のオーディオを処理 → **カットオフ周波数が約 2 倍にシフト**

## 技術的詳細

### computeParamsHash（問題の箇所）

```cpp
uint64_t EQProcessor::computeParamsHash(const convo::EQParameters& params) noexcept
{
    uint64_t hash = 0;
    for (int i = 0; i < 20; ++i) {
        // ... band frequency, gain, q, enabled, type, channelMode
    }
    // totalGainDb, agcEnabled, nonlinearSaturation, filterStructure
    return hash;
}
```

`sampleRate`, `maxBlockSize`, `generation` のいずれもハッシュに含まれない。

### getOrCreate のルックアップパス

```cpp
// AudioEngine.Cache.cpp:58-65
const uint64_t hash = EQProcessor::computeParamsHash(params);
auto it = currentMap->map.find(hash);
if (it != currentMap->map.end())
    return it->second;  // ← sampleRate が異なっても同一ハッシュでヒット！
```

`getOrCreate` の第2引数 `sampleRate` はハッシュミス時（新規作成時）のみ使用され、キャッシュヒット時には無視される。

### EQCoeffCache の不変性

`EQCoeffCache` は生成時に渡された `sampleRate` で係数を計算し、それを `cache->sampleRate` に保存する（`EQProcessor.ProcessingCache.cpp:57`）。しかし、この `sampleRate` はどこでも検証されない。

## 影響範囲

- サンプルレート変更後、次にユーザーが EQ パラメータを変更するまで、全 EQ バンドの周波数特性が設計値から乖離する
- 48kHz→96kHz の場合、全カットオフ周波数が約 2 倍にシフト（ナイキスト周波数の比率による）
- 発現はサンプルレート変更時に必ず発生し、ユーザーが気づかないまま不正な EQ で再生が継続する

## 修正案

`computeParamsHash` に `sampleRate`, `maxBlockSize`（または `generation`）を含める：

```cpp
uint64_t EQProcessor::computeParamsHash(
    const convo::EQParameters& params,
    double sampleRate,
    int maxBlockSize) noexcept
```

または `getOrCreate` のキャッシュヒット時に `sampleRate` を検証する：

```cpp
auto it = currentMap->map.find(hash);
if (it != currentMap->map.end()) {
    if (it->second->sampleRate == sampleRate
        && it->second->maxBlockSize == maxBlockSize)
        return it->second;
    // サンプルレート不一致 → 既存キャッシュを無効化して再作成
}
```
