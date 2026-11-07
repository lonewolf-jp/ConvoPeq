# #1A 係数キャッシュ探索上限の拡張（32 → 64）

- 実施: 2026-11-07
- 基準: `main` = `ae91385d`（最新 `ConvoPeq.md` は `7faf9fa9` のソースを反映）
- 承認: Owner 指示（#1A の探索上限改善）

## 1. 実装前の確認

- 該当箇所: `AudioEngine::EQCacheManager::getOrCreate`（`AudioEngine.Cache.cpp`）。
- 呼び出し経路: `AudioEngine.Snapshot.cpp` のスナップショット構築（NonRT）。
- RT 側は `pinByHash` の単発参照のみで、探索ループを通らない。
- 現行は 32 回で探索を打ち切り、全段衝突時は新規キャッシュを破棄して `nullptr` を返す（fail-closed）。呼び出し側はスナップショットを抑制する。
- `ConvoPeq.md` でも `probe < 32` が 2 箇所（L4732、L4770）に存在することを確認した。

## 2. 変更内容

| ファイル | 変更 |
|---|---|
| `src/audioengine/AudioEngine.h` | `EQCacheManager::kMaxCollisionProbes = 64` を追加し、探索上限を名前付き定数にした |
| `src/audioengine/AudioEngine.Cache.cpp` | 2 箇所の `probe < 32` を `probe < kMaxCollisionProbes` に置換。テストフック `debugRunProbeDepthProbe` を追加 |
| `src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp` | 40 段の衝突連鎖を作り、探索が成功することを確認する統合チェックを追加。探索 1 段のコストも出力 |

所有権・寿命・スレッド安全性は変更していない。`CacheMap` のコピーオンライト、`DSPHandle` の生成・破棄、`writeMutex` の範囲も従来のまま。

## 3. 検証

| 項目 | 結果 |
|---|---|
| 40 段衝突での探索 | PASS（`[#1A] cache probe depth probe: PASS`）。32 回打ち切りでは fail-closed になる深さ |
| 既存の衝突プローブ（#1） | PASS |
| #1B の正規化プローブ | PASS |
| ハーネス全体 | exit 0、`all publish pipeline tests PASS` |
| 探索 1 段のコスト | 3.15 ns/段（100 万回の平均、整数演算のみ）。64 段でも最悪 約 0.2 µs の追加 |

## 4. メモリ・RT への影響評価

- **メモリ**: 追加の確保はない。探索は既存のマップ読み取りと `probeCacheHash` の整数演算のみ。
- **RT**: `pinByHash` は変更しておらず、RT 経路の負荷は不変。探索ループは NonRT のスナップショット構築時だけ実行される。
- **最悪ケース**: 全段衝突時は 32 段から 64 段へ増えるが、整数演算 32 回分の追加に限られる。それでも衝突が続く場合は従来どおり fail-closed し、誤った係数は返さない。

## 5. 残存リスク・未検証

- 64 段でも全段衝突する構成では従来どおり fail-closed する（設計上の上限）。
- 実運用で 32〜64 段の衝突が起きる頻度は計測していない。発生条件の再現は別途。
- 性能改善ではなく、衝突時の可用性改善である。

## 6. 状態

| 項目 | 状態 |
|---|---|
| 実装 | 完了 |
| テスト | 追加・PASS |
| PR | 作成済み（マージ未実施） |
