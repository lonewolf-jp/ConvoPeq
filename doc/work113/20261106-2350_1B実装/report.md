# #1B 無効バンドのキー正規化 実装

- 実施: 2026-11-06
- 基準: `main` = `8e54a53a`（Owner 指定の `db77ab2b` 以降はドキュメントのみ）
- 承認: 設計案に基づく実装・テスト追加（2026-11-06 の Owner 決定）
- 範囲: #1B のみ。探索上限（#1A）、RT 処理、音声出力の変更は含まない

## 1. 変更内容

`EQProcessor::makeCacheKey` で、無効バンドの周波数・ゲイン・Q を 0 に正規化する。

- 有効バンド: 従来どおりクランプして `canonicalFloatBits` で格納する。
- 無効バンド: `frequencyBits`・`gainBits`・`qBits` を `0u` にする。
- `enabled`・`type`・`channelMode` は従来どおり保持する。
- `sampleRate <= 0` の扱いは変更しない（設計どおり今回は拡張しない）。

無効バンドは `createCoeffCache` でバイパス係数になり、処理側は `bandActive` を先に確認するため、値は係数にも出力にも影響しない。キーだけを粗くし、同じ有効設定が同じキャッシュを再利用できるようにする。

## 2. 差分レビュー

| ファイル | 変更 |
|---|---|
| `src/eqprocessor/EQProcessor.h` | `makeCacheKey` のバンドループを有効・無効で分岐。無効時は 3 つのビット列を 0 に固定 |
| `src/tests/AudioEngineHarness/STG11EQRetireTests.cpp` | D13 に正規化の単体テストを追加（キー・ハッシュ・非有限値・再有効化・係数一致） |
| `src/audioengine/AudioEngine.h` | テストフック `debugRunEqDisabledBandKeyProbe` の宣言（フック有効時のみ） |
| `src/audioengine/AudioEngine.Cache.cpp` | 上記フックの実装（同一キャッシュの再利用と、有効化時の別キャッシュを検証） |
| `src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp` | フックを呼ぶ統合チェック `checkEqCacheDisabledBandKeyProbe` を追加 |

確認点:

- 有効バンドの値・クランプ・`canonicalFloatBits` の扱いは変えていない。
- `enabled`・`type`・`channelMode`・`totalGain`・`saturation`・`agc`・`filterStructure`・`sampleRate`・`maxBlockSize` は従来どおりキーに含む。
- RT 経路（`process` / `pinByHash`）には触れていない。
- フックは `CONVOPEQ_ENABLE_TEST_HOOKS` が定義されたビルドでのみコンパイルされる。

## 3. `eqCoeffHash` の影響調査

`paramsHash` は `hashCacheKey(key)` であり、無効バンドの値に依存しなくなる。期待値への影響を調べた。

- `src/tests` に EQ キャッシュのハッシュ固定値は存在しない（`paramsHash == hashOf(key)` の相対比較のみ）。
- `AudioEngine.Cache.cpp` の衝突プローブは有効バンドのゲインを使うため影響なし。
- 設計資料・過去の実行記録にはハッシュ値の記載があるが、いずれも過去の観測記録であり、期待値として比較していない。

したがって、更新が必要な固定期待値はなかった。

## 4. テスト

| テスト | 結果 |
|---|---|
| `STG11EQRetireTests`（フックなし） | exit 0。`STG11-D13: coefficient cache key contract PASS`（追加した正規化の単体テストを含む） |
| `AudioEngineHarness`（`CONVOPEQ_ENABLE_TEST_HOOKS=ON`） | exit 0。`[#1B] disabled-band key normalisation probe: PASS`、`[#1]` も PASS、`all publish pipeline tests PASS` |
| `ctest -C Release` | **54 / 54 PASS**、332.7 秒（フック有効ビルド） |

フック有効ビルドは `-DCONVOPEQ_ENABLE_TEST_HOOKS=ON` で構成し、`CONVOPEQ_SOURCE_ROOT` を設定して実行した。

## 5. soak・TX の決定記録

`doc/work113/20261106-2030_残存対応/03-soak-tx-conditions.md` の §7 に、2026-11-06 の Owner 決定（soak の運用条件、TX の測定条件、承認境界）を追記した。

## 6. 状態

| 項目 | 状態 |
|---|---|
| #1B 実装 | 完了 |
| 単体テスト | 追加済み |
| 統合テスト（フック有効時） | 追加済み |
| コミット・PR | `a2aa12c0`、PR #163（マージは未実施） |
