# #1 EQ 係数キャッシュのハッシュ衝突修正

- 実施: 2026-11-06
- 対象: `ConvoPeq 構造・妥当性検証レポート` の #1（係数キャッシュの符号衝突）
- 制約: #1 の修正と回帰テストのみ。既定値、パラメータ範囲、係数生成、フィルター構造、#3、CMake、CI は変更していない。コミットと PR は Owner 承認後に実施した（`6772e536`、PR #155）
- 基準 HEAD: `da32ae63`

## 0. 決定状況（2026-11-06）

Owner は、実装差分のレビューを条件に本修正を承認した。レビュー完了後、コミットと PR の作成が承認された。差分レビューの材料は `doc/work113/20261106-1230_差分レビュー/report.md` に置いた。

| 項目 | 結果 |
|---|---|
| コミット | `6772e536`（承認済みの 4 ファイルのみ） |
| PR | #155 |
| CI | 全チェック通過（Windows Release、ASan、TSan、isr-verify、CodeQL、lint ほか） |
| マージ | Owner の承認待ち（未実施） |

## 1. 修正前の問題

| 項目 | 修正前 |
|---|---|
| ハッシュ | `floatToCanonicalBits` が `bits & 0x7FFFFFFF` で符号ビットを落とす |
| ヒット判定 | ハッシュ一致のみ。パラメータの比較はない |
| 帰結 | +6 dB と −6 dB が同一ハッシュになり、先に作られた係数を再利用する。帯域中心で 12 dB の誤差（±48 dB なら 96 dB） |

## 2. 修正内容

### 2.1 キーの導入

`EQCoeffKey` を追加した。float は符号を保持した canonical bits で持ち、次の正規化を行う。

| 対象 | 正規化 |
|---|---|
| 符号 | 保持する（+x と −x を区別） |
| ±0.0 | 0 に統一 |
| NaN | 単一の quiet NaN 表現に統一 |
| frequency | 係数生成と同じ範囲へクランプ（20 Hz〜min(20 kHz, 0.95·Nyquist)） |
| gain | −48〜+48 dB へクランプ |
| Q | 0.01〜20 へクランプ |
| totalGain | −48〜+48 dB へクランプ |
| saturation | 0〜1 へクランプ |
| その他 | enabled、type、channelMode、agc、filterStructure、sampleRate、maxBlockSize |

クランプ規約は `validateAndClampParameters`（`EQProcessor.Coefficients.cpp:91-103`）と揃えた。キーと係数生成が同じ意味の値を使う。

### 2.2 ハッシュ

`hashCacheKey` はキーの全フィールドを結合する。ハッシュ 0 は「キャッシュなし」の番兵なので 1 に置き換える。

### 2.3 ヒット判定

`AudioEngine::EQCacheManager::getOrCreate` を次のように変えた。

1. キーと基本ハッシュを算出する。
2. ロックなしでハッシュを引き、エントリのキーが一致すれば返す。
3. キーが違う場合は `probeCacheHash` で別のハッシュへ退避し、再探索する（最大 32 回）。
4. 空きが見つかれば、そのハッシュでキャッシュを作成して登録する。
5. 32 回すべて別キーで埋まる場合は、誤った係数を返さず fail-closed にする。

`EQCoeffCache` に `key` を追加し、`createCoeffCache` が設定する。スナップショットへ渡す `paramsHash` は、衝突回避後の最終ハッシュである。

### 2.4 変更ファイル

| ファイル | 変更 |
|---|---|
| `src/eqprocessor/EQProcessor.h` | `EQCoeffKey`、`EQCoeffCache::key`、キー生成・ハッシュ・照合・退避の宣言と実装 |
| `src/eqprocessor/EQProcessor.ProcessingCache.cpp` | 符号マスクの廃止。`computeParamsHash` をキー経由に変更。`createCoeffCache` でキーを保存 |
| `src/audioengine/AudioEngine.Cache.cpp` | キー照合と衝突時のハッシュ退避を追加 |
| `src/tests/AudioEngineHarness/STG11EQRetireTests.cpp` | 回帰テスト `checkCoeffCacheKeyContract()` を追加 |

## 3. 修正前後のヒット判定

| 状況 | 修正前 | 修正後 |
|---|---|---|
| 同一パラメータ | ハッシュ一致でヒット | キー一致でヒット |
| 符号だけ異なるゲイン | ハッシュ一致で誤ヒット | キー不一致。別ハッシュへ退避して新規作成 |
| クランプ後に同値（10 Hz と 20 Hz など） | ハッシュ不一致でミス | キー一致でヒット（係数も同じ） |
| ハッシュだけが偶然一致する別パラメータ | 誤ヒット | キー不一致を検出し、別ハッシュで登録 |
| sampleRate / maxBlockSize の差 | ハッシュ不一致 | キー不一致 |

## 4. 回帰テスト

`STG11EQRetireTests` に 15 項目を追加した（`STG11-D13`）。

| # | 検証 | 結果 |
|---|---|---|
| 1 | 同一パラメータのキーとハッシュが一致する | PASS |
| 2 | ゲインの符号を区別する（キーとハッシュ） | PASS |
| 3 | totalGain の符号を区別する | PASS |
| 4 | 周波数のクランプ後同値を同一キーにする | PASS |
| 5 | ゲインのクランプ後同値を同一キーにする | PASS |
| 6 | Q のクランプ後同値を同一キーにする | PASS |
| 7 | sampleRate の差を区別する | PASS |
| 8 | maxBlockSize の差を区別する | PASS |
| 9 | enabled の差を区別する | PASS |
| 10 | type の差を区別する | PASS |
| 11 | channelMode の差を区別する | PASS |
| 12 | saturation の差を区別する | PASS |
| 13 | filterStructure の差を区別する | PASS |
| 14 | agc の差を区別する | PASS |
| 15 | ハッシュが同じでもキーが違えばキャッシュヒットしない | PASS |
| 16 | 同一パラメータの係数生成が決定的である | PASS |

実行結果は次のとおり。

```
STG11-D13: coefficient cache key contract PASS
STG11EQRetireTests: PASS (TD1-1/TD1-2/TD1-3/TD1-4a/TD1-4b)
exit=0 elapsed=0.44s
```

既存の TD1-0〜TD1-4b もすべて PASS した。

## 5. 音響仕様への影響

| 項目 | 変更 |
|---|---|
| EQ パラメータの既定値 | なし |
| パラメータ範囲 | なし |
| 係数生成アルゴリズム | なし |
| フィルター構造 | なし |
| オーバーサンプリング利得 | なし（#3 は未着手） |
| 出力レベル | なし |

キャッシュの同一性判定だけを変えた。同じパラメータに対して生成される係数は従来と同じである。テスト 16 で決定性を確認した。

## 6. ビルドと実行

| 工程 | 結果 |
|---|---|
| configure（MKLROOT/IPPROOT を設定） | 成功（2.4 秒） |
| `STG11EQRetireTests` ビルド | 成功（798.5 秒、Release） |
| テスト実行 | exit 0、0.44 秒 |
| `ConvoPeq` ビルド | 結果は §7 |

`STG11EQRetireTests` の単独ビルドは、`MKLROOT` が未設定だと `ipp.h` を見つけられず失敗する。これは既存のターゲット構成によるもので、今回の変更とは無関係である。`MKLROOT` と `IPPROOT` を設定して configure し直すと通る。

## 7. アプリ本体のビルド

`ConvoPeq` ターゲットも Release でビルドした。

| 項目 | 結果 |
|---|---|
| configure | 成功（MKLROOT/IPPROOT 設定後） |
| `ConvoPeq` ビルド | 成功（exit 0）。`AudioEngine.Cache.obj` は 12:34 に更新され、今回の変更を含む |
| CI: Windows MSVC / Release | 成功（14 分 36 秒） |
| CI: ASan（MSVC、RelWithDebInfo） | 成功（1 時間 7 分 43 秒） |
| CI: TSan（Clang、best-effort） | 成功（38 秒） |
| CI: isr-verify | 成功（1 分 25 秒） |
| CI: CodeQL / lint / authority | すべて成功 |
| 実行ファイル | `build\ConvoPeq_artefacts\Release\ConvoPeq.exe`（45.4 MB、12:43 更新） |

`STG11EQRetireTests` のターゲットは `ConvoPeq` に依存するため、そのビルド中にアプリ本体のオブジェクトも更新された。最後の `ConvoPeq` ビルドはリンク確認のみで 1.4 秒だった。

## 8. 残存リスク

| リスク | 内容 |
|---|---|
| 無効バンドの値 | 無効バンドは係数がバイパス固定だが、キーには freq/gain/Q を含めるため、値だけ変えた場合に余分なキャッシュが作られる。誤ヒットはしない |
| 32 回の退避超過 | 32 個の候補がすべて別キーで埋まる場合は fail-closed（EQ バイパス）。実運用では起きない確率だが、挙動として明記する |
| 64 ビット衝突 | キー比較で検出するため誤ヒットしない。検出時はハッシュを退避する |
| RT 経路 | 変更なし。`pinByHash` は従来どおりハッシュで解決する |
| メモリ | キャッシュ 1 件あたり約 320 バイト増える |
| 統合テスト | 衝突時のハッシュ退避は `AudioEngine` に依存するため、単体テストでは判定プリミティブまでを検証した。退避ループ自体はコードレビューで確認した |

## 9. 位置づけ

本修正は #1 の範囲に限定した。承認済みの 4 ファイルを `6772e536` としてコミットし、PR #155 を作成した。マージは Owner の承認待ちである。
