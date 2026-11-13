# C3-D 実装 — head identity の取得 API 拡張

- 作業日時: 2026-11-13 22:15 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）＋未コミットの診断ログ
- 種別: **Owner 承認済み範囲（head identity の取得 API 拡張）のみ**。epoch 前進の実装は含まない。
- 承認条件: `peekHead` の out 引数拡張、identity を `(headSlotIndex, headSequence, headEpoch)` の 3 要素で
  構成、epoch/type/domain を identity の代替にしない、既存の読み取り整合性・sequence 検証・
  キュー非変更の契約を維持、診断側の利用箇所へ反映。

---

## 1. 実装範囲（承認との対応）

| # | 承認要件 | 実装 |
| --- | --- | --- |
| 1 | `peekHead` を拡張し `slotIndex` と `sequence` を out 引数で取得 | `peekHead(DeletionEntry&, uint32_t&, uint32_t&)` に変更（呼出し元は 1 箇所のみ） |
| 2 | identity を 3 要素で構成 | `RetireHeadSnapshot` に `headSlotIndex` / `headSequence` を追加（`headEpoch` と合わせて 3 要素） |
| 3 | epoch / type / domain を代替にしない | コメントで禁止を明記し、3 要素を個別に保持。比較は 3 要素の組で行う前提 |
| 4 | 既存の整合性・sequence 検証・キュー非変更を維持 | コピー前後の `seq` と `dequeuePos` の再確認をそのまま維持。書き込み・CAS・dequeue なし |
| 5 | 診断側の利用箇所へ反映 | `inspectRetireHead()` が 3 要素を返し、`[DRAIN_DIAG]` の同一行に `headSlot` / `headSeq` を追加 |

**含まないもの**: `waitForDrain` のループでの epoch 前進（C3-D の本体）。したがって
2 秒 drain タイムアウトは解消していない（§4 の実測どおり 3/3 で発生）。

## 2. 変更内容

| # | ファイル | 箇所 | 内容 |
| --- | --- | --- | --- |
| 1 | `src/DeferredDeletionQueue.h` | `:290-311` | `peekHead` のシグネチャ拡張と identity の out 出力 |
| 2 | `src/core/EpochDomain.h` | `:564-565`, `:581-590` | `RetireHeadSnapshot` に 2 フィールド追加、`inspectRetireHead` で充填 |
| 3 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | `:771-772` | `headSlot=` / `headSeq=` を同一行に追加 |
| 4 | `src/tests/ISRSemanticValidationTests.cpp` | `:20`, `:2656-2727`, `:2993-2995` | 契約テスト AC2 の追加と登録 |

### 2.1 契約（`peekHead`）

```
成功: out（entry）・slotIndex・sequence の 3 つを書く
失敗: 3 つとも一切書かない（推定値を返さない）
```

- 失敗条件は 2 つ。`seq != pos + 1`（空 / 未 publish）と、コピー中に `seq` または `dequeuePos` が
  変化した場合（上書き・破棄）。
- コピーは `DeletionEntry` の POD コピーのみ（ポインタを辿らない）。
- キュー状態は変更しない（`dequeuePos` / `sequences` / `ringBuffer` への書き込みなし）。

### 2.2 ログ（行数は増えない）

```
[DRAIN_DIAG] timedOut=1 ... head=present headSlot=6 headSeq=7 headRetireEpoch=12 headType=0
  headDomain=2 headScanned=1 headScanAgeMs=2002 minReaderEpoch=12 currentEpoch=12
  retireEpochLtMinReader=0
```

`head=unavailable` のときは `headSlot` / `headSeq` / `headRetireEpoch` などは `na` になる
（`valid=false` のときは書かない契約のため、推定値は出ない）。

## 3. 検証

### 3.1 ビルド

`cmake --build build --config Release` → 256 ターゲット、エラー 0（警告 2 件は `MklFftEvaluator.h` の
既存 C4390）。追加ビルド（テストターゲット単体）も成功。

### 3.2 追加テスト AC2（`testAC2_headIdentity`）

| # | ケース | 期待 | 結果 |
| --- | --- | --- | --- |
| 1 | 空キュー | false、out 引数は sentinel のまま（推定値を返さない） | PASS |
| 2 | 2 件 enqueue 後の先頭 | true、`slot=0` / `seq=1` / `epoch=10` / `domain=EngineISR` | PASS |
| 3 | peek を 5 回反復 | 毎回同じ 3 要素、`sizeApprox()==2`（キュー非変更） | PASS |
| 4 | 先頭だけ reclaim（`reclaim(11)`） | 戻り 1、次の identity は `slot/seq` が進む（同一でない） | PASS |
| 5 | 全回収後 | false、out 引数は sentinel のまま | PASS |
| 6 | 副作用 | deleter が各 1 回、キューは空 | PASS |

### 3.3 ミューテーション試験（テストが実際に効いていることの確認）

`if (h1.epoch != 10)` を一時的に `!= 99` に変えて再ビルド → 実行結果:

```
TEST FAILED: AC2: head identity (slot, sequence, epoch) contract failed
exit=1
```

確認後に元へ戻し、再ビルドして `exit=0`（PASS）に復帰。テストが実行され、誤った期待を検出できることを
確認した。

### 3.4 既存テスト

`ctest --test-dir build -C Release` → **61/61 成功**（340.80 s）。AC2 は
`ISRSemanticValidationRejects` の中で実行される。

### 3.5 実機 3 回（CLI）

`--cli-run --cli-exit-ms 2000 --verify-runtime-evidence-live` を 3 回。

| 実行 | exit | 総経過 | 行数 | identity（slot / seq / epoch） | `failed` | `c3` | `c9_rbl` | `retireEpochLtMinReader` |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 1 | 0 | 6.81 s | 2 | 6 / 7 / 12 | `c3,c9` | 1 | 143 | 0 |
| 2 | 0 | 6.30 s | 2 | 6 / 7 / 12 | `c3,c9` | 1 | 143 | 0 |
| 3 | 0 | 6.24 s | 2 | 6 / 7 / 12 | `c3,c9` | 1 | 143 | 0 |

- 3 要素の identity が 3 回とも同じ値で取得できた（`slot = dequeuePos & 4095 = 6`、`seq = pos + 1 = 7`、
  `epoch = 12`）。
- 既存の観測値（`failed=c3,c9`、`c3=1`、`c9_rbl=143`、`retireEpochLtMinReader=0`）は**変更前と同一**で、
  API 拡張が drain 判定や reclaim の挙動に影響していないことを示す。
- ログは 1 shutdown あたり 2 行のまま。

### 3.6 ConvoPeq.md の再生成

ソースが変わったため、実装完了後に再生成した。

| 項目 | 値 |
| --- | --- |
| stamp | 2026-11-13 22:16:37 |
| サイズ | 7,228,910 B |
| 検証 | `--verify-snapshot` で `name == header: OK` / `content == current: OK` |
| 収録確認 | `peekHead(DeletionEntry& out,` / `slotIndex = idx;` / `headSlotIndex` / `headSequence` / `testAC2_headIdentity` / `headSlot=` / `headSeq=` がいずれも存在 |
| 分割 | 4 ファイル（1,993,465 / 1,989,538 / 1,985,715 / 1,260,444 B）。再結合すると元の内容と一致する |

## 4. 未実施（承認範囲外）

| # | 内容 | 理由 |
| --- | --- | --- |
| 1 | `waitForDrain` での epoch 前進（C3-D 本体） | 今回の承認は API 拡張に限定。タイムアウトは未解消 |
| 2 | C9-A（Coordinator の drain 述語変更） | 別承認 |
| 3 | `retireBacklogCount_` の改名 | 別課題 |
| 4 | Debug 構成での確認 | 今回は Release のみ |

## 5. 残留リスク

- `peekHead` は読み取り専用だが、`DeletionEntry`（56〜64 B）のコピー 1 回分のコストがある。
  呼出しはタイムアウト時の 1 回のみ。
- identity の 3 要素は「その瞬間」の値である。2 ms ポーリングの直後なので、直前に観測した値と
  1 entry ずれる可能性は残る（値のスナップショットとして扱う）。
- 前進処理は未実装のため、2 秒タイムアウトは 3/3 で継続している。**C3-D の効果検証は未実施**。

## 6. ロールバック

| 対象 | 撤去方法 |
| --- | --- |
| `peekHead` の out 引数 | シグネチャを 1 引数へ戻し、`inspectRetireHead` の呼出しを戻す |
| `RetireHeadSnapshot` の 2 フィールド | 削除 |
| ログの `headSlot` / `headSeq` | 2 項目を削除 |
| テスト AC2 | 関数・include・登録を削除 |

`isFullyDrained()` と `reclaim()` は変更していないため、判定ロジックの復元は不要。

## 7. 判定

- 承認された範囲（head identity の取得 API 拡張）は**完了**。3 要素を 3 回とも取得でき、
  失敗時は推定値を返さず、キューと reclaim に副作用がないことをテストで固定した。
- **C3-D の本体（epoch 前進）は未実装**。2 秒タイムアウトは解消していない。
- 既存の回帰は 61/61 で維持。
