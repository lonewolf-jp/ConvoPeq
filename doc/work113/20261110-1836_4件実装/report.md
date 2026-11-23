# 4 件の実装完了報告（2026-11-10）

## §0 実装前の確認

| 項目 | 結果 |
|---|---|
| ブランチ / HEAD | `main` / `c94d3767` |
| レビュー対象との一致 | `src/InputInspection.cpp` の SHA-256 が `A4F4F01A31F2558B…` で、レビュー用コピーと一致 |
| パッチのドライラン | `patch -p0 --dry-run` が警告なしで成功 |
| ロールバック用バックアップ | `tmp/rollback/{item1_3,item2,item4}/` にバイトコピー、`tmp/rollback/manifest.txt` に SHA-256 を記録 |

実装は 4 件を独立に扱い、項目ごとの差分を `patches/` に置いた。既存の未コミット変更は保持している。

## §1 項目 4: `computeMaxElementDepth` の置換

### 1.1 適用

- `patches/item4_computeMaxElementDepth.patch`（承認済みの差分）を `patch -p0` で適用した。警告・fuzz なし。
- 適用後のファイルは 6,948 → 8,406 バイト、SHA-256 は `A4F4F01A…` → `C7259580…`。
- 適用後の関数は、レビュー時の提案実装と**バイト一致**することを確認した（148 行）。

### 1.2 同等性

適用後の実ソースから関数を抽出して 26 ケースを実行し、**26/26 一致**（`summary: pass=26 fail=0`）。

### 1.3 性能（同一環境・同一手順、2 回ずつ）

| 測定 | 置換前 | 置換後 |
|---|---|---|
| 起動（プロセス開始 → `Auto-exit scheduled`） | 12,573 / 12,290 ms | **2,177 / 1,744 ms** |
| `DeviceSettings::saveSettings`（`step 6`→`step 7`） | 10,498 / 10,313 ms | **41 / 42 ms** |
| プロセス総経過 | 27,146 / 26,726 ms | **6,490 / 5,913 ms** |

### 1.4 ビルド警告（未修正・報告）

置換により `warning C4267`（`size_t` → `const int`）が 1 件発生する。`juce::String::getNumBytesAsUTF8()` の戻り値が `size_t` であるためで、実ファイル（最大 10 MB 制限）では実害がない。承認済みパッチをそのまま適用する条件を守り、**修正は行っていない**。最小修正は `static_cast<int>` の 1 行だが、レビュー済み差分から逸脱するため提案に留める。

### 1.5 ロールバック

`tmp/rollback/item4/InputInspection.cpp` を戻すか、`patch -R` で逆適用する（SHA-256 で一致確認可能）。

## §2 項目 1: 退役遅延の計測拡張

### 2.1 追加した記録

- スロットに `safeTsNs`（EBR 安全到達）、`minReaderEpoch`、`minReaderSlot` を追加。`tsNs` を enqueueTs として使う。
- `recordSafe()` を新設し、D キューの `reclaim`（安全判定の直後）、`terminalReclaim` の同期経路、`TerminalReclaimAuthority::drain` の 3 箇所で記録する。強制解放（`drainAll`、`drainAllUnsafe`）は安全到達を記録しない（未観測として扱う）。
- 要求時に最小 reader epoch と保持スロットを求める（`getMinReaderEpoch()` と `getReaderSlotDetail()` の走査。Non-RT の `enqueueRetire` のみ）。
- 集計: `ebrNs = safeTs - enqueueTs`、`execNs = reclaimTs - safeTs`。件数・合計・最大を Snapshot／StageStats に追加。
- レポート: `timing`（runtime 段階を優先）と、観測ごとの `enqueueTsNs`・`safeTsNs`・`minReaderEpoch`・`minReaderSlot`。

### 2.2 既存の判定・reclaim 条件は不変

`evaluateRetireDelay` と安全判定の式は差分に含まれない（パッチ内に `evaluateRetireDelay` の変更 0 件）。追加は記録のみ。

### 2.3 実測（交互バースト 40 回）

| 項目 | 値 |
|---|---|
| 判定 | fail（`latency_exceeds_threshold`、観測 943、最大 4,404 ms） |
| EBR 待ち | 平均 **362.5〜364.9 ms**、最大 **4,404〜4,419 ms** |
| 実行待ち（safe 到達後） | 平均 0 ms、最大 **0.000〜0.016 ms** |
| safe 未観測（強制解放） | 903 件 |
| 生観測 256 件のうち safeTs あり | 130 件 |
| 最小 epoch 保持スロットが特定できた件数 | 129 / 256 |

EBR 待ちが支配的で、安全到達後の実行は 16 µs 以下である。§1 の分解仮説を実測で裏付けた。

## §3 項目 2: RuntimeWorld 寿命の記録

### 3.1 追加した記録

- 新規 `WorldLifetimeRecorder`（D1 とは別クラス・別 key）。256 スロット、アトミックのみ。
- commit: `enqueueRuntimePublicationFireAndForget` の owner channel enqueue 成功時。`shutdownRuntime_.isShutdownInProgress()` で `shutdown`、それ以外は `publish`。
- 拒否: `retireRejectedRuntimeWorldNonRt` で `rejected` として起点を記録する（commit に到達しない world のため）。
- destroy: 2 つの retire 関数の deferred delete の deleter 内（`ptr->~RuntimePublishWorld()` の直前）。
- レポート: `worldLifetime`（commits／destroys／unmatched／overflow、経路別の件数・合計・最大、生観測 128 件）。

### 3.2 実測（交互バースト 40 回・2 回）

| 項目 | 値 |
|---|---|
| commits / destroys | 40/40、41/41 |
| unmatched | 1 |
| overflow | 0 |
| `publish` 経路 | 40/40、41/41、最大 2,209〜2,217 ms |
| `shutdown`・`rejected` 経路 | この実行では 0 件 |

### 3.3 制約

- 経路は **commit 時点の文脈**で決まる。起動時の置換（bootstrap）と再構成（reconfigure）は `publish` に含まれる。区別するには呼出元から経路を渡す API 変更が必要で、今回の承認範囲外のため未実施。
- 終了時に破棄される world も、commit が publish なら `publish` として記録される。
- `unmatched = 1` は、commit を記録しない生成物（初期化時の一時 world など）の破棄と推定しているが特定には至っていない。

## §4 項目 3: `retireRT` の記録（TLS 区画）

### 4.1 追加した記録

- `recordRequestRT()`: TLS で 8 区画（各 64 スロット）の 1 つを払い出し、**その区画内だけ**を CAS で走査する。確保・ロック・ログ・文字列生成なし。
- `retireRT` で要求を先に記録し、`provider_->enqueueRetire` が失敗したら `cancelRequest` で取り消す（孤児を作らない）。
- `rtRequests` を集計し、レポートの `timing.rtRequests` に出す。

### 4.2 検証

| 測定 | 結果 |
|---|---|
| RT 記録（100,000 組） | **68.4 ns/組**（requests 100,000 / completions 100,000 / overflow 0） |
| 全スロット走査（比較） | 267.9 ns/組 |
| 区画満杯時の RT 呼出（20,000 回） | **283.1 ns/回**（overflow に計上） |
| アプリ実行での `rtRequests` | **0** |

`retireRT` は `RefCountedDeferred` からのみ呼ばれ、現行コードではその利用箇所がコメント（移行済み）だけである。したがってこの実行では RT 経路が使われず、**D1 の `unmatched = 3` は RT 経路ではなく retryable 経路の完了**によるものである（`rtRequests = 0` が根拠）。

### 4.3 制約

- RT 経路の実走例が現行コードに無いため、「呼出回数と記録件数の一致」はアプリ実行では確認できていない。単体（100,000 組）での一致のみ確認した。
- RT 側では最小 reader のスロット走査を行わない（O(1) を優先）。`minReaderEpoch` のみ 1 回読み取る。

## §5 横断検証

| 項目 | 結果 |
|---|---|
| Release ビルド | 成功（`ConvoPeq` ターゲット） |
| ctest（実装途中のビルド） | 60/60 合格・337.75 秒 |
| ctest（最終コード） | §5.1 に追記 |
| 既存の警告 | `MklFftEvaluator.h` の C4390（既存）、`RtMeasureRing.h` の C4324（既存）。新規は項目 4 の C4267 のみ |

### 5.1 ctest（最終コード）

Release 構成で全ターゲットをビルドし、`build/` で `ctest -C Release --output-on-failure` を実行した。

```
100% tests passed out of 60
Total Test time (real) = 334.96 sec
ctest_exit=0
```

`AudioEngineHarness` と RT 系を含む 60 件すべて合格した。4 件の実装による回帰は検出されていない。

## §6 未解決事項

1. `worldLifetime` の `unmatched = 1` の特定（初期化時の一時 world と推定）。
2. 経路の完全分離（bootstrap／reconfigure／deferred は現状 `publish` に含まれる）。API 変更が必要。
3. `retireRT` の実走例が無いため、RT 経路の件数一致はアプリでは未確認。
4. 項目 4 の C4267 警告（承認済みパッチを優先し未修正）。
5. D1 の `unmatched = 3`（retryable 経路）は引き続き未対応。

## §7 ロールバック

| 項目 | 対象 | 方法 |
|---|---|---|
| 4 | `src/InputInspection.cpp` | `patches/item4_...patch` の逆適用、または `tmp/rollback/item4/` から復元 |
| 1・3 | `ISRRetireLatency.h`、`ISRRetireRouter.cpp`、`DeferredDeletionQueue.h`、`ISRHB.cpp`、`ISRHB.h` | `patches/item1_item3_...patch` の逆適用、または `tmp/rollback/item1_3/` から復元 |
| 2 | `AudioEngine.h`、`AudioEngine.CtorDtor.cpp`（2 行）、`WorldLifetimeRecorder.h`（新規） | `patches/item2_...patch` の逆適用＋新規ファイル削除 |

復元後は `tmp/rollback/manifest.txt` の SHA-256 と照合できる。
