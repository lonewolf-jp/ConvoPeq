# `retire_delay` の 3 分類診断設計（判定式は変更しない・実装未承認）

Owner 判断（2026-11-11）を受け、現行の単一「latency」を 3 成分に分ける**診断**を設計する。**判定式（50 ms・N_req=59）と `evaluateRetireDelay` は変更しない。実装もしない。**

## §0 現行計測の構造と、その限界

```
要求 recordRequest(ts) ───────────────────────────────► 完了 recordComplete(now)
                                    ↑
                        recordSafe(safeTs) ─ deleter
現行: latency = now - ts
      ebr     = safeTs - ts          ← 「要求から安全判定まで」を全部 EBR と呼んでいる
      exec    = now - safeTs
```

`safeTs` は reclaim の**直前**に打たれる。したがって `ebr` には「reclaim ドライバが動かなかった時間」が含まれる。これが 4.4 秒の正体であり、EBR 待ち（reader の保持）と混同していた。

## §1 3 分類の定義

| 分類 | 定義 | 取得するタイムスタンプ |
|---|---|---|
| **reclaim ドライバ待ち** | 退役（enqueue）から、その entry が**最初に reclaim 走査で評価された**時点まで | `firstScanTs`（新設） |
| **EBR 待ち** | 最初の走査から、epoch 安全条件が成立した時点まで | `safeTs`（既存） |
| **実行待ち** | 安全条件成立から deleter 実行完了まで | `safeTs` → `now`（既存） |
| **未分類** | 上記 3 つの和が `latency` に満たない残差 | 残差として明示（0 で埋めない） |

### 1.1 `firstScanTs` の取り方（設計）

`DeferredDeletionQueue` の entry に `std::uint64_t firstScanTsNs` を追加（8 バイト）。`reclaim` が先頭 entry を評価したとき（`isOlder` を判定する直前、`scanPos == deqPos` の分岐に入る前）、未設定なら `nowNs` を CAS で 1 回だけ書く。

- 走査は FIFO の先頭だけを見る実装なので、`firstScanTs` は「その entry が先頭に到達して初めて評価された時刻」になる。これが「reclaim ドライバが動いた時刻」と一致する。
- 追加コストは 1 entry あたり 8 バイトと、初回のみの CAS 1 回。
- `recordComplete` で `firstScanTs` を読み、3 分類を計算して観測に含める。

### 1.2 計算と不変条件

```
driverWait = firstScanTs - ts           (firstScanTs が未設定なら未分類扱い)
ebrWait    = safeTs - firstScanTs       (safeTs >= firstScanTs のとき)
execWait   = now - safeTs
classified = driverWait + ebrWait + execWait
unclassified = latency - classified      (負なら 0 に丸めず「不整合」として別計数)
```

- 順序が壊れている場合（`firstScanTs < ts` など）は **`unclassified` を増やし、その要求を分類対象から外す**。0 として混ぜない。
- `safeTs` が未観測（`drainAll` の強制解放など）の要求は、従来どおり `unobservedSafe` として別計上し、分類には含めない。

## §2 出力（診断用の追加セクション）

既存の `retireLatencyD1` は**一切変更しない**。次のセクションを追加する。

```json
"retireLatencyClassified": {
  "note": "diagnostic only; the verdict uses retireLatencyD1.runtime and is unchanged",
  "thresholdNs": 50000000,
  "required": 59,
  "paths": [
    { "name": "queue", "count": 962,
      "driverWait": { "sumNs": 0, "maxNs": 0, "over50ms": 0 },
      "ebrWait":    { "sumNs": 0, "maxNs": 0, "over50ms": 0 },
      "execWait":   { "sumNs": 0, "maxNs": 0, "over50ms": 0 },
      "unclassified": { "count": 0, "sumNs": 0 } }
  ],
  "totals": { "classified": 0, "unclassified": 0 },
  "consistencyErrors": 0
}
```

## §3 既存判定との関係（明示）

| 項目 | 扱い |
|---|---|
| 50 ms 閾値 | **変更しない** |
| `N_req = 59` | **変更しない** |
| `evaluateRetireDelay` | **変更しない**（`runtime` 段の total latency のみを読む。バイト一致を維持） |
| verdict | 分類の追加で**変わらない**。分類は診断であり、判定を改善したようには見せない |
| 比較 | 同じ実行の `retireLatencyD1`（従来）と `retireLatencyClassified`（新）を並べる。従来の `ebr` が新分類の `driverWait + ebrWait` に相当することを明記 |

## §4 既存データへの適用例（推定）

直近の実行（bypass 40 回 × 250 ms）の最遅要求は次の値だった。

| 項目 | 実測値 |
|---|---|
| path | `queue` |
| `epoch` / `minReaderEpoch` / `minReaderSlot` | 1 / 1 / 0 |
| latency | 4,405 ms |
| `safeTs - ts`（現行の ebr） | ≈ 4,405 ms（latency とほぼ一致） |
| `now - safeTs`（現行 exec） | < 1 ms |

新分類では次になると推定される（`firstScanTs` は未計測のため**推定**と明記する）。

| 分類 | 推定値 | 理由 |
|---|---|---|
| reclaim ドライバ待ち | ≈ 4,404 ms | entry が先頭で評価されたのは解放の直前（安全判定と同時）であり、それ以前に走査が無かった |
| EBR 待ち | ≈ 0 ms | 走査時点で `isOlder(1, minReader)` が成立していた（reader 保持は最長 39.4 ms で、その時刻には終了している） |
| 実行待ち | < 1 ms | deleter は即時 |
| 未分類 | ≈ 0 ms | 3 成分で latency を説明できる（検証は実装後） |

**この推定は「現行の ebr が driverWait を含んでいた」ことを示すための例**であり、実装後の実測で置き換える。

## §5 分類不能時間の扱い

| 状況 | 扱い |
|---|---|
| `firstScanTs == 0`（一度も走査されずに drainAll で解放） | `unclassified` に **latency 全額**を計上（0 にしない） |
| `safeTs == 0`（`recordSafe` 未観測） | 既存の `unobservedSafe` に計上し、分類から除外 |
| 順序不整合（`firstScanTs < ts` など） | `consistencyErrors` を増やし、当該要求は `unclassified` |
| 丸め誤差（1 µs 未満） | `unclassified` に含める（0 埋めしない） |

## §6 実装対象（承認後・今回は未実装）

| ファイル | 変更 |
|---|---|
| `src/DeferredDeletionQueue.h` | entry に `firstScanTsNs` を追加し、初回走査時に CAS で記録 |
| `src/audioengine/ISRRetireLatency.h` | `Observation` に `firstScanTsNs` と 3 分類値、`PathStats` に分類別の集計 |
| `src/audioengine/ISRHB.cpp` | `retireLatencyClassified` セクションの出力（判定関数の外） |
| テスト | 分類の合成（driver+ebr+exec+unclassified == latency）と、`unclassified` が 0 埋めされないこと |

## §7 検証計画

| # | 検証 | 期待 |
|---|---|---|
| 1 | `evaluateRetireDelay` | バイト一致・verdict 不変 |
| 2 | 分類の合成 | 各要求で `driver + ebr + exec + unclassified == latency`（±1 µs） |
| 3 | 未分類の可視性 | 走査されない要求が `unclassified` に全額入る |
| 4 | 既存 60 テスト | 合格 |
| 5 | 比較 | 同一実行で従来 `ebr` ≈ `driverWait + ebrWait` になること（今回の推定の検証） |

## §8 Owner 推奨判断の反映（2026-11-11）

- 50 ms 閾値と N_req=59 は**判定用として不変**。50 ms は「reclaim ドライバ待ちの性能目標・診断閾値」と位置づける（保証ではない）。
- 3 分類の実装は承認対象。`firstScanTs` の追加が必須（「reclaim が動かなかった」と「reader が残った」を分離する唯一の手段）。
- 未分類を 0 で埋めない。`consistencyErrors` に順序不整合・時刻欠損を記録する。
- 関係式: `T_total = T_driverWait + T_ebrWait + T_executionWait + T_unclassified`。
- 実装後のゲート: ビルド → テスト → 同一実行計測（`globalEpoch` 推移・`publishEpoch` 回数・reclaim 呼出し回数・enqueue 時刻・reader 状態を 1 実行から取得）を別々に扱う。
- **今回のステータス**: 設計提出まで。実装は未着手。
