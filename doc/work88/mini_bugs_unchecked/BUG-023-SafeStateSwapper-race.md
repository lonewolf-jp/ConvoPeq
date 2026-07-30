# BUG-023: SafeStateSwapper tryReclaim() move-path と swap() のリングバッファ競合

## 発見日
2026-07-26

## ファイル
`src/SafeStateSwapper.h:103-131 (swap), 201-272 (tryReclaim)`

## 問題
`tryReclaim()` の move-path（head のエントリを tail へ移動するパス、lines 252-268）と
`swap()`（lines 113-130）がリングバッファの同じスロットに同時書き込みする可能性がある。

### 競合シナリオ

`tryReclaim()` が head のエントリを reclaim できない場合（epoch がまだ安全でない）、
以下の move-path を実行する:
1. head を進める（line 252: `publishAtomic(head, (h+1) % kMaxRetired, release)`）
2. tail を読む（line 254: `consumeAtomic(tail, acquire)`）
3. 空きスロット `t`（旧 tail）にエントリを書き込む（line 263）
4. tail を進める（line 265）

`swap()` は以下の処理を行う:
1. tail を読む（line 113: `consumeAtomic(tail, acquire)`）
2. 空きチェック（line 116: `next == head`）
3. 空きスロット `t`（tail）にエントリを書き込む（line 126）
4. tail を進める（line 130）

### 競合の発生

両関数が異なるスレッドで並行実行された場合:

1. `tryReclaim` head 進める → head = h+1（ring buffer に空きができる）
2. `swap` tail を読む → t = T（空きスロット）
3. `swap` retiredBuffer[T] に書き込む（line 126）
4. `tryReclaim` tail を読む → t = T（まだ古い値、step 7 未完了）
5. `tryReclaim` retiredBuffer[T] に書き込む（line 263）← **`swap` のエントリを上書き！**
6. `tryReclaim` tail を T+1 に進める（line 265）
7. `swap` tail を T+1 に進める（line 130、同じ値）

結果: `swap()` が書き込んだ ConvolverState エントリが消失する → **メモリリーク**

### リスク評価
- **重大度**: HIGH — ConvolverState のメモリリーク（MKL メモリを含む、1エントリ数十MBの可能性）
- **発生頻度**: 低〜中 — タイミング依存（tryReclaim が 1ms 周期、swap が Message Thread 経由）
- **影響**: 長時間動作時のメモリ消費増加、最悪の場合 OOM

### 修正方針
以下のいずれか:
1. `tryReclaim` の move-path で `compareExchangeAtomic` を使って tail の CAS を行い、
   `swap` と排他する
2. `swap` で `compareExchangeAtomic` を使って tail の CAS を行い、
   `tryReclaim` の move-path と排他する
3. move-path を削除し、reclaim 不可のエントリは常にフォールバックキューに送る
   （フォールバックキューは `std::mutex` で保護済みのため安全）
