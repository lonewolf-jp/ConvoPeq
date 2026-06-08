# BUG-03 DeferredDeletionQueue::reclaim() トレース検証報告書

**作成日**: 2026-06-08
**対象関数**: `DeferredDeletionQueue::reclaim(uint64_t minReaderEpoch)`
**検証方法**: 紙上トレース（Vyukov bounded MPMC queue アルゴリズム照合）
**並行パターン検証**: CAS競合・ABA・wrap-around・starvation

---

## 目次 {#toc}

1. [アルゴリズム概要](#sec1)
2. [現状のバグ](#sec2)
3. [修正案](#sec3)
4. [トレース検証シナリオ](#sec4)
5. [並行性検証](#sec5)
6. [結論](#sec6)

---

## 1. アルゴリズム概要 {#sec1}

### 1.1 Vyukov bounded MPMC queue

本キューは Dmitry Vyukov の bounded MPMC (Multi-Producer Multi-Consumer) キューをベースとする。

**核心理念**:

- リングバッファ (`ringBuffer[0..kQueueSize-1]`) + シーケンス番号 (`sequences[0..kQueueSize-1]`)
- `enqueuePos`, `dequeuePos` は単調増加カウンタ（決して減少しない）
- スロット選択は `pos & kMask` で行う（kQueueSize は2のべき乗）
- シーケンス番号でスロットの状態を管理 (解放/データ準備完了/再利用可能)

### 1.2 シーケンス番号の状態遷移

```
初期状態:   sequences[i] = i                    (スロットiは空、位置iからの書き込み可能)
  ↓
enqueue:   CAS(enqueuePos, pos, pos+1) 成功後
           → データ書き込み
           → sequences[pos & mask] = pos + 1   (スロット占有、読み取り可能)
  ↓
dequeue:   CAS(dequeuePos, deqPos, deqPos+1) 成功後
           → データ読み取り・削除
           → sequences[deqPos & mask] = deqPos + kQueueSize  (スロット解放、再利用可能)
```

**状態判定式**:

- `seq == pos`: スロット空き (enqueue可能)
- `seq == pos + 1`: データ準備完了 (dequeue可能)
- `seq == pos - kQueueSize + 1`: スロット解放済み (dequeue完了)

### 1.3 デキュー側のデータ有無判定

```cpp
const uint32_t seq = sequences[scanPos & kMask];
const intptr_t diff = seq - (scanPos + 1);
if (diff != 0) break; // Empty
```

- `diff == 0`: スロットにデータあり、処理可能
- `diff != 0`: まだデータなし、または解放済み

---

## 2. 現状のバグ {#sec2}

### 2.1 問題箇所

`reclaim()` の CAS 成功後ブロック (lines 143-148):

```cpp
if (convo::compareExchangeAtomic(dequeuePos,
                                 deqPos,                    // expected (CAS成功後も更新されない!)
                                 static_cast<uint32_t>(deqPos + 1),
                                 std::memory_order_release,
                                 std::memory_order_acquire))
{
    // ... エントリ削除 ...
    convo::publishAtomic(seq_atom, scanPos + kQueueSize, std::memory_order_release);

    scanPos = deqPos;  // ← deqPos は CAS前の旧値のまま!
    scanned = 0;
}
```

### 2.2 原因

`convo::compareExchangeAtomic` は `std::atomic_compare_exchange_strong_explicit` を使用。C++標準仕様により:

- **成功時**: `expected` パラメータ (`deqPos`) は **更新されない**
- 失敗時: `expected` に現在値が書き込まれる

CAS 成功後、`dequeuePos` は `deqPos + 1` に進んだが、ローカル変数 `deqPos` は `deqPos` のまま。`scanPos = deqPos` で解放済みスロットを指す。

### 2.3 トレース: バグの可視化

```
前提: kQueueSize=8, dequeuePos=5, deqPos(局所変数)=5

CAS成功: dequeuePos: 5 → 6
         deqPos:     5 (変わらず!)

scanPos = deqPos = 5  (解放済みスロット5を指す)
seq_atom[5] は解放時に 5 + 8 = 13 に設定済み
diff = 13 - (5 + 1) = 7 ≠ 0 → break!

→ 1回の reclaim() で最大1エントリしか解放できない
```

---

## 3. 修正案 {#sec3}

### 3.1 修正

```cpp
// Before (BUG):
scanPos = deqPos;

// After (FIX):
++deqPos;          // dequeuePos の新値 (deqPos+1) に追従
scanPos = deqPos;
```

### 3.2 正当性の根拠

CAS成功後、`dequeuePos` は確実に `deqPos + 1` である。

- CAS は不可分操作: 複数スレッドが同時に CAS しても1つだけ成功する
- 自スレッドが CAS 成功した時点で、`dequeuePos == deqPos + 1` が保証される
- 他スレッドの CAS は失敗しており、`dequeuePos` は自スレッドの書き込みのみで変更されている
- したがって `++deqPos` (= deqPos + 1) は `dequeuePos` と一致する

### 3.3 既存の類似パターン

`drainAllUnsafe()` は既に同様のパターンを使用:

```cpp
if (convo::compareExchangeAtomic(dequeuePos, pos, pos + 1, ...)) {
    // ... 削除 ...
    pos++;  // ← これと同じパターン!
}
```

---

## 4. トレース検証シナリオ {#sec4}

### 4.1 前提条件

全シナリオで `kQueueSize = 8`（簡略化）、`kMask = 7` とする。

### 4.2 シナリオ1: 5件連続 enqueue → reclaim (正常系)

**設定**: minReaderEpoch = 100（全エントリ削除可能）

**enqueue 5件**:

```
enqueuePos: 0→1→2→3→4→5
dequeuePos: 0

slot 0: seq=0→CAS→write→seq=1
slot 1: seq=1→CAS→write→seq=2
slot 2: seq=2→CAS→write→seq=3
slot 3: seq=3→CAS→write→seq=4
slot 4: seq=4→CAS→write→seq=5
```

**reclaim() [修正後]**:

```
deqPos=0, scanPos=0, scanned=0

[iter1] scanPos=0: seq[0]=1, diff=1-(0+1)=0 ✓
  canDelete=true, scanPos==deqPos(0==0) ✓
  CAS(0→1)成功
  → delete slot0, seq[0]=0+8=8
  ++deqPos → deqPos=1
  scanPos=1, scanned=0

[iter2] scanPos=1: seq[1]=2, diff=2-(1+1)=0 ✓
  canDelete=true, scanPos==deqPos(1==1) ✓
  CAS(1→2)成功
  → delete slot1, seq[1]=1+8=9
  ++deqPos → deqPos=2
  scanPos=2, scanned=0

[iter3] scanPos=2: seq[2]=3, diff=3-(2+1)=0 ✓
  canDelete=true, scanPos==deqPos(2==2) ✓
  CAS(2→3)成功
  → delete slot2, seq[2]=2+8=10
  ++deqPos → deqPos=3
  scanPos=3, scanned=0

[iter4] scanPos=3: seq[3]=4, diff=4-(3+1)=0 ✓
  canDelete=true, scanPos==deqPos(3==3) ✓
  CAS(3→4)成功
  → delete slot3, seq[3]=3+8=11
  ++deqPos → deqPos=4
  scanPos=4, scanned=0

[iter5] scanPos=4: seq[4]=5, diff=5-(4+1)=0 ✓
  canDelete=true, scanPos==deqPos(4==4) ✓
  CAS(4→5)成功
  → delete slot4, seq[4]=4+8=12
  ++deqPos → deqPos=5
  scanPos=5, scanned=0

[iter6] scanPos=5: seq[5]=5*(初期値), diff=5-(5+1)=-1≠0
  → break (キュー空)

結果: 5件すべて回収 ✓
enqueuePos=5, dequeuePos=5
```

**判定**: ✅ 全件回収成功

### 4.3 シナリオ2: EBRにより一部エントリが削除不可

**設定**: slot0〜4 の epoch が [1, 3, 5, 7, 9], minReaderEpoch=6

```text
slot0: epoch=1, isOlder(1,6)=true  → 削除可
slot1: epoch=3, isOlder(3,6)=true  → 削除可
slot2: epoch=5, isOlder(5,6)=true  → 削除可
slot3: epoch=7, isOlder(7,6)=false → 削除不可
slot4: epoch=9, isOlder(9,6)=false → 削除不可
```

**reclaim() [修正後]**:

```
deqPos=0, scanPos=0

[iter1] scanPos=0: canDelete=true → CAS成功 → delete slot0
  ++deqPos=1, scanPos=1, scanned=0

[iter2] scanPos=1: canDelete=true → CAS成功 → delete slot1
  ++deqPos=2, scanPos=2, scanned=0

[iter3] scanPos=2: canDelete=true → CAS成功 → delete slot2
  ++deqPos=3, scanPos=3, scanned=0

[iter4] scanPos=3: canDelete=false
  → else: scanPos=3, canDelete=false
  → else節: scanPos=4, scanned=1

[iter5] scanPos=4: canDelete=false
  → else: scanPos=5, scanned=2

[iter6] scanPos=5: seq=5, diff≠0 → break

結果: 3件回収 (0,1,2), 2件保留 (3,4)
dequeuePos=3
```

**判定**: ✅ EBR条件を正しく尊重。削除可能なエントリのみ回収

### 4.4 シナリオ3: Wrap-around

**前提**: kQueueSize=8, enqueuePos=12, dequeuePos=8（slot[8&7=0] が次, slot[11&7=3] が最新）

**状態**:

```
slot[8&7=0]: seq=8+1=9 (データ準備完了)
slot[9&7=1]: seq=9+1=10 (データ準備完了)
slot[10&7=2]: seq=10+1=11 (データ準備完了)
slot[11&7=3]: seq=11+1=12 (データ準備完了)
slot[12&7=4]: seq=4 (初期状態、空)
```

**reclaim() [修正後]**:

```
deqPos=8, scanPos=8

[iter1] scanPos=8: seq[0]=9, diff=9-(8+1)=0 ✓
  CAS(8→9)成功 → delete slot[0]
  seq[0]=8+8=16 (再利用可能)
  ++deqPos=9, scanPos=9, scanned=0

[iter2] scanPos=9: seq[1]=10, diff=10-(9+1)=0 ✓
  CAS(9→10)成功 → delete slot[1]
  seq[1]=9+8=17
  ++deqPos=10, scanPos=10, scanned=0

[iter3] scanPos=10: seq[2]=11, diff=11-(10+1)=0 ✓
  CAS(10→11)成功 → delete slot[2]
  seq[2]=10+8=18
  ++deqPos=11, scanPos=11, scanned=0

[iter4] scanPos=11: seq[3]=12, diff=12-(11+1)=0 ✓
  CAS(11→12)成功 → delete slot[3]
  seq[3]=11+8=19
  ++deqPos=12, scanPos=12, scanned=0

[iter5] scanPos=12: seq[4]=4, diff=4-(12+1)=-9≠0
  → break (キュー空)

結果: 4件すべて回収、dequeuePos=12
```

**判定**: ✅ Wrap-around でも正しく動作

### 4.5 シナリオ4: キュー空からの reclaim

**設定**: enqueuePos=dequeuePos=42

```
deqPos=42, scanPos=42
seq[42&7=2]=2, diff=2-(42+1)=-41≠0 → break
```

**判定**: ✅ 即 break、何もせず終了

### 4.6 シナリオ5: kMaxScan 制限

**設定**: slot0 のみ削除可、slot1〜1023 は削除不可

**reclaim() [修正後]**:

```
deqPos=0

[iter1] scanPos=0: delete slot0 → ++deqPos=1, scanPos=1, scanned=0

[iter2-1025] scanPos=1〜1024: canDelete=false
  → すべて else 節: scanPos++, scanned++
  scanned が kMaxScan(1024) に達する → break

結果: 1件回収、即時 break (正常。他スレッドの epoch advance 待ち)
```

**判定**: ✅ kMaxScan 制限は正常に機能

### 4.7 シナリオ6: 競合 — 2スレッド同時 reclaim

**設定**: slot0 にデータ、2スレッド同時に reclaim

```
両スレッド: deqPos=0, scanPos=0, seq[0]=1, diff=0

スレッドA: CAS(0→1)成功
スレッドB: CAS(0→1)失敗 → expected が 1 に更新される

スレッドA: delete slot0, seq[0]=8, ++deqPos=1, scanPos=1
スレッドB: deqPos = consumeAtomic(dequeuePos) = 1
            scanPos = 1, scanned = 0

スレッドA: scanPos=1, seq[1]=...(データなし)→break
スレッドB: scanPos=1, seq[1]=...(データなし)→break
```

**判定**: ✅ 競合解決。CAS失敗側は dequeuePos を再読取して退避

---

## 5. 並行性検証 {#sec5}

### 5.1 ABA

**結論**: ABA問題は発生しない。

理由:

- CAS の expected 値 (`deqPos`) は単調増加する位置カウンタ
- 同一スロットが解放→再確保→再解放されても、CAS expected は異なる値
- 例: slot0 が位置0, 8, 16, 24,... で再利用される。CAS expected は単調増加する位置値であり、同じ位置でCASが誤成功することはない

### 5.2 Slot Skip

**結論**: Slot skip は発生しない。

理由:

- `scanPos == deqPos` チェックにより、CAS は必ず現在の dequeue 先頭でのみ実行
- deqPos は必ず dequeuePos に追従する（CAS成功時は ++deqPos、失敗時は consumeAtomic）
- スロットを飛ばす経路は存在しない

### 5.3 Starvation

**結論**: 修正によって starvation リスクは低下する。

理由:

- 現状: 1回の reclaim() で1エントリのみ → 多数のエントリが滞留 → starvation
- 修正後: 1回の reclaim() で複数エントリ処理可能 → starvation リスク低減
- enqueue と reclaim は異なる atomic (enqueuePos / dequeuePos) を操作するため相互ブロックなし

### 5.4 FIFO 順序

**結論**: FIFO 順序は維持される。

理由:

- `scanPos == deqPos` チェックにより、先頭エントリのみ削除
- deqPos は常に dequeuePos に追従する
- 先頭以外のエントリを削除することはない

### 5.5 Sequence 範囲

各スロットの sequence 値は `[0, pos + kQueueSize, pos + 2*kQueueSize, ...]` と増加する。uint32_t の範囲内で運用可能。

- kQueueSize = 4096
- uint32_t max = 4,294,967,295
- 最大ラップ回数 ≈ 4,294,967,295 / 4096 ≈ 1,048,576 回
- 十分な余裕

---

## 6. 結論 {#sec6}

### 6.1 バグ判定

| 項目 | 判定 |
| --- | --- |
| 現状のバグ存在 | ✅ **確定** — CAS成功後に deqPos が更新されず、scanPos が解放済みスロットを指す |
| 影響 | 1回の reclaim() で最大1エントリしか解放できない |
| 発現条件 | 複数エントリが同時に削除可能な状態 → 解放遅延・キュー溢れリスク |

### 6.2 修正の妥当性

| 観点 | 評価 |
| --- | --- |
| `++deqPos` の正当性 | ✅ CAS成功後に dequeuePos = deqPos + 1 は確定。`++deqPos` (= deqPos+1) は正しい |
| 既存パターンとの一致 | ✅ `drainAllUnsafe()` の `pos++` と同一パターン |
| ABA | ❌ 発生しない (CAS expected は単調増加カウンタ) |
| Slot skip | ❌ 発生しない (scanPos == deqPos チェックあり) |
| Starvation | ✅ リスク低減（より多くのエントリを解放） |
| FIFO | ✅ 維持される |

### 6.3 最終判定

**修正 `++deqPos` は安全かつ正しい。実施推�む。**

変更は以下の1行追加のみ:

```cpp
// Before (line 146):
scanPos = deqPos;

// After:
++deqPos;
scanPos = deqPos;
```

### 6.4 検証手順

修正後、以下を実行:

1. `Build_CMakeTools (Debug)` — コンパイル確認
2. `Build_CMakeTools (Release)` — リリースビルド確認
3. `Strict Atomic Dot-Call Scan` — atomic 規約違反チェック
4. `work21 EpochDomain CI Gate` — epoch 関連ゲート通過確認
5. CLI Smoke Test — 基本動作確認

---

*作成日: 2026-06-08*
*トレース検証: GitHub Copilot (DeepSeek V4 Flash)*
