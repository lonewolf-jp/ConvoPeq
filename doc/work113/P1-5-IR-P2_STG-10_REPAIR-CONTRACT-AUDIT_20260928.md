# STG-10 Repair Contract Audit（2026-09-28）

> **Verdict: Concrete Defect = PROVEN / Repair Contract = PROVEN**
> **本書は read-only 監査である。production = 0 / test = 0 / CMake = 0 / ConvoPeq.md = 0 / commit = 0 / push = 0。**
> **Owner が Repair Contract Audit 以外の段階を明示的に GO しない限り、implementation には進まない。**

---

## 1. Authority

Owner 指定: `ConvoPeq(11).md == ConvoPeq.md`（統一 authority）。実際の source 検証は
repo 上の最新 `ConvoPeq.md` を基準に実施した。

| 項目 | 値 |
| --- | --- |
| HEAD | `13a80d66e8e58fbcb66e71eb162f8e95fe134eeb` |
| SHA-256 | `B15640ADCE15A38801DA8192652876EE4CA8EC6113247EEF9EB93E9EA754D7F3` |
| size | 5,642,888 B |
| Generated | 2026-09-28 22:19:21 |
| baseline Generated | 2026-09-28 22:19:21（一致） |
| NEWER_SRC_COUNT | **0** |
| STATUS | **FRESH**（`--check` exit 0） |

**FRESH。STOP 条件「latest authority が FRESH でない」には該当しない。**
監査前後で `git status --porcelain -- src CMakeLists.txt` は空（production 未変更）。

---

## 2. Quarantine semantics（Q1〜Q6 — 推測を使わず caller/callee/state transition で確定）

### 用語の分離（重要）

本 codebase には**別物**の quarantine が 2 系統ある。**STG-10 は前者に限る。**

| | 系統① Reader quarantine（**STG-10 の対象**） | 系統② DSP handle quarantine（**対象外**） |
| --- | --- | --- |
| 状態 holder | `EpochDomain::ReaderSlot::quarantineFlags` | `DSPQuarantineManager` / `QuarantineService` |
|  Exclusion 先の判定 | **`getMinReaderEpoch()` — 解放可否の安全境界そのもの** | `worldAuthority_.lifetime()` の lane |
| flag | `kQuarantinedFlag=0x01` / `kPendingQuarantineFlag=0x02` | `QuarantineReason` / lane |
|  主要 entry | `IEpochProvider::quarantineReader` | `submitQuarantine` / `executeQuarantine` |
| 判定 | **STG-10 の Concrete Defect** | STG-6/8 領域。**CLOSED。再 OPEN しない** |

### Q1. quarantine された reader slot は恒久的に再利用禁止なのか → **いいえ（No）**

**証拠（allocation 側の実装）**:

- `EpochDomain.h:87-114` `reserveReaderThread` — 「前回キャッシュ済み tid を観測」して
  **同一 slot の再利用を専用に存在する関数**。guard は `registrationClosed_`（`:92`）のみ。
- `RCUReader.h:166-172` — `exit()` が保存した `preferredThreadId` を毎回 `reserveReaderThread` に渡す。
  → **同一 slot の再利用は通常運用の一等経路**。
- `EpochDomain.h:48-85` `registerReaderThread` — guard は `registrationClosed_`（`:50`）のみ。
- `registrationClosed_` を立てる `closeReaderRegistration()`（`:628`）の production 呼出し元は
  `AudioEngine.Processing.ReleaseResources.cpp:274` と `AudioEngine.CtorDtor.cpp:236` の
  **shutdown のみ**。

**証拠（意味論の側）**: `IEpochProvider.h:82` は `unquarantineAllReaders` を
「★ Shutdown専用」と規定する。**恒久封印が意図なら clear 機構は不要**。clear が
「Shutdown 専用」と明記されていること自体が、封印ではなく**「shutdown まで保持する一時隔離」**
であることを示す。

**判定: Q1 = No。slot の再利用は許可されている。**

### Q2. quarantine は現在の reader instance だけを隔離するのか → **Yes（設計意図）**

すべての仕様コメントの主語は「**その Reader**」である。

| 位置 | 記述 |
| --- | --- |
| `IEpochProvider.h:76` | 「★ stuck Reader を quarantined にマーク（**killしない**）」 |
| `IEpochProvider.h:79` | 「戻り値: true=**即座隔離成功**, false=**遅延隔離（exitReader待ち）**」 |
| `AudioEngine.Timer.cpp:1798` | 「この Reader を quarantine（**この** stuck Reader の epoch を safe-epoch 計算から除外）」 |
| `EpochDomain.h:171` | 「**この Reader** は exit 後も quarantined フラグにより getMinReaderEpoch から除外される」 |
| `EpochDomain.h:276-278` | 「CAS を使うことで flags は 0x00→0x02 / 0x00→0x01 / 0x02→0x01 の**原子的遷移のみ**許し」 |

**判定: Q2 = Yes。隔離対象は「占有（occupancy）」であり「slot」ではない。**

### Q3. quarantine 後に同じ readerIndex を再入場することは仕様上許されるか → **Yes（むしろ通常運用で必然）**

- RT reader は `ConvolverProcessor.Runtime.cpp:240`（`RCUReaderGuard` を `process()` 内に置く）で
  **毎 audio block** enter/exit する。
- `RCUReader.h:134-140` `exit()`: `activeThreadId` を `-1` 化し、`epochProvider->exitReader(tid)` を
  呼んだ後 `preferredThreadId` に `tid` を保存する。
- `RCUReader.h:167-172` `acquireThreadSlot()`: `preferredThreadId` があれば
  `reserveReaderThread(preferredTid)` で**同じ tid を再取得**する。
- `EpochDomain.h:116-141` `enterReader` は `quarantineFlags` を**参照しない**（`rg` 実測 0 件）。

**判定: Q3 = Yes。再入場は禁止されていない。かつ同一 tid の再入場は毎 block 起きている。**

### Q4. quarantine 後に register / reserve が同じ slot を再利用することは許されるか → **Yes（構造的に可能）**

隔離済みかつ非参加の slot は `exitReader` が `epoch = kInactiveEpoch`（`:168`）を
publish 済みなので、allocation の選択基準 `epoch == kInactiveEpoch` を**満たす**。

| allocation | 選択基準 | quarantineFlags の検査 |
| --- | --- | --- |
| `registerReaderThread` `:52-61` | `epoch: kInactive → kReserved` の CAS | **無し（実測 0 件）** |
| `reserveReaderThread` `:95-103` | `epoch: kInactive → kReserved` の CAS | **無し（実測 0 件）** |

**判定: Q4 = Yes。隔離済み slot は新規 reader に貸出し可能である。**
この 2 関数が `quarantineFlags` を見ないことが、Q1（再利用が許される）と
Q2（隔離は instance 単位）が実装で接続されていない**唯一の結合欠落**である。

### Q5. `unquarantineAllReaders()` が shutdown 時だけ呼ばれる設計意図 → **「通常運行中は隔離を維持し、shutdown で一括解放」**

| 証拠 | 内容 |
| --- | --- |
| `IEpochProvider.h:82` | 「★ **Shutdown専用**: 全quarantined Readerを解放」 |
| `AudioEngine.Processing.ReleaseResources.cpp:431-435` | 「★★★ PR2: Quarantine **全スロット強制解放（シャットダウン専用）** / この時点で **GracefulDrain が activeReaderCount==0 を確認済み**」 |
| 呼出し元（production） | `ReleaseResources.cpp:435` の **1 箇所のみ** |

前提条件が「activeReaderCount==0 を確認済み」であること（＝**全 reader が非参加**）に
注目する。これは Q2 の裏付けである。**隔離保持が前提とされているのは
「まだ reader が参加しているうち」であり、参加者が終われば解放してよいという設計。**

**判定: Q5 = 「shutdown まで保持する一時隔離」。slot の恒久封印ではない。**

### Q6. `kQuarantinedFlag` / `kPendingQuarantineFlag` の状態機械

実コードから機械を**逆算**した結果（コードに存在しない名称は導入していない）:

```text
  flags 0x00                 0x02 pending                0x01 quarantined
     │                          │                            │
     │  quarantineReader          │  quarantineReader           │
     │  depth>0 (:319 CAS)        │  depth==0 (:294 CAS)        │
     ├──────────────────────────►│◄───────────────────────────┤
     │                           │                            │
     │                           │  exitReader 最終 exit        │
     │                           │  (:178 CAS 0x02→0x01)      │
     │                           ├───────────────────────────►│
     │                           │                            │
     │                           │              unquarantineAllReaders
     │◄───────────────────────────────────────────────────────┤
                                    (:332 publish 0x00 / shutdown のみ)
```

生成が禁止される遷移（`EpochDomain.h:276-278` の設計意図、`BUG-049` の修正方針）:
`0x03`（quarantined|pending）の生成禁止。現状の 3 系統 CAS はこれを満たしている。

**欠落している遷移**: 「隔離済み slot の再占有時の 0x01 → 0x00」。**これが本 STG の欠陥の正体**（§6.2）。

### 2.7 決定的な意味論（Q1〜Q6 の収束）

```text
仕様意味論（確定）:
  quarantine は「reader の占有(occupancy)」に対する隔離であり、slot に対する封印ではない。
  したがって:
    (a) 参加中の reader（depth>0）は、如何なる隔離状態でも安全境界から除外してはならない
    (b) 非参加の reader（depth==0）だけが隔離による除外の対象になる
    (c) slot が再占有されたとき、その占有は「新しい reader」であり、旧占有の隔離は継承されない
```

**実装は (a) を満たしていない。** §4 で証明する。

---

## 3. ReaderSlot state machine

`EpochDomain::ReaderSlot`（`EpochDomain.h:567-583`）の実フィールドから再構成した。
`epoch` / `depth` / `quarantineFlags` は全ファイルで `std::atomic`、
`ownerTag` は非 atomic（CAS 排他下のみ書き込み、`stale read 許容` とコメント）。

### 3.1 状態表（コードに実在する状態のみ）

| # | 状態 | poch | depth | quarantineFlags | 到達 writer | 避移元 | 状態の clear 条件 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | **Unregistered / Inactive** | `kInactive` | 0 | `0x00` | ctor `:32-35` | — | （allocation されるまで） |
| S1 | **Reserved**（取得済み・未入場） | `kReserved` | 0 | `0x00` | `registerReaderThread:57-66` / `reserveReaderThread:98-110` | S0, S3 | `enterReader` |
| S2 | **Active**（critical section 内） | `currentEpoch` | ≥1 | `0x00` | `enterReader:127,130` | S1, S3, S6 | `exitReader` |
| S2n | **Active-nested** | `currentEpoch` | ≥2 | `0x00` | `enterReader:130`（`previousDepth>0`） | S2 | `exitReader`（`previousDepth>1` で早期 return） |
| S2t | **Enter 遷移中（transient）** | `currentEpoch` | 0 | `0x00` | `enterReader:127` の後 `:130` の前 | S1/S3 | `enterReader:130` |
| S3 | **Idle-occupied**（exit 済・占有継続） | `kInactive` | 0 | `0x00` | `exitReader:168` | S2, S2n | 再入場 or 新規割当 |
| S4 | **QuarantinePending** | `kInactive` or `currentEpoch` | >0 | `0x02` | `quarantineReader:319`（depth>0） | S2 | `exitReader:178` → S5 |
| S5 | **Quarantined（非参加）** | `kInactive` | 0 | `0x01` | `quarantineReader:294,302`（depth==0）/ `exitReader:178` | S0, S3, S4 | `unquarantineAllReaders:332`（shutdown のみ） |
| **S6** | **Quarantined かつ Live（禁止状態）** | `currentEpoch` | **>0** | `0x01` | **`enterReader:130`（S5/S3/S1 から）** ← **本欠陥** | S1, S3, S5 | **解放機構なし** |
| S7 | **Quarantined-slotted Reserved** | `kReserved` | 0 | `0x01` | `registerReaderThread:57` / `reserveReaderThread:98`（S5 を受理） | S5 | `enterReader` → **S6** |

### 3.2 コードに存在しない状態（Owner §3「勝手に導入しない」）

- **`Reclaimed / released` 状態は `ReaderSlot` に存在しない。**
  slot を解放する API（`releaseReader` / `freeSlot` / `unregisterReaderThread`）は
  `IEpochProvider` に**定義されていない**。`readers` は `std::array<ReaderSlot, 64>` の固定長で、
  要素の寿命は `EpochDomain` と同命。唯一の「戻し」は `exitReader:168` の
  `epoch = kInactiveEpoch` であり、これは S3 への遷移であって reclaimed ではない。
- **`reserved` という reader 概念は無い。** `kReservedEpoch` は allocation 中の
  占有競合を防ぐための**番兵値**であって、独立した reader 状態ではない（S1 として本書が名付けた
  のは epoch/depth/flags の組から機械的に導いた名称）。
- **generation / epoch-generation と reader slot は結線していない。**
  `epochGeneration_`（`:611`）は `publishEpoch()` 回数であり、reader slot の世代ではない。

### 3.3 禁止状態 `quarantineFlags != 0 && depth > 0` は本当に禁止か

**Owner §3 の問いに対する答え: 「コメント上は禁止、実装上は防がれていない」。**

| 検査 | 結果 |
| --- | --- |
| コメント上の主張 | **禁止**。`EpochDomain.h:220-224`「`kQuarantinedFlag` 設定時は `depth==0` が不変条件: 即座 quarantine: depth==0 でのみ設定 / 遅延 quarantine: exitReader で depth:1→0 後に昇格」 |
| `verifyReaderInvariants` の主張 | **禁止**。`:369-372` `isQuarantined && depth==0 → epoch inactive/reserved`、`:377` `isPending → depth>0` |
| `getMinReaderEpoch` の Debug assert | **禁止**。`:228-229` `assert(depth == 0 && "quarantined reader must have depth==0")` |
| **全 writer からの証明** | **証明できない** |
| **全 reader からの証明** | **証明できない** |

**writer 側（全 4 系統）**:

| writer | quarantineFlags を読むか |
| --- | --- |
| `enterReader:130`（`depth` を無条件に `fetchAdd(+1)`） | **いいえ**。`enterReader` 内の `quarantineFlags` 参照は **0 件** |
| `exitReader:152,168,178` | はい（`:174`）。ただし depth は `fetchSub` 済み |
| `registerReaderThread:57` / `reserveReaderThread:98` | **いいえ**。両関数内の `quarantineFlags` 参照は **0 件** |
| `unquarantineAllReaders:332` | はい（writer 自身） |

**reader 側**: `exitReader:174` / `getMinReaderEpoch:225` / `quarantineReader:314` /
`quarantinedReaderCount:344` / `verifyReaderInvariants:358`。**これらのどれも
「フラグ立ち ∧ depth>0」を検出・拒否しない**（`verifyReaderInvariants` は
`isQuarantined && depth==0` を条件にしており、**S6 は検査対象外**）。

**結論**: S6 を生成する writer が存在し、それを拒否する reader も存在しない。
**禁止状態であることはコメント上の宣言に過ぎない。**
かつ `verifyReaderInvariants` は**呼出し元 0 件**（`rg` 実測: 定義 `:353` のみ）で、
Debug 側の独立検出器も死んでいる。**唯一の稼働する検出器は `getMinReaderEpoch:228` の assert** であり、
これは Release で消滅する。

---

## 4. flag / depth / epoch writer・reader audit（全量）

`src/core/EpochDomain.h` に対する実測（`rg`）。

### 4.1 `quarantineFlags`

| 種別 | 行 | 内容 |
| --- | --- | --- |
| **WRITE** | `:35` | ctor 初期化 `publishAtomic(..., 0, release)` |
| **WRITE** | `:178` | `exitReader` — CAS `0x02 → 0x01`（pending → quarantined） |
| **WRITE** | `:294` | `quarantineReader` — CAS `0x02 → 0x01`（depth==0 分岐①） |
| **WRITE** | `:302` | `quarantineReader` — CAS `0x00 → 0x01`（depth==0 分岐②） |
| **WRITE** | `:319` | `quarantineReader` — CAS `0x00 → 0x02`（depth>0 分岐） |
| **WRITE** | `:332` | `unquarantineAllReaders` — `publishAtomic(..., 0, release)`（**shutdown のみ**） |
| READ | `:174` | `exitReader` — pending 判定 |
| READ | `:225` | `getMinReaderEpoch` — **除外判定** |
| READ | `:314` | `quarantineReader` — 既に隔離済みなら早期 return |
| READ | `:344` | `quarantinedReaderCount` |
| READ | `:358` | `verifyReaderInvariants`（**呼出し元 0 件**） |

**WRITE 6 箇所 / READ 5 箇所。うち `enterReader` / `registerReaderThread` /
`reserveReaderThread` は WRITE も READ も 0。**

### 4.2 `depth`

| 種別 | 行 | 内容 |
| --- | --- | --- |
| WRITE | `:33` | ctor `publishAtomic(0, release)` |
| WRITE | `:64-66` | `registerReaderThread` — epoch CAS 成功後 `publishAtomic(0, release)` |
| WRITE | `:108-110` | `reserveReaderThread` — 同上 |
| WRITE | `:130` | `enterReader` — `fetchAddAtomic(1, acq_rel)`（**無条件。flag を見ない**） |
| WRITE | `:152` | `exitReader` — `fetchSubAtomic(1, acq_rel)` |
| WRITE | `:157` | `exitReader` — underflow 回復 `publishAtomic(0, release)` |
| READ | `:228` | `getMinReaderEpoch` — **assert 内のみ（Release で消滅）** |
| READ | `:234` | `getMinReaderEpoch` — 通常経路 |
| READ | `:256` | `activeReaderCount` |
| READ | `:286` | `quarantineReader` — 分岐選択（CAS で再検証しない） |
| READ | `:359` | `verifyReaderInvariants`（死コード） |
| READ | `:448` | `getReaderSlotDetail` |
| READ | `:489` | `detectStuckReaders` — 3 pass すべて `depth>0` を要求 |

### 4.3 `epoch`

| 種別 | 行 | 内容 |
| --- | --- | --- |
| WRITE | `:32` | ctor `publishAtomic(kInactive, release)` |
| WRITE | `:57-61` | `registerReaderThread` — CAS `kInactive → kReserved` |
| WRITE | `:98-103` | `reserveReaderThread` — CAS `kInactive → kReserved` |
| WRITE | `:127` | `enterReader` — `publishAtomic(currentEpoch, release)`（**depth++ より前**、BUG-050 の HB 修正） |
| WRITE | `:168` | `exitReader` — `publishAtomic(kInactive, release)` |
| READ | `:239` | `getMinReaderEpoch` |
| READ | `:360` | `verifyReaderInvariants` |
| READ | `:447` | `getReaderSlotDetail` |
| READ | `:484` | `detectStuckReaders` |

### 4.4 slot allocation / release sites

| 種別 | 位置 | 条件 |
| --- | --- | --- |
| **allocation** | `EpochDomain.h:52-61` `registerReaderThread` | `registrationClosed_==false` かつ `epoch==kInactiveEpoch` の先頭 slot |
| **allocation** | `EpochDomain.h:95-103` `reserveReaderThread` | `registrationClosed_==false` かつ `epoch==kInactiveEpoch`（指定 index） |
| **allocation 呼び出し元** | `RCUReader.h:169` / `:175`（`acquireThreadSlot`） | **RT**（`RCUReader::enter` から） |
| **release** | **存在しない** | `IEpochProvider` に release API なし。`readers` は固定長 `std::array` |
| **退避（準 release）** | `EpochDomain.h:168` `exitReader` の `epoch=kInactive` | S2 → S3。以降 allocation 対象になる |
| **一括 clear** | `EpochDomain.h:332` `unquarantineAllReaders` | shutdown のみ |

---

## 5. Path A（TOCTOU inside `quarantineReader`）

### 5.1 構造

```text
NonRT Timer:  quarantineReader(i)
  :286   d = load(slot.depth, acquire)        ← 読み（1 回だけ）
  :288   if (d == 0)
  :302     CAS(quarantineFlags, 0x00 → 0x01)  ← CAS。depth の再検証なし
                    ↑
RT audio:   enterReader(i)
  :127     store(slot.epoch, currentEpoch, release)
  :130     fetchAdd(slot.depth, +1, acq_rel)
```

**読みと CAS の間に `depth` の再検証が無い**ため、この窓に `enterReader` が入ると
`kQuarantinedFlag == 0x01` かつ `depth == 1`（S6）が生成される。

### 5.2 到達性

`detectStuckReaders` は 3 pass すべてで `depth > 0` を要求する（`:502` / `:521` / `:524`）。
したがって検出時点では `depth > 0` であり、`:286` で `d == 0` になるには
**NonRT の health-monitor → callback → Timer ディスパッチ中に RT が exit を完了**する必要がある。
この区間は NonRT hop を挟むため現実的な時間幅があり、到達可能。

### 5.3 判定

**Path A は実在する。** ただし §6 のとおり **B に吸収される**ため、
A の単独修正では根本原因は残る（Owner §6 の要求どおり明示する）。

---

## 6. Path B（quarantine 後の再入場）— **STG-10 の最重要主張**

### 6.1 Owner 指定の B-1 〜 B-7 逐項実証

#### **B-1. `RCUReader::exit()` が `preferredThreadId` を保存する → YES**

`src/core/RCUReader.h:133-141`

```cpp
// acq_rel: activeThreadId を原子的に取得（+クリア）
const int tid = convo::exchangeAtomic(activeThreadId, -1, std::memory_order_acq_rel);
if (tid >= 0)
{
    epochProvider->exitReader(tid);                                   // :137
    convo::publishAtomic(preferredThreadId, tid, std::memory_order_release);  // :140
}
```

**caller**: `RCUReaderGuard::~RCUReaderGuard()`（`RCUReader.h:197`）→ `RCUReader::exit()`。
**callee**: `EpochDomain::exitReader`（`EpochDomain.h:144`）。

#### **B-2. 次回 `enter()` が同じ slot を優先取得する → YES（毎 block）**

`src/core/RCUReader.h:159-182`

```cpp
int acquireThreadSlot() noexcept
{
    const int activeTid = convo::consumeAtomic(activeThreadId, std::memory_order_acquire);
    if (activeTid >= 0) return activeTid;                    // :162-164

    const int preferredTid = convo::consumeAtomic(preferredThreadId, acquire);  // :167
    int reservedTid = -1;
    if (preferredTid >= 0 && epochProvider->reserveReaderThread(preferredTid)) // :169 ← 同一 slot
        reservedTid = preferredTid;
    else
        reservedTid = epochProvider->registerReaderThread();  // :175
    convo::publishAtomic(activeThreadId, reservedTid, release);
    return reservedTid;
}
```

**caller**: `RCUReader::enter()`（`RCUReader.h:62`）= **RT**。
**callee**: `EpochDomain::reserveReaderThread`（`EpochDomain.h:87`）。

#### **B-3. `enterReader()` が quarantine flag を見ない → YES（0 件）**

`src/core/EpochDomain.h:116-141` — `quarantineFlags` 参照 **0 件**（`rg` 実測）。
`depth` は `:130` で無条件に `fetchAdd(+1)`。

#### **B-4. quarantine された slot が register / reserve で別 reader に再割当可能か → YES**

隔離の成立状態 S5 は `epoch == kInactiveEpoch`（`exitReader:168` が publish）かつ
`flags == 0x01`。これは allocation の選択基準（`epoch == kInactiveEpoch`）を**満たす**。

| 経路 | 成立条件 | quarantineFlags 検査 |
| --- | --- | --- |
| `reserveReaderThread(preferredTid)`（**B-2 の毎 block 経路**） | S5 は `epoch==kInactive` なので CAS **成功** | **無し（`:87-114` に 0 件）** |
| `registerReaderThread()`（新規 reader） | 同上、走査で S5 を **最初に_** 掴む可能性 | **無し（`:48-85` に 0 件）** |

**決定的な事実**: `exitReader:178` が `0x02 → 0x01` へ昇格させた直後の状態 S5 は、
`reserveReaderThread:95-103` が**成功する condition と一字一句一致**する。
**S5 は「次の enter で必ず再利用される」状態である。**
→ **Path B は race ではなく、公称経路（nominal path）である。**

#### **B-5. 再入場後に `getMinReaderEpoch()` が実際にその slot を除外する → YES**

`src/core/EpochDomain.h:225-231`

```cpp
const uint8_t flags = convo::consumeAtomic(slot.quarantineFlags, std::memory_order_acquire);
if ((flags & ReaderSlot::kQuarantinedFlag) != 0)
{
    assert(convo::consumeAtomic(slot.depth, std::memory_order_acquire) == 0
        && "quarantined reader must have depth==0");
    continue;                                    // ← depth に関係なく除外
}
```

**除外条件に `depth` が含まれていない。** S6（S5 → `enterReader`）でも
`flags == 0x01` なので `continue` が必ず取られる。
Release では `assert`（`:228`）が `NDEBUG` で消滅するため、**無言で除外される**。

#### **B-6. 除外により `minReaderEpoch` が安全境界を越えて前進し得る → YES**

`EpochDomain.h:214-248` の構造:

```cpp
uint64_t minEpoch = currentEpoch();          // :216  seed
for (const auto& slot : readers) {           // :218
    ... if (quarantined) continue;           // :226-231  ← ここでの除外
    const uint32_t depth = ...;              // :234
    if (depth == 0) continue;
    const uint64_t epoch = ...;              // :239
    if (isOlder(epoch, minEpoch)) minEpoch = epoch;   // :243-244
}
```

live reader（`epoch == E_R`、`depth == 1`）を除外すると、
`minEpoch` は **E_R ではなく seed `currentEpoch()` で確定**する（他の reader がいなければ）。

```text
具体例（既存コードの値域で成立する）:
  globalEpoch = 10
  audio thread reader: epoch = 3, depth = 1  ← 実際に critical section 内
  他の reader: なし
  → minReaderEpoch = 10   （本来は 3 でなければならず）
```

**EBR の安全条件は `entry.epoch < minReaderEpoch` のときだけ解放、である。**
`entry.epoch == 5` の entry は本来 `isOlder(5, 3) == false` で**解放されない**が、
除外により `isOlder(5, 10) == true` となり**解放される**。

#### **B-7. その値が `DeferredDeletionQueue::reclaim()` の delete 条件へ到達する → YES**

**caller/callee チェーン（production 実測）**:

```text
EpochDomain::tryReclaim()                            EpochDomain.h:386
  └─ :394  deferredDeletionQueue.reclaim(getMinReaderEpoch())      ← getMinReaderEpoch の唯一の実使用
        └─ DeferredDeletionQueue::reclaim(uint64_t minReaderEpoch)  DeferredDeletionQueue.h:110
              └─ :132   if (isOlder(entry.epoch, minReaderEpoch)) canDelete = true;
              └─ :137   if (canDelete && scanPos == deqPos)
                    └─ :138-142  CAS(dequeuePos, ...) で所有権獲得
                    └─ :143-146 if (entry.deleter && entry.ptr) entry.deleter(entry.ptr);   ← 解放
```

**判定: B-1〜B-7 すべて YES。B は source 上で完全に証明された。**

### 6.2 Path A / Path B の分離（Owner §6）

| | Path A | **Path B** |
| --- | --- | --- |
| 発生位置 | `quarantineReader:286` の read と `:294`/`:302` の CAS の間 | quarantine 成立後、次の `enter()` |
| 時間窓 | 有限（2 命令 + NonRT hop） | **不要** |
| 成立条件 | 交差した reader が**ちょうど** enter している | 同一 tid の**次の audio block**（必ず来る） |
| 到達性 | 条件的 | **必達** |
| A を直したら B は残るか | — | **残る**（B には A 毫不相关） |

**Owner §6 の判定基準を適用する: B が成立するなら B だけで Concrete Defect である。**
本監査は B の成立を §6.1 で証明した。**したがって A の修正は本欠陥の根治ではない。**
A の修正を主契約とするのは誤りである。

---

## 7. UAF lifetime proof（read-only structural proof）

Owner §10 の鎖を source 上で確認する。**再現は不要。**

### 7.1 寿命を跨越する 2 つの production 経路（matched pair）

#### 鎖① Convolver `IRState`

| 段 | 位置 | 内容 |
| --- | --- | --- |
| RT reader 取得 | `ConvolverProcessor.h:1424` `RCUReader runtimeRcuReader { m_epochDomain }` | provider は `EpochDomain` |
| RT guard | `ConvolverProcessor.Runtime.cpp:240` `RCUReaderGuard guard(runtimeRcuReader);`（`process()` 冒頭） | **毎 audio block** |
| RT が読む対象 | `Runtime.cpp:245-250` `auto& activeCrossfadeGain = crossfadeGain;` / `mixSmoother` / `oldDelay` 等 | `process()` 内 |
| RT reader が保持する共有ポインタ | `Lifecycle.cpp:48` `auto* oldState = exchangeAtomic(currentIRState, newState.release(), acq_rel);` の **旧 `IRState*` を参照中の reader** | 同一変数 |
| retire | `ConvolverProcessor.Lifecycle.cpp:58-59` `provider->enqueueDeferredDeleteNonRt(oldState, deleter);` | → `AudioEngine.h:4455` → `:4463` → `ISRRetireRouter::enqueueWithRetry` → `EpochDomain::enqueueRetireTyped`（`EpochDomain.h:406`）→ `deferredDeletionQueue` |
| 解放 | `DeferredDeletionQueue.h:145` `entry.deleter(entry.ptr);` → `Lifecycle.cpp:51-56` `state->~IRState(); mkl_free(state);` | **破壊 + メモリ解放** |

#### 鎖② EQ `EQState`

| 段 | 位置 |
| --- | --- |
| RT reader | `EQProcessor.h:497` `RCUReader rcuReader { m_epochDomain }` |
| RT guard | `EQProcessor.Processing.cpp:488` `RCUReaderGuard guard(rcuReader);` → `:489` `loadCurrentState()` |
| retire | `EQProcessor.Core.cpp:104` `enqueueDeferredDeleteWithFallback(state, deleteEQStatePtr, epoch)`（→ `:26` → `AudioEngine::enqueueDeferredDeleteNonRt`） |
| 解放 | 同一 `DeferredDeletionQueue.h:145` |

**RT reader と retire 側が同一の `EpochDomain` provider を通っていることが
`{ m_epochDomain }` の初期化で示されている。これは偶然の一致ではない。**

### 7.2 影響範囲は D queue 1 個ではない — `getMinReaderEpoch()` は 4 系統の解放判定Somosる

| # | 判定利用者 | 位置 | 解放形态 |
| --- | --- | --- | --- |
| 1 | `deferredDeletionQueue.reclaim` | `EpochDomain.h:394` | `DeferredDeletionQueue.h:145` |
| 2 | `m_retireQuarantine.drain` | `ISRRetireRouter.cpp:401`（`drainQuarantineStore`） | store drain |
| 3 | `m_emergencyQuarantine.drain` + `m_terminalReclaim.drain` | `ISRRetireRouter.cpp:544` / `:555`（`drainEmergencyAndTerminal`） | store drain |
| 4 | **`terminalReclaim` の同期破棄** | `ISRRetireRouter.cpp:517-524` | **`deleter(ptr)` を同期実行** |
| 5 | `m_coordinator.reclaim` | `AudioEngine.Retire.cpp:65` / `:373` | Coordinator reclaim |

**誤った除外は 5 系統すべてに伝播する。** 特に #4 は**同期 `deleter(ptr)`** であり、
 非同期 queue を経由しない。

### 7.3 RT 到達性

| 機能 | スレッド | 根拠 |
| --- | --- | --- |
| `enterReader` / `exitReader` | **RT** | `RCUReader::enter` は `RCUReaderGuard`（`ConvolverProcessor.Runtime.cpp:240` / `EQProcessor.Processing.cpp:488`）から呼ばれ、RT |
| `quarantineReader` | NonRT | `AudioEngine.Timer.cpp:1799`（health event callback） |
| `getMinReaderEpoch` / 5 系統の解放 | NonRT | 呼出し元は Timer / `tryReclaimDeferredResources` / `drainDeferredRetireQueues` / `releaseResources` / `SnapshotCoordinator`。RT の `retireRT`（`ISRRetireRouter.cpp:283`）は `provider_->enqueueRetire` を呼ぶだけで reclaim に到達しない（`enqueueWithRetry` は `jassert(!isAudioThread())` / `ISRRetireRouter.cpp:299`） |

**決定的な帰結: 「RT は解放しない」をAGT維持している。RT は被害者のみ。**
欠陥は「RT が保護を受ける側なのに、その保護の入力が NonRT によって破壊される」構造である。

### 7.4 不変条件違反

| 不変条件 | 判定 |
| --- | --- |
| **Retire は Epoch を通る** | **破る**。`entry.epoch < minReaderEpoch` が、存在する reader（`E_R`）より大きい値に対して成立する |
| RT は判断しない | RT 側の判断関数は未変更。**入力が破壊される** |
| RT は解放しない | 維持（§7.3） |
| RT は待たない / 解放しない / ownership なし | 維持（RT 経路の修正を伴わない） |
| Shutdown は完全 Drain | 維持。`unquarantineAllReaders`（`ReleaseResources.cpp:435`）は activeReaderCount==0 後にのみ呼ばれ、全 reader を非参加にするため、RC-1 の影響を受けない（§10.2） |
| Publish / Crossfade / Retire authority を増やさない | 維持（authority の new 認可は本 STG で導入しない） |
| RuntimeWorld ownership を増やさない | 維持 |

---

## 8. Existing invariants の現状評価

| # | invariant | 現状 | 本 STG の扱い |
| --- | --- | --- | --- |
| INV-A | 「quarantined ⇒ depth==0」 | **宣言のみ、強制なし**（§3.3） | **強制側を実装する** |
| INV-B | 「active reader は必ず minReaderEpoch に寄与する」 | **破られている**（S6） | **回復する** |
| INV-C | 1:1 ownership（retire → authority → deleter） | 維持。本件は release **タイミング**の誤りであり ownership 移-Sanの欠落ではない | 変更しない |
| INV-D | `0x03` を生成しない | 維持（3 系統 CAS） | 変更しない |
| INV-E | RT は判断しない | 維持（RT 経路を変更しない） | 変更しない |

**既存 invariant の改変は INV-A の「強制の追加」のみで、INV-B の「回復」のみ。
invariant の意味を変える更新はしない。**

---

## 9. Repair candidates 比較評価

### 9.1 Candidate A — `enterReader()` が quarantine flag を検査する → **REJECT**

| 評価項目 | 判定 |
| --- | --- |
| RT での判断追加になるか | **なる。** `enterReader` は RT（`RCUReader::enter` → `acquireThreadSlot` → `enterReader`）。現在 **分岐が 0 個**の関数に policy 分岐が増える |
| RT で新しい failure policy が発生するか | **発生する。** 現状 `enterReader` は `void noexcept` で**失敗経路が無く、成功を返すことだけ**が契約。拒否を追加すると**新しい failure policy** の導入になる |
| audio thread の実行契約を変更するか | **変更する。** 「enter は必ず epoch 保護を保証する」から「enter が拒否しうる」へ変わる |
| realtime cost | 増分（atomic load 1 回）。**小さいが 0 ではない** |
| existing API への影響 | `IReaderEpochProvider::enterReader` の契約が「必ず成功」から「失敗しうる」に変わる。全実装（mock 含む）に影響 |

**REJECT の決定的理由（安全性）**:

`enterReader` が失敗した場合の既存挙動を確認する。`RCUReader.h:62-82` は
`tid < 0`（slot 取得失敗）時に `rootEnterSucceeded_ = false` を立てるが、
**guard 自体は生きたまま caller は状態を読む**。

- fail-closed 消費者は **1 箇所のみ**: `ObservedRuntime.h:49` `if (!guard.rootEnterSucceeded())`
- **fail-open 経路**: `ConvolverProcessor::process()`（`Runtime.cpp:240`）と
  `EQProcessor::process()`（`EQProcessor.Processing.cpp:488`）は
  `rootEnterSucceeded()` を**検査していない**。guard 構築に失敗しても
  `loadCurrentState()` / `crossfadeGain` を読む。

→ **Candidate A を採用すると、隔離済み slot で enter が失敗した RT block が
epoch 保護なしで共有状態を読むことになる。Candidate A は UAF を换一个 UAF に変える。**

（`rootEnterSucceeded()` 非検査は**既存の別問題**であり、本 STG の scope 外。
本 STG で是正しない。**Candidate A の採用を妨げる理由としてのみ記録する。**）

### 9.2 Candidate B — `quarantineReader()` 後に depth を再検証する → **REJECT（主契約として）**

| 評価項目 | 判定 |
| --- | --- |
| TOCTOU だけを閉じるのか | **そう。Path A のみ** |
| 経路 B を閉じられるのか | **閉じられない。** B は quarantine **成立後**の再入場で、`quarantineReader` 之后再検証しても無関係 |
| flag/depth 不整合を残さないか | **残る。** CAS 成功（`0x01`）後に `depth>0` が判明しても、**`0x01 → 0x02` の降格経路が存在しない**。`0x03` を生成しない設計（`:276-278`）とCAS 3 系統（`:294`/`:302`/`:319`）的下、**昇格方向のみを持つ** |

**Owner §6「A を先に修正しても根本原因は残ります」に該当する。REJECT。**

### 9.3 Candidate C — quarantine state と depth を単一 atomic state に統合 → **REJECT（本次 scope）**

| 評価項目 | 判定 |
| --- | --- |
| ReaderSlot layout 変更 | **必要**（`depth: uint32` と `quarantineFlags: uint8` を 1 個の `uint32` に統合） |
| ABI / cache layout | `ReaderSlot` は `EpochDomain` の private nested struct。外部 ABI には出ない。ただし `alignas` と cache line 構成は要確認 |
| atomic state transition | `enterReader` の `fetchAdd` → **CAS loop 化**が必要。RT 経路の書き換え |
| RT path | **変更される**（`enterReader` / `exitReader` 両方） |
| memory ordering | `exitReader:168` の `epoch=kInactive` publish → `:178` の flag CAS の**順序不变条件**を再設計する必要がある。現行の暗黙保証（`:367` コメントが指摘）に影響 |
| existing reader registration semantics | `registerReaderThread` / `reserveReaderThread` の `epoch` CAS との相互関係を再定義する必要 |

**REJECT 理由**: RT 経路の大規模書き換え + memory ordering の再設計を伴う一方、
**RC-1 だけで §7.4 の安全条件が満たされる**（§10.3 で証明）。**最小修正で最大効果原则に照らして不採用。**
将来課題として記録する。

### 9.4 Candidate D — `getMinReaderEpoch()` で不整合時に安全側へ倒す → **ADOPT（主契約）**

| 評価項目 | 判定 |
| --- | --- |
| Release で assert が消える問題 | **直接対処する。** 除外条件に `depth==0` を含めることで、Release でも Debug と**同一の判定**になる（assert は副作用として消える） |
| reclaim 停止による resource retention | **あり、しかし設計上許容。** `SafeStateSwapper.h:167-171` が「exitReader を呼ばずに Reader が停止した場合…reclaim がブロックされる。これは RCU の仕様（**設計上は安全側フェール**）」と明記している。**RC-1 は S6 をこの既知の安全側フェールに戻す** |
| fail-closed 性 | **フェイルクローズ方向**（除外しない = より多く保護 = 解放を減らす） |
| shutdown への影響 | **なし**（§10.2 で検証） |
| UAF防止として十分か | **十分。§10.3 で read-only 構造証明する** |

---

## 10. Recommended repair contract

### 10.1 RC-1（唯一の必須変更 — **唯一の契約**）

**`EpochDomain::getMinReaderEpoch()` の除外条件を、`kQuarantinedFlag` 単独から
「`kQuarantinedFlag` **かつ `depth == 0`**」に変更する。**

契約上の要求（実装ではない）:

```text
RC-1.1  getMinReaderEpoch() の除外は、reader が「非参加」でなければならない。
        すなわち除外条件は (quarantineFlags & kQuarantinedFlag) != 0
                              AND depth == 0
        のみに限られる。

RC-1.2  depth > 0 かつ kQuarantinedFlag の状態は、通常経路で処理し
        その reader の epoch を minEpoch に寄与させる。
        → EBR の安全条件「minReaderEpoch <= 全 active reader の epoch」を回復する。

RC-1.3  RT 経路（enterReader / exitReader / acquireThreadSlot /
        registerReaderThread / reserveReaderThread）は変更しない。

RC-1.4  quarantineFlags の writer 集合を変更しない。
        （unquarantineAllReaders を shutdown 以外に呼ばない）

RC-1.5  既存 API の署名・virtual 集合を変更しない。

RC-1.6  quarantineFlags 単独の reader は 0 件のまま維持する。
        （検出器を enterReader 側に増やさない）
```

**変更範囲**: `src/core/EpochDomain.h` の `getMinReaderEpoch()` **1 関数のみ**。
分岐 1 箇所の条件変更。atomic 操作の追加なし（既存 `consumeAtomic` を共有できる）。

### 10.2 Shutdown との非衝突（read-only 検証）

`unquarantineAllReaders` の前提条件は「`activeReaderCount == 0` を確認済み」
（`ReleaseResources.cpp:432`）。RC-1 は `depth > 0` の reader を除外に **使わない** ので、
shutdown 時の全 reader 非参加という前提と**完全に整合**する。
さらに RC-1 は flags を変更しないため、shutdown の強制解放機構はそのまま有効である。

**衝突なし。**

### 10.3 RC-1 が UAF を防止し得る理由（read-only 構造証明）

**補題（本監査の中核）**:
`getMinReaderEpoch()` が reader slot を scanning において「捕获し損ねた」場合、
その reader が保護すべきオブジェクトが誤解放されることはない。

**証明**:

1. `getMinReaderEpoch` の seed は `minEpoch = currentEpoch()`（`EpochDomain.h:216`）。
2. 解放条件は `isOlder(entry.epoch, minReaderEpoch)`、すなわち
   `entry.epoch < minReaderEpoch`（`DeferredDeletionQueue.h:132`）。
3. 捕获し損ねた reader は、`enterReader:127` で `epoch = currentEpoch()` を publish した直後
   （`:130` の `depth++` 前）に scanning を通過した reader であり、
   その epoch は **scan 開始時の `currentEpoch()` と同一またはそれ以上**。
   `globalEpoch` は `publishEpoch()`（`:196-200`）で単調非減少にのみ進む。
4. よってその reader の epoch は seed `minEpoch` 以上であり、
   その reader を minEpoch に加算しても結果は変わらない。
5. したって scanning の取り見落としは **minEpoch を本来より大きすることが不可能**だから。
   誤解放を生むのは、**minEpoch を本来より大きくする**操作だけである。
6. 現状の S6（`kQuarantinedFlag` 単独での除外）はまさにその操作であり、
   **本欠陥の正体**である。
7. RC-1 は除外条件に `depth == 0` を加えることで、**「minEpoch を大きくしうる」
   経路を全て閉じる**。残る取りこぼしは補題により無害である。

**結論: RC-1 は安全性の**必要十分条件**である。**
RC-1 の適用後、`quarantineFlags != 0 && depth > 0`（S6）は**無害な状態**になる。
したがって **Path A と Path B の両方が、個別に修正されなくても同時に閉じる。**

### 10.4 副次効果（明記）

- **S6 状態の RT block は reclaim を block する**（安全側フェール）。
  ただし真の stuck reader（`depth>0` + `pending`）は **RC-1 により挙動が変わらない**
  （`kPendingQuarantineFlag` は RC-1 の対象外であり、元来除外されない。`:226` は
  `kQuarantinedFlag` のみを見る）。
- よって **RC-1 は quarantine 機能の正常動作を一切壊さない**（§11 T10-3 が証明する）。
- RC-1 は**「flag が残る」異常状態（S6）を無害化する**。
  flag の stale 残存（RC-2 で解消すべき范畴）は**診断上の問題**として残るが、
  **安全性上の問題ではなくなる**。これは R-4 diagnostics 領域であり、本 STG に併合しない。

---

## 11. Test contract

**原则: expected result を先に固定し、その結果に production code を合わせる。
production code の実装詳細（分岐位置・変数名）に oracle を依存させない。**

### 11.1 T10-1（**強制**）: 「quarantine かつ参加中の reader は `getMinReaderEpoch` から除外されない」

**Owner §11 の 2 択の答え: 後者を採用する。**

- 採用: `T10-1: quarantine + live reader 状態が getMinReaderEpoch から除外されない`
- 不採用: `quarantine 後に同一 reader が再入場できない`（= Candidate A の主張。
  §9.1 で安全性をasons により REJECT した。採用すると test が production の
  誤った policy を固定してしまう）

**oracle**: 戻り **`minReaderEpoch` の値**のみ。内部 state の観測はしない。

決定論的な単一スレッド手順（`EpochDomain` 直叩き、race 不要）:

```text
1.  publishEpoch() を N 回進め、globalEpoch を十分に進める
2.  i = registerReaderThread("t")
3.  quarantineReader(i)            // depth==0 ⇒ 即座隔離 ⇒ flags=0x01
4.  expectedEntryEpoch = currentEpoch()  (まだ更大的に)
5.  enterReader(i)                 // depth 0→1  ⇒ S6 を決定論的に生成
6.  dq.enqueue(ptr, deleter, retireEpoch = 3)   // 过去の epoch で retire
7.  minEpoch = getMinReaderEpoch()
8.  tryReclaim()
9.  ORACLE:
   - minEpoch <= step5 で enter した reader の epoch  （保護されている）
   - ptr の deleter が**呼ばれていない**              （解放されていない）
   - reclaimSuccessCount の増加が reader の epoch を超えない
```

**修正前**: `minEpoch == currentEpoch()`（== step4 値）となり、entry epoch 3 は
`isOlder(3, currentEpoch())` で真 → **deleter が呼ばれる**。→ **FAIL（危険側）**。
（Release で FAIL する。Debug では加えて `:228` の assert が発火するが、
**oracle は戻り値であり assert ではない**。）

**修正後**: `minEpoch == 3`（reader の epoch）となり、`isOlder(3,3) == false` →
**deleter は呼ばれない**。→ **PASS（安全側）**。

**この test は Path A と Path B のどちらの経路でも成立する。**
（手順 3-5 は Path B の非 realloc 変種。手順 3 の後に
`reserveReaderThread(i)` を挟むことで Path B の alloc 変種（S7→S6）も別途生成できる。）

### 11.2 T10-2（**強制**）: quarantined slot の再割当後も、新 reader は保護される

**Owner の記述「quarantined slot が新 reader へ再利用されない」は、本契約の妥当な一部ではない。**
本契約は再割当を**禁止しない**（Q1 = No）。したがって test の形が変わる。

```text
1.  publishEpoch() を進める
2.  i = registerReaderThread("old")
3.  quarantineReader(i)              // flags=0x01, epoch=kInactive  ⇒ S5
4.  reserveReaderThread(i) == true   // ★ S5 が allocation に受理されることを観測
                                    //   （Q4 の実証。PASS である）
5.  enterReader(i)                   // ⇒ S7 → S6
6.  dq.enqueue(ptr, deleter, 古い epoch)
7.  tryReclaim()
8.  ORACLE: deleter が**呼ばれていない**
9.  teardown: unquarantineAllReaders()  ⇒ quarantinedReaderCount() == 0
```

- 手順 4 が `true` を返すこと自体が **Q1/Q4 の実証**であり、**PASS 期待**。
- 手順 8 が PASS することが **RC-1 の実効**。
- 手順 9 が **shutdown 契約の非回帰**。

### 11.3 T10-3（**強制**）: quarantine / exit / reclaim の正常経路が保たれる

**これが RC-1 の非回帰担保であり、機能を-disable していないことの証明。**

```text
1.  publishEpoch() を進める
2.  i = registerReaderThread("stuck")
3.  enterReader(i)                          // depth = 1
4.  r = quarantineReader(i)                 // depth>0 ⇒ pending
5.  ORACLE A: r == false                    （即座隔離でない = deferred）
6.  dq.enqueue(ptr, deleter, epoch = enter 時より古い値)
7.  tryReclaim()
8.  ORACLE B: deleter は**未**呼出         （参加中は保護される）
9.  exitReader(i)                           // depth 0 ⇒ pending→quarantined
10. ORACLE C: quarantinedReaderCount() == 1
11. tryReclaim()
12. ORACLE D: ptr の deleter が**呼ばれる** （隔離により reclaim 進行）
```

**ORACLE D が PASS することが、RC-1 が quarantine 機能を壊していないことの決定的証拠。**
（`kPendingQuarantineFlag` は RC-1 の対象外なので、この経路は修正前後で同一。）

### 11.4 T10-4（**強制**）: shutdown unquarantine / drain

```text
1. 上記 T10-3 まで実行し、隔離（ORACLE C）を成立させる
2.  snapshot = getMinReaderEpoch()            （隔離により == currentEpoch()）
3.  enterReader(i) → exitReader(i)            // 非参加状態で再入場/退出
4.  unquarantineAllReaders()                  // shutdown 模擬
5.  ORACLE A: quarantinedReaderCount() == 0
6.  ORACLE B: getMinReaderEpoch() <= step2 の snapshot
              （除外が解除された。将来 i が参加するなら epoch が反映される）
7.  drainAll()                                // DeferredDeletionQueue::drainAllUnsafe
8.  ORACLE C: 全 entry の deleter が呼ばれ、pendingRetireCount() == 0
```

### 11.5 T10-5（**不要と判定**）: Path A の race 再現

**不要。** 理由: RC-1 により **S6 は生成経路によらず無害**（§10.3）である。
したがって「どの経路で S6 を作るか」を検証する必要がなく、
nondeterministic な race 再現 test を追加する価値がない。

**ただし** §11.1/§11.2 の 2 本は S6 を**決定論的に生成する**ため、
Path A / Path B の**両方**を状態としてカバーしている（生成経路は問わない）。
→ **race 再現 test を作らずに両経路を担保できる。**

### 11.6 既存 test oracle の変更

**不要。** 監査実測: `EpochDomain::quarantineReader` / `unquarantineAllReaders` /
`quarantinedReaderCount` を**実 `EpochDomain` に対して呼ぶ既存 test は 0 件**。
`src/tests` 内の `quarantine` 参照 102 件のうち、`D8_2_B_2_Tests.cpp:160-162` は
**mock**、残りは**別系統の DSP handle quarantine**（`QuarantineService` /
`submitQuarantine` / `dspQuarantineManager`）。

**既存 oracle の改变 = 0 を宣言できる。**

---

## 12. Prohibited changes（Owner §9 の 10 項目への準拠）

| # | 禁止項目 | 本契約の準拠 |
| --- | --- | --- |
| 1 | ❌ assert を増やすだけ | **规避。** RC-1 は Release でもRelease 存活する条件変更。assert（`:228`）は副作用として消えるが**安全性の根拠ではない** |
| 2 | ❌ Release でも単純 continue/skip するだけ | **规避。** 除外条件が `depth == 0` を**含む**ため単純 skip ではない。Release で Debug と同一判定になる |
| 3 | ❌ quarantine flag を勝手に clear する | **规避。** RC-1.4: writer 集合を変更しない |
| 4 | ❌ shutdown 以外で unquarantine する | **规避。** 呼出し元を追加しない |
| 5 | ❌ RT で新しい policy 判断を追加する | **规避。** RC-1.3: RT 経路は変更しない |
| 6 | ❌ reader slot を別の slot へ勝手に移動する | **规避。** slot の割当・移動を行わない |
| 7 | ❌ epoch を人工的に進める | **规避。** `publishEpoch` の呼出しを追加しない |
| 8 | ❌ reclaim を強制する | **规避。** `drainAll` / `reclaim` の呼出しを追加しない。reader の参加を誤って再取得する新規侵入を 0 にする |
| 9 | ❌ memory_order を根拠なく変更する | **规避。** 既存 `consumeAtomic(..., acquire)` を再利用する。memory order の変更は 0 |
| 10 | ❌ STG-9-D1 を再 OPEN する | **规避。** STG-9-D1 の差分（`AudioEngine.h` / `AudioEngine.Retire.cpp`）に一切触れない。本 STG の production diff は `EpochDomain.h` のみ |

**特にOwner §9 の注記「`assert(depth == 0);` を追加するだけでは修正とは認めません。
Release で安全性を回復する必要があります。」に対する明示的準拠:
本契約は `assert` の追加を**一切含まない**。Release で安全性が回復されるのは
`getMinReaderEpoch()` の**条件式**が変わるためであり、`NDEBUG` の影響を受けない。**

---

## 13. STOP conditions の判定

| # | STOP 条件 | 判定 | 根拠 |
| --- | --- | --- | --- |
| 1 | quarantine の仕様意味論を確定できない | **該当せず** | §2 Q1-Q6 + §2.7 で確定（instance 単位の一時隔離） |
| 2 | Path B を source 上で証明できない | **該当せず** | §6.1 B-1〜B-7 すべて YES。**S5 が allocation の受理条件と一致** |
| 3 | UAF までの lifetime chain を証明できない | **該当せず** | §7.1 matched pair（Convolver / EQ）+ §7.2 影響 5 系統 |
| 4 | repair candidate が複数あり一意に収束しない | **該当せず** | **RC-1 に一意収束**。A/B/C は §9 で理由付きで REJECT |
| 5 | RT decision 問題を解決できない | **該当せず** | **RT 経路を変更しないことで解消**。RT 側の既存不備（`rootEnterSucceeded()` 非検査）は Candidate A の REJECT 根拠としてのみ記録 |
| 6 | reader slot ownership が不明 | **該当せず** | §3.1/§3.2 で state machine と release 機構の不在を確定 |
| 7 | memory ordering が不明 | **該当せず** | §10.3 の補題で `globalEpoch` の単調性 + seed 構造により無害性を証明。RC-1 は memory order を変更しない |
| 8 | existing shutdown semantics との衝突 | **該当せず** | §10.2 で検証。前提（activeReaderCount==0）と整合 |
| 9 | 既存 STG との交差が発見された | **該当せず** | §7.4 参照。`EpochDomain.h` は STG-1〜9 で未変更。系統② DSP quarantine は対象外 |
| 10 | production 変更が必要になった | **該当せず** | read-only 完遂。§14 に宣言 |
| 11 | test oracle の変更が必要になった | **該当せず** | §11.6。既存 test 0 件のため改変不要 |
| 12 | latest authority が FRESH でない | **該当せず** | §1。NEWER_SRC_COUNT 0 / FRESH |

**STOP 条件に該当しない。Audit を完了し、Owner へ差し戻す。**

---

## 14. §17 最終報告

```text
STG-10 Repair Contract Audit
============================

Concrete Defect:      PROVEN — quarantined 状態で退出した reader slot に
                      新規 reader が入場し、その reader が
                      getMinReaderEpoch() から除外される（状態 S6）。
                      epoch 保護不复、reclaim 判定が危険側に倒れる。

Severity:             Critical / memory safety (UAF)。
                      Release で無言、Debug で EpochDomain.h:229 assert 発火。
                      ※ Discovery の「有力な仮説」から Audit で確定。

Root cause:           quarantine が「reader の占有(occupancy)」に対する
                      一時隔離であるのに、実装が「slot の latch」として
                      実装されている。占有の終わり（exitReader:168 で
                      epoch=kInactive）と allocation の受理条件が
                      同一であるのに、allocation 経路
                      (registerReaderThread / reserveReaderThread) が
                      quarantineFlags を読まないため、隔離の寿命が
                      slot の寿命と一致してしまう。
                      さらに enterReader は flag を見ないため
                      quarantine 解除も一切行われない。

Path A:               TOCTOU（quarantineReader:286 の depth read と
                      :294/:302 の CAS の間）。実在するが**窓があり**、
                      単独の修正では根本原因が残る。→ 主契約にしない。

Path B:               PROVEN（B-1〜B-7 全て YES）。
                      さらに S5（epoch=kInactive ∧ flags=0x01）は
                      reserveReaderThread:95-103 の受理条件と一字一句
                      一致するため、**race ではなく公称経路**。
                      → B だけで Concrete Defect が成立する。

UAF proof:            PROVEN（read-only structural proof）。
                      鎖① Convolver:  RT guard
                        (ConvolverProcessor.Runtime.cpp:240) →
                        oldState 参照 ⇄ NonRT retire
                        (ConvolverProcessor.Lifecycle.cpp:58-59) →
                        DeferredDeletionQueue → 解放 (:145)
                      鎖② EQ:         RT guard
                        (EQProcessor.Processing.cpp:488) ⇄ NonRT retire
                        (EQProcessor.Core.cpp:104) → 同上
                      両者とも RCUReader の provider は { m_epochDomain }。
                      影響は D queue 1 個ではなく
                      getMinReaderEpoch() を消費する 5 系統の解放判定
                      （terminalReclaim は同期 deleter を実行）。
                      補題（§10.3）で scanning 取りこぼしの無害性を
                      証明 → RC-1 は必要十分。

Quarantine semantics: 確定 = 「reader の占有に対する一時隔離」。
                      Q1 No / Q2 Yes / Q3 Yes / Q4 Yes /
                      Q5 shutdown まで保持・shutdown で一括解放 /
                      Q6 0x00 ⇄ 0x02 ⇄ 0x01（0x03 禁止）。
                      意味論を 3 条件に収束（§2.7）。
                      現状は (a)「depth>0 は如何なる隔離でも除外しない」
                      を満たしていない。

Recommended repair:   RC-1（唯一の必須変更）
                      getMinReaderEpoch() の除外条件を
                        (quarantineFlags & kQuarantinedFlag) != 0
                      から
                        (quarantineFlags & kQuarantinedFlag) != 0
                        AND depth == 0
                      へ変更。
                      変更範囲: src/core/EpochDomain.h の 1 関数のみ。
                      RC-1.1 〜 RC-1.6 の 6 条件を契約として提示。

Alternative repairs rejected:
                      A: enterReader が flag を検査            → REJECT
                         理由: RT に新しい failure policy を導入し、
                         失敗時 ConvolverProcessor/EQProcessor は
                         rootEnterSucceeded() を検査しないため
                         **epoch 保護なしで共有状態を読む**
                         （UAF を別の UAF に置換）。
                      B: quarantineReader 後に depth 再検証    → REJECT
                         理由: Path A のみを閉じ Path B は残る。
                         かつ 0x01→0x02 の降格経路が存在せず
                         0x03 禁止設計と両立しない。
                      C: quarantine state と depth を单一 atomic
                         state に統合                          → REJECT
                         理由: RT 経路の大規模書き換えと
                         memory ordering 再設計を伴うが、
                         RC-1 だけで必要十分条件が満たされる。
                         将来課題として記録。
                      (RC-2「allocation 時に flag を reset」は
                       **安全性を回復することは不要**であり、
                       且つ RT allocation 経路へ 1 store を追加するため
                       本 STG の scope 外。liveness 課題として記録。)

RT impact:            ゼロ。RC-1.3 により RT 経路
                      (enterReader / exitReader / acquireThreadSlot /
                       registerReaderThread / reserveReaderThread) を
                      一切変更しない。既存 API の署名・virtual 集合も不変。
                      したがって
                        RT: no wait / no lock / no allocation /
                            no delete / no ownership /
                            **no new policy decision**
                      のすべてが現状どおり維持される。
                      Owner §12 の問い（enterReader での flag 読が
                      新しい policy decision か）に対する答えは
                      「Candidate A なら YES なので採用しない」。

Ownership impact:     ゼロ。ownership の 1:1 移管機構
                      (retire → authority → deleter) は不変。
                      変更するのは「解放タイミングの判定」であり
                      「所有権が移るか」ではわない。
                      RuntimeWorld ownership を増やさない。

Shutdown impact:      ゼロ。unquarantineAllReaders
                      (ReleaseResources.cpp:435) の前提
                      (activeReaderCount==0) は RC-1 と完全に整合。
                      RC-1 は flags を変更しないため強制解放機構は有効。
                      Shutdown は完全 Drain を維持。

Test contract:        T10-1（強制）quarantine + live reader が
                        getMinReaderEpoch から除外されない。
                        oracle は戻り minReaderEpoch の値。
                        決定論的・単一スレッド・race 不要。
                      T10-2（強制）quarantined slot の再割当
                        (reserveReaderThread == true) は PASS 期待、
                        且つ新 reader は解放されない。
                        併せて unquarantineAllReaders → count==0。
                      T10-3（強制）quarantine / exit / reclaim の
                        正常経路。隔離成立後に **deleter が呼ばれる**ことを
                        oracle とし、RC-1 が機能を無効化していない
                        ことを証明する。
                      T10-4（強制）shutdown unquarantine / drain。
                      T10-5（不要）Path A の race 再現。
                        S6 は生成経路によらず RC-1 で無害となるため
                        決定論的 test の 2 本で両経路を担保できる。
                      既存 test oracle の改変 = 0
                      （実 EpochDomain に対する quarantine 呼出しは 0 件）。

Production changes:   0（read-only）
Test changes:         0（read-only）
CMake changes:        0
Authority changes:    0（SHA 不変: B15640AD…4D7F3 / FRESH）
Commit:               0
Push:                 0

Verdict:              Concrete Defect      = PROVEN
                      Repair Contract     = PROVEN
                      Test Contract       = PROVEN
                      Implementation      = NOT STARTED
```

**`Repair Contract = PROVEN` に収束した。ただし implementation には進まない。**
Owner の Repair GO を待つ。
