# STG-10-D1 Repair Implementation（2026-09-29）

> **Verdict: STG-10-D1 Repair Implementation = PASS**
> **COMMIT = NOT AUTHORIZED / PUSH = NOT AUTHORIZED（未実施）**
> 本書は Owner GO に基づく RC-1 実装と全 validation の実施記録である。

---

## 1. Authority

Owner 指定: `ConvoPeq(11).md == ConvoPeq.md`（統一 authority）。実 source 検証は repo の
`ConvoPeq.md` を基準に実施した。

### 1.1 実装開始時（Owner 提示 baseline との一致確認）

| 項目 | Owner baseline | 実測 | 一致 |
| --- | --- | --- | --- |
| HEAD | `13a80d66e8e58fbcb66e71eb162f8e95fe134eeb` | `13a80d66e8e58fbcb66e71eb162f8e95fe134eeb` | **YES** |
| SHA-256 | `B15640AD…4D7F3` | `B15640ADCE15A38801DA8192652876EE4CA8EC6113247EEF9EB93E9EA754D7F3` | **YES** |
| size | 5,642,888 B | 5,642,888 B | **YES** |
| Generated | 2026-09-28 22:19:21 | 2026-09-28 22:19:21 | **YES** |
| NEWER_SRC_COUNT | 0 | 0 | **YES** |
| STATUS | FRESH | FRESH（`--check` exit 0） | **YES** |

**完全一致。STOP 条件「Authority が FRESH でない」に該当せず、実施を開始した。**

### 1.2 全 validation PASS 後の再生成（§12）

| 項目 | 値 |
| --- | --- |
| HEAD | `13a80d66e8e58fbcb66e71eb162f8e95fe134eeb` |
| SHA-256 | `E9FBD8F9E47DACA786D1BC6DB9596180A8659961BE34973CE4C4CC738CF29D09` |
| size | 5,662,249 B（+19,361 B） |
| Generated | 2026-09-29 00:19:43 |
| NEWER_SRC_COUNT | **0** |
| STATUS | **FRESH** |

authority 内容の検証:
- RC-1 の条件 `kQuarantinedFlag) != 0 && depth == 0` が `ConvoPeq.md:57535` に存在
- 旧 assert `quarantined reader must have depth==0` の出現回数 = **0**
- `STG10ReaderQuarantineTests` の出現回数 = 20（test TU 本文＋CMake wiring）

---

## 2. Concrete defect

`P1-5-IR-P2_STG-10_REPAIR-CONTRACT-AUDIT_20260928.md` で PROVEN とした下列 defect の修正。

> `quarantineReader()` が立てた `kQuarantinedFlag` は **reader の占有に対する一時隔離**であり
> slot の封印ではない。`registerReaderThread` / `reserveReaderThread` は `quarantineFlags` を
> 読まず `epoch==kInactiveEpoch` だけで再割当するため、隔離済み slot には次の audio block で
> 新規 reader が入場する（`RCUReader::exit:140` が `preferredThreadId` を保存し
> `RCUReader::acquireThreadSlot:169` が毎 block 同一 tid を再取得する）。
> 旧 `getMinReaderEpoch()` はフラグ**単独**で除外していたため、その live reader（depth>0）が
> safe-epoch 計算から抜け、`minReaderEpoch` が真値より大きくなり、その reader が参照中の
> `DeferredDeletionQueue` entry（および `terminalReclaim` の同期 `deleter`）が解放され得た。
> Release では `assert` が消滅するため無言。

**Path B が本体**（`S5` は `reserveReaderThread` の受理条件と一致するため公称経路）。
**Path A の TOCTOU は RC-1 により自動的に無害化される**（state が生成経路によらず安全になるため）。

---

## 3. RC-1 contract（実装内容）

Audit §10.1 の RC-1.1〜RC-1.6 を実装した。

### 3.1 production diff（`src/core/EpochDomain.h` のみ）

`getMinReaderEpoch()` 内の**除外条件 1 箇所**のみを変更。

```diff
         for (const auto& slot : readers)
         {
-            // ★ Phase 3: quarantined Reader は safe-epoch 計算から除外
-            //   kQuarantinedFlag 設定時は depth==0 が不変条件:
-            //     - 即座 quarantine: depth==0 でのみ設定
-            //     - 遅延 quarantine: exitReader で depth:1→0 後に昇格
-            //   したがって depth の再チェックは不要だが、防衛的アサートで担保する。
+            // acquire: enterReader release の depth 書き込みと HB し、depth 読み取り後に epoch を読む。
+            const uint32_t depth = convo::consumeAtomic(slot.depth, std::memory_order_acquire);
+
+            // ★ STG-10-D1 (RC-1): quarantined Reader は safe-epoch 計算から除外するが、
+            //   その根拠は「非参加（depth==0）」でなければならない。
+            //   ...（設計根拠コメント 8 行）
             const uint8_t flags = convo::consumeAtomic(slot.quarantineFlags, std::memory_order_acquire);
-            if ((flags & ReaderSlot::kQuarantinedFlag) != 0)
-            {
-                assert(convo::consumeAtomic(slot.depth, std::memory_order_acquire) == 0
-                    && "quarantined reader must have depth==0");
+            if ((flags & ReaderSlot::kQuarantinedFlag) != 0 && depth == 0)
                 continue;
-            }
 
-            // acquire: enterReader release の depth 書き込みと HB し、depth 読み取り後に epoch を読む。
-            const uint32_t depth = convo::consumeAtomic(slot.depth, std::memory_order_acquire);
             if (depth == 0)
                 continue;
```

`14 insertions(+) / 12 deletions(-)` / 1 ファイル / 1 関数。

### 3.2 obsolete assert の除去（RC-1 条件変更の不可分部分）

旧 `assert(depth == 0 && "quarantined reader must have depth==0")` を除去した。理由は 2 つ:

1. **RC-1 により `kQuarantinedFlag && depth>0` は「正当な active reader」になった。**
   旧 assert の前提は不変条件ではなくなった。
2. **残すと Debug build が正当な S6 状態で abort する。** T10-1 / T10-2 は
   S6 を**意図的に生成する**ため、残置すると Debug の test 自体が成立しない。

これは Owner §3 の指示形（「`quarantined && depth == 0` だけを除外」）から必然的に導かれる
変更であり、除外条件と同じ branch 内の要素である。**assert の追加は行っていない**
（Owner §9「assert を増やすだけ」の禁止にも抵触しない）。
**安全性の根拠は assert ではなく条件式であり、Release（`NDEBUG`）でも完全に有効**である
（Release で T10-1〜T10-4 が PASS した実測が証明）。

### 3.3 変更していないもの（RC-1.3 / RC-1.4 / RC-1.5）

| 対象 | バイト一致検証 |
| --- | --- |
| `enterReader` | HEAD L117-141 ≡ WORK L117-141 → **identical** |
| `exitReader` | HEAD L144-184 ≡ WORK L144-184 → **identical** |
| `registerReaderThread` | HEAD L39-42 ≡ WORK L39-42 → **identical** |
| `reserveReaderThread` | HEAD L87-114 ≡ WORK L87-114 → **identical** |
| `quarantineReader` | HEAD L279-325 ≡ WORK L281-327 → **identical** |
| `unquarantineAllReaders` | HEAD L328-336 ≡ WORK L330-338 → **identical** |
| `publishEpoch` | HEAD L193-202 ≡ WORK L193-202 → **identical** |
| `currentEpoch` | HEAD L186-190 ≡ WORK L186-190 → **identical** |
| `tryReclaim` | HEAD L386-396 ≡ WORK L388-398 → **identical** |
| `drainAll` | HEAD L425-428 ≡ WORK L427-430 → **identical** |
| `getMinReaderEpoch` | HEAD L214-248 vs WORK L214-250 → **差分あり（意図通り）** |

`RCUReader.h` / `DeferredDeletionQueue.h` / `ISRRetireRouter.{h,cpp}` / `ISRShutdown.h` /
`ISRRuntimePublicationCoordinator.{h,cpp}` / RuntimeWorld 関連 = **変更 0**。

> 検証方法 note: 最初の grep ベースの静的検証では `registerReaderThread` /
> `reserveReaderThread` / `currentEpoch` に偽陽性 FAIL が出た（追加**コメント**内の関数名
> マッチ、および sed 範囲の取り違え）。関数本体を正規化して抽出する
> バイト単位比較で再検証し、上記のとおり**全件 identical** であることを確定した。

---

## 4. Test contract 実装

### 4.1 追加 test TU

`src/tests/STG10ReaderQuarantineTests.cpp`（**新規 / 454 行**）。

| 方針 | 実装 |
| --- | --- |
| 決定論性 | 単一スレッド。race 再現不使用。S6 を手順で明示的に生成 |
| oracle | `getMinReaderEpoch()` の**戻り値**と「deleter 起動の有無」。**Debug assert を oracle にしない** |
| internal 依存なし | 公開 API のみ。private state・内部変数名に oracle を依存させない |
| test hook | **0 件**。production に test 用の足場を追加していない |
| 既存 oracle | **改変 0**。既存 test への追加・変更なし |

### 4.2 T10-1 — quarantined かつ live reader は minEpoch から除外されない

```text
1. advanceEpoch(5)                      currentEpoch == 5
2. i = registerReaderThread()
3. quarantineReader(i)   == true        depth==0 → 即座隔離（flags=0x01）
   quarantinedReaderCount() == 1
4. enterReader(i)                       depth 0→1  →  S6
   activeReaderCount() == 1
5. bumpEpoch(5)                         currentEpoch == 10（reader の epoch 5 は stale）
6. retire entry at epoch 7              （reader より新しく、currentEpoch より古い）
7. minEpoch = getMinReaderEpoch()       ★ ORACLE: == 5（reader の epoch）
8. tryReclaim()                         ★ ORACLE: deleter 未起動 / pendingRetireCount==1
                                          / reclaimSuccessCount==0
9. control: exitReader(i) → tryReclaim() ★ ORACLE: deleter == 1（test が「解放不能」で
                                          PASS していないことの証明）
```

### 4.3 T10-2 — quarantined slot の再利用を禁止しない（＋ live reader は保護）

```text
1. advanceEpoch(5)
2. i = registerReaderThread()
3. enterReader(i) → exitReader(i)       RT block 1（epoch==kInactive, depth==0）
4. quarantineReader(i) == true          S5
5. reserveReaderThread(i) == true   ★ ORACLE: slot は恒久封印ではない（Q1/Q4 の固定）
6. enterReader(i)                       RT block 2 → S6
7. bumpEpoch(5)
8. retire at 7 / getMinReaderEpoch() ★ ORACLE: == 5
9. tryReclaim()                    ★ ORACLE: deleter 未起動
10. teardown: exitReader → unquarantineAllReaders → count==0
              → drainAll → deleter==1 / pending==0 / leak 0
```

**step 5 が PASS であることが契約の核心**（quarantine を slot 封印にしない）。
**step 8/9 が PASS することが RC-1 の実効**（Q1/Q4 と RC-1 の同時固定）。

### 4.4 T10-3 — 正常 quarantine / exit / reclaim（RC-1 の非回帰担保）

```text
1. advanceEpoch(5); i = register
2. enterReader(i) → bumpEpoch(5) → retire at 7
3. tryReclaim()                ★ ORACLE: deleter 未起動（active 中は保護）
4. quarantineReader(i) == false ★ ORACLE: depth>0 なので deferred
5. quarantinedReaderCount()==0  ★ ORACLE: pending は quarantined として計上されない
6. tryReclaim()                ★ ORACLE: 依然として保護（kPending は除外対象外）
7. exitReader(i)                pending → quarantined へ昇格
8. quarantinedReaderCount()==1  ★ ORACLE
9. tryReclaim()                ★ ORACLE: deleter == 1（隔離が reclaim  진행を可能にする）
```

**step 9 が PASS することが「RC-1 が quarantine 機能を無効化していない」ことの決定的証拠。**

### 4.5 T10-4 — shutdown unquarantine / drain（既存 shutdown semantics の非回帰）

```text
1. advanceEpoch(5); i = register; enter → exit
2. quarantineReader(i) == true; count == 1
3. activeReaderCount() == 0   ★ ORACLE: ReleaseResources.cpp:432 の shutdown 前提
4. retire at 3 → tryReclaim() → deleter==1（隔離により epoch-safe な entry は解放）
5. bumpEpoch(5); getMinReaderEpoch() == 10  ★ ORACLE: 非参加 reader しか居ないため seed 一致
6. retire at 10 → tryReclaim() → 未解放 ★ ORACLE: isOlder(10,10)==false
7. unquarantineAllReaders() → count == 0  ★ ORACLE
8. drainAll() → deleter==2 / pending==0 / leak 0  ★ ORACLE
```

### 4.6 test 衛生の改善（実施記録）

初版は `TestObject::aliveCount` を TU 全体で共有していたため、T10-1 / T10-2 が失敗した場合に
残存オブジェクトが T10-3 / T10-4 の leak 判定に連鎖し、failure の帰属が曖昧になっていた
（§5.2 の negative control 初回実行で実際に観察）。
**test ごとに独立した生存カウンタ（`DeleterTracker::aliveCount`）に変更し解消した。**

---

## 5. Negative control（test が実際に欠陥を検出することの証明）

production を**一切変更せず**、隔離環境（`%TEMP%`）に header 閉包をコピーして実施した。

### 5.1 Variant B — HEAD byte-exact 版（Debug）

```text
Assertion failed: convo::consumeAtomic(slot.depth, std::memory_order_acquire) == 0
  && "quarantined reader must have depth==0",
  file src\core\EpochDomain.h, line 229
B_RUN_EXIT = -1073740791 (0xC0000409 / abort)
```

**`EpochDomain.h:229` の assert が発火する。** これは STG-9-D1 Gate-R2 期間および
Post-Commit Gate に記録された intermittent observation
（`EpochDomain.h:229` assert 連発後の stack overflow）の**署名を再現したもの**であり、
STG-10 Concrete Defect の現場証拠と一致する。
（root cause の確定は本 STG の Audit で完了済み。STG-9-D1 の再 OPEN ではない。）

### 5.2 Variant A — RC-1 条件のみ差し戻し（assert 除去）

```text
T10-1: minReaderEpoch=10 expected 5 (reader epoch)
FAIL: T10-1 quarantined + live reader must not be excluded
T10-2: minReaderEpoch=10 expected 5 (reused slot's live reader)
FAIL: T10-2 quarantined slot is reusable but its live reader is protected
STG10ReaderQuarantineTests: 2 test(s) FAILED
```

| 結果 | 意味 |
| --- | --- |
| **T10-1 / T10-2 が FAIL** | **値 oracle のみで欠陥を検出する。assert に一切依存していない**（Owner §6-T10-1 の要件充足） |
| **T10-3 / T10-4 が PASS** | これらは真の**非回帰 test**であり、RC-1 適用前後で中立。RC-1 が quarantine 機能を壊していないことの反面証拠 |

repo 側のファイルは post-fix のまま保持されていることを setup / build の両方で確認した
（`repo still post-fix = True`）。隔離環境は検証後に削除済み。

---

## 6. Build / Regression 結果

### 6.1 Phase 1 — Debug

| 項目 | 結果 |
| --- | --- |
| Debug build | **PASS**（`cmake --build build --config Debug` exit 0） |
| T10-1 | **PASS** |
| T10-2 | **PASS** |
| T10-3 | **PASS** |
| T10-4 | **PASS** |
| STG10ReaderQuarantineTests.exe | **PASS**（exit 0） |
| Debug CTest | **43/43 PASS**（`100% tests passed out of 43` / 196.50 sec / exit 0） |

```text
STG10ReaderQuarantineTests: T10-1 PASS (quarantined live reader contributes to minReaderEpoch)
STG10ReaderQuarantineTests: T10-2 PASS (slot reuse allowed, live occupancy protected)
STG10ReaderQuarantineTests: T10-3 PASS (pending->quarantined unblocks reclaim)
STG10ReaderQuarantineTests: T10-4 PASS (shutdown unquarantine + drainAll)
STG10ReaderQuarantineTests: PASS (T10-1/T10-2/T10-3/T10-4)
```

CTest 件数が 42 → 43 になったのは新規 test 登録による（既存 42 件は全て PASS のまま）。

### 6.2 Phase 2 — Release（Release validation 必須）

| 項目 | 結果 |
| --- | --- |
| Release build | **PASS**（exit 0） |
| T10-1 / T10-2 / T10-3 / T10-4 | **PASS**（4 件全て） |
| STG10ReaderQuarantineTests.exe (Release) | **PASS**（exit 0） |
| Release CTest | **43/43 PASS**（154.93 sec / exit 0） |

**Debug と Release の結果は矛盾しない**（STOP 条件 13 に該当せず）。
**Release で PASS したことが、本 defect が「assert 消滅による無言の安全境界後退」で
あったことの直接の証拠である**（STOP 条件 9 に該当せず）。

### 6.3 Phase 3 — 既存 STG regression

`AudioEngineHarness` を Debug / Release 両方で直接実行し、明示行を確認した。

| STG | Debug | Release |
| --- | --- | --- |
| STG-1（bypass staging preservation 6/6） | PASS | PASS |
| STG-2（StateIO round-trip 3/3） | PASS | PASS |
| STG-2-A（order transition ceiling） | PASS | PASS |
| STG-2-B（bypass transition ceiling） | PASS | PASS |
| STG-2-C（save/load idempotent） | PASS | PASS |
| STG-2-R（R4/R5/R6 preservation+stability） | PASS | PASS |
| STG-4-1（no stale overwrite） | PASS | PASS |
| STG-6-D1（failure reported, state preserved） | PASS | PASS |
| STG-7-D1（autoGain flag round-trip） | PASS | PASS |
| STG-8-D1（discard terminalizes recovery） | PASS | PASS |
| STG-8-D2（overwrite terminalizes evicted） | PASS | PASS |
| STG-8-D2b（same obligation retained） | PASS | PASS |
| STG-8-D3（pressure signal redrives） | PASS | PASS |
| STG-9-D1 T-01 / T-02 / T-03 / T-04 | PASS | PASS |

**CLOSED 済み項目の regression 発生 = 0**（STOP 条件 12 に該当せず）。
Release harness は全 suite が `: PASS`（`: FAIL` なし、exit 0）。
Debug harness も exit 0 で FAIL 行なし。

### 6.4 Intermittent observation の再発判定

STG-9-D1 Post-Commit Gate で記録した 4 項目:

| 観測項目 | 本 implementation での再発 |
| --- | --- |
| `STG-8-D2`（`checkSTG82OverwriteTerminalizesEvicted`） | **再発なし**（Debug / Release harness とも PASS） |
| `STG-8-D1`（`checkSTG81DiscardTerminalizesRecovery`） | **再発なし** |
| `EpochDomain.h:229` assert 連発 + stack overflow | **再発なし**。かつ本 STG で**根因を修正済み**（§2 / §3）。Variant B で同 assert の再現を確定済み |
| `testRebuildPublishCompletes`（`audio thread did not run`） | **再発なし** |

**STG-9-D1 の intermittent observation 3 件は，本次 STG-10-D1 と非交差のまま記録継続。**
**`EpochDomain.h:229` の 1 件は，本次で独立に根因が特定・修正された**（STOP 条件 16 に該当せず）。
**R-4 diagnostics は本 STG で一切修正していない**（STOP 条件 17 に該当せず）。

---

## 7. Static validation

| # | 検査 | 結果 |
| --- | --- | --- |
| 1 | production 変更ファイル = `src/core/EpochDomain.h` のみ | **PASS** |
| 2 | 変更関数 = `getMinReaderEpoch()` のみ | **PASS**（他 10 関数はバイト一致） |
| 3 | RT path（`enterReader` / `exitReader` / `RCUReader` / `acquireThreadSlot` / `registerReaderThread` / `reserveReaderThread`）に production diff なし | **PASS**（`RCUReader.h` 変更 0、他はバイト一致） |
| 4 | quarantine writer（`quarantineReader` / `unquarantineAllReaders` / `exitReader` の flag writer）に diff なし | **PASS**（バイト一致） |
| 5 | `publishEpoch` / `currentEpoch` / `epochGeneration_` に diff なし | **PASS**（`epochGeneration_` 変更行 0） |
| 6 | reclaim 経路（`tryReclaim` / `drainAll` / `DeferredDeletionQueue.h` / `ISRRetireRouter.*`）に diff なし | **PASS**（バイト一致 / ファイル変更 0） |
| 7 | 追加行の raw `std::atomic` method call | **PASS = 0 件** |
| 8 | 追加行の `memory_order_*` | **PASS**（`memory_order_acquire` 1 件のみ。既存パターンと同一。変更された memory ordering 0） |
| 9 | 新しい `getMinReaderEpoch` 内の atomic アクセス | **PASS**（`convo::consumeAtomic` 3 件のみ。raw `std::atomic` 0） |
| 10 | Coordinator authority / RuntimeWorld ownership / Retire authority / Publish・Crossfade authority | **PASS**（該当ファイル変更 0） |
| 11 | API signature / virtual interface | **PASS**（変更 0） |
| 12 | shutdown authority（`ISRShutdown.h`） | **PASS**（変更 0） |
| 13 | STG-9-D1 files（`AudioEngine.h` / `AudioEngine.Retire.cpp`） | **PASS**（変更 0 → 再 OPEN なし） |
| 14 | R-4 diagnostics（`AudioEngine.Threading.cpp` / `AudioEngine.CtorDtor.cpp`） | **PASS**（変更 0） |

**Static ISR validation = PASS / Raw atomic audit = PASS / Authority audit = PASS。**

### 7.1 RT invariant の再確認（Owner §12）

| 項目 | 状態 |
| --- | --- |
| RT: no wait | **維持**。`getMinReaderEpoch` は NonRT 専用のため変更なし。RT 経路の命令追加 0 |
| RT: no lock | **維持**。lock 追加 0 |
| RT: no allocation | **維持** |
| RT: no delete | **維持**。RT は `reclaim` に到達しない（`retireRT` は `provider_->enqueueRetire` のみ） |
| RT: no ownership | **維持**。ownership 移管機構は不変 |
| RT: **no new policy decision** | **維持**。RC-1 は `getMinReaderEpoch`（NonRT）の条件式のみを変更。`enterReader` に failure policy を追加していない |

**「RT で新しい policy 判断を追加する」禁止の回避**: Candidate A（`enterReader` での flag 検査）は
Audit §9.1 で「`ConvolverProcessor::process()` / `EQProcessor::process()` が
`rootEnterSucceeded()` を検査しないため、enter 失敗時に **epoch 保護なしで共有状態を読む**
（UAF を別 UAF に置換）」として REJECT 済み。RC-1 はその経路を一切採用していない。

---

## 8. 不変条件の実測確認

| 不変条件 | 判定 | 根拠 |
| --- | --- | --- |
| **Retire は Epoch を通る** | **回復** | T10-1: live reader の epoch 5 が `minReaderEpoch` に参加し、epoch 7 の entry は `isOlder(7,5)==false` で解放されない。Release で実証 |
| RT は判断しない | 維持 | RT 経路の関数本体がバイト一致 |
| RT は解放しない | 維持 | 解放は NonRT（`tryReclaim` / 4 drain 系統） |
| Shutdown は完全 Drain | 維持 | T10-4: `activeReaderCount()==0` 前提 → `unquarantineAllReaders()` → `drainAll()` で pending 0・leak 0 |
| 1:1 ownership | 維持 | ownership 移管は不変。変更したのは解放**タイミングの判定** |
| `0x03` を生成しない | 維持 | quarantine writer 変更 0 |
| Publish / Crossfade / Retire authority を増やさない | 維持 | 該当ファイル変更 0 |
| RuntimeWorld ownership を増やさない | 維持 | 該当ファイル変更 0 |

---

## 9. 変更ファイル一式

| 種別 | ファイル | 内容 |
| --- | --- | --- |
| **production** | `src/core/EpochDomain.h` | `getMinReaderEpoch()` の除外条件 1 箇所（+14 / -12） |
| **test（新規）** | `src/tests/STG10ReaderQuarantineTests.cpp` | 454 行。T10-1 / T10-2 / T10-3 / T10-4 |
| **build** | `CMakeLists.txt` | +22 行（`add_executable` ブロック / `add_test` / IPO-OFF / ASAN リスト） |
| **authority** | `ConvoPeq.md` | 全 validation PASS 後に再生成（+498 / -13） |

**CMake wiring の内訳**（`StuckReaderFallbackDrainTests` のパターンを踏襲）:
- `add_executable(STG10ReaderQuarantineTests ...)` ＋ include / link / features / options
- `add_test(NAME STG10ReaderQuarantine COMMAND STG10ReaderQuarantineTests)`
- `set_target_properties(... INTERPROCEDURAL_OPTIMIZATION OFF)`
- `CONVOPEQ_ASAN_TEST_TARGETS` への登録

`EpochDomain` / `AtomicAccess` / `ThreadHash` はいずれも header-only のため、追加 `.cpp` は不要。

**pre-existing かつ STG-10-D1 と無関係な差分（commit してはいけない）**:
`AGENTS.md`(M) / `headroom-proxy-start.ps1`(M) /
`doc/work113/P1-5-IR-P2_STG-8-D1-D3_REPAIR-GATE_20260928.md`(M) /
`.memories/` / `scripts/gen_clangdb.py` / `vc140.pdb` / `doc/ConvoPeqMD/` /
その他未追跡文書。**本 implementation ではこれらに一切触れていない。**

---

## 10. Residual issues（残留・次期項目）

### 10.1 本 STG の scope 外として保持（併合していない）

| # | 項目 | 状態 |
| --- | --- | --- |
| R1 | **R-4 diagnostics**（`collectDrainAudit()` の reclaim 系 field 欠落、`[FAULT]` ログ文言の誤誘導） | **未着手。独立した P2/P3 として保持**（Owner §11） |
| R2 | **RC-2（allocation 時に `quarantineFlags` を reset）** | **不採用。** Audit §9 で「安全性を回復することでは不要」かつ「RT allocation 経路へ 1 store を追加するため scope 外」と判定。liveness 課題として記録 |
| R3 | **Path A（`quarantineReader` の TOCTOU）** | **明示的に未修正**（Owner §2）。RC-1 により S6 は生成経路によらず安全になるため、現状でも安全 |
| R4 | **Candidate C（quarantine state と depth の atomic state word 統合）** | **不採用。** Audit §9 で RT 経路の大規模書き換えと memory ordering 再設計が必要と判定。将来課題 |
| R5 | **`quarantinedReaderCount()` の production 消費者が 0 件** | 未着手。**R-4 diagnostics 領域**と同一系統。observability の欠落 |
| R6 | **erifyReaderInvariants() の呼び出し元 0 件（死コード）** | 未手づき。RC-1 により S6 は検査対象外の状態として本質的に無害化済み |
| R7 | **`RCUReader::enter()` 失敗時に `ConvolverProcessor::process()` / `EQProcessor::process()` が `rootEnterSucceeded()` を検査しない** | **既存の別問題**。RC-1 の適用で顕在化はしない。Audit §9.1 で Candidate A の REJECT 根拠としてのみ記録。**本 STG の scope 外** |
| R8 | **STG-8-D1 / STG-8-D2 / `testRebuildPublishCompletes` の intermittent observation 3 件** | **再発なし。** STG-10-D1 と非交差のまま記録継続 |

### 10.2 Owner への明示事項

1. **旧 assert の除去は RC-1 の不可分部分**として実施した（§3.2）。理由:
   旧 assert の前提は RC-1 により不変条件でなくなり、残置すると T10-1 / T10-2 が
   生成する**正当な** S6 状態で Debug build が abort する。assert の**追加**は行っていない。
2. **negative control を 2 種実施した**（§5）。Variant B で `EpochDomain.h:229` の assert が
   発火することを再現し、これが STG-9-D1 期間に記録された intermittent observation と
   同一の署名であることを確認した。Variant A で**値 oracle のみが欠陥を検出する**ことを確認した。
3. **R3（Path A）は Owner 指示どおり明示的に未修正**である。RC-1 により S6 は無害であるため
   実害はないが、**診断上の stale flag 残存 possibility** は R5 / R6 とともに残る。

---

## 11. STOP 条件の判定

| # | STOP 条件 | 判定 |
| --- | --- | --- |
| 1 | Authority が FRESH でない | 該当せず（baseline 完全一致） |
| 2 | `EpochDomain.h` 以外の production modification が必要 | 該当せず |
| 3 | `getMinReaderEpoch()` 以外の production function の変更が必要 | 該当せず（他 10 関数バイト一致） |
| 4 | RT path の変更が必要 | 該当せず（RC-1.3） |
| 5 | memory ordering の変更が必要 | 該当せず（`consumeAtomic(acquire)` を再利用。追加行の memory_order は acquire 1 件のみ） |
| 6 | quarantine writer の変更が必要 | 該当せず（RC-1.4） |
| 7 | API / virtual interface の変更が必要 | 該当せず |
| 8 | Coordinator / RuntimeWorld / Retire authority に変更が必要 | 該当せず |
| 9 | T10-1 が Release で PASS しない | 該当せず（Release PASS） |
| 10 | T10-3 の正常 quarantine/reclaim が壊れる | 該当せず（T10-3 PASS） |
| 11 | T10-4 の shutdown drain が壊れる | 該当せず（T10-4 PASS） |
| 12 | STG-1〜9 の CLOSED regression が発生する | 該当せず（全 PASS） |
| 13 | Debug / Release の結果が矛盾する | 該当せず（両者 PASS） |
| 14 | raw atomic access が増える | 該当せず（0 件） |
| 15 | production diff が RC-1 の 1 条件変更を超える | 該当せず（`getMinReaderEpoch` のみ） |
| 16 | STG-9-D1 の再 OPEN が必要になる | 該当せず（STG-9-D1 files 変更 0。`EpochDomain.h:229` の真因は別系統） |
| 17 | R-4 diagnostics を同時修正したくなる | 該当せず（未着手で保持） |
| 18 | test oracle を弱めないと PASS できない | 該当せず（value oracle を厳格化。negative control で FAIL を確認） |

**全 18 項目に該当しない。実装と全 validation が PASS したため、Owner へ差し戻す。**

---

## 12. §16 最終報告

```text
STG-10-D1 Repair Implementation
===============================

Authority:
HEAD = 13a80d66e8e58fbcb66e71eb162f8e95fe134eeb
SHA (pre)  = B15640ADCE15A38801DA8192652876EE4CA8EC6113247EEF9EB93E9EA754D7F3  (Owner baseline と一致)
size (pre) = 5,642,888 B / Generated 2026-09-28 22:19:21 / NEWER_SRC_COUNT 0
SHA (post) = E9FBD8F9E47DACA786D1BC6DB9596180A8659961BE34973CE4C4CC738CF29D09
size (post)= 5,662,249 B / Generated 2026-09-29 00:19:43
NEWER_SRC_COUNT = 0
STATUS = FRESH (pre) / FRESH (post)

Production diff:
src/core/EpochDomain.h only          = PASS
getMinReaderEpoch condition only     = PASS
(+14 / -12, 1 file, 1 function; obsolete assert in the same branch removed as
 an inseparable part of the condition change — see §3.2)

T10-1 = PASS   (quarantined + live reader contributes to minReaderEpoch;
                value oracle; detected in negative control WITHOUT any assert)
T10-2 = PASS   (quarantined slot IS reusable — reserveReaderThread==true is the
                PASS condition; its live occupancy is protected)
T10-3 = PASS   (deferred pending -> quarantined still unblocks reclaim;
                genuine non-regression — PASS in both pre-fix and post-fix builds)
T10-4 = PASS   (shutdown unquarantine + drainAll; pending 0, leak 0)

Debug build   = PASS
Debug CTest   = PASS  43/43 (196.50 s, exit 0)
Release build = PASS
Release CTest = PASS  43/43 (154.93 s, exit 0)

Existing STG regression = PASS
  STG-1 / STG-2 / STG-2-A / STG-2-B / STG-2-C / STG-2-R / STG-4-1 / STG-6-D1 /
  STG-7-D1 / STG-8-D1 / STG-8-D2 / STG-8-D2b / STG-8-D3 / STG-9-D1(T-01..T-04)
  all PASS in Debug and Release. 0 intermittent recurrences.

Static ISR validation = PASS  (RT path / quarantine writer / epoch / reclaim /
                              API / authority byte-identical to HEAD)
Raw atomic audit        = PASS  (0 raw std::atomic method calls in added lines)
Authority audit         = PASS  (Coordinator / RuntimeWorld / Retire /
                              Publish / Crossfade / Shutdown authority unchanged)

ConvoPeq regeneration   = PASS  (regenerated after ALL validations passed)
Post-regeneration freshness = PASS  (NEWER_SRC_COUNT 0 / FRESH)

Production changes = 1 file / 1 function / 1 condition
                     (src/core/EpochDomain.h :: EpochDomain::getMinReaderEpoch)
Test changes       = 1 new TU / 454 lines
                     (src/tests/STG10ReaderQuarantineTests.cpp)
CMake changes      = +22 lines (add_executable / add_test / IPO-OFF / ASAN list)
Authority changes  = 1 (ConvoPeq.md regenerated)
Existing oracle changes = 0
test hook additions     = 0

Commit = 0
Push   = 0
staged = 0 files
branch = main...origin/main [ahead 8]

Verdict:
STG-10-D1 Repair Implementation = PASS
```

**COMMIT / PUSH は未実施。Owner へ差し戻す。**
