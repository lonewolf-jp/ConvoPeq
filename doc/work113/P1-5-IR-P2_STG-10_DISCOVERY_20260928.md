# STG-10 Concrete Defect Discovery（2026-09-28）

> **Verdict: Concrete Defect 判定 — PROPOSED（Owner 判定待ち）**
> **本書は read-only discovery である。production code 変更 = 0。test 変更 = 0。CMake 変更 = 0。**
> **Owner GO なしに implementation へ進まない。**

---

## 0. 前提と authority

Owner 指示 §3 に従い、`ConvoPeq.md` / `ConvoPeq(11).md` を**同一の統一 authority** として扱った。

| 項目 | 値 |
| --- | --- |
| HEAD | `13a80d66e8e58fbcb66e71eb162f8e95fe134eeb` |
| authority SHA-256 | `B15640ADCE15A38801DA8192652876EE4CA8EC6113247EEF9EB93E9EA754D7F3` |
| size | 5,642,888 B |
| Generated | 2026-09-28 22:19:21 |
| baseline Generated | 2026-09-28 22:19:21（一致） |
| NEWER_SRC_COUNT | **0** |
| STATUS | **FRESH** |

**FRESH を確認。STOP 条件（latest authority が FRESH でない）に該当しない。**

直前の作業状態: `STG-9-D1 = CLOSED`（commit `13a80d66`、post-commit gate PASS、Push 未実施）。

---

## 1. §4 Bug ledger 再構成

`doc/work113/` 全 Markdown を走査した結果。STG-1/2/4-1/6-D1/7-D1/8-D1/D2/D2b/D3/9-D1 は
**理由なく再 OPEN しない**（§1.1 で照合済み、§1.2 に記録）。

### 1.1 CLOSED 済み（再 OPEN 禁止）

| STG | 主題 | 判定 | commit |
| --- | --- | --- | --- |
| STG-1 | Bypass toggle が user staging gain を破壊 | CLOSED | — |
| STG-2 | Order/Bypass 変更が headroom を mode ceiling 上に残す | CLOSED | — |
| STG-4-1 | Stale overwrite | CLOSED | — |
| STG-6-D1 | SilentIR 失敗報告 + state 保持 | CLOSED | — |
| STG-7-D1 | autoGain flag round-trip | CLOSED | — |
| STG-8-D1 | discard が recovery を terminal 化しない | CLOSED | — |
| STG-8-D2 | overwrite が evicted を terminal 化しない | CLOSED | — |
| STG-8-D2b | 同一 obligation の retention | CLOSED | — |
| STG-8-D3 | pressure signal の再駆動 | CLOSED | — |
| STG-9-D1 | deferred retry を通る reclaim accounting | **CLOSED** | `13a80d66` |

CLOSED 判定の正本は各 `*_POST-COMMIT-GATE_20260928.md`。STG-9-D1 について
commit `13a80d66` と production diff（`AudioEngine.h` + `AudioEngine.Retire.cpp` のみ）を
照合し、**本 STG-10 の候補はこれらの領域と一切交差しない**ことを確認した（§3.4）。

### 1.2 未解決項目の分類

| 区分 | 件数（代表） | 次期 STG の対象になり得るか |
| --- | --- | --- |
| **runtime code の concrete defect** | **1 件（本書 §2 の候補）** | **YES** |
| gate checker / process defect（`RY2` gate checker normaliser、`N2` acceptance-record audit、process-defect exact-token 系、`P1-5-IR-P2_AMENDMENT-1_Decimal-Literal`） | 多数 | NO — runtime code の defect ではない（Owner §5 により Concrete Defect ではない） |
| R-4 diagnostics（`collectDrainAudit` の reclaim 系 field 欠落、`[FAULT]` ログ文言の誤誘導） | 1 系統 | **後続 P2/P3 候補**。STG-9-D1 で明示的に分離済み。severity は本候補より低い |
| measurement observation（`C0_provenance_matrix` STALE、`p1_2_double_compensation_review` observation 等） | 多数 | NO — measurement/characterization 事項。Owner §5 により defect 判定不可 |
| historical observation（work92 系、Step5C*P3-5-FPM* provenance 系） | 多数 | NO |
| intermittent observation（STG-8-D1 / STG-8-D2 / `testRebuildPublishCompletes`） | 3 系統 | NO — STG-9-D1 と非交差。post-commit で再発なし |

**runtime code の concrete defect 証拠を持つ未解決項目は本書 §2 の 1 件のみである。**

> 参考（Owner §5「単なる設計改善案・コードスメル・将来リファクタリングでは
> Concrete Defect と判定しない」に対する区別）:
> `registerReaderThread` に tag 付きオーバーロードがある／`quarantineReader` の
> 戻り値が 3 系統で[bool] に潰されている 之类は**コードスメル**であり本 STG の
> 対象ではない。§2 はいずれも production の実行時安全性に直結する。

---

## 2. 候補

### Candidate: `EpochDomain` の reader quarantine flag が再入場した live reader を
### `getMinReaderEpoch()` から恒久的に除外し、Release ビルドで UAF を生む

| 評価軸 | 内容 |
| --- | --- |
| **Candidate** | `EpochDomain::quarantineReader()` が設定する `kQuarantinedFlag` と、`enterReader()` / `reserveReaderThread()` / `registerReaderThread()` の間に**無効化の経路が存在しない**ため、隔離済みの reader slot が再入場した live reader を `getMinReaderEpoch()` から恒久除外する。Release ビルドでは assert が消滅するため、この除外が UAF に直結する。 |
| **Severity** | **Critical（memory safety）**。Debug では `EpochDomain.h:229` の assert として顕在化、Release では無言で安全境界が破れる。 |
| **Current status** | **未修正・未記録**。production code にこの欠陥を检出する test が 1 件も存在しない（§5）。ledger には「quarantineReader の設計意図」の記述のみあり、本欠陥の記録はない。 |
| **Concrete defect evidence** | §3.1 の source 事実 + §3.2 の到達可能性 + §3.3 の実測 assert 署名 |
| **Affected production path** | `RuntimeHealthMonitor::diagnoseRetireStall` → `onHealthEvent(EVENT_READER_STUCK)` → `AudioEngine::quarantineReader` → `EpochDomain::quarantineReader` → `getMinReaderEpoch` → `DeferredDeletionQueue::reclaim` → `entry.deleter(entry.ptr)` |
| **RT reachability** | **直接 RT 到達**。`ConvolverProcessor::process()`（RT / audio thread、`src/convolver/ConvolverProcessor.Runtime.cpp:240`）が `RCUReaderGuard` により**毎 audio block** で `enterReader()` を呼ぶ。`EQProcessor.Processing.cpp:488`、`ObservedRuntime.h:69`、`SafeStateSwapper.h:151` も同じ経路。 |
| **Ownership impact** | **RuntimeWorld ownership は増やさない。** RT reader が参照中のオブジェクトの**寿命**（reclaim 可能か否か）が誤判定される。ownership の**移管**は無い。 |
| **Shutdown impact** | 隔離済み slot のフラグを明示的にクリアする `unquarantineAllReaders()` は **shutdown のみ**の呼出し（`AudioEngine.Processing.ReleaseResources.cpp:435`）。したがって通常運行中は**恒久状態**。shutdown はむしろこの状態を強制的に解除するため、shutdown 経路の正しさには影響しない。 |
| **Data-loss / crash / UAF impact** | **UAF。** 実質的なデータ loss / silent drop はないが、RT reader が所有する DSP state（delay buffer / filter state / crossfade gain / runtime state）が reclaim により解放され得る。 |
| **Existing test coverage** | **実質ゼロ。** `EpochDomain::quarantineReader` を実 `EpochDomain` に対して呼ぶ test は存在しない（§5）。 |
| **Existing repair history** | なし。本 STG-10 が**最初の記録**である。 |

---

## 3. Concrete Defect の証拠

### 3.1 §6-A. Authority / §6-B. 実装 — source 上の事実

authority（`ConvoPeq.md` / HEAD `13a80d66`）から直接確認した。行番号は HEAD の
`src/core/EpochDomain.h` におけるもの。

#### (1) `quarantineReader` は depth を**一度**読んで、CAS 時に再検証しない

`src/core/EpochDomain.h:279-325`

```cpp
[[nodiscard]] bool quarantineReader(int readerIndex) noexcept override
{
    if (readerIndex < 0 || readerIndex >= kMaxReaders)
        return false;

    auto& slot = readers[static_cast<size_t>(readerIndex)];
    // acquire: enterReader/exitReader の深度変更を観測
    const uint32_t d = convo::consumeAtomic(slot.depth, std::memory_order_acquire);   // :286  読み

    if (d == 0)
    {
        uint8_t expected = kPendingQuarantineFlag;                                    // :293
        if (compareExchangeAtomic(slot.quarantineFlags, expected,
                                  kQuarantinedFlag, ...))                             // :294
            return true;
        expected = 0;
        if (compareExchangeAtomic(slot.quarantineFlags, expected,
                                  kQuarantinedFlag, ...))                             // :302
            return true;                                                              // CAS 成功
        return false;
    }
    ...
}
```

`:286` の read と `:294` / `:302` の CAS の間に `depth` の再検証が無い。**TOCTOU。**

#### (2) `enterReader` は quarantine flag を**一切**確認しない

`src/core/EpochDomain.h:116-141`

```cpp
void enterReader(int readerIndex) noexcept override
{
    if (readerIndex < 0 || readerIndex >= kMaxReaders) return;

    auto& slot = readers[static_cast<size_t>(readerIndex)];
    const uint64_t epoch = currentEpoch();
    convo::publishAtomic(slot.epoch, epoch, std::memory_order_release);   // :127

    const uint32_t previousDepth = convo::fetchAddAtomic(slot.depth, 1, acq_rel);  // :130
    ...
}
```

`quarantineFlags` への参照は **0 件**。`kQuarantinedFlag` が立ったまま `depth` は 0→1 になる。

#### (3) slot の取得経路 2 本とも flag を見ない

- `reserveReaderThread()` `EpochDomain.h:87-114` — `epoch` の CAS のみ。`quarantineFlags` 参照 0 件。
- `registerReaderThread()` `EpochDomain.h:48-85` — `epoch == kInactiveEpoch` を走査して
  `kReservedEpoch` へ CAS。`quarantineFlags` 参照 0 件。
  → **隔離済みで depth==0 の slot は epoch が `kInactiveEpoch` であるため、新規 reader へ
  そのまま貸出され得る。**

#### (4) `getMinReaderEpoch` は flag を優先し、Release では depth を見ずに除外する

`src/core/EpochDomain.h:225-231`

```cpp
const uint8_t flags = convo::consumeAtomic(slot.quarantineFlags, std::memory_order_acquire);
if ((flags & ReaderSlot::kQuarantinedFlag) != 0)
{
    assert(convo::consumeAtomic(slot.depth, std::memory_order_acquire) == 0
        && "quarantined reader must have depth==0");        // :228-229  Debug のみ
    continue;                                              // :230     Release でも除外
}
```

ソースコメント（`:221-224`）はこの不変条件を前提としている:

> `kQuarantinedFlag` 設定時は `depth==0` が不変条件:
> - 即座 quarantine: `depth==0` でのみ設定
> - 遅延 quarantine: `exitReader` で `depth:1→0` 後に昇格

**この前提が (1)(2)(3) により破られている。** さらに `assert` は `NDEBUG`（Release）で消滅するため、
**Debug でしか検出できない**。

#### (5) 除外は reclaim を**攻撃的**側に倒す

- `EpochDomain.h:394` `deferredDeletionQueue.reclaim(getMinReaderEpoch());`
- `src/DeferredDeletionQueue.h:132` `if (isOlder(entry.epoch, minReaderEpoch)) canDelete = true;`
- `DeferredDeletionQueue.h:143-146` `entry.deleter(entry.ptr);`

live reader を除外すると `minReaderEpoch` が**真値より大きくなる**。大きいほど
`isOlder(entry.epoch, minReaderEpoch)` が真になる entry が増える = **解放が増える**。
したがって誤った除外は安全側ではなく**危険側**に作用する。

#### (6) フラグを消せるのは shutdown だけ

`quarantineFlags` への書き込みは全 8 箇所（`:35` 初期化 / `:178` 昇格 / `:294` / `:302` /
`:319` pending 設定 / `:332` 全クリア）。クリアは `unquarantineAllReaders()`（`:328-336`）のみで、
production 呼出し元は `AudioEngine.Processing.ReleaseResources.cpp:435`（shutdown）の**1 箇所のみ**。

→ **一度立ったフラグはプロセス寿命中、二度と消えない。**

### 3.2 §6-B. 到達可能性（Production 経路の逐次追跡）

| # | 段階 | 位置 | スレッド |
| --- | --- | --- | --- |
| 1 | RT reader が毎 block で enter/exit | `ConvolverProcessor.Runtime.cpp:240` `RCUReaderGuard guard(runtimeRcuReader)` | **RT** |
| 2 | `RCUReader::enter()` が `acquireThreadSlot()` で tid を取得 | `RCUReader.h:62-65` | RT |
| 3 | 同一 tid を再利用（`preferredThreadId` 経路） | `RCUReader.h:166-172` | RT |
| 4 | `epochProvider->enterReader(tid)` | `RCUReader.h:65` → `EpochDomain.h:117` | RT |
| 5 | `depth` 0→1、`epoch` = `currentEpoch()` | `EpochDomain.h:127` / `:130` | RT |
| 6 | `exit()` が `preferredThreadId` に tid を保存 | `RCUReader.h:140` | RT |
| 7 | 次の block で 3 に戻り**同一 slot**を再入場 | — | RT |
| 8 | health monitor が `detectStuckReaders(10)` | `RuntimeHealthMonitor.cpp:492` | NonRT |
| 9 | `EVENT_READER_STUCK` 発火（`readerIndex` 付き） | `RuntimeHealthMonitor.cpp:500-508` | NonRT |
| 10 | Timer が `onHealthEvent` を受信 | `AudioEngine.Timer.cpp:1791` | NonRT |
| 11 | `m_retireRouter->quarantineReader(event.readerIndex)` | `AudioEngine.Timer.cpp:1799` | NonRT |
| 12 | `EpochDomain::quarantineReader` | `EpochDomain.h:279` | NonRT |
| 13 | `depth` を 1 回読む | `EpochDomain.h:286` | NonRT |
| 14 | `kQuarantinedFlag` を CAS | `EpochDomain.h:294` / `:302` | NonRT |
| 15 | 次の block で同一 slot に再入場（フラグは残る） | 4-5 に復帰 | **RT** |
| 16 | `getMinReaderEpoch()` が live reader を除外 | `EpochDomain.h:225-230` | 任意 |
| 17 | `reclaim` が誤って解放 | `EpochDomain.h:394` / `DeferredDeletionQueue.h:132,145` | NonRT |

**2 つの独立した到達経路が存在する。**

- **経路 A（TOCTOU・窓あり）**: 手順 13 の read と 14 の CAS の間に RT が 5 を実行する。
  `detectStuckReaders` は `depth > 0` を要求するため（`EpochDomain.h:502` / `:521` / `:524`）、
  検出時点では depth>0 だが、手順 11 までの NonRT hop 中に RT が exit 完了 → depth==0 になり得る。
  その上で 14 が成功し、15 の再入場と競合する。

- **経路 B（窓に依存しない・構造的）**: 14 が `depth==0` 時に成功した後、**同一 tid が
  次の block で必ず再入場する**（手順 6→3 の `preferredThreadId` 再利用）。
  さらに新規 reader が `registerReaderThread()`（手順 3 の別経路）で**隔離済み slot を
  貸出**され得る。**印は一度立つと消えない**ため、この経路は竞争的ではない。

経路 B により、本件は「狭い窓の race」ではなく **一度発生すると恒久的に続く状態破損**である。

### 3.3 §6-E. 実測証拠（既に観測された assert）

`doc/work113/P1-5-IR-P2_STG-9-D1_REPAIR-GATE-R2_20260928.md:241` および
`P1-5-IR-P2_STG-9-D1_POST-COMMIT-GATE_20260928.md:163,177` に記録済み:

> `EpochDomain.h:229` assert 連発後の stack overflow（`0xC00000FD`）1 回

`EpochDomain.h:229` は本候補の (4) の assert そのもの
（`"quarantined reader must have depth==0"`）である。

**したがって既に観測されたクラッシュは、本欠陥の署名である。**
過去記録はこれを「STG-9-D1 の変更と因果経路なし → intermittent observation」と判定しており、
**その判定自体は正しい**（STG-9-D1 の差分は `pendingReclaimHandles_` の会計のみ）。
しかし「STG-9-D1 と無関係」は **「production code に欠陥がない」ことを意味しない**。
本 STG-10 はその別にある真因を特定するものであり、**STG-9-D1 の再 OPEN ではない**。

### 3.4 既存修正との非交差（Owner §6-D）

| 既存 STG | 変更領域 | 本候補との交差 |
| --- | --- | --- |
| STG-8-D1/D2/D2b/D3 | `RuntimePublicationOrchestrator` の deferred slot、`RecoveryAdmissionTable` の obligation | **なし**。本候補は `EpochDomain` の reader slot 寿命 |
| STG-9-D1 | `pendingReclaimHandles_` の +1/−1 会計、epoch 判断の Coordinator 一本化 | **なし**。本候補は `EpochDomain::ReaderSlot.quarantineFlags` |
| STG-6-D1 / STG-7-D1 / STG-4-1 / STG-2 / STG-1 | filter / IR / swap / staging | **なし** |

STG-9-D1 の production diff は `AudioEngine.h` と `AudioEngine.Retire.cpp` の 2 ファイルのみ。
**`src/core/EpochDomain.h` は STG-1〜9 のいずれでも変更されていない。**
`kMaxReaders` / `ReaderSlot` / quarantine 機構は STG-1 以前からの設計 그대로残る。

---

## 4. §6-C. 不変条件への影響（Practical Stable ISR Bridge Runtime）

| 不変条件 | 判定 | 根拠 |
| --- | --- | --- |
| **RT は待たない** | **維持** | `enterReader` / `getMinReaderEpoch` は lock-free atomic のみ。本修正候補も atomic のみに限定され得る |
| **RT は解放しない** | **維持** | 解放は `deferredDeletionQueue.reclaim()`（NonRT）。RT は `reclaim` を呼ばない |
| **RT は判断しない** | **要確認** | `getMinReaderEpoch()` は判断関数。**ただし本候補の主張は「RT の判断が誤る」のではなく「NonRT の隔離が RT の判断入力を破壊する」**。RT 側の判断ロジックは変更しない |
| **Retire は Epoch を通る** | **破壊** | 誤って抬高した `minReaderEpoch` が reclaim を許すため、Retire が epoch ガードを実質的に通過する。**これが本欠陥の核心** |
| **Shutdown は完全 Drain** | **維持** | `unquarantineAllReaders()` が shutdown で全 clear し、以後 drain 可。shutdown 経路は本欠陥の影響を受けない（むしろ回復させる） |
| **Overflow は silent loss しない** | **影響なし** | 本件は overflow ではなく premature reclaim |
| **Publish / Crossfade / Retire authority を増やさない** | **維持可能** | 修正は `EpochDomain` 内部の整合に閉じ、authority の new 認可は増やさない |
| **RuntimeWorld ownership を増やさない** | **維持可能** | 同上 |

**`Retire は Epoch を通る` の 1 項目が破られている。これが本候補を Critical とする理由である。**

---

## 5. §6-E. Tests — oracle 欠落の確認

### 5.1 `quarantineReader` の production コードは test で 1 回も呼ばれていない

```
$ rg -n 'quarantineReader' src/tests
src/tests/D8_2_B_2_Tests.cpp:160:    bool quarantineReader(int) noexcept override { return false; }   // mock
src/tests/PriorityIntegrationTests.cpp:79-80: // ★ Phase3/5連動: quarantineReader → High優先度投入  // コメントのみ
```

- `D8_2_B_2_Tests.cpp:160` は **`IReaderEpochProvider` の mock**であり、実装を呼ばない。
- `PriorityIntegrationTests.cpp:79-80` は**コメント**であり、呼出しコードを含まない。

→ **`EpochDomain::quarantineReader`（`EpochDomain.h:279-325`）の test カバレッジはゼロ。**

### 5.2 既存 test は本欠陥を検出できない

`src/tests/StuckReaderFallbackDrainTests.cpp` は `enterReader` を 4 箇所（`:139` / `:193` /
`:256` / `:331`）で呼ぶが、**いずれも「enter した reader を exit させない stuck reader として
保持する」使用方法**であり、quarantine 後の再入場を単独で検証しない。
`src/tests/AudioEngineHarness/T2Measurement.cpp:122` / `T3Measurement.cpp:119` /
`T4Measurement.cpp:148` も同じく enter 保持型。

**gap の要約**: 「隔離済み slot に新しい reader が入り直す」という状態遷移を
既存 test ならび CMake に oracle が存在しない。

### 5.3 検出可能性の予感（read-only 判定）

本欠陥は Debug では `EpochDomain.h:229` の assert で**既に 1 回実際に発火**している（§3.3）。
したがって test 追加は不要ではなく、**再現可能な oracle を 1 本加えるだけで
回帰として固定できる見込みが高い**。ただし production の nondeterminism に依存せず
**決定論的に生成できる unit-level oracle（EpochDomain 直叩き）を優先すべき**と考える。
これは次期 Repair Contract Audit の議題であり、**本 discovery では test を一切作成していない**。

---

## 6. 想定される修正の形状（**本 discovery では一切実装していない**）

Owner §7 により production code は変更していない。以下は **Audit 段階の入力情報**であり、
Repair Contract ではない。採用可否は Owner GO に依存する。

候補となる方向（いずれも未採用）:

1. **nterReader で quarantine flag を監視し、フラグ設定中は epoch 保護を維持したまま
   入場を拒否 or deferred 扱いにする。** → Fail-Closed だが RT での判断追加になるため
   「RT は判断しない」不変条件の精査が必要。
2. **`quarantineReader` の CAS 成功後に `depth` を再検証し、0 でなければ pending へ降格させる。** →
   TOCTOU (1) のみを閉じる。経路 B（恒久フラグ）は残る。
3. **`kQuarantinedFlag` と `depth>0` の両方が成立する状態を構造的に不可能にする
   （flag と depth を 1 つの atomic state word に統合）。** → 1 と 2 を同時に閉じる。
   ただし reader slot の state word 化は `ReaderSlot` layout を変えるため
   「RuntimeWorld ownership を増やさない」不変条件の精査を要する。
4. **getMinReaderEpoch で「flag 設定済かつ depth>0」を、除外ではなく不変条件違反として扱う**
   （Release でも生き残る fail-closed）。** → 診断は可能だが、安全側 retrofit のみ。

**いずれも Repair Contract Audit で「RT は判断しない」「RT は解放しない」を
壊さないか、生命安全を回復しつつ authority を増やさないかを判定する必要がある。
本 discovery では採用していない。**

---

## 7. §7. Production 変更禁止の遵守

```text
src/**                変更 0
CMakeLists.txt        変更 0
tests                 変更 0
ConvoPeq.md           変更 0（source 変更がないため再生成も不要）
build                 実行 0
git mutation          0
```

HEAD は `13a80d66` のまま。`ConvoPeq.md` の SHA-256 も §0 の値から変化していない。
本 discovery は読み取り専用で完了した。

---

## 8. §9. STOP 条件の判定

| STOP 条件 | 判定 |
| --- | --- |
| latest authority が FRESH でない | **該当せず**（FRESH / NEWER_SRC_COUNT 0） |
| latest bug ledger と source 状態が一致しない | **該当せず**（§1.1 の CLOSED 10 項目を commit と照合し一致。§3.4 で非交差を確認） |
| Concrete Defect が判定できない | **該当せず**（§3 に source 事実・到達経路・実測署名を提示） |

**STOP 条件には該当しないため、本 cycle はここで停止する。**
Owner §2「今回は **Discovery GO のみ**。production code の変更 GO ではない」に従い、
**Repair Contract Audit にも進まない。**

---

## 9. 報告事項

1. **Concrete Defect 判定（PROPOSED）**: `EpochDomain` の reader quarantine flag 恒久化欠陥。
   Severity **Critical（Release で UAF、Debug で `EpochDomain.h:229` assert）**。
2. **本件は intermittent observation の「真因特定」に相当するが、STG-9-D1 の再 OPEN ではない。**
   STG-9-D1 の差分との非交差を再確認済み（§3.4）。
3. **R-4 diagnostics**（`collectDrainAudit` の reclaim 系 field 欠落）は本候補と別系統の
   P2/P3 項目として引き続き保留。**本 STG で併合しない。**
4. **次段階は Owner 指示待ち**（Repair Contract Audit / Owner GO）。本 discovery はそこで停止する。
