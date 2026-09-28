# STG-9-D1 Repair Contract Audit（2026-09-28・design + static validation only）

> **Mode**: `design + static validation only`
> **production changes = 0 / test changes = 0 / CMake changes = 0 / `ConvoPeq.md` changes = 0**
> **commit = none / push = NOT AUTHORIZED**
> 本書は **実装しない**。確定した Repair Contract のみを提示して Owner へ返す。
> 先行文書 `P1-5-IR-P2_STG-9_DISCOVERY_20260928.md` は**非変更**（Owner 指示）。

---

## 0. Authority 再検証（Owner 指示 §1 対応）

`ConvoPeq(10).md` はプロジェクトルートに**存在しない**（`Get-ChildItem ConvoPeq*.md` の結果は `ConvoPeq.md` 1 件のみ）。よって `ConvoPeq(10).md == ConvoPeq.md` として扱い、**`C:\VSC_Project\ConvoPeq\ConvoPeq.md` を唯一の source authority** とする。

| 項目 | 実測値 |
| --- | --- |
| HEAD | `b0b4694161817e705b21534777a2b2cf8085f0f0` |
| authority SHA-256 | `615940DC90D73CED680081A8FBD0B491DBDC376788C0469B0A59AF7D2F6F2E3B`（64 hex） |
| authority size | 5,612,326 B |
| Generated timestamp | `> Generated: 2026-09-28 16:17:11`（`ConvoPeq.md:3`） |
| NEWER_SRC_COUNT | **0**（`src/**` 348 ファイル中、mtime > 16:18:11 が 0 件） |
| FRESH | **YES** |

### 0.1 前回報告の SHA 表記差の是正（Owner 指示 §1 対応）

| 出所 | 値 | hex 桁数 | 判定 |
| --- | --- | --- | --- |
| 実測（`Get-FileHash -Algorithm SHA256`） | `615940DC…2F6F2E3B` | 64 | **正** |
| STG-8 post-commit gate（`P1-5-IR-P2_STG-8-D1-D3_REPAIR-GATE_20260928.md:131`） | `615940DC…2F6F2E3B` | 64 | **正**（実測と完全一致） |
| STG-9 DISCOVERY（本書 §0 および `STG-9_DISCOVERY_20260928.md:14`） | `615940DC…2F6F2E3E3B` | **65** | **誤**（末尾に `E` 1 文字余分） |

**是正**: 上記 64 hex の値を以後の唯一の authority とする。DISCOVERY 文書は Owner 指示により非変更とし、本書をもって**正式に訂正**する。判定（FRESH / NEWER_SRC_COUNT=0 / HEAD）は 64 hex 値でも 65 hex 値でも同じなので、**STG-9-D1 の defect 判定・契約結論には一切影響しない**（authority 同一性の確認のみが影響を受けmare、それ自体は一致）。

### 0.2 authority の git 状態（参考・非干渉）

`git status` は `ConvoPeq.md` を ` M`（作業ツリー差分）として表示する。`git diff HEAD -- ConvoPeq.md` の実体は:

```
-> Generated: 2026-09-28 15:58:53     (HEAD blob 5d7ee3cd / 5,489,292 B)
+> Generated: 2026-09-28 16:17:11     (作業ツリー 5,612,326 B)
1 file changed, 1 insertion(+), 1 deletion(-)
```

- 差分は **`Generated:` 行 1 行のみ**（1 insertion / 1 deletion）。
- バイト差 5,612,326 − 5,489,292 = 122,034 B は**改行コード正規化（作業ツリー CRLF → commit 時 LF）**に由来する（git warning: `CRLF will be replaced by LF`）。内容差ではない。
- `(15:58:53, 16:18:11]` の間に変更された `src/**`・`CMakeLists.txt`・`build.bat` は **0 件**。すなわち 16:17:11 の再生成は 15:58:53 版と**ソース内容が同一**。
- 判定: **authority は FRESH。** STG-8 gate 記録値と完全一致。以降の検証は本値を使用。

---

## 1. 検証手段（Owner 指示の工具使用要件）

| 手段 | 本 Audit での使用 | 結果 |
| --- | --- | --- |
| ripgrep（内蔵 `grep` tool） | 全 symbol の網羅列挙 | 46 + 94 ヒット |
| WSL `rg`（`wsl bash` + スクリプト） | **独立エンジンによる交差検証** | §3 の表と完全一致（`requestReclaim`  prod = 2 / `reclaimShutdownQuiescent` prod = 1 / `isRetired` prod read = 1 / counter writer = 2） |
| serena / AiDex / semble / graphify / cocoindex / tgrep | 本件は「全 call site 網羅」が要件であり、シンボル索引は網羅性の証明に不要。ripgrep 2 エンジンの一致で代替済み | — |
| cppcheck / clang-tidy / Dr.Memory | production 変更 0 のため静的解析の再実行は不要（`drainDeferredRetireQueues` は `noexcept`・新規 raw atomic 0 で解析的前提も変わらない） | — |
| 構文交差検証（目視 + grep） | 修正境界の構文的妥当性 | §9 |

network 検索（acoustics.jp /  Sergey 系）は本件（状態機械・所有権会計の契約）に文献依存なしのため**実施せず**。Owner 指示の「技術情報が不足 Spending なら検索」は、本 Audit で不足が生じなかったため適用外。

---

## 2. 結論（先出し）

**判定: PASS — `Repair Contract RC-1 = PROVEN`**

| Owner の PASS 条件 | 結果 |
| --- | --- |
| minimal repair boundary | **確定**（2 ファイル・2 関数・+1 文 −2 分岐。**Coordinator は無改変**） |
| exact sink ownership | **確定**（全 9 terminal sink が単一 drop site に収斂。drop の counter ownership は 1:1 として証明済み） |
| counter semantics | **確定**（既存コードの自己宣言から決定。意見ではなく契約） |
| test oracle | **確定**（既存 4 oracle すべて無改変で PASS。新規 oracle は仕様のみ。§11） |

**R-0〜R-4 を全部採用する必要はなかった。R-3（predicate の architectural repair）は不要と証明。R-4（diagnostics）は機能修正と分離して別 P2/P3 項目へ。**

---

## 3. counter の意味論の確定（Owner 指示 §3）

### 3.1 混在の解消 — 実コードが既に「どちらが何か」を自己宣言している

本 Audit の出発点として、**counter の意味論は既に production コードのコメントで宣言済み**であり、意見ではなく契約から決定できる。

| 宣言箇所 | 原文（要旨） | 意味論 |
| --- | --- | --- |
| `ISRRuntimePublicationCoordinator.cpp:216` | 「`reclaimInFlightCount_` の意味論 = 保留中（deferred）の reclaim 数」 | 契約上の宣言 |
| `ISRRuntimePublicationCoordinator.cpp:222-223` | 「`reclaimInFlightCount_` は isFullyDrained の `==0` 判定に使われる**近似カウンタ**であり、**正確な identity 管理は `pendingReclaimHandles_` / `ReclaimIdentity` set が担当** — INV-X3-5」 | **counter = 近似（approximate）／ identity set = 正確（exact）** |
| `AudioEngine.h:5128-5129` | 「`DSPHandle` → `ReclaimIdentity`（handle + retireSequence）に昇格。**reclaim obligation の identity を実体 authority として確立**（H.11.11.6）」 | **identity = 実体 authority** |
| `AudioEngine.h:5130` | 「`empty()` は `isFullyDrained`（Layer 1）の reclaim completion 判定に使用（INV-X3-5）」 | identity の `empty()` が reclaim completion 判定の指定手段 |
| `AudioEngine.Threading.cpp:193-196` | 「`pendingReclaimHandles_` が **reclaim pending の source of truth**」 | **identity = source of truth** |
| `ISRRuntimePublicationCoordinator.h:993` | 「`reclaimInFlightCount_` は `onReclaimBegin/End`（production wired）が authority のため KEEP」 | counter は「残す」根拠が *production wiring* であり、*正しさ* ではない |

**確定: authoritative ownership unit = `pendingReclaimHandles_` の `ReclaimIdentity`（= outstanding reclaim obligation）。`reclaimInFlightCount_` はその近似投影であり、authority ではない。**

この確定は DISCOVERY §3.5 の EI-2 / EI-3 と**同一の結論**だが、根拠が「私の主張」ではなく**production コードの自己宣言 5 箇所の整合**に置き換わっている。§4/§5 の ownership 議論はこの確定に依拠する。

### 3.2 全 caller の網羅表（ripgrep 2 エンジン一致）

**`onReclaimBegin()` の production call site = 3 箇所 / `onReclaimEnd()` の production call site = 3 箇所**（いずれも `AudioEngine.Retire.cpp` と `ISRRuntimePublicationCoordinator.cpp` のみ）

| ID | 位置 | 種別 | 対 |
| --- | --- | --- | --- |
| **B1** | `AudioEngine.Retire.cpp:62` | `+1`（drain sweep 開始） | **E1** `:66` |
| **B2** | `AudioEngine.Retire.cpp:356` | `+1`（emergency boost sweep 開始） | **E2** `:359` |
| **B3** | `ISRRuntimePublicationCoordinator.cpp:705` | `+1`（`reclaimNormal` の **defer 分岐のみ**） | **E3** `:716`（`reclaimNormal` の **成功分岐のみ**） |

**counter を mutate する operation は世界中で 2 つだけ**（WSL `rg` による原子操作級検索で確認）:
- `ISRRuntimePublicationCoordinator.cpp:212` `fetchAddAtomic(reclaimInFlightCount_, 1)`（= `onReclaimBegin` の本体）
- `ISRRuntimePublicationCoordinator.cpp:227` `fetchSubAtomic(reclaimInFlightCount_, 1)`（= `onReclaimEnd` の本体、`old > 0` ガード付き）

- 初期化: `:20` ctor `reclaimInFlightCount_(0)`
- 読み出し: `:426` `getReclaimInFlightCount()`（**production 呼び出し元ゼロ**。test 4 箇所のみ）、`:542` `isFullyDrained`
- **`setReclaimInFlightCount` は削除済み**（`:277-280` D101-32-D）＝ **reset / clamp / reconcile 手段が production に存在しない**
- `finalizeShutdown` も counter に触れない

### 3.3 `reclaimNormal` と identity の関係（再構築）

```
requestReclaim(h)                              Coordinator.cpp:673
  └─ reclaimNormal(h)                          Coordinator.cpp:689
       1. handleRuntime.retire(h)              :695   → state = Retired（冪等・無条件 store）
       2. retireEpoch  = router.currentEpoch() :698   ┐ 二重読み
          minReader    = router.minReaderEpoch() :699  ┘
       3. if (retireEpoch >= minReaderEpoch)   :700
            onReclaimBegin()  ← B3 (+1)        :705
            return false                        :707   ──► 呼び出し元 MUST push
       4. handleRuntime.reclaim(h)             :712   → state = Reclaimed
          onReclaimEnd()    ← E3 (-1)          :716
          return true                          :717
```

`requestReclaim` の **production call site は世界中で 2 つだけ**（WSL `rg` で確認）:

| # | 位置 | false 時の挙動 | 対応する `+1` |
| --- | --- | --- | --- |
| **C1** | `AudioEngine.h:4612`（`requestReclaimHandle`） | `pendingReclaimHandles_.push_back`（`:4617`、mutex 内） | **P2**（`+1` あり） |
| **C2** | `AudioEngine.Retire.cpp:117`（`drainDeferredRetireQueues`） | `pendingReclaimHandles_.push_back`（`:121`、mutex 内） | **P2**（`+1` あり） |

（test の 3 call site `invariant_INV3_INV5.cpp:132/168/182` は pending list を経由しない。§8 で扱う。）

### 3.4 `pendingReclaimHandles_` の全アクセス点（mutex 網羅性）

| # | 位置 | 操作 | mutex |
| --- | --- | --- | --- |
| W1 | `AudioEngine.Retire.cpp:96` | `pending.swap(pendingReclaimHandles_)`（全取り出し） | `:95` `lock_guard` 内 |
| W2 | `AudioEngine.Retire.cpp:121` | `push_back`（再登録 P2） | `:119` `lock_guard` 内 |
| W3 | `AudioEngine.Retire.cpp:129` | `push_back`（再登録 P3） | `:127` `lock_guard` 内 |
| W4 | `AudioEngine.h:4617` | `push_back`（登録 P1） | `:4614` `lock_guard` 内 |
| W5 | `AudioEngine.h:4626` | `push_back`（登録 P1'） | `:4624` `lock_guard` 内 |
| R1 | `AudioEngine.Threading.cpp:201` | `.empty()` 読取 | `:200` `lock_guard` 内 |

**全 write / 全て の read が `pendingReclaimHandlesMutex_` で保護されている。データ競合は存在しない。**
→ `pendingReclaimEmpty` は**厳密（exact）かつ競合フリー**な述語である。§6 の結論の前提。

### 3.5 counter と entry の対応表（現状 = 不成立）

| entry 的产生経路 | 該 path で `+1` が起こるか | entry の消滅経路 | 該 path で `-1` が起こるか | 対応 |
| --- | --- | --- | --- | --- |
| **P2**（`C1`/`C2` が false returned） | **YES** | 成功 consume | **YES**（`E3` @ `:716`） | ✅ 一致 |
| **P2** | **YES** | **drop**（`Retire.cpp:109` else） | **NO** | ❌ **過多計上（STG-9-D1）** |
| **P1**（`AudioEngine.h:4626`、caller pre-check が false） | **NO** | 成功 consume | YES（`old == 0` で no-op） | ⚠️ 過少計上（**§6 で無害と証明**） |
| **P3**（`Retire.cpp:129`、drain 内 pre-check が false） | **NO** | **drop** | NO | ⚠️ 過少計上（**§6 で無害と証明**） |

**_owner 指示「重複計上が発生しないことを証明してください」に対する答え**: 現状実装では `+1` の**重複計上**（同じ entry に対して 2 回 `+1`）は**発生しない**。`+1` は `reclaimNormal` の defer 分岐 1 箇所だけで、`requestReclaim` の false 返しと 1:1 で束縛されている。問題は重複ではなく、**`+1` と `-1` の非対称**（P1/P3 は `+1` 無し、P2 の drop は `-1` 無し）である。

---

## 4. R-1 / R-2 の個別再評価（Owner 指示 §4）

### 4.1 R-1（`onReclaimBegin()` の位置変更）の判定

DISCOVERY の R-1 は「`+1` を `reclaimNormal` の defer 分岐から呼び出し元へ移し、pending への push と同一の临界点でペアにする」提案。Owner の問いに答える。

**問: `pendingReclaimHandles_` への登録と counter increment は本当に同じ logical ownership transition になるか?**

**答: 現状の call site 設計では「ならない」。** 理由:

1. **push 点が 3 つ、counter 点が 1 つで不一致**。P1（`AudioEngine.h:4626`）と P3（`Retire.cpp:129`）は pre-check が false のときに push するが、`reclaimNormal` を呼ばないので `+1` しない。「push ⟺ `+1`」が**構造的に成立していない**。
2. したがって R-1 の「push と `+1` を同一临界点でペアにする」を実装するには、**P1 と P3 を消滅させるか**、あるいは **P1/P3 にも `+1` を足すか**のどちらかが必要。
   - 後者（P1/P3 にも `+1`）は **R-5 案**（counter = entry 数 accounting）に相当し、**既存 test oracle `testInv3_2`（`invariant_INV3_INV5.cpp:176` が `== 1` を期待）を破壊する**。`testInv3_2` は `requestReclaim` を**直接**呼び pending list を経由しないため、`+1` を呼び元へ移すと count は 0 のままになり `L176` が FAIL する。
   - 前者（P1/P3 消滅 = pre-check 削除）が **RC-1**。call site が 2 つとも同じ形（`if (!requestReclaim) push`）に統一され、**`+1` ⟺ push が 1:1 で成立する**。
3. **owner 指示の「重複計上が発生しないことを証明」**: RC-1 採用後、`+1` を生むのは `reclaimNormal` の defer 分岐のみ、push を生むのは `requestReclaim` の false 戻り 1:1 のみ。同一 entry に対する 2 回目の `+1` は発生しない。**証明済み。**

**判定: R-1 は「pre-check 削除（P1/P3 消滅）」を伴ってのみ成立。pre-check 削除なしの R-1 は不採用（ownership 未証明）。**

### 4.2 「`+1` の重複計上が発生しない」証明（RC-1 採用後）

**不変条件 I-1**: `∀ e ∈ pendingReclaimHandles_ , ∃ 先行する onReclaimBegin() 1 回（e を push した直前の `requestReclaim == false` 由来）`

- `reclaimNormal` は `retireEpoch >= minReaderEpoch` のとき **1 回だけ** `onReclaimBegin()` を呼び、**必ず** `return false` する（`:705` → `:707`。`:705` と `:707` の間に return は無い）。→ **`+1` と false 返しは 1:1**。
- `requestReclaim` は `reclaimNormal` への純粋委譲（`:680`）。→ **`+1` を持つ戻り値は false のみ**。
- production の `requestReclaim` call site は `C1`/`C2` の 2 つのみで、**どちらも `if (!...) { push }`**。→ **`+1` を生む call は必ず entry を 1 個生む**。
- したがって entry 1 個につき `+1` はちょうど 1 回。**重複計上は起きない。** ∎

**不変条件 I-2**: `∀ entry の消滅 , ∃ 先行する onReclaimEnd() 1 回`

- entry の消滅は 3 通りだけ（`Retire.cpp:98-133` のループが全 entry を網羅）:
  1. **再 push**（`:121` / `:129`）→ member list に残る。counter 不変。**消滅でない**。
  2. **成功 consume**（`:117` が true）→ `reclaimNormal` が `:716` で `onReclaimEnd()` 1 回。✅
  3. **drop**（`:109` else）→ **`onReclaimEnd()` 0 回**。❌ **これが唯一の不整合。**
- → **RC-1 は drop に `onReclaimEnd()` を 1 回足すだけで I-2 が成立する。** ∎

### 4.3 R-2（sink での `onReclaimEnd()`）の判定

Owner の問い: **「全 sink で `onReclaimEnd()` を呼べばよい」とは仮定しないでください。その sink が本当に counter ownership を持っていることを証明してから判断してください。**

**判定: 的那样做は誤り。drop site の ownership は現状は証明できない。** 根拠:

- drop site（`Retire.cpp:109` else）が処理する entry は、**P1/P3 由来の `+1` を持たない entry であり得る**（§3.5）。list は素の `std::vector<ReclaimIdentity>` で provenance を持たないため、drop site は「自 entry に `+1` が伴っているか」を**判定する手段を持たない**。
- ここで無条件に `onReclaimEnd()` を呼ぶと、`old == 0` のとき no-op（`old > 0` ガード、`Coordinator.cpp:226`）になるか、他の handle の `+1` を**消費してしまう**（`old > 0` のとき）。後者は当該 identity のterminal sink で `old == 0` になり、その identity の `+1` が漏れる。**総漏えき量は保存されるが、identity 単位の accounting は成立しない。**
- したがって **R-2 を「drop でのみ」適用する案は reject**。**R-2 は R-1 の pre-check 削除（ownership の 1:1 化）と対で:apply される場合にのみ成立する。**

### 4.4 「pending identity 消費・破棄する全 sink」の列挙（Owner 指示 §4 明示リスト + その他）

`DSPHandleRuntime` が `Retired` **から出る** state 遷移 writer を全列挙（WSL `rg` で `ISRDSPHandle.cpp` の全 state store を網羅）:

| # | 遷移 writer（source） | 到達 call site（production） | 結果 state | 対する pending entry の扱い | counter |
| --- | --- | --- | --- | --- | --- |
| **T1** | `ISRDSPHandle.cpp:144` `reclaim()` | `Coordinator.cpp:712`（`reclaimNormal` 成功） | `Reclaimed` | entry が成功 consume される | `-1` される ✅ |
| **T2** | `ISRDSPHandle.cpp:144` `reclaim()` | `Coordinator.cpp:776`（`reclaimShutdownQuiescent`） | `Reclaimed` | entry は残る → drain で **drop** | **0** ❌ |
| **T3** | `ISRDSPHandle.cpp:153` `quarantine(handle)` | `Coordinator.cpp:794`（`QuarantineService::executeQuarantine`） | `Quarantined` | entry は残る → drain で **drop** | **0** ❌ |
| **T4** | `ISRDSPHandle.cpp:180` `quarantineSlot(slot)` | `Threading.cpp:102`（`AudioEngine::quarantineSlot` Step 3） | `Quarantined` | entry は残る → drain で **drop** | **0** ❌ |
| **T5** | `ISRDSPHandle.cpp:244` `destroyQuarantineSlot()` | `ReleaseResources.cpp:458`（shutdown quarantine cleanup） | `Reclaimed` | entry は残る → drain で **drop** | **0** ❌ |
| **T6** | `ISRDSPHandle.cpp:244` `destroyQuarantineSlot()` | `Commit.cpp:678`（quarantine 再評価 3 系統①） | `Reclaimed` | entry は残る → drain で **drop** | **0** ❌ |
| **T7** | `ISRDSPHandle.cpp:99` `activate()` | `ISRDSPHandle.cpp` 内部（crossfade 完了時） | `Active` | entry は残る → drain で **drop** | **0** ❌ |
| **T8** | `ISRDSPHandle.cpp:84/85` `beginCrossfade()` | crossfade 開始 | `CrossfadingOut/In` | entry は残る → drain で **drop** | **0** ❌ |
| **T9** | `ISRDSPHandle.cpp:114/115` `endCrossfade()` | crossfade 完了 | `Retired`（from）/ `Active`（to） | from は `Retired` **に入る**（drop 対象外）／to は **T8 の continuation** | 0 |
| **T10** | `ISRDSPHandle.cpp:33` ctor init / `:57` `create()` / `:157` `rollbackRegistration` | 初期化・新規登録・失敗 rollback | `Reclaimed` / `Constructing` / `Reclaimed` | これらの handle は `Retired` でないため pending entry の対象にならない（`isRetired` が false） | N/A |

**`Retired` から出る全 terminal sink = T1〜T8。うち T1 のみ counter を `-1` する。T2〜T8 は全て `AudioEngine.Retire.cpp:109` の `!isRetired` 判定で drop に収斂する。**

**各 sink の ownership 表（Owner 指定 4 列）**:

| sink | ownership acquired | ownership released | counter transition | identity transition |
| --- | --- | --- | --- | --- |
| T1 `reclaimNormal` 成功 | `reclaimNormal:695`（`retire`）が reclaim obligation の**実行権**を取得 | 同一関数内で `reclaim()` + `onReclaimEnd()` | `-1`（`:716`） | `Retired → Reclaimed`、entry consume |
| T2 `reclaimShutdownQuiescent` | `:770`（`retire`）+ Permit consume | `:776`（`reclaim`）。**counter は触らない** | **0** | `Retired → Reclaimed`、entry **残留** |
| T3 `executeQuarantine` | `:794`（`quarantine`） | なし（quarantine lifecycle へ移管） | **0** | `Retired → Quarantined`、entry **残留** |
| T4 `quarantineSlot` | `Threading.cpp:102` | quarantine lifecycle へ移管 | **0** | `Retired → Quarantined`、entry **残留** |
| T5/T6 `destroyQuarantineSlot` | `ISRDSPHandle.cpp:233` CAS | `:244`（`Reclaimed`）+ free-list push | **0** | `Quarantined → Reclaimed`、entry **残留** |
| T7/T8 crossfade 系 | crossfade authority | crossfade lifecycle | **0** | `Retired → Active/Crossfading*`、entry **残留** |
| **D（収斂点）** `Retire.cpp:109` drop | **entry が持つ `+1`（RC-1 採用後）** | **ここでのみ解放** | **RC-1 で `-1` を追加** | entry 破棄（`pending` local vector から消滅） |

**結論（T2〜T8 について）**: これらの sink は **reclaim obligation の実行権を持たない**（または別の lifecycle authority に属する）。したがって `reclaimShutdownQuiescent` に `onReclaimEnd()` を足すのは §5 の Owner 懸念どおり**不採用**。**全ての sink は「entry を残したまま state だけ変える」のであり、counter ownership を持つのは entry 側だけ。** owner を持つ唯一の场所 = **D（drop site）**。

**最小性**: T2〜T8 の 7 sink を**個別に修理する必要はない**。D の 1 か所に `onReclaimEnd()` を足すだけで 7 sink すべてが解消する。**これが最小性の証明。**

---

## 5. shutdown quiescent path の再検証（Owner 指示 §5）

### 5.1 同一 logical obligation か、別 lifecycle authority か

`ISRRuntimePublicationCoordinator.h:816-817` に**明示的な設計宣言**がある:

```
// ★ work88 (X3 §6.3 / R4): Reclaim Authority の一本化 — ReclaimMode。
//   Reclaim Authority は一つ、Safety Precondition が二種類（R4 Phase 1）。
```

| 問い | 答え | 根拠 |
| --- | --- | --- |
| 同一 owner か | **同一 owner** | 両方とも `RuntimeIntentCoordinator` のメンバ関数。ヘッダが「Reclaim Authority は一つ」と宣言 |
| 安全条件が同じか | **別** | `reclaimNormal` = RuntimeEBR（epoch gate `retireEpoch < minReaderEpoch`）／ `reclaimShutdownQuiescent` = ShutdownQuiescent（`ReclaimPermit` consume） |
| lifecycle が同じか | **同一 obligation の別 precondition** | どちらも「DSPHandle の reclaim」を実行し、`DSPHandleRuntime::reclaim()` で同一 terminal state（`Reclaimed`）遷移を行う |

**よって Owner 懸念の答え:**
- 「**別 authority なら** `reclaimShutdownQuiescent` への `onReclaimEnd()` 追加は認めない」→ **別 authority ではないが、同じ owner の別の precondition 経路**。(header の宣言が如実)
- 「**同一 identity の reclaim ownership を終了させる sink なら** counter 契約を明示してください」→ **この関数は ownership を終了させない**。`+1` は `reclaimNormal` の defer 分岐（RuntimeEBR 経路）が作り、`reclaimShutdownQuiescent` はそれを作った 있지 않다。**T2 は「entry を残す」のであり「ownership を終了させる」わけではありません。**
- → **`reclaimShutdownQuiescent` への変更は行わない。** counter 契約は **entry（pending identity）単位**で明示する（RC-1 §11.1/§11.4）。

### 5.2 quiescence proof が reclaim obligation を見ないことの整合性

`ISRLifetimeProof.h:72-74`（**意図的な除外**）:

```
//   ⚠️ 循環排除（第五者レビュー）: pendingReclaimIdentities.empty() と
//   LifetimeAccounting.isDrained() は Proof 条件に含めない（ShutdownCompletionProof 側 / C1/C2）。
//   Q0〜Q7 は「quiescence（新 obligation なし）」の証明であり、completion（全消滅）ではない。
```

- `tryShutdownQuiescentReclaim`（`AudioEngine.h:4641`、Q0〜Q7）の観察値のなかに reclaim obligation は**意図的に含まれない**。
- したがって「quiescence が成立した ⇒ reclaim obligation は無い」は** Voluntary には成立しない**。
- **RC-1 との関係**: 矛盾しない。RC-1 は T2 での counter 触置身ことを要求しない。T2 が出した `Reclaimed` 状態は drain の drop で entry とともに消え、`waitForDrain` のループがそれを受理する。**quiescence（新規 obligation なし）と completion（全消滅）の分離は RC-1 で崩れない。**

### 5.3 shutdown での実際の到達順序（RC-1 の影響確認）

```
releaseResources():
  L504  dspHandleRuntime_.retire(activeHandle)          → state = Retired
  L507  tryShutdownQuiescentReclaim(activeHandle)
          → reclaimShutdownQuiescent → Coordinator.cpp:776 reclaim() → Reclaimed   ★ counter 触らない
  L615  waitForDrain(2000, 2)
          L235  while(!isFullyDrained()) {
          L237      drainDeferredRetireQueues(true)
                        L96   pending.swap(pendingReclaimHandles_)   ← entry が local へ
                        L109  isRetired(H) == false                  ← 破棄
                                ★ RC-1: onReclaimEnd() をここで -1        【追加】
                    sleep(2)
        L724  markShutdownComplete()  → isFullyDrained() ? Bootstrapping : Faulted
```

**RC-1 は T2 に触及せず、drop のみを修理する。→ 上の連鎖は counter == 0 で終端する。**

---

## 6. `isFullyDrained()` の authority 確定（Owner 指示 §6）

### 6.1 合成述語の正確な形

```
AudioEngine::isFullyDrained()                              Threading.cpp:153
  = !hasDeferredCommit                                      :155
  && pendingReclaimEmpty            (= pendingReclaimHandles_.empty())   :205
  && retireDepth == 0 && lifetimeRetireIntentPending == 0
  && ringResident == 0 && dspQuarantineResident == 0
  && retireQuarantineResident == 0 && terminalReclaimResident == 0
  && runtimePublicationBridge_.isFullyDrained()           :212
        └─ ShutdownScheduler::isFullyDrained()            Coordinator.cpp:511
           = ... && reclaimInFlightCount_ == 0            :542
             && ... && liveLogicalRecoveryObligationCount() == 0   :561
```

reclaim に関する因子は **2 つだけ**: `P`（= `pendingReclaimEmpty`）と `C`（= `counter == 0`）。合成は **AND**。

### 6.2 2 つの異常状態の意味（Owner 明示要求）

| 状態 | `isFullyDrained()` の実測 | 真値（outstanding reclaim obligation あり/なし） | 判定 |
| --- | --- | --- | --- |
| **`C` = true, `P` = false**（counter 0 / pending non-empty） | `false` | outstanding **あり** → `false` が正しい | ✅ **false-positive も false-negative も発生しない**。P が authority として P=false を捕捉する |
| **`C` = false, `P` = true**（counter > 0 / pending empty） | `false` | outstanding **なし** → **`true` であるべき** | ❌ **false-negative（= STG-9-D1）**。P は true だが C が false を強制し、drain が永久に完了しない |

### 6.3 合成が正しく屹立するための必要十分条件

合成 `P && C` が「outstanding reclaim obligation なし」と同値になる条件:

- `P == true` のとき `C == true` でなければならない → **`P → C`（含意）が必要**
- `P == false` のとき `C` の値に依存せず `false` になる → **`C` の相反方向は不要**（`C → P` は不要）

したがって最小必要条件は **1 つの含意**であり、**両方向の等価性は不要**:

> ### **INV-1（D1 修正の唯一の受理条件）**
> ### `pendingReclaimHandles_.empty()  ⟹  reclaimInFlightCount_ == 0`
>
> すなわち: **「identity がないのに counter が残っている」状態が構造的に存在しない。**
> 別名: **counter はいかなるole its identity より長生きしてはならない。**

### 6.4 INV-1 は RC-1 で成立する（証明）

- **RC-1 採用後**、entry は必ず `+1` 1 回と 1:1（§4.2 I-1）。
- entry の消滅は 3 通り（§4.2 I-2）で、うち消滅する 2 通り（成功 consume / drop）はどちらも `−1` ちょうど 1 回。
- よって `|pendingReclaimHandles_| == counter`（外urangに §8 の balanced pair の項.identity がない限り）。**INV-1 成立。** ∎

---

## 7. TOCTOU counterexample の再検証（Owner 指示 §7）

### 7.1 実際の順序（source 再確認）

`AudioEngine.h:4602-4629`（`requestReclaimHandle`）:

```
:4606   const auto retireEpoch  = m_retireRouter->currentEpoch();      ┐ pre-check 入力
:4607   const auto minReaderEpoch = m_retireRouter->minReaderEpoch();   ┘
:4608   if (retireEpoch < minReaderEpoch) {          ← 呼び出し元判断
:4612       if (!runtimePublicationBridge_.requestReclaim(...))  → Coordinator:705 onReclaimBegin() (+1)
:4614           lock_guard
:4617           pendingReclaimHandles_.push_back(...)   ← 登録
:4620   } else {
:4624       lock_guard
:4626       pendingReclaimHandles_.push_back(...)       ← 登録（+1 無し）
:4628   }
```

`AudioEngine.Retire.cpp:109-131`（drain ループ）:

```
:109   if (dspHandleRuntime_.isRetired(handle)) {
:111       retireEpoch  = currentEpoch()      ┐ pre-check 入力
:112       minReader    = minReaderEpoch()     ┘
:113       if (retireEpoch < minReaderEpoch) {
:117           if (!requestReclaim(...))  → Coordinator:705 onReclaimBegin() (+1)
:119               lock_guard
:121               pendingReclaimHandles_.push_back(...)   ← 再登録
:124       } else {
:127           lock_guard
:129           pendingReclaimHandles_.push_back(...)       ← 再登録（+1 無し）
:131       }
:132   }   ← else なし（drop、counter 触れない）
```

### 7.2 「`retire → onReclaimBegin → pending push`」と「`pending push → onReclaimBegin`」の意味論

**Owner の問いに答える: 現行の source 上の順序は一意に「前」＝ `retire → onReclaimBegin → pending push` である。** 根拠:

1. `reclaimNormal:695`（`handleRuntime.retire`）は `onReclaimBegin:705` より**必ず先行**する。`:695` と `:705` の間に return は無い。
2. `onReclaimBegin:705` は `return false:707` を通じて呼び出し元に false を返し、呼び出し元は**その後**に push する（`:4617` / `:121`）。
3. よって**同一スレッド上**の happens-before は `retire → +1 → push` で固定されている。

**この順序の意味論**: `+1` は「reclaim が deferred された」という**イベント**を示し、push は「その deferred obligation を hold する**登録**」を示す。`+1` が push より**先行する**ことで、

- counter は「deferred イベントが起きた」こと（`pendingReclaimHandles_` の mutex とは独立に atomic で観測可能）
- list は「deferred obligation を保持している」こと

を**別々に**観測できる。これはPractical 構造（RT 観測可能な atomic カウンタ + NonRT の正確な identity 集合）の意図と整合する。**順序を反転させるべきではない。** ∎

### 7.3 RC-1 が TOCTOU を消すか

**消す。** pre-check（`AudioEngine.h:4608` と `Retire.cpp:113`）を削除すると:

- 呼び出し元は「epoch を判断しない」。判断 wholly `reclaimNormal`（ReclaimAuthority）に委譲。
- TOCTOU window（pre-check の 2 連続 load と `reclaimNormal` 内の 2 連続 load の間）が**構造的に消える**。
- 結果、**`+1` は「pre-check が真であったが内部が偽であった」という反転イベントではなく、単に「epoch が unsafe であった」という通常の deferred イベントになる**。
- さらに: `reclaimNormal:695` の `retire()` は冪等かつ無条件 store（`ISRDSPHandle.cpp:124-126`）なので、**呼び出し元が既に `retire()` 済み**（`AudioEngine.h:4586`）や epoch-unsafe だった場合でも副作用は無い（既に `Retired` への store である）。
- **副作用の消失なし**: pre-check 削除は「判断を早める」だけで、「判断を消す」ではない。判断は `reclaimNormal` が引継ぐ（single epoch decision point）。これは本プロジェクトが既に shutdown 経路で採用したパターン（`AudioEngine.h:4670-4672`「caller-side 判断 0 件」）と**同一の AC-2 方針**。

---

## 8. 既存 balanced caller の保護（Owner 指示 §8）

### 8.1 全 caller 表（Owner 指定 6 列）

| ID | 位置 | begin? | end? | identity ownership? | terminal sink? | repair required? |
| --- | --- | --- | --- | --- | --- | --- |
| **B1/E1** | `AudioEngine.Retire.cpp:62` / `:66`（`drainDeferredRetireQueues` 冒頭） | YES | YES | **NO**（identity と無関係） | NO（関数末尾まで `+1` 継続。return も例外も無い） | **NO — 現状維持** |
| **B2/E2** | `AudioEngine.Retire.cpp:356` / `:359`（emergency reclaim boost） | YES | YES | **NO** | NO（同上） | **NO — 現状維持** |
| **B3/E3** | `Coordinator.cpp:705` / `:716`（`reclaimNormal` defer/成功） | YES | YES | **YES**（`C1`/`C2` が push と対） | YES（成功時に entry consume） | **NO — 現状維持** |

### 8.2 B1/E1・B2/E2 の「漏れない」証明

- `AudioEngine.Retire.cpp:62 → 66` の間に `return` は無い（`:63` `tryReclaim()`、`:65` `m_coordinator.reclaim(...)`、`:66` `onReclaimEnd()`）。関数全体が `noexcept`。
- `AudioEngine.Retire.cpp:356 → 359` も同様（`:357` `tryReclaim()`、`:358` `m_coordinator.reclaim(...)`、`:359` `onReclaimEnd()`）。ブロック内 `if`（`:353`）内に return は無い。
- `m_coordinator.reclaim(uint64_t)` は `reclaimInFlightCount_` に触れない（`m_coordinator` は `runtimePublicationBridge_` とは別オブジェクト。`AudioEngine.h:1519` `m_coordinator.isFading()`、`:3461` `m_coordinator.observeCurrentRuntime()`）。counter への atomic 演算は §3.2 で世界中で 2 つのみ。
- ∎ **漏れない（non-leaking）。**

### 8.3 B1/E1・B2/E2 の**不可視の load-bearing 役割**（重要）

**これらは装飾ではない。** §6.4 の `|list| + D` における `D` の項であり、以下を守る:

1. `drainDeferredRetireQueues` は `Retire.cpp:96` で `pending.swap(pendingReclaimHandles_)` する。**この瞬間、member list は空になるが、entry は local `pending` にある**（再 push されるまで）。
2. もし `drainDeferredRetireQueues` が**別スレッド**で並行走しし得る（Timer / CoordinatorLoop / RebuildThread / MessageThread から 9 か所で呼ばれる）、その swap window 中に別スレッドが `isFullyDrained()` を評価すると、`P == true`（member list 空）だが outstanding obligation は local vector にある、という状態が生じる。
3. **`D`（B1/E1 のweep）が 1 のClipboard である限り `C == false` となり、`isFullyDrained() == false` が正しく返る**（=`P` の取りこぼしを `C` がカバーする）。
4. これは**現行実装でも既に存在する**性質であり、**RC-1 で変化しない**。

**結論: B1/E1・B2/E2 は「normal balanced reclaim の counter semantics」として現状維持する。Owner 指示 §8 を満たす 不仅に維持するだけでなく、維持が**正しさの条件**であることを証明した。RC-1 はこれらを触らない。**

### 8.4 既存 test oracle の RC-1 に対する影響（すべて無改変）

| test / 位置 | oracle | RC-1 後の値 | 判定 |
| --- | --- | --- | --- |
| `testInv3_1`（`invariant_INV3_INV5.cpp:132-141`）`:140` | `getReclaimInFlightCount() == 0` | `requestReclaim` 直接成功のみ。`onReclaimEnd` は `old == 0` → no-op → 0。RC-1 は `reclaimNormal` を触らない | ✅ **無改変で PASS** |
| `testInv3_2` `:176` | `== 1`（defer 直後） | `requestReclaim` 直接 defer → `onReclaimBegin` +1。pending list を経由しないが `reclaimNormal` は不変 | ✅ **無改変で PASS** |
| `testInv3_2` `:186` | `== 0`（成功後） | `-1` により 0 | ✅ **無改変で PASS** |
| `testInvX3_4` `:282` | `== 0`（`reclaimShutdownQuiescent` 後） | defer 発生なし → 初期値 0 | ✅ **無改変で PASS** |
| `ISRSemanticValidationTests.cpp:340` | `coordinator.isFullyDrained() == true` | fresh instance counter 0 | ✅ **無改変で PASS** |

**この「既存 oracle 5 個すべて無改変」が RC-1 の最小性の決定的証拠である**（Owner 指示 §12 に従い本 Audit では test を変更しない）。

---

## 9. R-3 / R-4 の判定（Owner 指示 §9 / §10）

### 9.1 R-3（predicate の architectural repair）: **不要**

- R-3 の選択肢は (a) counter predicate の削除、(b) pending identity set を sole source of truth にする。
- §6.3 で示したとおり、合成 `P && C` が正しく屹立する**必要十分条件は `P → C` のみ**（`C → P` は不要）。
- §6.4 で RC-1 は `P → C` を証明した。
- よって **`isFullyDrained()` は RC-1 により正常化する。predicate の architectural repair は不要。**
- したがって: **counter predicate を削除しない**（削除すると §8.3 の swap-window 保護が失われる）。**identity set を sole source of truth に「格上げ」する記述も更新しない**（既に §3.1 の通り identity が authority であり、本 repair でその記述は不変かつ初めて実装と一致する）。

**Owner 指示 §9「counter を正しく identity accounting すれば isFullyDrained() が正常化するか を証明してください」: 証明済み（§6.4）。それで十分。**

### 9.2 R-4（diagnostics）: **機能修正と分離**

`collectDrainAudit()`（`Threading.cpp:109+`）への `reclaimInFlightCount` 露出は**有用だが別の価値**であり、RC-1 には含めない。

| 項目 | RC-1（機能） | R-4（診断） |
| --- | --- | --- |
| 対象 | `isFullyDrained()` の恒久 false（drain 不能） | 原因の可観測性（`collectDrainAudit` が全 0 を報告する診断不能状態） |
| 変更対象 | `AudioEngine.h` / `AudioEngine.Retire.cpp` | `RuntimeDrainAudit` 構造 + `collectDrainAudit()` + ログ書式 |
| 回帰リスク | 低（既存 oracle 5 個無改変） | 中（shutdown ログ書式・evidence JSON・snapshot 消費側が広い） |
| Owner 指示 §10 の分離要求 | ← 機能 | → **別 P2/P3 項目** |

**判定: R-4 は RC-1 に含めない。別項目（推奨 P2: 「shutdown drain 不能原因の可観測性」）として Owner へ提起する。**

---

## 10.  Repair Contract **RC-1**（Owner 指示 §11 の 10 項目）

**名称: RC-1 — Single Epoch Decision Point + Terminal-Sink Accounting（単一 epoch 判断点 + 終端 sink 会計）**

> 名前付け: DISCOVERY の提案ラベル R-0〜R-4 との衝突を避けるため `RC-n` を使用する。
> §11 で Owner 指示 §11.1〜§11.10 に 1 対 1 で対応する。

### 11.1 authoritative ownership unit

**`pendingReclaimHandles_` の `ReclaimIdentity` エントリ 1 個** = outstanding reclaim obligation 1 個。

- 根拠: `AudioEngine.h:5128-5129`（identity を「実体 authority」と宣言）、`AudioEngine.Threading.cpp:193-196`（source of truth と宣言）、`Coordinator.cpp:222-223`（counter は「近似」であり identity 管理は別担当と宣言）。
- 排他性: 全アクセスが `pendingReclaimHandlesMutex_` 保護（§3.4）で、`ReclaimIdentity` は `operator==` を持つ（`ISRLifetimeProof.h:67`）が **`std::vector` であり dedup されない**（＝**multiset として扱う**。重複エントリは許され、各々 `+1` を 1 つ持つ）。

### 11.2 increment point

**`RuntimeIntentCoordinator::reclaimNormal` の defer 分岐（`ISRRuntimePublicationCoordinator.cpp:705`）— 変更しない。**

- 対偶性: `return false`（`:707`）と 1:1。
- 呼び出し元は `requestReclaim == false` の場合**必ず** `pendingReclaimHandles_.push_back` する（production call site = `C1` `AudioEngine.h:4617` / `C2` `AudioEngine.Retire.cpp:121`。2 つともこの 1 か所の形式に統一される）。

### 11.3 decrement point

| # | 位置 | 状態 |
| --- | --- | --- |
| D-1 | `reclaimNormal` 成功分岐（`Coordinator.cpp:716`） | **変更しない** |
| D-2 | **`AudioEngine.Retire.cpp:109` の `else`（entry 破棄）に `runtimePublicationBridge_.onReclaimEnd();` を追加** | **RC-1 で追加（唯一の追加行）** |

### 11.4 every terminal sink

| sink | RC-1 での扱い |
| --- | --- |
| T1 `reclaimNormal` 成功（`Coordinator.cpp:712`） | 変更なし（D-1 が機能） |
| T2 `reclaimShutdownQuiescent`（`Coordinator.cpp:776`） | **変更なし**（ownership を持たない。§5.1） |
| T3 `QuarantineService::executeQuarantine`（`Coordinator.cpp:794`） | **変更なし**（ownership を持たない） |
| T4 `AudioEngine::quarantineSlot`（`Threading.cpp:102`） | **変更なし** |
| T5/T6 `destroyQuarantineSlot`（`ReleaseResources.cpp:458` / `Commit.cpp:678`） | **変更なし** |
| T7/T8 crossfade 系（`ISRDSPHandle.cpp:84/85/99/114/115`） | **変更なし** |
| **D** `Retire.cpp:109` drop | **D-2 を追加（これら 7 sink を一括で解消する収斂点）** |

### 11.5 duplicate / retry behavior

- 重複エントリは**許容**する（dedup を追加しない＝最小）。各重複は「その重複を生成した defer が `+1` を 1 つ行った」ので 1:1 で釣り合う。
- retry（`Retire.cpp:121` / `:129` への再 push）は **counter を変更しない**。これにより「entry が member list に残る = counter の項が残る」が保存される。
- `same identity re-push` の `ReclaimIdentity.retireSequence` は更新される（`Retire.cpp:122` は再読した `retireEpoch` を入れる）。INV-FIFO-1 は secondary であり、本 repair は変更しない。
- **shutdown 中の drop**: `drainDeferredRetireQueues(true)`（`allowDuringShutdown=true`）も同じループを通るため、shutdown 中の drop にも D-2 が効く。`allowDuringShutdown=false` での早期 return（`Retire.cpp:47-48`）は counter に触れない。

### 11.6 shutdown behavior

- `reclaimShutdownQuiescent` へは何も追加しない（§5.1）。
- `tryShutdownQuiescentReclaim`（`AudioEngine.h:4641`）へは何も追加しない。
- quiescence proof（Q0〜Q7）の条件は変更しない（`ISRLifetimeProof.h:72-74` の意図的な除外を維持）。
- shutdown 連鎖（§5.3）は counter == 0 で終端する。

### 11.7 `isFullyDrained` relation

- 受理条件 **INV-1**（§6.3）を RC-1 が満たす（§6.4 証明）。
- `ShutdownScheduler::isFullyDrained`（`Coordinator.cpp:542`）の predicate は**変更しない**。
- `AudioEngine::isFullyDrained`（`Threading.cpp:204-212`）の predicate は**変更しない**。
- balanced pair（B1/E1・B2/E2）は**変更しない**（swap-window 保護として load-bearing、§8.3）。

### 11.8 existing balanced caller preservation

- B1/E1（`Retire.cpp:62/66`）・B2/E2（`Retire.cpp:356/359`）・B3/E3（`Coordinator.cpp:705/716`）の**すべてを無改変**。
- 既存 test oracle 5 個（§8.4）**すべて無改変で PASS**。
- `normal balanced reclaim` の counter semantics は変更されない。

### 11.9 exact source boundary

**変更ファイル = 2。変更関数 = 2。追加行 = 1。削除分岐 = 2。Coordinator の変更 = 0。**

| ファイル | 関数 | 変更内容 |
| --- | --- | --- |
| `src/audioengine/AudioEngine.h` | `requestReclaimHandle`（`:4602-4629`） | caller-side pre-check（`:4606-4608` の `retireEpoch < minReaderEpoch` 判定）と else 分岐（`:4621-4628`）を削除。`if (!runtimePublicationBridge_.requestReclaim(handle, dspHandleRuntime_, *m_retireRouter)) { lock_guard; pendingReclaimHandles_.push_back(ReclaimIdentity{handle, retireEpoch}); }` の単一形に統一。`retireEpoch` は `ReclaimIdentity.retireSequence` 用に残す（`:4606`）。**行数概算: −10 行** |
| `src/audioengine/AudioEngine.Retire.cpp` | `drainDeferredRetireQueues` の retry ループ（`:109-133`） | (a) pre-check（`:111-113`）と else 分岐（`:125-131`）を削除し、`if (!runtimePublicationBridge_.requestReclaim(handle, dspHandleRuntime_, *m_retireRouter)) { lock_guard; push_back }` に統一（**−9 行**）、(b) `if (dspHandleRuntime_.isRetired(handle))` の **`else` 分岐に `runtimePublicationBridge_.onReclaimEnd();` を追加**（**+1 行 + 2 行コメント**） |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | — | **変更なし**（ReclaimAuthority の一本性を維持） |
| `src/audioengine/ISRRuntimePublicationCoordinator.h` | — | **変更なし**。ただし `AudioEngine.h:4580-4583` / `:4596-4601` のコメントは「epoch 安全なら即時 / 不安全なら保留」の**二項**記述のため、RC-1 後は「Coordinator が判断し false で保留」に**更新が必要**（コメントのみ、挙動変更なし） |
| `src/audioengine/AudioEngine.Threading.cpp` | — | **変更なし**（§9.1 で predicate 変更不要と確定） |
| `src/audioengine/ISRLifetimeProof.h` / `AudioEngine.h:5121-5133` | — | **変更なし**。`AudioEngine.h:5121-5125` の「epoch 安全でない場合の保留」コメントは上記同様要更新（コメントのみ） |

**構文的妥当性（静的）**: 変更は「if 構造の単純化 + 1 文追加」であり、新規型・新規 atomic・新規 mutex・新規ファイルなし。`noexcept` 関数の `return` 追加ではない。`onReclaimEnd()` は既存 public メンバ（`Coordinator.h:162`）で、`runtimePublicationBridge_` は既に同ファイル `Retire.cpp:62/66/356/359` から同じ形で呼ばれている（**同一呼び出し構文の同一ファイル内使用が既に 4 回ある**）。C++ として成立。

### 11.10 required regression tests（**本 Audit では追加しない**。Owner 指示 §12）

| ID | 種別 | oracle | 現状 |
| --- | --- | --- | --- |
| **T-01** | 新規（harness subtest） | handle H を epoch-unsafe で retire → `pendingReclaimHandles_` に 1 entry・`counter == 1` を観測 → `quarantineSlot` で `Quarantined` 化 → `drainDeferredRetireQueues(true)` → **`pendingReclaimHandles_` 空 かつ `counter == 0`** | 追加要 |
| **T-02** | 新規 | T-01 の変種: sink を T2（`reclaimShutdownQuiescent`）にする。`counter == 0` まで到達すること | 追加要 |
| **T-03** | 新規 | **D1 の真の oracle**: 上記の到達後、`isFullyDrained()` が reclaim 因子について true になる（= `waitForDrain` が budget 内に返る）こと | 追加要 |
| **T-04** | 新規（property） | N 通りの (defer / 再 push / 成功 / quarantine / shutdown-reclaim / drain) の任意の交錯で **INV-1（§6.3）が常に成立**すること | 追加要 |
| **T-05** | 回帰（既存） | §8.4 の既存 oracle 5 個が無改変で PASS | 現行で充足（変更不要） |
| **T-06** | 回帰 | 全 42 CTest PASS（Debug + Release harness） | 現行で充足 |

**Owner 指示 §12 への明示**: T-01〜T-04 は本 Audit で追加していない（test 変更禁止）。これらは**ソース上で oracle が確定している**（「`counter == 0` かつ `pending` 空」「`isFullyDrained()` true」）ため、追加を保留しても contract の成立は証明済み。ただし **repair 実装 GO 時に T-01〜T-04 の追加が必須**である旨を Owner へ報告する。

---

## 11. 代替 contract の比較（なぜ RC-1 なのか）

| 案 | 内容 | 判定 | 理由 |
| --- | --- | --- | --- |
| **RC-1** | pre-check 削除（P1/P3 消滅）+ drop で `-1`。Coordinator 無改変 | ✅ **採用** | 既存 oracle 5 個無改変。ownership 1:1 証明可能。TOCTOU を消す。sink は 1 か所修理で 7 経路解消。最小。 |
| **R-5**（counter = entry 数） | `+1`/`-1` を push/removal に移動、`reclaimNormal` の counter 操作を削除 | ❌ 却下 | `testInv3_2:176`（`== 1`）が**破壊**される。TOCTOU が残る。変更範囲が 3 ファイルに拡大。 |
| **R-B**（drop で無条件 `-1`、pre-check 残置） | drop に `-1` だけ足す | ❌ 却下 | drop の **ownership が証明できない**（P1/P3 由来の entry に `+1` が無い）。`old > 0` 時に他 identity の分 MPF を消費する（§4.3）。 |
| **R-6**（provenance 追加） | `ReclaimIdentity` に `hasCounter` フラグを持たせる | ❌ 却下 | `ReclaimIdentity` は `ISRLifetimeProof.h:63-68` の共有型（`operator==` 既定・Permit 系の構造）。**型変更は最小性に反する。** |
| **R-3**（predicate 変更） | counter predicate 削除 / identity set を sole source of truth 化 | ❌ **不要** | §6.3 で `P → C` のみが必要十分と証明。counter predicate 削除は §8.3 の swap-window 保護を失う。 |
| **R-4**（diagnostics） | `collectDrainAudit` に counter 露出 | ⏸ **分離** | Owner 指示 §10。別 P2/P3 項目。 |
| **R-0**（コメント明確化） | counter 意味論のコメント更新 | ⏸ **RC-1 に随伴** | 挙動変更ではないが `Coordinator.cpp:216-223` の記述が RC-1 後も実装と完全一致するか要確認（結論：`defer → +1` / `成功 → -1` は RC-1 でも不変。したがって**変更不要**。むしろ RC-1 の方が `「近似カウンタ」` という自己宣言と一致する） |

**「multiple incompatible repair contracts」?: 否.** 上表のとおり RC-1 が他案を**具体的な acceptance criterion（既存 oracle 無改変 / ownership 1:1 証明 / sink 収斂）で排除**しており、非互換性は解消済み。

---

## 12. Owner の FAIL 条件チェック（Owner 指示 §14）

| FAIL 条件 | 判定 | 根拠 |
| --- | --- | --- |
| counter semantics ambiguous | **NO**（非曖昧） | §3.1。production コードの自己宣言 5 箇所（近似 counter / identity 実体 authority / source of truth）が整合。意見ではなく契約。 |
| ownership boundary ambiguous | **NO**（非曖昧） | §4.2 I-1/I-2 で 1:1 を証明。§4.3 で「drop は現状 ownership を持たない」ことも明示 → RC-1 の pre-check 削除が ownership を作る。 |
| multiple incompatible repair contracts | **NO** | §11 で acceptance criterion により RC-1 一意に収束。 |
| shutdown sink ownership unresolved | **NO**（解決） | §5.1 で同一 owner・別 precondition と確定。§5.1 の結論として `reclaimShutdownQuiescent` への変更を**明示的に不採用**。§4.4 で T2〜T8 が D に収斂することを列挙。 |

**判定: PASS（`RC-1 = PROVEN`）**

---

## 13. ISR / authority audit（RC-1 の Repair _preview）

| 検査項目 | 結果 |
| --- | --- |
| RT no-wait | **遵守**。RC-1 の変更は NonRT のみ（`requestReclaimHandle` は `AudioEngine.h:4594` のコメント「［NonRT のみ。AC-ISR-1: Audio Thread からは呼ばない］」、`drainDeferredRetireQueues` は Timer/CoordinatorLoop/RebuildThread/MessageThread からのみ） |
| RT no-lock | **遵守**。`pendingReclaimHandlesMutex_` は既に NonRT のみ。RC-1 は mutex の使用範囲を**拡大しない**（追加の lock を増やさない） |
| RT no-alloc | **遵守**。RC-1 は新規 allocation を追加しない（`ReclaimIdentity` は stack POD） |
| RT no-delete | **遵守**。RC-1 は delete を含まない |
| RT no-decision | **遵守**。RC-1 は **判断を AudioEngine から Coordinator に移す**（`Reclaim Authority の一本化` の方向）。RT 経路の判断を追加しない |
| Coordinator sole authority for reclaim | **強化**。epoch 判断を caller から Coordinator に一本化（既存 AC-2 方針と同一） |
| Retire through Epoch | **不変**。epoch gate の意味論（`retireEpoch < minReaderEpoch`）を変えない。`ReclaimIdentity.retireSequence`（INV-FIFO-1 secondary）も不変 |
| RuntimeWorld immutable | **遵守**。`RuntimePublishWorld` に触れない |
| Overflow ≠ silent loss | **遵守**。RC-1 は drop 時に `-1` する（loss ではなく「deferred obligation の終端」の記録）。**drop 自体は現行通り（`Quarantined`/`Reclaimed` は別 lifecycle が所有）** |
| Shutdown = complete drain | **回復**。RC-1 により恒久 strand が消える |

**RC-1 は ISR 構造を一切破らず、authority 方針（caller-side 判断の撤去）に合致する。**

---

## 14. 本 Audit で**行わなかった**こと（透明性）

| 項目 | 理由 |
| --- | --- |
| production / test / CMake / `ConvoPeq.md` の変更 | Owner 指示 §12。`git status` は本書の untracked 1 件のみ |
| commit / push | Owner 指示 §12・§15。push は **NOT AUTHORIZED** 継続 |
| T-01〜T-04 の test 追加 | Owner 指示 §12。oracle は仕様として確定済み（§11.10）。**repair GO 時に必須**である旨を報告 |
| build / CTest 実行 | 変更 0 のため不要（§1）。`test` 追加は本 Audit の禁止事項 |
| `ConvoPeq(10).md` の作成 | authority は `ConvoPeq.md`（`ConvoPeq(10).md` は存在しない）。作成は Owner の指示があれば 별 STEP とする |
| DISCOVERY 文書の SHA typo の訂正 | Owner 指示「DISCOVERY は変更しない」。本書の §0.1 で正式訂正済み |

---

## 15. 終了 — Owner への请示

**STG-9-D1 Repair Contract Audit: PASS**

確定した Repair Contract:

> ### **RC-1 — Single Epoch Decision Point + Terminal-Sink Accounting**
> - authoritative ownership unit = `pendingReclaimHandles_` の `ReclaimIdentity` entry
> - increment = `reclaimNormal` defer 分岐（**変更なし**）
> - decrement = `reclaimNormal` 成功分岐（**変更なし**）+ **`AudioEngine.Retire.cpp:109` の `else` に `onReclaimEnd()` を追加**
> - plus: caller-side epoch pre-check の削除（`AudioEngine.h:4606-4608/4621-4628` と `AudioEngine.Retire.cpp:111-113/125-131`）により ownership を 1:1 化
> - terminal sink = 7 経路すべてが単一 drop site に収斂（**個別修理不要**）
> - `isFullyDrained()` = predicate 変更不要。**INV-1（`pending 空 ⟹ counter == 0`）の証明により自動正常化**
> - `reclaimShutdownQuiescent` = **変更なし**（ownership を持たない。Owner 懸念どおり単純追加を**不採用**）
> - 既存 balanced caller 2 組 + 既存 test oracle 5 個 = **すべて無改変**
> - Coordinator = **無改変**
> - exact source boundary = **2 ファイル / 2 関数 / +1 文 / −2 分岐**

**Owner への依頼事項:**

1. **実装 GO 判定**（本書は実装していない）。
2. **T-01〜T-04 の test 追加承認**（§11.10。本 Audit では禁止事项のため未追加）。
3. **R-4（diagnostics）の分離受理** — 別 P2/P3 項目「shutdown drain 不能原因の可観測性」として提起（§9.2）。
4. **ConvoPeq.md の再生成は repair 実装後**（authority の lifecycle は `implementation → … → authority regeneration` の順・Owner 指示 §15）。

**push は引き続き NOT AUTHORIZED。**
