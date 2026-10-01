# STG-11-D1 Repair Contract Audit — D1 のみ（2026-09-29）

> **Verdict: D1 = CONFIRMED / Repair Contract = PROVEN（候補 B を推奨）**
> **本書は read-only 監査である。production = 0 / test = 0 / CMake = 0 / ConvoPeq.md = 0 / commit = 0 / push = 0。**
> **Owner が implementation を明示的に GO しない限り、実装には進まない。**
> **D2 / D3 / D4 は scope 外**（既知 finding として保持。D4 との dependency は §11 で判定）。

---

## 1. Authority

Owner 指定: `ConvoPeq(20260928-152951).md == ConvoPeq.md`（統一 authority）。
実 source 検証は repo の現行 `ConvoPeq.md` を基準に実施した。

| 項目 | 値 |
| --- | --- |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| SHA-256 | `E9FBD8F9E47DACA786D1BC6DB9596180A8659961BE34973CE4C4CC738CF29D09` |
| size | 5,662,249 B |
| Generated | 2026-09-29 00:19:43 |
| NEWER_SRC_COUNT | **0** |
| STATUS | **FRESH**（`--check` exit 0） |

**FRESH。監査前後で `src` / `CMakeLists.txt` / `ConvoPeq.md` は未変更。**

---

## 2. D1 defect chain の再証明（Owner 必須項目 1）

STG-11 Discovery §6.1 の chain を source 実読で**再確認**した。行番号は HEAD `f95f524c` におけるもの。

### 2.1 Chain（caller → callee）

```text
caller : EQProcessor::retireEQStateDeferred   EQProcessor.Core.cpp:98-105
         EQProcessor::retireBandNodeDeferred   EQProcessor.Core.cpp:107-114
         ← EQProcessor.Parameters.cpp:29,49,69,93,118,139,173,196,222,257（全 setter）
         ← EQProcessor.Coefficients.cpp:75（updateBandNode / Message Thread）
         ← EQProcessor.Core.cpp:142,148（~EQProcessor）
         ← EQProcessor.Core.cpp:242（resetToDefaults） / :739（param path）
callee : EQProcessor::enqueueDeferredDeleteWithFallback  EQProcessor.Core.cpp:26-68
         ├─ :49  ISRRetireRouter stackRouter(m_epochDomain)     ← スタック上
         ├─ :52  m_retireCoordinator->enqueueRetire(Granted, stackRouter, ptr, deleter, retireEpoch)
         │        └─ RuntimeIntentCoordinator::enqueueRetire
         │            ISRRuntimePublicationCoordinator.cpp:147-172
         │            └─ :160  router.enqueueWithRetry(...)     ← 完全委譲
         └─ :61  stackRouter.enqueueWithRetry(ptr, deleter, retireEpoch, Generic)
                  └─ ISRRetireRouter::enqueueWithRetry  ISRRetireRouter.cpp:303-384
```

### 2.2 D → Q → E → T の各 ownership transition（再確認）

| Stage | 位置 | 昇格先の実体 | 実体の lifetime |
| --- | --- | --- | --- |
| D | `ISRRetireRouter.cpp:322` `enqueueRetire` → `EpochDomain::enqueueRetireTyped`（`EpochDomain.h:406`）→ `deferredDeletionQueue` | `stackRouter.provider_` = **EQ private `m_epochDomain`** の内蔵 D queue | EQProcessor と同命（member） |
| retry | `:327-336` `provider_->tryReclaim()` + `drainEmergencyAndTerminal()` + 再 enqueue | 同上 | 同上 |
| Q | `:342` `m_retireQuarantine.quarantine(...)` | `ISRRetireRouter.h:401` の **member**（inline `entries_`、`kMaxQuarantinedEntries = 512`） | **stackRouter と同命 = 関数 return で消滅** |
| E | `:352` `m_emergencyQuarantine.quarantine(...)` | `ISRRetireRouter.h:403` の **member**（同型・同容量） | **同上** |
| T | `:365` `terminalReclaim(...)` | `ISRRetireRouter.h:405` の **member**（`std::vector<Entry>` growable） | **同上（heap ブロックごと leak）** |

### 2.3 関数 return 後の各 storage の lifetime（再確認）

- `ISRRetireRouter` に**ユーザー定義デストラクタは存在しない**（`::~ISRRetireRouter` を `rg` で検索し 0 件）。
- `RetireQuarantineStore` / `TerminalReclaimAuthority` にもユーザー定義デストラクタは存在しない（同 0 件）。
- したがって `stackRouter` 破棄時、Q/E に残る entry は**記憶ごと消滅**し、T に残る entry は
  **heap ブロックごと leak** する。**いずれも deleter は一度も呼ばれない。**
- `:65-67` は `Success / QueuePressure / TerminalReclaim` をすべて `true` として返すため、
  呼び出し元（14 箇所の setter / dtor）は「ownership 移転成立」と誤認し、
  `m_retireDropCount` も増えない（`Core.cpp:142/148/242/739`、`Coefficients.cpp:76`、
  `Parameters.cpp:30` 等は `false` 時のみ increment）。

### 2.4 shutdown drain の到達先（再確認）

| drain | 対象 | stackRouter の Q/E/T に届くか |
| --- | --- | --- |
| `ReleaseResources.cpp:435` `m_retireRouter->unquarantineAllReaders()` | engine router | **NO** |
| `ReleaseResources.cpp:444` / `:555` `drainAllQuarantineStore()` | engine router | **NO** |
| `CtorDtor.cpp:285-291` `m_retireRouter->drainAll()` / `drainAllQuarantineStore()` | engine router | **NO** |
| `~EQProcessor` `Core.cpp:158-160` `m_epochDomain.tryReclaim(); drainAll(); tryReclaim();` | private D のみ | **NO（Q/E/T を知らない）** |

**到達する drain は存在しない。→ shutdown reachability: FAIL（現状）。**

### 2.5 小結

**D1 status = CONFIRMED。** STG-11 Discovery の主張は source 実読で再現した。
STG-8 / STG-9-D1 / STG-10-D1 との交差なし（`EQProcessor` / `ISRRetireRouter` の domain であり、
`pendingReclaimHandles_` でも `RecoveryAdmissionTable` でも `getMinReaderEpoch` でもない）。

---

## 3. Epoch-domain provenance（Owner 必須項目 2 — 最重要）

**結論を先に述べる: Candidate A（engine shared router）は epoch provenance を破壊する。
単なる置換は安全ではない。**

### 3.1 二重構造の確定（最新 source）

| 要素 | file:line | 内容 |
| --- | --- | --- |
| EQ private domain | `EQProcessor.h:488` | `convo::EpochDomain m_epochDomain;` |
| EQ RT reader | `EQProcessor.h:497` | `convo::RCUReader rcuReader { m_epochDomain };` |
| EQ 注入済み router（未使用） | `EQProcessor.h:492` / `:494` | `m_retireCoordinator` / `m_retireRouter` |
| EQ retire epoch | `EQProcessor.Core.cpp:43` | `m_epochDomain.currentEpoch()`（private counter） |
| EQ retire epoch（state） | `EQProcessor.Core.cpp:103` | 同上 |
| EQ retire epoch（node） | `EQProcessor.Core.cpp:112` | 同上 |
| engine domain | `AudioEngine.h:5015` | `convo::EpochDomain m_epochDomain;`（別オブジェクト） |
| engine router の provider | `AudioEngine.CtorDtor.cpp:39` | `make_unique<ISRRetireRouter>(m_epochDomain, ...)`（engine domain） |

### 3.2 `m_retireRouter`（EQ member）が使用する domain

**`setRetireRouter` の production 呼び出しは 0 件**（`rg` 実測。唯一の同名呼び出しは
`AudioEngine.CtorDtor.cpp:53` の `m_healthMonitor.setRetireRouter` で別クラス）。
したがって `EQProcessor::m_retireRouter` は production では**常に nullptr** であり、
「どの domain を使うか」は未定義である。**仮定してはならない。**

### 3.3 対照実験: ConvolverProcessor は正しい provenance を持つ

| 要素 | Convolver（正） | EQ（現状） |
| --- | --- | --- |
| RT guard | private `runtimeRcuReader`（`ConvolverProcessor.h:1424`）＋ engine `enterGlobalReader`（`Runtime.cpp:217` → engine `m_retireRouter->enterReader`） | private `rcuReader`（`EQProcessor.h:497`）**のみ** |
| engine domain への接触 | **あり**（`enterGlobalReader` 7 箇所: `Lifecycle.cpp:151,238,527`、`Runtime.cpp:117`、`StateAndUI.cpp:426,730`） | **なし**（`enterGlobalReader` / `getRcuProvider` / `enterRcuReader` / `snapshotRcuEpoch` の参照 0 件） |
| retireEpoch | engine `snapshotRcuEpoch()`（`Lifecycle.cpp:320,495`）= `currentRetireEpoch()`（`AudioEngine.Publication.cpp:11-14`） | private `m_epochDomain.currentEpoch()`（`Core.cpp:43,103,112`） |
| retire 先 | engine router（`provider->enqueueDeferredDeleteNonRt`） | stack-local router（private domain） |

### 3.4 Candidate A が provenance を壊す 2 つの独立証明

**証明 1 — epoch 値の mismatch。**
`EpochDomain` は各インスタンスが独立の `globalEpoch`（ctor `globalEpoch(1)`、
`EpochDomain.h:26`）を持ち、独立の cadence で進む。
engine domain の advance（`m_retireRouter->publishEpoch()` 等）と
EQ private domain の advance（`flushPendingEpochAdvance` のみ:
`Core.cpp:204` releaseResources / `:748` param path）は**無関係**。
EQ-private epoch 値（例: 5）を engine D queue に入れても、`isOlder(5, engineMin)` の
比較は**無意味な数値同士の比較**になる。

**証明 2 — reader 可視性の mismatch（UAF）。**
EQ の RT reader は private domain にのみ登録される（`rcuReader { m_epochDomain }`、
`RCUReader.h:159-182` の `acquireThreadSlot` が private `reserve/registerReaderThread` を呼ぶ）。
engine `getMinReaderEpoch()` は EQ private reader を**見ない**。
EQ `process()`（`EQProcessor.Processing.cpp:488-489`）は
`RCUReaderGuard guard(rcuReader); stateSnapshot = loadCurrentState(...)` の下で
`stateSnapshot->nonlinearSaturation`（`:562`）、`->agcEnabled`（`:626`）、
`->filterStructure`（`:868`）、各 `BandNode`（`:660`）を**毎 block 参照**する。
これら（`EQState` / `BandNode`）こそ retire 対象である。
engine 境界で reclaim すれば、**RT 参照中の解放 = UAF**。

**したがって「`stackRouter` → `m_retireRouter`（engine shared）に置換」は安全ではない。
Candidate A = REJECT。**

### 3.5 epoch provenance 判定

```text
epoch provenance: PASS（現状の private-domain provenance は正しい。
                 Candidate A はこれを破壊するため不採用）
```

---

## 4. `setRetireRouter()` の lifecycle wiring 全追跡（Owner 必須項目 3）

| # | 項目 | source evidence |
| --- | --- | --- |
| 1 | declaration | `EQProcessor.h:461-464` `void setRetireRouter(ISRRetireRouter* router)` |
| 2 | member | `EQProcessor.h:494` `ISRRetireRouter* m_retireRouter{nullptr};` |
| 3 | constructor | `EQProcessor::EQProcessor()`（`Core.cpp:119-133`）は `m_retireRouter` に触れない |
| 4 | factory / builder | **存在しない**（`setRetireRouter` の production 呼び出し 0 件） |
| 5 | AudioEngine 側の生成 | `m_retireRouter = make_unique<ISRRetireRouter>(m_epochDomain, ...)`（`CtorDtor.cpp:39`）は **engine の router** であり、EQ には渡されない |
| 6 | production call site | **0 件**（同名の `m_healthMonitor.setRetireRouter` のみ） |
| 7 | `EQProcessor` destruction より前後の router lifetime | 論点自体が不成立（注入が無い）。なお `EQProcessor` は `DSPCore` の member（`AudioEngine.h:895` `EQProcessor eq;`）であり、`DSPCore` は publish ごとに heap 生成（`RuntimeBuilder.cpp:469`、`PrepareToPlay.cpp:263`）され publication pipeline で retire される |
| 8 | `m_retireRouter` の読み取り | **0 件**（代入 `:463` のみ。write-only member） |

**結論: `setRetireRouter` 配線は死んでいる。Candidate A は「既存配線を使う」こともできない。
注入配線自体の新設が必要であり、その新設が §3.4 の mismatch を生む。**

---

## 5. Candidate repair 比較（Owner 必須項目 4）

### 5.1 定義

| 候補 | 内容 |
| --- | --- |
| **A** | engine shared router を使用（`m_retireRouter` に engine router を注入） |
| **B** | EQProcessor 所有 router を member lifetime に延長（`ISRRetireRouter` を member 化し private domain に束縛） |
| **C** | private epoch domain を維持し、その domain に対応する durable retire authority を別途 lifetime 延長（例: EQ 所有の Q/E/T 相当 store を member 化） |

### 5.2 比較表

| 評価軸 | A: engine shared router | B: member router（推奨） | C: private domain + 別 durable authority |
| --- | --- | --- | --- |
| ownership | Q/E/T は engine router に移るが、**関数 return 問題は解消**。ただし後述の provenance 破壊により無効 | **解消**。Q/E/T が EQProcessor と同命になり、dtor 前 drain で回収可能 | 解消可能だが **escalation 段を自作する必要** |
| epoch provenance | **破壊**（§3.4 の 2 証明）。EQ-private epoch を engine counter と比較／EQ private reader が engine min に含まれない → **UAF** | **維持**。provider は `m_epochDomain` のまま。retireEpoch も reclaim 境界も同一 domain | 維持可能（設計次第） |
| shutdown drain | engine 既存 drain に乗る（到達はする） | **要配線**。member router に対し `~EQProcessor` で `drainAllQuarantineStore` 相当＋定期的 `tryReclaim` の driver が必要（§6.3） | 同左（要配線） |
| RT safety | RT 経路の変更なし。ただし UAF（上記） | **RT 経路の変更なし**。`process()` の guard / load 経路に触れない | 同左 |
| object lifetime | EQState/BandNode が engine epoch で誤解放され得る | **private epoch で正しく保護**（現状の D と同一条件） | 同左（設計次第） |
| authority singularization | engine に集約されるが、**EQ オブジェクトの authority としては誤った集約** | 同一 chain（D→Q→E→T）を同一オブジェクト群に適用。**chain の意味は不変** | 新 authority の追加 = **singularization の後退** |
| existing architecture との整合 | `ISRRetireRouter` の設計（provider 束縛）を**誤用**する | `ISRRetireRouter` の設計どおり（provider 束縛＋drain 契約）。`enqueueWithRetry` の改変 0 | 新規 escalation 段 = 新規契約 |
| 追加要素 | 注入配線（＋ provenance 破壊） | member 1 個＋ctor init＋drain 配線 | store 群＋escalation＋drain 配線 |

### 5.3 判定

| 候補 | 判定 | 理由 |
| --- | --- | --- |
| **A** | **REJECT** | §3.4 の 2 独立証明（epoch 値 mismatch＋reader 可視性 mismatch→UAF）。STG-10-D1 で閉じた UAF と同型の UAF を再導入する |
| **B** | **ADOPT（推奨）** | provenance 維持＋chain 不変＋最小新 logic。`ISRRetireRouter` の改変 0（Owner 必須項目 5 に適合） |
| **C** | **FALLBACK** | B の member 化に構造的障害が見つかった場合のみ。escalation の再実装が必要で B より新規契約が多い |

---

## 6. D1 の修正範囲の最小化（Owner 必須項目 5）

### 6.1 最小修正 scope（候補 B）

```text
files:
  src/eqprocessor/EQProcessor.h          (member 1 個追加＋ctor init 宣言)
  src/eqprocessor/EQProcessor.Core.cpp   (ctor init＋drain 配線)

functions:
  EQProcessor::EQProcessor()             (member router 初期化)
  EQProcessor::~EQProcessor()            (破棄前 drainAllQuarantineStore 相当)
  EQProcessor::enqueueDeferredDeleteWithFallback()  (stackRouter → member router)
  ＋ 定期的 drain の driver 1 箇所（§6.3）

NOT touched:
  src/audioengine/ISRRetireRouter.*      (設計変更なし。chain 維持)
  src/core/EpochDomain.h                 (STG-10-D1 の RC-1 を維持)
  RT path（process / guard / load）       (変更なし)
  Coordinator / RuntimeWorld / Retire authority（変更なし）
```

### 6.2 実現可能性の事前確認（read-only）

| 条件 | 判定 |
| --- | --- |
| `ISRRetireRouter` の member 化 | **可能**。copy/move 削除済み（`h:175-178`）だが、`EQProcessor` は既に `RCUReader` member（copy/move 削除、`RCUReader.h:31,33`）により non-movable。追加 member で movability は変わらない |
| member 初期化順 | `m_epochDomain`（`h:488`）より後に router member を宣言すれば provider 参照は有効 |
| ctor 引数 | `ISRRetireRouter(IEpochProvider&, observer=nullptr)`（`h:172-173`）。observer は既定で可 |
| `~EQProcessor` の既存 drain との順序 | 現状 `:158-160` で private D を drain。member router の Q/E/T drain を**その前**に置く（Q/E/T → D の順ではなく、Q/E/T を epoch-gated drain した後に D を drain する既存順序と整合） |

### 6.3 残課題（implementation stage で確定すること）

1. **定期的 drain の driver**: member router の Q/E/T は epoch-gated のため、破棄前 drain だけでは
   epoch-unsafe entry が残る。`ISRRetireRouter::tryReclaim()`（public、`cpp:386-394`:
   provider tryReclaim＋`drainQuarantineStore`＋`drainEmergencyAndTerminal`）を呼ぶ
   NonRT driver が必要。候補: param path（`Core.cpp:748` 付近・Message thread）、
   `releaseResources`、engine Timer からの到達。**driver の選択は implementation stage の決定事項**。
   なお `signalDrainWakeup()`（`:378`）が起こすのは engine の CoordinatorLoop であり、
   member router を見ない点に注意。
2. **旧 DSPCore 破棄タイミング**: `DSPCore`（したがって `EQProcessor`）は publication で retire され、
   fade 完了後に破棄される。破棄前 drain は「audio が当該 DSPCore を触らない」前提に立つが、
   これは現状の `~EQProcessor:158-160` の `drainAll()`（epoch-agnostic 強制）と**同一の前提**であり、
   B は前提を悪化させない。
3. **`m_retireDropCount` の意味**: B 適用後も `false`（Shutdown 等）は返りうる。telemetry の意味は不変。

---

## 7. 既存 invariant との衝突確認（Owner 必須項目 6）

| 不変条件 | 候補 B の判定 |
| --- | --- |
| RT は ownership を解放しない | **維持**。RT 経路（`process` / guard / load）に触れない。retire 呼び出しは全て NonRT（setter / Message / dtor） |
| Retire は Epoch を通る | **維持・回復**。retireEpoch（private）⇄ reclaim 境界（private min）が同一 domain。現状の D と同一条件を Q/E/T に拡張する |
| TerminalReclaimAuthority が terminal ownership を受領する | **維持**。chain（D→Q→E→T）をそのまま使う。member 化しても T の growable 性質は不変 |
| shutdown は完全 Drain | **要配線**（§6.3）。破棄前 drain＋定期的 drain で達成。現状（到達 drain なし）より改善 |
| authority は単一化する | **維持**。EQ オブジェクトの authority は private-domain chain に一本化。engine chain とは対象オブジェクトが disjoint（§3.3 対照表） |
| private EBR domain と shared retire authority の混在による epoch mismatch | **発生しない**。B は shared authority を使わない |

**衝突なし。**

---

## 8. D2 / D3 / D4 の扱い（Owner 必須項目 7）

- **D2**（`SnapshotCoordinator` Q-full assert のみ）: 既知 finding として保持。**本 Audit の結論に混ぜない。**
  なお D2 の sink は engine singleton Q であり、D1 の stack-local 問題とは**到達先が異なる**。
  混同しないこと。
- **D3**（`applyMmcssPriority` 失敗分岐の RT mutex）: 既知 finding として保持。**混ぜない。**
- **D4**（`closeReaderRegistration` と slot 再取得）: P0 化の判定を**本 Audit の結論に混ぜない**。
  §11 で dependency の有無のみ報告する。

---

## 9. Test contract（実装段階への申し送り）

**test は追加も変更もしていない**（read-only）。実装時に固定すべき oracle の方向のみ示す。
expected result を先に固定し production に合わせることはしない。

| ID | 方向 | oracle の形 |
| --- | --- | --- |
| TD1-1 | D 満杯 → Q 昇格した entry が**関数 return 後も生存**し、epoch-safe 後に deleter が呼ばれる | deleter 起動＋`pendingRetireCount` 遷移。決定論的（private domain 直叩き相当の単一スレッド手順が望ましい） |
| TD1-2 | E / T まで到達した entry も同様に生存＋回収される | 同上（Q 512 満杯 scenarios は capacity を埋める手順が必要） |
| TD1-3 | `~EQProcessor` 後に leak なし（全 entry の deleter が呼ばれたか drain された） | leak カウンタ 0 |
| TD1-4 | **epoch provenance の非回帰**: EQ retire entry の epoch が private domain の advance でのみ reclaim 可能になること（engine epoch を進めても解放されない） | engine `publishEpoch` 後に deleter 未起動 |
| TD1-5 | 既存 `D8_2_B_2_Tests` / `RetireGraceSemanticsTests` の非回帰（改変 0） | 既存 suite PASS |

---

## 10. Prohibited changes への準拠（本 Audit 自体の遵守）

| 禁止 | 遵守 |
| --- | --- |
| `getMinReaderEpoch()` の変更 | **なし**（STG-10-D1 の RC-1 を維持） |
| quarantine flag の変更 | **なし** |
| RT path の変更 | **なし**（候補 B も RT に触れない） |
| Coordinator の変更 | **なし**（`RuntimeIntentCoordinator::enqueueRetire` は委譲のまま） |
| production / test / CMake / ConvoPeq.md の変更 | **すべて 0**（read-only 完遂） |
| commit / push / worktree cleanup | **すべて 0**。pre-existing 差分（`AGENTS.md` / `headroom-proxy-start.ps1` / STG-8 gate）に未接触 |

---

## 11. D4 との dependency 判定（Owner 必須項目 7 ただし書き）

**D1 の検証中に D4 との直接的な dependency は発見されなかった。**

| 観点 | 結果 |
| --- | --- |
| D1 の domain | EQ **private** `m_epochDomain` |
| `closeReaderRegistration()` の対象 | engine `m_epochDomain` のみ（`ReleaseResources.cpp:274`、`CtorDtor.cpp:236`） |
| EQ private domain の close | **存在しない**（`closeReaderRegistration` を EQ が呼ぶ箇所 0 件） |
| したがって D4 の failure mode（close 後の再取得失敗）は D1 の chain に**到達しない** | **dependency なし** |

---

## 12. STOP conditions の判定

Owner の指定する STOP 条件（STG-10 Audit §13 に準ずる）に該当するものはない。
特に: quarantine の意味論は STG-10 で確定済み・Path B は PROVEN 済み・UAF chain は PROVEN 済み・
本 Audit は単一候補（B）に収束・RT decision 問題なし（RT を変更しない）・
slot ownership 明確・memory ordering 変更なし・shutdown 意味論と非衝突（§7）・
既存 STG との交差なし・production/test 変更なし・authority FRESH。

---

## 13. 最終報告

```text
D1 status:
  CONFIRMED

ownership proof:
  PASS（§2: D→Q→E→T の昇格先が stack-local member であることを再証明。
        Q/E inline・T heap のいずれも関数 return で失われる。
        ユーザー定義 dtor なし。呼び出し元は true を誤認）

epoch provenance:
  PASS（§3: 現状の private-domain provenance は正しい。
        Candidate A は epoch 値 mismatch＋reader 可視性 mismatch の
        2 独立証明により UAF を再導入するため REJECT。
        Convolver との対照（engine epoch＋engine reader）で裏付け）

lifetime proof:
  PASS（§2.3: ISRRetireRouter／各 store にユーザー定義 dtor なし。
        stack 破棄で deleter 未実行のまま消滅）

shutdown reachability:
  PASS（§2.4: engine 既存 drain は engine router のみ。
        ~EQProcessor は private D のみ。stack Q/E/T に届く drain は存在しない。
        ※「PASS」は「到達不能であることの証明が成立」の意味）

RT contract:
  PASS（全 retire 呼び出しは NonRT。RT process は guard/load のみ。
        候補 B は RT 経路を変更しない）

repair candidate:
  B（EQProcessor 所有の member router。private domain 束縛を維持）
  A は REJECT（§3.4／§5.2）。C は FALLBACK（§5.2）。

minimum repair scope:
  src/eqprocessor/EQProcessor.h（member 1 個＋ctor init 宣言）
  src/eqprocessor/EQProcessor.Core.cpp（ctor init＋破棄前 drain＋定期的 drain の driver）
  ISRRetireRouter／EpochDomain／RT path／Coordinator／authority は不変

new invariant required:
  1. EQ retired objects are reclaimed only through the private-domain-gated
     authority（現状の D と同一条件を Q/E/T に拡張。実質的には既存 invariant の確認）
  2. member router は破棄前に drain されること（destruction-drain。§6.2）

implementation readiness:
  GO（候補 B。§6.3 の driver 選択を implementation stage で確定すること）
```

**`Repair Contract = PROVEN`（候補 B）に収束した。ただし implementation には進まない。**
Owner の implementation GO を待つ。
