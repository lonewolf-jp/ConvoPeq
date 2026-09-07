# D172-2 — MEM_SNAP Repair Contract / Authority Impact Audit（evidence）

- 日付: 2026-09-07
- Type: **read-only audit** — production source 0 / test source 0 / CMake 0 / build 0 / CTest 0
- Authority stamp: `ConvoPeq.md` — `Generated: 2026-09-07 21:38:42`
- 前提: D172-1 STOP（CONFIRMED LIFETIME HAZARD・diagnostic build scope）
- 目的: 「UAF を直すこと」ではなく「**どの既存 authority を使えば Observer の lifetime guarantee を追加せずに安全化できるかを証明し、実装境界を固定すること**」
- 原則: Practical Stable ISR Bridge Runtime — RuntimeWorld は Publish 後 immutable / Observer は観測専用・所有権を持たない / Retire/Delete は専用経路に分離

---

## P1 — 修復案A の source preflight — **PASS**

### P1-1. 変更対象の現状

```cpp
// Timer.cpp:1079-1088（MEM_SNAP block・#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS 内）
auto* activeDSP = getActiveRuntimeDSP();                 // legacy slot（D172-1 で CONFIRMED HAZARD）
if (activeDSP != nullptr)
    auto stats = activeDSP->collectTrackedMemoryStatistics();
```

### P1-2. 案A が接続する既存 authority の完全列挙

| 要素 | アンカー | 実測 |
|---|---|---|
| `makeRuntimeReadHandle()` | AudioEngine.h:3227-3302 | Message/Publication channel は `debugAssertNotAudioThread()` → world observe → telemetry → coordinator observe → handle 構築 |
| `getRuntimeWorldFromReadHandle()` | AudioEngine.h:3324-3327 | `runtimeReadHandle.runtimeWorldPtr()` を返すのみ（追加解決なし） |
| **`resolveActiveRuntimeDSPFromRuntimeWorldOnly()`** | AudioEngine.h:3398-3404 | `runtimeWorld->engine.current` を DSPCore\* に返す — **RT path（Latency.cpp:85-89 comment「RT 処理パス…と同じ解決」）と同一の authoritative resolver** |
| `RuntimeReadHandle` lifetime | AudioEngine.h:2282-2313 | move-only RAII・`ObservedRuntime`（= `RCUReaderGuard` 保持、ObservedRuntime.h:26-31）を move で保持。デストラクトで `exitReader` |
| current DSP の authoritative path | Timer.cpp:503/685/700 で `resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle)` 既に使用中 | timerCallback は本 resolver を通常 path で常用（案A は**新経路を作らない**） |
| `collectTrackedMemoryStatistics()` 呼び出し条件 | DSPCoreLifecycle.cpp:337-338 | `ASSERT_NON_RT_THREAD()` — timerCallback（MessageThread）は通過（現行と同一） |

### P1-3. 「runtimeReadHandle が MEM_SNAP dereference 完了時点まで有効」の再証明

1. **取得位置**: `AudioEngine.Timer.cpp:428` — `timerCallback()` 関数本体トップレベルで宣言。ブロックスコープをまたぐ条件付き宣言ではない。
2. **move 有無**: rg 実測 — `runtimeReadHandle` の全使用は `:429/471/472/503/504/685/700/702/909` の**const 参照渡しのみ**。`std::move(runtimeReadHandle)` は 0 件。move 消滅なし。
3. **MEM_SNAP block の位置**: `:1022-1115`（`#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`）。`:428` 宣言の生存スコープ内。
4. **既存の world dereference**: MEM_SNAP block は `:1027` で既に `runtimeWorld->generation` を読んでいる — world 経由の access は現行でも実施済み（案A は dereference 先を slot → world.current へ寄せるのみ）。
5. **RAII 効果**: `ObservedRuntime` ctor 内 `RCUReader::enter()` → `EpochDomain::enterReader`（EpochDomain.h:114-131）が **enter 時点の currentEpoch を pin**。`exitReader`（:141-160）は handle デストラクト時（callback 終了時）にのみ発火。

> **結論: handle は MEM_SNAP dereference 完了時点まで生存し、その期間 MessageThread は EBR reader として active（epoch pin 中）。**

---

## P2 — RuntimeWorld → DSP lifetime chain の再証明 — **PASS（2 層構成・Layer 2 は既知境界として明記）**

### P2-1. retire の正式経路（OBSERVE-1）

```text
World W(gen G) current = D
  → W の replacement publish（新しい world swap・新 publicationEpoch）
  → fade 完了時: Timer が runtimePublicationBridge_.submitObserve(fadingHandle, currentPublicationEpoch())
      （Timer.cpp:972/1125/1706 — Coordinator.h:224 OBSERVE-1 契約「Timer → submitObserve → Coordinator が retirePublishedDSP を起動」）
  → CoordinatorLoop worker が processIntent → AudioEngine::retirePublishedDSP(D, lifetimeMgr)
      （Timer.cpp:1924 — publication epoch を伝搬する retire パス）
  → DSPLifetimeManager::retire(D, E_r)
      （DSPLifetimeManager.cpp:36-69 — retireDSPHandleForRuntime 台帳解除 + router_->enqueueWithRetry(D, &destroyDSPCoreNode, E_r, Generic)）
  → CoordinatorLoop drain（Threading.cpp:383 runCoordinatorPhase 末端）
  → isOlder(E_r, minReaderEpoch) 成立時のみ destroyDSPCoreNode（Threading.cpp:38-40 ~DSPCore + aligned_free）
```

**契約の骨子（AudioEngine.h:4332-4348）**: retireDSPHandleForRuntime は台帳解除 primitive。物理破壊は DSPLifetimeManager::retire → EBR（enqueueWithRetry による破壊権取得 → epoch 安全確認後に destroyDSPCoreNode）。**破壊は必ず epoch gate を通る。**

### P2-2. Epoch 基本契約の接続（`retireEpoch < minReaderEpoch`）

| 要素 | アンカー | 意味 |
|---|---|---|
| `enterReader(tid)` | EpochDomain.h:114-131 | reader slot に **enter 時の currentEpoch** を pin（`slot.epoch = currentEpoch()` release）+ depth++ |
| `exitReader(tid)` | EpochDomain.h:141-160 | 最終 exit で `slot.epoch = kInactiveEpoch`（pin 解除） |
| `getMinReaderEpoch()` | EpochDomain.h:201-210 | **active（depth>0）reader の pin epoch 最小値**（active 0 なら currentEpoch） |
| destroy 条件 | ISRRetireRouter.h:44/77-79 | `isOlder(entry.epoch, minReaderEpoch) == true` で deleter 実行 |

→ **Reader が entry の epoch より前の epoch を pin し続ける限り destroy は遅延する。**

### P2-3. lifetime proof（Layer 1 — 主 hazard の解消）

Reader（改修後の MEM_SNAP）が観測した world W の current DSP = D について:

```text
read section 開始:   enterReader → E_pin = currentEpoch() を pin（depth>0）
current DSP capture: runtimeWorld->engine.current = D（world は Publish 後 immutable）
DSP retire:          W の replacement 発行時に D が retire 対象になる
                     → retire enqueue epoch E_r = replacement 時の publicationEpoch
                     → replacement は reader enter 後に起きた場合 E_r > E_pin
EBR grace:           reader が active（callback 実行中）の間 minReaderEpoch ≤ E_pin < E_r
                     → isOlder(E_r, minReaderEpoch) 不成立 → destroy 遅延
destroy:             reader の exit（callback 終了）後のみ実行可能
```

- **Case 1（replacement が reader enter 後）**: E_r > E_pin → read section 中 D は破壊されない。✔
- **Case 2（replacement が reader enter 前）**: reader が observe する world は swap 後の新 world → current = 新 DSP → 旧 D は参照しない。✔
- **world の immutable 性**: RuntimeWorld は Publish 後 immutable（RuntimeWorldAuthority publish 契約・seal-before-bake ordering）→ 観測済み world の engine.current は書き換わらない。✔

> **「取得した DSP が read section 終了まで破壊されない」— Layer 1 として成立。**
> 対比: 現行 slot 経路は「slot 値が前世代 generation で書かれた stale pointer」であり、epoch 規律の下で一度も取得されていないため EBR が無力だった（D172-1）。案A では pointer が **read section 内の観測** として取得され、EBR の保護対象になる。

### P2-4. Layer 2 — 既知境界（Case 3 window）の明記

`makeRuntimeReadHandle` 内の順序は **world observe（`runtimeStore_.observe()` 単一 acquire load・pin 前）→ enter**。このμs window 内に replacement publish + retire enqueue + drain 完了 + 全 reader 通過が同時進行した場合、reader は「pin 済みだが古い world の current DSP（destroy 済み）」を観測し得る（理論上・確率的）。

- **本 repair が新規に生むリスクではない**: この window は RT path（`resolveActiveRuntimeDSPFromRuntimeWorldOnly` を AudioProcessor の RT 契約で使用）・Latency・Publication channel を含む**全 world reader が共有する既存性質**であり、既存 invariant 運用（D158〜D172 の 60+ gen soak・CTest 40/40）で未観測。
- MEM_SNAP 固有の差分はゼロ（同一 resolver・同一 handle・同一スレッド）。
- **対処**: (1) 本契約文書に既知境界として記録（完了）。(2) D172-3 の実装では dereference を handle 取得済み区間内で完結させ、追加の window を作らない。(3) enter-first 順序逆転は全 world reader に影響するため**本 repair の scope 外**として別 track 記録（着手理由は現時点でなし — 未観測・理論窓・scope 拡大）。

### P2-5. P2 判定

> **PASS — Layer 1（主 hazard = 確定的 UAF の解消）は EBR 既存 authority による lifetime proof として成立。Layer 2（Case 3 window）は既存全 reader 共有の既知境界として明記し、MEM_SNAP 固有の新規リスクはゼロ。**

---

## P3 — TRK telemetry semantic audit — **PASS（intentional change として契約明記）**

### P3-1. producer / consumer 完全列挙

| 区分 | 実測 |
|---|---|
| producer | `AudioEngine.Timer.cpp:1092-1105`（`[MEM_SNAP] PUBLISH … TRK: total=… OS=… EQ=… AL=… LT=…`）**1 箇所のみ** |
| test assertion | **0 件**（`rg "TRK" src/tests/` → 0 hit） |
| regression threshold / diagnostic parser | **0 件**（tools/・scripts/ に parser なし。tools/work70_add_memsnap.py は MEM_SNAP 挿入用の生成スクリプトで parser ではない） |
| external tooling | なし |
| evidence | D162-1P/D162-1R-B で観測**補助**として言及（主指標は DSPCore::liveCount / DSP_FOOTPRINT / LiveAllocRegistry）。TRK 数値そのものへの判定依存なし |
| crash analysis | D169-2-5 の crash は logger race（契約違反）で TRK 値依存なし |

### P3-2. 意味論の差分

| | 現行 | 案A 適用後 |
|---|---|---|
| TRK の対象 | `activeRuntimeDSPSlot` が指す placeholder DSP（**破壊後は dangling から garbage 値** — D172-1 実証） | RuntimeWorld current DSP（生存が EBR 保証された実 active DSP） |
| world 未公開期 | slot に placeholder があればその stats | TRK = 0（world current なし） |

### P3-3. 判定

> **semantic change = intentional として許容。** 根拠: (1) consumer が存在しないため互換性破壊の当事者がいない。(2) 現行 TRK は placeholder 破壊後 garbage 値を出力しており、現行値自体が「意味のある baseline」として機能していない（むしろ案A は TRK を初めて「信頼できる統計」にする）。(3) 変更は diagnostic ログの意味論であり production 動作に影響しない。**契約条件**: D172-3 実装時に MEM_SNAP block へ「TRK source = RuntimeWorld current DSP（旧: legacy placeholder slot）」の意味論変更を明記する。

---

## P4 — R2（Latency fallback）— **PASS（lifetime-safe 構造の再確認）**

### P4-1. fallback 構造

```cpp
// Latency.cpp:80-97
publishedWorld = worldAuthority_.consumeWorldHandle(readToken);
dsp = (publishedWorld != nullptr) ? publishedWorld->engine.current : nullptr;
if (dsp == nullptr) dsp = getActiveRuntimeDSP();   // placeholder フォールバック
if (dsp == nullptr) return breakdown;               // ← dereference は dsp 非null 時のみ
```

fallback dereference が問題になるのは **「world null（または current null）∧ slot 値が dangling」の同時成立** 時のみ。

### P4-2. 順序証明

| 状態遷移 | slot | world | 同時成立の可否 |
|---|---|---|---|
| 初期化直後 | null（h:2228 既定） | 未 publish（null） | dangling なし ✔ |
| prepare 成功後 | =P（生存・world current） | gen1 current=P | world non-null → fallback 非到達 ✔ |
| rebuild 置換後（P destroy） | **dangling P** | gen2 current=新DSP（non-null） | **world non-null → fallback 非到達** ✔ |
| releaseResources | :180 で null（rebuildMutex 下） | :531-537 clearPublishedRuntimeSnapshotsNonRt → null（同一関数・slot clear が先行） | world null 観測時点で slot null は program order で先行・release/acquire で可視 ✔ |
| dtor | CtorDtor:153 で null（:141-168 validate 前後） | 同時 clear | 同上 ✔ |
| tryAdmit 失敗 rollback | :317 destroy → :318 null（同一 MessageThread 連続コード） | publish 失敗で旧状態（null or 旧 gen） | window は同一 MessageThread 内 → MessageThread reader（fallback の全 caller が MessageThread 系 — 下記）は割り込み不可 ✔ |

- **fallback reader のスレッド実測**: `getCurrentLatencyBreakdown` の caller = `MainWindow.cpp:1544`（UI・MessageThread）と `AudioEngineProcessor::prepareToPlay:38`（**`audioEngine.prepareToPlay` と同一シーケンス** — slot writer と同一スレッド）。RT thread は呼ばない（RT は world 解決のみ — Latency.cpp:85-89 comment 実測）。
- destroy が cross-thread（CoordinatorLoop worker）で起きるのは **published DSP（world non-null 期間）のみ** — その期間 fallback は非到達。

> **結論: 「world null ∧ slot dangling」は atomic ordering / critical-section ordering 上で成立しない。R2 は現行構造のまま lifetime-safe。変更不要。**

---

## P5 — R3（logRuntimeTransitionEvent dormant reader）— **PASS（dormant 維持 + 復活時契約を D172-3 に含める）**

- production caller 0 件を再確認（`rg "logRuntimeTransitionEvent\(" src/audioengine/*.cpp` → 0 hit・AudioEngine.h:3844 定義のみ）。
- dereference 内容: `getActiveRuntimeDSP()` → `current->runtimeUuid`（h:3850-3851）— slot 経路の dereference 潜在は残る。
- **判定: dormant API として残す**（削除は scope 最小化の観点で本 repair に含めない）。ただし **D172-3 implementation contract に以下を含める**: 復活・再利用時は `resolveActiveRuntimeDSPFromRuntimeWorldOnly` 経由に統一することを契約コメントで明記（slot dereference の新規追加を禁止する boundary）。

---

## P6 — 案B の formal rejection — **PASS（invariant 5 項目で固定）**

案B「destroy/retire 側で `activeRuntimeDSPSlot` を CAS clear」を、以下の invariant 単位で却下する:

| # | 判定軸 | 却下理由（source 実測に基づく） |
|---|---|---|
| 1 | Observer mirror が lifetime authority 化しないか | **化する**。RC-D169-1-2（ReleaseResources.cpp:151-161）は「capture は topology observation 用にのみ使用する（**ownership authority に昇格させない**）」と明文。destroy 側 clear は slot を lifetime coordination participant に昇格させる契約違反 |
| 2 | destroy path が pointer identity を扱うことにならないか | **扱うことになる**。destroy 側が「対象 DSP が slot に入っているか」を判定するには raw pointer 値の突合が必須。AudioEngine.h:4332-4348 契約は terminal disposition を DSPLifetimeManager::retire 一本路に限定しており、slot 突合はこの契約外の操作 |
| 3 | address reuse race が発生し得ないか | **発生し得る**。destroyDSPCoreNode は `convo::aligned_free(core)`（Threading.cpp:40）で address を解放 — 同一 address が直後の新 DSPCore 生成に再割当され、旧 destroy 側の突合が「生存中の新 DSP」を誤対象化する。D169-1 で実測した障害クラス（probe8/9: address reuse による生存 DSP 誤 lookup → 二重破壊 0xC0000005）と同型 |
| 4 | Retire Authority の単一性を壊さないか | **壊す**。D170 修復の核心は「destroy authority の handle 一本路収束」（D169-1R RC-D169-1-1: pointer-value retirement 廃止）。slot clear は第 2 の terminal disposition 経路の萌芽になる |
| 5 | D170/D169-1 で排除した設計を再導入しないか | **再導入する**。旧 :352-359 の `lifetimeForShutdown.retire(activeToRelease 等)` は「raw-pointer map key lookup（address reuse で生存 DSP を誤 lookup）」として廃止済み（ReleaseResources.cpp:155-159 comment 実測）。slot 突合 clear は同一パターンの変種 |

> **案B = REJECT（5/5 invariant 違反）。**

---

## P7 — 案C の必要性 — **PASS（不採用判定）**

- 案A で lifetime proof が成立する（P2 Layer 1）以上、**新規観測 authority / registry を導入する理由は存在しない**。
- 案C が要求する「生存 DSP の列挙機構」は handle map（mutex）か新 registry の追加を意味し、Practical 原則（既存 authority を使用・新 mechanism 追加は理由があるときのみ）に反する。
- さらに現行 TRK の計算（`collectTrackedMemoryStatistics`）は config scalar からの推定値であり、registry 化の精度上の利得も限定的。
- **判定: 案C 不採用**（A の proof が崩れた場合の再設計候補として記録保持）。

---

## P8 — Repair contract（GO/NO-GO）

| 項目 | 判定 |
|---|---|
| P1 RuntimeWorld resolution | **PASS** |
| P2 DSP lifetime proof | **PASS**（Layer 1 完全 proof・Layer 2 既知境界明記） |
| P3 TRK semantic compatibility | **PASS**（intentional・consumer なし・契約明記条件付き） |
| P4 R2 latency fallback | **PASS**（現行構造のまま lifetime-safe・変更不要） |
| P5 R3 dormant reader | **PASS**（dormant 維持・復活時契約を D172-3 に含める） |
| P6 Option B rejection | **PASS**（invariant 5/5 で REJECT 固定） |
| P7 Option C necessity | **PASS**（不採用） |

> ## **判定: GO — 案A 採用・implementation contract 以下のとおり固定 → D172-3 implementation**

### Implementation contract（実装境界 — コードは D172-3 で書く）

1. **変更箇所は 1 箇所のみ**: `AudioEngine.Timer.cpp` MEM_SNAP block（:1079 付近）の `getActiveRuntimeDSP()` → `resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle)`。既存 accessor・既存 handle・既存 authority のみ使用。**新規 atomic / helper / queue / authority の追加禁止**。
2. **禁止事項**: `activeRuntimeDSPSlot` の writer 変更（W1-W4 は現状維持）/ destroy path 変更 / slot clear 追加（案B 相当）/ test source 変更（MEM_SNAP の挙動を直接 assert する新規テストは scope 外 — 既存 CTest 回帰のみ）/ CMake 変更。
3. **TRK 意味論の明記**: MEM_SNAP block コメントに「TRK source = RuntimeWorld current DSP（旧: legacy placeholder slot）・world 未公開期は 0」を記載（P3-3 契約）。
4. **R3 契約コメント**: `logRuntimeTransitionEvent` に「復活時は world resolution 経由に統一（slot dereference 新規追加禁止）」を記載。
5. **実装前 baseline evidence**: D172-1 P4 方式 α（log 相関 `[D117_DESTROY] dsp=P` × `[MEM_SNAP] TRK≠0`）を diagnostic build で 1 回取得し、修復前の dangling dereference 状態を記録（P7 のとおり dynamic reproduction は baseline evidence としてこの位置で実施）。
6. **検証**: 既存 CTest 40/40（3 config）+ diagnostic build での MEM_SNAP 出力確認（TRK が world current DSP の実値を示すこと・dangling 状態の解消）。
7. **scope 外として記録**: makeRuntimeReadHandle の enter-first 順序逆転（Case 3 window の完全解消 — 全 world reader 影響のため別 track・着手理由なし）。

### D172-2 で固定しないもの（指示どおり保留）

- 具体的コード変更 / helper 追加 / 新 atomic 追加 / slot writer 変更 / destroy path 変更 / test source 変更 / CMake 変更 — **すべて D172-3 以降**。
- `activeRuntimeDSPSlot` の destroy-side clear — **実装禁止継続**（P6 で invariant 単位に REJECT 済み）。
- inventory 編集 / buildErrorCount_ trigger 登録 — doc-only maintenance task として別処理（D171-1 判定どおり）。

---

## 実測コマンド系譜（主要分）

```bash
sed -n '190,260p' src/audioengine/RuntimeWorldAuthority.h   # consumeWorldHandle = runtimeStore_.observe() 単一 acquire load
sed -n '3324,3345p;3398,3404p' src/audioengine/AudioEngine.h # getRuntimeWorldFromReadHandle / resolveActiveRuntimeDSPFromRuntimeWorldOnly
sed -n '2282,2313p;3227,3302p' src/audioengine/AudioEngine.h # RuntimeReadHandle RAII / makeRuntimeReadHandle 順序
sed -n '26,45p'  src/core/ObservedRuntime.h                  # ctor で RCUReaderGuard（enter）
sed -n '114,160p;201,210p' src/core/EpochDomain.h            # enterReader pin / exitReader kInactiveEpoch / getMinReaderEpoch
sed -n '60,100p' src/core/SnapshotCoordinator.h              # coordinator observe は enter-first（snapshot は pin 下）
sed -n '218,232p' src/audioengine/ISRRuntimePublicationCoordinator.h  # OBSERVE-1 契約
sed -n '960,975p' src/audioengine/AudioEngine.Timer.cpp      # submitObserve(fadingHandle, currentPublicationEpoch())
sed -n '1924,1984p' src/audioengine/AudioEngine.Timer.cpp    # retirePublishedDSP（publication epoch 伝搬）
sed -n '36,69p'  src/audioengine/DSPLifetimeManager.cpp      # retire → enqueueWithRetry(D, &destroyDSPCoreNode, epoch)
rg -n "runtimeReadHandle" src/audioengine/AudioEngine.Timer.cpp  # move 0 件・const 参照渡しのみ
rg -rn "TRK" src/tests/ tools/ scripts/                       # consumer 0 件
rg -n "logRuntimeTransitionEvent\(" src/audioengine/*.cpp     # production caller 0 件（R3 dormant 再確認）
sed -n '80,97p' src/audioengine/AudioEngine.Processing.Latency.cpp + caller 実測  # R2 thread model
sed -n '4332,4348p' src/audioengine/AudioEngine.h + ReleaseResources.cpp:151-161  # P5 案B 却下アンカー
# 交差検証: serena（reader set）・semble（位置裏付け）・cppcheck（Timer.cpp 指摘 0）
```

## 限界

- 動的実証（方式 α baseline）は本 audit scope 外（D172-3 実装前に実施）。
- Case 3 window（world observe → enter のμs 窓）は理論的存在として記録した。実測観測はなく、解消には既存全 reader への影響を伴うため本 repair scope 外。
