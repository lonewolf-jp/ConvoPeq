# STG-11-D18 Repair Implementation

- Document: `doc/work113/P1-5-IR-P2_STG-11-D18_REPAIR-IMPLEMENTATION_20261001.md`
- Work item: **STG-11-D18-1** — Published Runtime ↔ Crossfade Semantic Consistency
- Contract: `doc/work113/P1-5-IR-P2_STG-11-D18_REPAIR-CONTRACT-AUDIT_20261001.md`
- Date: 2026-10-01
- Authority: リポジトリルート `ConvoPeq.md`（`ConvoPeq(20261001-054723).md` と同一ファイル）
- Commit / push: **未実施**（Owner の明示承認待ち）

---

## 1. Verdict

```
GO（実装完了）— concrete defect 1 件（STG-11-D18-1）を特定し最小修正を適用
```

| 層 | 判定 |
| --- | --- |
| World identity（A） | concrete defect なし |
| Plan identity（B） | concrete defect なし（Plan は CrossfadeRecord{id, from, to, startEpoch, active}） |
| Publish/Crossfade ordering（C） | **defect 1 件**（§2） |
| Plan / World semantic mismatch（§3） | concrete defect なし（duration は `CrossfadeRuntime::start()` が plan と runtime の単一供給点） |
| Crossfade cancellation / replacement（§4） | **defect 1 件**（§2。overlap publish） |
| Lifetime / Retire boundary（§5） | **defect 1 件**（§2。overlap 時に先行 fade の source が orphan） |

---

## 2. STG-11-D18-1（唯一の concrete defect）

### 2.1 Defect statement

`DSPTransition::onPublishCompleted` は fading-slot の CAS claim に**失敗した場合にも**
crossfade record の登録と `beginCrossfade` を実行していた。claim 失敗 =
先行 fade が実行中（`fadingRuntimeDSPSlot` が別 DSP を占有）で、
その場合 `oldDSP` は**即時 retire**されるにもかかわらず、実行されない fade を
主張する phantom record `(old→new)` が登録される。

この phantom record と無条件書き込みの `endCrossfade` が組み合わさると、
fade 完了時に以下が同時発生する。

1. `registry_[toHandle] = Active` — **retire 済み DSP の Active 復活**
2. `activeRuntimeDSPHandle_ = toHandle` — **stale 値の上書き**（実際の active は更に新しい publish の DSP）
3. record table への phantom 残留（完了シグナルの取りこぼし、`jassert(records.size()==1)` 発火）

次 publish は stale な `activeRuntimeDSPHandle_` を `oldHandle` として使い、
その handle は retire 済みなので claim 成功時に**reclaim 済み object の
fading 復活**（audio thread からの UAF）に至り得る。

さらに `activate()` が `fadingRuntimeDSPHandle_` を無条件 null 化していたため、
overlap publish が先行 fade の source DSP（R1）の retire 経路を消滅させていた。

- `Timer.cpp:1142` / `:1305` の `submitObserve` が null を見て skip → orphan
- `Timer.cpp:2129` `retirePublishedDSP` の `currentHandle` が null となり
  receipt と不整合 → Emergency Retire（誤 `submitQuarantine` / `fatal_` 化）

### 2.2 Reproduction（deterministic / timing 非依存）

```text
1. R1 active、R2 publish → claim(R1) 成功、fade R1→R2 開始
   （record id1、slot=R1、fadingRuntimeDSPHandle_=R1）
2. fade 完了前に R3 publish（fire-and-forget 経路は fading gate を持たない）
   step1: activate(R3)                      → fading handle が null 化（旧実装）
   step2: registerCrossfade(R2,R3)=id2      → phantom record 登録
          beginCrossfade(R2,R3,id2)         → registry[R2]=CrossfadingIn, [R3]=CrossfadingOut
          claim(R2) 失敗                    → lifetime.retire(R2)（即時 retire）
3. fade 完了（Timer.cpp:1100 tryCompleteFade）
   records=[id1,id2] → front=id1 → endCrossfade(id1)
       registry[R1]=Retired
       registry[R2]=Active      ★ retire 済み DSP の復活
       activeRuntimeDSPHandle_=R2  ★ stale（実際は R3）
       fadingRuntimeDSPHandle_=null
   id2 は active=true のまま残留
4. 次 publish は oldHandle=R2（retired）を使用
```

step 2 の「fade 完了前に R3 publish」は正常な publish/parameter/UI sequence で到達可能。
fire-and-forget 経路（`PublicationExecutor.cpp:61`）には fading gate がなく、
gate を持つのは `Orchestrator::trySubmitImpl` の `DeferredFadingActive` のみ。

### 2.3 Root cause（3 点の無条件操作）

| # | 場所 | 内容 |
| --- | --- | --- |
| 1 | `DSPTransition.h:113-135`（旧） | register/begin が claim 結果に依存しない |
| 2 | `ISRDSPHandle.cpp:106-112`（旧） | `endCrossfade` の toHandle/activeHandle/fading 書き込みが現状非依存 |
| 3 | `ISRDSPHandle.cpp:96`（旧） | `activate()` の fading-handle null が in-flight fade の retire 経路を消す |

---

## 3. 修正（NonRT のみ。3 点）

### F-A. `DSPHandleRuntime::activate` — fading handle を消さない

`src/audioengine/ISRDSPHandle.cpp`

```diff
     convo::publishAtomic(registry_[handle.slot].state, DSPState::Active, std::memory_order_release);
     convo::publishAtomic(activeRuntimeDSPHandle_, handle, std::memory_order_release);
-    convo::publishAtomic(fadingRuntimeDSPHandle_, DSPHandle::null(), std::memory_order_release);
+    // ★ STG-11-D18 (F-A): in-flight fade の fading handle を消さない。
 }
```

- 通常 path（先行 fade 無し）では handle は既に null のため**無変更と等価**。
- 終端は `endCrossfade`（F-C のガード付き）が担う。
- caller は `DSPTransition.h:71` / `:104` の 2 箇所のみ（重みなしを確認済み）。

### F-B. `onPublishCompleted` — claim を register/begin より先行

`src/audioengine/DSPTransition.h`

```diff
 if (decision.needsCrossfade && oldDSP != nullptr && oldDSP != newDSP) {
+    const bool claimed = engine_.claimFadingRuntimeDSP(oldDSP);
+    if (!claimed) {
+        lifetime.retire(oldDSP);      // register/begin は行わない
+    } else {
+        // 既存の register/begin/storeReceipt
+    }
     // crossfadeRuntime_.start(...) は無条件実行のまま
```

- 通常 path の効果順序は同一（claim → register → begin → receipt → ramp）。
- `ramp.start` は**無条件実行**を保持（先行 fade の可聴タイミングを維持）。
- claim 失敗時に phantom record が生まれないため single-record invariant が構造的に維持される。

### F-C. `endCrossfade` — stale 書き込みのガード

`src/audioengine/ISRDSPHandle.cpp`

| 書き込み | 修正後の条件 | 根拠 |
| --- | --- | --- |
| `fromHandle → Retired` | **無条件のまま** | 到達状態は `CrossfadingOut` か `Retired` のいずれか。両方とも `Retired` が正解 |
| `toHandle → Active` | 現在 `CrossfadingIn` の場合のみ | 重ね合わせで即時 retire 済みの DSP を復活させない |
| `activeHandle = toHandle` | 現在 `== fromHandle` または `== toHandle` の場合のみ | より新しい publish が進んでいれば上書きしない |
| `fading → null` | 現在 `== fromHandle` または既に null の場合のみ | より新しい fade の handle を消さない |

すべて既存 `convo::consumeAtomic` / `publishAtomic` のみ。新規 atomic・mutex・
authority は追加していない。

### ISR 構造への影響

**なし。** 3 点すべて NonRT（CoordinatorLoop / Timer）パス。
RT path の mutex / allocation / delete を追加していない。既存 atomic への
load-then-conditional-store のみ。

---

## 4. 修正ファイル

| ファイル | 種別 | 内容 |
| --- | --- | --- |
| `src/audioengine/ISRDSPHandle.cpp` | production | F-A（activate）、F-C（endCrossfade） |
| `src/audioengine/DSPTransition.h` | production | F-B（claim-first） |
| `src/tests/AudioEngineHarness/STG11D18CrossfadeTests.cpp` | test（新規） | T-CFX-A〜D |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | test（配線） | 宣言＋呼び出し追加 |
| `src/tests/NormalRetireDSPHandleCompareTests.cpp` | test（契約更新） | §6 参照 |
| `CMakeLists.txt` | build | 新規 test TU 登録（+1 行） |

production diff 行数（HEAD `f95f524c` 起点。D18 は両ファイルが未 commit 変更の
唯一の差分）：`DSPTransition.h` +26/-14、`ISRDSPHandle.cpp` +30/-4、
`NormalRetireDSPHandleCompareTests.cpp` +16/-3、`CMakeLists.txt` +1、
`PublishPipelineIntegrationTests.cpp` +3。

---

## 5. Regression test

`src/tests/AudioEngineHarness/STG11D18CrossfadeTests.cpp`（harness sub-test）

| ID | 内容 | 対応 |
| --- | --- | --- |
| T-CFX-A | overlap completion: `begin(h1,h2)` → `retire(h2)` → `activate(h3)` → `endCrossfade(id1)`。h2 は `Retired` のまま、`activeHandle==h3`、h1 `Retired` | F-C |
| T-CFX-B | `activate()` が in-flight の fading handle を保持する | F-A |
| T-CFX-C | 通常完了: `begin(h1,h2)` → `activate(h2)` → `endCrossfade` → h1 `Retired`/h2 `Active`/`active==h2`/fading null（production 順を再現） | 正常 path 等価性 |
| T-CFX-D | 構造検査: `DSPTransition.h` で claim が register/begin より前、`ISRDSPHandle.cpp` の `endCrossfade` にガード群、`activate` に fading 書き込みなし | F-A/F-B/F-C |

fixture: `DSPHandleRuntime` + `CrossfadeAuthorityRuntime` の単体インスタンス
（default 構築可・状態は instance-local）。`create(void*)` は未参照の
distinct ダミー pointer。engine / DSPCore は不要。timing 非依存・完全 deterministic。

### CFX-1〜CFX-7 対応表（Owner 指定）

| 要求 | 対応 |
| --- | --- |
| CFX-1 single publish → crossfade | T-CFX-C（通常完了の等価性）＋既存 `PublicationPipelineIntegrationTests` publish 系 |
| CFX-2 successive publish R1→R2→R3 | T-CFX-A（overlap 後の registry/active 状態） |
| CFX-3 publish during active crossfade | T-CFX-A / T-CFX-B |
| CFX-4 crossfade completion → retire | T-CFX-A（`isRetired(h1)`） |
| CFX-5 crossfade cancellation/replacement | T-CFX-A（claim 失敗＝replacement） |
| CFX-6 different sample-rate / topology | 既存の `PublicationValidatorIsolationTests`（`CrossfadeAuthority::evaluate` の OS factor・structuralHash 分岐）に委譲。D18-1 の被害は topology ではなく handle lifecycle であり、sampleRate/topology に依存しないため新規 test 不要 |
| CFX-7 negative control | §5.1 実施済み |

### 5.1 Negative control（実施結果）

`ISRDSPHandle.cpp` を一時的に原始欠陥へ復元（F-C の無条件書き込み＋F-A の無条件 null 化）
→ Debug rebuild → harness rc=1。

```text
D18-A: h2 resurrected (expected Retired)
D18-A: active handle clobbered (expected h3)
FAIL: D18-A overlap completion
D18-B: activate() destroyed the in-flight fading handle
FAIL: D18-B activate preserves fading handle
FAIL: D18-D structural ordering
```

復元後に元ファイルから復帰し、Debug/Release rebuild 後 45/45 PASS 再確認済み。

---

## 6. 既存 test の契約更新（明示記録）

`src/tests/NormalRetireDSPHandleCompareTests.cpp` の `testFadingRuntimeDSPHandle` は
`activate()` が fading handle を null 化する**実装詳細を encode していた**。
F-A により契約が変わったため該当アサーション（`activate` 直後の 1 行）と
コメントを更新した。更新の根拠は §2.1 の 2 つの production 経路
（`Timer.cpp:1142/:1305` の observe skip、`Timer.cpp:2129` の Emergency Retire 化）。
最終段落（`endCrossfade` で null 化・active==to）は**変更なし**で PASS 継続。

production の defect を test が固定していたため test 側を直した、という位置づけ。
production diff には含まれない。

---

## 7. 通常 path への非影響（等価性の根拠）

| 観点 | 内容 |
| --- | --- |
| 実行順序 | 通常 path は claim 成功 → register → begin → receipt → ramp-start。変更前後で同一 |
| fading handle 終端 | 通常 path は `endCrossfade` が null 化（F-C の条件 `== fromHandle` が成立）。変更前後で同一 |
| F-C 3 ガード | 通常 path では `toHandle` は `CrossfadingIn`、`activeHandle` は `toHandle`（step1 activate 直後）、`fading` は `fromHandle` — 3 条件とも成立するため書き込みは実行される。**no-op 化されない** |
| 可聴動作 | `ramp.start` は無条件実行。変更なし |
| 新規 authority / atomic / mutex | 0 |

---

## 8. 検証結果

| 項目 | 結果 |
| --- | --- |
| Debug build | exit=0 |
| Release build | exit=0 |
| Debug CTest | **45/45 PASS** |
| Release CTest | **45/45 PASS** |
| D18 targeted tests | T-CFX-A/B/C/D すべて PASS |
| Negative control | 上記 3 項目 FAIL を確認 |
| 静的解析 cppcheck | D18 hunk の指摘 **0**（残 2 件は hunk 外の既存 `uninitMemberVarNoCtor` / JUCE `unknownMacro`） |
| 静的解析 clang-tidy | D18 hunk の `clang-analyzer-*` / `bugprone-*` 指摘 **0**（残 1 件は harness から可視性が必須のエントリポイント `runSTG11D18CrossfadeTests` の `misc-*`） |
| ASan | **block** — `0xC0000139`（STATUS_ENTRYPOINT_NOT_FOUND、pre-main）。3 試行（既定 / LLVM runtime on PATH / MSVC runtime on PATH）すべて同一。PASS 扱いせず環境制約として記録 |
| ConvoPeq.md 再生成 | Generated `2026-10-01 18:40:06`、SHA-256 `64C4AD043D052772FEF4B596701F63737032AC2C21F3100E3E98A00327535CDC`、5,907,579 B、NEWER_SRC_COUNT 0 = **FRESH** |

---

## 9. D1〜D17 との境界

- D1〜D5（retire / epoch / MMCSS）に触れない（`DSPLifetimeManager` / `ISRRetireRouter` /
  `SnapshotCoordinator` / MMCSS のいずれも未変更）。
- D7〜D11（session 値 finite/enum guard）に触れない。
- D12〜D17 は監査のみ（NO-GO）。
- D14 の generation identity、D15 の DSP buffer、R4/P3 の recovery delivery は別 authority。
- 変更した production surface は `DSPHandleRuntime`（handle registry の NonRT 状態機械）と
  `DSPTransition::onPublishCompleted` の publish 完了尾巴の **2 ファイルのみ**。

---

## 10. 未確定事項 / 残存リスク

1. **fire-and-forget 経路に fading gate が無い**（`PublicationExecutor.cpp:61`）。
   D18-1 は「gate を追加する」のではなく「gate が無い状態での結果を incoherent にしない」
   ことで修正した。gate 追加は latest-wins / collapse の**仕様判断**を要し、
   Owner の基準（設計優先・speculative）に該当するため**行っていない**。
   重叠時に R1 が orphan になり得る点は §2.1 の通り F-A/F-C で緩和されるが、
   「R1 を fade 完了まで保持する」ことを明示する gate が必要かは将来設計課題。
2. **`Timer.cpp:1142` の observe は通常 path では既に null を見る**（`endCrossfade` が先に
   null 化するため）。本修正の対象外だが、self-contained intent として到達不能の
   可能性があり、別途の検討対象。harm の立証は本 STG の範囲外。
3. **既知 flake (R8)**: Debug `AudioEngineHarness` の STG-6 area `0xC0000005` と
   STG-8-D2b の timing assert は散発する。本次 4 ランの harness はすべて rc=0。