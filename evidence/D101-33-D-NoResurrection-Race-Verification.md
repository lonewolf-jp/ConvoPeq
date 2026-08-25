# D101-33-D — No-Resurrection Race Verification（独立監査報告書）

- **実施日**: 2026-08-25
- **作業種別**: read-only independent audit（ソースコード変更 **0**）
- **判定**: **PASS**（全 16 条件充足。§4 に 2 件の誠実な観測事項を記録 — いずれも GAP 非該当）
- **基準**: ConvoPeq.md **2026-08-25 14:08 再生成版**（ローカル実ソースから）
- **入力**: D101-33-C 実装（実装者検証結果を鵜呑みにせず再検証）

---

## 1. 基準化

```
python output_sourcecode_markdown.py   → ConvoPeq.md 再生成 (2026-08-25 14:08)
git status                             → 変更 = D101-31-D〜D101-33-C の累積のみ（本監査中の変更なし）
git diff --check                       → whitespace エラーなし
```

---

## 2. D-1 Admission / Close の linearization 独立追跡

### 2.1 同一 atomic word CAS の再確認（コード構造から）

```c
// ISRShutdown.cpp tryAdmit:
uint32_t expected = consumeAtomic(packedState_, acquire);
...
compareExchangeAtomic(packedState_, expected, desired, acq_rel, ...)   // ← 同一変数

// ISRShutdown.cpp closeAdmission:
uint32_t expected = consumeAtomic(packedState_, acquire);
...
compareExchangeAtomic(packedState_, expected, desired, acq_rel, ...)   // ← 同一変数
```

✅ **両者とも `packedState_` 単語への CAS。** 全順序な modification order が成立し、
`P.tryAdmit < C` または `C < P.tryAdmit` の二択が構造的に保証される（第三の interleaving 不存在）。

### 2.2 tryAdmit は副作用より前か

✅ `enqueueRuntimePublicationFireAndForget`(AudioEngine.h:4524) の **最初の文**が
`tryAdmit(1)`（:4541）。失敗時は即 `return {Failed, CallerDestroy}` —
handle 登録 / registry / ownerChannel / X5 residency のいずれにも触れない（コメントに明記・実測一致）。

### 2.3 closeAdmission は実際の閉鎖点か

✅ ReleaseResources.cpp:77（requestShutdown :75 直後）に移動済み。
T_s1' 以降の全 tryAdmit は Closing/Closed を読んで拒否される。
旧位置(:210 相当)には呼び出し残骸なし、joinProducers retry loop は同位置に維持。

### 2.4 CoordinatorState::ShuttingDown の admission authority 復活チェック

✅ 残存使用箇所は以下のみ（全て非 admission 用途）:

| 位置 | 用途 |
|---|---|
| Coordinator.cpp:553 | requestShutdown による store（drain-mode signal の設定） |
| Coordinator.cpp:558 | markShutdownComplete の precondition check |
| Coordinator.cpp:825 | **submitRecoveryRequest の Recovery gate（Path C）** — D101-33-B/C で意図的に維持した Recovery 専用 authority。Publication admission ではない |
| ISRShutdown.h:169 | コメント |

→ Publication admission authority への復活 **なし**。

---

## 3. D-2 Path B 全 producer 到達性監査

### 3.1 obligation 生成 site の全列挙

| 操作 | production site | 判定 |
|---|---|---|
| `enqueuePublicationIntent(` | **AudioEngine.h:4609（facade 内）のみ** | ✅ bypass なし |
| `ownerChannel().enqueue` | **AudioEngine.h:4591（facade 内）のみ** | ✅ |
| `registry().registerPublish` | **AudioEngine.h:4587（facade 内）のみ** | ✅ |
| `IntentType::Publish` 設定 | enqueuePublicationIntent 内部（choke point）+ facade intent 構築のみ | ✅ |

他の caller はテストのみ（AdmissionPackedStateTests / SemanticValidation / SoakTests —
いずれも coordinator 直接操作の unit test）。

### 3.2 直接 producer → token 到達性

```
PrepareToPlay.cpp:155/:277 ─┐
ReleaseResources.cpp:175  ──┤
Timer.cpp:994             ──┼─→ commitRuntimePublication(:4602)
Transition.cpp:25         ──┘        └─ enqueueRuntimePublicationFireAndForget(:4524)
PublicationExecutor.cpp:57-66            └─ tryAdmit(1) (:4541) ★全経路通過
   （CoordinatorLoop deferred resubmit）
```

✅ **publication obligation を生成する site はすべて token 取得を通過する。bypass は存在しない。**

---

## 4. D-3 Case B/C/D 独立再検証

### 4.1 Case B（副作用ゼロ）

Test 13 が検証する不変条件（close → tryAdmit fail 後）:
outstanding==0 / pendingIntentCount_==0 / residency==0 / isFullyDrained()==true / state==Closing。
→ **独立確認 OK**（テストコード再読・条件過不足なし）。

### 4.2 Case C（最重要 — テスト実装そのものの critical review）

Test 14 の構造を監査者の視点で再検証した結果:

| 検証ポイント | 評価 |
|---|---|
| race 結果の拘束 | ✅ 各 round で「admit 成功 ⇒ push 側処理へ」「admit 失敗 ⇒ 即 return（push しない）」が排他に分岐し、round 後に `residency == pushed数` で**過剰予約/漏れ予約の双方**を検出する |
| outstanding()==0 が「producer 終了後を見ているだけ」か | ❌ ではない。producer thread 内で tryAdmit 成功直後に `outstanding() >= 1` を**即時 assert**しており、token の観測可能性を並行区間で確認している。join 後の ==0 は追加の漏れ検査 |
| failed admission 後の副作用ゼロ検査 | ✅ 失敗時は return のみで pushed/residency に触れない。round 後の `residency == pushed` により「失敗したのに reserve された」ケースを検出可能 |
| residency==pushed 比較で ownership leak を見逃すか | ⚠️ **限界あり（誠実な記録）**: 本 unit test は choke point + token primitive を直接駆動し、ownerChannel/registry を含まないため、facade レベルの ownership leak はこのテストでは観測不可。→ **代替保証**: (a) ownerChannel/registerPublish への到達は token取得後のみ（facade 構造・§3 実測）、(b) facade 失敗経路の take/unregister/release はコード検査済み（D101-33-C）、(c) token なしでの Owner 移譲は構造的に不可能。よって ownership leak は **by-construction で排除**されており、unit test の限界は GAP ではない。将来の強化として AudioEngineHarness レベルの race stress test を推奨（observation 記録） |

複数回実行: **3 回連続 PASS**（0.29-0.31 sec/回）。admit-first/close-first の比率は
スケジューラ依存のため PASS 条件にしない（指示どおり）。

### 4.3 Case D

Test 15 が queue-full 到達を実際に確認した上で、admit→push fail→X5 rollback→token release→
pendingIntent 非計上の chain を検証。→ 再確認 OK。

---

## 5. D-4 shutdown 後 Publish commit の不存在

証明すべきこと: 「queue 内の Publish Intent は必ず closeAdmission 前に tryAdmit 済み」

```text
queue への Publish 入口 = enqueuePublicationIntent（production 唯一 caller = facade :4609）
facade は tryAdmit 成功後にのみこの関数に到達する（:4541 が最初の文）
tryAdmit と closeAdmission は同一 word CAS → close 後の到達は不可能（§2.1）
∴ queue 内の全 Publish Intent は close 前に admitt 済みである ＝ Q.E.D.（構造的証明）
```

- ProcessIntent に discard を追加していない ✅（変更不要の証明が成立）
- ProcessIntent.cpp 自体も未触碰（grep で変更 diff なし確認）

---

## 6. D-5 Q0 の意味の再検証

```
Path A publication   → Orchestrator.cpp:69      tryAdmit ┐
Path B publication   → AudioEngine.h:4541       tryAdmit ├─ packedState_
Recovery             → RebuildDispatch:319 /
                       AudioEngine.h:4443       tryAdmit ┘
Build                → （Recovery build 経路に含まれる: :319）
        ↓
outstanding() = 上記すべての未解放 reservation 数
        ↓
joinProducers() は count!=0 を拒否 → waitForDrain retry
```

✅ **Path B を含む全 publication transaction が Q0 の観測対象に収束。**
`tryAdmit success → outstanding()>0` は Test 14 が producer thread 内で直接 assert。
D101-33-A の GAP（Path B 観測外）は解消済みことを独立確認。

---

## 7. D-6 authority resurrection 監査

| 項目 | 要求 | 実測 | 判定 |
|---|---|---|---|
| publicationBacklogCount_ | Publish admission に使用されない | facade/choke point 参照ゼロ（dead counter のまま） | ✅ |
| pendingIntentCount_ | Publish admission に流用されない | Observe/Recovery/Quarantine 専用（ProcessIntent 分岐不変） | ✅ |
| publicationIntentResidencyCount_ | transport authority のまま | drain 条件 + ProcessIntent decrement のみ。admission 判定には不使用 | ✅ |
| CoordinatorState::ShuttingDown | admission authority 復活なし | §2.4 のとおり Recovery gate（意図的維持）+ drain signal のみ | ✅ |
| AdmissionPackedState | Publication admission authority | 4 path すべての token source | ✅ |
| outstanding() | Path B 含む in-flight を観測 | §6 のとおり | ✅ |

D101-32-F の authority separation（token=shutdown obligation / residency=transport）は
崩れていない。

---

## 8. テスト再実行結果

| テスト | 結果 |
|---|---|
| #21 AdmissionPackedState ×3 回連続 | ✅ Passed / Passed / Passed（15 tests internal） |
| 全 CTest Debug | ✅ **100% passed out of 38** |
| 全 CTest Release | ✅ **100% passed out of 38** |

---

## 9. PASS 条件チェックリスト

* [x] 最新 ConvoPeq.md 再生成（14:08 版）
* [x] Path B 全 producer が tryAdmit に収束（§3 — bypass ゼロ）
* [x] tryAdmit / closeAdmission が同一 atomic word の CAS（§2.1）
* [x] Case B の副作用ゼロ（§4.1）
* [x] Case C の P→C が token により outstanding() 観測下（§4.2 — producer 内即時 assert）
* [x] Case C の C→P が副作用ゼロ reject（§4.2）
* [x] close 後の新規 Publish obligation が構造的に不可能（§5 Q.E.D.）
* [x] ProcessIntent discard 不要の証明を維持（§5 — 未実装のまま証明成立）
* [x] Path B が Q0 の観測対象（§6）
* [x] publicationBacklogCount_ 非依存（§7）
* [x] pendingIntentCount_ を Publish admission に流用しない（§7）
* [x] X5 residency authority 不変（§7）
* [x] Path A/Recovery/Build の admission regression なし（当該ファイル diff なし・#19/#20/#21 PASS）
* [x] Debug 全CTest PASS（38/38）
* [x] Release 全CTest PASS（38/38）
* [x] ソースコード変更 0

# VERDICT: D101-33-D = **PASS**

---

## 10. 観測事項（GAP 非該当・将来参照）

1. **Unit test の観測限界**: Test 14 は choke point + token を直接駆動するため、race 下の
   ownerChannel/registry leak は direct 観測対象外。by-construction 保証（token 先行 +
   facade 失敗経路の rollback）とコード検査で補完済み。将来の強化として
   AudioEngineHarness レベルの facade race stress test 追加を推奨。
2. **デストラクタ経路**: CtorDtor.cpp:113 の ~AudioEngine は requestShutdown() を呼ぶが
   closeAdmission() とのペアがない（pre-existing 特性）。destruction 中に publication
   producer は存在しないため低リスクだが、完全性を求めるなら dtor path への
   closeAdmission ペアリングを将来タスク候補とすること。

## 11. 次ステップ

D101-33-D PASS により **D101-33-E — 未コミット変更群の最終差分監査 / Commit Readiness Audit**
へ進行可能（コミット判断はその後）。
