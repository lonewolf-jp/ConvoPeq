# work89 R1/R2/R3 残存リスク 改修計画案（v2 — 監査フィードバック反映版）

- **作成日**: 2026-09-10（v1）→ v2（ユーザー監査「条件付き GO」の修正指摘を反映・R-3a/R-3b 証明を実施済み）→ **v2.1（第2回監査「R-3c 修正必須・他 GO」の反映 — 本版で実装着手可）**
- **対象**: `doc/work89/INTEGRATED-BUG-LIST.md` §7.2「残存リスクの本質（3 点）」R1/R2/R3（BUG-065 関連）
- **調査基準**: 現行 HEAD `3b43a35d`（2026-09-10 10:31:31 +0900・working tree clean）
- **authority**: `ConvoPeq.md Generated: 2026-09-10 02:03:43` / NEWER_SRC_COUNT=0（work92 閉鎖後）
- **位置づけ**: work92（18 項目・CLOSED）に含まれない**新規 work item** の計画案。
- **分類**（work89 §17.8 の体系）: いずれも現行稼働経路に影響しない監査衛生化 = **OPTIMIZATION / PROCESS DEBT**。ACTIVE/OPEN BUG ではない。

## 0. v2 変更履歴（ユーザー監査 2026-09-10「条件付き GO」の反映）

| 監査指摘 | v2 での対応 |
|---|---|
| R-1「dead code」表現は過剰 — 「現行直接呼出し 0 の休眠コード」が正確 | §2 R-1 のコメント文案を「休眠コード（dormant・wrapper 経由の活性化配線あり）」表現に修正 |
| R-2 の allowlist（行番号許可）は監査上の穴になり得る | §2 R-2 を **DORMANT_EDGE 明示状態 + 構造的 allowlist（file/function/expected callee/reason 固定）**設計に変更 |
| R-3 の subsumption 主張は証明不足 — reset() と prepareToPlay() を分離し実装前に証明せよ | §1.5/§1.6 に **R-3a（reset() subsumption）・R-3b（prepareToPlay fresh-instance）の証明を実施・確定**（本 v2 で証明完了）。§2 R-3 を R-3a/R-3b/R-3c に分離 |
| wraparound の「無害」断定には世代識別/ABA の明記が必要 | §1.7 R-3c に正確な表現で記述 |
| G4 の grep AC とコメント例外の混在 | §3 の G4 を **G4-A/B/C に分離** |
| R-4 に D-2/D-3 を混ぜるな（検証境界が曖昧になる） | §2 R-4 を「R-1〜R-3 CLOSED 後に別 work item で再評価」と明確化 |

**最終判定（ユーザー監査 2026-09-10）**: 計画承認・**条件付き GO**。R-1 は実装可、R-2 は verifier 仕様修正（DORMANT_EDGE）、R-3 は再証明後に実装。**本 v2 で R-3a/R-3b の再証明は完了（§1.5/§1.6）**。

### 0.1 v2.1 変更履歴（第2回監査 2026-09-10「R-3c 修正必須・他 GO」の反映）

| 監査判定 | v2.1 での対応 |
|---|---|
| **R-1: GO** | 変更なし（§2 R-1） |
| **R-2: GO**（条件: 完全一致時のみ WARN・乖離は FAIL を明文化） | §2 R-2 に判定フロー（検出 → file/function 特定 → 実 callee 集合取得 → expected_callee と完全一致？ → Yes=WARN / No=FAIL）を実装仕様として明文化（§2 R-2・AC-R2-1 強化） |
| **R-3a: GO**（条件: subsumption 主張の表現を厳密化・§1.8 限定条件をコメントに必ず残す） | §2 R-3a のコメント文案を「mask 消費意味が filterState の band reset に限定される限り、reset() の全体 memset により subsume される」表現に修正 |
| **R-3b: GO** | 変更なし（§2 R-3b） |
| **R-3c: 修正必須** — 「ABA 実到達不能」「到達しても no-op 1 回」の断定は証明不成立（2^32 publication の有限上限を別途証明していない） | §1.7/§2 R-3c を全面的に書き換え。**R-3c は「バグ修正対象（CLOSED 対象）」から「serial protocol の有限世代識別限界 = DOCUMENTED LIMITATION」に再分類**。R-3a/R-3b の CLOSED 判定に影響しない |
| G4-B: 行番号を AC 本体にしない | §3 G4-B を「process() の Audio Thread 実行経路にのみ存在すること」を主条件、行番号（:589/:599/:1073/:1083）を補助 evidence に分離 |

**第2回監査の最終判定**: **R-1/R-2/R-3a/R-3b = GO・R-3c = 修正後 GO（DOCUMENTED LIMITATION 化）・R-4 = 今回除外で正しい。v2.1 は実装着手可能レベル。**

---

## 1. 調査結果の確定（計画の前提・v2 証明込み）

### 1.1 R1（非 atomic shadow への Non-RT 直接書込 = data race / UB）

**判定: 稼働経路は解消済み / 休眠コード内に同型書込が残存**

- reset()/prepareToPlay() からの rt シャドウ直接書込 6 行は work92 B-7a で削除済み（現行 Core.cpp:298-301 / :822-824 は契約コメントのみ）。
- しかし休眠コード関数内に同型書込が 6 行残存:

| 場所 | 書込内容 | UB 該当性 |
|---|---|---|
| `EQProcessor::syncStateFrom` — EQProcessor.Core.cpp:623 | `rtDeferredBandResetMask.store(syncedMask, relaxed)` | UB でない（atomic）。ただし Audio Thread の RMW と relaxed 同士の論理競合 |
| 同 :624 | `rtSeenBandResetSerial = syncedSerial` | **UB 該当**（plain uint64_t への Message Thread 書込） |
| 同 :625 | `rtSeenAgcResetSerial = syncedSerial` | **UB 該当**（同上） |
| `EQProcessor::syncGlobalStateFrom` — :687 | `rtDeferredBandResetMask.store(syncedMask, relaxed)` | UB でない（論理競合のみ） |
| 同 :688 | `rtSeenBandResetSerial = syncedSerial` | **UB 該当** |
| 同 :689 | `rtSeenAgcResetSerial = syncedSerial` | **UB 該当** |

- 呼出元は 0 件（`dead_code_callers_verifier.py` PASS・`rg` 全数実測）のため現状発火しない。
- **表現の訂正（v2・監査指摘反映）**: 両関数は「完全に dead」ではなく **「現行直接呼出し 0 の休眠コード（dormant）」** である。syncStateFrom には Message Thread jassert が、syncGlobalStateFrom には「Worker Thread からも安全」コメントが残り、将来の復活を意識した構造。R-1 のコメント文言もこれに合わせる。

### 1.2 R2（bandResetPacked の serial 0 巻き戻し）

**判定: work92 B-7b で解消済み。新変異（CAS による mask clobber）は R-3a/R-3b に分離して証明**

- B-7b により reset()/prepareToPlay() の `publishAtomic(bandResetPacked, 0)` は **CAS（serial+1・mask=0）ループ**に置換済み（Core.cpp:279-291 / :808-820 実測）。serial 単調性は回復。
- 新変異: B-7b の CAS は mask=0 で上書きするため、並行 `requestBandReset()` が置いた mask を clobber し得る。→ **R-3a（reset()）と R-3b（prepareToPlay()）に分離して証明（§1.5/§1.6）**。

### 1.3 R3（rtDeferredBandResetMask.store(0) と Audio Thread fetch_or の競合）

**判定: 稼働経路は解消済み**

- R3 の根拠となっていた `store(0)` は reset()/prepareToPlay() から削除済み。
- 残存する `rtDeferredBandResetMask` 操作は全て Audio Thread 専有（全 site 実測）: Processing.cpp:509 / :600 / :603(exchange) / :608 / :1084 / :1087(exchange) / :1110 — 全て process() 内。`exchange(0)` → `fetch_or(mask)` 再装填は同一スレッドの逐次 RMW で競合不能。
- 休眠コード内 :623/:687 は R-1 に包含。

### 1.4 【新発見】EQProcessor::reset() は「デッドコード」ではなく「休眠だが配線済み」

```
DSPCore::reset()                                    (DSPCoreLifecycle.cpp:381 — 現在は直接呼出元 0 件)
  ├─ convolverState->resetForRuntime()              (:383)
  └─ eqState->resetForRuntime()                     (:384)
      └─ EQRuntimeState::resetForRuntime()          (AudioEngine.h:731-734)
          └─ ref().reset()  == EQProcessor::reset()  ← 一跳びで活性化
```

- `dead_code_callers_verifier.py` はレシーバ whitelist ベースであり、wrapper 内 `ref().reset()` を検出できない（スクリプト自己申告の Known limitations）。PASS の意味は「DSPCore::reset() の直接呼出がない」。
- RuntimeBuilder.cpp:430 は fresh instance への `prepare()` のみ（reset() を呼ばない）— 実測確認済み。
- 活性化時の安全性前提: reset() の `memset(filterState)` は RT 所有データへの直接書込のため **Audio Thread 停止中**であることが前提（work89 §8.2 維持）。

### 1.5 【R-3a 証明・確定】reset() の CAS mask clobber は filterState 全 memset で包含される

**証明方針**: 「bandReset mask の唯一の消費効果 = filterState への memset」であり「reset() の memset はその全集合を覆盖する」ことをコード実査で閉じる。

| # | 証明項目 | 実測結果 | 判定 |
|---|---|---|---|
| P-A1 | **mask の唯一の消費効果は filterState の memset か** | mask の全消費 site は Processing.cpp:603-617（float 版）と :1087-1108（double 版）のみ。float 版: `exchange(0)` → `!canSafelyResetState` なら `fetch_or(mask)` 再装填 → `mask==0xFFFFFFFF` なら `memset(activeFilterState.data(), 0, sizeof(activeFilterState))` → さもなくば band 毎に `memset(activeFilterState[ch][i].data(), 0, sizeof(double)*2)`。double 版: 同一構造（`isAudioBlockSilent` 判定込み）。**mask は telemetry・他 state・serial acknowledgement には一切使われない**（`rg 'rtDeferredBandResetMask'` 全 11 site 実測・全て上記 reset 処理内）。 | ✅ 成立 |
| P-A2 | **reset() の memset は mask reset の対象集合を完全包含するか** | reset() は `std::memset(filterState.data(), 0, sizeof(filterState))`（Core.cpp:265）。`filterState` は `std::array<std::array<std::array<double,2>, NUM_BANDS>, kFilterChannels>`（EQProcessor.h:660、NUM_BANDS=20 / kFilterChannels=4 / 宣言子 `{}` 値初期化）。消費者の band 毎 memset（`[ch][i]` の 2 double × 4ch × 該当 band）はこの配列の部分集合。**全 memset ⊇ 任意の部分 memset**。 | ✅ 成立 |
| P-A3 | **serial/mask が他の機構（telemetry・acknowledgement 以外）に影響しないか** | `bandResetPacked` の全読出 site: Processing.cpp:595-600/:1079-1084（消費者）・EQProcessor.h:536/:545（requestBandReset の CAS）・Core.cpp:607（休眠 syncStateFrom）。serial の消費者側の意味は「rtSeen 更新（acknowledge）+ fetch_or(mask)」のみ。telemetry 参照 0 件。 | ✅ 成立 |
| P-A4 | **reset() 後に requestBandReset の要求を別途処理する必要はないか** | 3 ケースに分けて検討: (i) 要求が reset() の CAS より**前に** publish され消費者が未処理 → reset() の CAS が mask を 0 にするが、reset() 自身の全 memset が当該要求の効果（filterState クリア）を包含 → 要求は「処理済みと等価」。(ii) 要求が消費者により**処理済み** → 冪等（reset() が再度全 memset）。(iii) 要求が reset() の CAS より**後に** publish（(S+2, 0) を読んで (S+3, M') を書く）→ 消費者が通常どおり検知・処理。**損失は (i) のみで、(i) は包含により無害。** | ✅ 成立 |
| P-A5 | **係数・BandNode への影響はないか** | band reset は係数を触らない（係数は RCU BandNode exchange = Parameters.cpp の `exchangeBandNode` 経由で別機構）。mask がクリアされても係数更新は損なわれない。reset() 自身も `updateBandNode` ループで係数を同期する（Core.cpp:268-270 付近）。 | ✅ 成立 |

**R-3a 結論**: ✅ **証明完了**。reset() の CAS mask clobber は「filterState 全 memset ⊇ band reset の要求集合（P-A1/P-A2/P-A5）+ 全 interleaving で損失が (i) のみ且つ包含される（P-A4）」により**機能的に無害**。subsumption 主張は P-A1〜P-A5 を根拠として成立する。reset() は休眠（§1.4）であり、発火は Audio Thread 停止中に限定される点も前提として文書化する（R-3a コメントに記載）。

### 1.6 【R-3b 証明・確定】prepareToPlay() の CAS mask clobber は fresh-instance ownership により無害（subsumption 論拠すら不要）

**証明方針**: prepareToPlay() の呼出経路を全数列挙し、対象 EQProcessor が「未 publish の fresh instance」に限定されること（= 当該インスタンスの bandResetPacked に並行 writer が存在し得ないこと）を証明する。

| # | 証明項目 | 実測結果 | 判定 |
|---|---|---|---|
| P-B1 | **EQProcessor::prepareToPlay の呼出経路の全数列挙** | (a) `EQRuntimeState::prepare()`（AudioEngine.h:727-730: `ref().prepareToPlay(...)`）→ 呼出元は `eqState->prepare(...)` の **2 箇所のみ**（DSPCoreLifecycle.cpp:200〔diagLog 版〕と :270〔非 diag 版〕）— 両方とも **`DSPCore::prepare()`（同ファイル :72 定義）の内部**。(b) 直接呼出（`eqRt().prepareToPlay` / `eq.prepareToPlay`）: **0 件**（rg 実測）。(c) `DSPCore::prepare()` の呼出元: **RuntimeBuilder.cpp:430 の `runtime->prepare(...)` 1 箇所のみ**。 | ✅ 成立 |
| P-B2 | **対象インスタンスは必ず fresh か** | RuntimeBuilder.cpp:425 で `aligned_make_unique<AudioEngine::DSPCore>()` により新規生成 → :430 で `runtime->prepare(...)`（この間に publish なし）→ :435 `result.runtime = runtime.release()`。publish（activeRuntimeDSPSlot 経由の admission）は build 完了後の別段階。**prepare 時点で Audio Thread は当該インスタンスを一切参照していない**。同様に PrepareToPlay.cpp:259-263 の placeholderDSP も fresh 生成 → prepare → :286 release（未 publish）。 | ✅ 成立 |
| P-B3 | **fresh instance の bandResetPacked に並行 writer が存在し得ないか** | requestBandReset/requestAllBandReset/requestAgcReset の呼出元（全数）: Core.cpp:255/:256（resetToDefaults 内 — `this` が対象）・Core.cpp:505（setStateFromXML 系）・Core.cpp:571/:572（状態ロード系）・Parameters.cpp:89/:169/:192（band 単位 setter）。これらは **UI/Message Thread が `eqRt()`（= 現行 publish 済み DSPCore の EQ）または uiEqEditor（EQEditProcessor — 別インスタンス）に対して呼ぶ**。Rebuild Thread が構築中の fresh instance に対して UI 側が request を発行する経路は存在しない（fresh instance はまだ eqRt() でない）。 | ✅ 成立 |
| P-B4 | **live instance への prepareToPlay 再呼出は存在しないか** | サンプルレート/ブロックサイズ変更は rebuild（fresh DSPCore 生成）で対応し、live instance の再 prepare は行わない（ISR bridge 設計・D169-2 系の collapse no-op は engine/device レベル prepare の話で DSPCore::prepare とは別系統）。rg で live instance への prepareToPlay 経路 0 件を確認。 | ✅ 成立 |

**R-3b 結論**: ✅ **証明完了**。EQProcessor::prepareToPlay() は **「未 publish の fresh instance に対して Rebuild Thread から呼ばれる」ことが P-B1〜P-B4 により構造的に保証**され、当該インスタンスの bandResetPacked には並行 writer が存在しない。したがって B-7b の CAS(mask=0) は clobber し得る相手が存在せず、**subsumption 論拠すら不要**。ユーザー監査の予測（§5「より良い証明が可能」）どおり、こちらは ownership/lifetime property で閉じた。

### 1.7 【R-3c・v2.1 修正】serial 32bit wraparound と ABA の正確な記述（DOCUMENTED LIMITATION）

**確定してよい部分**:

- serial の意味は「順序比較」ではなく「**変更世代識別**」である。consumer は `bandResetSerialNow != rtSeenBandResetSerial` の**等価比較のみ**で検知し、monotonic 大小比較を行わない。したがって単純な `0xFFFFFFFF → 0` の wraparound 自体は順序判定を破壊しない。
- wrap 時（rtSeen=0xFFFFFFFF → published=0）も `!=` が成立するため検知され、以後 0→1→…と通常どおり追従する。

**ABA の理論的残存（v2 の「実到達不能」断定を撤回）**:

- serial は 32bit であるため、consumer が同一値 `S` を `rtSeen` に保持したまま **2^32 回以上の publication** が発生すると published serial が再び `S` に戻り、`serial != rtSeen` が false となって**その世代の変更を見逃す理論的可能性（ABA）が残る**。これは「通常の wraparound」(`0xFFFFFFFF→0` は検知できる) とは**別の問題**である。
- この ABA を「実到達不能」とするには、**publication 総回数または consumer 停滞時間について別途有限上限を証明する必要がある**。本 work item ではその上限を証明していない。v2 が置いた「UI 操作起因・秒間高々数百回」の見積もりは**運用上のリスク評価**としては有効だが、将来の publication source 追加（automation・preset loading・MIDI・host parameter automation・batch reset・stress test 等）を考慮すると、監査上の「到達不能」証明としては不十分である。
- したがって R-3c の分類は「無害」ではなく **「通常の wraparound は無害、2^32 世代再利用による ABA は理論上残存するが、現行運用では極低頻度」= 有限だが極低確率の世代識別限界**とする。

**R-3c の扱い（v2.1 再分類）**: R-3c は **R-3a/R-3b と異なり「バグ修正対象（CLOSED 対象）」ではない**。serial protocol の**有限世代識別限界**として限界を正確に記録する（DOCUMENTED LIMITATION）。これにより:

```text
R-3a  機能的 clobber     → 証明済み（§1.5）→ CLOSED 対象
R-3b  fresh instance     → 証明済み（§1.6）→ CLOSED 対象
R-3c  32-bit generation  → 限界を正確に記録 → DOCUMENTED LIMITATION（CLOSED としない）
```

という監査区分とする。将来 publication source を大幅に拡張する（automation 等の導入）場合は、本限界の再評価（64bit serial 化や検知方式の見直しを含む）をその work item の前提条件とする。

### 1.8 【R-3a/R-3b 証明の共通限定条件】（R-3c は限界記録であり本条件の適用外）

上記証明（R-3a subsumption / R-3b fresh-instance）は **「requestBandReset の mask が現在の消費実装（filterState memset のみ）を指す限り」** 有効である。将来 mask に新たな意味（telemetry、他 state、ack 以外の副作用）を追加する変更を行う場合は、本証明を再実施すること（R-3a コメントにその旨を明記する）。R-3c（serial 世代識別限界）は本条件の適用外であり、§1.7 のとおり DOCUMENTED LIMITATION として独立に記録する。

---

## 2. 改修項目（v2）

### R-1: 休眠コード内 shadow 書込の削除 + 契約コメント【実装可】

| 項目 | 内容 |
|---|---|
| 対象 | `src/eqprocessor/EQProcessor.Core.cpp` — syncStateFrom(:623-625)・syncGlobalStateFrom(:687-689) の計 6 行 |
| 内容 | 3 shadow 書込を削除し、両関数に次の契約コメントを配置: 「**本関数は現行直接呼出し 0 の休眠コード（dormant）**。EQRuntimeState wrapper 経由の活性化配線は DSPCore::reset() チェーンに存在するが、当該チェーン自体も休眠（`dead_code_callers_verifier.py` の DORMANT_EDGE 監視下・R-2 参照）。rt シャドウ（rtDeferredBandResetMask / rtSeenBandResetSerial / rtSeenAgcResetSerial）は Audio Thread 専有（work92 B-7a 契約・Core.cpp:298-301 参照）。**将来本関数を活性化する場合も rt シャドウへの Non-RT 書込は禁止** — serial は fetchAddAtomic 前進 + Audio Thread の自己更新に一任すること。」 |
| 根拠 | R1 の UB 該当行（:624/:625/:688/:689）の実体をゼロにする。動作影響ゼロ（直接呼出 0 件）。 |
| サイズ | ±10 行・1 ファイル |

### R-2: verifier の DORMANT_EDGE 監視設計【仕様修正済み】

| 項目 | 内容 |
|---|---|
| 対象 | `tools/dead_code_callers_verifier.py` |
| 設計（監査指摘どおり FAIL/ALLOW 二値から変更） | WATCHED エントリに `state` フィールドを導入: **ACTIVE**（検出 = FAIL・現行動作）と **DORMANT_EDGE**（検出 = 構造照合）。allowlist は行番号ではなく **{file, function, expected_callee, reason} の構造的照合**で固定する。「その行だから許可」ではなく「**この dormant edge は設計上意図されたもの**」という監査。 |
| **判定フロー（v2.1 明文化・監査条件）** | DORMANT_EDGE は「検出されたら WARN」ではない。**構造一致した場合だけ WARN とする**: (1) 検出 → (2) file/function 特定 → (3) **実際の callee 集合を取得** → (4) **expected_callee と完全一致？** → **Yes = WARN（DORMANT_EDGE 成立）** / **No = FAIL**。具体的に FAIL となるのは: expected にない callee が**追加**された / expected callee の一方が**消えた** / function が別関数に変わった / file が変わった。 |
| DORMANT_EDGE 登録エントリ（初期） | (1) `{file: src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp, function: AudioEngine::DSPCore::reset, expected_callee: eqState->resetForRuntime + convolverState->resetForRuntime, reason: intentional dormant wiring — DSPCore::reset 自体は直接呼出 0（work89 §8.2/§17.8・REMEDIATION_PLAN §1.4）}`。(2) EQProcessor.Core.cpp の syncStateFrom/syncGlobalStateFrom は R-1 適用後「shadow 書込なし」を ACTIVE 検証に格上げ（watch は継続）。 |
| 追加パターン | `eqState->resetForRuntime()` / `convolverState->resetForRuntime()` / `ref().reset()` を検出対象に追加（ACTIVE 検出 = FAIL・DORMANT_EDGE 登録 site のみ WARN）。 |
| 根拠 | 「wrapper 経由の休眠配線」という今回発見の問題構造を将来確実に監視できる。行番号 allowlist の監査上の穴（変更を見逃す）を構造照合で塞ぐ。 |
| サイズ | スクリプト +50 行程度 |

### R-3a: reset() subsumption 証明のコメント固定【証明完了・コメント化のみ】

| 項目 | 内容 |
|---|---|
| 対象 | `src/eqprocessor/EQProcessor.Core.cpp` reset() 内 B-7b コメント（:279-282） |
| 内容 | 追記: 「CAS は並行 requestBandReset() の mask を clobber し得るが、**現行の mask 消費意味が filterState の band reset に限定される限り、本関数の全体 memset により subsume される**: (i) 消費者の mask 効果は filterState memset のみ（Processing.cpp:603-617/:1087-1108・REMEDIATION_PLAN §1.5 P-A1〜P-A5）。(ii) 本関数は休眠（直接呼出 0）であり発火は Audio Thread 停止中に限定される。(iii) **mask に filterState リセット以外の意味（telemetry・他 state・ack 以外の副作用）を追加する変更を行う場合は、本 subsumption 証明を再実施すること（REMEDIATION_PLAN §1.8）**。」 |
| 状態 | **証明は §1.5 で完了済み**。実装はコメント追記のみ。 |

### R-3b: prepareToPlay() fresh-instance 証明のコメント固定【証明完了・コメント化のみ】

| 項目 | 内容 |
|---|---|
| 対象 | `src/eqprocessor/EQProcessor.Core.cpp` prepareToPlay() 内 B-7b コメント（:807-810） |
| 内容 | 追記: 「本 CAS は clobber し得る相手が構造的に存在しない: prepareToPlay の呼出経路は DSPCore::prepare()（RuntimeBuilder.cpp:430・fresh DSPCore 生成 :425 → prepare :430 → publish は :435 release 後の別段階）に限定され（P-B1/P-B2）、fresh instance の bandResetPacked に並行 writer は存在しない（P-B3 — request 系は eqRt()=現行 instance / uiEqEditor=別 instance 宛）。live instance への再 prepare 経路は 0 件（P-B4）。REMEDIATION_PLAN §1.6 参照。」 |
| 状態 | **証明は §1.6 で完了済み**。実装はコメント追記のみ。subsumption 論拠は不要（ownership/lifetime property で閉じた）。 |

### R-3c: serial 世代識別限界の記録【v2.1 再分類: DOCUMENTED LIMITATION — CLOSED 対象外】

| 項目 | 内容 |
|---|---|
| 分類（v2.1 変更） | **「バグ修正対象」ではなく「serial protocol の有限世代識別限界」**。R-3a/R-3b の CLOSED 証明を弱めないため、R-3c は CLOSED とせず限界を正確に記録する（§1.7 参照）。 |
| 対象 | EQProcessor.h:508-512 付近（bandResetPacked 宣言横） |
| 内容 | 追記: 「serial は順序比較ではなく**変更世代識別**であり、consumer は `!=` 等価比較のみで検知するため、単純な 32bit wraparound（0xFFFFFFFF→0）は順序判定を壊さない。ただし **consumer が同一値を保持したまま 2^32 回以上の publication が発生すると同一 serial 値が再利用され（ABA）、変更を見逃す理論的可能性が残る**。これを排除するには publication 総数または consumer 停滞時間の有限上限証明が別途必要であり、本設計では証明していない（REMEDIATION_PLAN §1.7 DOCUMENTED LIMITATION）。現行運用では極低頻度。publication source を大幅に拡張する（automation/MIDI/batch 等の導入）場合は、64bit serial 化または検知方式の見直しを前提条件とすること。」 |
| 状態 | **v2 の「ABA 実到達不能」「到達しても no-op 1 回」の断定を撤回**（§1.7 参照）。 |

### R-4: D-2 / D-3 との統合再評価【今回スコープ外・明確化】

| 項目 | 内容 |
|---|---|
| 方針（監査指摘どおり） | **R-1〜R-3 を CLOSED にした後**、別 work item として D-2（AGC state architecture の二重管理整理）/ D-3（デッドコード 6 関数削除）を再評価する。本計画に混ぜない（shadow ownership / dead code / AGC state architecture / verifier semantics の検証境界を分離するため）。 |

---

## 3. 実装順序と検証 gate（v2）

```
実装順序: R-1 → R-2 → R-3a/R-3b（証明は §1.5/§1.6 で完了済み・コメント化のみ）→ R-3c（文書化のみ・CLOSED 対象外の DOCUMENTED LIMITATION 記録）
前提: §1.5/§1.6 の証明は実装前に確定済み（監査条件を充足）

検証 gate（v2.1・G4 を 3 分割）:
  G1. Debug + Release ビルド PASS（MSVC）
  G2. CTest 40/40 ×2 config
  G3. tools/dead_code_callers_verifier.py PASS（DORMANT_EDGE 設計適用後の新仕様込み）
  G4-A. Core.cpp 内の assignment token 検出 0 件:
        `grep -n -E 'rtSeen(Band|Agc)ResetSerial\s*=' src/eqprocessor/EQProcessor.Core.cpp`
        → 契約コメント文も含め assignment token（`rtSeen…Serial =` の形）が 0 件であること
        （コメント内に例示が必要な場合は `=` を含めない表記にする）
  G4-B.【主条件】Processing.cpp の対象 assignment が process() の Audio Thread 実行経路に
        のみ存在すること（AST/関数スコープ照合で確認）。
        【補助 evidence】現行の該当 site は :589/:599/:1073/:1083 — 補助情報であり行番号は AC の本体ではない
        （将来の行番号ドリフトで AC が無意味化するのを避ける）
  G4-C. Core.cpp に契約コメントは存在してよい（検査対象外・G4-A の token 条件のみで判定）
  G5. cppcheck warning 変化なし（work92 §3 水準維持・修正箇所への直接指摘 0 件）
```

### 受け入れ基準（AC・v2）

| AC | 内容 | 達成条件 |
|---|---|---|
| AC-R1-1 | Core.cpp から shadow assignment が消滅 | G4-A 0 件 |
| AC-R1-2 | 契約コメント配置（休眠コード表現） | 両関数に「現行直接呼出し 0 の休眠コード」+「将来活性化時も Non-RT shadow 書込禁止」のコメントが実在 |
| AC-R2-1 | DORMANT_EDGE 設計の実装（完全一致時のみ WARN） | state フィールド + 構造的 allowlist（file/function/expected_callee/reason）が実装され exit 0。**判定フロー §2 R-2 どおり: 実 callee 集合が expected_callee と完全一致の場合のみ WARN・追加/欠落/関数変更/file 変更は FAIL** |
| AC-R2-2 | 活性配線の監視下化 | DSPCoreLifecycle.cpp:383-384 が DORMANT_EDGE 登録され、構造変化で FAIL する経路がテストされている |
| AC-R3a-1 | subsumption コメント（証明参照つき・厳密表現） | Core.cpp reset() B-7b コメントに「mask 消費意味が filterState の band reset に限定される限り subsume される」+ §1.8 限定条件（再証明義務）+ PLAN 参照が実在 |
| AC-R3b-1 | fresh-instance コメント（証明参照つき） | Core.cpp prepareToPlay() B-7b コメントに §1.6 の要旨 + PLAN 参照が実在 |
| AC-R3c-1 | DOCUMENTED LIMITATION コメント | EQProcessor.h に「wraparound は `!=` 検知を壊さないが、2^32 世代再利用による ABA は理論上残る・有限上限証明は未実施」の記録（§1.7）が実在。**CLOSED とはしない** |

---

## 4. 留意事項

1. **work item 分類**: 本計画は work92（CLOSED）の 18 項目には含まれない新規 work item。ACTIVE/OPEN BUG ではない（work89 §17.8 分類で OPTIMIZATION / PROCESS DEBT）。
2. **証明の有効範囲**: §1.5/§1.6 の証明は「mask が filterState memset のみを指す現行消費実装」に限定して有効。mask への新意味追加時は再証明（R-3a コメントに明記）。**§1.7（R-3c）は証明ではなく限界記録（DOCUMENTED LIMITATION）であり、CLOSED 判定の対象に含めない** — 将来 publication source を大幅に拡張する場合は 64bit serial 化等の再評価を前提条件とする。
3. **TSan**: 本環境に実行基盤なし（work92 FINAL_INVENTORY §4）。G4-A/B の grep 表明で代替検証。
4. **静的解析ツール**: cppcheck/clang-tidy は既知の環境制約あり（work92 §3）。gate 主柱はコンパイラ警告 + CTest + verifier + grep 表明。
5. **ConvoPeq.md 再生成**: R-1/R-3a/R-3b/R-3c のソース変更後は `python output_sourcecode_markdown.py` で再生成し NEWER_SRC_COUNT=0 を確認（既存運用ルール）。
6. **「Practical Stable ISR Bridge Runtime」原則との整合**: R-1 の shadow ownership 明確化は「RT は状態遷移・所有権管理を担わず、危険操作を Non-RT 側へ隔離する」原則と整合する（ユーザー監査 2026-09-10 確認済み）。

---

## 5. 参照

- `doc/work89/INTEGRATED-BUG-LIST.md` §7.2・§8・§15.4・§17・§17.8
- `doc/work92/IMPLEMENTATION_REPORT_20260910.md` §B-7a/B-7b
- `doc/work92/FINAL_INVENTORY_20260910.md` §3・§5-1
- `tools/dead_code_callers_verifier.py`
- 証明の主要実測行:
  - R-3a: EQProcessor.Processing.cpp:603-617（float 消費者）/:1087-1108（double 消費者）、EQProcessor.h:660（filterState 宣言・`std::array<std::array<std::array<double,2>,20>,4>{}`）/:153-155（NUM_BANDS=20・kFilterChannels=4）、EQProcessor.Core.cpp:265（reset() 全 memset）
  - R-3b: RuntimeBuilder.cpp:421-435（fresh 生成→prepare→release）、DSPCoreLifecycle.cpp:72（DSPCore::prepare 定義）/:200/:270（eqState->prepare）/:381-407（DSPCore::reset — 直接呼出 0）、AudioEngine.h:727-734（EQRuntimeState::prepare/resetForRuntime）、PrepareToPlay.cpp:256-286（placeholderDSP fresh 生成→prepare→release）、AudioEngine.h:1292-1300（uiEqEditor = EQEditProcessor 別 instance）、EQEditProcessor.h:28（`final : public EQProcessor` 継承だが prepareToPlay/reset の override なし・呼出 0 件）
  - request 系呼出元全数: Core.cpp:255/:256/:505/:571/:572、Parameters.cpp:89/:145/:169/:192（すべて `this`/eqRt()/uiEqEditor 宛・fresh instance 宛は 0 件）
  - bandResetPacked 消費者全数: Processing.cpp:595-600/:1079-1084（telemetry 参照 0 件）

---

## 6. 実施完了記録（2026-09-10・v2.1 実装）

**実装**: R-1 / R-2 / R-3a / R-3b / R-3c すべて実装完了。authority = ConvoPeq.md `Generated: 2026-09-10 14:44:04`。

| 項目 | 実装内容 | 状態 |
|---|---|---|
| R-1 | EQProcessor.Core.cpp: syncStateFrom / syncGlobalStateFrom の shadow 書込 3 行×2（:623-625/:687-689 → 実装後 :626-632/:693-699 付近に契約コメント）+ 未使用ローカル宣言 4 行×2 削除。EQProcessor.h: 旧「Worker sync」プロトコルコメントを shadow ownership 契約に置換（rtShadow ブロック + band reset shadow ブロック）。syncGlobalStateFrom のセクションヘッダに休眠注記追加 | ✅ CLOSED |
| R-2 | dead_code_callers_verifier.py に DORMANT_EDGE 機構実装（構造的 allowlist {file, function, expected_callee, reason}・完全一致のみ WARN・追加/欠落/変更は FAIL）。登録 edge 3 件: DSPCore::reset 配線（eqState/convolverState->resetForRuntime ×1 ずつ）+ EQRuntimeState/ConvolverRuntimeState wrapper（ref().reset() ×1 ずつ）。check_file に ACTIVE パターン（resetForRuntime / ref().reset()）追加 | ✅ CLOSED |
| R-3a | reset() B-7b コメントに subsumption 証明（§1.5 参照つき・厳密表現 + 再証明義務）を追記 | ✅ CLOSED |
| R-3b | prepareToPlay() B-7b コメントに fresh-instance ownership 証明（§1.6 参照つき）を追記 | ✅ CLOSED |
| R-3c | EQProcessor.h bandResetPacked 宣言横に DOCUMENTED LIMITATION コメント（§1.7 参照つき・CLOSED としない）を追記 | ✅ 記録済み（LIMITATION） |

**検証 gate 結果**:

| Gate | 結果 |
|---|---|
| G1 Debug ビルド | ✅ PASS（Ninja Multi-Config・414/414・`-j2` — 初回は全並列で cl.exe C1060 ヒープ枯渇、`-j2` で解消） |
| G1 Release ビルド | ✅ PASS（554/554・`-j2`） |
| G2 CTest Debug | ✅ **40/40 PASS**（UTF-8 コードページ付き） |
| G2 CTest Release | ✅ **40/40 PASS** |
| G3 verifier | ✅ PASS（`3 dormant edge(s) verified`・ACTIVE 違反 0 件） |
| G4-A | ✅ Core.cpp 内 `rtSeen(Band|Agc)ResetSerial =` assignment token **0 件** |
| G4-B | ✅ Processing.cpp:589/:599/:1073/:1083（process() Audio Thread 経路のみ・補助 evidence） |
| G4-C | ✅ 契約コメント 6 箇所実在（Core.cpp）+ EQProcessor.h ownership 契約 |
| G5 cppcheck | ✅ 変更箇所への新規指摘 0 件（pre-existing uninitMemberVar のみ・work92 §3 水準維持） |
| ConvoPeq.md 再生成 | ✅ `Generated: 2026-09-10 14:44:04`・work89 R-1/R-3 マーカー 11 件反映 |

**実装過程の環境記録**:

1. **並列ビルドの cl.exe C1060（ヒープ枯渇）**: 全コア並列（`/MP1` × ninja 全 jobs）で Debug ビルドが C1060 多発 → `ninja -j2` で解消。マシン負荷に依存する環境問題でありコード問題ではない。
2. **test 33 一時失敗（実装ミスではなく作業工程の事故）**: EQProcessor.h 変更前の依存 TU touch 操作で、bash `while read` がパス区切り/CR を誤処理し **`src/eqprocessorEQProcessor.Core.cpp\r` 等の CR 付き空ファイル 10 個**を src/ 下に生成。RuntimeWorldAuthorityProjectionTests が `fs::recursive_directory_iterator` で src/ を走査するため namespace 解決が崩れ test 33 が FAIL。`find src -name $'*\r*' -delete` で除去後 **40/40 PASS** — R-1〜R-3 の変更自体は無関係と確定。**教訓**: src/ 下のファイル一括 touch は `read -r` を使用し、実行後に `find src -name $'*\r*'` でゴミ生成を必ず確認する。
3. **evidence**: ビルド/CTest ログ `evidence/work89_r123_*.log`、実行スクリプト `evidence/work89_r123_*.bat`（commit 時に一緒に登録）。

**残置**: なし（R-4 = D-2/D-3 統合再評価は次回 work item・PLAN §2 R-4 のとおり）。本計画は **全項目 CLOSED（R-3c のみ DOCUMENTED LIMITATION として記録継続）**。

## 7. 閉鎖記録（2026-09-10 ユーザー最終判定）

ユーザー判定（現行 ConvoPeq.md 2026-09-10 02:03:43 基準との照合報告）により以下が確定した:

| 項目 | 最終状態 |
|---|---|
| R-1 | **CLOSED** |
| R-2 | **CLOSED** |
| R-3a | **CLOSED** |
| R-3b | **CLOSED** |
| R-3c | **DOCUMENTED LIMITATION**（CLOSED に含めない — 32bit wraparound 自体と 2³² 世代再利用 ABA を分離・後者の有限上限証明未実施の限定が現行ソースコメントに残存） |
| R-4 | **未着手・別 work item**（D-2/D-3 再評価） |
| work89 R1/R2/R3 | **実装・検証完了** |

**補足（ユーザー確認事項の応答）**:
- 現行 ConvoPeq.md での整合確認: `bandResetPacked` は `std::atomic<std::uint64_t>` で R-3c の DOCUMENTED LIMITATION がコードコメントに明記 / `syncStateFrom` の Non-RT shadow 書込廃止契約が R-1 どおり反映 — いずれも確認済み。
- **Practical Stable ISR Bridge Runtime 原則との整合**: 「RT は Read → Execute → Output に限定し、Publish/Retire/Delete の判断・実行を RT から分離する」設計に対し、R-1 の shadow ownership 明確化（shadow の唯一の書込主体 = Audio Thread・Non-RT は atomic publish 前進のみ）はこの原則に沿う。Publish/Retire 責務分離とも矛盾しない。
- 変更境界の評価: R-1 = dormant code の衛生化 / R-2 = verifier の監視能力向上 / R-3a/b = 証明の永続化 / R-3c = 限界の明文化 — **新たな production behavior は導入しない**きれいな境界。

> **work89 R1/R2/R3 改修: CLOSED**（R-3c は DOCUMENTED LIMITATION として限界記録を継続）。authority = ConvoPeq.md `Generated: 2026-09-10 14:44:04`。変更は commit 待ち。
