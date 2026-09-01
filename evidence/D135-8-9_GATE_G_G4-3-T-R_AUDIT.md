# D135-8/9 Gate G-4.3-T-R — Regression / Compile-Fix Audit (read-only)

**Date:** 2026-08-30 23:42–23:55 (+09:00)
**Type:** read-only audit. Production source changes: **0**. Test source changes: **0**.
**Inputs:** working tree @ HEAD `5f6f48c` + uncommitted G-4.1…G-4.3-T deltas（cpp +85/−15, h +49/−8, tests +99/−0）
**Verdict: PASS**

> 副次作業（非ソース）: 監査対象確認のため `ConvoPeq.md` を再生成（`python output_sourcecode_markdown.py`、`Generated: 2026-08-30 23:47:13`。再生成前は G-4.2 helper マーカー 0 件＝旧世代だった）。ビルド evidence 取得用に `tools/g43tr_ctest.bat`（監査ハーネス、src 外）を追加。

---

## A. compile-only 修正の監査 → PASS

修正本体（`src/audioengine/ISRRuntimePublicationCoordinator.cpp:821-825`、anonymous namespace 冒頭）:

```cpp
namespace {
// G-4.2 helpers live at convo::isr anonymous-namespace scope; ObligationDomains is a nested
// member type of RuntimeIntentCoordinator — alias it here so the helpers can name it.
using convo::isr::RuntimeIntentCoordinator;
using ObligationDomains = RuntimeIntentCoordinator::ObligationDomains;
```

### A.1 宣言元・型定義（最新 ConvoPeq.md で確認）
- 宣言元: `ISRRuntimePublicationCoordinator.h:228-235`、`class RuntimeIntentCoordinator` の **public ネスト enum**:
  `enum class ObligationDomains : std::uint8_t { None=0, IR=1<<0, Conv=1<<1, EQ=1<<2, Config=1<<3, OS=1<<4 };`（G-4.1 (R1) 導入、I4 D12.2/D13）
- 最新 `ConvoPeq.md`（Generated 2026-08-30 23:47:13）: enum 定義 = line 57703、`using ObligationDomains` = 1 件（cpp 転写内）。ソース／派生スナップショット／実ツリーの三者一致。
- クロスチェック: AiDex（更新後）= h:228 type / cpp:825 alias / cpp:853-863 helper、serena `find_symbol` = cpp:852-864（0-based）で同一本体。

### A.2 戻り値型の解決
- `computeDomainCoverage(const convo::RuntimeBuildSnapshot&) noexcept`（cpp:853）の戻り値 `ObligationDomains` はエイリアス経由で `RuntimeIntentCoordinator::ObligationDomains` に解決。`SemanticRecoveryTarget::domainCoverage` フィールド型（h:280）と**同一型**であり、cpp:892 の集約初期化は型変換なしで適合。
- 実測: Debug/Release フルビルド成功（本監査 `evidence/g43tr_ctest.log`）、`ISRRuntimePublicationCoordinator.cpp.obj` 両構成コンパイル済み。

### A.3 型名解決以外の意味を持たないこと
- 修正は anonymous namespace（cpp:821-866、`namespace convo::isr` 内・cpp:10）の**中に閉じる**。このブロックの実体は G-4.2 helper 2 関数のみ。
- `using convo::isr::RuntimeIntentCoordinator;` — 外側スコープで既に可視の**同一エンティティ**を内側スコープに再導入する using-declaration。同一エンティティ参照のため曖昧化・遮蔽なし（合法、意味論的効果ゼロ）。
- `using ObligationDomains = ...` — 型エイリアス宣言。定義・ストレージ・overload set・ADL 集合を変えない。anonymous namespace は内部リンケージなので他 TU に一切露出しない。

### A.4 computeDomainCoverage の呼び出し箇所・戻り値・bitmask が G-4.2 から不変
- 呼び出し箇所: **cpp:892 の 1 箇所のみ**（`srt` 構築内）。
- 本体（serena 実測）: IR←`rebuildFingerprint.irIdentityHash!=0`、Conv←`convolutionConfigHash!=0 || convolverFingerprint!=0`、EQ←`dspParameterHash!=0`、Config←`computeBuildInputHash(buildInput)!=0`、`OS` ビットは設定されない（dormant bit）。`None` 初期化→ビット OR のみ、副作用なし。
- G-4.2 報告（`doc/work88/D135-8-9_GATE_G_G4-2_REPORT.md:19-20`）の記述（necessary-condition metadata / 構築順序 `{ir, conv, dspParam, coverage, convolverFingerprint, buildInputHash}`）と cpp:888-895 の現状が完全一致。G-4.3-T の cpp デルタは 822-825 の 4 行（コメント 2 + using 2）のみ。

### A.5 domainCoverage = necessary-condition metadata only
- `grep -rn domainCoverage src/` の全ヒット: h:280（宣言）、h:268/270・cpp:886（コメント）。**判定パスでの読み取り 0 件**。
- `SemanticRecoveryTarget::operator==`（h:283-289）は 5 semantic 値（ir/conv/dspParam/convolverFingerprint/buildInputHash）のみ比較、domainCoverage 除外。`findByKey`（h:382-389）/`tryInsert`（h:392-413）/capacity/`resolve`（h:423-436）/supersession（未実装）のいずれも domainCoverage を参照しない。

### A.6 ODR / namespace / shadowing / 型変換の新たな問題なし
- 名前空間スコープに別の `ObligationDomains` 宣言なし（grep で h:228 ネスト宣言のみ）→ エイリアスの遮蔽対象なし。
- using-declaration は同一エンティティ参照 → ODR 違反なし。underlying type `std::uint8_t` と `static_cast<std::uint8_t>` 演算が整合、窄め変換なし。
- 新警告なし: ビルドログの `C4324`（LogicalRecoveryObligation パディング、h:351）は **D116/D117/D119/D123 以前からの既存**（evidence 各 log に存在）。compile fix 由来の新警告・新エラー 0。

---

## B. G-4.3-T regression coverage 再監査 → PASS

### ソース（`src/tests/ISRSemanticValidationTests.cpp`、+99 行、main 登録 5 本）
| T | テスト | 検証内容 | 判定 |
|---|---|---|---|
| T1 | `testG43_T1_sameIdentityCoalesces` | same {h=3,target(501)} 再 submit → true / L==1（ΔL=0）/ coalescedCount+1 / `pop` 後 oblId==初回 id（T6 再利用） | ✓ |
| T2 | `testG43_T2_sameHandleDifferentTarget_New` | h=4、irIdentityHash 601 vs 602 → L==2（NEW） | ✓ |
| T3 | 既存 `testRLOE_C4_distinctHandlesSameTarget` | 別 handle×同一 target → L+2（R8 §4） | ✓（既存でカバー） |
| T4 | 既存 `testRLOE_C3_coalesceAtFull` | L==32 + matching key → COALESCE（L 不変・coalesced+1） | ✓（既存でカバー） |
| T5 | 既存 `testRLOE_C2_33rdRejected` | L==32 + non-matching → reject（L 据え置き） | ✓（既存でカバー） |
| T7 | `testG43_T7_terminalThenSameIdentity_NewAdmission` | submit→pop→`resolve(Published)`→L=0→同一 identity 再 submit→**NEW L=1**。terminal は `findByKey`（Live のみ一致）の候補にならないことを実証 | ✓ |
| T8 | `testG43_T8_coalesceKeepsRecoveryGeneration` | `g1=pop1.recoveryGeneration != 0` → COALESCE → `pop2.recoveryGeneration == g1`（再生成なし） | ✓ |
| T9 | 静的検証 | `recoveryGeneration = intent.intentId` 代入 **0 件**（grep）。intentId=`nextRecoveryIntentId_`（cpp:908）、recoveryGeneration=`++nextRecoveryGeneration_`（h:404、tryInsert のみ）— 独立カウンタでドメイン分離成立 | ✓ |
| T10 | `testG43_T10_buildSourceMetadataDriftCoalesces` | 同一 rebuildFingerprint（=同一 target）で `RuntimeBuildSnapshot.sampleRate`（**トップレベルフィールド**、RuntimeBuildTypes.h:64。target 導出は rebuildFingerprint 3 ハッシュ＋convolverFingerprint＋buildInput ハッシュのみで、このフィールドを読まない）だけ 48001 に drift → COALESCE・L==1。**D18.8 境界の正しい実装**: 「同一 semantic target → coalesce / semantic target 変化 → distinct」であり、snapshot-level metadata drift は identity を変えない | ✓ |
| T11 | 未実装（明示） | deterministic race 用 production synchronization hook は**追加されていない**（diff に該当なし）。既存 public API のみ使用 | ✓（対象外として正） |

`makeRecoverySnapshot(n)`: irIdentityHash=n / convolutionConfigHash=0x21 / dspParameterHash=0x42 / buildInput 既定ゼロ。T1/T2/T7/T8/T10 の target 操作は上記導出と整合。

### 実行結果（二重の実測）
1. **G-4.3-T 当初**（`evidence/g43t_build.log`、23:08/23:10）: exe 直接実行 Debug `DBG_EXIT=0` / Release `REL_EXIT=0`（失敗時 throw 構造 → 全テスト合格）。
2. **本監査**（`evidence/g43tr_ctest.log`、23:42–23:55）: **フルビルド→CTest** を両構成で実施:
   - Debug: full build OK → `100% tests passed out of 40`（40.12s）、`DBG_CTEST_EXIT=0`、#21 `ISRSemanticValidationRejects` Passed 0.40s
   - Release: full build OK（239 targets）→ `100% tests passed out of 40`、`REL_CTEST_EXIT=0`、#21 Passed 0.18s、#40 AudioEngineHarness Passed 15.85s
   - CTest 登録名 `ISRSemanticValidationRejects` は `add_test(... COMMAND ISRSemanticValidationTests)`（CMakeLists.txt:827）= 新規 5 テストを含む exe そのもの。

---

## C. G-4.3 invariant 再確認 → PASS

| # | 項目 | 実証 |
|---|---|---|
| C-1 | `CoalesceIdentity = {quarantinedHandle, SemanticRecoveryTarget}` | h:297-304（operator== = handle && target）。episode/generation 非関与 |
| C-2 | same identity → findByKey → CAS Live→Live → 既存 oblId 再利用 → ΔL=0 | cpp:926→931→937。`tryInsert` 非経由・`liveCount_` 非変動 |
| C-3 | different identity → tryInsert → ΔL=+1 / capacity≥32 → reject | cpp:946-950 + h:393-394（`liveCount_ >= kCapacity` → nullopt）+ h:408（+1 唯一サイト） |
| C-4 | terminal → resolve() → **同一 state atomic** → ΔL=−1 | h:423-436（id 再スキャン + `slots_[i].state.compare_exchange_strong`）— C-2 の CAS と同一 atomic。`resolveRecoveryObligation`（cpp:1023-1048、Retry は早期 return で Live 維持）と `markTransientFailure` 枯渇分岐（cpp:1087）の 2 サイトのみが −1 を発火 |
| C-5 | COALESCE 後の obligation slot mutation = 0 | cpp:935-944 は `slot(existing)` 読み取りのみ（id.load 937 / delivery 読 943）+ coordinator telemetry `recoveryCoalescedCount_`（939、slot 状態ではない）。共有経路の delivery bookkeeping（978/993/1008）は D105-R5-10 既存の単一書込者動作で G-4.3-T 無変更 |
| C-6 | RecoveryGeneration 再発行なし（coalesce 時） | `++nextRecoveryGeneration_` は h:404（tryInsert）のみ |
| C-7 | BuildGeneration との混同なし | cpp 内 `.generation =` 書込 0 件・`buildSource.generation` 参照 0 件。durable 転写は `intent.recoveryGeneration`（cpp:1000） |
| C-8 | RecoveryEpisodeId production reference = 0 | grep（design-only/Phase-II コメント除外後）**0 ヒット** |
| C-9 | ResolvedSuperseded production transition = 0 | h:314（enum 定義）のみ。cpp:1033-1044 の switch は Published/StaleSuperseded/ShutdownDiscarded/Failed の 4 終端のみ生成 |
| C-10 | canSupersede 系 production decision path = 0 | h:222/226/280/314 のコメントのみ、実装・呼び出しなし |
| C-11 | durable fallback / blind overwrite 未変更 | cpp:991-996（distinct-obligation 上書き防止ガード）＋ cpp:997-1007（同一 obligation のみ上書き）は G-4.3-T デルタ外（G-4.4 対象として温存） |
| C-12 | reservation semantics 未変更 | cpp:976（push 前 `pendingIntentCount_` +1）/ 983（drop 時 −1）無変更（INV-5 / P2-1 §1.1.4） |

---

## D. Diff boundary → PASS

- `git status`: 変更は **3 ファイルのみ**（cpp / h / tests）。AudioEngine 系・reclaim・shutdown・publish・durable・CMake 無変更。
- G-4.3-T デルタ（G-4.3-RF 比）= **テスト +99 行**（T1/T2/T7/T8/T10 + main 登録）+ **cpp compile-only fix 4 行**（822-825: コメント 2 + using 2）。h の変更は全て G-4.1/G-4.2 由来（既監査）。
- **明記（教訓）**: G-4.2/G-4.3 の read-only 監査（A1-A12 / R1-R12）は意味論検証に留まり**ビルド検証を含まなかった**ため、G-4.2 helper の型解決欠陥（C4430/C2146/C2143/C2447/C3861 @ cpp:849-850/888）は G-4.3-T のビルドで初めて露見した。本監査はこの教訓に従い grep ではなく **Debug build → Debug CTest → Release build → Release CTest の実測**（`evidence/g43tr_ctest.log`、40/40×2）を成立証拠とする。以後の Gate では read-only 監査にも早期ビルド検証を組み込むこと。

## 非ブロッキング観察（G-4.3-T 起因ではない）
1. h:960 コメント「入ってきた時点の rebuildRequestGeneration（coalesce 判定用）」は G-4.2 で意味が変わった（専用 ordinal・coalesce 判定非使用）→ **コメントドリフト**。次回編集窓で訂正推奨。
2. h:274-276 コメントの「cpp:836-841 3-field 集約初期化」参照は G-4.2 で cpp:888-895 の 6-field 化により陳腐化（行番号・内容とも）。
3. C4324（h:351）は D116 以前からの既存警告。
4. ツール状況: rg(WSL)/serena/AiDex/ctx で監査完遂。ccc（daemon 起動失敗）・semble（traceback）・graphify（グラフ未構築で 0 nodes）は本件では代替不要。

## 判定: **PASS**
compile-only fix は型名解決のみ（意味論不変）、G-4.3 invariant（identity/CAS 線形化/ΔL/generation 分離/episode=0/supersession dormant/durable 温存）は全て維持、regression coverage（T1-T10 + T11 明示対象外）はソース・実行の双方で成立。G-4.4 には進まない（指示どおり停止）。
