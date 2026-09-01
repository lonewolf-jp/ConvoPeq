# D140 — G-4.4-P4 Memory-Order / Cross-Thread Delivery + Intermittent AV Audit (read-only)

**Date:** 2026-08-31 (+09:00)
**Type:** read-only diagnostic audit. **Production source changes: 0. Test source changes: 0.**
**P4-0 基準:** `ConvoPeq.md Generated: 2026-08-31 09:28:16`（P3 完了時再生成版）。`git status` = P3 完了時と同一の 5 ファイル（RebuildDispatch +11 / Threading +16 / coordinator.cpp +121 / coordinator.h +70 / tests +399、HEAD `5f6f48c` 比）— **P3 以降の余計な変更なしを確認**。
**Final classification: B — DATA RACE / UB（delivery と durable-slot plain フィールド）** ／ 断続 AV の因果は **UNPROVEN**（H1/H2 不採用、§7 参照）

---

## 1. delivery 全アクセス（read/write 列挙・スレッド・同期）

### Writer
| # | 場所 | 値 | thread | plain/atomic | lock | ordering | 対象 entry |
|---|---|---|---|---|---|---|---|
| W1 | h:402 tryInsert | None | CoordinatorLoop | plain | なし | n/a | 新規 slot |
| W2 | cpp:978 submit push 成功 | Transport | CoordinatorLoop | plain | なし | push 後 | slotIdx |
| W3 | cpp:993 submit defer | None | CoordinatorLoop | plain | なし | n/a | slotIdx |
| W4 | cpp:1008 submit durable | Durable | CoordinatorLoop | plain | なし | predicate release(1007) 後 | slotIdx |
| **W5** | **cpp:1078 markTransientFailure** | **None** | **RebuildThread** | **plain** | **なし** | **fetch_add(acq_rel) は別フィールド** | **slot(i)** |
| W6 | cpp:1169 redrive durable 付着 | Durable | CoordinatorLoop | plain | なし | predicate release 後 | slot(idx) |
| W7 | cpp:1194 redrive transport 付着 | Transport | CoordinatorLoop | plain | なし | push 後 | slot(idx) |
| W8 | cpp:1186 P3 repair 再同期 | Durable | CoordinatorLoop | plain | なし | n/a | slot(idx) |

**W5 のスレッド確定（コメントでなく呼び出し連鎖で）**: `markTransientFailure` の production caller は
- RebuildDispatch:1006/1033（P1 transport recovery 失敗）と :1091/1115（durable recovery 失敗）— **rebuildThreadLoop 内 = RebuildThread**。
- Orchestrator:311（trySubmitImpl 内、関数開始 line 40）/ :401（submitPublishRequest 内、開始 357）。`submitPublishRequest` の caller は Commit.cpp:822 = `enqueuePublicationIntentForRuntimeCommit`（Commit.cpp:782）の内部呼び出しで、その production caller は RebuildDispatch:1051/1131/1337 のみ（全て rebuildThreadLoop）。もう一处 Orchestrator:731 = `processDeferredAdmission`（RebuildDispatch:914 経由、RebuildThread）。
→ **全 markTransientFailure 実行は RebuildThread。CoordinatorLoop からの呼び出しは 0 件。**

### Reader
| # | 場所 | thread | 用途 |
|---|---|---|---|
| R1 | cpp:900 wasDeferredBefore | CoordinatorLoop | submit 事前スナップショット |
| R2 | cpp:943 coalesce 早期 return 判定 | CoordinatorLoop | 二重 delivery 防止 |
| R3 | cpp:1116 redrive 候補選別 | CoordinatorLoop | None のみ再付着 |
| R4 | cpp:1142 redrive 冪等 check | CoordinatorLoop | None でなければ no-op |
| （tests）| 間接のみ（pop/take/liveCount で観測） | 単一スレッド | — |

RebuildThread 側の delivery 読み取りは**存在しない**（markTransientFailure は読む前に無条件 None を書く）。

## 2. Data race 形式判定（P4-2）

対象ペア:
- **A** = RebuildThread: W5 `delivery = None`（plain store）
- **B** = CoordinatorLoop: R3/R4 `delivery == None`（plain load）、W7/W8（plain store）

### A → B（Builder の None 書込が redrive に伝わるか）
- 非枯渇経路（count 1-3）: W5 の直後に同一 obligation で Builder が発行する release は**なし**（consecutiveFailureCount の acq_rel fetch_add は別メモリ位置であり、CoordinatorLoop は redrive 経路でそれを読まない。slot.state も触らない）。CoordinatorLoop 側の acquire 読取（state@1114）は**別の** release（resolve CAS）にしか reads-from しない。→ **synchronizes-with エッジ無し = HB 無し**。
- 枯渇経路（count 4）: W5 → resolve CAS（acq_rel、h:428）→ CoordinatorLoop の state acquire 読取（1114）が当該 CAS 値を読む → HB 成立。この経路のみ SAFE。
### B → A
- W5 は delivery を読まないため read-write 競合なし。ただし **W7/W8（CoordinatorLoop store）× W5（RebuildThread store）は write-write 競合**（同一 byte、両方 plain、無 lock）。到達例: 窓（slot=O∧None）で Builder が while ループ再 take→失敗→W5(None) と、同一 tick の CoordinatorLoop redrive による W8(Durable 再同期) が並行。
### 判定（3 値）
- 非枯渇経路の A→B: **DATA RACE / UB**（[intro.races]: 同一 object の conflicting accesses、少なくとも一方非 atomic store、HB 無し）。
- 枯渇経路: **SAFE**（resolve CAS が HB）。
- B→A write-write: **DATA RACE / UB**。
- 実務注記（判定には用いない）: x86-64 TSO + 単一 aligned byte アクセスのため tearing せず、観測される最終値はいずれの順序でも論理的に収束（None 後 Durable=整合、Durable 後 None=窓再開で冪等）。**しかしこれは「たぶん同時実行されない」ではなく「同時実行されるが UB」という認定**であり、コメント h:324「CoordinatorLoop-only — non-atomic by design」は **D105-R18 以降偽**（D138 追認）。

### 同型の追加競合（durable-slot plain フィールド — D136-D の系）
CoordinatorLoop の redrive 読取（state cpp:1158 / recoveryObligationId cpp:1185 / submit ガード 991-992）に対し、RebuildThread が take/settle/discard で同一 plain フィールドへ書込（cpp:1208/1242/1246/1229/1248）。submit 経路の wake は rebuildMutex エッジで順序付けられるが、**requestRebuild 起床時の take() 読取と redrive 書込の間には HB が無い**（D136-D で記録済み、今回再確認）。→ 同じ **DATA RACE / UB** 分類。

## 3. P3 repair は新規 race を作ったか（P4-3）

- **新規 race クラスの追加なし**: W8 は既存 W7 と同一フィールド・同一スレッド対（CoordinatorLoop vs RebuildThread W5）の書込であり、P3 は新しい共有状態も新しいスレッドも導入していない。
- **既存 race による新しい失敗モード（1 件、狭い）**: repair は `state(1158)` と `recoveryObligationId(1185)` を**無 HB で 2 段階読取**する。Builder の settle(false)（構造体リセット）が両読取の間に挿さると、`stale DurablePending ∧ stale oblId==O` → 実際は空 slot に対して delivery=Durable を再同期 → **O が実体無く Durable 表示で stranded**（同一 {h,target} の再 submit か shutdown まで）。P3 以前は同じ stale 読取でも transport push に落ちるため実体が残り、この失敗モードは存在しなかった。
- 分離結論: **P3 が race を作ったのではない。P3 は既存 durable-slot race の下で新しい（狭い）帰結を追加した。** P3 の逐次整合性上の正しさ（T-P3-1..6）とは無関係（混同禁止指示に従い分離して記載）。

## 4. redriveWakePending_ のスレッド安全性（P4-4）

- Writer: cpp:1170/1187/1195（redriveDeferredRecovery のみ）— 呼び出し元は runCoordinatorPhase（Threading:270）と submitRecoveryRequest（cpp:902）の 2 箇所、**いずれも CoordinatorLoop**。
- Reader/clear: `consumeRedriveWake()`（h:505-508）の production 呼び出しは Threading.cpp:279（runCoordinatorPhase = CoordinatorLoop）のみ。テストは単一スレッド。
- **RebuildThread からのアクセス 0 件**（grep 全数で確認）。set/clear とも同一スレッド → **SAFE**（新規 cross-thread ordering なし。P2 設計意図どおり）。

## 5. 断続 AV の独立診断（P4-5）— 部分特定・因果 UNPROVEN

### 再現（今回実測）
- 最終 P3 バイナリ（Release）で **100 回中 3 回**（iter 69/86/98）exit=0xC0000005 を再現。D139 の 20/20 合格は低頻度ゆえの未検出だったことを訂正。
- bisect 済み事実（D139）: **P2+P3 テストブロックをスキップしても再現**（その構成で same-holder repair はデッドコード）→ P3 実装・テストは誘因でない。

### 関数形状（逆アセンブル 3 ビルドで一致）
- 現在のビルド: 関数開始 0x1400176C0、フォールト形状命令 0x140017771 ほか。bisect2 ビルド 0x140013275、bisect1 ビルド 0x1400179bc/179ca — **同一関数形状**（ビルドレイアウト差のみ）。
- 形状: ①`[rdx+0x64]=1`、`[rdx+0x65]=2|6` の 2 バイト書込、②`[r9+0x198..0x1B0]` の 4 qword 書込、各前に `sub/cmp 0x8000000000000000` による **double `!=`（NaN 安全）ガード**、③宛先 r9 は非 NULL チェック済みだが不正領域（**「非 NULL ≠ 有効オブジェクト」**）。
- 呼び出し元は 1 関数内 2 箇所（0x140018A2E/0x140018A4D → call 0x1400177C0 経由）。
- **この形状は delivery 経路のコードと一致しない**: delivery は単一 byte（LogicalRecoveryObligation 内 0x148 付近）で、4 連続 double の NaN ガード書込も 0x64/0x65 の 2 バイト書込も coordinator の delivery 操作には存在しない。→ AV の直接原因関数は delivery 書込経路ではない。

### シンボル解決の限界（正直な記録）
Release リンクは PDB を生成しない設定（/DEBUG なし）で、`SymFromAddr` は 487（ERROR_INVALID_ADDRESS）。Debug PDB はレイアウトが別物で対応付け不能。**関数のソース名特定は未達**（専用診断ビルドが必要 — §8）。

## 6. H1-H4 評価（P4-6/7）

| 仮説 | 評価 | 根拠 |
|---|---|---|
| H1 delivery race → UB → heap 破損 → 遅延 AV | **不採用（証拠なし）** | 破損を示す直接証拠（アロケータメタデータ異常等）未取得。ただし §2 の形式 UB は**理論的に H1 を可能にする**ため排除もできない。フォールト関数形状が delivery 経路と不一致であることは間接的に H1 に不利 |
| H2 recovery race → stale lifetime → AV | **不採用（証拠なし）** | bisect2 で recovery 系テストスキップでも再現。ただし既存テスト（C11-C16/R18/R20）は recovery 経路を多用するため分離不能 |
| H3 既存 DSP/object lifetime bug → 破損 → 後段顕在化 | **可能性あり（未証明）** | 4-double NaN ガード + 2 状態バイトの書込先が stale pointer の形状。テストバイナリにリンクされる retire 系（ISRRetireRouter/ISRRetireRuntimeEx）が候補だが特定未達 |
| H4 P2/P3 無関係の既存 UB | **最有力（ただし証明は §8 の診断ゲート待ち）** | P2 era の CTest は各 1 回の実行のみで ~3% 頻度を検出できず、P3 変更で挙動不変という時間的一貫性 |

### lifetime chain（P4-7、フォールト関数について）
allocation→init→publication→reader→mutation→retire→destruction の各段を特定するにはシンボルが必要。**現状 reader access と destruction の HB 関係は判定不能**（UNPROVEN の理由の内訳: ①関数未特定、②r9 の由来未特定、③破損発生点とクラッシュ点が異なる可能性＝遅延顕在化）。

## 7. 修復契約候補（P4-9 — 実装禁止、確定は次 Gate）

**先決の問い**: なぜ delivery を cross-thread shared にするのか。現設計は「失敗の観測者（Builder/Orchestrator）が即座に adjudicate する」ため RebuildThread から plain 書込を行っている。これは ownership protocol の選択であり、**atomic 化は byte レベルの tearing 競合を隠すが、§3 の 2 段階 stale 読取（state→oblId）や durable-slot 全体の整合は救わない**（「delivery だけ atomic にすれば全体が正しくなる」とは限らない — 指示どおり明記）。

候補（優先順・要ユーザー判断）:
1. **(i) adjudication の CoordinatorLoop 化**: 失敗観測を CoordinatorLoop 宛イベント（既存 intentQueue_ 相当）へ転送し、delivery/durable-slot の全書込を単一スレッドに寄せる。HB 問題が構造的に消える。P4-2/§3/§5(durable 側) を一括修復。変更範囲大。
2. **(iii) 状態の atomic ドメイン統合**: delivery を state atomic と同一 CAS 領域へ統合（または durable slot を atomic 化 + 単一 release/acquire プロトコル）。中変更。
3. **(ii) delivery のみ atomic**: 最小だが §3 の 2 段階読取と durable-slot 残りを放置。**単独では不十分と判定**。

**P4 実装の要否**: 形式 UB の是正として**要**。ただし契約の形（i/iii）は §8 の AV 診断結果に依存する（AV が H1 由来なら (ii) は不可）。

## 8. 推奨次マイクロゲート（本 Gate の範囲外・未実施）
- AV 専用診断: Release+`/DEBUG`（または ASan/`/fsanitize=address`）ビルドでの再現実行、WER LocalDumps + minidump の Exception 解析、フォールト関数のソース名確定 → H1-H4 の決着。
- これが終わるまで P4 実装契約は (i)/(iii) のどちらとも確定しない。

## 成果物チェックリスト（P4-10）
1. delivery read/write 全数 ✅（W1-W8/R1-R4）
2. thread ownership 全数 ✅（W5=RebuildThread を呼び出し連鎖で確定）
3. happens-before graph ✅（§2: 非枯渇 A→B エッジ無し／枯渇は resolve CAS／B→A write-write）
4. data-race 判定 ✅（DATA RACE / UB、3 値の内訳付き）
5. redriveWakePending_ 安全性 ✅（CoordinatorLoop 専用・SAFE）
6. P3 repair との因果分離 ✅（新規 race なし・新失敗モード 1 件）
7. AV の call/lifetime chain △（形状特定・シンボル未達＝部分）
8. H1-H4 評価 ✅（H1/H2 不採用、H3/H4 未証明）
9. 修復契約候補 ✅（(i)>(iii)>(ii)、単独 atomic 化は不十分と明記）
10. P4 実装要否 ✅（要・ただし契約確定は AV 診断ゲート後）

**P4 audit 最終判定: DATA RACE（B）— delivery/durable-slot plain フィールドの形式 UB を確定。断続 AV との因果は UNPROVEN（H1/H2 不採用）。**
**STOP — 実装 0。P5/P6/AV 修正には進まない。**
