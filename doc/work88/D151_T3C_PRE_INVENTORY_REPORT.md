# D151 — T3c Implementation Pre-Inventory（Work Report）

**Status: inventory 確定 → D152（Patch Specification）進出可。変更 0・実装 0。**
**詳細:** `evidence/D151_T3C_PRE_INVENTORY.md` / **基準:** ConvoPeq.md `15:32:58`（mtime 再実測: より新しい src 0 件）。全 site 現行ソースへ戻って実測（rg + ccc grep 二重一致）。

## 成果 4 表
- **A. W Field Contract**: `alignas(16) {id:u64, state:u8, pending:u8, adjudicated:u8, delivery:u8, pad[4]}`（byte 格納・意味幅 75bit）。ObligationState 8 値のうち本番格納は 6 値（ResolvedRetry=cpp:1023 で格納不能、ResolvedSuperseded=dormant）。K=4。`pending` は**新設**（consecutiveFailureCount の改名ではない — 前者=未 adjudicate、後者=adjudicated 置換）。
- **B. Transition Matrix**: 7 遷移すべて Expected W/New W/CAS owner/失敗挙動/liveCount 効果を固定。禁止条件「CAS→state 再判定→別 atomic 更新」を明文化。
- **C. Source Patch Inventory**: 16 項目を実 site で固定（C1 構造/C3 tryInsert CAS/C4 resolve/C8 markT 分割/C12 runCoordinatorPhase 挿入点=processIntent 後・redrive cpp:272 前/C13-14 6 サイト置換/C15 telemetry）。
- **D. Test Adaptation Inventory**: markTransientFailure テスト呼び出し約 30 site・**ISRSemanticValidationTests.cpp のみ**（AudioEngineHarness/invariant_INV3_INV5 は 0）。テストからの delivery/counter/ObligationState 直接アクセス **0 件実測** → 適配は「同期 adjudication 前提」の置換に限定。**TEST-ONLY ラッパ（postSignal+即 adjudicate）**で全 40 テストの意味を保存、production 呼び出し 0 を grep 強制。新規テスト 5 本（postSignal 単体/一括 drain/飽和/terminal drop/2 スレッド）を D152 設計へ。

## 一次資料による確定・訂正（§2）
1. **intrinsic 正式名は `_InterlockedCompareExchange128`**（D150 の「16b」は通称。MicrosoftDocs cpp-docs を raw 取得）。仕様: Destination は **16 バイト整列必須（違反は GP fault）**、失敗時 comparand バッファに現在値が戻る（再試行が再 load 不要）、x64/ARM64、基本形はフルバリア。
2. **`std::atomic<16B>` は不採用**: MSVC STL `stl/inc/atomic` 一次ソース実測 `_Is_always_lock_free = _TypeSize <= 8 && pow2` → 16B は lock pool。intrinsic ラッパで lock-free 実装。
3. **新規設計帰責**: 16B の素の load は 2×u64 で**分断可能性あり** → 「load は advisory、commit は CAS のみ（全文比較のため偽成功生成不能）」の規律を全擬似コードに明記（D152 反映）。

## 中心論点の確定
- **delivery = W only・writer=CL only**（本番 11 サイト棚卸し、**cpp:1071 の RebuildThread plain 書込は C8 分割で消去可能**、実装後 grep 機械検証可）。
- **liveCount_ 単一 authority**: +1=tryInsert CAS 勝者 / −1=resolve・adjudicate 枯渇の各 CAS 勝者のみ。resolve∥adjudicate 枯渇は同一 W の Live 要求で相互排他 → duplicate decrement 不可。
- **markTransientFailure 責務境界**: 観測（postSignal・RebuildThread）/ 所有権遷移（adjudicate・CL）/ テスト互換（ラッパ）の 3 分離。production 6 サイトは postSignal へ機械置換。
- **tryInsert publication**: payload 先書→W CAS(Live 公開・release)→liveCount++。id が W 内に入ったことで「id 可視∧Live 可視」が単一語原子（現行 h:399-401 より強い）。

## 次アクション
**D152 — T3c Patch Specification** へ（exact struct/bit layout/CAS wrapper API/各関数擬似コード/memory ordering/tryInsert CAS/markT 変更/delivery 移設/liveCount authority/shutdown/R18-R20 適配を実装可能単位で固定）。open items 10 件を evidence §7 に列挙。**Phase 2 実装凍結は継続**（D153 read-only audit 通過まで build/CTest 非着手）。

**STOP — 実装 0。D152 の指示を待つ。P5/P6 非着手。**
