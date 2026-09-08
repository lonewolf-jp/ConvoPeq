# D153 — T3c Read-only Implementation Audit（Work Report）

```text
Production/Test/CMake changes: 0 / Build: NOT RUN / CTest: NOT RUN
基準: ConvoPeq.md 15:32:58（newer source 0 実測・git worktree はセッション開始時既存の G-4.x 変更のみ）
詳細: evidence/D153_T3C_READ_ONLY_IMPLEMENTATION_AUDIT.md
```

**Status: NO-GO（条件付き・単一ブロッカー D153-D）。それ以外の全トポロジー検査 PASS。**

## 主要照合結果

- **D153-A（6 sites 矛盾）= 解消・PASS**: ソース直接再計測で production `markTransientFailure(` 呼び出し = **6**（RebuildDispatch:1006/1033/1091/1115 の builder 4 + Orchestrator:311/401 の orchestrator 2）。過去資料の「5（builder2+orchestrator3）」は旧時点（builder 側 2 件は G-4.4-P1/D136-A で追加済み）で現行不支持。D152=6 と一致。`postRecoveryFailureSignal(` = 0（正常）。
- **D153-B = PASS**: `tryInsert(` call site 実測 = **1**（cpp:947、定義 h:392 除く）。expected 1 / actual 1。
- **§2 .h 11 項目 = 全 MATCH**: 旧構造（atomic id h:348/state h:350、plain delivery h:356、atomic counter h:361、check-then-act markT cpp:1059-1085、resolve の別語 counter reset h:428/429）が現存ことを実証。`RecoveryLifecycleWord` 0 hit = **expected baseline**（実装漏れではない）。
- **T3/T4/T5 前提 = 成立**: T4 の payload writer/Live publisher は全て CL（跨スレッド payload reader ゼロ・Builder は durable コピーのみ読む）→ 現行の Live 後 payload 書込（h:404・cpp:956-959）は安全だが D152 の順序強化が正。T5 は現行 1 発 CAS（state 単語）では 1 発設計が正しいこと、全文 CAS 化で contention 再試行が必須になること（R1）を分岐構造 cpp:927-947 上で確認。T3 の別 atomic 構造（CAS h:428 → store h:429）現存確認。
- **§5 挿入点 = 一意に確定**: runCoordinatorPhase processIntent 閉じ（:264）〜 redrive 注釈（:266）の間。adjudicate 既存 call site 0、CL 経路は ISRCoordinatorLoop:39 のみ。
- **§6 telemetry**: exhausted++ の現行位置 cpp:1079（resolve cpp:1080 の前・CAS 成否未確認）を**修正対象として記録**（本監査では変更せず）。
- **V1-V10 baseline**: V1=6 / V2=7 / V3=3+1 / V4=16 / V5=0 / V6=0 / V7=0 / V8=6 / **V9=NOT RUN — D153 read-only** / V10=存在。

## ブロッカー D153-D（新規発見）

**D152 §2.3「MSVC の std::atomic<16B> は lock pool」は事実誤認**で、現行コード一次資料と矛盾:
1. プロジェクト自身の前例 ISRDSPHandle.cpp:18-22:「MSVC で 16B atomic の is_lock_free() が false を返すのは **STL の保身的判定**。実際は CMPXCHG16B で lock-free に動作」— `std::atomic<DSPHandle>`（16B・alignas16）を runtime 検証付きで実運用中。
2. MSVC STL 本体ソース（raw 取得）: `_Atomic_storage<_Ty&, 16> { // lock-free using 16-byte intrinsics`（_WIN64）。compile-time `_Is_always_lock_free<=8` は保守的定数にすぎない。
→ 停止条件 #10 発火。実装の安全性はどちらの選択でも保たれるが、誤った事実記述の仕様では実装者・将来監査を誤誘導する。

## 解除手順（次アクション）

**D152-R1（read-only 仕様修正）**: §2.3 を撤回し、(a) `std::atomic<RecoveryLifecycleWord>` + alignas(16) + DSPHandle 前例の runtime `is_lock_free()` 検証（推奨・標準 memory_order API・前例整合）または (b) 明示 intrinsic wrapper（根記述のみ修正）のいずれかに確定。§12 V5 条件を選択に整合。static_assert 4 点は DSPHandle 前例と同型で妥当（is_always_lock_free を含めない点は D152 正しい）。
→ **D153-R**（修正節の軽量再監査）→ GO → T3c production implementation → build/CTest。

**Phase 2 実装凍結継続。P5/P6 非着手。STOP。**
