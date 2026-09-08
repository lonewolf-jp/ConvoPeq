# D137 — G-4.4-P2 Builder Wake / Liveness 修復（Work Report）

**Status: 実装完了・AC-P2-1..5 成立・Debug/Release CTest 40/40×2 PASS**
**詳細:** `evidence/D137_G4-4-P2_WAKE_FIX.md` / **ビルド:** `evidence/g44p2_ctest.log`

## 監査（実装前）→ 欠陥確定
redrive 付着（durable cpp:1158-1170 / transport cpp:1173-1178）から Builder wake への配線が 0 本であることを再確認（rebuildCV.notify 全 4 サイトに redrive 由来なし・wait predicate は recoveryPending のみ・markTransientFailure/settle は wake 非発火）。4 遷移 A-D を実装前後で対比証明。

## 実装（wake/liveness のみ、最小）
- coordinator `.h` +13: `redriveWakePending_`（CoordinatorLoop 専用 plain bool — 同一スレッド set/consume）+ `consumeRedriveWake()`
- coordinator `.cpp` +2: **実付着（None→Transport/Durable）2 箇所のみ**で latch set
- `AudioEngine.Threading.cpp` +16: runCoordinatorPhase の redrive 直後、latch 消費 → `{rebuildMutex} recoveryPending=true` → `rebuildCV.notify_all()` — **既存 submitRecoveryIntent wake プロトコルの再利用**（毎 tick notify への逆戻りなし = F6-5 意図維持）。opportunistic redrive（capacity reject 経路）も同一 latch でカバー
- tests +150: T-P2-1（latch は実付着のみ・1 回のみ発火）/ T-P2-2/3/4（生産 triple を模した**実 2 スレッド** wake プロトコルテスト: signal-before-wait / signal-after-wait / 64 ラウンド混合順序）/ T-P2-5（K=4 終端 + wake 源消滅）

## Lost-wake 証明（notify ではなく predicate state × linearization）
recoveryPending は mutex 下 set の状態信号。3 順序（redrive→notify→wait / redrive→wait→notify / wait→redrive→notify）すべてで predicate が信号を保持するため取りこぼし不成立。Builder の clear（:974）後付着も同一 tick の latch consume で再設定され、pop/take ドレインか次 wait 進入で確実に消費。

## 開発中の重要発見（記録）
初版 T-P2-4 が **coalescing フラグの誤モデル**（signal 数=consumed 数の期待）でハング — CTest #21 が 9 分停止（実測・プロセス強制終了）。生産セマンティクス（1 wake で全ドレイン、flag はイベントカウンタでない）に合わせ 1 outstanding + progress ハンドシェイクへ修正して合格。この失敗自体が recoveryPending の状態信号意味論の実証となった。

## 結果
```text
Debug:   full build OK → CTest 100% tests passed out of 40（DBG_CTEST_EXIT=0）
Release: full build OK → CTest 100% tests passed out of 40（REL_CTEST_EXIT=0）
既存 40 テスト回帰なし / ConvoPeq.md 再生成（Generated: 2026-08-31 01:33:50、G-4.4-P2 マーカー 15 件）
```

## 禁止事項遵守
durable slot 設計・take/settle・memory-order・recoveryIntentQueue_・coalesce/supersession・RecoveryEpisodeId・capacity・retry 回数・markTransientFailure 意味・**P3 double representation（B 遷移窓は従来どおり温存）**・retry scheduler・shutdown — 全て無変更。

## 残存（次 Gate 入力）
P3（二重住処）/ P4（memory-order 契約）/ P5（Building overwrite 契約）/ P6（実 Engine 統合 wake テスト — 本 P2 の実スレッドテストはプロトコル層であり、AudioEngineHarness レベルは未整備）。

## STOP
P2 完了。**P3 には進まない（指示待ち）。**
