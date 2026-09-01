# D135-8/9 Gate G-4.4-P1 — Deterministic Stranding 修復（Work Report）

**Status: 実装完了・Debug/Release CTest 40/40×2 PASS**
**詳細:** `evidence/D135-8-9_GATE_G4-4-P1_STRANDING_FIX.md` / **ビルド:** `evidence/g44p1_ctest.log`

## 変更（1 ファイル +11 行、他 0）
`src/audioengine/AudioEngine.RebuildDispatch.cpp` — transport recovery の
1. build 失敗（`recoveryResult.runtime == nullptr`）
2. warmup 失敗（`validateWarmup != None`、未コミット DSP 破棄後）

の両 `continue` 直前に `runtimePublicationBridge_.markTransientFailure(recovery->obligationId);` を追加（+コメント）。これにより pop 済み obligation は delivery=None + retry counter 経由で redrive 候補化（次 tick で exactly one delivery 再取得）、4 回枯渇で ResolvedFailed 終端。D136-A の決定論的 stranding（Live+Transport 恒久固定）を解消。

## 実装前トレースで確認した不変条件
- durable slot 書込は cpp:998（push 失敗後のみ）/ 1159（NoAdmission のみ）/ consumer 側遷移に限定 → **通常 transport 失敗経路に durable slot は存在しない**。
- 唯一の例外は D136-B 既存窓（durable transient failure 由来）で、そこでは既存 durable call site（:1091/:1115）が同一 helper を既に呼んでいるため、**P1 は新規相互作用を導入しない**（窓の修復は P3 のまま）。
- markTransientFailure は Live のみ・id=0 no-op・カウンタ不変・枯渇時 resolve CAS 冪等（台帳無傷）。

## 結果
```
Debug:   full build OK → CTest 100% tests passed out of 40（DBG_CTEST_EXIT=0）
Release: full build OK → CTest 100% tests passed out of 40（REL_CTEST_EXIT=0）
```
既存テストに変化なし（stranding は Builder スレッド実在環境の問題で単一スレッドテストは不変 — 回帰なしの確認）。ConvoPeq.md を再生成（`Generated: 2026-08-31 00:52:36`、G-4.4-P1 マーカー 2 件）。

## 禁止事項遵守
durable 設計 / take / settle / memory order / SemanticRecoveryTarget / coalesce / supersession / retry scheduler / capacity / **wake 機構** — 全て無変更。wake を混ぜない指示により、**P1 で stranding は「恒久」から「次 wake まで遅延」へ改善された段階**であり、liveness 完全修復は P2（redrive→Builder wake）待ちである点を明示。

## STOP
P1 のみ実装・検証・提示まで。**P2 は未開始（指示待ち）。**
