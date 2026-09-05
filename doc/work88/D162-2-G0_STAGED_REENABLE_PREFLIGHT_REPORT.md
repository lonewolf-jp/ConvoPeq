# D162-2-G0 Work Report — S3/V-D Staged Re-enable Preflight（read-only）

- Work item: D162-2-G0（D162-2-G 開始前 preflight audit・production source 変更 **0**）
- Date: 2026-09-04
- 基準ソース: ConvoPeq.md `Generated: 2026-09-04 18:29:10`
- 判定: **PASS（G1 実施可）** — 詳細は evidence/D162-2-G0_PREFLIGHT_AUDIT.md

## 0. Executive Summary

1. **G0-1**: S3 の switch は「retire 呼び出しの有無」（`clearDeferredForShutdown` 内
   disposition block が注記付きで空・Orchestrator md:67397-67442）。V-D の switch は
   `if (false && ...)` 2 行（ReleaseResources VerifyDrained・md:38372/:38381、
   B-era direct destroy 形状で休眠）。S1/S2/S4/E-4d/E-1/E-2/E-3 は全て有効維持を確認。
2. **G0-2**: authority 経路（retireRegisteredDSP → map erase + EBR enqueue）と
   direct 経路（destroyRolledBackDSP）を実コードで再構成。S3 の呼び出し経路は 3 系統
   （EmergencyDrain / C1 fallback / midrun-E-4d 後）で、いずれも RebuildThread join 後 or
   RebuildThread 自身 → INV-D162-6/7 適合。
3. **G0-3**: AudioSegmentBuffer は DSPCore subtree に**含まれない**（NoiseShaperLearner
   専属・engine member teardown で破壊）。F 修正の効き方は「B-era で S3/V-D destroy の
   MKL churn が mismatch による registry 汚染に接触して AV が顕在化していた」機構の
   汚染源除去。**B-era 分離マトリクス（S3-EBR crash / S3-direct clean）は F 後は予測力なし**。
4. **G0-4**: G1(S3 only) → G2(V-D only) → G3(both) → G4(final soak) の gate matrix を確定。

## 1. G0 の新規所見（G1/G2 設計に影響）

| # | 所見 | 影響 |
| --- | --- | --- |
| N-1 | **V-D direct destroy（現行休眠コード）は `runtimeDSPHandleMap_` を erase しない**。registry は Reclaimed + slot freelist 返却のため、slot 再利用時に E-2 `retireByHandle` が破壊済み DSP を map HIT → 二重破壊する latent 経路が静的に成立（shutdown 後は admission closed で不発） | V-D-b（authority retire・map erase あり）を推奨。V-D-a（`if(false)` フリップの最小差分）を選ぶ場合は注記書き換えが必要 |
| N-2 | **V-D-b は INV-D162-8（shutdown 破壊は EBR 単経路）に文言適合**。direct destroy 選択の根拠だった「EBR 遅延が AV の原因」は D0 で誤帰属と判明（実因 = allocator mismatch） | G2 の推奨案 = V-D-b（`retire(resolved)` に差し替え）。V-D-a/b の確定は G1 PASS 後にユーザー指示 |
| N-3 | **S3 は標準 soak で発火しない可能性**（EmergencyDrain body は `isEmergencyDrainRequested()` true 時のみ・ReleaseResources md:38193-38194） | G1 の PASS 証跡に `CLEAR_SHUTDOWN_DISPOSITION` 件数 > 0 を要求。60-gen で 0 件なら targeted trigger（EmergencyDrain 要求条件 / C1 到達条件）を G3 前に確定 |
| N-4 | S3 を `clearDeferredForShutdown` 内に挿入すると、midrun 経由（E-4d 済み）でも S3 retire が走るが、**1 回目の retire で map erase 済みのため find 失敗 → no-op**（二重 EBR enqueue 構造的に不可能・INV-D162-3） | G1 実装は 1 block で 3 経路を一括カバー可能。E-4d との併存は安全 |
| N-5 | `detectStuckReaders` jassert（pend=1 滞留・Debug-only pre-existing）と **E-3 jassert（pendingRetire 残留・INV-D162-8 違反）は別物**。G 期間中は Signature 分類規約（A=D0 chain 禁止 / B=detectStuckReaders 記録のみ / C=その他即 stop）で運用 | ユーザー指示の「別シグネチャ扱い」を G1-G4 全段階に組込み済み（audit §4/§7.2） |

## 2. Staged Gate Matrix（確定版）

| Stage | S3 | V-D | production 変更 | Gate ladder |
| --- | --- | --- | --- | --- |
| **G1** | **ON（EBR authority）** | OFF | Orchestrator `clearDeferredForShutdown` に `retireRegisteredDSP(..., "shutdown-clear")` + DIAG 挿入（1 block） | Build(Debug/Release/RWDI) → CTest(Debug 40 / Release 39) → Debug 6-gen → RWDI 6-gen → RWDI 60-gen |
| **G2** | OFF（G1 revert） | **ON** | ReleaseResources V-D block（V-D-b 推奨 / V-D-a 選択可・G1 PASS 後に確定） | 同一 ladder |
| **G3** | **ON** | **ON** | G1 + G2 の結合（D162-2-B で 0xC0000005 だった組合せの F 後再証明） | 同一 ladder（本命 Gate） |
| **G4** | ON | ON | なし（G3 状態で 60-gen x2 決定論） | exit 0x0 / crash 0 / residual 0 / EBR pend 0 / ovf 0 / lifecycle 突合 / shutdown Drain→Reclaim→VerifyEmpty |

各 stage PASS 条件（10 項目）: 0xC0000005 なし・Signature A（D0 chain）なし・
S3 destruction count == CLEAR disposition count == destroy 実件数・EBR pend 永続増加なし・
residual 0・shutdown clean（E-3 DIAG 0 件）・audio drop/XRUN なし・AudioSegmentBuffer 破壊正常・
E-4 会計 closure（CREATE = CONSUME + DISCARD + OVERWRITE + CLEAR 全 disposition 対応）・
INV-D162-6/7/9 静的適合。

## 3. G1 最小変更（次工程・承認待ち）

```cpp
// clearDeferredForShutdown() 内・deferredSlot_.reset() 直前に挿入
retireRegisteredDSP(deferredSlot_->request, "shutdown-clear");   // ★ D162-2-G1
```

- 対象 1 関数 1 block（Orchestrator のみ）・V-D は未接触。
- 既存 `event=CLEAR ... (no disposition)` DIAG 文言更新（mismatch 解消済みのため）。
- INV-D162-8（EBR 単経路）/ INV-D162-6（dtor body 内 digest）/ INV-D162-9（CLEAR 出口
  disposition カバー）の全てに適合。

## 4. 次のアクション

1. **D162-2-G1 実装**（上記最小変更）+ Gate ladder 実行 — ユーザーの go 待ち。
2. G1 の 6-gen 時点で `CLEAR_SHUTDOWN_DISPOSITION` 発火有無を確認（N-3 の解決）。
3. G1 PASS 後: V-D-a / V-D-b の確定指示（推奨 V-D-b）→ G2。
