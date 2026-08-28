# D113-A — CacheMap destructor comment cleanup (evidence)

**Date:** 2026-08-29
**Scope:** `src/audioengine/AudioEngine.h` — `EQCacheManager::CacheMap` destructor 直上のコメントブロックのみ
**Baseline:** D114 PASS (`evidence/D114_PHASE_I_OPERATIONAL_VALIDATION_AUDIT.md` — 15/15 PASS, Debug 40/40, Release 40/40, 0 regression)

## 状況

D113-A の初回適用（前セッション、未コミット）により、旧コメント

```
// ★ P0-2: デストラクタ。
//   通常パス: retire のみ（コピー先マップが参照中のため delete 不可）。
//   Shutdown: resolve → delete → reclaim（全マップ同時破棄のため安全）。
```

が詳細版コメントブロックに置き換えられていた。旧コメントには 2 点の事実誤認があった:

1. 「Shutdown: resolve → delete → reclaim」— 実コードは **reclaim → resolve → delete** の順（reclaim 成功時のみ物理解放）。
2. CacheMap dtor の起動経路として direct delete（~EQCacheManager）しか記述しておらず、deferred delete 経由（m_retireRouter consumer thread, epoch-gated）が欠落。

本セッションでは、この未コミットの D113-A コメントブロックを実コードに対して全文検証し、乖離のみを修正した。

## 実コード検証（コメント記述の根拠）

| コメントの主張 | 検証結果 | 根拠 |
| --- | --- | --- |
| (a) ~EQCacheManager — writeMutex 下、cacheMapPtr を nullptr 交換後に直接 delete | ✅ 一致 | `AudioEngine.Cache.cpp:161-171`（lock → exchangeAtomic(nullptr) → unique_ptr delete） |
| (b) enqueueDeferredDeleteNonRt 経由の deleter — m_retireRouter consumer thread, epoch-gated | ✅ 一致 | `AudioEngine.Cache.cpp:16`（tryEnqueueDeferredMap）, `:40`（storeNewMap — `delete static_cast<CacheMap*>(p)` デリータ登録）, `AudioEngine.h:4219` enqueueDeferredDeleteNonRt 定義 |
| shutdownReclaim (ShutdownReclaimAuthority) | ✅ 一致 | `ISRRetireRouter.h:382` |
| directDelete は禁忌 | ✅ 一致 | `doc/work88/REPAIR_PLAN.md:2412` 付近（「directDelete は RT 参照中の UAF リスクがあるため禁忌」） |
| Destroy branch 順序: tryShutdownQuiescentReclaim → rt.resolve → delete EQCoeffCache、reclaim 失敗時は物理解放スキップ | ✅ 一致 | `AudioEngine.h:2127-2133` |
| 通常パス: retire のみ、CacheMap 自体の delete はしない | ✅ 一致 | `AudioEngine.h:2138-2143`（else branch） |
| H.11.11.9.4 delete-before-reclaim blocker への対応 | ✅ 一致 | ccc search で `doc/work88/REPAIR_PLAN2-dash2.md:3535-3545` を確認（「❌ delete → reclaim」→「✅ ReclaimPermit → ReclaimStarted → physical destruction → ReclaimCompleted」）。現行コードは修正後の順序 |

## 本セッションの修正内容（コメントのみ・3箇所）

実コード検証の結果、コメント内の**相互参照の乖離**を修正した（記述内容の変更はなし）:

1. `AudioEngine.h:4198 enqueueDeferredDeleteNonRt` → `AudioEngine.h enqueueDeferredDeleteNonRt (定義 4219)`
   — 原因: コメント挿入 (+21 行) 自身による行番号の自己ずれ。
2. `RELEASE-3:422 drainAllQuarantineStore` → `ISRRetireRouter.h:247 drainAllQuarantineStore (shutdown drain: ReleaseResources.cpp:410/505)`
   — 原因: 「RELEASE-3:422」は実在しない参照（リポジトリ全体検索で該当なし）。実コードの shutdown drain 呼び出し箇所に修正。
3. `実コードの順序 (line 2106-2111):` / `通常パス (shutdownPhase < Destroy, line 2117-2123):` → branch 識別による参照（`shutdownPhase >= Destroy branch — 本 dtor の if ブロック` / `本 dtor の else ブロック`）
   — 原因: 同上の自己ずれ。行番号参照はコメント挿入で再びずれるため、安定した branch 識別に変更。

## 制約遵守

- production logic / destructor / ownership / release() / retire() / API / atomic / queue / Recovery / I4 / test: **変更なし**（D115-A で証明）
- RecoveryEpisodeId / RecoveryGeneration / SemanticRecoveryTarget / 5-field fingerprint / snapshot freeze / MPSC化 / semantic supersession: **未実装のまま**（Phase-II 要素に触れていない）

## 生成物

- `evidence/D113A_AudioEngine.h.diff` — AudioEngine.h の git diff（単一 hunk `@@ -2081,7 +2081,30 @@`、CacheMap dtor コメントブロックに局在）
- 本ファイル

## 使用ツール

WSL: rg / sed / fd / gcc（コメント strip 証明）、ccc search（H.11.11.9.4 履歴確認）、semble search、graphify explain。
serena / AiDex MCP は本セッション未接続（次回セッション開始時に接続。CLI 経由で AiDex インデックスは 2026-08-28 に增量整備済み）。

## 判定

**D113-A: PASS（コメント変更のみ・実コード差分 0）** — commit candidate のみ。**commit は未実施**（D115-A 監査後に凍結判断）。
