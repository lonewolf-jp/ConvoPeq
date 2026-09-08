# D162-2-G2 Work Report — V-D-b Staged Re-enable（authority retire 統一）

- Work item: D162-2-G2（D162-2-G ステージ 2/4・S3=ON / V-D=ON via authority retire）
- Date: 2026-09-04
- 基準ソース: G1 適用済み working tree + G2 変更（ConvoPeq.md `Generated: 2026-09-04 20:28:50`）
- 判定: **PASS**（詳細: evidence/D162-2G2_VD_REENABLE_EVIDENCE.md）

## 0. Executive Summary

V-D（VerifyDrained 最終 active/fading DSP 破壊）を direct destroy（`destroyRolledBackDSP`）から
authority（`DSPLifetimeManager::retire` = map erase + registry Retired + requestReclaim +
EBR enqueue）に統一して有効化した。**3 soak（Debug 6-gen / RWDI 6-gen / RWDI 60-gen）すべて
exit 0x00000000・crash dump 0**。V-D が **retired=1 → EBR Success → destroy → remaining=0**
の完全閉包を実測達成し、**direct destroy 0 件・stale map entry 0（retireByHandle MISS 実測）**
を証明。Signature A/B/C・E-3 違反とも 0 件。

## 1. 変更ファイル数と diff

| 項目 | 内容 |
| --- | --- |
| 変更 production ファイル数 | **1**（`src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`） |
| 変更関数 | `releaseResources()` 内 VerifyDrained 破壊 block のみ |
| 実質 diff | (a) `if (false && ...)` 2 行を解除し `destroyRolledBackDSP` → `retire` に差し替え、(b) fading 側に `!= activeDSPToDestroy` の二重 retire 防止ガード追加、(c) DIAG を `[D162-2G2_VD_RETIRE] target=active-final/fading-final` に変更、(d) D0/F 経緯込みの契約コメントに更新 |
| 触れていないもの | Orchestrator（G1 S3 維持）/ destroyRolledBackDSP 関数本体（削除せず rollback 経路で残存）/ DSPLifetimeManager / EBR / tryShutdownQuiescentReclaim / handle registry / E-3 / S1-S4 / AudioSegmentBuffer / テスト |

## 2. 静的チェック（指示 §3・全 ✓）

- V-D active/fading とも `retire(...)` 化・`false &&` は **src 全走査で 0 件**（全ソースから消滅）
- S3 `shutdown-clear` retire 維持（Orchestrator.cpp:632）
- `destroyRolledBackDSP` 呼び出しは V-D で 0（関数自体は DSPLifetimeManager.cpp:149 に残存）
- map erase は authority 内（AudioEngine.h:4367）で実行されることを再確認

## 3. Gate ladder 結果

| Gate | 結果 |
| --- | --- |
| G2-1 Build（Debug/Release/RWDI） | 全 **EXIT=0** |
| G2-2 CTest | Debug **40/40**・Release **39/40**（AudioEngineHarness 0xC0000374 = G1/B/E/F と同一 pre-existing・新規失敗なし。Test #36 HeadlessAudioPathVerification は PASS） |
| G2-3 Debug 6-gen | **exit 0x00000000**・dump 0 |
| G2-4 RWDI 6-gen | **exit 0x00000000**・dump 0 |
| G2-5 RWDI 60-gen | **exit 0x00000000**・dump 0 |

## 4. V-D 固有観測（指示 §8 の分離報告）

| 項目 | Debug 6-gen | RWDI 6-gen | RWDI 60-gen |
| --- | ---: | ---: | ---: |
| V-D active target count | 1 | 1 | 1 |
| V-D fading target count | 0 | 0 | 0（final fading は crossfade 経路で先に消化・VerifyDrained 残存なし） |
| V-D retire(retired=1) count | **1** | **1** | **1** |
| V-D retire(retired=0) count | 0 | 0 | 0 |
| V-D EBR enqueue count | 1（Success） | 1（Success） | 1（Success） |
| V-D D117_DESTROY count | 1 | 1 | 1 |
| V-D footprint released count | 1（remaining=0） | 1（remaining=0） | 1（remaining=0） |
| **direct destroy count** | **0** | **0** | **0** |
| stale map entry | 0（**retireByHandle MISS 実測**） | 0 | 0 |

**V-D が retired=1 で動作するケースを 3/3 run で観測** — G2 の本質検証条件（§8 の
「no-op 経路を V-D と解釈しない」要求）を満たす。

追加観測: 60-gen の V-D 対象アドレスは address reuse を経た個体だったが、generation 検証付き
registry のもとで新 lifecycle を正しく resolve（retired=1）し旧 lifecycle と混同しなかった
（D162-2-C §10 の address reuse 懸念への authority 経路の健全性実証）。

## 5. 全体観測（60-gen）

| 指標 | 値 |
| --- | --- |
| E-4 会計 | CREATE 26,603 = CONSUME 26,558 + OVERWRITE 37 + CLEAR 7 + DISCARD 1（完全収支） |
| S3（G1 維持） | CLEAR_SHUTDOWN_DISPOSITION 7・全 disposition 対応 |
| destroy / released | **61 / 61**（residual 0・destroy 61 = 構築 60 + placeholder 1） |
| EBR pend / ovf | 最終 0 / 0（pend 最大 2 は運転中一時） |
| E-3（INV-D162-8） | 0 件 |
| Signature A / B / C | 0 / 0 / 0 |
| shutdown sequence complete | 1 |
| XRUN | 17（Callback ≤1.87ms・E baseline 17 と同水準 / F 15 — 新規クラスなし） |
| DC live / Priv | 1–3 収束 / **442MB**（F=445 / E=461 と同水準） |

最終 destroy は `~AudioEngine: enter` 後（dtor body D5/D8 drain 内）に実行 — INV-D162-8 適合。

## 6. crash dump 分類

dump 2 件（ConvoPeq.exe.25884 / AudioEngineHarness.exe.4228）は Release CTest 由来の
pre-existing（旧 build-icx binary static teardown AV / AudioEngineHarness 0xC0000374）。
G2 soak 3 run の dump は **0 件**。

## 7. 判定と次工程

**D162-2-G2 = PASS。** 現行 source は既に S3=ON / V-D=ON であるため、
**D162-2-G3（両経路同時の本命再証明）は production 変更 0 で Gate ladder（Build → CTest →
Debug 6-gen → RWDI 6-gen → RWDI 60-gen）のみを実行**する形になる。G3 の結果が出るまで
G4（60-gen × 2 final deterministic soak）には進まない。
