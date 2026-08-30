# D117 — IR Reload × Rebuild Memory Retention Root-Cause Audit

**Date:** 2026-08-29
**対象:** D116-6 で検出された ~150MB/publish のメモリ滞留の原因監査（**修正実装なし・commit 凍結継続**）
**Binary:** MSVC Release 診断ビルド（`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON`、`build\`、head a65ace1 + 観測専用トレース追加）
**重要:** icx crash（0xc0000005 ×3）は別トラックのまま。本監査は MSVC のみ。

---

## D117-0 — 実行バイナリと source revision の固定

| 項目 | 値 |
| --- | --- |
| Git HEAD | `a65ace1df9b2012a12fa6b656a91f33ad4da9000`（ユーザーにより D113-A 含め commit 済み。src/ は D116 検証バイナリと同一ソース） |
| D116 baseline binary | `build\ConvoPeq_artefacts\Release\ConvoPeq.exe`（47,778,304 bytes, Aug 29 00:52, md5 23a5f2eca84a8bb728cc23426282eae9） |
| D117 観測 binary | 同パス、診断ビルド（Aug 29 08:0x）。**マクロ OFF の production 構成には影響なし**（全トレースは `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` ゲート） |
| 観測専用追加（D117 GO 範囲） | `DSPLifetimeManager.cpp`（retire/retireByHandle トレース）、`AudioEngine.Threading.cpp`（destroyDSPCoreNode トレース）— logging のみ・semantics 不変 |

**ビルド手順メモ**: `build.bat` の `-D` 引数は cmd の引数解析で `=` が分割され、さらに cmake のコンパイラ変更時キャッシュ再構築で `-D` が消失するため、`evidence/D117_diag_build.bat`（2回 configure + キャッシュ検証）を使用。

---

## D117-2/3 — 診断ビルドによる観測（10 pairs reload+rebuild / burst only / trace）

### D117-1「145 MB/pair」の正体 = **DSPCore 1オブジェクト（StereoConvolver + MKLNonUniformConvolver×2 を含む）**

MEM_SNAP 時系列（`evidence/D117_reload_rebuild.log`、reload+burst 10 pairs）:

| 時点 | NUC live | NUC alloc | DC live | SC live | pend | quarantine | Priv |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 初期 | 0 | 455MB | 2 | 0 | 1 | 0 | 505MB |
| +3 pair | 6 | 1147MB | 5 | 3 | 0 | 0 | 975MB |
| +6 pair | 12 | 1838MB | 8 | 6 | 0 | 0 | 1427MB |
| +9 pair（最終） | **18** | 2529MB | **11** | **9** | 0 | 0 | 1881MB |

- **ΔDC = ΔSC = +1/publish、ΔNUC = +2/publish（stereo L/R）** — 9 publishes で正確に 9 セット増加
- **NUC tA=2GB / tF=0.00GB — tracked 解放が一度も発生していない**
- pend=0（retire queue 空き）・quarantine=0（[D101_9_T5_OBS] Q_resident=0 / E_resident=0 / T_resident=0）・ovf=0
- IR_LOAD=18 / IR_RELEASE=18 は **同一 NUC の layers 差し替え**（SetImpulse 時の旧 layer 解放）であり、オブジェクト破棄ではない

### D117-9 分類: **Case 3 — 「retire pipeline に入っていない object が存在」**

Case 1/2/4 は棄却:
- Case 1（遅い/backlog）→ pend 常に 0、queue 滞留なし
- Case 2（quarantine 滞留）→ quarantine resident 全て 0
- Case 4（DSP lifetime 以外）→ DC/SC/NUC live が毎 publish 正確に +1/+1/+2 で増加（ Private Δ≈153MB/publish と一致）

### D117-6 順序依存性試験の結果 — **仮説の訂正**

burst only（reload 無し）10 intents を診断ビルドで実行（`evidence/D117_burst_only.log`）:
**DC 2→11、NUC 0→18、SC 0→9、Priv 505→1880MB — reload なしでも全く同じ蓄積。**

→ **D116-6 の「reload+rebuild の組合せでのみ増加」という結論は訂正される**。正しくは:
- **すべての publish で ~150MB が漏出する**（reload の有無は無関係）
- D116-6 の隔離試験の誤り原因: (a) burst-only 実行のメモリランプを稀なサンプリング（tail 8サンプル）で見落とした、(b) CSV の上書き（デフォルト出力パス衝突）、(c) reload-only は publish 1回のみで漏出 1 回分は視認不可だった
- 長時間試験（6分・59 publishes）の線形増加（+1.45GB/min）は本質を正しく捉えていた

---

## D117-4/5 — 一本道追跡と DSPGuard（実測）

`retirePublishedDSP` / `DSPLifetimeManager::retire` / `retireByHandle` / `destroyDSPCoreNode` に観測専用トレースを追加し、reload+burst 6 publishes を実行（`evidence/D117_lifetime_trace.log`）:

**`[D117_RETIRE]` / `[D117_RETIRE_BY_HANDLE]` / `[D117_DESTROY]` が 6 publishes 中「0 回」出力。**

- **破壊チェーンの第一リンク（retire 呼び出し）が実行時に一度も発火していない**
- DSPGuard（RebuildDispatch.cpp:882）について: retire 失敗時の `destroyDSPCoreNode` 直接呼出しフォールバックは現行ソースに存在する（過去の「未登録→リーク」仮説は不成立）。ただし本監査では、guard は rebuild-obsolete（未 publish）DSPCore 専用であり、今回の漏出経路（publish 済み DSPCore）とは無関係 — guard 自体は今回 0 回発火
- destroyDSPCoreNode の実装は正常（~DSPCore → aligned_free）。呼ばれないことが問題

## CONFIRMED CAUSE — fadeCompleted ブロック内のシーケンス欠陥

`AudioEngine.Timer.cpp` fadeCompleted ブロック（929-1002 行）:

```cpp
while (crossfadeRuntime_.consumeCompletedFade(ev))
{
    dspHandleRuntime_.endCrossfade(ev.id);   // (1) fadingRuntimeDSPHandle_ を null にする
    ...                                       //     （旧 slot は Retired 状態へ）
}
// ★ B-1: CAS-based fading slot clear
{
    DSPCore* current = ...fadingRuntimeDSPSlot...;   // (2) slot CAS は成功
    if (... CAS success ...)
    {
        const auto fadingHandle = dspHandleRuntime_.getFadingRuntimeDSPHandle();  // (3) ← すでに null
        if (!fadingHandle.isNull())
            runtimePublicationBridge_.submitObserve(fadingHandle, ...);            // (4) ← 実行されない
    }
}
```

- `endCrossfade()`（ISRDSPHandle.cpp:103-119）は `fadingRuntimeDSPHandle_` を **null に publish** する
- 直後の (3) は常に null を読み、(4) の `submitObserve` が **常にスキップ**される
- → Coordinator の Observe → `retireByHandle` → `enqueueWithRetry(destroyDSPCoreNode)` が **一切呼ばれない**
- 同一構造の `DSPTransition::onTransitionComplete` と `AudioEngine.Timer.cpp:1102-1106`（!isFading パス）も同一時点の null 読みのため全経路が漏出

**補助因（セーフティネットの不在）**: crossfade 開始時 `storeReceipt(fadingHandle, epoch)` で handle は保存されており、receipt ベースで retire する `retirePublishedDSP`（Timer.cpp:1868）が存在するが、**呼び出し元ゼロの dead code**（"Timer CAS retire パスで呼ばれる" / "Coordinator Loop で取り出して retirePublishedDSP を実行する" という契約コメントに反し、どちらの経路にも未接線）。このハンドル競合レースに対する保険が未配線だった。

**なぜ CTest 40/40 が PASS するか**: テストは `DSPLifetimeManager::retire/retireByHandle` を unit レベルで直接駆動するため、Timer の fadeCompleted 配線欠落を検出できない（実 app の結合経路のみで発現）。

---

## FACT（実測された事実）

1. publish 1 回毎に DSPCore×1 + StereoConvolver×1 + MKLNonUniformConvolver×2 が生成され、破棄されない（Δlive と ΔPriv≈153MB が 9/6/10 publishes の全試行で一致）
2. retire queue（pend）・quarantine（Q/E/T resident）・overflow は常に 0 — retire pipeline に **入ってすらいない**
3. `endCrossfade` が fading handle を null 化した後で observe 提出が読むため、`submitObserve` が毎回スキップされる（静的解析 + トレース 0 回で確定）
4. `retirePublishedDSP` は呼び出し元ゼロ（dead code）
5. NUC の IR_RELEASE は同一オブジェクトの layers 差し替えであり、tF（tracked freed）は終始 0
6. restart サイクルで跨ぎ残留なし（プロセス終了で全解放）— shutdown 経路は正常
7. 診断ビルドの CTest 全体実行で test 21 が 1 回 SEGFAULT（単独実行 ×2 は PASS、非決定的・要調査）

## HYPOTHESIS（検証中に棄却/修正されたもの）

| 仮説 | 判定 |
| --- | --- |
| reload+rebuild の組合せでのみ増加（D116-6 の結論） | **棄却** — burst-only でも同一蓄積（サンプリング誤読） |
| 旧「未登録 DSPCore → retire false → 何もしない → leak」（DSPGuard） | **不成立** — guard に fallback 実装済み・且つ本経路未発火 |
| quarantine / deferred queue 滞留 | **棄却** — 全 resident カウンタ 0 |
| MKL/IPP/allocator retention（Case 4） | **棄却** — live object 数の増加が主因 |
| transferIRStateFrom の意図しない複製（D117-7） | **棄却（本件との因果）** — updateIRState は copy だが新 DSPCore 側の正常割当。漏出は旧側の未破棄 |

## CONFIRMED CAUSE

**fadeCompleted ブロック内の観測順序欠陥**: `endCrossfade()` による `fadingRuntimeDSPHandle_` の null 化が、Timer による observe 提出より先行するため、fading DSPCore の retire intent が 100% 未送出になる。加えて receipt ベースの retire 経路（retirePublishedDSP）が未配線のため、競合に対するフォールバックが存在しない。結果、**publish 済み DSPCore（+SC+2×NUC ≈150MB）がアプリ稼働中に一切破棄されない**。

## UNRESOLVED（次監査へ）

1. 診断ビルドで test 21 が全体実行時に 1 回のみ SEGFAULT（単独実行は PASS）— 診断マクロ由来のテスト分離問題の可能性。**修正は未実施**
2. `RuntimeWorld` 自体も `[WORLD] RetireQueue=1` として滞留が見える — world 側の回収経路の詳細（world は DSPCore とは別管理）は未深耕
3. 修正方針の決定（endCrossfade の handle null 化を observe 提出後に遅延するか、receipt ベース retire を配線するか、両方か）は **I4/D2 契約（Authority・DELETE-1/2/3 順序）との整合監査が前提** — 本監査では実装しない
4. icx 0xc0000005（別トラック）— 本件の destroy 未配線とは無関係と考えられるが、icx ビルドにも同一レイアウトが存在するため、修正後に再評価

## 生成物

- `evidence/D117_diag_build.bat` / `D117_DIAG_build.log` — 診断ビルド手順とログ
- `evidence/D117_reload_rebuild.log` — 10 pairs（MEM_SNAP 887 / IR_LOAD 18 / IR_RELEASE 18）
- `evidence/D117_burst_only.log` — burst only 差分試験
- `evidence/D117_lifetime_trace.log` — retire チェーントレース（0 発火の証明）
- `evidence/D117_memory.csv` — プロセスメモリ時系列
- `evidence/D117_ctest.log` — 診断ビルド CTest（39/40 + test 21 非決定 SEGFAULT、単独実行 PASS）
- 本ファイル

**commit は引き続き凍結。修正実装は次段階（D118: 契約整合を含む修正設計監査）での判断事項。**
