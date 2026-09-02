# D162-1 Work Report — Rebuild Generation 系メモリ滞留の原因監査（read-only）

- Work item: D162-1（D116-OP-2 TRIGGERED の原因監査）
- Date: 2026-09-01
- Mode: **完全 read-only**（production source / tests / CMakeLists / build 設定の変更 = 0。`git status src/` の差分は CR-α/ND 既存分のみで D162-1 追加 0）
- Authority: 現行 ConvoPeq.md（Generated 2026-09-01 21:47:45）。旧 work70 資料の行番号・仮説は authority としない
- Evidence: evidence/D162-1_RETENTION_CHAIN_EVIDENCE.md（E-1〜E-7）、evidence/D116_OP2_soak.log、evidence/D116_OP2_memory_soak.csv

---

## 0. 完了条件チェックリスト（10 項目）

| # | 条件 | 判定 | 参照 |
| --- | --- | --- | --- |
| 1 | 59 generation の retention chain をコード上で再構成 | **PASS** | §1 |
| 2 | obsolete / committed / published を分離 | **PASS** | §1.2 |
| 3 | 現行 DSPGuard direct-destroy path を検証 | **PASS** | §2 |
| 4 | DSPCore allocation/destruction site を全件照合 | **PASS** | §2.1 |
| 5 | retire → router → reclaim chain を照合 | **PASS** | §2.2 |
| 6 | generation-scoped allocation 候補を分類 | **PASS** | §3 |
| 7 | diagnostic build で取得すべき counter を確定 | **PASS** | §4 |
| 8 | H1〜H4 を FACT / HYPOTHESIS として整理 | **PASS** | §5 |
| 9 | 修正対象をまだ決定しない | **PASS**（修正判断は D162-2/1R 分岐でユーザー指示） | §6 |
| 10 | production source 変更 = 0 | **PASS** | §7 |

---

## 1. Phase 1 — Generation lifecycle 完全追跡

### 1.1 chain

59 generations すべて（soak で REBUILD_REQUESTED 119 → dispatch 59 → build 59）が以下を通過:

```
RuntimeBuilder.cpp:425  DSPCore allocated (aligned_make_unique)
AudioEngine.RebuildDispatch.cpp:1358-1390  dspGuard.ptr → dspToCommit → enqueuePublicationIntentForRuntimeCommit
AudioEngine.Commit.cpp:804  registerDSPHandleForRuntime(newDSP)   ← 全 59 件 handle 登録済み
Commit.cpp  submitPublishRequest(req)                              ← submit 後に DSP 処分コードなし
PublicationAdmission.cpp:10-60  Admission evaluate
PublicationExecutor.cpp:55-95  [PUBLISH]（成功 10 件のみ: gen 6,9,…,33）
```

- publish は **10 / 59**（every-3rd-generation、間隔 ≈36 s = crossfade 窓 ≒ fade 完了周期）。
- 非 publish の **49 generations の最終 disposition は soak ログ上観測不能**（CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF のため Deferred/Rejected/discard telemetry が compile-out。`build/CMakeCache.txt` で BOOL=OFF 確認）。

### 1.2 成功 / obsolete / committed の分離

| 分類 | 件数 | 処分経路 | 判定 |
| --- | --- | --- | --- |
| published | 10 | 正規 retire pipeline（DSPLifetimeManager → RetireRouter → reclaim） | 滞留の直接原因ではない |
| obsolete（isObsolete→continue） | 0 件の痕跡／guard dtor が救済 | RebuildDispatch.cpp:1208-1221 → guard dtor direct-destroy | リークなし（コード上） |
| committed 未 publish | 49 | **経路未確定** — Deferred(overwrite 消化 or 残存 or discard) or Rejected\*。discard/Rejected は DSP 処分なし（§1.3） | **構造的リーク候補・実行時 UNPROVEN** |

### 1.3 処分の非対称（構造的中心事実）

- **Rejected\*** 全 5 case（Orchestrator.cpp:380-436）: telemetry + obligation API のみで **DSP 処分なし**。
- **Deferred → discard**（:685-691）: `view->discard()` は reason 記録 + finishView のみで **DSP 処分なし**。
- **Deferred → overwrite**（:441-534, :461-468）: 旧 slot DSP のみ retire — 非 publish 経路で**唯一**の処分点。
- commit 側（Commit.cpp:787-870）は submit 後に処分しない → **admission が Rejected/discard を返した DSPCore は handle 登録のまま誰にも retire されない**。

---

## 2. Phase 2 — DSPGuard direct-destroy path 検証

1. **dtor fallback 実装確認**: RebuildDispatch.cpp:932-964 — `retireDSPHandleForRuntime == false`（未登録）時 `destroyDSPCoreNode` 直接触破（work70-FIX(rebuild-obsolete) 現行実装）。旧 work70 の「handle map 未登録 → EBR 未投入リーク」仮説は現行コードでは成立しない。
2. **dspGuard.ptr の全代入/release 点照合**: :1018 / :1037 / :1053 / :1106 / :1119 / :1133 / :1208 / :1369（:1358-1390 で `dspToCommit` に譲渡後 `nullptr` 化）。未解放経路なし。
3. **warmup retry 経路**: Schedule → `retryScheduler_->schedule()` + continue — ptr 設定済みのまま dtor が retire/destroy。リークなし。
4. **obsolete 経路**: :1213-1221 isObsolete → continue → guard dtor。リークなし。
5. **recovery lease 破棄**: :1036 / :1118 で `destroyDSPCoreNode(dspGuard.ptr)`。二重解放なし（ptr nullptr 化とセット）。
6. **destroyDSPCoreNode 呼び出し元 7 箇所照合**: Threading.cpp:17-25（~DSPCore + aligned_free）の全 caller が guard dtor / recovery lease / DSPLifetimeManager enqueue コールバック / destroyRolledBackDSP / quarantine に収束。浮遊 caller なし。

### 2.1 allocation site 全件

- `RuntimeBuilder.cpp:425`（generation ごとの DSPCore 本体）
- `AudioEngine.Processing.PrepareToPlay.cpp:234-270`（placeholder DSP・初回 IR 前のみ）

### 2.2 retire → router → reclaim chain

`DSPLifetimeManager.cpp:40-160`: retire()（**handle 未登録なら early return**）→ enqueueWithRetry(dsp, &destroyDSPCoreNode, epoch) → RetireRouter（overflow ring / quarantine なし流れ）→ reclaim → destroyDSPCoreNode。quarantineSlot（Threading.cpp:40-71）も retireDSPHandleForRuntime へ委譲。Shutdown audit 実測: routerPendingRetire=1 / quarantine=0 → **backlog 滞留の兆候なし（H2 不支持の実測）**。

---

## 3. Phase 3 — 129.8 MB/generation の allocation 分類

### 3.1 実測

private +7,658.5 MB / 422 s ≒ **129.8 MB/gen**。WS 7,403.7 MB が並行追従 → **実 resident 滞留**（OS lazy release / 仮想 commit 幻影ではない）。

### 3.2 NUC 入力長の確定

rebuild 経路 `loadedSR = 192000`（LoaderThread.cpp:428）→ `computeTargetIRLength(192000, 76800)` = `192000 × 1.0 s`（CONV_STATUS irLen=192000 から逆算・cap MAX_IR_LATENCY=2,097,152 非拘束）→ **NUC 入力 192,000 samples**。

### 3.3 NUC 構造計算（part0=2048, tailStart=4,080, mult=8）

L0 0.25 MB + L1 12.50 MB / ch → **25.5 MB / DSPCore**（irFreq+fdl+scratch ×2ch）。

### 3.4 per-DSPCore 構造推定と残差

| 構成要素 | 推定 |
| --- | --- |
| NUC ×2ch | 25.5 MB |
| irData (double 192000×2ch) | 2.9 MB |
| EQ（384 kHz / block 8192・scratch 65536） | 1-2 MB |
| oversampling（osFactor 2） | 1-3 MB |
| その他（TP detector / ramp / misc） | < 1 MB |
| **合計** | **≈ 33 MB / DSPCore** |

**残差 ≈ 97 MB/gen が DSPCore 単体では説明不能**。分類表:

| 候補 | 内容 | 起源が DSPCore か | 判別手段 |
| --- | --- | --- | --- |
| (a) NUC 実サイズ過小推定 | MKL DFTI descriptor/work buffer が diag allocator 追跡外の可能性 | DSPCore 内 | [MEM_SNAP] NUC alloc MB |
| (b) generation-scoped 非 DSPCore 割当 | RuntimePublishWorld / engine 側 IR AudioBuffer 群 / AUTO_GAIN / EQ・OS 大型 workspace | DSPCore 外 | [MEM_SNAP] Other MB（= Priv − NUC − retire） |
| (c) retained DSPCore 複数化 | H1 経路で複数 DSPCore/generation 生存 | DSPCore | [MEM_SNAP] DC live / SC live |
| (d) allocator arena 滞留 | JUCE/aligned arena の freed-held | どちらでも | DC/NUC live flat かつ Priv 増加 |

結論: **Phase 4 診断ビルド実測なしに残差の帰属は確定できない**（先に結論を置かない旨の指示遵守）。

---

## 4. Phase 4 — 診断ビルド実測設計（変更 0 で実施可能な部分を明確化）

### 4.1 既存診断器（CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON で出力・ソース変更不要）

| 要求 counter（指示最低限項目） | 既存取得手段 |
| --- | --- |
| timestamp | [Seq=] diag sequence + [MEM] 1 s 周期（Timer.cpp:1280-1308） |
| generation | diagPrefix(gen) / [MEM_SNAP] PUBLISH gen= |
| DSPCore::liveCount | [MEM_SNAP] `DC: live=`（Timer.cpp:1060,1072） |
| StereoConvolver::liveCount | [MEM_SNAP] `SC: live=` |
| NUC live / alloc bytes | [MEM_SNAP] `NUC: live= alloc=MB peak= tA= tF= lost= zero=`（convo::diag 追跡） |
| pendingRetireCount / tracked / bytes | [MEM_SNAP] `Ret: pend= trBytes= tr= ovf=` |
| reclaimCount | [MEM_SNAP] `rec=` |
| runtimeRetireCount | [MEM_SNAP] `gen=`（retiringGeneration・DSPLifetimeManager 唯一 Authority） |
| published generation | [PUBLISH] gen=（常時） |
| quarantineResident | shutdown 時 collectDrainAudit（Threading.cpp:74-110）のみ — 周期サンプルなし |
| DSP pointer register/retire/destroy trace | [D117_RETIRE] dsp=%p retired/enqueue/epoch（DSPLifetimeManager.cpp:46/63）+ [D117_DESTROY]（Threading.cpp:17-25）。**register 側のログは未実装** |
| Non-NUC 残差分離 | [MEM_SNAP] `Other=MB`（= Priv − NUC − retire） |
| active DSP 内訳 | [MEM_SNAP] `TRK: total/OS/EQ/AL/LT`（active DSP のみ） |

### 4.2 gap（診断ビルドでも未計測 — 将来の最小追加候補・本監査では実装しない）

1. `runtimeDSPHandleMap_` size の周期ログ（AudioEngine.h:4307-4346・size accessor なし）
2. generation ごとの admission 分類累積（Accepted / DeferredFadingActive / Rejected\* / discard reason — [PUBLISH] は成功のみ）
3. discard reason 累積 counter（slot の lastDiscardReason は最終値のみ）
4. retire 時点の per-DSP TrackedMemoryStatistics（TRK は active DSP のみサンプリング → retiring DSP の内訳が観測できない）
5. obsolete generation count（guard dtor fallback が silent）

### 4.3 実施手順（設計）

1. `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON` で同一 CLI soak 条件（`--cli-ir-reload-count 60 --cli-ir-reload-interval-ms 6000 … --cli-exit-ms 420000`）を Release で再実行
2. [MEM_SNAP] 時系列から ΔPriv / ΔNUC / ΔOther / ΔDC-live / ΔSC-live / ΔRet-pend を generation 境界（[CONV_REBUILD] 59 境界）ごとに差分分解
3. [D117_RETIRE]/[D117_DESTROY] の pointer 集合差分から generation ごとの register→retire→destroy の有無を pointer 単位で判定
4. ΔDC-live ≈ +1/gen なら H1 確定、ΔOther ≈ +97 MB/gen なら H3 確定、ΔNUC ≈ +129 MB/gen なら (a) 確定

---

## 5. Phase 5 — H1〜H4 分類（FACT / OBSERVED / INFERRED / UNPROVEN）

| 仮説 | 内容 | 分類 | 根拠 |
| --- | --- | --- | --- |
| **H1** DSPCore destruction 漏れ（discard / Rejected\* 経路の非処分） | admission Rejected / deferred-discard を受けた generation の DSPCore が handle 登録のまま retire されない | 構造 = **FACT（コード上）** / 実行時 = **UNPROVEN** | §1.3 の非対称は現行コードで確定。ただし soak は diagnostics OFF で 49 件の経路不明。旧 work70 の handle-map 仮説とは別物（direct-destroy fallback は現行実装済み） |
| **H2** Retire-EBR backlog 滞留 | RetireRouter pending / epoch gap で DSP 回収遅延 | **OBSERVED で不支持**（弱） | shutdown audit: routerPendingRetire=1 / quarantine=0。ただし in-run 時系列は diagnostics OFF で未観測 → 完全否定は Phase 4 後 |
| **H3** DSPCore 以外の generation-scoped allocation | ~97 MB/gen の残差の帰属先 | **INFERRED**（実測から強く示唆・割当先未確定） | 129.8 MB/gen 実測 vs 33 MB/gen 構造推定。候補 (a)-(d) は §3.4。**H1 が仮に全件成立しても最大 49×33 ≈ 1.6 GB ≈ 21 % で、H3 なしには 7.7 GB を説明できない** |
| **H4** allocator-OS retention | freed 領域の OS 返還遅延 | **OBSERVED で不支持**（主因として） | WS が Private に並行追従（7,403 MB）= 実 resident 増加。freed-every-gen の arena は platform しない（線形増加と矛盾）。二次的要因の可能性は残す |

**統合見解（結論の先出しではなく分類）**: 滞留は (i) 非 publish generation の DSPCore がどこまで実際に retain されるか（H1 実行時証明）と (ii) 残差 ~97 MB/gen の帰属（H3 確定）の 2 問題に分解される。両者とも既存 [MEM_SNAP] 診断ビルドで discriminate 可能（§4.3 判定ルール）。

---

## 6. Phase 6 — 修正禁止の確認

- 本監査で決定した修正対象: **なし**。「メモリが増えたので destroyDSPCoreNode() を追加する」類の修正先行は行っていない。
- discard / Rejected 経路への retire 追加・slot 仕様変更・guard 変更はすべて D162-2（confirmed 後）の設計 gate に委ねる。
- D162-1R 分岐（H3 帰属未確定の場合）は §4.2 gap 項目の最小 instrumentation 設計から始める。

## 7. 変更確認

- production source / tests / CMakeLists: 変更 0（`git status src/` = CR-α/ND 既存差分のみ・D162-1 追加 0 行）
- 新規作成: 本報告書 + evidence/D162-1_RETENTION_CHAIN_EVIDENCE.md のみ

## 8. 分岐提案（ユーザー指示待ち）

- **H1/H3 が実行時 confirmed になった場合** → D162-2（修正設計 gate）へ
- **diagnostics ビルドでも帰属が確定しない場合** → D162-1R（§4.2 gap の最小 instrumentation → 再測定）へ
- 本監査単体では D162-2 起票の条件（実行時 confirmed）は未充足 — §4.3 の診断ビルド実測が次の最短ステップ
