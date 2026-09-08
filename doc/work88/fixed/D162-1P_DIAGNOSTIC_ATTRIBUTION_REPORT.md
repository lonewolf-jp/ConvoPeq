# D162-1P Work Report — Diagnostic Attribution Soak（診断 ON 実測帰属）

- Work item: D162-1P（D162-1 Phase 4 の実行フェーズ・診断ビルド同一条件再測定）
- Date: 2026-09-02
- Mode: production source 変更 **0**（src/ / tests/ / CMakeLists.txt いずれも無変更。新規 build dir `build-diag`・evidence 配下の bat/ps1 のみ追加）
- Evidence: evidence/D162-1P_soak.log（91,105 行）/ D162-1P_memory.csv（212 サンプル）/ D162-1P_asan_stderr.txt（ASAN 解析）/ D162-1P_build*.log

---

## 0. 実行条件と逸脱（正直な記録）

| 項目 | OP-2-A baseline | D162-1P | 逸脱評価 |
| --- | --- | --- | --- |
| CLI 条件 | `--cli-ir-reload-count 60 --cli-ir-reload-interval-ms 6000 --cli-intent-burst-count 60 --cli-intent-burst-interval-ms 6000 --cli-exit-ms 420000` | **同一** | なし |
| IR / SR / block / OS | 76800 src @192kHz / 1024 / 2x | 同一（[CONV_STATUS] irLen=192000 sr=192000 osFactor=2 processingRate=384000） | なし |
| binary | build/ Release・DIAG=OFF | **build-diag RelWithDebInfo・DIAG=ON** | **あり（下記理由）** |
| 結果 | private 7,740 MB / 426 s | private 7,746 MB / 434 s | 再現性确认 |

**逸脱理由**: Release + `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON` のバイナリは起動早期（[EQ_CTOR] exit 後・最初の REBUILD_REQUESTED 前）に **0xC0000374 (heap corruption, ntdll) で 3/3 決定論的にクラッシュ**（§4.2）。Debug DIAG は完走、ASAN Release DIAG も完走（= 最適化構成依存）。RelWithDebInfo（/O2 /Ob1）DIAG は完走するため本測定に採用。メモリ増加率は OP-2-A と同オーダー（+1,215 vs +1,088 MB/min）で比較可能性を確認。

---

## 1. 主要実測（59+1 generation の帰属）

### 1.1 カウント対応表（ユーザー指定の最低限表）

| 項目 | 60 gen 実測 | 判定 |
| --- | ---: | --- |
| generation | 60（gen 5..64・[CONV_REBUILD]×60・D133 enqueue×60） | FACT |
| published | 11（gen 5,7,10,…,34・every-3rd・約36s間隔） | FACT |
| non-published | 49 | FACT |
| DSPCore live 増加 | 1 → **50**（+49 net・MEM_SNAP DC live） | **H1 判定材料** |
| StereoConvolver live | 0 → **50**（DC live と 1:1） | H1 整合 |
| NUC live | 0 → **100**（= 2 × SC live） | H1 整合 |
| D117 destroy | **11 件のみ**（placeholder 1 + 代替された旧 published 10） | **H1 判定材料** |
| retire（D117_RETIRE / RETIRE_BY_HANDLE） | 11 dsp・全て destroy と対成立・**retire なし destroy なしの DSP = 49** | **H1 判定材料** |
| RetireRouter pending | 最大 1・終端 0・**ovf=0・quarantine=0**・reclaim 12,982,510 単調増加 | **H2 判定材料** |
| handle map residual | 未計測（size accessor なし・既知 gap） | — |

**算術的閉包**: 総 DSPCore = placeholder 1 + 60 generation = **61**。destroy **11**。終端 live **50**。
61 − 11 = 50 ✓。すなわち **49 件の非 publish generation の DSPCore は全件、retire も destroy もされず生きたまま残存**。MEM_SNAP の live カウンタ・D117 pointer trace・世代算術の 3 者が完全一致。

### 1.2 pointer 単位の照合（H1 実行時実証）

- 11 件の destroy はすべて **published chain** のみ: gen5 publish 時に旧 published（placeholder）を retire → destroy、以降 gen7 publish で gen5 の DSP、gen10 publish で gen7 の DSP… と「publish 1 件につき直前 published 1 件を retire」。`[D117_RETIRE_BY_HANDLE] lookup=HIT` + `[D117_DESTROY]` が 11 組対成立、retire-only / destroy-only の浮遊 pointer は 0。
- **非 publish generation には retire も destroy も一切発生していない**（destroy 11 = 61 − 50 と一致し、消えた DSP が別経路で解放された可能性が算術的に排除される）。
- publish は gen34 で停止（gen35..64 の 30 世代は全件非 publish・全件残存）。

### 1.3 per-retained-DSP フットプリント（世代境界差分）

| 指標 | per retained DSPCore（MEM_SNAP 世代境界差分・56 セグメント） |
| --- | --- |
| diag tracked aligned alloc（`NUC alloc` カラム） | **+123 MB** |
| Private | **+141 MB**（範囲 103–145） |
| WorkingSet | **+~130 MB**（WS が追従 = 実 resident） |
| placeholder DSP（IR なし・比較基準） | +5.5 MB |

- CSV 全区間傾き **+1,214.9 MB/min**（active window 線形）→ **141.7 MB/gen**（7s/gen）と MEM_SNAP 差分が一致。
- destroy 時の private 落ちは −15〜−35 MB にとどまる（構築時 +123 MB に対し）→ **解放済みメモリの arena 滞留が実在**（H4 は二次寄与として実在するが主因ではない）。

---

## 2. H1〜H4 再分類（最終）

| 仮説 | 判定 | 根拠 |
| --- | --- | --- |
| **H1** DSPCore destruction 漏れ | **CONFIRMED**（実行時・pointer 算術込み） | §1.1/§1.2。49/60 generation の DSPCore が retire/destroy されず残存。構造的根拠（D162-1: discard/Rejected 非対称）と実行時結果が一致 |
| **H2** Retire-EBR backlog 滞留 | **REFUTED** | pend 最大 1・ovf=0・quarantine=0・reclaim 単調増加で正常 drain。そもそも非 publish DSP は router に一度も入っていない |
| **H3** DSPCore 外の generation-scoped allocation | **REFUTED**（主因として） | private 増加が DC live 増加と 1:1 で追従（+141 MB × 50 live ≒ 7,450 MB 全額説明）。DSPCore 外の独立増加分は検出されず。**再枠組み**: 未帰属分は「retained DSPCore の内部」にある（§3） |
| **H4** allocator-OS retention | **REFUTED**（主因）・**実在（二次）** | WS が private に追従（実 resident 増加）。一方 destroy 時の priv 落ちが DSP サイズより大幅に小さい = freed-held arena は実在するが、線形増加の主因は live retained DSP |

---

## 3. 残存未帰属分（D162-2 前に要確認）

per retained DSPCore ≈ 123–141 MB の内訳（現時点で特定できている分）:

| 構成要素 | 推定 | 根拠 |
| --- | --- | --- |
| latency buffers ×4（kMaxLatencySamples=1,536,000・384kHz 2s cap → 769,026 samples ×8B ×4） | **23.5 MB** | PrepareToPlay.cpp:177-196 |
| NUC ×2ch（SoA モデル・part0=1024・irLen=192000・mult=8） | **26.8 MB** | MKLNonUniformConvolver.cpp:805-870 の静積分 |
| irData 2ch + impulseForFft 2ch | **5.8 MB** | StereoConvolver / SetImpulse |
| EQ scratch（65536×8B）+ DSPCore aligned（8192） | **0.6 MB** | [EQ_PREPARE]/[DSPCORE_PREPARE] |
| **特定済合計** | **≈ 57 MB** | |
| **未帰属** | **≈ 66–84 MB / DSP** | NUC 実割当が静的 SoA モデルの約 3.5–4 倍の可能性（L1 numParts を capacity 予約した場合 ≈+28 MB/ch で部分説明）が最有力候補。確定には per-allocation-site サイズログが要 |

**計測器の制約（本測定で判明した新規知見）**:
- `NUC alloc` カラム（`convo::diag::allocatedBytes()`）は **累積確保量**（`aligned_free` 経由の解放が counter を減らさない。実測 tF=0GB のまま DSP 11 件 destroy されたことで証明）→ live bytes ではない。
- `Other=MB` は常に 0（tracked 値が Priv を上回るため clamp）。帰属手段として無効。
- `TRK`（TrackedMemoryStatistics）は常時 **0.0**（active DSP でも計測未配線）。

---

## 4. 新規発見（TRIGGERED 候補・本 work item では修正しない）

### 4.1 Teardown UAF（全ビルド共通・ASAN で実証）

```
AudioEngine::~AudioEngine (CtorDtor.cpp:288) が ISRRetireRouter を破壊
  → 後続のメンバ破壊で EQCacheManager::~EQCacheManager → CacheMap::~CacheMap
    → tryShutdownQuiescentReclaim (AudioEngine.h:4428) が解放済み router を read
```
- ASAN: heap-use-after-free（read 8B・freed by ~ISRRetireRouter・同 dtor チェーン内）
- 本日 OFF ベースライン Release も終了時 0xC0000005 で同一経路と整合（D116-OP-2 当時は APP_EXIT=0 で非発現 = タイミング依存）
- 影響: 終了時のみ・本監査の retention 数値には影響しない

### 4.2 Release + DIAG ON の起動時 heap corruption

- 0xC0000374（ntdll）・3/3 決定論的再現・bootstrap log は [EQ_CTOR] exit で停止
- Debug DIAG / RelWithDebInfo DIAG / Release DIAG+ASAN は完走 → 最適化構成依存
- 現行ツリーで DIAG=ON Release が未検証だったことが初めて判明（D135-2 時代の diag build は旧ツリー）

### 4.3 診断器の計測限界（将来の instrumentation 要件）

1. `aligned_free` が diag counter を通らない（live bytes 計測不能）
2. TRK（TrackedMemoryStatistics）未配線（常時 0）
3. `runtimeDSPHandleMap_` size 未ログ
4. admission 分類（Deferred/Rejected/discard）の累積ログは DIAG ON でも出力されない（本 soak で DEFERRED/Rejected 行 = 0 を確認）

---

## 5. D162-2 起票条件の判定（ユーザー基準適用）

| 条件 | 判定 |
| --- | --- |
| H1: non-publish gen → DSPCore retained → retire なし / destroy なし（pointer 単位実証） | **充足**（§1.2） |
| H3: ~97 MB/gen の帰属先が再現性をもって特定 | **未充足**（DSPCore 外説は否定。残り ~70 MB/DSP が DSPCore 内部・未帰属） |

→ ユーザー基準「H1 単独では D162-2 直行しない」に従い、**D162-2 は起票条件未充足**。推奨次工程は **D162-1R（最小 instrumentation による per-site 帰属測定）**: ① NUC SetImpulse の allocSizes 集計ログ ② aligned_free の diag counter 配線（または free-size 渡し）③ TRK 配線 ④ destroy 時の per-DSP footprint ログ。これで retained DSP 123–141 MB の全額帰属 → D162-2 の修正設計対象が確定する。

---

## 6. production source 変更確認

- src/ / tests/ / CMakeLists.txt: 変更 0（`git status` = CR-α/ND 既存差分のみ）
- 追加物: build-diag/・build-diag-asan/（ビルド生成物）+ evidence 配下の bat/ps1/log のみ
