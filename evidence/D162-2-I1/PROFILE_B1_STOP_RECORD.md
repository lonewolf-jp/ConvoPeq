# D162-2-I1-B — Profile B-1 Record / STOP 発生（residual ≠ 0）

```text
Date:     2026-09-04
Type:     Profile B-1 ×1（production 変更 0 / test 0 / build 0）
Baseline: ConvoPeq.md Generated 2026-09-04 20:28:50・binary = G4-1 と同一 RWDI（21:16 build）
条件:     --cli-run --cli-ir D162-1P_active.wav --cli-intent-burst-count 3
          --cli-intent-burst-interval-ms 2000 --cli-exit-ms 15000（I0 契約どおり）
判定:     **STOP — residual ≠ 0（deferred slot 残留 DSP 1 個体が無処分で shutdown）**
          → B-2〜B-6 は未実施。I1 契約どおり原因分類まで実施済み（本書）。
```

---

## 1. 測定表（指示テンプレート）

| 項目 | B-1 |
| --- | --- |
| exit | **0x00000000**（crash ではなく論理 leak） |
| new dump | **0**（CrashDumps 10 不変） |
| shutdown zone | **clean**（Auto-exit flush → SHUTDOWN_BEGIN → reset completed → LOGGER_DETACH/END 全出現） |
| IR load | **1 回**（D162-1P_active.wav） |
| rebuild count / generation | **3 build（gen 5/6/7）**・burst 3 intent |
| deferred CREATE | **69**（全て gen=7 dsp=000001E2DE2D0080・retention re-defer churn） |
| deferred disposition | **CONSUME 68 / CLEAR 0 / CLEAR_MIDRUN 0 / CLEAR_SHUTDOWN 0 / DISCARD 0 / OVERWRITE 0** → **最終 CREATE 1 件が無処分** |
| retire | gen5: retireByHandle HIT（midrun・epoch 14）/ gen6: V-D authority retire（retired=1・epoch 17）/ gen7: **retire なし** |
| EBR pend / ovf | 0 / 0（E-3 assert 通過 — leak DSP は EBR 未 enqueue のため E-3 は捕捉不能） |
| destroy / release | **3 / 3**（gen4 placeholder・gen5・gen6）— **gen7 は destroy 0 件** |
| generation 1:1 | **破れ**（構築 4 = placeholder+gen5/6/7 / 破壊 3 → **residual = 1**） |
| direct destroy | registered DSP について **0** |
| stale-map MISS/HIT | MISS 2（handle=251/252・dtor E-2）/ HIT 1（midrun の正常 authority destroy・許容） |
| E-3 | 0 |
| Signature A/B/C | 0 / 0 / 0 |
| residual | **1（gen7 dsp=000001E2DE2D0080・TOTAL 149,382,008 bytes）** |
| XRUN / Pressure / Callback max | **0 件**（IR+rebuild でも 0） |
| memory / NUC | 最終 MEM_SNAP: DC live=2 / SC live=2 / Priv=593MB（leak 149MB を含む） |
| **STOP 条件** | **該当: residual ≠ 0**（＋ E-4 会計不成立: CREATE 69 ≠ exits 68） |

## 2. Root cause（ lifecycle 再構成）

```text
gen5 build → publish(seq=5) → gen5 DSP retireByHandle HIT で midrun destroy（epoch 14・正常）
gen6 build → publish SUCCEEDED（executor_.publish gen=6）
gen7 build（:1424）→ publish（[PUBLISH] seq=7 gen=7 worldId=7 :848）
    ↓
★ gen7 DSP が deferred slot で retention re-defer churn を開始（:1445）
   CREATE → CONSUME → submitPublishRequest → DeferredFadingActive → CREATE …
   （D135-8 F3 の re-defer churn。TTL 30s 未満で run 終了のため churn 継続）
   69 CREATE / 68 CONSUME（:1445-:3743）
    ↓
shutdown（Auto-exit flush :3763）時点で **slot に gen7 DSP が残留**（[ISR][Shutdown] 自身が
   deferred=1 oldestAgeMs=3323 を観測 — :3814）
    ↓
S3 の clearDeferredForShutdown が **一度も呼ばれない**:
   (a) EmergencyDrain → isEmergencyDrainRequested()==false → "(diagnostic only)" branch で body 未実行
   (b) Timer C2/C3/C4 midrun clear → 15s run で健康回復 trigger 不発
   (c) C1 fallback → requestDeferredClear 未呼出
    ↓
VerifyDrained の V-D は gen6（active-final）のみ destroy・deferred slot には触れない
    ↓
dtor も deferred slot を処分しない → **gen7 DSP は retire/EBR/destroy 経路を一切通らず
   process exit で OS 回収**（E-3 は EBR 未 enqueue のため捕捉不能）
```

## 3. 発見事項の分類

### 3-1. residual = 1 の本体: **shutdown pipeline の deferred-slot cleanup が無条件でない**（pre-existing 構造 gap）

- S3（G1）は「`clearDeferredForShutdown` が呼ばれた場合の disposition」を修正したものであり、
  **呼び出し自体の無条件化はしていない**。呼び出し口 3 経路はすべて条件付き:
  (a) EmergencyDrain（health request 条件）/ (b) C1 fallback（RebuildThread 停止後の request 時）/
  (c) midrun E-4d（Timer C2/C3/C4 trigger 条件）。
- 60-gen の G 系 soak では 420s の実行中に必ず Timer trigger が発火し midrun clear が閉じていた
  ため gap が顕在化しなかった。**B-1 の短時間 profile（15s・健康 trigger なし）で初めて
  実測顕在化した。**
- D162-1P 時代に測定された「shutdown 時 1 件 leak（S3 起源）」と同一クラスの現象であり、
  **G1/G2 の regression ではない**（G 前ならこの DSP は S3/E-4d どちらでも処分されていなかった
  同一 leak。G4 の 60-gen 証明は「Timer trigger が発火する profile」での closure 実証）。
- 副次所見: gen7 は **publish済み**（[PUBLISH] seq=7 worldId=7）の DSP が churn で
  re-defer されていた。published DSP が slot に残留する経路の意味論（D135-8 F3 churn と
  publish 状態の相互作用）は次工程の root-cause 課題。

### 3-2. **V-D-b が実際に UAF を防止した（G0 設計判断の実証）**

- V-D の fading-final target（000001E2C356B080）は、**midrun（line 1418）に既に破壊済みの
  gen5 DSP の dangling pointer** だった（registry slot が Retired のまま reclaim pending・
  address 未再利用のため resolve が旧 instance を返した）。
- V-D-b（authority retire）: map 不在 → `retired=0` → **破壊せず no-op** — 安全。
- **V-D-a（destroyRolledBackDSP direct destroy）だったら**: 解放済み pointer への
  `~DSPCore + aligned_free` = **double-free / UAF crash** になっていた。
- → G0 N-1（stale map latent 二重破壊）は latent ではなく、**この profile で実発火し得る
  経路だった**ことが確定。V-D-b 採用判断（G0 → G2）が crash を防止した。

### 3-3. D120 race との関係

- **crash なし・zone clean** → D120 teardown race の再現ではない。
- ただし「fading handle が shutdown 時に dangling 解決し得る」事実（3-2）は、D120-7 仮説
  (c)（audio device close と DSP destroy の HB）系の観測材料として residual register に記録。

## 4. I1 契約上の処置

| 契約 | 実行 |
| --- | --- |
| STOP 条件該当 run の即停止 | **実行**（B-2〜B-6 未実施） |
| log / dump 保存 | A1-A8・B1 log は `evidence/D162-2-I1/` に保存済み・dump 0 |
| shutdown zone 特定 | clean（zone T1 不該当） |
| lifecycle phase 特定 | deferred slot 残留（`[ISR][Shutdown] deferred=1`）・S3 site 未到達 |
| residual か新規 defect か | **pre-existing 構造 gap の実測顕在化**（G regression ではない・新規 crash でもない） |

## 5. 次の選択肢（ユーザー判定事項）

1. **修復実装（推奨候補）**: shutdown sequence に **無条件の deferred-slot disposition** を追加
   （例: releaseResources の VerifyDrained または dtor body で、slot 保持 DSP に対して
   S3 と同一の authority retire を実行）。これにより (i) B-1 型 leak が消滅、
   (ii) S3 standalone `retired=1` ケースが実測可能になる（G1-G4 の未観測事項の解消）。
   設計確認点: 実装 phase の選定（releaseResources vs dtor）・INV-D162-8 の EBR drain 順序との
   整合・churn 中（TTL 未満）の shutdown での retired 対象の意味論。
2. **profile 側回避**: B profile を EmergencyDrain が要求される条件へ変更する（測定を変えるだけで
   構造 gap は残るため非推奨）。
3. **residual register 登録のみで保留**（60-gen 運用では顕在化しないが、短時間 restart 運用では
   毎回 149MB leak になり得る）。

いずれの場合も **B-2〜B-6 の継続は、この gap の扱いを決めてから**が妥当
（同一 leak が 6/6 run で発生することが確定しているため）。
