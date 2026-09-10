# ConvoPeq 残存バグ改修計画書（ISR 準拠・work92）— v3 gate 完了版

- **v3.1 補訂日**: 2026-09-09（未確定事項の全件確定 — 詳細は `RECONCILIATION_20260909.md`。行番号訂正 B-8/B-6/B-7a/C-7・B-4 設計補完（DspNumericPolicy は cachedThreadHash 維持）・C-8 設計補完（icx :1622 target 固有 /QxCORE-AVX2 の config-gated 化を追加）・B-3 拡張オプション（OversamplingType 同型キャスト）を確定）
- **v3 改訂日**: 2026-09-09（v2 レビューの mandatory gate G1〜G6 を全て実施・全 PASS）
- **v2 改訂日**: 2026-09-09（レビュー指摘「CONDITIONAL APPROVAL」を受けての全面改訂）
- **v1**: 2026-09-09 初版（B-2/C-8 を設計確定としていた点を修正）
- **基準**: `doc/work92/CORRECT_INTEGRATED_BUG_LIST.md` / authority: ConvoPeq.md `Generated: 2026-09-09 01:32:32` / HEAD **40f4229e**（v3.1 で実測・54ba7b40 からの src 差分は diag flag plain-bool 化+NOLINT+コメントのみで全判定不変）
- **設計基準**: Practical Stable ISR Bridge Runtime — 「RT は待たない・解放しない・判断しない」「Publish/Crossfade/Retire authority を増やさない」「Retire は Epoch を通る」「Overflow は silent loss にしない」

---

## 0. v3 最重要更新 — mandatory gate G1〜G6 全 PASS

v2 レビューで実装開始前の mandatory gate とされた 6 項目を全て read-only 監査として実施し、**全て PASS** した。結果サマリ:

| Gate | 項目 | 結果 | 要点（詳細は各節） |
|---|---|---|---|
| **G1** | B-2 V-B2-1 CFG 確認 | **PASS** | `setIRChangeFlag()`（UIEvents.cpp:177）は submit（:162）と同一ローカル変数 `needsStructuralRebuild` で gate され、:158〜:174 間に変数への再代入なし。deferred 分岐（:132-147）は submit と flag を**両方**スキップし Timer 経路（:794→:800）に委譲 — submit 先行不変 |
| **G2** | A-3 L/R 片側 failure semantics | **PASS** | composite score は L+R blend で部分成功は意味論的に無効 → zero-both が唯一の整合解。`evaluate()` は NonRT（NoiseShaperLearner worker :1346）のため DBG 許可 |
| **G3** | B-4 thread_local 契約 | **PASS** | steady-state = thread_local 読取のみ。first-init は「そのスレッドの初回 enter()」で発生（audio thread は初回 callback block）— **既存 `cachedThreadHash()` と同一の初回タイミング**。AC-B4-3 を steady-state 表記に改訂 |
| **G4** | B-5 int64→int narrowing proof | **PASS** | `MAX_FILE_LENGTH = 2147483647`（**samples** 限定 — エラーメッセージ :454「2GB samples limit」で確認）。ループ不変式 `offset + chunk ≤ fileLength ≤ INT32_MAX`（int64 算術）により全 narrowing が証明。belt-and-braces の assert 追加を設計に含める |
| **G5** | B-7b 全 writer + linearization | **PASS + 新発見** | writer は 3 系統（requestBandReset CAS・reset :280・prepareToPlay :791）。**現行 `publishAtomic(0)` は並行する requestBandReset の累積 mask+serial を clobber する pre-existing 競合**（work89 R3 の具体化）— B-7b がこの競合も修正する。consumer は serial-advance+mask=0 を acknowledge として正しく処理（無変更で可） |
| **G6** | C-8 icx generated ISA 実測 | **PASS** | 代表 DSP ループを icx 2026.1 で実コンパイル → `/QxCORE-AVX2` と `/arch:AVX2` の **ISA セット完全同一**（VADDPD/VMOVUPD/VZEROUPPER + スカラーのみ）・**Intel 専用命令（vpcompressd/zmm/gather/kmov 等）ゼロ**。差分は load/store スケジューリング順のみ。→「AMD execution likely possible / officially unsupported」の整理が実証と整合 |

**v3 の最終ゲート判定**: 6 gate 全 PASS により、**A-1 / A-2 / B-1 / B-3 / B-4 / B-6 / B-8 / B-7a / B-7b / C-1〜C-9 が実装着手可能**。B-2 はコード変更なし（コメント文書化のみ）。C-8 は G6 実測を根拠に Phase C で実施可。

---

## 0-1. v2 レビュー判定の反映（v2 からの継続事項）

レビューの CONDITIONAL APPROVAL を受け、各項目を実装ゲート付きで再分類した:

### G1 詳細 — B-2 V-B2-1 CFG 確認（v3 実施・PASS）

`convolverParamsChanged()`（UIEvents.cpp:57-193）の制御フローを実測:

```text
:73   needsStructuralRebuild = false（初期化）
:91-99 lock 内で判定（uiHasIr != committedHasIr / hash 不一致）
:132-147 deferred 分岐（prepared IR apply 200ms window）:
        setRebuildReason(DeferredStructural) + needsStructuralRebuild = false
        → :158 submit も :174 setIRChangeFlag も両方スキップ
        → 後続の Timer deferred_structural_due 経路（:794 submit → :800 flag）で実行
:158  if (needsStructuralRebuild) → :162 submitRebuildIntent(Structural)
:174  if (needsStructuralRebuild && srForRebuild > 0.0) → :177 setIRChangeFlag("UI")
```

**CFG 証明**: `:177` は `needsStructuralRebuild`（ローカル bool）が true のときのみ到達し、`:162` も同一変数で gate される。`:158` から `:174` の間に `needsStructuralRebuild` への再代入は存在しない（実測: :158-193 の間に代入文なし）。したがって **「:177 に到達 ⇒ :162 が同一呼び出し内で実行済み」が CFG レベルで成立**。deferred 分岐は両方をスキップするため flag 単独セットの経路は存在しない。

**G1 = PASS**: V-B2-1 完了。B-2 は「コメント文書化のみ」で実装可能（コードロジック変更ゼロ）。

### G5 詳細 — bandResetPacked 全 writer + CAS linearization（v3 実施・PASS + 新発見）

**writer 全数列挙**（rtk rg 全 src 実測）:

| writer | 操作 | 備考 |
|---|---|---|
| `EQProcessor.h:536-545` requestBandReset() | CAS ループ（serial+1・mask OR 累積） | Parameters.cpp:89/:169/:192 から呼出 |
| `Core.cpp:255, :488, :554` requestAllBandReset() | 上記の mask=0xFFFFFFFF 版 | resetToDefaults 系 |
| `Core.cpp:280` reset() | `publishAtomic(bandResetPacked, 0)` | **serial を 0 に巻き戻す** |
| `Core.cpp:791` prepareToPlay() | 同上 | 同上 |
| `Core.cpp:590, :650` syncStateFrom/syncGlobalStateFrom | consume のみ（reader） | デッドコード確定（work89 §9 F-2） |

**新発見（pre-existing 競合）**: 現行 `publishAtomic(bandResetPacked, 0)` は、並行して実行中の `requestBandReset()` が累積した mask+serial を**無条件に clobber**する。例: requestBandReset(0xFF) が serial=5/mask=0xFF を書いた直後に reset() が 0 を publish → **保留中のバンドリセット要求が消失**。これは work89 §7.2 R3 の具体化であり、B-7b（CAS serial+1 & mask=0）がこの競合も修正する。consumer（Processing.cpp:595-601）は serial-advance+mask=0 を `fetch_or(0)` = acknowledge として正しく処理するため **consumer 無変更**。

**linearization 確定**: B-7b 適用後、全 writer が CAS（単一 linearization point）に統一され、serial は単調増加のみ。reset/prepareToPlay は「mask=0 への意図的クリア」（reset は filterState を memset するため保留 mask は redundant）という意味論で正当化される。

**G5 = PASS**: B-7b 設計確定。AC-B7b-5（pre-existing clobber 競合の解消）を追加。

### G2/G3/G4/G6 の詳細は各項目節（A-3 / B-4 / B-5 / C-8）に記載。

| ゲート | 項目 | v1 からの変更 |
|---|---|---|
| 🟢 **実装着手可能** | A-1, A-2, B-1, B-3, B-6, B-8, C-1〜C-7（protocol 非変更分）, C-9 | A-1 テスト方式変更・A-2 テスト分離・C-9 文言修正 |
| 🟡 **実装前再監査が必要 → 本 v2 で再監査実施済み** | A-3, B-4, B-5, B-7 | §2〜§4 に再監査結果を反映し設計更新 |
| 🔴 **現状承認不可 → 本 v2 で再監査完了・判定更新** | B-2, C-8 | B-2 は「実装案撤回・プロトコル監査結果による再分類」。C-8 は compiler matrix 確定版に更新 |
| Phase B 内部順序 | 再編成 | 独立項目 → identity → B-7a → (B-2 判定) → B-7b → B-5 の順に修正 |

**B-2 再監査の結論（§2 で詳述）**: プロトコル全体を追跡した結果、**元バグ指摘（big 1-10「createImpl nullptr で IR 変更要求が永久消失」）のメカニズムは現行コード順序では成立しない**ことが実証された。`m_pendingIRChange` は「構造 rebuild intent が先行発行済みであること」を示す level flag であり、Snapshot.cpp:95 の消費は正当な acknowledge である。したがって B-2 は「バグ修正」から「**設計文書化 + 残余検証 1 項目**」に再分類。

**C-8 再監査の結論（§5 で詳述）**: CMakeLists は既に `CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"` ガードが確立（:79, :93, :637, :849, :897, :933, :954）しており、compiler-aware 条件は既存パターン。MSVC / icx / AMD の 3 軸 matrix を確定して設計更新。

---

## 1. 🟢 Phase A — 実装着手可能（P0）

### A-1. big 1-1 — nucHCMode / nucLCMode のセッション永続化【着手可能】

**現状（実測）**: `getState()`（StateAndUI.cpp:202-250）の setProperty 20 箇所に nucHCMode/nucLCMode なし。`setState()`（:289-）にも読込なし。実行時は正しく同期済み（:142-143 同期・:194-199 jlimit・:55-56/:865 ハッシュ・`setNUCFilterModes()` :816-838）。

**修正設計**（v1 から変更なし・既存 authority 崩さず）:
```cpp
// getState() — :242（maxCacheEntries の後・irPath ブロックの前）に追加:
v.setProperty ("nucHCMode", static_cast<int>(snapshot.nucHCMode), nullptr);
v.setProperty ("nucLCMode", static_cast<int>(snapshot.nucLCMode), nullptr);

// setState() — :350（tailL1L2Multiplier の後）に追加:
if (v.hasProperty ("nucHCMode") && v.hasProperty ("nucLCMode"))
    setNUCFilterModes(static_cast<convo::HCMode>(static_cast<int>(v.getProperty("nucHCMode"))),
                      static_cast<convo::LCMode>(static_cast<int>(v.getProperty("nucLCMode"))));
// setNUCFilterModes（:816）内部の jlimit 正規化に一任（重複クランプ禁止）
```

**ISR 適合性**: runtime state → snapshot → pendingOverride の既存 authority を崩さず、ValueTree ↔ nucHCMode/nucLCMode の往復を追加するのみ。二重管理を作らない。

**テスト（v2 修正）**: `BuildSnapshotTests` への寄せではなく、**実際の `getState()` → `setState()` round-trip を直接通す**テストを実装する。
- AC-A1-1: round-trip 後 `getNucHCMode()/getNucLCMode()` が元値一致
- AC-A1-2: 範囲外値（-1, 99）で jlimit 正規化・クラッシュなし
- AC-A1-3: プロパティ不在の旧セッションでデフォルト維持（後方互換）
- AC-A1-4: round-trip で rebuild が 1 回だけ発火（setNUCFilterModes の coalesce 動作確認）

**リスク**: Low — pure addition

---

### A-2. big 1-3 — `_mm256_store_pd` → `_mm256_storeu_pd` 統一【着手可能】

**現状（実測）**: `InputBitDepthTransform.h:114-115` のみが契約なし store_pd（:60/:81 は storeu 済み）。呼び出し元 3 箇所（DSPCoreIO.cpp:216 / LoaderThread.cpp:481 / ResampleAndFallback.cpp:325）。MKLNonUniformConvolver.cpp:1407, 1668 は mkl_malloc 64-byte 保証付き accumBuf で **修正不要**。TruePeakDetector.cpp:85 はローカル alignas 配列。

**修正設計**（v1 から変更なし）: :114-115 を `_mm256_storeu_pd` に変更（2 行）。

**テスト（v2 修正: 2 つに分離）**:
- AC-A2-1（alignment independence）: 非アライン `dst`（double 1 個分オフセット）を渡して #GP 不発
- AC-A2-2（numerical equivalence）: 正常系 IR 読込の数値結果が旧実装と等価（store/storeu は演算結果を変えないため、これは独立した回帰テストとして位置づける）

**リスク**: Low

---

## 2. 🔴→✅ B-2 再監査結果 — m_pendingIRChange acknowledgement protocol（v2 の中心）

### 2-1. 全サイト実測（rtk rg 全数）

| サイト | 操作 | 役割 |
|---|---|---|
| `AudioEngine.h:1524`（`setIRChangeFlag()` 内） | `publishAtomic(true, release)` | **唯一の writer** |
| `Timer.cpp:800` | writer 呼び出し | **`submitRebuildIntent(Structural, DeferredStructuralRebuildRequested)`（:794）の直後に set** |
| `UIEvents.cpp:177` | writer 呼び出し | **`submitRebuildIntent(Structural, EnqueueSnapshotCommand)`（:162）の後に set**（同一関数内・:162 は無条件経路） |
| `Snapshot.cpp:95` | `exchangeAtomic(false, acq_rel)` | **唯一の消費クリア（acknowledge 点）** |
| `Timer.cpp:849` | `consumeAtomic`（非消費） | finalizeReady gate —「IR 変更 in flight 中は finalize しない」 |
| `Parameters.cpp:381, :452` | `consumeAtomic`（非消費） | shouldDeferRebuild —「IR 変更 in flight 中は新規 rebuild を deferred に」 |
| `AudioEngine.h:4989` | 宣言 `std::atomic<bool>` | — |

### 2-2. 確定した acknowledgement protocol（状態機械）

```text
[IR change requested]（UI 操作 / deferred-structural due）
   ↓ ① submitRebuildIntent(Structural)  ← intent が先行発行される
   ↓ ② setIRChangeFlag()（level flag = true）
   ↓
[IR change pending]（flag = true の間）
   ├─ Timer:849: finalize gate が閉じる（finalize 遅延）
   ├─ Parameters:381/452: 新規 rebuild が DeferredFinalizeAware に遅延
   └─ snapshot builder（:95）: flag を exchange で消費 → promoteToStructural=true →
      param-only snapshot を作らず早期 return（:101-107）＝ acknowledge
   ↓
[structural rebuild が処理]（①で発行済みの intent が RebuildThread で消化・collapse latest-wins で IR 変更を内包）
   ↓
[complete]（flag=false・irLoading/isIRFinalized/outstandingRebuild の他条件が finalize/defer を制御）
```

### 2-3. 元バグ指摘の検証結果 — **現行コードでは不成立**

元指摘（big 1-10）: 「createImpl が nullptr を返す場合、IR 変更要求は永久に失われる」。

**実測による訂正**: 現行コードは `:95 exchange → :101 promoteToStructural なら早期 return` の順序であり、**flag がセットされている状態では createImpl は呼び出されない**。createImpl が到達するのは flag=false の場合のみで、消失しうる flag は存在しない。2026-07-26 の元レポートは当時のコード順序（createImpl 後クリア）を想定したものであり、現行順序とは一致しない。

### 2-4. 残る真の妥当性確認点（1 件のみ）

protocol の健全性は「**setIRChangeFlag を呼ぶ全経路が、同一呼び出し内または先行して構造 rebuild intent を発行している**」ことに依存する。実測では 2 writer とも充足（Timer:794→800 / UIEvents:162→177）だが、UIEvents.cpp の :162 と :177 が**別の if ブロック**に分かれているため、実装時に 1 項目だけ静的確認を行う:

> **V-B2-1（実装前チェック）**: UIEvents.cpp の convolverParamsChanged において、「:177 に到達する全パスが :162 の submit を通る（または別経路の構造 intent が in-flight）」を制御フローで確認。不通の場合は :177 直前に submit 追加（1 行）。

### 2-5. B-2 の最終判定（v2）

| 項目 | 判定 |
|---|---|
| バグとしての big 1-10 | **不成立（バグではない）** — 現行コード順序で消失メカニズムが存在しない |
| 実装方式（consume への変更） | **撤回** — 現行 exchange が正しい acknowledge 点 |
| 採用する対応 | **(a) 本プロトコルの設計文書化（Snapshot.cpp:95 直前に状態機械コメント追加）+ (b) V-B2-1 の実装前チェック + (c) 任意の観測性向上（promote 経路の diagLog に in-flight rebuild generation を追記）** |

ISR 適合性: 既存 protocol は「intent 先行発行 → flag は marker → snapshot 側 acknowledge」であり、authority を増やさない・RT に触れない。**コード変更はコメントのみ**。

---

## 3. 🟡→✅ Phase B — 再監査反映後の設計（実装順序を再編成）

実装順序（レビュー推奨どおり・独立項目を先行）:

```text
B-3 enum validation → B-8 saturating timing → B-1 rename → B-6 fftSize
   ↓（独立項目完了後）
B-4 thread identity（Low-Medium）
   ↓
B-7a reset shadow cleanup
   ↓
B-2 → 判定確定済み（§2: コード変更はコメントのみ・V-B2-1 チェック）
   ↓
B-7b bandResetPacked serial protocol（独立 pre-audit 完了: §3-7）
   ↓
B-5 streaming loader（reader API int64 修正込み）
```

### B-1. emitRetireIntentRT → emitRetireIntentNonRT リネーム【着手可能・v1 から変更なし】
- 宣言 ISRRetire.h:59 + 定義 ISRRetire.cpp:94-104 + 呼び出し元全数置換
- AC-B1-1: ビルド通過 / AC-B1-2: `emitRetireIntentRT` grep 0 件 / AC-B1-3: CTest 40/40
- Finding 9 コメントは新名前に合わせて更新

### B-3. StateIO.cpp:90 の enum 範囲チェック【着手可能・v3.1 確定】
- `NoiseShaperType::Psychoacoustic(0)〜Fixed15Tap(3)` の範囲ガード追加（~6 行）
- **v3.1 実測**: `setNoiseShaperType`（Parameters.cpp:405）内部に正規化なし・**StateIO.cpp:120 の setOversamplingType も同型の無検証キャスト**（同時修正は B-3 拡張オプション・ユーザー gate で判定）
- AC: 範囲外値（4, -1）でデフォルト維持

### B-6. fftSize int64 化【着手可能・v3.1 影響リスト確定】
- `MKLNonUniformConvolver.h:330` `int fftSize` → `int64_t` + cpp 側 15 箇所の cast 整理（v3.1 実測: :297/:355/:778-781/:792/:797/:841/:843/:845/:847/:893/:894/:906/:907 — コンパイルエラー駆動で全数消去）
- AC: 5秒@768kHz の割当サイズ正確性 / 既存テスト全通過

### B-8. タイミング計算 saturating subtraction【着手可能・v3.1 行番号訂正】
- **AudioBlock.cpp:632/:638 + BlockDouble.cpp:589/:594**（v3.1 実測訂正 — 旧 :624/:630/:664 はズレ。減算+uint32_t キャストは実測 4 箇所のみ）
- AC: 逆転注入で巨大値なし / 通常計測値不変

### B-4. thread token 単調 ID 化【要修正反映・リスク Low→Low-Medium に訂正・v3.1 設計補完】
- **v3.1 追加確定**: token 消費点は RCUReader.h:47/:127/:150-152 + EpochDomain.h:75 に加え、**DspNumericPolicy.h:44/:120 が `cachedThreadHash()` を直接使用**。DspNumericPolicy の tag は衝突許容の役割タグのため **cachedThreadHash を維持**し、置換対象は `src/core/` の RCU/Epoch token 系のみ。PLAN §7 の静的監査パターンは `src/core/` → **`src/` 全域**に拡張（DspNumericPolicy.h が src/ 直下のため）

**レビュー指摘の反映（v3 確定）**: 「既存と同じだから安全」という比較はリスク評価には使えるが safe の証明にはならない、という指摘を採用。正確な契約は「**既存 `cachedThreadHash()` と同等の初回 thread_local initialization pattern を置換する。新規の blocking/allocation API は導入しない**」。

**G3 監査結果（thread_local 初期化契約の明文化・PASS）**:
- **steady-state**: thread_local 読み取り 1 回（dynamic init 完了後）。追加コストゼロ
- **first-init**: **そのスレッドでの初回 `enter()` 呼び出し時**に 1 回のみ発生。audio thread の場合は**ストリーム開始後の最初の callback block**（BlockDouble.cpp:151 が初回 enter 点）。既存 `cachedThreadHash()`（ThreadHash.h:9-16）も同一タイミングで初回 init するため、**本変更は初回 init のタイミングを変えない**（計算内容が std::hash\<thread::id\> → relaxed fetch_add に置き換わるのみ）
- C++ 実装上、thread_local dynamic init の内部ガード機構は implementation-defined だが、**既存コードと同一の機構に依存するため新規リスクは導入されない**（本計画は「既存 pattern の置換」であり「新 pattern の導入」ではない）
- warm-up は non-viable: thread_local は per-thread のため、Audio Thread 以外からの事前初期化は効果がない（実測: audioThreadRcuReader は AudioEngine メンバ（h:4856）として構築時生成されるが、token は enter 時に初回評価）

**改訂設計**:
1. `ThreadHash.h` に `acquireUniqueThreadId()` を追加（counter は 1 起点静止・0 は無効値予約）
2. `RCUReader.h` / `EpochDomain.h` の token 取得を新関数に置換
3. **契約コメントを ThreadHash.h に固定**: 「first-init はそのスレッドの初回 enter() で発生（audio thread = 最初の callback block）。既存 cachedThreadHash() と同一タイミング・同一機構。新規 blocking/allocation なし」
4. `EpochDomain.h:74, 523-525, 571` の ownerThreadId（BUG-063 修正）と同一 ID 系に統一

**AC-B4-1**: 100+ スレッド生成で ID 重複ゼロ / **AC-B4-2**: 既存 RCU/Epoch テスト全通過 / **AC-B4-3（v3 改訂）**: **steady-state では thread_local token の読み取りのみ**（first-init は既存 cachedThreadHash と同一の初回 pattern・新規 blocking/allocation なし）

**リスク**: Low-Medium（first-init は初回 callback block で 1 回・既存と同種の pattern 置換）

### B-7a. reset/prepareToPlay の rt シャドウ直接書込削除【分離・着手可能・v3.1 行確定】
- **削除行確定（v3.1 実測）**: reset() 側 **Core.cpp:282-284**・prepareToPlay() 側 **Core.cpp:793-795**（各 3 行）

**スコープ（変更 A のみ）**: `EQProcessor.Core.cpp` の reset()（:284 付近）と prepareToPlay()（:795 付近）から以下 3 行を削除:
- `rtSeenAgcResetSerial = 0`
- `rtSeenBandResetSerial = 0`
- `rtDeferredBandResetMask.store(0, relaxed)`

**安全性の根拠（実測）**: serial は先行する `fetchAddAtomic`（:281, :792）で前進済み。Audio Thread 側検知（Processing.cpp:586-601）は `serial != rtSeen` で shadow を自己更新するため、Non-RT 側の事前書込は不要。prepareToPlay は publish 前の新規 DSPCore（rtSeen 初期値 0・serial 1）に対して呼ばれ、RT が `1 != 0` を検知して 1 回リセット — 意図どおり。

**ISR 適合性**: 「Non-RT → Audio Thread の通信は atomic publish のみ・rt シャドウは Audio Thread 専有」の不変条件を確立。**契約違反の削除のみで新規 protocol 導入なし**。

**AC-B7a-1**: prepareToPlay() 後の最初の process で AGC shadow が 1.0/0.0/0.0 に 1 回だけリセット / **AC-B7a-2**: 連続 reset() で二重リセットなし / **AC-B7a-3**: TSan クリーン

**リスク**: Low

### B-7b. bandResetPacked serial protocol 変更【分離・G5 再監査 PASS 反映】

**スコープ（変更 B）**: `EQProcessor.Core.cpp:280, :791` の `publishAtomic(bandResetPacked, 0)` を「serial 前進 + mask=0」の CAS に変更。

**G5 再監査結果（全 writer + CAS linearization の確認・PASS）**:

writer 全数列挙（rtk rg 実測）:
1. `requestBandReset()`（EQProcessor.h:536-545）— CAS ループ（serial+1・mask OR 累積）。呼び出し元: Parameters.cpp:89/:169/:192（per-band）、Core.cpp:255/:488/:554（requestAllBandReset = mask 0xFFFFFFFF）
2. `reset()`（Core.cpp:280）— `publishAtomic(bandResetPacked, 0)` ← **B-7b の置換対象**
3. `prepareToPlay()`（Core.cpp:791）— 同上 ← **B-7b の置換対象**
4. `syncStateFrom()` / `syncGlobalStateFrom()`（Core.cpp:590/:650）— reader のみ（デッドコード確定）

**G5 新発見（pre-existing 競合）**: 現行 `publishAtomic(bandResetPacked, 0)` は、並行して実行中の `requestBandReset()` が累積した mask+serial を**無条件に clobber** する（例: requestBandReset(0xFF) が serial=5/mask=0xFF を書いた直後に reset() が 0 を publish → 保留中の要求が消失）。これは work89 §7.2 R3 の具体化であり、B-7b がこの競合も修正する。**reset() が保留 mask を捨てること自体は正当**（reset() は filterState を memset するため保留 mask は redundant — work89 §9 F-6）。

**linearization 確定**: B-7b 適用後は全 writer が CAS（単一 linearization point）に統一され、serial は単調増加のみ。ユーザーレビューの interleaving 例「A: serial 10 mask X / B: serial 11 mask Y / A: reset → serial 12 mask 0」でも、CAS ループが最新 packed を観測するため lost update なし。**consumer（Processing.cpp:595-601）は serial-advance+mask=0 を `fetch_or(0)` = acknowledge として正しく処理し、無変更で可**。

**pre-audit 結果（Audio Thread consumer の追跡・v2 から継続）**:
- consumer は `Processing.cpp:595-601`: `bandResetSerialNow != rtSeenBandResetSerial` を検知 → `rtSeenBandResetSerial = bandResetSerialNow` + `rtDeferredBandResetMask.fetch_or(bandResetMaskFromPacked(packed))`
- **serial 前進 + mask=0 の packed を publish した場合**: RT は serial 変化を検知 → `fetch_or(0)` = no-op → rtSeen を新 serial に同期。**consumer 側の変更は不要**
- 既存ヘルパーが全て実在（EQProcessor.h:515-529）+ `requestBandReset()`（:531-537）がコピーすべき CAS パターンの参照実装

**修正設計**:
```cpp
// ★ B-7b: serial 前進 + mask=0 クリア（serial 巻き戻し R2 の解消 + G5 clobber 競合の解消）
std::uint64_t packed = convo::consumeAtomic(bandResetPacked, std::memory_order_acquire);
for (;;)
{
    const auto serial = static_cast<std::uint32_t>(bandResetSerialFromPacked(packed) + 1u);
    const std::uint64_t desired = makeBandResetPacked(serial, 0u);
    if (convo::compareExchangeAtomic(bandResetPacked, packed, desired,
                                     std::memory_order_acq_rel, std::memory_order_acquire))
        break;
}
// 削除: publishAtomic(bandResetPacked, 0, ...)
```

**ISR 適合性**: `requestBandReset()`（既存の安全パターン）と同一の CAS 単調増加設計に統一。Audio Thread は無変更。

**AC-B7b-1**: bandResetPacked serial が reset() 後も巻き戻らない / **AC-B7b-2**: silent 中の不意な全バンド memset が発生しない / **AC-B7b-3**: reset() → process 1 回で deferred mask が正しく処理される / **AC-B7b-4**: TSan クリーン / **AC-B7b-5（v3 追加）**: **並行 requestBandReset 実行中の reset() で保留 mask が clobber されない**（CAS 統一による競合解消・G5 新発見の検証）

**リスク**: Low-Medium（state machine 変更・ただし全 writer 列挙 + consumer 追跡 + 参照実装パターン準拠）

### B-5. LoaderThread ストリーミング読込【G4 再監査 PASS 反映】

**レビュー指摘の反映（修正必須）**: `reader->read(..., static_cast<int>(offset), ...)` のオフセット cast を廃止。実測で JUCE `AudioFormatReader::readSamples(int* const* destChannels, int numDestChannels, int startOffsetInDestBuffer, **int64 startSampleInFile**, int numSamples)`（juce_AudioFormatReader.h:279-283）を確認 — **startSampleInFile は int64**。

**G4 監査結果（int64→int narrowing の全境界 proof・PASS）**:

境界列挙（実測: LoaderThread.cpp:448-485）:
| 変数 | 型 | narrowing 点 | 証明 |
|---|---|---|---|
| `fileLength` | int64（reader->lengthInSamples） | なし（ループ条件は int64 比較のまま） | — |
| `MAX_FILE_LENGTH` | `static constexpr int64 = 2147483647`（:450） | — | **ガードは samples 限定**（:454 エラーメッセージ「exceeds 2GB samples limit」で確認 — bytes ではない） |
| `offset + chunk` | int64 算術 | `copyFrom(ch, static_cast<int>(offset), ...)` | **ループ不変式**: `offset + chunk ≤ fileLength`（`chunk = min(kStreamChunk, fileLength - offset)` より）かつ `fileLength ≤ INT32_MAX`（:452 ガード）→ `offset + chunk ≤ INT32_MAX` が int64 算術で成立。static_cast は値域証明済みの narrowing |
| `chunk` | int64 算術結果 | `reader->read(..., chunk, ...)` の numSamples（int） | `chunk ≤ kStreamChunk = 256*1024` で自明に int 域内 |
| `offset` | int64 | `reader->read(..., offset, ...)` の startSampleInFile | **narrowing なし** — int64 のまま渡す（本修正の本体） |

**belt-and-braces**: 実装時に `jassert(offset + chunk <= INT32_MAX);` をループ先頭に追加（デバッグ時のみ・Release でゼロコスト）。理論証明済みだが、将来 MAX_FILE_LENGTH 変更時の事故防止。

**「2GB」表記の明確化（レビュー指摘採用）**: 本リストの 2GB は **samples 数**（fileLength ≤ 2^31-1 samples）。実メモリは float 4B + double 8B のチャンネル倍率で決まり、ステレオで float ~16GB / double ~32GB が旧実装の確保量。改修後はチャンク 2MB×2 のみ。

**改訂設計**:
```cpp
constexpr int64_t kStreamChunk = 256 * 1024;
// MAX_FILE_LENGTH ガード（≤ INT32_MAX samples）は維持 — loadedIR（juce::AudioBuffer）の
// copyFrom/setSize は int 位置系のため、IR パイプライン全体が int sample 範囲を前提とする
juce::AudioBuffer<float> tempFloatBuffer(numChannels, static_cast<int>(kStreamChunk));  // ~2MB
auto tempAligned = convo::makeAlignedArray<double>(static_cast<size_t>(kStreamChunk)); // ~2MB
for (int64_t offset = 0; offset < fileLength; offset += kStreamChunk)
{
    if (shouldCancel && shouldCancel()) { ...; return false; }   // キャンセル応答性も改善
    const int64_t remaining = fileLength - offset;               // ループ不変: remaining > 0
    const int chunk = static_cast<int>(std::min<int64_t>(kStreamChunk, remaining));
    jassert(offset + chunk <= 2147483647);                       // ★ G4: narrowing 前の belt-and-braces
    // ★ B-5: offset は int64 のまま reader に渡す（static_cast<int> 禁止）
    if (!reader->read(&tempFloatBuffer, 0, chunk, offset, true, true)) { ...; return false; }
    for (int ch = 0; ch < numChannels; ++ch)
    {
        convo::input_transform::convertFloatToDoubleHighQuality(
            tempFloatBuffer.getReadPointer(ch), tempAligned.get(), chunk);
        stepResult.loadedIR.copyFrom(ch, static_cast<int>(offset), tempAligned.get(), chunk);
        // copyFrom の destStartSample は int — 上記不変式により値域証明済み
    }
}
```

**メモリ効果（実測計算）**: 現行 = tempFloatBuffer(float 全長 ~4GB) + tempAligned(double 全長 ~4GB) + loadedIR(~4GB) ≈ 12GB → 改修後 = 2MB + 2MB + loadedIR(~4GB) ≈ 4GB

**AC-B5-1**: 通常 IR で bit-exact 不変 / **AC-B5-2**: 2^31 samples 近傍 IR（bytes ではなく samples 基準 — G4 明確化）でメモリ一定（チャンク分のみ）・**offset は int64 で reader に渡る**（>500M samples 位置でも正しく読める） / **AC-B5-3**: キャンセルが 256k サンプル以内で応答 / **AC-B5-4（v3 追加）**: チャンク境界ちょうど・端数・offset+chunk=INT32_MAX 境界の 3 パターンで narrowing assert が発火しない（G4 proof の動的検証）

**リスク**: Medium（読込ロジック変更）→ 境界テスト（ちょうどチャンク境界・端数）を追加

---

## 4. A-3 再設計 — IPP FFT 失敗時の failure semantics（G2 再監査 PASS 反映）

**レビュー指摘の採用**: 「ゼロ結果 = 無音」という新規 semantics の発明は撤回し、**既存の failure semantics に整合**させる。

**実測した既存 failure semantics**:
- `MklFftEvaluator.h:78-107` — コンストラクタの IPP 初期化失敗は `fftSpec = nullptr` を残し、`evaluate()` 冒頭（:250-254）のガードが `Result{}`（全フィールド 0.0）を返す。**`Result{}` = 「FFT 利用不可」は既に確立された failure semantics**（[Bug 2/3 fix] コメントで文書化済み）
- 唯一の呼び出し元 `NoiseShaperLearner.cpp:1346` — Result を学習スコア（timeScore/freqScore blend）として消費。`Result{}` は「候補のノイズペナルティゼロ」に寄るが、学習は NonRT 自己修復系で継続動作

**G2 結果 — L/R 片側 failure semantics の確認（PASS）**:

レビュー指摘「L success / R failure のとき、L の成功結果を捨ててよいか」を検証した。composite score は:
- `compositeScore` は L+R スペクトラムの加重合成（noisePower/flatness/hf penalty が両チャンネル統合値）
- timeDomainRms は L+R の二乗和平均

つまり **片側でも失敗したら composite score は意味論的に無効**（L のみの部分スコアは R をゼロとしたノイズ評価として不正）。zero-both（L/R をともにゼロ化 + fftFailed=true）が **既存 Result{} semantics と唯一整合する解**。部分成功の保存は行わない。

**evaluate() の NonRT 確認**: 唯一の呼び出し元は `NoiseShaperLearner.cpp:1346`（worker thread・NonRT）。RT path からは到達しないため **DBG ログは許可**。ISR「RT で log しない」契約と整合。

**改訂設計（v3 確定）**:
```cpp
// Result 構造体（:40-46）に 1 フィールド追加（additive）:
struct Result
{
    double noisePower = 0.0;
    double spectralFlatnessPenalty = 0.0;
    double hfPenalty = 0.0;
    double timeDomainRms = 0.0;
    double compositeScore = 0.0;
    bool   fftFailed = false;      // ★ big 1-6: IPP 実行時失敗の明示（既存 Result{} = FFT 利用不可 semantics と整合）
};

// :270-271 — After:
const IppStatus stL = ippsFFTFwd_RToCCS_64f(inputLeft,  reinterpret_cast<Ipp64f*>(spectrumLeft),  fftSpec, fftWorkBuf);
const IppStatus stR = ippsFFTFwd_RToCCS_64f(inputRight, reinterpret_cast<Ipp64f*>(spectrumRight), fftSpec, fftWorkBuf);
if (stL != ippStsNoErr || stR != ippStsNoErr)
{
    // ★ big 1-6: 片側失敗でも composite score は意味論的に無効のため L/R ともゼロ化
    //   （G2 監査確定 — 部分成功の保存は行わない）
    spectrumLeft->fill(0.0);
    spectrumRight->fill(0.0);
    Result r {};
    r.fftFailed = true;
    return r;
}
// :425-426（void 版）も同様に status チェック + memset ゼロ埋め + メンバに失敗記録

// 初回失敗時 diagLog（NonRT 確認済み — G2）:
if (ippFailureCount_.fetch_add(1, std::memory_order_relaxed) == 0)
    DBG("MklFftEvaluator: ippsFFTFwd_RToCCS_64f failed — returning zero Result (fftFailed=true)");
```

**AC-A3-1**: 正常系 bit-exact 不変 / **AC-A3-2**: `fftFailed` フィールドが失敗時に立つ / **AC-A3-3**: 失敗時にゴミ（非ゼロ garbage）が spectrum に残らない / **AC-A3-4**: **L/R 片側のみの IPP failure でも既存 `Result{}` semantics と一致する（zero-both + fftFailed）**（G2 で確定・実装時にテスト化）

**リスク**: Low

---

## 5. C-8 再設計 — compiler matrix 確定版（G6 実測 PASS 反映）

**実測**: CMakeLists は既に `CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"` ガードが 7 箇所（:79, :93, :637, :849, :897, :933, :954）で確立済み。本修正も同一パターンに従う。

### 5-0. G6 実測結果（icx generated ISA 検証・PASS）

レビュー指摘「CMake 上の意図だけで AMD 非対応を確定しない」を受け、**icx 2026.1（oneAPI 2026.1.0 Build 20260617）で代表 DSP ループ（MKLNonUniformConvolver.cpp:1665-1673 相当の load/add/store パターン）を実コンパイルし、生成アセンブリを比較**した:

| 項目 | `/QxCORE-AVX2` | `/arch:AVX2` |
|---|---|---|
| 生成命令セット | VADDPD / VMOVUPD / VZEROUPPER + スカラー（ADDQ/CMPQ/MOVQ 等） | **完全同一** |
| Intel 専用命令（vpcompressd / zmm / gather / kmov / valignd 等） | **0 件** | 0 件 |
| 命令シーケンス | 一部 load/store のスケジューリング順のみ差分 | — |

**結論（v3 確定）**: この代表ループでは `/QxCORE-AVX2` は AVX2 サブセット内の codegen に留まり、Intel 専用命令は生成されなかった。ただしこれは**代表パターン 1 件の実測**であり、全コードパスでの保証ではない。したがって文書上は「**icx build: AMD execution = unsupported**（実行可能性は別概念として断定しない）」の整理を採用。将来 AVX-512/AMX 依存コードが追加された場合は再実測する。

**AC-C8-4（v3 追加）**: icx 生成 ISA の実測手順（本計画付録の icx -FA コマンド）をビルドドキュメントに記録し、codegen 変更時に再実行可能にする。

### 5-1. Compiler / CPU matrix（確定）

| ビルド | Compiler | AVX2 codegen | /fp | AMD 実行 |
|---|---|---|---|---|
| MSVC 版 | MSVC (cl) | `/arch:AVX2`（target 固有・:1235 系） | `/fp:precise` に変更（本修正） | **サポート**（MSVC は CPU 非依存 codegen） |
| icx 版 | IntelLLVM | `/QxCORE-AVX2`（**target_compile_options に移行**・G6 実測で AVX2 サブセット内を確認） | `/fp:fast` 維持（性能要件・:1585-1587 コメント） | **unsupported（明文化のみ・実行可能性は断定しない）** |

### 5-2. 修正設計

```cmake
# (1) MSVC 版 :1519-1520 — /fp:fast を除去（DSP 数値精度）:
set(CMAKE_CXX_FLAGS_RELEASE "/Zm400 /bigobj /O2 /Ob2 /DNDEBUG /Gw /Gy /Zi /utf-8 /EHsc")
set(CMAKE_C_FLAGS_RELEASE   "/Zm400 /bigobj /O2 /DNDEBUG /Gw /Gy /Zi /utf-8 /EHsc")

# (2) icx 版 :1606 — /QxCORE-AVX2 を global から除去:
set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /fp:fast /Gy /Zi /utf-8 /EHsc")

# (2') v3.1 追加: :1622 の target 固有 /QxCORE-AVX2（現行は全 config 無条件）を
#     Release-gated に置換する（除去漏れがあると Debug も AVX2 要求のまま残る）:
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_compile_options(ConvoPeq PRIVATE $<$<CONFIG:Release>:/QxCORE-AVX2>)
endif()

# (3) icx 専用 ISA を target 固有に（compiler-aware ガード — 既存 :897 パターン準拠）:
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_compile_options(ConvoPeq PRIVATE $<$<CONFIG:Release>:/QxCORE-AVX2>)
    # 注: icx の /QxCORE-AVX2 は G6 実測（§5-0）で AVX2 サブセット内の codegen を確認済み。
    #     ただし全コードパスの保証ではないため、icx バイナリの AMD 実行は
    #     「unsupported（公式サポート対象外）」として文書化する（実行可能性は断定しない）。
endif()
```

### 5-3. 明文化すべき 2 点（レビュー指摘どおり分離）

1. **「AMD CPU 上で MSVC ビルドが動く」**: `/arch:AVX2` は標準 AVX2（AMD Ryzen 対応）+ `/fp:precise` により動作保証
2. **「icx build を AMD で公式サポートしない」**: G6 実測で代表ループは Intel 専用命令ゼロを確認したが、全コードパスの保証ではないため、icx バイナリの AMD 実行は**公式サポート対象外**と README/ビルドドキュメントに明記（「実行不能」とは断定しない）

**AC-C8-1**: MSVC Release ビルド + テスト全通過（/fp:precise 変更後の数値回帰確認 — 既存テストの期待値が変わらないことを確認） / **AC-C8-2**: icx Release ビルド通過（LLVM OOM 再発なし — /O2 維持のため） / **AC-C8-3**: CMake configure で compiler 判定が正しく分岐 / **AC-C8-4（v3 追加）**: icx ISA 実測手順のドキュメント化（§5-0 のコマンド）

**リスク**: Medium（ビルド設定変更）→ MSVC/icx 両 Debug/Release ビルド + CTest を必須化

---

## 6. 🟢 Phase C — 残り項目

### C-1. SoftClipPadéPolicy リネーム（着手可能）
- `FastTanhApprox.h:63` → `SoftClipPadePolicy`（U+00E9 除去）・使用箇所 DSPCoreDouble.cpp:127, :191 を置換
- AC: cppcheck 正常解析・Float/Double 出力 bit-exact 不変

### C-2. テスト矛盾条件修正（着手可能）
- `EQProcessorMaxGainTests.cpp:357` の `if (delta > 1e-6)` 削除
- AC: tiny delta ループで `logBound > 0` が検証される

### C-3. 入力側 DC ブロッカー後 sanitize 追加（着手可能）
- `DSPCoreIO.cpp` processInputFloat/Double の DC ブロッカー直後に `sanitizeFiniteChunk()`（4 箇所）
- AC: NaN/Inf 入力後の出力に NaN/Inf なし

### C-4. CacheManager strict-aliasing 修正（着手可能）
- `CacheManager.cpp:267` をバイトオフセット + memcpy に置換
- AC: invalidPointerCast 警告ゼロ・数値不変

### C-5. LockFreeRingBuffer::size() 読取順序固定（着手可能）
- `LockFreeRingBuffer.h:76-81` — writeIndex 先読み + `(w >= r) ? (w - r) : 0`
- AC: SPSC ストレステストで負・capacity 超過なし

### C-6. SnapshotFactory NaN 等価誤判定（着手可能）
- `areSnapshotsEquivalent()` に `std::isnan` チェック追加
- AC: NaN 入力で false

### C-7. volatile sink / alignas ループ内（着手可能・Low・v3.1 行訂正）
- `CacheManager.cpp:203, 243` → `std::atomic_signal_fence` パターン / `SpectrumAnalyzerComponent.cpp:474` → ループ外 + alignas(32)

### C-9. lastResortQueue_ 値初期化（着手可能・文言修正）
- `ISRRuntimePublicationCoordinator.h:1016` — `RetireOverflowEntry lastResortQueue_[kLastResortQueueCapacity]{};` に `{}` 追加
- **v2 文言修正**: 「UB の芽の除去」ではなく「**未初期化状態への依存可能性の排除**」（未初期化値が observable behavior に到達する証拠はないため）。安全側の初期化として実施
- ISR 適合性: Retire authority 構造不変・producer 実装なし（big_bug §10-3-1 設計確定維持）

---

## 7. 実装完了条件（共通）と ISR 不変条件

各項目完了ごとに: targeted test PASS → **targeted static audit PASS（v3 追加）** → Full Debug build PASS → CTest 40/40 → `output_sourcecode_markdown.py` 再生成（NEWER_SRC_COUNT=0）→ 作業報告書を doc/work92/ に記録。

**targeted static audit（v3 追加 — レビュー指摘「テストだけでは structural regression を捕捉できない」への対応）**:
| 対象 | 監査コマンド |
|---|---|
| B-2（実施済み） | `rg -n "setIRChangeFlag" src/` — writer 2 箇所（Timer:800 / UIEvents:177）が変わっていないこと・各 writer の直前に submitRebuildIntent(Structural) があること |
| B-4 | `rg -n "currentThreadToken\|cachedThreadHash\|acquireUniqueThreadId" src/core/` — token 生成が 1 関数に統一・呼び出し側の全置換確認 |
| B-5 | `rg -n "static_cast<int>\\(offset\\)" src/convolver/` — reader への offset cast がゼロであること（copyFrom の int 位置は許容） |
| B-7b | `rg -n "publishAtomic\\(bandResetPacked" src/` — 0 件（旧 publish が全廃されたこと）・`rg -n "bandResetPacked" src/eqprocessor/EQProcessor.h` で CAS writer 3 系統の確認 |
| C-8 | `rg -n "fp:fast\|QxCORE-AVX2" CMakeLists.txt` — global flag からの除去と IntelLLVM ガード内への移行確認 |

ISR 不変条件チェック（全 Phase 完了時）:
| # | 条件 | 検証 |
|---|---|---|
| I-1 | RT path に新規 lock/allocation/blocking なし | `rg "lock_guard\|std::mutex\|new \|malloc" src/audioengine/AudioEngine.Processing.*` 差分ゼロ（B-4 の thread_local は既存契約パターン継続） |
| I-2 | Retire/Publish/Crossfade authority 不変 | authority_source_count_verifier PASS |
| I-3 | HealthMonitor decision authority 化なし | RuntimeHealthMonitor への schedule/decision 追加ゼロ |
| I-4 | Overflow/Failure が silent loss でない | A-3 の fftFailed + diagLog・BUG-015 退避経路の維持 |
| I-5 | atomic 操作は convo wrapper 統一 | check-src-atomic-dotcall.ps1 PASS |
| I-6 | thread_local は既存 RT-SAFE 契約 | NOLINT(thread-local) RT-SAFE: コメント維持 |
| I-7 | T3c/RecoveryLifecycleWord にロジック変更なし | Coordinator CAS path diff = コメントのみ |
| I-8 | isFullyDrained / shutdown 順序に触れない | ShutdownScheduler 系 diff ゼロ |

---

## 8. 残置（本計画対象外）

| 項目 | 理由 |
|---|---|
| big 1-2 の producer 実装・big 1-7 案 B | Retire authority 設計変更。Closed boundary — 設計確定イベント時に別 work item |
| R-新規A〜D | incremental rebuild 有効化まで対応不要 |
| big 3-1/3-3/3-4/3-8 | 実害なし確認済み・監視のみ |
| BUG-040 | 誤報確定（3 段フォールバック実装済み） |
| BUG-044 | 解消済み（MklFftEvaluator.h:138 `= delete` 実測） |

---

## 9. 改訂履歴

| 版 | 日付 | 内容 |
|---|---|---|
| v1 | 2026-09-09 | 初版策定 |
| v2 | 2026-09-09 | レビュー（CONDITIONAL APPROVAL）反映: ①B-2 を read-only 再監査 → 元バグ指摘は現行コード順序で不成立と実証・プロトコル状態機械を文書化・実装案（consume 変更）は撤回・V-B2-1 チェック 1 件のみ残置 ②C-8 を compiler matrix 確定版に更新（IntelLLVM ガード既存パターン準拠）③A-3 を fftFailed フィールド + diagLog 方式に再設計（silent loss 観測性）④B-4 を Low-Medium に訂正（RT callback 初回 thread_local init は既存動作と同種・コスト低減を文書化）⑤B-5 に int64 offset 修正（JUCE readSamples 実測）⑥B-7 を a/b に分離（consumer 追跡完了）⑦A-1 テストを round-trip 直接方式に変更 ⑧A-2 テストを alignment/numerical の 2 本に分離 ⑨C-9 文言修正 ⑩Phase B 順序再編成 |
| **v3** | **2026-09-09** | v2 レビューの mandatory gate G1〜G6 を全て実施（全 PASS）: **G1** B-2 V-B2-1 CFG 確認 — :177 は :162 と同一ローカル変数 gate・deferred 分岐は両方スキップで Timer 経路に委譲・submit 先行不変を CFG レベルで証明 **G2** A-3 L/R 片側 failure — composite score は L+R blend で部分成功は意味論的に無効・zero-both が唯一整合・evaluate は NonRT（DBG 許可）を確認・AC-A3-4 追加 **G3** B-4 thread_local 契約を steady-state/first-init に分離して明文化（「既存と同じ=安全」表現を撤回・AC-B4-3 を steady-state 表記に改訂） **G4** B-5 narrowing 全境界 proof — MAX_FILE_LENGTH は **samples**（:454 メッセージで確認）・ループ不変式 offset+chunk ≤ INT32_MAX を int64 算術で証明・jassert 追加・AC-B5-4 追加 **G5** B-7b 全 writer 列挙 — **現行 publishAtomic(0) が並行 requestBandReset の累積 mask+serial を clobber する pre-existing 競合を新発見**（B-7b が解消）・AC-B7b-5 追加 **G6** C-8 icx ISA 実測 — /QxCORE-AVX2 と /arch:AVX2 の ISA セット完全同一・Intel 専用命令ゼロ（代表 DSP ループ）・「AMD unsupported」表記に整理・AC-C8-4 追加。**共通完了条件に targeted static audit を追加**（レビュー #8 対応） |
| **v3.1** | **2026-09-09** | 未確定事項の全件確定（`RECONCILIATION_20260909.md`・HEAD 40f4229e 実測）: ①B-8 対象行を実測 4 箇所に訂正（AudioBlock:632/:638・BlockDouble:589/:594）②B-6 cpp 側影響 15 箇所を確定 ③B-4 に DspNumericPolicy.h:44/:120 の扱いを追加（tag は cachedThreadHash 維持・監査パターンは src/ 全域）④B-3 に OversamplingType(:120) 同型キャストの拡張オプションを記録（ユーザー gate）⑤C-8 に icx :1622 target 固有 /QxCORE-AVX2（全 config 無条件）の config-gated 化を追加 ⑥C-7 行訂正 :203/:243 ⑦B-7a 削除行確定 :282-284/:793-795 ⑧BUG-044（=delete :138-141）・OPEN 3 sigma（clamp :85）を HEAD で再確認 ⑨A-2 の MKL :1407/:1668 修正不要を実証 ⑩B-5 JUCE 署名 int64 を本ツリーで確認 |
