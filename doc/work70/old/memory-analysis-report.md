# メモリ 2.5GB 肥大化 完全解析レポート

**日付**: 2026-07-09
**対象**: ConvoPeq (Release, icx, `-DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1`)
**ログ**: `doc/work70/ConvoPeq.log` (24375行)

---

## 1. 修正確認: MEM_SNAP が正常動作

MEM_SNAP は **248回の定期出力** を生成。修正が有効であることを確認。

```
[MEM_SNAP] PUBLISH gen=4 | NUC: live=0 alloc=0MB peak=0MB tA=0GB tF=0GB ...
[MEM_SNAP] PUBLISH gen=5 | NUC: live=2 alloc=35MB peak=35MB tA=0GB tF=0GB ...
```

---

## 2. メモリ増加タイムライン（実測値）

| フェーズ | Private | Other (MEM_SNAP) | Δ | 要因 |
|:---------|:-------|:-----------------|:--|:-----|
| 起動直後 | 74MB | N/A | ─ | JUCE 初期化 |
| DSPCore #1 prepare (blk=524288) | 555MB | N/A | +481MB | 初回 prepare（異常に大きいブロック） |
| DSPCore #2 prepare (osFactor=2) | 1354MB | 1354MB | +799MB | SR 切替 + リビルド |
| Gen=4 安定稼働 | 1354~1381MB | 1354~1381MB | ~27MB | コールバック駆動による微増 |
| **↓ Gen=4 内 rebuild obsolete** | | | | |
| DSPCore #3 prepare | 1662MB | 1662MB | **+308MB** | 無駄な prepare（wasted） |
| DSPCore #4 prepare | 1915MB | 1915MB | **+253MB** | 無駄な prepare（wasted） |
| DSPCore #5→#6 prepare | 2467MB | 2467MB | **+552MB** | 2回分の無駄な prepare |
| MixedPhase | 2500MB | 2500MB | +33MB | phase 最適化 |
| Gen=5 publish (IR load) | 2468MB | 2433MB | −32MB | NUC 確保(35MB) + 他微動 |
| Stable (15秒後) | 2477MB | 2442MB | +9MB | ほぼ安定 |

---

## 3. 決定的発見: MKL Convolution は無実（僅か 1.4%）

**Gen=5 安定時の MEM_SNAP より:**

```
Priv=2477MB | NUC: live=2 alloc=35MB peak=35MB | Other=2442MB
```

| カテゴリ | 実測値 | 割合 | 備考 |
|:---------|:-------|:-----|:------|
| MKL convolution (tracked) | **35MB** | **1.4%** | IRFreq=10MB + FDL=22MB + Accum=2MB + Tail=0MB |
| OtherPrivate (非 MKL) | **2442MB** | **98.6%** | DSPCore, JUCE, CRT 等 — 大部分を非MKL領域が占める |

### IR_LOAD/IR_LAYOUT 実測値

```
NUC#A seq=1: IRFreq=5MB FDL=11MB Accum=1MB Tail=0MB Direct=0MB Ring=0MB Total=18MB
NUC#B seq=2: IRFreq=5MB FDL=11MB Accum=1MB Tail=0MB Direct=0MB Ring=0MB Total=18MB
MKL total: 35MB (ステレオ)
```

**Convolution は 35MB であり、メモリ肥大の原因ではない。**

---

## 4. 実測で確認できた事実と、それから導かれる仮説

本セクションでは、事実として確認できたこと（🔍）と、それに基づく推論（📐）を明確に区別して記載する。

### 4a. ログから確認できた事実 🔍

| カウンタ | gen=4 | gen=5 | 変化 |
|:---------|:------|:------|:-----|
| `lifecycle(pub/ret/reclaim)` | `4/0/0` | `5/0/0` | **ret=0, reclaim=0 のまま** |

- `lifecycle(retire)=0` は**事実として確認済み**
- `DSPCORE_PREPARE` が **6回実行** されたことは**事実として確認済み**
- `[XFADE] start` / `[XFADE] completed` が **0回** であることは**事実として確認済み**
  - **ただし**: `[XFADE]` は DSP エンジン遷移 (`crossfadeRuntime_`) のログ。`SnapshotCoordinator`（パラメータ遷移 EQ/NS/AGC）とは別機構である。`startFade()` の実行有無とは無関係。
- `advanceFade()` が誰からも呼ばれていないことは**コード解析で確認済み**
- `SnapshotCoordinator::startFade()` は実行された（`[VERIFY] EQ createdHash=...` が gen=4 で出力されていることから確定）

### 4b. これらの事実から導かれる推論 📐

上記事実に基づき、以下の動作が発生していると考えるのが合理的である（コード解析による裏付けあり）:

```
createSnapshotFromCurrentState
  ↓
startFade(fadeSamples=256/1024/128)
  state = FadingIn, remaining > 0
  ↓
advanceFade(numSamples) → 誰も呼ばない（コード欠落 確認済み）
  ↓
remainingCount() = 初期値のまま
  ↓
tryCompleteFade() が常に false
fadeCompleted ブロックに到達しない
  ↓
Timer 経路の retire() が実行されない

⚠️ **ただし**: DSPTransition には独立した retire 経路が存在する。SnapshotCoordinator とは別の機構であり、この経路は advanceFade に依存しない可能性がある。したがって「すべての retire 経路が停止している」とは断定できず、「Timer 経路の retire が停止している」ことのみが確定している。
DSPCore が解放されず累積
```

ただし「DSPCore が解放されない」により Private Memory がどの程度上昇しているかは、advanceFade() を実際に配線した後に計測で確認する必要がある。

### 4c. リビルド obsolete と DSPGuard RAII

```
rebuild obsolete phase=prepare gen=1 currentGen=3 wasted=98.9ms
rebuild obsolete phase=prepare gen=5 currentGen=4 wasted=140.1ms
rebuild obsolete phase=warmup gen=7 currentGen=4 wasted=224.4ms
```

> **確定事実** 🔍: 上記の 3 回の `rebuild obsolete` で破棄された DSPCore インスタンスは、`DSPGuard` (RAII) により適切に EBR retire されている。コード解析で確認済み。`lifecycle(retire)=0` に見えるのは、この経路では DSPHandle が未登録のため `runtimeRetireCount` カウンタがインクリメントされないため。**メモリリークは発生していない。**

> **推論** 📐: 真の問題は「いったん commit された DSPCore（publish 後に current/fading となったもの）が、後続の publish で置き換えられた後に advanceFade 未配線のため retire されない」ことである。こちらは advanceFade() の Audio Callback 配線が完了するまで解消しない。

> **確定補足** 🔍: DSPTransition には publish 完了直後に oldDSP を retire する別経路が存在する（`DSPTransition::onPublishCompleted` → `lifetime.retire(oldDSP)`）。しかしログ上 `lifecycle(retire)=0` であることから、この経路でも `retireDSPHandleForRuntime()` が false を返しカウンタが更新されていないか、oldDSP 自体が nullptr であった可能性が高い。advanceFade 配線後にもう一度ログを確認し、lifecycle(retire) > 0 となることを確認する必要がある。

---

## 5. 各 DSPCore インスタンスのメモリ内訳（推定値）

**このセクションはすべて推定値である。実測値ではない。**

DIAG ログから得られたバッファサイズから計算した、1インスタンスあたりのおおよその内訳:

| コンポーネント | 推定サイズ | 根拠 |
|:-------------|:----------|:-----|
| EQ scratch | 32MB | 4,194,304 samples × 8B（ログから確定） |
| EQ msWorkBuffer | 16MB | 2,097,152 samples × 8B（ログから確定） |
| EQ dryBypass | 8MB | 1,048,576 samples × 8B（ログから確定） |
| EQ parallel/xfade 等 | ~12MB | **推定**（アドホック） |
| Internal aligned buffers | ~40MB | **推定**（10本 × 4MB） |
| Oversampling / SoftClip | ~50MB | **推定** |
| DC Blocker/Filters/etc | ~30MB | **推定** |
| TruePeak/etc | ~20MB | **推定** |
| CRT/STL | ~12MB | **推定** |
| **小計** | **~220MB** | **ほとんどが推定値** |

> **注意**: 上記のうちログから確定しているのは `EQ scratch(32MB)`、`EQ msWorkBuffer(16MB)`、`EQ dryBypass(8MB)` のみ。残りは全て推定。

> **6 インスタンス累積の計算も推定**: 6 × 220MB = ~1,320MB（アドホックな積み上げ）。内訳の配分（DSPCore ~1,320MB、JUCE ~500MB など）には実際の測定根拠がなく、コード上のバッファサイズから機械的に割り振ったものであることに留意。

---

## 6. JUCE Framework（全推定値）

JUCE のメモリ使用量は未測定。以下の数値は推定。

| カテゴリ | 推定値 | 根拠 |
|:---------|:-------|:-----|
| JUCE 全体 | ~500MB | **推定**。起動時 74MB から prepare 直後 555MB への増分の一部 |

JUCE が消費する可能性がある領域:
- グラフィックスレンダリングバッファ
- GUI コンポーネントツリー
- イベントループ、タイマー
- フォントキャッシュ、イメージキャッシュ

---

## 7. 未計装領域の候補: IR 処理一時バッファ（全推定値）

以下は `DIAG_MKL_MALLOC` の対象外で、OtherPrivate に含まれている。サイズは推定値、実測不可能。

| 候補 | 推定値 | 根拠 |
|:-----|:-------|:-----|
| IRConverter | ~200MB | **推定**（raw mkl_malloc, 未計装） |
| CacheManager | ~100MB | **推定** |
| MixedPhase | ~100MB | **推定**（186ms 処理に伴う一時確保） |
| ScopedAlignedPtr 一時 | ~?MB | **推定**（impulseForFft, tempTime, tempFreq 等） |

> **注意**: これらの値はすべて推定。実測には該当箇所を `DIAG_MKL_MALLOC` 化する追加計装が必要。

---

## 8. メモリ配分サマリ（全推定値）

**以下の内訳はすべて推定値である。実測確定値は MKL convolution の 35MB のみ。**

```
Private 2477MB の推定構成（全推定値、確定値は MKL のみ）:

  DSPCore 累積（commit後未解放）     = ~800MB (32%)  ★ 推定、主要因として極めて有力
  JUCE Framework                     = ~500MB (20%)   推定
  IR processing temp buffers         = ~400MB (16%)   推定
  CRT Heap / Threads / DLLs          = ~150MB (6%)    推定
  Build phase 一時バッファ           = ~100MB (4%)    推定
  Publication World objects          =  ~72MB (3%)    推定
  Convolution (tracked, 確定値!)     =   35MB (1.4%)  ☆ 確定
  Other (誤差)                       = ~420MB (17%)   推定
```

> **確定値は MKL Convolution 35MB のみ**。残りは OtherPrivate 2442MB に含まれており、現状の計装では分割不可能。

---

## 9. 対策案

| 優先度 | 対策 | 期待削減量 | 備考 |
|:-------|:-----|:----------|:-----|
| **P0** | **advanceFade() を Audio Callback に配線** | 確認後評価 | ISR の原則上必須。配線後に MEM_SNAP の liveCount/Private が減少するか確認 |
| **P1** | **初回 DSPCore の processingBlockSize=524288 の検証** | 要調査 | osFactor=0 時の異常値。影響範囲大のためアーキテクチャ判断が必要 |
| **P2** | **MEM_SNAP に liveCount 追加（StereoConvolver/DSPCore）** | — | 監視強化。配線後の効果確認に有用 |
| **P3** | **DSPCore 1個あたりのメモリコストの実測** | — | build.bat 以外の計装拡張か、ヒーププロファイラの使用を検討 |

**advanceFade 配線後に期待される効果（概算、未実証）**:

常時必要な DSPCore は 2 インスタンス（current + fading）。DSPCore 1個あたり ~220MB という**推定が正しいと仮定した場合**、解放される 4 インスタンスにより約 **~800MB** 程度の削減が期待される（2500MB → ~1700MB）。ただしこの数値は以下の理由から**過大または過小の可能性がある**:

- DSPCore 1個のコスト ~220MB は**推定値**であり、実際には異なる可能性
- advanceFade 配線後に retire が正常動作しても、DSPCore 以外の OtherPrivate 領域は変わらない
- JUCE や IR 処理バッファは retire と無関係

> **注意**: advanceFade() を正しく配線した後に、MEM_SNAP の `liveCount` と `Private` の変化を実測して初めて、効果が確定する。

---

## 10. 検証に使用したツール

| ツール | 使用目的 |
|:-------|:---------|
| grep / sed / awk (WSL) | ログ抽出、統計計算 |
| serena MCP | コードパスのトレース、型情報取得 |
| cocoindex-code (ccc.exe) | 関数間依存の grep |
| graphify (graphify.exe) | 依存関係グラフパス検索（DSPCore→AudioEngine 経路） |
| semble (semble.exe) | セマンティックコード検索 |
| AiDex MCP | コードインデックス検索、シンボル特定 |

---

## 11. 更新履歴

| 日付 | 更新内容 |
|:-----|:---------|
| 2026-07-09 | 初版作成 |
| 2026-07-09 | DSPCore が `aligned_make_unique` でヒープ割り当てされることを確認 |
| 2026-07-09 | advanceFade の Timer 配線否定的見解を反映。ISR 責務に基づく正しい配線先（Audio Callback）を特定 |
| 2026-07-09 | ISR 設計原則を追記。事実と推定の区別を明確化。削減見積もりを推定注釈付きで修正（~800MB削減、2500→~1700MB） |
| 2026-07-09 | [XFADE]=0 は SnapshotCoordinator とは別機構であることを明確化。重複セクションを削除。ISR 推奨 advanceFade 完了通知設計を追記。 |
