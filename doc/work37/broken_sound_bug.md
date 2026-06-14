## ログ解析結果

添付のログを詳細に解析しました。以下に内部状態の推定結果を報告します。

---

### 全体のフェーズ

ログは以下の3フェーズに明確に分割できます：

| フェーズ | 概要 | 状態 |
|---|---|---|
| **Phase 1** (開始〜`convolverParamsChanged`) | 初期化・DSP準備完了 | ✅ 正常 |
| **Phase 2** (transition) | Retire Stall 発生 → **全Rebuild抑制** | ⚠️ 破綻 |
| **Phase 3** (以降〜シャットダウン) | 全Snapshot抑制 + NoiseShaper学習継続 | ❌ 音声劣化 |

---

### Phase 1: 正常動作（開始〜各種Prepare完了）

- サンプリングレート: 48kHz → 192kHz（`prepareToPlay`）
- Oversampling: 4x → 処理レート 768kHz → 最終的に **2x（384kHz）** に変更
- 複数世代の `REBUILD` が正常に `DISPATCHED` される
- DSPCore の各モジュール（Ramp, Oversampling, Convolver, EQ, DC Blocker, NoiseShaper, OutputFilter）が正常に `prepare` 完了
- Runtime Publish が `rev=4` まで進行

### Phase 2: Retire Stall 発生（破綻の起点）

**最も重要なイベント**:

```
[HEALTH] eventCode=1011 severity=2 value=4572875800036643
[HEALTH] Retire stall detected, throttling rebuild and forcing reclaim
```

`Retire stall` とは、**エポックベースのDSPリソース retirement（旧設定の廃棄）が停止**した状態です。

**tx counters が示す異常**:
```
lifecycle(pub/ret/reclaim)=4/0/1
```
- `pub=4` → 4回のパブリッシュが行われた
- **`ret=0`** → **1度も retirement が完了していない** ← 異常
- `reclaim=1` → 強制リクレームが1回走った

**結果**: Retire圧力が `severe` と判定され、以降の**すべて**の `REBUILD` が抑制されます。

### Phase 3: 全Rebuild抑制 + コンボルバIR未適用

**抑制されたイベント一覧**:

| intentId | 理由 | 結果 |
|---|---|---|
| 14,15 | Snapshot | SUPPRESSED (retire_pressure_severe) |
| 16 | 構造的リビルド（Convolver IR変更） | SUPPRESSED |
| 17 | UI EQ変更 | SUPPRESSED |
| 19,24-30 | Snapshot（全定期） | SUPPRESSED |
| 23 | **Deferred Structural rebuild release** | SUPPRESSED |

**特に重大**: intentId 18 の `convolver_params_changed` は **DEFERRED**（準備済みIR適用ウィンドウ待ち）となり、intentId 22 で `released` された後、intentId 23 で **SUPPRESSED** されました。つまり:

> **`applyComputedIR: applied scaleFactor=0.132589` のIR変更は計算されたが、DSP実行系にデプロイされなかった。**

`toNull=-1`（有効な遷移先がない）という状態が確定し、システムは**古いDSP構成で動作し続ける**ことになります。

---

### NoiseShaper Learner の状態

```
startNoiseShaperLearning with mode=5, resume=true
```

- **mode=5**: 適応型ノイズシェーピングのフル学習モード
- Learner は `iter=0` から `iter=1150` まで継続稼働
- `bufferedSamples=3840000`（FIFO最大容量）で張り付いている → **Producer = Audio Thread が Consumer = Learner より高速**
- `accepted` 値の変動（3016 → 610812 → 212776 → 449836）は学習の進行段階を示す

### シャットダウン時の問題

```
[ISR][Shutdown] Drain incomplete: routerPendingRetire=2
```

- `routerPendingRetire=2` — ルーターに2つの保留retireが残っている
- **Drain timeout reached** — 強制 drain がタイムアウト
- `oldestAgeMs=557` — 最も古い保留が557ms放置

---

### 「途中から音がおかしくなる」の推定原因

複合的な要因が重なっています：

| # | 原因 | メカニズム |
|---|---|---|
| **①** | **Retire Stall によるSnapshot凍結** | 新DSP構成が一切デプロイ不可に。Convolver IR変更、EQ変更が未適用 |
| **②** | **適応型NoiseShaper学習中の音質劣化** | 学習途中（iter 0〜1150）のノイズシェーピング係数は収束前で、**ノイズ特性が不安定** |
| **③** | **FIFOバッファフル状態持続** | `bufferedSamples=3840000` 張り付き → リアルタイム処理にバックプレッシャー |
| **④** | **Oversampling倍率変更の未適用** | processingRate 768kHz→384kHz 変更が、Snapshot抑制により切り替わらず不整合 |

**総合推定**: 音声破綻の直接的原因は **②（NoiseShaper学習中の適応型フィルタ係数変動）** で、**①（Retire Stall）がそれを回復不能に固定化**したため劣化状態が継続・悪化したと考えられます。

---

### 推奨される調査・修正ポイント

1. **Retire Stall の根本原因調査**
   - Epoch advanceが停止した理由 → 何が epoch の完了をブロックしているか
   - `routerPendingRetire=2` の正体 → どのDSP状態がルーターでスタックしているか
   - `oldestAgeMs=557` → 557msもretireされなかった理由

2. **NoiseShaperの学習開始条件の再検討**
   - `mode=5` + `resume=true` の学習完了条件を確認
   - 学習中のDSP状態が不整合でも学習を続行してよいか（ガード条件追加）

3. **Rebuild抑制からの回復パス**
   - 現在 `retire_pressure_severe` は抑制のみで回復ロジックがない
   - 強制epoch advance または 強制reclaim のタイムアウト再試行機構を追加

4. **Snapshot枯渇時のフォールバック**
   - Snapshotが長時間抑制された場合、最後の既知正常状態を維持する明示的なハンドリングが必要