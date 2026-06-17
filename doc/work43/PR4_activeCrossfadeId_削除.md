# PR4: activeCrossfadeId_ 削除（CompletedFadeEvent SPSC 維持版） — 詳細設計書

- **作成日**: 2026-06-17
- **ステータス**: 設計済み（未実装）
- **親計画**: `doc/work43/crossfade_refactoring_checklist.md`
- **前提**: PR2 完了済み（Timer が `getActiveCrossfades()` で ID を取得するようになった状態）
- **改訂履歴**: v5 — SPSC 経由を維持。`activeCrossfadeId_` のみ削除。`CompletedFadeEvent` 設計資産を活用。

---

## 0. 目的

PR2 により Timer は `getActiveCrossfades()` で Crossfade ID を取得するようになった。
これにより `activeCrossfadeId_` は冗長になった。PR4 では `activeCrossfadeId_` のみを削除する。

**重要な設計判断**: `CompletedFadeEvent` SPSC は削除せず維持する。
理由:

1. SPSC は将来 AudioThread→Timer 完了通知経路のための設計資産
2. `notifyFadeComplete` / `consumeCompletedFade` / `completedFadeQueue_` / `crossfadeEventDropCount_` は既に実装済み
3. Timer 完了経路では `getActiveCrossfades()` で取得した ID を SPSC に流し、即消費する（同一スレッド round-trip）
4. Authority (`getActiveCrossfades()`) は ID 取得源として使い、完了検出権威には昇格させない
   — 完了検出は `CrossfadeRuntime` の責務

| 役割 | 移行先 |
|------|--------|
| Timer への完了 ID 伝達 | `getActiveCrossfades().front().id` |
| DSPTransition での開始記録 | `registerCrossfade()` の戻り値 |
| shutdown 時の進行中 Crossfade 有無確認 | `CrossfadeRuntime::isPending()` |
| タイムアウト回復時の ID 特定 | `getActiveCrossfades()` 全件ループ |

---

## 1. 変更ファイル一覧

| # | ファイル | 変更種別 |
|---|---------|---------|
| 1 | `src/audioengine/AudioEngine.h` | `activeCrossfadeId_` 宣言削除（L3608） |
| 2 | `src/audioengine/DSPTransition.h` | `publishAtomic(activeCrossfadeId_)` 削除（L82-86） |
| 3 | `src/audioengine/AudioEngine.Timer.cpp` | 完了経路: SPSC round-trip 除去、直接 `endCrossfade/unregisterCrossfade` 化 |
| 4 | `src/audioengine/AudioEngine.Timer.cpp` | タイムアウト経路: `publishAtomic(0)` 削除 |
| 5 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | 3箇所の参照削除 |
| 6 | `.github/scripts/isr-verify-phase4-generation-drift.ps1` | チェック全面更新 |

---

## 2. 変更詳細

### 2.1 AudioEngine.h — 宣言削除 (L3608)

```cpp
// 削除:
std::atomic<convo::isr::CrossfadeId> activeCrossfadeId_{ 0u };
```

### 2.2 DSPTransition.h — publishAtomic 削除 (L82-86)

```cpp
// 変更前:
const auto xfadeId = engine_.crossfadeAuthorityRuntime_.registerCrossfade(oldHandle, newHandle);
engine_.dspHandleRuntime_.beginCrossfade(oldHandle, newHandle, xfadeId);
engine_.publishAtomic(engine_.activeCrossfadeId_,          // ★ 削除
                     static_cast<convo::isr::CrossfadeId>(xfadeId),
                     std::memory_order_release);

// 変更後:
const auto xfadeId = engine_.crossfadeAuthorityRuntime_.registerCrossfade(oldHandle, newHandle);
engine_.dspHandleRuntime_.beginCrossfade(oldHandle, newHandle, xfadeId);
// activeCrossfadeId_ publish 削除 — SPSC が唯一の通知経路
```

### 2.3 Timer 完了検出経路 — SPSC round-trip 維持、activeCrossfadeId_ のみ削除

**変更前** (PR2 適用後):

```cpp
if (!records.empty())
{
    jassert(records.size() == 1);
    const auto xfadeId = records.front().id;
    crossfadeRuntime_.notifyFadeComplete(xfadeId);       // SPSC push
    convo::isr::CompletedFadeEvent ev;
    if (crossfadeRuntime_.consumeCompletedFade(ev))      // SPSC pop
    {
        dspHandleRuntime_.endCrossfade(ev.id);
        crossfadeAuthorityRuntime_.unregisterCrossfade(ev.id);
    }
    convo::publishAtomic(activeCrossfadeId_, static_cast<convo::isr::CrossfadeId>(0u),
                         std::memory_order_release);     // ★ 削除
}
```

**変更後**:

```cpp
if (!records.empty())
{
    jassert(records.size() == 1);
    const auto xfadeId = records.front().id;
    // ★ PR4: SPSC round-trip 維持（将来 AudioThread 通知用の設計資産）
    crossfadeRuntime_.notifyFadeComplete(xfadeId);       // SPSC push
    convo::isr::CompletedFadeEvent ev;
    while (crossfadeRuntime_.consumeCompletedFade(ev))   // SPSC pop（全件消費）
    {
        dspHandleRuntime_.endCrossfade(ev.id);
        crossfadeAuthorityRuntime_.unregisterCrossfade(ev.id);
    }
    // ★ activeCrossfadeId_ publish 削除 — 変数自体が消滅
}
```

**変更点**:

- `publishAtomic(activeCrossfadeId_, 0)` 削除（変数消滅のため）
- `if` → `while` で SPSC キュー内の全イベントを消費
- SPSC round-trip は維持（`notifyFadeComplete` → `consumeCompletedFade`）
- ID は `getActiveCrossfades()` から取得（PR2 の変更を継承）

### 2.4 Timer タイムアウト回復経路 — publishAtomic 削除

**変更後** (PR2 適用後から `publishAtomic` のみ削除):

```cpp
auto records = crossfadeAuthorityRuntime_.getActiveCrossfades();
jassert(records.size() <= 1);
if (records.size() > 1) { /* diagLog */ }
for (const auto& record : records)
    crossfadeAuthorityRuntime_.unregisterCrossfade(record.id);
// ★ PR4: activeCrossfadeId_ は存在しない。Authority の records_ が既にクリア済み。
crossfadeRuntime_.complete();
```

### 2.5 ReleaseResources.cpp — 3箇所の参照削除

**(1) EmergencyDrain (L247):**

```cpp
// 変更前:
crossfadeRuntime_.reset();
convo::publishAtomic(activeCrossfadeId_, uint64_t{0}, std::memory_order_release);

// 変更後:
crossfadeRuntime_.reset();  // reset() が pending を含む全状態をクリア
```

**(2) VerifyDrained (L284):**

```cpp
// 変更前:
// ... retire/reclaim ...
convo::publishAtomic(activeCrossfadeId_, static_cast<convo::isr::CrossfadeId>(0u), ...);

// 変更後:
// publishAtomic 行を削除。既に retire/reclaim 済みで、activeCrossfadeId_ は存在しない
```

**(3) shutdown counter (L378):**

```cpp
// 変更前:
const auto activeCrossfadeCount = consumeAtomic(activeCrossfadeId_, std::memory_order_acquire)
    != static_cast<convo::isr::CrossfadeId>(0u) ? 1u : 0u;

// 変更後:
const auto activeCrossfadeCount = crossfadeRuntime_.isPending() ? 1u : 0u;
```

### 2.7 Gen-Drift Script — 全面更新

#### 更新方針

| 変更 | 内容 |
|------|------|
| 検査対象ファイル追加 | `DSPTransition.h` (PR2 で追加済み) |
| 検査対象ファイル維持 | `ISRDSPHandle.cpp`, `AudioEngine.Timer.cpp`, `AudioEngine.h` |
| 検査対象ファイル削除 | `AudioEngine.Commit.cpp` (crossfade 開始ロジックが `DSPTransition.h` に移動したため) |
| 削除するチェック | `activeCrossfadeId_` 関連の全チェック |
| 追加するチェック | Authority→DSPHandle ID 連携、SPSC round-trip、`crossfadeRecords_` 維持 |

#### 完全版スクリプト

```powershell
# ファイル: .github/scripts/isr-verify-phase4-generation-drift.ps1

# --- PR2 で追加済み ---
$requiredFiles = [ordered]@{
    dspHandleCpp  = (Join-Path $audioRoot 'ISRDSPHandle.cpp')
    dspTransition = (Join-Path $audioRoot 'DSPTransition.h')    # ★ 追加 (PR2)
    timerCpp      = (Join-Path $audioRoot 'AudioEngine.Timer.cpp')
    audioHeader   = (Join-Path $audioRoot 'AudioEngine.h')
}
# commitCpp (AudioEngine.Commit.cpp) は削除

# --- PR4 で全面更新するチェック ---
# ★ 削除: チェック1 (stale generation) — 維持
# ★ 削除: チェック2-a (commitCpp の beginCrossfade) → DSPTransition 版に置換
# ★ 削除: チェック2-b (commitCpp の activeCrossfadeId_ publish) → 消滅
# ★ 削除: チェック3-a (timer の consumeAtomic(activeCrossfadeId_)) → getActiveCrossfades に置換
# ★ 削除: チェック3-d (timer の publishAtomic(activeCrossfadeId_, 0)) → 消滅
# ★ 削除: チェック4 (header の activeCrossfadeId_ 宣言) → 消滅

# 1) stale generation handle は reject 必須（維持）
if ($null -ne $dspHandleCpp) {
    if ($dspHandleCpp -notmatch 'currentGen\s*!=\s*handle\.generation') {
        $violations.Add('Phase4 drift gate: stale generation mismatch check missing in DSPHandleRuntime::resolve.')
    }
}

# 2) Crossfade 開始: DSPTransition で registerCrossfade + beginCrossfade 連携
if ($null -ne $dspTransition) {
    if ($dspTransition -notmatch 'crossfadeAuthorityRuntime_\.registerCrossfade\(') {
        $violations.Add('Crossfade gate: registerCrossfade call missing in DSPTransition.')
    }
    if ($dspTransition -notmatch 'dspHandleRuntime_\.beginCrossfade\(') {
        $violations.Add('Crossfade gate: beginCrossfade call missing in DSPTransition.')
    }
    # ★ activeCrossfadeId_ への publish が存在しないことを確認
    if ($dspTransition -match 'activeCrossfadeId_') {
        $violations.Add('Crossfade gate: activeCrossfadeId_ reference must be removed from DSPTransition.')
    }
}

# 3) Crossfade 完了: Timer で Authority → SPSC → endCrossfade/unregisterCrossfade
if ($null -ne $timerCpp) {
    # 完了検出: Authority から ID を取得
    if ($timerCpp -notmatch 'getActiveCrossfades\(\)') {
        $violations.Add('Crossfade gate: timer must use getActiveCrossfades() for completion.')
    }
    # SPSC round-trip
    if ($timerCpp -notmatch 'notifyFadeComplete\(') {
        $violations.Add('Crossfade gate: timer must notifyFadeComplete to SPSC.')
    }
    if ($timerCpp -notmatch 'consumeCompletedFade\(') {
        $violations.Add('Crossfade gate: timer must consumeCompletedFade from SPSC.')
    }
    # 状態遷移
    if ($timerCpp -notmatch 'dspHandleRuntime_\.endCrossfade\s*\(') {
        $violations.Add('Crossfade gate: timer must call endCrossfade.')
    }
    if ($timerCpp -notmatch 'crossfadeAuthorityRuntime_\.unregisterCrossfade\s*\(') {
        $violations.Add('Crossfade gate: timer must unregister crossfade authority entry.')
    }
    # ★ activeCrossfadeId_ が Timer から完全に消えていること
    if ($timerCpp -match 'activeCrossfadeId_') {
        $violations.Add('Crossfade gate: activeCrossfadeId_ must be removed from Timer.cpp.')
    }
}

# 4) activeCrossfadeId_ が全ソースコードから消滅したこと
$activeRefCount = (Select-String -Path "$audioRoot\*.cpp", "$audioRoot\*.h" -Pattern 'activeCrossfadeId_' | Measure-Object).Count
if ($activeRefCount -gt 0) {
    $violations.Add("Crossfade gate: activeCrossfadeId_ still referenced in $activeRefCount places.")
}

# 5) crossfadeRecords_ は維持されていること（削除禁止）
if ($null -ne $dspHandleCpp) {
    if ($dspHandleCpp -notmatch 'crossfadeRecords_') {
        $violations.Add('Crossfade gate: crossfadeRecords_ must be preserved in DSPHandleRuntime.')
    }
}
```

```

---

## 3. 削除されるもの

| 項目 | 削除理由 |
|------|---------|
| `AudioEngine::activeCrossfadeId_` | `getActiveCrossfades()` で代替 |
| `DSPTransition::publishAtomic(activeCrossfadeId_, xfadeId)` | Authority の `records_` が唯一情報源 |
| Timer 完了経路の `publishAtomic(activeCrossfadeId_, 0)` | 変数消滅により不要 |
| Timer タイムアウト経路の `publishAtomic(activeCrossfadeId_, 0)` | 同上 |
| ReleaseResources の `publishAtomic(0) x2` | `reset()` で十分 |
| ReleaseResources の `consumeAtomic() != 0` | `isPending()` で代替 |

---

## 4. 維持されるもの

| 項目 | 状態 | 理由 |
|------|------|------|
| `CompletedFadeEvent` 構造体 | アクティブ | Timer 完了経路で SPSC round-trip に使用 |
| `SPSCRingBuffer<CompletedFadeEvent, 32>` | アクティブ | 同上 |
| `notifyFadeComplete()` / `consumeCompletedFade()` | アクティブ | Timer 完了経路で push/pop に使用 |
| `getActiveCrossfades()` | アクティブ | Timer 完了検出の ID 取得源 |
| `crossfadeRecords_` (DSPHandleRuntime) | アクティブ | 複数 Crossfade 対応 |
| `CrossfadeAuthorityRuntime::records_` | アクティブ | Registry |
| `endCrossfade(CrossfadeId id)` | アクティブ | ID→状態遷移の契約 |

---

## 4.5 調査で確定した事項

### getActiveCrossfades() 戻り値型

- `CrossfadeAuthorityRuntime::getActiveCrossfades()` は `std::vector<CrossfadeRecord>` 値返し
- `CrossfadeRecord`: `{uint32_t id, DSPHandle from(2×uint32_t), DSPHandle to(2×uint32_t), uint64_t, bool}` ≈ 32 bytes
- 現在の単一 Crossfade 前提では 0〜1 件のコピー、Timer 周期への影響は無視可能
- 将来複数対応時の最適化案: `SmallVector` / `std::array` / `forEachActiveCrossfade(fn)` コールバック

### SPSCRingBuffer (CompletedFadeEvent キュー)

- 実装: `src/core/CommandBuffer.h` — ロックフリー SPSC, acquire/release セマンティクス, 64-byte alignment
- 容量: 32 エントリ（power-of-2）, `CompletedFadeEvent` は trivially copyable ✅
- `crossfadeEventDropCount_` で溢れ監視 → HealthMonitor で参照 ✅
- RT-safe 確認済み ✅

### gen-drift スクリプトの更新必要性

- 現行スクリプトは `AudioEngine.Commit.cpp` を検査 — PR1 により crossfade 開始は `DSPTransition.h` に移動
- PR4 では以下のチェックに全面更新が必要:
  1. DSPTransition: `registerCrossfade` + `beginCrossfade(from, to, id)` 連携
  2. Timer: `getActiveCrossfades()` + `notifyFadeComplete()` + `consumeCompletedFade()` + `endCrossfade()` + `unregisterCrossfade()`
  3. `activeCrossfadeId_` 全ソースから消滅
  4. `crossfadeRecords_` 維持確認

### activeCrossfadeId_ 全12参照の内訳

| 箇所 | 行 | PR4 での処置 |
|------|-----|-------------|
| `AudioEngine.h` | 3608 | 宣言削除 |
| `AudioEngine.Timer.cpp` | 386 | `consumeAtomic` → `getActiveCrossfades()` (PR2) |
| `AudioEngine.Timer.cpp` | 397 | `publishAtomic(0)` 削除 |
| `AudioEngine.Timer.cpp` | 618 | `consumeAtomic` → `getActiveCrossfades()` (PR2) |
| `AudioEngine.Timer.cpp` | 622 | `publishAtomic(0)` 削除 |
| `ReleaseResources.cpp` | 247 | `publishAtomic(0)` 削除 |
| `ReleaseResources.cpp` | 284 | `publishAtomic(0)` 削除 |
| `ReleaseResources.cpp` | 378 | `consumeAtomic` → `isPending()` |
| `CrossfadeRuntime.h` | 50 | コメント（変更不要） |
| `CrossfadeRuntime.h` | 170 | コメント（変更不要） |
| `DSPTransition.h` | 83-84 | `publishAtomic(xfadeId)` 削除 |

### CrossfadeAuthorityRuntime::records_ の状態

- `registerCrossfade`: DSPTransition.h のみから呼ばれる ✅（CI gate 充足）
- `unregisterCrossfade`: Timer 完了経路 + タイムアウト経路からのみ ✅
- 唯一の CrossfadeId 発行源 ✅（PR1 で確立）

---

## 5. 検証項目

- [ ] ビルドが通ること（Debug + Release）
- [ ] `activeCrossfadeId_` が全ソースコードから消滅したこと
- [ ] 通常の Crossfade 完了経路が正しく動作すること
- [ ] タイムアウト回復が正しく動作すること
- [ ] shutdown が正しく動作すること
- [ ] generation-drift スクリプトが PASS すること
- [ ] CI ゲートが PASS すること

---

## 6. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| `activeCrossfadeId_` 参照が残存 | コンパイルエラー | 宣言削除により C++ コンパイラが検出 |
| `getActiveCrossfades()` が空を返す | 完了処理スキップ | `size()==1` ガードで 0 件時は何もしない（正常動作） |
| `isPending()` と Authority の records_ の不整合 | shutdown カウンタ誤差 | `isPending()` は `pending_` atomic の直接読み取り。Authority の records_ より軽量 |

---

## 7. 事後状態（最終形）

```

DSPTransition (commit):
  registerCrossfade(from, to) → id (Authority 唯一権威)
  beginCrossfade(from, to, id) (DSPHandleRuntime 状態遷移)
  ★ activeCrossfadeId_ publish なし（削除）

Timer 完了検出:
  tryCompleteFade() → true
  getActiveCrossfades() → [{id=17, ...}]
  jassert(size == 1)                    ← 単一前提表明
  notifyFadeComplete(17)                → SPSC push
  while (consumeCompletedFade(ev))      ← SPSC pop（全件）
      endCrossfade(ev.id)
      unregisterCrossfade(ev.id)
  ★ activeCrossfadeId_ なし

Timer タイムアウト回復:
  getActiveCrossfades() → [{id=17, ...}]
  unregisterCrossfade(17) (for)
  complete()
  ★ activeCrossfadeId_ なし

ReleaseResources:
  crossfadeRuntime_.reset()
  isPending() → activeCrossfadeCount
  ★ activeCrossfadeId_ なし

CrossfadeRuntime:
  CompletedFadeEvent SPSC — アクティブ使用
  notifyFadeComplete(id) → Timer 完了経路から呼ばれる
  consumeCompletedFade(ev) → Timer 完了経路で消費
  ★ 将来 AudioThread が notifyFadeComplete を呼べば Timer 側の変更不要

```

  getActiveCrossfades() で全 active レコード取得
  unregister (for ループ)
  ★ activeCrossfadeId_ なし

```
