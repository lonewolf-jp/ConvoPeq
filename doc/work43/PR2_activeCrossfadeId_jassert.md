# PR2: activeCrossfadeId_ 単一前提の表明 — 詳細設計書

- **作成日**: 2026-06-17
- **ステータス**: 設計済み（未実装）
- **親計画**: `doc/work43/crossfade_refactoring_checklist.md`
- **前提**: PR1 完了済み

---

## 0. 目的

`activeCrossfadeId_` は単一 Crossfade を前提とした atomic 変数である。
現状の実装はこの前提を暗黙的に仮定しているが、明示的な表明がない。

本 PR では以下を追加する：

- `consumeAtomic(activeCrossfadeId_)` → `getActiveCrossfades()` に置換
- `!records.empty()` で安全なガード
- `jassert(records.size() == 1)` で単一前提を表明

`!records.empty()` を採用する理由:

- `fadeCompleted == true` かつ `records.empty()` は Crossfade Registry と Runtime の不整合を示唆する。
  `size() == 1` で黙殺するより `!records.empty()` + `jassert(size == 1)` で診断性を確保する。

`jassert(size == 1)` の意味:

- Debug: 複数 Crossfade 発生時に停止（size>=2）
- Release: jassert 消滅後も `records.front()` で先頭1件を処理。残りはタイムアウト回復へ委譲

これにより：

1. 単一前提が暗黙的でなくなり、設計文書として機能する
2. 複数 Crossfade が発生した場合もデバッグビルドで即検出できる
3. Release ビルドでも異常時は安全側（処理スキップ）に倒れる
4. 将来の複数対応時の改修箇所が明確になる

---

## 1. 変更ファイル一覧

| # | ファイル | 変更種別 |
|---|---------|---------|
| 1 | `src/audioengine/AudioEngine.Timer.cpp` | jassert + ガード追加 |
| 2 | `.github/scripts/isr-verify-phase4-generation-drift.ps1` | `DSPTransition.h` を検査対象に追加、DSPTransition チェック追加 |
| 3 | `src/audioengine/AudioEngine.h` | （変更なし） |

---

## 2. 変更詳細

### 2.1 AudioEngine.Timer.cpp — 完了検出経路 (L380〜398)

**変更前**:

```cpp
const bool fadeCompleted = m_coordinator.tryCompleteFade();
if (fadeCompleted)
{
    const auto completedId = convo::consumeAtomic(activeCrossfadeId_, std::memory_order_acquire);
    if (completedId != 0u)
    {
        crossfadeRuntime_.notifyFadeComplete(completedId);
        convo::isr::CompletedFadeEvent ev;
        if (crossfadeRuntime_.consumeCompletedFade(ev))
        {
            dspHandleRuntime_.endCrossfade(ev.id);
            crossfadeAuthorityRuntime_.unregisterCrossfade(ev.id);
        }
        convo::publishAtomic(activeCrossfadeId_, static_cast<convo::isr::CrossfadeId>(0u),
                             std::memory_order_release);
    }
    // ...
}
```

**変更後**:

```cpp
const bool fadeCompleted = m_coordinator.tryCompleteFade();
if (fadeCompleted)
{
    // ★ PR2: Authority の Registry から active crossfade を取得
    auto records = crossfadeAuthorityRuntime_.getActiveCrossfades();
    if (!records.empty())
    {
        // 単一 Crossfade 前提を表明（size=0 は正常、size>=2 は異常）
        jassert(records.size() == 1);
        if (records.size() > 1) {
            diagLog("[DIAG] Crossfade: multiple active crossfades detected (count="
                + juce::String(static_cast<int>(records.size()))
                + "), processing first only");
        }

        const auto xfadeId = records.front().id;
        crossfadeRuntime_.notifyFadeComplete(xfadeId);
        convo::isr::CompletedFadeEvent ev;
        if (crossfadeRuntime_.consumeCompletedFade(ev))
        {
            dspHandleRuntime_.endCrossfade(ev.id);
            crossfadeAuthorityRuntime_.unregisterCrossfade(ev.id);
        }
        convo::publishAtomic(activeCrossfadeId_, static_cast<convo::isr::CrossfadeId>(0u),
                             std::memory_order_release);
    }
    // ...
}
```

### 2.2 AudioEngine.Timer.cpp — タイムアウト回復経路 (L618〜626)

**変更前**:

```cpp
const auto activeId = convo::consumeAtomic(activeCrossfadeId_, std::memory_order_acquire);
if (activeId != 0u)
{
    crossfadeAuthorityRuntime_.unregisterCrossfade(activeId);
    convo::publishAtomic(activeCrossfadeId_, uint64_t{0}, std::memory_order_release);
}
crossfadeRuntime_.complete();
```

**変更後**:

```cpp
// ★ PR2: Authority の Registry から全 active レコードを取得
auto records = crossfadeAuthorityRuntime_.getActiveCrossfades();
jassert(records.size() <= 1);  // 複数 Crossfade 時はデバッグビルドで停止
if (records.size() > 1) {
    diagLog("[DIAG] Crossfade: multiple active crossfades detected (count="
        + juce::String(static_cast<int>(records.size()))
        + "), clearing all via timeout recovery");
}
for (const auto& record : records)
    crossfadeAuthorityRuntime_.unregisterCrossfade(record.id);
crossfadeRuntime_.complete();
```

**変更理由**:

- タイムアウト回復は「すべての進行中 Crossfade を強制終了」するセマンティクス
- そのため `getActiveCrossfades()` の全件ループが適切（完了検出とロジックが異なる）
- `jassert` は監視用。複数発生時もループで安全に処理

---

## 3. 確認済み事項（調査結果）

### 3.1 getActiveCrossfades() 戻り値型

- `std::vector<CrossfadeRecord>` 値返し
- `CrossfadeRecord` 構造体サイズ: `{uint32_t id, DSPHandle from, DSPHandle to, uint64_t startEpoch, bool active}` ≈ 24〜32 bytes
- 現在の単一 Crossfade 前提では 0〜1 件のコピー。Timer コールバック (~30ms) への影響は無視可能

### 3.2 SPSCRingBuffer (CompletedFadeEvent キュー)

- `src/core/CommandBuffer.h` に実装
- ロックフリー、acquire/release セマンティクス
- 64-byte cache-line alignment（false sharing 防止）
- 容量 32、power-of-2 制約充足
- `CompletedFadeEvent` は trivially copyable（static_assert 確認済み）
- RT-safe 確認済み

### 3.3 gen-drift スクリプトの現状問題と更新計画

**現状の問題:**

- 現行スクリプトは `AudioEngine.Commit.cpp` を検査対象としている
- PR1 により crossfade 開始ロジックは `DSPTransition.h` に移動
- チェック2-a (`beginCrossfade` 呼び出し) は `DSPTransition.h` を検査すべきだが `commitCpp` を見ている → **PR2 時点で PASS しない**
- チェック2-b (`publishAtomic(activeCrossfadeId_, crossfadeId`) も `commitCpp` を見ている → 同上

**PR2 での更新内容:**

不要なチェックは削除せず、まず `DSPTransition.h` を検査対象に追加する。
`commitCpp` に対するチェックは PR4 まで残しておく（activeCrossfadeId_ が存在する間は正当）。

```powershell
# --- PR2 での変更 ---

# 1. 検査対象ファイルに DSPTransition.h を追加
$requiredFiles['dspTransition'] = Join-Path $audioRoot 'DSPTransition.h'

# 2. DSPTransition.h の内容を取得
$dspTransition = $fileContents['dspTransition']

# 3. DSPTransition.h に対するチェックを追加
#    チェック2-a の DSPTransition 版: beginCrossfade 呼び出し
if ($dspTransition -notmatch 'dspHandleRuntime_\.beginCrossfade\s*\(') {
    $violations.Add('Phase4 drift gate: beginCrossfade call missing in DSPTransition.')
}
#    チェック2-b の DSPTransition 版: activeCrossfadeId_ publish
if ($dspTransition -notmatch 'publishAtomic\(.*activeCrossfadeId_,') {
    $violations.Add('Phase4 drift gate: activeCrossfadeId publish on crossfade start missing in DSPTransition.')
}
```

**PR4 での更新内容:**

- `commitCpp` に対する全チェックを削除（activeCrossfadeId_ 消滅）
- DSPTransition のチェックを `activeCrossfadeId_ publish` から `getActiveCrossfades()` 連携に変更
- Timer チェックを `consumeAtomic(activeCrossfadeId_)` から `getActiveCrossfades()` + SPSC round-trip に変更
- `crossfadeRecords_` 維持チェックを追加

### 3.4 PR1 事後検証（差分確認）

- `DSPHandleRuntime::beginCrossfade` → `(from, to, id)` 外部注入 ✅
- `nextCrossfadeId_` 削除 ✅
- `DSPTransition.h` → `registerCrossfade` + `beginCrossfade(from, to, xfadeId)` ✅
- `DSPLifetimeManager.h` → 引数追加 ✅

---

## 4. 変更しないもの

| 項目 | 理由 |
|------|------|
| `activeCrossfadeId_` 宣言 | PR4 で削除するまで維持 |
| `DSPTransition.h` の `publishAtomic(activeCrossfadeId_, xfadeId)` | PR4 で削除するまで維持 |
| `ReleaseResources.cpp` の参照 | PR4 で一括削除 |
| `endCrossfade(CrossfadeId id)` のシグネチャ | 変更なし |
| `CrossfadeRuntime` | 変更なし |

---

## 4. 検証項目

- [ ] デバッグビルドが通ること（`jassert` は Debug 限定）
- [ ] 通常の Crossfade 完了経路で `jassert` が発火しないこと
- [ ] タイムアウト回復経路で全 active レコードが unregister されること
- [ ] Release ビルドで `jassert` が消滅し、動作に影響しないこと
- [ ] CI ゲート（generation-drift）が PASS すること

---

## 5. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| `jassert` が Release ビルドで消滅 | 監視機能喪失 | CI のデバッグビルドで検出。本番影響なし |
| `getActiveCrossfades()` が 0 件を返す | 完了処理スキップ | `if (records.size() == 1)` でガード。0 件なら何もしない |
| タイムアウト回復で全件ループが多すぎる | 非効率 | 現状は高々1件。複数時もループは即完了 |

---

## 6. 事後状態

```
完了検出:
  Authority.getActiveCrossfades() → [{id=17, ...}]
  jassert(size <= 1)              ← 単一前提を表明
  endCrossfade(17) / unregister(17)

タイムアウト回復:
  Authority.getActiveCrossfades() → [{id=17, ...}]
  jassert(size <= 1)              ← 単一前提を表明
  unregister(17) / complete()
```
