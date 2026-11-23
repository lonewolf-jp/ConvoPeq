# reclaim ドライバ改訂設計 v3（実装未承認・設計のみ）

Owner 指摘（2026-11-11）の 3 点（滞留判定の lock-free 化、Message Thread 制約の明確化、publish 理由の実経路への一致）を反映した。**製品コードは変更していない（実装未着手）。**

## §1 滞留判定の修正（v2 §1.2 の置き換え）

### 1.1 何が問題だったか

v2 は `quarantineResidentCount()` と `terminalReclaimResidentCount()` を使う設計だったが、両者は **mutex を取る**。

| API | 実装 | ロック |
|---|---|---|
| `RetireQuarantineStore::residentCount()` | `std::lock_guard<std::mutex> lock(mtx_); return size_;`（`RetireQuarantineStore.h:194-198`） | **取る** |
| `RetireQuarantineStore::residentCountAtomic()` | `consumeAtomic(residentAtomic_)` | 取らない |
| `ISRRetireRouter::quarantineResidentCount()` | `m_retireQuarantine.residentCount() + m_emergencyQuarantine.residentCount()`（`ISRRetireRouter.cpp:673-677`） | **取る** |
| `ISRRetireRouter::terminalReclaimResidentCount()` | `m_terminalReclaim.residentCount()`（`ISRRetireRouter.cpp:781-784`） | **取る** |
| `ISRRetireRouter::residentCountAtomic()` | Q + EmergencyQ + Terminal の atomic 合算（`ISRRetireRouter.h:428-433`） | **取らない** |

タイマー（Message Thread）から毎 100 ms に mutex を 3 回取るのは避けるべきであり、既存の lock-free 集約 API がある。

### 1.2 修正後のコード（設計）

```cpp
if (m_ownedRetireRouter.pendingRetireCount() > 0
    || m_ownedRetireRouter.residentCountAtomic() > 0)
{
    m_ownedRetireRouter.tryReclaim();
}
```

- `pendingRetireCount()` は D キュー滞留（atomic）。
- `residentCountAtomic()` は Q + EmergencyQ + Terminal の滞留（atomic 3 つの合算）。
- したがって **mutex もロックも取らない**。新しい authority・カウンタは追加しない。
- これで「D に滞留」「Q/E/T に滞留」の両方を 1 回の判定で確認できる。

## §2 Message Thread 制約の明確化

`jassert(!isAudioThread())` は「Audio Thread ではない」ことしか示さない。新設メソッドは public なので、将来の誤用を検出するため **Message Thread 固有の assertion** を併置する。

```cpp
void driveRetireReclaimNonRt() noexcept
{
    jassert (juce::MessageManager::getInstanceWithoutCreating() != nullptr
             && juce::MessageManager::getInstanceWithoutCreating()->isThisTheMessageThread());

    if (convo::exchangeAtomic (m_epochAdvancePending, false, std::memory_order_acq_rel))
    {
        m_epochDomain->publishEpoch();
        m_ownedRetireRouter.tryReclaim();
        return;
    }

    if (m_ownedRetireRouter.pendingRetireCount() > 0
        || m_ownedRetireRouter.residentCountAtomic() > 0)
    {
        m_ownedRetireRouter.tryReclaim();
    }
}
```

- `isThisTheMessageThread()` は JUCE の公開 API。`getInstanceWithoutCreating()` で MessageManager 未生成時の null も扱う。
- 呼出し側（`AudioEngine::timerCallback`）にも `!isShutdownInProgress()` と `!isAudioThread()` を残す（二重の防御）。

## §3 publish 理由の分類（実経路に一致させる）

### 3.1 現在のツリーで確認した事実

| 項目 | 確認結果 | 根拠 |
|---|---|---|
| EQ ドメインの publish 箇所 | **1 か所のみ**（`m_epochDomain->publishEpoch()` は `flushPendingEpochAdvance()` 内） | `grep -rn "m_epochDomain->publishEpoch()" src` → `EQProcessor.Core.cpp:87` のみ |
| `flushPendingEpochAdvance()` の呼出し | **2 か所**: `releaseResources()`（`Core.cpp:231`）、`prepareToPlay()`（`Core.cpp:790`） | 関数定義の帰属を確認 |
| `resetToDefaults()` | **呼出し元ではない**（このツリー）。`resetToDefaults()` は `Core.cpp:238` に定義され、`retireEQStateDeferred` は呼ぶが flush は呼ばない | `grep "flushPendingEpochAdvance();"` で 231 と 790 のみ |

> Owner の 11/8 スナップショットでは `resetToDefaults()` が呼出し元として見えた可能性があるが、**現在の HEAD ツリーでは 2 か所**である。実装時はこの版で再確認する。

### 3.2 計測の設計（理由別カウンタ）

publish は 1 か所なので、**理由＝どの入口が pending を消費したか**で分類する。

| 理由 | 入口 |
|---|---|
| `timerPending` | 新設 `driveRetireReclaimNonRt()`（タイマー） |
| `prepare` | `prepareToPlay()` → `flushPendingEpochAdvance()` |
| `release` | `releaseResources()` → `flushPendingEpochAdvance()` |

実装（計測時のみ）: 各入口で理由タグを立て、`flushPendingEpochAdvance()` 内の exchange 成功時に `publishReasonCount[reason]` を 1 増やす。`flushPendingEpochAdvance()` の**挙動は変えない**（タグは計測用の `std::atomic<uint8_t>` 1 個）。

- 理由タグは「最後に入口を通過した理由」であり、同時実行は無い（Message Thread のみ）ため競合しない。
- `flushPendingEpochAdvance()` を直接呼ぶ第 3 の経路が将来増えた場合は `reason = other` とし、**未分類を 0 にしない**。

### 3.3 同一実行計測の 9 項目（v2 §2 を維持し、理由分類を上表に置換）

| # | 項目 | 変更 |
|---|---|---|
| 2 | publish 回数（理由別） | `timerPending` / `prepare` / `release` / `other` の 4 区分（`resetToDefaults` は本ツリーでは存在しない） |
| その他 1,3〜9 | v2 §2 のとおり | 変更なし |

## §4 v2 からの差分一覧

| 節 | v2 | v3 |
|---|---|---|
| §1.2 | `pendingRetireCount` ＋ `quarantineResidentCount` ＋ `terminalReclaimResidentCount` | **`pendingRetireCount` ＋ `residentCountAtomic`** |
| §1.5（スレッド） | `jassert(!isAudioThread())` のみ | **＋ `isThisTheMessageThread()`** |
| §2（計測） | publish 理由を「prepare/release 由来」と記述 | **実測した 2 経路＋タイマー＋other の 4 区分**、`resetToDefaults` を除外 |

v2 のその他の節（タイマー周期 100 ms、寿命、RT、再入、D→Q→E→T、案の比較、実装順序、必須条件）は維持する。

## §5 状態

- 本改訂は設計のみ。製品コードは未変更（HEAD `c94d3767`）。
- 実装着手の承認は未取得。承認対象は**段階①（ドライバ実装＋競合・寿命テスト）のみ**であり、段階③以降（`firstScanTs`・3 分類）と Reader レジストリ統合は含まない。
