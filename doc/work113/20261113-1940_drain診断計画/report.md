# drain 診断ログの実装計画（編集前の提示）

- 作業日時: 2026-11-13 19:40 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）
- 種別: **実装計画の提示のみ**。ソースは変更していない。
- 前提: Owner 判断（2026-11-13）により、診断ログは「条件付き承認」。編集前に変更箇所・取得方法・副作用を示すこと。

---

## 1. 現行ソースの確認

Owner の手元にある最新スナップショット（`ConvoPeq.md`、2026-11-12 生成）は今回の working tree を反映していない。
そこで、**working tree を直接読んで**行番号を確定した。

| ファイル | サイズ | SHA-256（先頭 16） | 用途 |
| --- | ---: | --- | --- |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | 927 行 | `b6f5ac36836c7901` | 挿入箇所 2 つ |
| `src/audioengine/AudioEngine.h` | — | `3d72e88dbe436eab` | getter 追加 1 つ |
| `src/audioengine/AudioEngine.Threading.cpp` | — | `93f0ccfbb59fd157` | 参照のみ（`isFullyDrained` / `collectDrainAudit`） |

## 2. 変更箇所（3 つ）

### 2.1 `ReleaseResources.cpp:662`（タイムアウト分岐）— ログ 1 行

現行:

```cpp
    if (timedOut) {
        auto audit = collectDrainAudit();                 // 既存の呼出しをそのまま使う
        auto reason = convo::isr::ShutdownBlockingReason::Unknown;
        if (audit.stuckReaderCount > 0) ...
        shutdownRuntime_.markTimedOut(reason);
    }
```

追加（`collectDrainAudit()` の直後、`markTimedOut` より前）:

```cpp
        diagLog("[DRAIN_DIAG] timedOut=1"
                " pendingReclaimHandles=" + juce::String(pendingReclaimHandleCount())
                + " retireDepth=" + juce::String(audit.routerPendingRetire)
                + " lifetimeIntent=" + juce::String(audit.pendingRetire)
                + " overflowRing=" + juce::String(audit.overflowRingResident)
                + " dspQuarantine=" + juce::String(audit.quarantineResident)
                + " retireQuarantine=" + juce::String(m_retireRouter->quarantineResidentCount())
                + " terminalReclaim=" + juce::String(m_retireRouter->terminalReclaimResidentCount())
                + " bridgeDrained=" + juce::String(runtimePublicationBridge_.isFullyDrained() ? 1 : 0)
                + " deferredPublish=" + juce::String(audit.deferredPublish)
                + " readers=" + juce::String(audit.activeReaderCount)
                + " domain=" + juce::String(static_cast<int>(m_epochDomain.domainId()))
                + " reclaimAttempts=" + juce::String(audit.reclaimAttemptCount)
                + " reclaimSuccess=" + juce::String(audit.reclaimSuccessCount));
```

- `if (timedOut)` の中なので **1 回の shutdown につき 1 行**。2 ms ポーリングごとには出ない。
- `collectDrainAudit()` は既にこの分岐で呼ばれているため、**呼出しを増やさない**（値を再利用する）。

### 2.2 `AudioEngine.h` — getter 1 つ

`pendingReclaimHandles_` は private で getter が無い。9 条件の 2 番目を記録するため、読み取り専用の
getter を 1 つ足す。

```cpp
    // ★ 診断用: pending reclaim の件数（値の取得のみ。状態は変えない）
    [[nodiscard]] std::size_t pendingReclaimHandleCount() const noexcept
    {
        const std::lock_guard<std::mutex> lock (pendingReclaimHandlesMutex_);
        return pendingReclaimHandles_.size();
    }
```

`pendingReclaimHandlesMutex_` は `mutable` であることを確認済み（`isFullyDrained()` が const から
同じ mutex を取っている）。

### 2.3 `ReleaseResources.cpp:579-584` — World の clear→retire の成否

現行は戻り値（`clearedWorld`）の null 判定だけで、retire の結果は `void` のため取れない。
**処理は変えず**、判定結果を診断ログに 1 行だけ残す。

```cpp
    auto* clearedWorld = worldAuthority_.clearPublishedRuntimeSnapshotsNonRt();
    if (clearedWorld != nullptr)
    {
        RuntimePublicationBridge clearBridge{ *this, runtimePublicationValidator_ };
        clearBridge.retirePublishedRuntimeWorldNonRt(clearedWorld, true);
    }
    diagLog("[DRAIN_DIAG] worldClear cleared=" + juce::String(clearedWorld != nullptr ? 1 : 0)
            + " published=" + juce::String(static_cast<juce::int64>(worldLifecycleAudit_.publishedCount()))
            + " retired=" + juce::String(static_cast<juce::int64>(worldLifecycleAudit_.retiredCount()))
            + " terminalReclaim=" + juce::String(m_retireRouter->terminalReclaimResidentCount()));
```

- `retirePublishedRuntimeWorldNonRt` は `void`（`AudioEngine.h:3973`）なので、**成否は直接取れない**。
  代わりに「clear が何かを返したか」と「audit の published/retired 件数」「terminal 退避の件数」で状態を示す。
- これは処理の変更ではなく、既存の戻り値と既存のカウンタの記録である。

## 3. 値の取得方法（9 条件との対応）

| # | `isFullyDrained()` の条件 | 取得方法 | 既存/新規 |
| --- | --- | --- | --- |
| 1 | `!hasDeferredCommit` | `audit.deferredPublish == 0`（同じ `hasDeferredRequest()` 由来） | 既存 |
| 2 | `pendingReclaimHandles_.empty()` | 新 getter `pendingReclaimHandleCount()` | **新規（getter 1 つ）** |
| 3 | `pendingRetireCount() == 0` | `audit.routerPendingRetire`（router の値 + fallback depth） | 既存 |
| 4 | `lifetime().pendingIntentCount() == 0` | `audit.pendingRetire` | 既存 |
| 5 | overflow ring resident == 0 | `audit.overflowRingResident` | 既存 |
| 6 | DSP quarantine resident == 0 | `audit.quarantineResident` | 既存 |
| 7 | retire quarantine resident == 0 | `m_retireRouter->quarantineResidentCount()` | 既存（audit 外） |
| 8 | terminal reclaim resident == 0 | `m_retireRouter->terminalReclaimResidentCount()` | 既存（audit 外） |
| 9 | `runtimePublicationBridge_.isFullyDrained()` | 同じ関数を呼ぶ | 既存 |

補助として `readers`・`reclaimAttempts`・`reclaimSuccess`・`domain` も同じ行に出す。

## 4. 残存 2 件のドメイン（コードで確定）

`m_retireRouter` は `AudioEngine.CtorDtor.cpp:62` で
`std::make_unique<convo::isr::ISRRetireRouter>(m_epochDomain, &worldRetirementReference_)` として作られ、
`pendingRetireCount()` は provider（＝`m_epochDomain`）へ委譲する（`ISRRetireRouter.cpp:878-883`）。
`m_epochDomain` は `AudioEngine.h:5456` で
`convo::EpochDomain m_epochDomain { convo::RetireDomainId::EngineISR };` と宣言されており、
**ドメインは EngineISR である**（コード上の直接の根拠）。EQ のドメインは別インスタンス（`EQProcessor` 側）である。

したがって **`isFullyDrained()` が見ている `pendingRetireCount()` は EngineISR ドメインの値**であり、
残存 2 件は EngineISR ドメインである。EQ ドメインの滞留はこの判定に含まれない。
ログの `domain=` はこの前提を実行時に裏づけるために出す。

## 5. 副作用

| 観点 | 内容 |
| --- | --- |
| 実行される処理 | `pendingReclaimHandleCount()` は短時間 mutex を取る。他はすべて既存の読み取りアクセサ |
| 処理順序・同期 | 変更しない。`collectDrainAudit()` の呼出し位置・回数も変えない |
| drain / reclaim / epoch 前進 / retire | **追加しない**（診断ログは値を読むだけ） |
| RT 経路 | 触れない（`releaseResources` は非 RT） |
| ログ量 | 1 shutdown あたり 2 行（タイムアウト分岐 1 行、world clear 1 行）。ポーリングごとには出さない |
| 失敗時の挙動 | ログ文字列の生成のみ。例外は投げない（`juce::String` の連結） |
| 性能 | 文字列連結が 2 回。終了経路の 1 回だけであり、計測可能な影響は想定しにくい（§6 で確認する） |

## 6. 検証計画

| # | 検証 | 方法 |
| --- | --- | --- |
| 1 | 1 shutdown あたり 2 行だけ出ること | `--cli-run --cli-exit-ms 2000` を 3 回実行し、`[DRAIN_DIAG]` の行数を数える（期待: 各 2 行） |
| 2 | 値が再現すること | 3 回の `retireDepth` が 2 で一致するか確認（前回の証跡と照合） |
| 3 | 通常終了の時間に影響しないこと | 変更前後で総経過と C 全体を比較（前回実測: 総 6.26〜6.57 s、C 2.38〜2.42 s）。差がノイズ範囲かを確認 |
| 4 | 既存テスト | `ctest -C Release` 61 件 |
| 5 | ログ追加以外の差分が無いこと | `git diff` を確認し、追加行がログと getter だけであること |

## 7. ロールバック

- 追加した 3 箇所（2 行のログ、getter 1 つ）を削除するだけで元に戻る。既存の制御フローを書き換えない。

## 8. 未確定・代替

| # | 内容 | 扱い |
| --- | --- | --- |
| 1 | `retirePublishedRuntimeWorldNonRt` の成否 | 戻り値が `void` のため直接取れない。§2.3 の代替（clear の null 判定＋audit カウンタ）で状態を示す |
| 2 | EQ ドメインの滞留 | 今回の判定対象外。必要なら別途、EQ 側のキューを読む計測が要る |
| 3 | 残存 2 件の「中身」（型・サイズ） | キューを走査する API が無いため、今回は件数とドメインまで。必要なら別承認で走査用 API を追加する |

## 9. 停止条件の確認

- ソース変更・ビルド・テストは行っていない。
- 本書は実装計画であり、診断ログの実装承認を意味しない。
