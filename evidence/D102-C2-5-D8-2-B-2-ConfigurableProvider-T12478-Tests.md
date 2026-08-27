# D102-C2-5-D8-2-B-2 — Configurable Epoch Provider + T1/T2/T3/T4/T7/T8 Test-First Implementation

- **実施日**: 2026-08-26
- **判定**: **ALL PASS** — T1/T2/T3/T4/T7/T8 = PASS (6/6)、T5/T6/T9 = DEFERRED、T10 = source audit
- **production runtime semantics 変更**: **0**
- **production source 変更有無**: **なし**（`src/` 配下の既存ファイルは全て無変更）
- **I4 contract 変更**: 0 / Terminal bounded 化: 0 / QueueFull・Shutdown 人工 path 追加: 禁止遵守 / resetFade public 化: 禁止遵守 / DSPLifetimeManager 抽象化: 禁止遵守

---

## 1. 変更ファイル一覧

| ファイル | 種別 | 内容 |
|---|---|---|
| `src/tests/D8_2_B_2_Tests.cpp` | **新規 (test-only)** | ConfigurableEpochProvider + T1/T2/T3/T4/T7/T8 |
| `CMakeLists.txt` | **変更（test target 登録のみ）** | `D8_2_B_2_Tests` ターゲット追加（ソース登録 + include + link + Windows定義 + icx LTCG OFF + add_test） |

### production source 変更有無

```
git status --short:
 M CMakeLists.txt              ← test target 登録のみ（production ビルドに影響なし）
 M ConvoPeq.md                 ← タイムスタンプのみ（自動生成ファイル）
?? src/tests/D8_2_B_2_Tests.cpp ← 新規テストファイルのみ

git diff --stat src/ → 変更ゼロ
```

---

## 2. ConfigurableEpochProvider（test-only helper）

`D8_2_B_2_Tests.cpp` の anonymous namespace 内に定義。`src/tests/` 配下限定。

### 制御インターフェース（全て atomic — `std::function` 不使用）

```text
enqueueRetireResult : std::atomic<bool>   true→D受入 / false→Q/E/T escalation 誘発
minReaderEpoch_     : std::atomic<uint64_t> getMinReaderEpoch() の戻り値制御
currentEpoch_       : std::atomic<uint64_t> currentEpoch()/publishEpoch() 実体
```

### 観測カウンタ

```text
enqueueRetireCallCount      3-arg版呼出数（SnapshotCoordinator::enqueueWithRetry が使用）
enqueueRetireTypedCallCount 4-arg版呼出数（ISRRetireRouter::enqueueRetire(4-arg) が使用）
tryReclaimCount / publishEpochCount
lastEnqueuedPtr             D 受入時の最終 ptr（T1/T7 の D-path cleanup 用）
```

### 設計上の注意

- `enqueueRetire`(3-arg) と `enqueueRetireTyped`(4-arg) を独立オーバーライドし、
  カウンタを分離（default 委譲による二重カウントを排除）。
- RT 相当経路に `std::function` を持ち込まない — 将来の RT caller テスト転用に備え atomic 制御のみ。
- 既存 stub の統合: RetireGraceSemanticsTests の `TestProvider`(false固定) +
  D8_1_WrapperCacheTests の `TestEpochProvider`(true固定) → `enqueueRetireResult` で切替可能に統一。

---

## 3. テスト結果（実行ログ）

ビルド: MSVC (VS18 Enterprise, cl.exe) + Ninja Multi-Config Debug、exit code **0**。

```text
[T1] SnapshotCoordinator → D (Success)...        PASS: D owns, Q/E/T empty
[T2] SnapshotCoordinator → Q (QueuePressure)...  PASS: Q owns exactly 1, D/E/T empty, no double-transfer
[T3] ISRRetireRouter Q → E ...                   PASS: E owns 1, Q=512 full, T empty, deleteCount==0
[T4] ISRRetireRouter Q+E → T ...                 PASS: T owns 1, growable store accepted, deleteCount==1 after drain
[T7] Snapshot ownership conservation...
     [A] D owns snap1 (enqueueRetireCount=1), Q=0          PASS
     [B] Q owns snap1 (D rejected=2), Q=1                  PASS
[T8] Router ownership conservation...
     [A] Q path — admitted 1, deleted exactly once         PASS
     [B] E path — admitted 1, deleted exactly once         PASS
     [C] T path — admitted 1 (growable), deleted exactly once PASS

D8-2-B-2 Tests PASS  (exit code: 0)
```

### 各テストの検証内容と PASS 条件の対応

| Test | 検証した disposition | 主要アサーション | 判定 |
|---|---|---|---|
| T1 | Snapshot → **D owns** | enqueueRetireCallCount≥1, Q/E/T==0 | **PASS** |
| T2 | Snapshot → **Q owns** | enqueueRetireCallCount==2(retry), Q==1, E/T==0, pendingRetire==0 | **PASS** |
| T3 | Router Q(full) → **E owns** | result==QueuePressure, E==1, T==0, deleteCount==0 | **PASS** |
| T4 | Router Q+E(full) → **T owns** | result==TerminalReclaim, T==1, drain後 deleteCount==1 | **PASS** |
| T5 | QueueFull | **DEFERRED**（growable Terminal 下では到達不能 — D2-1 bounded Terminal 設計時に再評価） | DEFERRED |
| T6 | Shutdown | **DEFERRED**（production path 追加禁止遵守） | DEFERRED |
| T7 | Snapshot conservation | admitted=1、D/Q の排他的所有を両パスで固定 | **PASS** |
| T8 | Router conservation | admitted=1、Q/E/T 全パスで drain 後 deleteCount==exactly 1 | **PASS** |
| T9 | DSPLifetimeManager | **DEFERRED**（source audit に分解 — B-2 では触らない） | DEFERRED |
| T10 | RT boundary | source audit（friend 追加却下遵守） | audit |

### T10 source-audit 裏付け（本実装時に再確認済み）

`SnapshotCoordinator.cpp:81-96` `resetFadeStateAndRetireTarget()`:

```text
├─ m_slots.exchangeTarget(nullptr)   … target 取得
├─ publishEpoch()                    … epoch 公開
├─ enqueueRetire(target, snapshotDeleter, retireEpoch)  … D のみ（bool 戻り値は無視＝RT-safe）
└─ resetToIdle()                     … fade 状態解除

NO: quarantineRetire / emergencyQuarantine / terminalReclaim / enqueueWithRetry — 呼び出しゼロ（grep 確認済み）
```

---

## 4. テスト設計上の重要ポイント

1. **ownership transfer 一回性の固定（T2）**: `enqueueWithRetry` 内 retry は 2 回の `enqueueRetire`
   呼び出しで構成されるが、ptr 自体は 1 回だけ `quarantineRetireSink` へ移送される。
   `quarantineResidentCount == 1` により caller→Q transfer の単回性を検証。

2. **E/T escalation の事前充填方式（T3/T4/T8-B/C）**: `enqueueWithRetry` を 512/1024 回回す代わりに
   public API `quarantineRetire()` / `emergencyQuarantine()` で直接 pre-fill。
   epoch=0 & minReaderEpoch=0 のため `isOlder(0,0)==false` となり、retry loop 内
   `drainEmergencyAndTerminal()` でも drain されないことを利用。

3. **exactly-once deletion（T8）**: counting deleter により
   - drain 前 `deleteCount == 0`（store 保持中は未削除）
   - `drainAllQuarantineStore()` 後 `deleteCount == 1`（exactly once）
   を Q/E/T 全経路で検証。noop deleter の pre-fill エントリは deleteCount に寄与しない。

4. **cleanup の二重解放回避（T7-A）**: D-path エントリ（stub 受入・未保存）は
   `lastEnqueuedPtr` から手動 destroy、Q-path エントリは destructor → Q → drainAllQuarantineStore
   で破棄し、同一 ptr の二重 free を構造的に回避。

5. **SnapshotCoordinator と Router は別 provider を共有しない構成**:
   production (`m_coordinator(m_epochDomain)` + `setRetireSink(m_retireRouter.get())`) と同様に、
   test でも provider(ConfigurableEpochProvider) を coordinator に、ISRRetireRouter を retire sink に接続。

---

## 5. 未実施（DEFERRED）と根拠

| 項目 | 根拠 |
|---|---|
| T5 QueueFull | `RetireEnqueueResult::QueueFull` は enum 存在のみで production return path なし。Terminal growable のため Q+E exhausted 後も `TerminalReclaim` で閉じる。**D2-1 bounded Terminal 設計時に別 gate 再評価** |
| T6 Shutdown | `return Shutdown;` の追加はテストのための runtime semantics 変更。shutdown drain/discard protocol は既存テスト系（ShutdownRetireIntentDrainTests 等）で検証すべき |
| T9 DSP | Router 単体での ownership contract 証明は完了（本 B-2）。次段は `DSPLifetimeManager::retire()` が contract を壊さないことの **source audit**（抽象化ではなく） |
| T10 friend | 却下確定。static/source-level audit として §3 に記載の通り検証済み |

---

## 6. 次ステップ

**D8-2-C（disposition contract audit）へ進行可能。**

- T1–T4/T7/T8 が現行 ownership contract（D→Q→E→T、Terminal always accepts）と一致することを実行レベルで確認済み
- T5/T6/T9 の DEFERRED 判断と根拠は本報告書 §5 に確定済み
- T10 は source audit 完了済み（§3）
