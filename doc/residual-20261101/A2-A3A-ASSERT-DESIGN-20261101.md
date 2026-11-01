# A2 → A3-a 設計案 — Debug assert 違反の修正（2026-11-01）

- 状態: **設計案のみ。製品コードは変更していない**（Owner 承認待ち）
- 対象: `DSPHandleRuntime::destroyQuarantineSlot` と reclaim tick の再試行範囲
- 基準: `main` = `fd92c6e7`
- 前提: 原因は `doc/residual-20261101/A-ASSERT-RACE-ANALYSIS-20261101.md` §3 で特定済み

## 1. 前提（再掲）

`assert(prevState == DSPState::Quarantined)`（`ISRDSPHandle.cpp:529`）は **CAS（同 `:565`）より前**にあり、
状態遷移を排他しているのは CAS である。commit 経路（`AudioEngine.Commit.cpp:682`）と
reclaim tick（`retryDeferredQuarantineDestroy`）が別スレッドで同一 slot を破棄しようとすると、
**遷移は 1 回に保たれたまま assert だけが発火**する。二重解放は起きない（Release は安全）。

## 2. 設計方針（Owner 条件の写像）

| # | 条件 | 本設計での扱い |
|---|---|---|
| A-1 | **assert を削除しない** | 述語 `prevState == Quarantined` は**CAS 成功時の不変条件**として保持する（§3.2）。加えて**真の異常**を検出する assert を新設する（§3.3）。「検査を消して問題を隠す」ことはしない |
| A-2 | A1 単独は採用しない | 呼び出し側の再確認は **A3-a の一部（再試行範囲の限定）**としてのみ扱い、単独案としては採らない |
| A-3 | A2 と A3-a の責務・順序・競合時の挙動を明確化 | §4 の責務表と §5 の順序 |
| A-4 | Debug と Release の両方で状態遷移の整合性 | §6 V3（同一の遷移表を両構成で検証） |
| A-5 | 「二重解放が起きない」と「呼び出し側が誤った状態を扱わない」を**別々**に検証 | §6 V1 と V2 を独立させる |

## 3. A2: assert の位置を遷移の所有者側へ移す

### 3.1 現行の流れ（`ISRDSPHandle.cpp:512-585`）

| 段 | 行 | 内容 |
|---|---|---|
| 1 | 515-524 | slot / generation 検査 |
| 2 | **527-531** | **state を読み、`assert(prevState == Quarantined)`、Quarantined でなければ `false`** |
| 3 | 533-541 | active / fading / crossfade 検査 |
| 4 | 543-559 | pin gate（pin が残れば **deferred 計上**して `false`） |
| 5 | 561-572 | **CAS で `Quarantined → DestroyPending`**（唯一の遷移所有者） |
| 6 | 574-584 | instance クリア → `Reclaimed` → フリーリストへ push |

### 3.2 A2 の形（案）

```cpp
// 段 2（現行 527-531）を「前提の検査」から「競合の正常系」へ変える
const auto observed = convo::consumeAtomic(registry_[slot].state, std::memory_order_acquire);
if (observed != DSPState::Quarantined)
    return false;            // ★ 他者が先に回収した正常系。ここでは assert しない

// 段 3・4 は現行どおり（active/fading/crossfade、pin gate）

// 段 5（CAS）
auto expected = observed;
while (!convo::compareExchangeAtomic(registry_[slot].state, expected,
                                     DSPState::DestroyPending,
                                     std::memory_order_acq_rel,
                                     std::memory_order_acquire))
{
    if (expected != DSPState::Quarantined)
        return false;        // 競合に敗北（正常）。何も遷移させていない
    // expected == Quarantined のまま CAS が失敗 = 状態が Quarantined へ「逆行」した
    assert(!"DSPState regressed to Quarantined without a transition owner");
    return false;
}

// 段 5 成功：遷移の所有者は自分だけ
assert(expected == DSPState::Quarantined);   // ★ 述語を保持（CAS が保証する不変条件）
```

### 3.3 assert の意味づけ（削除ではないことの説明）

| 観点 | 現行 | A2 |
|---|---|---|
| 述語 `prevState == Quarantined` | CAS **前**の観測値に対して検査（競合で容易に偽になる） | CAS **成功時**の不変条件として保持（「自分が遷移させた」ことの保証） |
| 真の異常 | 検出できない（正常な競合と区別できない） | **`Quarantined` への逆行**（`DestroyPending`/`Reclaimed` から戻る）を検出 |
| 誤検出 | **する**（Debug の abort 原因） | **しない**（正常な競合は `false` を返す） |
| 問題の隠蔽 | — | なし。検査は減らず、**判別力が上がる** |

**代替案 A2'（Owner が現行位置の assert 保持を望む場合）**:
段 2 の assert を残しつつ、**発火時に「競合による正常終了」かを判定できる材料**（相手スレッドが遷移させた証拠）を
併記する。ただし assert は abort するため、**正常な競合でも Debug が止まる問題は残る**。
本設計は A2（位置の移動）を推奨する。

### 3.4 競合時の挙動（A2 単独での保証）

| 競合の型 | 結果 | 二重解放 |
|---|---|---|
| 他者が先に `Reclaimed` | 段 2 で `false` | 起きない（遷移していない） |
| 他者が CAS を先に成功 | CAS 失敗 → `expected != Quarantined` → `false` | 起きない（遷移していない） |
| CAS が spurious に失敗 | assert（真の異常）→ `false` | 起きない（遷移していない） |
| 自分が CAS 成功 | 段 6 を実行 | 1 回だけ push（`freeListMutex_` 下） |

## 4. A3-a: reclaim tick の再試行範囲を限定する

### 4.1 現行の問題

`retryDeferredQuarantineDestroy()`（`:590-614`）は `deferredQuarantineDestroy_`（**グローバルなカウンタ**）が
0 でなければ、**Quarantined な全スロット**を走査して破棄を試みる。
そのため「commit 経路が今まさに後片付けしているスロット」も対象になり、不要な競合を生む。

### 4.2 A3-a の形（案）

```cpp
// グローバルカウンタ → スロット単位のマーク（MAX_DSP_SLOTS は 256 程度なので 4 ワードのビットマスク）
std::atomic<std::uint64_t> deferredMask_[4];   // 実装は実装フェーズで確定

// destroyQuarantineSlot: pin gate で止まったとき
if (pinnedCount(slot) != 0) {
    setDeferred(slot);          // ★ グローバルの +1 ではなく、このスロットをマーク
    return false;
}

// retryDeferredQuarantineDestroy: マークされたスロットだけを対象にする
for (each slot in markedSlots()) {
    if (state(slot) != Quarantined) { clearDeferred(slot); continue; }   // 既に他者が終端
    if (destroyQuarantineSlot(slot, 0)) clearDeferred(slot);
    else if (pinnedCount(slot) == 0) clearDeferred(slot);               // 破棄できた/不要になった
}
```

- `deferredQuarantineDestroyCount()`（診断 API）は**マーク数の合計**を返すようにする（既存の呼び出し互換）。
- マークの漏れは slot リークになるため、**「Quarantined かつ instance 非 null だが未マーク」を検出する
  診断カウンタ**を追加する（実装フェーズで検討）。

## 5. 責務分担と適用順序

| 案 | 責務 | 競合時の挙動 |
|---|---|---|
| **A2** | 「観測と遷移」の契約を正す（関数 1 個の内部） | 敗者は `false`、勝者だけが遷移・解放 |
| **A3-a** | 「誰が再試行を起動するか」を限定する（tick の対象） | マークされた slot のみ tick が触る。commit 経路との重複が構造的に減る |

**順序**: **A2 → A3-a**。

1. A2 は A3-a の前提（敗者が安全に `false` を返す契約）を確立する。
2. A3-a は A2 の上で「不要な競合の発生源」を除く。
3. 逆順（A3-a → A2）にすると、A3-a だけでは commit 経路と tick の競合が残り、
   assert の誤検出が残る。

## 6. 検証計画（**実行は別途承認**）

### V1. 二重解放が起きないこと（独立検証）

| 項目 | 内容 |
|---|---|
| 手段 | フリーリストへの push を**スロット×世代**でカウントする計装（テストビルドのみ） |
| 合格 | 同一スロット・同一世代の push が **1 回以下** |
| 負荷 | 同一スロットを 2 スレッドから同時に破棄しようとする競合テスト |
| 対象 | `V11EngineRetireProtectionTests`、`DeferredFlowIntegrationTests`、`STG-8` 群 |

### V2. 呼び出し側が誤った状態を扱わないこと（独立検証）

| 呼び出し元 | `false` を受け取ったときの期待 | 検証方法 |
|---|---|---|
| `AudioEngine.Commit.cpp:682` | `continue` し、**レーン解放（`:684`）とフラグ解放（`:685`）を行わない** | 競合を注入し、`false` 時に lane/flag が変化しないことを確認 |
| `AudioEngine.Processing.ReleaseResources.cpp:468` | 破棄をスキップし、後続の drain に委ねる | 同上 |
| `retryDeferredQuarantineDestroy` | マークを維持し、次 tick で再試行 | マーク状態の遷移を確認 |

**V1 と V2 は別々に実施する**（一方の成功をもう一方の根拠にしない）。

### V3. Debug / Release の両構成

| 構成 | 内容 |
|---|---|
| Debug | assert 有効。ハーネス反復で **assert 発火 0 回** |
| Release | assert 無効。状態遷移の最終状態（slot の state / free list / 世代）が Debug と一致 |

**同一の遷移表**（入力となる競合パターンと期待状態）を両構成で使い、結果を比較する。

### V4. 回帰

| 対象 | 合格 |
|---|---|
| `ctest` 51/51（ASan） | 維持 |
| `isr-verify` / authority 系 | 全 PASS |
| F2 系（`[F2]`） | 全 PASS |
| STG-8 群 | 全 PASS |

## 7. 変更対象（実装フェーズ）

| ファイル | 変更内容 |
|---|---|
| `src/audioengine/ISRDSPHandle.cpp:527-531, 561-572` | A2（観測の扱いと assert の位置） |
| `src/audioengine/ISRDSPHandle.cpp:543-559, 590-614` | A3-a（deferred マークと再試行範囲） |
| `src/audioengine/ISRDSPHandle.h` | マーク用のメンバと診断 API |
| （テスト）`src/tests/AudioEngineHarness/*` | V1 の計装と V2 の競合注入（**テスト変更は別途承認**） |

## 8. 未確定事項（実測で解決したもの／残るもの）

### 8.1 解決済み（2026-11-01 20:20 実測）

| # | 項目 | 結果 |
|---|---|---|
| 1 | `MAX_DSP_SLOTS` | **256**（`ISRDSPHandle.h:197`）。マークは `std::uint64_t` × 4 のビットマスクで足りる |
| 2 | `deferredQuarantineDestroyCount()` の呼び出し元 | `AudioEngine.Retire.cpp:44`（`quarantineEmpty` 判定）と `:72`（早期 return 条件）、`DeferredPublicationTestAccess.h:175`。**「保留中のスロット数」を返す契約**を保つ必要がある |
| 3 | 既存の deferred 計上 | `destroyQuarantineSlot` の pin gate（`ISRDSPHandle.cpp:551-559`）で `deferredQuarantineDestroy_` に `+1` |

### 8.2 残る未確定事項

1. 「Quarantined かつ instance 非 null だが未マーク」の検出方法（診断カウンタで足りるか、
   マーク漏れを構造的に防ぐか）。
2. A2 の CAS 失敗時のリトライ回数の上限（spurious 失敗が連続する場合の扱い）。
3. Debug 実行での assert 発火頻度（未計測）。A2 適用後に V3 で 0 回を確認する。

## 9. 実施しないこと

- 製品コード・テストの変更（本設計は文書のみ）
- assert の削除（述語は CAS 成功時の不変条件として保持し、真の異常検出を追加する）
- A1 単独案の採用（呼び出し側の再確認だけでは TOCTOU が残る）
- ビルド・デバッガの実行（別途承認）
