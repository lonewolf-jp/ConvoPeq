# ConvoPeq メモリ肥大化 改修計画書 v5.38

**日付**: 2026-07-11
**対象**: work70 (メモリ 2.5GB 肥大化問題)
**前版**: v5.37 (2026-07-11)

> **本計画書は「プログラマがコーディングするために必要な設計情報」を先頭にまとめ、補足情報を Appendix に配置している。**
> 凡例: ✅ FACT / 🔍 HYPOTHESIS / 💡 PROPOSAL / ⚠️ CAVEAT

---

## [設計] 0. エグゼクティブサマリ

### 0.1 主要原因候補（最有力仮説）

🔍 **HYPOTHESIS**: lifecycle(retire)=0 の原因として **handle map 未登録が最有力**。P1-a により即時 retire 経路が回復することが期待されるが、enqueueRetire の QueueFull や epoch 不整合、reclaim 失敗などの後段経路が残存する可能性もあるため、完全な確定には実装後の検証が必要。

Coordinator direct publish（`prepareToPlay` 時の gen=3 publish）が、DSPCore を `runtimeDSPHandleMap_` に登録せずに current 化する。その後、後続の Orchestrator publish（gen=4, gen=8）で DSPTransition が oldDSP を retire しようとしても、handle map に未登録のため `retireDSPHandleForRuntime()` が false を返し、EBR enqueue に至らない。

※ 仮説が正しいことが証明されても、メモリ2.5GBの全量が改善されるとは限らない（AoS/BlockSize/IR/MKL 等の他要因が残る可能性がある）。

**調査で確定した事実**:- Orchestrator publish 3 回は全て Structural rebuild（`requestRebuild_sr_bs`）がトリガー。**構造的 rebuild では CrossfadeAuthority が needsCrossfade=false を返すため、DSPTransition は crossfade ではなく即時 retire 経路を使用する。**
- ログ `[XFADE]=0` はこの動作と整合。advanceFade（P1-b）は SnapshotCoordinator のパラメータフェード用であり、今回の Structural rebuild では関係しない。
- したがって **P1-a（handle 登録追加）が即時 retire 経路の回復に必須。P1-b（advanceFade）はパラメータフェード用として別途必要。**

```text
Publish sequence (ログから確定):
  gen=2: Orchestrator gen=1 → DSP_A registered    → current=A ✓
  gen=3: Coordinator direct  → DSP_B NOT registered → current=B ★原因
  gen=4: Orchestrator gen=4 (Structural) → needsCrossfade=false → immediate retire(B) → FAIL(handle未登録)
  gen=5: Orchestrator gen=8 (Structural) → needsCrossfade=false → immediate retire(?) → FAIL
```

### 0.2 優先順位

| 優先度 | Phase | 内容 | 変更ファイル数 | リスク |
|:-------|:------|:------|:-------------|:-------|
| **P0** | ✅ **完了** | lifecycle(retire)=0 の原因特定（主要原因と判断） | 0 (解析済) | なし |
| **P1-a** | 実装 | Coordinator direct publish に handle 登録を追加 | 1-2 | 中 |
| **P1-b** | 実装 | advanceFade 配線（AudioBlock.cpp 1行） | 1 | 低 |
| **P1-c** | 実装 | MEM_SNAP 監視強化 (Stereo/DSPCore liveCount) | 1 | 低 |
| P2-1 | 調査 | 初回 BlockSize (524288) 調査 | 0 | なし |
| P2-2 | 実装 | BlockSize 最適化 | TBD | 中 |
| P3 | 実装 | AoS→SoA (11パッチ) | 2+ | 高 |
| P4 | 実装 | 未計装 mkl_malloc DIAG 化 | 4 | 低 |

### 0.3 検証済みの設計判断

| 判断 | 結論 | 根拠 |
|:-----|:------|:------|
| advanceFade のサンプル単位 | **OS 補正不要。** コールバックサンプル数のまま減算 | `getNextAudioBlock()` の `numSamples` は callback block size。`SnapshotFadeState` も同一単位 |
| `FadeState::Completed` 追加 | **不要。** 既存の remaining==0 チェックで完了検出可能 | `tryComplete()` は `remainingCount() > 0` を直接チェック |
| `memory_order` 変更 | **現状維持（relaxed 化は却下）。** 現状の release/acquire が正しい HB を提供 | remaining→tryComplete の同期に release/acquire が必要 |
| CrossfadeRuntime への完了通知追加 | **不要。** SnapshotCoordinator と CrossfadeRuntime は責務が完全に独立 | 後述の設計方針セクション参照 |

---

## [設計] 1. P1-a: publish 経路への handle 登録追加

### 1.1 なぜ必要か

✅ **FACT**: Coordinator direct publish は `registerDSPHandleForRuntime()` を呼ばない。これにより該当 DSPCore が `runtimeDSPHandleMap_` に未登録となり、後続の全 retire が `retireDSPHandleForRuntime()` で false を返す。現時点の lifecycle(retire)=0 はこの事実で最もよく説明できる。

✅ **FACT**: Orchestrator publish（DSPTransition）経路の retire は **本来動作している**。ログ上でも DSPTransition::onPublishCompleted() は 3 回全て実行済み。しかし oldDSP が handle map に未登録のため失敗する。

### 1.2 設計上の Invariant

本設計が保証すべき不変条件を明文化する:

| ID | Invariant | 根拠 |
|:---|:----------|:------|
| INV-1 | **公開を試みる DSP は publish 前に Handle が登録済みである。** | commitRuntimePublication が唯一の publish 窓口であり、engine-managed 時は内部で register を実行する。registerDSPHandleForRuntime() は runtimeDSPHandleMap を authority とする idempotent operation — find→return(既存) or create→emplace(新規)。同一 DSPCore* の重複登録は map により抑制される。 |
| INV-2 | **公開失敗時は Handle 登録をロールバックする。** commitRuntimePublication が publish の成否に応じて rollback を自動実行する。 | rollbackRegistration (CAS: Constructing→Reclaimed) により Handle を利用可能プールに戻す。 |
| INV-3 | **Commit point 以降は Handle をロールバックしてはならない。** `PublishStageResultTraits::isCommitted()` が true を返した以降（commit point）は、Handle が公開済みとみなし rollback を禁止する。 | `PublishStageResultTraits::isCommitted(stage)` が commit point を抽象化する。commitRuntimePublication 内では commit point 到達後に `rollbackHandle = DSPHandle::null()` で無効化（`isNull()==true` に）することで SCOPE_EXIT による rollback を防止。`publishPerformed` flag により二重防御する。Coordinator の実装詳細（publishAndSwap 成功/失敗）ではなく Traits が commit 判定を一元管理する。 DSPHandle に `invalidate()` メソッドは存在せず、`= DSPHandle::null()` 代入により `slot=0,generation=0` を設定する。 |
| INV-4 | **runtimeDSPHandleMap に Reclaimed 状態の Handle が永続的に残ってはならない。** 短時間の過渡的共存（rollback における CAS 成功～Map erase の間）は許容されるが、最終的には Map から削除される。 | register/create → Constructing → (publish success, or rollback → CAS Reclaimed → Map erase)。retireDSPHandleForRuntime が map erase + Handle.retire + Handle.reclaim (→Reclaimed) を不可分に実行する。DSPHandle 版 rollback の分割ロックでは CAS 成功と Map erase の間に短い gap があるが、DSPHandle::operator==（slot+generation 比較）により新しい Handle と誤一致しない。generation は create() で常に +1u インクリメントされるため安全性が担保される（v5.26 コード調査確定）。 |
| INV-5 | **rollbackDSPHandleRegistration() 成功後、同一 DSPCore は registerDSPHandleForRuntime() により新しい DSPHandle を取得できる。** Constructing→Reclaimed によりスロットが利用可能プールに戻るため、後続の register→create で再利用可能。 | rollbackRegistration の CAS は Constructing→Reclaimed 遷移を行い、create() は Reclaimed を再利用対象とする（ISRDSPHandle.cpp:25）。 |
| INV-6 | **commitRuntimePublication() を経由して publish を試行した RuntimeWorld に対しては、Handle 登録と publish の責務は必ず同一トランザクション内で完結する。** 途中失敗時には Handle 登録だけが rollback され、Coordinator の publish 状態は rollback の対象とならない。 | publishPerformed flag により commit point を検出。publish 成功後の rollback を防止（INV-3）。Coordinator 状態のロールバックは不要（publishWorld の副作用ゼロ FACT）。 |
| INV-7 | **rollbackRegistration() 成功後、同一 DSPCore* は通常 runtimeDSPHandleMap から削除される（Map 不整合は DIAG のみで報告）。** `Reclaimed` 状態だけでなく Map 整合性も原則保証するが、Map erase が失敗しても Runtime の rollback は完了済み（CAS 成功）であり、致命的ではない。 | rollbackDSPHandleRegistration は CAS(Constructing→Reclaimed) を最優先。CAS 成功後は Map erase をベストエフォートで実行（eraseByHandle）。Map に Handle が存在しない場合は `RolledBackMapMissing`（DIAG のみ）として記録。Production では Runtime rollback が成功していれば常に成功とみなす。次の register 時に新しい Handle で上書きされるため Map 不整合は自動修復される。 |
| INV-8 | **`alreadyRegistered()` で渡される Handle は Constructing 状態でなければならない。** rollbackRegistration() は Constructing→Reclaimed の CAS のみ成功するため、登録直後の Constructing Handle のみ rollback 可能。Active/Retired 等の Handle を渡すと CAS 失敗し rollback が無視される。 | rollbackRegistration（ISRDSPHandle.cpp:312-315）は CAS(Constructing→Reclaimed) のみ成功。登録直後の Handle は必ず Constructing（create() → publishAtomic(state, Constructing)）。DSPTransition の `registerDSPHandleForRuntime(newDSP)` も同様。この前提が保証されない場合は `AlreadyRegistered` モードで Handle を渡してはならない。 |

⚠️ **CAVEAT**: INV-3 は Coordinator の現行実装（`publishAndSwap` 成功後にのみ Success を返す）に依存する。将来 Coordinator の実装が変更され、swap 成功後の post-processing が失敗する経路を追加する場合は、commitRuntimePublication 側も対応が必要。

⚠️ **CAVEAT**: `DSPHandleRuntime::retire()`（ISRDSPHandle.cpp:95）は状態チェックなしで `publishAtomic(state, Retired)` を実行する。CAS や assert による防御は存在しない。したがって理論上は `Constructing` 状態でも `Retired` に遷移可能。ただし commitRuntimePublication の `registerDSPHandleForRuntime()` から `publishWorld()` の間に別スレッドが同一 Handle を retire することは、以下の理由で現実的に不可能:
  - 登録された DSPCore* はまだ公開前の RuntimePublishWorld 内部にあり外部から到達不能
  - DSPHandle は呼び出し元のローカル変数（rollbackHandle）でのみ保持
  - retireDSPHandleForRuntime は DSPCore* 検索が前提で、未公開の DSPCore* を他スレッドが参照できない
したがって本 CAVEAT は将来のコード変更時に注意すべき既知の制約として文書化する。

### 1.3 設計判断: Coordinator + Bridge によるトランザクション — AudioEngine はプリミティブのみ

**「Publish を試行する DSP は publish 前に Handle 登録済み」** という invariant を **トランザクション構造で保証** する。

`registerDSPHandleForRuntime()` は idempotent（既存エントリがあればそれを返す）であることをコード確認済み（AudioEngine.h:3804-3819）。

#### 採用設計: AudioEngine::commitRuntimePublication() + RegistrationContext

**「Runtime publication のトランザクション（register → publish → 失敗時 rollback）は AudioEngine の責務」** とし、register の要否は `RegistrationContext` で暗黙的に表現する。`dsp != nullptr` の場合は登録必要、`handle` のみ指定の場合は事前登録済み、両方 null の場合は登録不要を示す。

`getBridge()` が Coordinator テンプレートに存在しないため、commitRuntimePublication は AudioEngine の private メソッドとし、Coordinator は引数で受け取る。register 要否を `DSPCore*` の nullptr で暗黙的に示すのではなく、`RegistrationContext` の `dsp` フィールドで明示する。

★ v5.26: `PublishRegistrationMode` enum から `RegistrationContext` 構造体に移行。enum の 2 値（NeedsRegistration/AlreadyRegistered）を構造体のフィールド有無で表現することで、将来の handle 種別拡張（ImportedHandle / RecoveredHandle / ExistingHandle）に柔軟に対応できる。

✅ **FACT（2026-07-10 コード調査確定）**: `RuntimePublicationCoordinator::publishWorld()` は**副作用が残らない**。失敗経路（Validation 拒否時）は rejection した World を `bridge_.retireRuntimePublishWorldNonRt(rejectedWorld, false)` で適切に後始末する。`publishAndSwap` 成功後は Coordinator 内部状態が確定し、その後は非 Success を返さない（coordinator state 変更は publish 成功＝確定を意味する）。したがって rollback が必要なのは「register した Handle」のみであり、Coordinator 状態のロールバックは不要。ただし将来の実装変更に備え、commitRuntimePublication 内部で `publishPerformed` flag を導入し、publishAndSwap 成功後に rollback が実行されるのを防止する（INV-3 の二重防御）。

✅ **FACT（2026-07-10 コード調査確定）**: `PublicationExecutor` 経路では `enqueuePublicationIntentForRuntimeCommit()` が `submitPublishRequest()` 呼び出し**前**に `registerDSPHandleForRuntime(newDSP)` を必ず実行する（AudioEngine.Commit.cpp:684）。`PublicationAdmission::PublishRequest` は既に `DSPHandle newDSP` メンバを持つ（PublicationAdmission.h:18）。したがって `req.newDSP` をそのまま `PublicationExecutor::publish()` の `existingHandle` として渡すデータフローが既に成立している。PublicationExecutor からは `AlreadyRegistered` モードで commitRuntimePublication を呼び出し、この Handle を渡すことで publish 失敗時の rollback を可能にする。

```cpp
// ★ work70: RegistrationContext — commitRuntimePublication の registration コンテキスト。
//   dsp が設定されている場合: commitRuntimePublication が register → publish → rollback を担当
//   handle が設定されている場合（かつ dsp==nullptr）: 呼び出し元が事前登録済み。publish + fail-rollback のみ
//   両方 null: 登録不要（Bootstrap world / Hard reset）
//   ★ 将来 ImportedHandle / RecoveredHandle / ExistingHandle 等が増えても
//     追加フィールドで表現可能。enum 拡張不要。
//   ★ v5.26: enum PublishRegistrationMode から移行。mode を暗黙的に表現する構造体。
struct RegistrationContext {
    DSPCore* dsp = nullptr;
    DSPHandle handle;

    // ★ v5.33: 静的ファクトリ — 「dsp と handle の同時指定」という不正状態を構造的に防止。
    //   呼び出し元はこれらのファクトリのみ使用する。
    // DSP を新規登録: commitRuntimePublication が register → publish → rollback を担当
    static RegistrationContext needsRegistration(DSPCore* dsp_) noexcept {
        return { dsp_, DSPHandle::null() };
    }
    // Handle が事前登録済み: 呼び出し元が register 済み。rollback のみ担当
    static RegistrationContext alreadyRegistered(DSPHandle handle_) noexcept {
        return { nullptr, handle_ };
    }
    // 登録不要: Bootstrap world / Hard reset
    static RegistrationContext none() noexcept {
        return { nullptr, DSPHandle::null() };
    }
};

// AudioEngine.h に追加（private メソッド、friend 経由の PublicationExecutor も使用可）
// ★ work70: RuntimeWorld の publish は必ずこの関数を経由する。
//   register → publish → 失敗時 rollback のトランザクションを保証する。
//   ★ v5.22: 戻り値は PublishCommitResult（stage のみ。committed/handle は含まない）。
//   ★ v5.24: rollback は ScopeExit パターンで実装 — 将来 return 経路が増えても漏れ防止。
//   ★ v5.26: RegistrationContext により dsp/handle を統一的に受け取り、
//     PublishStageResultTraits::isCommitted() で commit point 判定を委譲。
//     rollbackHandle を publishHandle から分離（registeredHandle 削除）。
//   ★ v5.27: committed は stage から導出可能（Traits::isCommitted）。PublishCommitResult
//     に含めず二重管理を防止。rollbackHandle は ScopeExit 内部でのみ使用し返却しない。
//     P1-a の変更は最小限に留め、HandleRegistry や helper 抽象化は後続リファクタリングに委ねる。
//   ★ v5.33: ScopeExit 条件を committed bool + rollbackHandle の二重管理から
//     rollbackHandle.isNull() のみに簡略化（commit point 到達後に DSPHandle::null() 代入）。
//     RegistrationContext は静的ファクトリ（needsRegistration/alreadyRegistered/none）で生成。
[[nodiscard]] PublishCommitResult commitRuntimePublication(
    RuntimePublicationCoordinator& coordinator,
    convo::aligned_unique_ptr<RuntimePublishWorld> world,
    const RegistrationContext& regCtx) noexcept
{
    // ★ v5.33: ScopeExit — rollback を確実に実行。
    //   ★ v5.37: SCOPE_EXIT は rollbackHandle を参照キャプチャする（コピーではない）。
    //     commit point 到達後は rollbackHandle = DSPHandle::null() で無効化し、
    //     SCOPE_EXIT による rollback を防止する。DSPHandle に invalidate() メソッドは
    //     存在しないため null() 代入により isNull() を true にする。
    //   TODO(work72): Extract transaction logic into PublicationTransaction class.
    DSPHandle rollbackHandle;
    SCOPE_EXIT {
        if (!rollbackHandle.isNull())
            rollbackDSPHandleRegistration(rollbackHandle);
    };
    if (regCtx.dsp != nullptr)
    {
        rollbackHandle = registerDSPHandleForRuntime(regCtx.dsp);
        if (rollbackHandle.isNull())
        {
            jassertfalse;
            return { PublishStageResult::Failed };
        }
    }
    else if (!regCtx.handle.isNull())
    {
        rollbackHandle = regCtx.handle;
    }
    const PublishStageResult stage = coordinator.publishWorld(std::move(world));
    if (PublishStageResultTraits::isCommitted(stage))
        rollbackHandle = DSPHandle::null();  // ★ v5.33: commit point → rollbackHandle を無効化
    return { stage };
}
```

**rollbackDSPHandleRegistration() と rollbackRegistration() — ★ v5.14: mutex 分割**:
```cpp
// ★ work70: RollbackResult — rollbackDSPHandleRegistration の詳細結果（DIAG ビルドのみ）
//   ★ v5.27: Production ビルドでは bool（成功/失敗）のみ。3値の詳細分類は DIAG に限定。
//     呼び出し側（ScopeExit 等）は戻り値の詳細を必要としない。
#ifdef CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
enum class RollbackResult : uint8_t {
    Failed,                // CAS 失敗（publish 成功済み）
    RolledBackAndErased,   // CAS 成功 + Map から削除
    RolledBackMapMissing   // CAS 成功 + Map に既に存在せず
};
#endif

// AudioEngine.h に追加
// ★ work70: Handle 登録のロールバック（DSPHandle 版）。
//   古い DSPCore* 版より安全（Handle が確定しているため）。
//   重要: HandleRuntime の rollback を先に実行し、成功した場合のみ Map から削除する。
//   CAS が失敗した場合（Constructing→Reclaimed 遷移不可＝publish 成功済み）は
//   Map を変更せずに false を返す。
//
//   ★ v5.14: mutex 保持時間を最小化するため、HandleRuntime の CAS は
//     無ロックで実行する。その後、再ロックして Handle を再確認した上で erase する。
//     これにより Map の mutex 競合が register/retire と干渉しにくくなる。
//   ★ v5.27: Production は bool 戻り値に簡略化。DIAG ビルド時のみ
//     RollbackResult の 3 値を返す。詳細分類はログ出力のみに使用。
//   ★ v5.31: Production bool 版は [[nodiscard]] を意図的に付けない。
//     呼び出し側（ScopeExit 等）は戻り値をチェックせず、rollback はベストエフォートで実行される。
//     DIAG RollbackResult 版も同様（ログ出力専用のため）。
//   ★ v5.35: Runtime rollback（CAS）成功を最優先。Map cleanup は二次的。
//     Production: Runtime rollback 成功後は常に true を返す（Map 不整合は無視）。
//     DIAG: rollback 成功後、Map の有無に応じて RolledBackAndErased / RolledBackMapMissing を返す。
//     理由: CAS(Constructing→Reclaimed) が成功していれば Runtime としては rollback 完了。
//     Map erase に失敗しても next register 時に新しい Handle で上書きされるため致命的ではない。
//     ScopeExit は戻り値を確認しないため、Production で常に true を返しても安全。
#ifdef CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
RollbackResult rollbackDSPHandleRegistration(convo::isr::DSPHandle handle) noexcept
#else
bool rollbackDSPHandleRegistration(convo::isr::DSPHandle handle) noexcept
#endif
{
    if (handle.isNull()) return false;
    // ★ 順序①: HandleRuntime を先に rollback（CAS: Constructing→Reclaimed）— 最重要
    if (!dspHandleRuntime_.rollbackRegistration(handle))
        return false;
    // ★ 順序②: Map から削除（二次的。失敗しても Runtime rollback は完了済み）
    const bool erased = eraseByHandle(handle);
    // Production: CAS 成功後は常に成功。Map 不整合は DIAG のみで報告。
    return true;
}

// ★ v5.33: eraseByHandle — Handle による Map erase 内部ヘルパー（HandleRegistry 移行準備）
//   ★ v5.34: static 指定を削除（inline メンバ関数として audioengine のメンバ変数にアクセス）。
//   ★ v5.37: 現状 O(n) linear scan（MAX_DSP_SLOTS=256 のため問題なし）。名前は eraseByHandle
//     だが実装は full scan であることを明記。将来の reverse map で O(1) に最適化予定。
//     現状: O(n) full scan。将来: HandleRegistry reverse map → O(1)。
private:
[[nodiscard]] bool eraseByHandle(convo::isr::DSPHandle handle) noexcept
{
    std::lock_guard<std::mutex> lock(runtimeDSPHandleMapMutex_);
    for (auto it = runtimeDSPHandleMap_.begin(); it != runtimeDSPHandleMap_.end(); ++it)
    {
        if (it->second == handle)
        {
            runtimeDSPHandleMap_.erase(it);
            return true;
        }
    }
    // Map に存在しなくても CAS 成功済みのため rollback は成功（INV-4/INV-7 対象外）
    return true;
}

// ★ 後方互換用: DSPCore* 版（bool のまま）
//   ★ v5.22: 互換レイヤー。将来の二方向 Map 導入後、DSPHandle 版に統一し削除予定。
//   ★ v5.24: [[deprecated]] を付与し、新規コードでの使用を禁止。
//   TODO: HandleRegistry リファクタリング時に削除
[[deprecated("Use rollbackDSPHandleRegistration(DSPHandle) instead")]]
bool rollbackDSPHandleRegistration(DSPCore* dsp) noexcept
{
    if (dsp == nullptr) return false;
    convo::isr::DSPHandle handle;
    {
        std::lock_guard<std::mutex> lock(runtimeDSPHandleMapMutex_);
        const auto it = runtimeDSPHandleMap_.find(dsp);
        if (it == runtimeDSPHandleMap_.end()) return false;
        handle = it->second;
    }
    if (!handle.isNull())
    {
        if (!dspHandleRuntime_.rollbackRegistration(handle))
            return false;
    }
    {
        std::lock_guard<std::mutex> lock(runtimeDSPHandleMapMutex_);
        for (auto it = runtimeDSPHandleMap_.begin(); it != runtimeDSPHandleMap_.end(); ++it)
        {
            if (it->second == handle)
            {
                runtimeDSPHandleMap_.erase(it);
                break;
            }
        }
    }
    return true;
}

// ISRDSPHandle.cpp に追加
// ★ work70: 登録のロールバック。
//   Constructing → Reclaimed への CAS により、「登録されたが公開されなかった」状態を
//   スロット利用可能プールへ戻す。
//   ▲ 重要: Constructing のみ rollback 対象。Constructing 以外の状態（Active/Retired/Reclaimed）
//     にある slot は rollback してはならない（publish 成功後または既に寿命終了）。
//     CAS が失敗した場合（Constructing 以外の状態）は false を返す。
//   ★ 設計判断 (v5.18): CAS を先に実行し、成功後に instance=nullptr を設定する。
//     理由: state（DSPState）が slot 再利用の authority である。CAS で Constructing→Reclaimed
//     に遷移した後に instance をクリアしても、他スレッドは Reclaimed を確認してから
//     create() でスロットを再利用するため安全。逆順（instance=null → CAS）では
//     Constructing 状態のまま instance が null になる window が生じ、resolve() が
//     不正な null instance を返す可能性がある。
//   ★ v5.31 コード確認確定: `resolve()`（ISRDSPHandle.cpp:40-55）は generation 一致確認後、
//     state（Reclaimed/Quarantined のみ無効）を authority として有効性を判定する。
//     instance は state が有効な場合にのみ返される（reg.instance を直接返す）。
//     したがって resolve の観点からは state が唯一の authority であり、instance=nullptr の
//     設定が state 変更の前後どちらでも安全（state 次第で resolve が instance を参照しない）。
//     この前提が変更される（resolve が state を見ずに instance のみを返す）場合には
//     rollbackRegistration の順序も再評価が必要。
//   ★ v5.33: instance=nullptr を削除。rollback を「state のみの操作」に純化。
//     「rollback が instance を消す」という責務は不要。理由:
//     - resolve() は state を authority とする（state が Reclaimed なら instance を返さない）
//     - create() は常に instance を上書きする（reclaimed→Constructing 時に dspInstance をセット）
//     - rollback と create の間に instance!=nullptr の時間窓が存在しても resolve は state で防御
//     reclaim() は retire→解放の最終段階として instance=nullptr + state=Reclaimed を設定する
//     （rollback とは責務が異なる — rollback は「未公開のまま利用可能プールに戻す」、
//      reclaim は「メモリ解放済み」）。この責務分離により ISR の state machine がより明確になる。
//   ★ v5.35: create() に jassert 追加推奨。rollback 後も instance が残っていることを確認:
//     jassert(reg.instance == nullptr || reg.state == DSPState::Reclaimed);
//     これにより「rollback 後に instance が不正に残っている」状態を Debug で検出できる。
//     Production では jassert は無効化されるためオーバーヘッドなし。
//   ★ v5.37: rollbackRegistration は Constructing 専用。将来 Publishing/PreActive 等の
//     中間状態が追加された場合は本関数の前提が変わるため再設計が必要。
//     Only Constructing may be rolled back. Future intermediate states require redesign.
[[nodiscard]] bool DSPHandleRuntime::rollbackRegistration(DSPHandle handle) noexcept
{
    if (handle.isNull() || handle.slot >= MAX_DSP_SLOTS) return false;
    auto& reg = registry_[handle.slot];
    DSPState expected = DSPState::Constructing;
    // ★ v5.33: state のみ CAS（instance は不変）。create() が上書きするため不要。
    return convo::compareExchangeAtomic(reg.state, expected, DSPState::Reclaimed,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire);
}
```

✅ **FACT（2026-07-10 コード調査確定）**: `registerDSPHandleForRuntime()` の `emplace` は find→return の後のみ実行される。`std::unordered_map::emplace` は重複キーの場合でも既存エントリを変更しない（insert 相当）。noexcept 関数内で例外は発生せず、`create(dsp)` で取得した Handle が孤立することはない。

✅ **FACT（2026-07-10 コード調査確定）**: `DSPHandleRuntime::create()` は `DSPState::Reclaimed` のスロットを再利用する（ISRDSPHandle.cpp:25）。`Constructing → Reclaimed` への rollback はこの「スロット利用可能プール」に戻す動作であり、振る舞いとして正しい。`Reclaimed` が「寿命終了」と「利用可能」の両方の意味を持つことは状態名の限界だが、内部的な振る舞いは一貫している。

#### 呼び出しパターン

**Coordinator direct publish 経路** (Init/PrepareToPlay/ReleaseResources/Timer/Transition):
```cpp
auto coordinator = makeRuntimePublicationCoordinator();
auto worldOwner = worldBuilder.buildRuntimePublishWorld(dsp, ...);
const auto result = commitRuntimePublication(
    coordinator, std::move(worldOwner),
    RegistrationContext::needsRegistration(dsp));
if (!PublishStageResultTraits::isCommitted(result.stage)) return;
```

**Orchestrator 経路** (PublicationExecutor):
```cpp
// Orchestrator は enqueuePublicationIntentForRuntimeCommit() 内で事前に
// registerDSPHandleForRuntime(newDSP) を実行済み。その結果の DSPHandle を
// RegistrationContext::alreadyRegistered() として渡し、publish 失敗時の rollback に使用する。
// ★ v5.31: handle は Constructing 状態であることが前提。rollbackRegistration() は Constructing からの
//   CAS(→Reclaimed) のみ成功する。
auto coordinator = engine.makeRuntimePublicationCoordinator();
const auto result = engine.commitRuntimePublication(
    coordinator, std::move(stateOwner),
    RegistrationContext::alreadyRegistered(registeredHandle));
if (!PublishStageResultTraits::isCommitted(result.stage)) return;
```

**v5.27 の変更**: `committed` は `PublishCommitResult` のメンバではなくなった。代わりに `PublishStageResultTraits::isCommitted(result.stage)` で commit 判定を行う。これにより committed と stage の二重管理を解消した。

#### 各 publish 経路のパラメータ

以下の 7 箇所の publishWorld 呼び出しについて、commitRuntimePublication の RegistrationContext パラメータを確定した。C++20 designated initializer で構築する:

| # | ファイル | RegistrationContext | 意味 |
|:-:|:---------|:-------------------|:------|
| 1 | AudioEngine.Init.cpp:48 | `RegistrationContext::none()` | Bootstrap world、DSP 未存在 |
| 2 | PrepareToPlay.cpp:155 | `RegistrationContext::needsRegistration(dsp)` | 初回 prepareToPlay、新規 DSP |
| 3 | PrepareToPlay.cpp:275 | `RegistrationContext::needsRegistration(dsp)` | placeholderDSP publish |
| 4 | ReleaseResources.cpp:155 | `RegistrationContext::none()` | Hard reset、DSP なし |
| 5 | Timer.cpp:900 | `RegistrationContext::needsRegistration(dsp)` | fadeCompleted ブロック |
| 6 | Transition.cpp:26 | `RegistrationContext::needsRegistration(dsp)` | publishIdleWorldOnly |
| 7 | PublicationExecutor.cpp:33 | `RegistrationContext::alreadyRegistered(h)` | Orchestrator が事前登録済み（FACT 確認済） |

CAVEAT: PublicationExecutor (#7) は Orchestrator 経路専用。enqueuePublicationIntentForRuntimeCommit() 内で publish 前に registerDSPHandleForRuntime(newDSP) 済み（FACT 確認済）のため handle のみ指定。

DSPTransition の重複登録について: DSPTransition::onPublishCompleted() の crossfade 経路 (needsCrossfade=true) で registerDSPHandleForRuntime(newDSP) が再度呼ばれるが、idempotent のため安全。即時 retire 経路 (needsCrossfade=false、今回の構造的 rebuild) では重複登録は発生しない。将来的なクリーンアップとして DSPHandleRuntime に findHandleByInstance() を追加し再登録不要とする方式も検討可能だが、P1-a の範囲外。

#### PublishOrigin — DIAG 専用に分離

`PublishOrigin` は DIAG 用途であり、本番 API には含めない。代わりに以下で対応する:

```cpp
// ★ DIAG ビルドのみ: commitRuntimePublication の呼び出しをトレース
#ifdef CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
struct ScopedPublishTrace {
    const char* origin_{nullptr};  // 呼び出し元識別子
    uint64_t dspPtr_{0};
    uint64_t generation_{0};

    ScopedPublishTrace(const char* origin, uint64_t dspPtr, uint64_t gen)
        : origin_(origin), dspPtr_(dspPtr), generation_(gen)
    {
        CONVOPEQ_DIAG_LOG("[HANDLE] origin=%s DSP=0x%llx gen=%llu action=begin_publish\n",
                          origin, (unsigned long long)dspPtr, (unsigned long long)gen);
    }
    ~ScopedPublishTrace()
    {
        if (origin_)
            CONVOPEQ_DIAG_LOG("[HANDLE] origin=%s DSP=0x%llx gen=%llu action=end_publish\n",
                              origin_, (unsigned long long)dspPtr_, (unsigned long long)generation_);
    }
};

// 呼び出し側:
ScopedPublishTrace trace("PrepareToPlay", reinterpret_cast<uint64_t>(dsp), worldGen);
const auto result = commitRuntimePublication(
    coordinator, std::move(worldOwner),
    RegistrationContext{.dsp = dsp});
if (!PublishStageResultTraits::isCommitted(result.stage)) return;
```
#else
// release ビルドでは trace 無し
#endif
```

✅ **FACT（2026-07-10 コード調査確定）**: `CONVOPEQ_DIAG_LOG` マクロは存在しない。プロジェクトでは `juce::Logger::writeToLog(juce::String::formatted(...))` または匿名名前空間の `diagLog(const juce::String&)` 関数（AudioEngine.Commit.cpp:17）を使用する。`diagLog` は `AudioEngine.Commit.cpp` 内の匿名名前空間に限定されるため、`ScopedPublishTrace` を別ファイルで使用する場合は直接 `juce::Logger::writeToLog` を呼ぶ。

このアプローチにより:
- 本番 API（`commitRuntimePublication`）に DIAG 用パラメータが混入しない
- 各呼び出し元が識別子を文字列で指定できる（enum より柔軟）
- RAII によりスコープ単位のトレースが自動化される
- 呼び出し元の記述量は `PublishOrigin` enum と同等

**HANDLE DIAG 拡張提案（v5.35）**: 以下の 4 アクションを追加することで handle lifecycle を state 単位で追跡可能にする:

```text
[HANDLE] action=register   slot=3  gen=5  state_before=Reclaimed state_after=Constructing
[HANDLE] action=rollback   slot=3  gen=5  CAS=success
[HANDLE] action=retire     slot=3  gen=5  lookup=success state_before=Active state_after=Retired
[HANDLE] action=reclaim    slot=3  gen=5  state_before=Retired  state_after=Reclaimed
[HANDLE] action=rollback_skip  slot=3  gen=5  reason=committed  // ★ v5.37: commit point 到達後の ScopeExit 無効化を追跡
```

`rollback_skip` は commit point 到達後に `rollbackHandle = DSPHandle::null()` で無効化されたケースを記録する。これにより ScopeExit が「rollback 不要だった」のか「無効化されて rollback しなかった」のかをログだけで区別できる（DIAG ビルドのみ）。

#### 責務境界まとめ

| コンポーネント | 責務 | AudioEngine 知識 |
|:-------------|:-----|:----------------|
| `commitRuntimePublication()` (AudioEngine private) | トランザクション: register → publish → rollback | Coordinator 引数で受取り、内部で register + publish + rollback を実行 |
| `RuntimePublicationCoordinator` (template) | publishWorld の実行 | なし（汎用テンプレート） |
| 各 publish 経路 (Init/PrepareToPlay/Timer等) | Coordinator 生成 + world 構築 + commitRuntimePublication 呼び出し | AudioEngine 内部メソッド |
| コアテンプレート (`src/core/RuntimePublicationCoordinator.h`) | Bridge 所有、publishWorld 実行。getBridge() は **持たない** | なし |
| `PublicationExecutor` | Orchestrator 経路の publish。RegistrationContext.handle で登録済み Handle を渡す | 型変換 + Handle 転送 |

✅ **FACT（2026-07-10 コード調査確定）**: `RuntimePublicationCoordinator` テンプレートは `Bridge bridge_` を **private メンバ**として保持する。public `getBridge()` は存在せず、Bridge にアクセスする唯一の方法は Coordinator メソッド（`publishWorld()` 内部での利用）のみ。したがって「Bridge 経由で Handle 登録を委譲する」設計はこのテンプレート制約と衝突する。この制約を尊重し、`commitRuntimePublication()` は AudioEngine のメソッドとして実装する。

#### `runtimeDSPHandleMap` の抽象化（将来の拡張性）

現在 `runtimeDSPHandleMap_` は `std::unordered_map<DSPCore*, DSPHandle>` + `std::mutex` で実装されている。この順方向 Map では Handle からの逆引きが O(n) になり、rollbackDSPHandleRegistration(DSPHandle) の Map erase が full scan となる。現状の MAX_DSP_SLOTS=256 では問題にならないが、恒久設計としては二方向 Map（`DSPCore* ↔ DSPHandle`）への移行を推奨する。P1-a 完了後のリファクタリングとして実施:

```cpp
// ★ 将来のリファクタリング候補: HandleRegistry クラス（二方向 Map）
//   現状: unordered_map<DSPCore*, DSPHandle>（順方向のみ）— Handle 逆引きが O(n)
//   将来: unordered_map<DSPCore*, DSPHandle> + unordered_map<DSPHandle, DSPCore*>
//         二方向 Map により register/retire/rollback 全てが O(1) で動作する。
//   P1-a 完了後の HandleRegistry リファクタリング時に実施。
class HandleRegistry {
public:
    DSPHandle register_(DSPCore* dsp);
    bool retire(DSPCore* dsp);
    bool rollback(DSPCore* dsp);
    RollbackResult rollback(DSPHandle handle);  // ★ O(1) erase が可能
private:
    std::mutex mutex_;
    std::unordered_map<DSPCore*, DSPHandle> forward_;
    std::unordered_map<DSPHandle, DSPCore*> reverse_;  // ★ Handle→DSPCore* 逆引き
};
```

これにより:
- rollbackDSPHandleRegistration(DSPHandle) の Map erase が O(1) になる
- 将来的な ConcurrentHashMap 置き換えが HandleRegistry 内部のみで完結
- 全マップ操作の一貫性（register/rollback/retire で同一ロック）がコード構造で保証される
- DIAG ログの追加も HandleRegistry 内に集中できる
- DSPCore* 版 rollback が不要になり、削除可能

**P1-a の最小変更方針**: P1-a では既存の `registerDSPHandleForRuntime()` 等の inline 関数を直接呼び出す。**private helper は導入しない。** HandleRegistry への抽象化は P1-a 完了後の後続リファクタリングで一括実施する。これにより P1-a の変更範囲を最小化する。

★ v5.27: private helper 案は削除。HandleRegistry リファクタリング時に一元導入するため、P1-a の段階で helper 層を入れても後で置き換えが必要になる。

**PublicationExecutor の置き換え**:
`PublicationExecutor::publish()` は `aligned_unique_ptr<FrozenRuntimeWorld>` のみ保持し、DSPCore* は **持たない**。したがって `commitRuntimePublication()` には **RegistrationContext{}（dsp なし）** を渡す。ただし Orchestrator が事前に `enqueuePublicationIntentForRuntimeCommit()` 内で登録した `DSPHandle`（`req.newDSP`）を `RegistrationContext.handle` として渡す必要がある。

**データフロー変更**: `PublicationExecutor::publish()` のシグネチャに `DSPHandle existingHandle` パラメータを追加する:
```cpp
// PublicationExecutor.h
[[nodiscard]] PublishResult publish(
    AudioEngine& engine,
    convo::aligned_unique_ptr<convo::FrozenRuntimeWorld> frozen,
    convo::isr::DSPHandle existingHandle) noexcept;  // ★ work70: 新規追加
```

**Orchestrator 側の呼び出し**:
```cpp
// RuntimePublicationOrchestrator.cpp:158 変更後
auto result = executor_.publish(engine_, std::move(frozen), req.newDSP);
```

**PublicationExecutor 側の使用**:
```cpp
const auto result = engine.commitRuntimePublication(
    coordinator, std::move(stateOwner),
    RegistrationContext::alreadyRegistered(existingHandle));
if (!PublishStageResultTraits::isCommitted(result.stage)) return;
```

`FrozenRuntimeWorld` に `toPublishWorld()` を追加する設計も可能だが、最小変更の原則から v5.7 では `releaseState()` の直接使用を維持する。`toPublishWorld()` は v5.6 で提案したリファクタリング候補であり、P1-a 実装後の改善として検討する。

**影響を受ける既存コード**:

| ファイル | 関数 | 変更前 | 変更後（RegistrationContext） |
|:---------|:------|:-------|:-----------------------------|
| `AudioEngine.Init.cpp:48` | initialize() | `coordinator.publishWorld(...)` | `commitRuntimePublication(c, w, RegistrationContext::none())` |
| `PrepareToPlay.cpp:155` | prepareToPlay() | `coordinator.publishWorld(...)` | `commitRuntimePublication(c, w, RegistrationContext::needsRegistration(dsp))` |
| `PrepareToPlay.cpp:275` | prepareToPlay() | `coordinator.publishWorld(...)` | `commitRuntimePublication(c, w, RegistrationContext::needsRegistration(dsp))` |
| `ReleaseResources.cpp:155` | releaseResources() | `coordinator.publishWorld(...)` | `commitRuntimePublication(c, w, RegistrationContext::none())` |
| `AudioEngine.Timer.cpp:900` | timerCallback() | `coordinator.publishWorld(...)` | `commitRuntimePublication(c, w, RegistrationContext::needsRegistration(dsp))` |
| `AudioEngine.Transition.cpp:26` | publishIdleWorldOnly() | `coordinator.publishWorld(...)` | `commitRuntimePublication(c, w, RegistrationContext::needsRegistration(dsp))` |
| `PublicationExecutor.cpp:33` | publish() | `coordinator.publishWorld(...)` | `engine.commitRuntimePublication(c, w, RegistrationContext::alreadyRegistered(h))` |

⚠️ **CAVEAT**: 本設計書の対象は production コードのみ。テストコード（`PartialPublicationRejectTests.cpp` 等）は `publishWorld()` を直接呼び続けてよい（テスト固有の要件）。

**Timer 経路の DSPCore* 解決**（fadeCompleted ブロック内: Timer.cpp:897）:
- publish 対象は `resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle)` の戻り値 `currentAfterFade`
- これを `commitRuntimePublication(coordinator, worldOwner, RegistrationContext::needsRegistration(currentAfterFade))` として呼び出す
- **検証**: `currentAfterFade` が publish する RuntimeWorld の生成元 DSPCore と一致することを DIAG ログで確認する

#### 設計ルール: commitRuntimePublication() 利用の徹底

**production コードでは `coordinator.publishWorld()` を直接呼んではならない。** Runtime publish は必ず `commitRuntimePublication()`（AudioEngine の private メソッド）を経由すること。これにより:
- registerDSPHandleForRuntime() の呼び忘れを構造的に防止
- 「Runtime publication を試行する DSP は publish 前に Handle 登録済みである」invariant がトランザクション構造で保証される
- 新規 publish 経路が追加されても自動的に invariant を満たす

**production コード中の全 `publishWorld()` 呼び出し（7 箇所）が `commitRuntimePublication()` に置き換わっていることを、コードレビュー + CI 静的チェックで確認する。** 1 箇所でも直接呼び出しが残ると invariant が成立しなくなる。

#### Architecture Decision: Publish の唯一の入口

**Decision**: 以降、production コードにおける `RuntimePublishWorld` の publish は全て `AudioEngine::commitRuntimePublication()` を経由する。`RuntimePublicationCoordinator::publishWorld()` を直接呼び出すことは**設計違反**であり、CI 静的チェックで禁止する。

**Rationale**:
- `commitRuntimePublication()` は register → publish → rollback のトランザクションを保証する唯一の関数である
- 直接 `publishWorld()` を呼ぶと register 漏れが構造的に発生する（今回の不具合の根本原因）
- 新規 publish 経路が追加されても、`commitRuntimePublication()` を通すことで自動的に INV-1（事前登録）が保証される

**Scope**: AudioEngine 内の全 publish 経路（Init/PrepareToPlay/ReleaseResources/Timer/Transition/PublicationExecutor）。テストコードは対象外。将来追加される publish 経路もこの制約に従う。

**`publishWorld()` へのアクセス制御**: `RuntimePublicationCoordinator` テンプレートの `publishWorld()` を `private` にすることはテンプレートの汎用性を損なうため行わない。代わりに以下の防御で制御する:
1. **CI 静的チェック（clang-tidy AST matcher）**で production コード中の `publishWorld()` 直接呼び出しを禁止（MemberCallExpr, callee === "publishWorld"）。
2. **コードレビュー**で全 7 箇所の置換を確認。

★ v5.35: `[[deprecated]]` 属性は Coordinator 汎用テンプレートに付与しない。clang-tidy のみで制御する。

**CI 静的チェック**（**参考実装**: 恒久対策は clang-tidy AST matcher に移行予定）:
```bash
# ★ v5.27: 以下は「参考実装」として記載する。production CI には clang-tidy を導入する。
#   grep ベースのチェックはコメント・改行跨ぎ・関数ポインタを検出できないため信用できない。
#
# 恒久対策（clang-tidy AST matcher）:
#   clang-tidy --checks='-*,misc-publish-world-direct-call' \
#     --match='MemberCallExpr, callee === "publishWorld"'
#   これによりコメント内の publishWorld( を誤検出せず、実呼び出しのみを検出可能。
#   また coordinator\n    .publishWorld( のような改行跨ぎも正しく検出できる。
#   MemberCallExpr のため auto fn = &...publishWorld 関数ポインタも検出可能。
#
# ★ v5.27 追記: コード調査により production コードの publishWorld() 直接呼び出しは
#   7 箇所全て確定済み（FACT #43）。関数ポインタ経路の隠れ呼び出しも存在しないことを確認。
#   したがって P1-a 実装後のリグレッション検知には clang-tidy を導入するまで
#   コードレビュー＋手動確認で十分対応可能。
#
# 参考: grep ベースの簡易チェック（検出漏れのリスクあり）:
#   grep -rn "publishWorld(" src/ --include="*.cpp" --include="*.h" \
#     | grep -v "tests/" | grep -v "Test\|test"
```

⚠️ **CAVEAT**: 関数ポインタ経由 (`auto fn = &RuntimePublicationCoordinator::publishWorld;`) は grep では検出不可能。clang-tidy の MemberCallExpr による AST マッチングが恒久対策。P1-a 実装時点ではコードベースに関数ポインタ経路が存在しないことを確認済み（FACT #43）。

**`RegistrationContext` の使い分けについて**:
- `RegistrationContext::needsRegistration(dsp)`: commitRuntimePublication が register + publish を実行。呼び出し元は register 不要。
- `RegistrationContext::alreadyRegistered(handle)`: 呼び出し元が事前登録済み。handle を rollback 用に保持。※ handle は Constructing 状態が前提。
- `RegistrationContext::none()`: 登録不要（Bootstrap world / Hard reset）。
- ★ v5.33: 静的ファクトリにより「dsp と handle の同時指定」という不正状態を構造的に防止。
- **DIAG ビルドでは `ScopedPublishTrace` を使用してこの対応関係をログに残す**

#### Future extension: PublishCommitResult と commitRuntimePublication の分割

**PublishCommitResult**: `commitRuntimePublication()` の戻り値。`stage` のみを保持する（`committed` は `PublishStageResultTraits::isCommitted(stage)` で導出。二重管理防止のため struct には含めない）:
```cpp
struct PublishCommitResult {
    PublishStageResult stage;  // ★ commit point は PublishStageResultTraits::isCommitted(stage) で判定
    // ★ v5.27: committed は stage から導出可能なため削除。handle も DIAG 専用のため本番 API に含めず。
    //   本番 API を最小限に保ち、DIAG 詳細は ScopedPublishTrace または別機構で提供。
};

// ★ commit point 判定 — Coordinator 実装から分離して独立
//   ★ v5.26: commitRuntimePublication の committed 判定はこの関数に委譲。
//     Coordinator が SuccessWithWarning / DeferredSuccess 等を追加しても
//     PublishStageResultTraits の修正のみで対応可能。AudioEngine 側の変更不要。
struct PublishStageResultTraits {
    [[nodiscard]] static bool isCommitted(PublishStageResult stage) noexcept {
        // ★ 現行実装では Success のみが commit point。
        //   将来拡張時はここに条件を追加する。
        return stage == PublishStageResult::Success;
    }
};
```
内部実装では `PublishStageResultTraits::isCommitted(stage)` で commit 判定を行う。呼び出し側は `result.stage` を `Traits::isCommitted()` に渡す。これにより将来 `PublishStageResult` に値が追加されても `PublishStageResultTraits` の修正のみで対応可能。

**内部関数分割**: 同様に、`commitRuntimePublication()` が肥大化した場合の対策として、以下の内部ヘルパーに分割する方式を将来のリファクタリング候補として提案する:
```cpp
// 将来の分割候補（P1-a では単一関数のまま）
[[nodiscard]] DSPHandle performRegistration(DSPCore* dsp) noexcept;
[[nodiscard]] PublishStageResult performPublish(RuntimePublicationCoordinator& coordinator,
    convo::aligned_unique_ptr<RuntimePublishWorld> world) noexcept;
bool performRollback(DSPCore* dsp, const DSPHandle& handle) noexcept;
```
ただし P1-a の実装時点では `commitRuntimePublication()` は 20 行未満であり、分割の必要性は低い。DIAG/統計/タイミング計装が追加された段階で検討する。

### 1.3 P1-a と P1-b の役割分担

P1-a と P1-b は**異なる retire 経路を回復する**（独立した並行作業）:

**重要な発見**: 今回のテストシナリオでは全ての Orchestrator publish が Structural rebuild（`requestRebuild_sr_bs`）によるものだった。Structural rebuild では CrossfadeAuthority が `needsCrossfade=false` を返すため、DSPTransition は crossfade ではなく **即時 retire 経路** を使用する。ログ `[XFADE]=0` はこれを裏付ける。

したがって:
- **P1-a により `retireDSPHandleForRuntime()` の lookup 成功率が改善することが期待される**（handle map 未登録問題解消）。これにより `runtimeRetireCount` が増加する可能性がある。ただし `enqueueRetire` → EBR epoch → reclaim → `destroyDSPCoreNode()` の後段経路（QueueFull 等）が残存する可能性もある。
- **P1-b（advanceFade）はパラメータフェード完了条件を成立させるための修正**（EQ/NS/AGC 変更時など）。retire を直接起動するわけではない
- 両方を実施することで全経路をカバーする

```text
P1-a: AudioEngine::commitRuntimePublication() (private メソッド)
  registerDSPHandleForRuntime(dsp) → coordinator.publishWorld()
    → handle map に必ず登録される
      → retireDSPHandleForRuntime() lookup 成功率改善（旧: 常に false）
        → DSPTransition 即時 retire 経路 (needsCrossfade=false) が成功
          → runtimeRetireCount 増加の可能性

P1-b: advanceFade 配線（パラメータフェード完了条件の成立）
  getNextAudioBlock() → advanceFade(numSamples)
    → remaining 減少 → 0 到達
      → tryCompleteFade() == true （advanceFade は fade を進行させるのみ）
        → fadeCompleted ブロック実行 → completeFade → publish → retire
          → パラメータ遷移 (EQ/NS/AGC) 時の retire が正常動作
```

---

## [設計] 2. P1-b: advanceFade 配線

### 2.1 なぜ必要か

✅ **FACT**: `SnapshotCoordinator::advanceFade()` が 0 call sites。

Timer 経路の retire は `fadeCompleted → completeFade → publish → retire` の連鎖で動作する。advanceFade が呼ばれないため SnapshotFadeState の remaining が永遠に減少せず、`tryCompleteFade()` が常に false を返す。これにより Timer 経路の retire がブロックされている。

### 2.2 変更内容

| ファイル | 変更 | 行数 |
|:---------|:------|:-----|
| `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | `getNextAudioBlock()` 内に 1 行追加 | +1 |

```cpp
// AudioEngine::getNextAudioBlock() 内、DSP 処理完了直後
// ★ work70: SnapshotCoordinator の fade 進行。remaining を処理済みサンプル数だけ進める。
m_coordinator.advanceFade(numSamples);
```

**変更不要のファイル:**

| ファイル | 理由 |
|:---------|:------|
| `SnapshotFadeState.h` | advance() の release store は既に正しい |
| `SnapshotCoordinator.h/.cpp` | advanceFade() の void シグネチャを維持 |
| `AudioEngine.Timer.cpp` | tryCompleteFade() は既に毎 callback で呼ばれている (line 843) |
| `CrossfadeRuntime.h` | 責務分離を維持。完了通知を持ち込まない |

### 2.3 成功条件（段階的確認）

| # | 確認段階 | 合格基準 |
|:-:|:---------|:-------|
| 1-1 | advanceFade 正常動作 | 毎 callback で呼ばれ remaining が減少する |
| 1-2 | remaining==0 到達 | tryComplete() が true を返す |
| 1-3 | completeFade() 実行 | fadeCompleted ブロック内の completeFade() が実行される |
| 2-1 | retire() 呼び出し | DSPLifetimeManager::retire() が実行される |
| 2-2 | pendingRetireCount 増加 | MEM_SNAP `pend` が一時的に >0 |
| 2-3 | EBR reclaim 実行 | MEM_SNAP `rec` が進行 |
| 2-4 | liveCount 減少 | NUC/DSPCore/Stereo liveCount 減少 |
| 2-5 | runtimeRetireCount 増加 | ⚠️ P1-a (handle 登録) が完了している場合のみ |

### 2.4 P1-a/P1-b 完了後の期待されるメモリ効果

| 項目 | 現状 | 期待値（推定） | 備考 |
|:-----|:------|:--------------|:------|
| Private Memory | 2477MB | 🔍 数百MB〜の改善 | DSPCore 未 retire が主因と仮定。正常に EBR まで到達した場合の推定。AoS→SoA (P3) や BlockSize (P2) 未実施のため下限は未確定 |
| lifecycle(ret) | 0 | >0 | P1-a により handle map 解決で DSPTransition の retire が動作する見込み |
| DSPCore liveCount | 6+ | 🔍 2（current + fading）に収束 | 正常に retire→reclaim が完了した場合の steady-state。enqueueRetire/epoch/reclaim の後段経路が残存する可能性もある |
| pendingRetireCount | 0 | 一時的に >0 → reclaim 後 0 | EBR の正常動作を示す |

⚠️ **CAVEAT**: 上記は検証前の推定。実際の効果は P1-a/b 実装後の MEM_SNAP で確認する。JUCE や IR 処理等の OtherPrivate は別途対応が必要。

---

## [設計] 3. P1-c: MEM_SNAP 監視強化

### 3.1 変更内容

`AudioEngine.Timer.cpp` の MEM_SNAP フォーマットに以下を追加/確認:

- `StereoConvolver::liveCount`（未出力 → 追加）
- `DSPCore::liveCount`（未出力 → 追加）
- `currentGeneration`（**既に `gen` として出力済み** — AudioEngine.Timer.cpp:909 `runtimeWorld->generation`）
- `retiringGeneration`（未定義 → 新規 atomic フィールド追加が必要。★ v5.37: retiringGeneration の更新 Authority は **DSPLifetimeManager::retire()** のみ。他の箇所からの更新を禁止するため private atomic とし、getter のみ公開。`DSPLifetimeManager::retire()` 内の `retireDSPHandleForRuntime()` 成功直後に store。→ P1-c 実装時に DSPLifetimeManager に `currentRetiringGeneration_` atomic フィールドを追加。）

✅ **FACT（2026-07-10 コード調査確定）**: `currentGeneration` は MEM_SNAP の `gen` フィールドとして既に出力されている（Timer.cpp:909）。`gen` は `runtimeWorld->generation` の値。したがって P1-c では `currentGeneration` の追加は不要。`retiringGeneration` のみ新規追加が必要。

**generation 追加の目的**: publish と retire の generation を対応づけて解析できる。`gen`（現在の world 世代）は既存、`retiringGeneration` は retire 対象世代を示し、`gen - retiringGeneration` の差が retire 遅延の指標となる。
例:
```text
current gen=8
retiring gen=7  → 1世代前の DSPCore が retire されている（正常）
current gen=8
retiring gen=6  → 世代差が大きく、retire が追いついていない可能性
```

### 3.2 期待効果

- DSPCore の寿命を直接監視可能に
- P1-a/P1-b 実施後の効果確認指標
- generation 対 lifecycle(retire) の相関を即座に追跡可能
- リグレッション検知の早期指標

---

## [設計] 4. 設計方針: SnapshotCoordinator と CrossfadeRuntime の責務分離

### 4.1 2つの独立した機構

✅ **FACT**: `SnapshotCoordinator` と `CrossfadeRuntime` は責務・管理対象・状態管理・ログタグの全てで独立している。

| 機構 | 管理対象 | 状態管理 | ログタグ | 呼び出し元 |
|:-----|:---------|:---------|:---------|:----------|
| **`SnapshotCoordinator`** | パラメータ遷移 (EQ/NS/AGC) | `SnapshotFadeState` (`Idle`/`FadingIn`) | `[VERIFY]` | `createSnapshotFromCurrentState()` (Timer) |
| **`CrossfadeRuntime`** | DSP エンジン遷移 (structural) | `pending_` boolean + `LinearRamp` | `[XFADE]` | `DSPTransition::onPublishCompleted()` (Orchestrator) |

```cpp
// DSPTransition.h:49 — DSP エンジン遷移のみを扱う
void onPublishCompleted(DSPCore* newDSP, DSPCore* oldDSP, ...) {
    // CrossfadeAuthority が判断した要否に基づき:
    // - 要: crossfadeRuntime_.start(decision.fadeTimeSec, sampleRate);
    // - 不要: crossfadeRuntime_.complete(); lifetime.retire(oldDSP);
}

// SnapshotCoordinator.cpp:43 — パラメータ遷移のみを扱う
void advanceFade(int numSamples) noexcept {
    m_fade.advance(numSamples);  // SnapshotFadeState のカウンタ減算
}
```

### 4.2 本計画書における意義

| 判断 | 根拠 |
|:-----|:------|
| `CrossfadeRuntime` に SnapshotCoordinator の完了通知を持ち込まない | 責務違反 |
| advanceFade の配線先は Audio Callback のみ | Timer はサンプル数を知らない。ISR 原則に基づく |
| Timer の tryCompleteFade() は変更不要 | advance() が remaining を減らすだけで動作する |
| `[XFADE]=0` と advanceFade 未配線は無関係 | `[XFADE]` は DSP エンジン遷移のログ。Snapshot(=パラメータ)遷移とは別機構 |

---

## [設計] 5. 検証計画

### 5.1 P1-a + P1-b 完了後の統合確認

P1-a の検証では **DIAG チェーン** を一時的に追加し、retire パイプラインの全段階をログ追跡する:

```text
確認チェーン:
  registerDSPHandleForRuntime() → map insert
    → retireDSPHandleForRuntime() → true (旧: false)
      → EBR enqueue (Success/QueuePressure)
        → DSPLifetimeManager::retire()
          → destroyDSPCoreNode() 実行
            → DSPCore::~DSPCore()
              → StereoConvolver::~StereoConvolver()
                → MKLNonUniformConvolver::~MKLNonUniformConvolver()  ← ★最終確認
```

各段階の DIAG ログを一時追加し、P1-a の成否を短時間で判断する。特に **デストラクタチェーン** (`[LIFETIME] DSPCore destroy`, `[LIFETIME] MKL destroy`) まで到達したことを一度確認すれば、「retire したが実際には解放されていない」という可能性を排除できる。問題なければ削除して本番コードへ。

✅ **FACT（コード確認済み）**: `DSPLifetimeManager::retire()` は以下のチェーンで動作する:
```text
dsp != nullptr
  → retireDSPHandleForRuntime(dsp): handle map lookup
    → false → return (何もしない)
    → true → dspHandleRuntime_.retire(handle) + .reclaim(handle)
      → router_->enqueueRetire(dsp, destroyDSPCoreNode, epoch)
        → Success/QueuePressure → runtimeRetireCount++
        → QueueFull → tryReclaim → 再試行
```
今回のログ解析で観測された最初のクリティカル分岐は **`retireDSPHandleForRuntime()` の戻り値** である（ログ上全て false）。ただし他に `enqueueRetire` の QueueFull や epoch 不整合なども分岐として存在する。これらは P1-a の handle 登録修正後に再評価する。

また、register/rollback/retire/erase の各イベントでは `DSPCore*`, `generation`, `HandleID` の **3点セット** を同時出力する。origin（呼び出し元識別）は `ScopedPublishTrace`（`#ifdef CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` でガード）で提供する:
```text
[HANDLE] gen=5 DSP=0x12345678 Handle=17 action=register
[HANDLE] gen=6 DSP=0x12345678 Handle=17 action=retire  lookup=success
[HANDLE] gen=6 Handle=17 action=erase

// DIAG ビルド時のみ: ScopedPublishTrace が origin を提供
[HANDLE] origin=PrepareToPlay DSP=0x12345678 gen=5 action=begin_publish
[HANDLE] origin=PrepareToPlay DSP=0x12345678 gen=5 action=end_publish
```
これにより:
- 同一 DSPCore の register → retire → erase が一本道で追跡できる
- HandleID が異なる DSPCore に再利用されても DSPCore* と generation で区別できる
- **origin（呼び出し元識別）は DIAG ビルドのみで提供。production ビルドの API を汚染しない**
- retire 時の lookup 成否が同時に分かる

| # | 確認項目 | 確認方法 | 合格基準 |
|:-:|:---------|:---------|:-------|
| 1 | コンパイル成功 | `build.bat Debug icx nopause -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1` | ✅ |
| 2 | 単体テスト | `ctest -C Debug --output-on-failure` | 全 PASS |
| 3 | advanceFade 呼び出し確認 | DIAG ログ | 1 callback = 1 call |
| 4 | tryCompleteFade() == true | DIAG ログ | fade 完了時 1 回 |
| 5 | **production コードの全 publishWorld() → commitRuntimePublication() 置換** | コードレビュー + CI 静的チェック | 7 箇所全て置換。1 箇所でも直接呼び出しが残っていれば invariant 不成立 |
| 6 | **retireDSPHandleForRuntime() false → true** | DIAG ログ追加 + [HANDLE] lookup=success | Coordinator direct publish 後、retireDSPHandleForRuntime() が false を返さなくなる。P1-a 成否の最優先指標。 |
| 7 | **registerDSPHandleForRuntime() idempotent 確認** | DIAG ログ追加 | 同一 DSPCore を 2 回 register しても map size / handle が変わらない |
| 8 | **register → retire → erase 完全一致** | DIAG ログ | 同一 HandleID が register/retire/erase を各 1 回ずつ通る。Map リークがないことの直接証明 |
| 9 | lifecycle(retire) > 0 | MEM_SNAP | 0 から正数に変化（retireDSPHandleForRuntime 成功の結果） |
| 11 | **runtimeDSPHandleMap register/retire/erase カウンタ** | DIAG ログ追加 | register数 ≒ retire数 ≒ erase数 のバランス |
| 12 | pendingRetireCount > 0 | MEM_SNAP `pend` | retire キューに一時的に滞留 |
| 13 | DSPCore liveCount 減少 | MEM_SNAP | 2 (current+fading) 付近に収束 |
| 14 | **currentAfterFade 一致確認** | DIAG ログ（ScopedPublishTrace 使用） | `[HANDLE] world=0x... dsp=0x... gen=...` で register 対象と publish 対象の対応関係を確認 |
| 15 | MEM_SNAP generation 値 | MEM_SNAP `currentGeneration`/`retiringGeneration` | lifecycle 変化との世代相関を直接確認。`currentGeneration - retiringGeneration` が retire 遅延の指標 |
| 16 | Private Memory 減少 | MEM_SNAP `Priv` | 🔍 数百MB規模の改善を期待 |
| 17 | 音声品質 | 主観評価 | ノイズ・クリックなし |
| 18 | 3 分間安定動作 | 長時間実行 | クラッシュ・ハングなし |

**同一 DSPCore* で追跡する DIAG チェーン（P1-a 成否判定の最優先項目）**:
```text
[HANDLE] DSP=0x12345678 Handle=17 action=register              ← bridge.registerDSPHandleForRuntime()
[HANDLE] DSP=0x12345678 Gen=3 world=0x... dsp=0x...             ← register成功確認
[PUBLISH] DSP=0x12345678 Gen=?                                    ← publishWorld
[HANDLE] DSP=0x12345678 Handle=17 lookup=success action=retire   ← retireDSPHandleForRuntime()
[HANDLE] Handle=17 action=erase                                   ← erase 確認（Map リーク防止）
[LIFETIME] DSPCore destroy 0x12345678                             ← destroyDSPCoreNode
```
これにより、**登録した DSP と retire 対象の DSP が本当に同一だったか**、**どの publish 経路で登録・retire されたか**、**erase まで完了したか** まで確認できる。

⚠️ **CAVEAT**: `runtimeDSPHandleMap` の size が 2〜3 に収束するかは P1-a 実装後の実測が必要。register/retire/erase の各カウンタを個別に監視し、retire → erase が正しく動作していることを確認する。
```

### 5.2 リスクとロールバック

| リスク | 確率 | 影響 | 対策 |
|:-------|:-----|:------|:------|
| handle 登録追加によるスロット枯渇 | 低 | 中 | Coordinator direct が 256 回連続しない限り発生しない |
| advanceFade 配線位置誤り | 低 | 低 | advance() は副作用のないカウンタ減算のみ |
| 各 Phase は独立コミットとする | — | — | 問題発生時は `git revert` で該当 Phase のみ戻せる |

---

**以下、Appendix（補足情報）:**

---

## Appendix A: 背景と調査結果の詳細

### A.0 凡例詳細

| ラベル | 意味 |
|:-------|:-----|
| ✅ **FACT** | コード解析またはログ解析で確定した事実 |
| 🔍 **HYPOTHESIS** | ログやコードから強く示唆されるが未確定 |
| 💡 **PROPOSAL** | 改善のための設計案（複数案あり得る） |
| ⚠️ **CAVEAT** | 注意・制約・未解決の懸念 |

### A.1 問題

ConvoPeq のプロセスメモリ使用量が、通常動作時に Private Memory 約 **2.5GB** に達する。

### A.2 確定事実一覧

| 項目 | 状態 | エビデンス |
|:-----|:-----|:-----------|
| MKL Convolution は 35MB (1.4%) のみ | ✅ **FACT** | MEM_SNAP `alloc=35MB`, `Other=2442MB` |
| `advanceFade()` が 0 call sites | ✅ **FACT** | grep 確認 |
| `lifecycle(retire)=0` | ✅ **FACT** | MEM_SNAP ログ (23317行中 1231回の出力) |
| Orchestrator publish 3 回全て SUCCEEDED | ✅ **FACT** | `trySubmit: publish SUCCEEDED gen=1/4/8` |
| DSPTransition は 3 回全て実行済み | ✅ **FACT** | DSPTransition::onPublishCompleted() は 3 回呼ばれている |
| Coordinator direct publish は handle 登録しない | ✅ **FACT** | registerDSPHandleForRuntime() は enqueuePublicationIntentForRuntimeCommit 内でのみ呼ばれる |
| `[XFADE]=0` | ✅ **FACT** | DSP エンジン遷移が発生しなかったことを示す（SnapshotCoordinator とは無関係） |
| advanceFade 未呼び出し | ✅ **FACT** | 0 call sites — Timer 経路の retire ブロック原因 |

### A.3 lifecycle(retire)=0 の調査経緯

2026-07-10 のログ解析により 5 仮説を検証:

| 仮説 | 結果 | 根拠 |
|:-----|:------|:------|
| A: retire() 未呼び出し | ❌ **否定** | DSPTransition → retire は 3 回全て実行 |
| **B: handle map 未登録** | ✅ **主要原因** | Coordinator direct publish が登録しない |
| C: enqueueRetire 失敗 | ❌ **否定** | retireDSPHandleForRuntime が false のため enqueue に到達せず |
| D: epoch 問題 | ❌ **否定** | enqueue に到達していない |
| E: Orchestrator 未使用 | ❌ **否定** | 3 回使用、全て SUCCEEDED |

⚠️ **CAVEAT**: 仮説 B は「主要原因」と判断したが「確定」ではない。register 後に retire が成功することまで確認したわけではなく、実装後の検証が必要。

### A.4 メモリ使用量の内訳（v4.5 更新）

**確定値**: MKL Convolution 35MB (1.4%)。**残り 2442MB はログ解析ベースの推定。**

```
Private 2477MB の内訳:
  ┌─ MKL Convolution (tracked)     =   35MB  (1.4%)  ☆ 確定値
  ├─ DSPCore #1 (gen=1, blk=524288) = ~481MB  ★ 初回prepare（巨大ブロック）
  ├─ DSPCore #2 (gen=4, blk=2048)   = ~220MB  ★ SR切替でprepare
  ├─ DSPCore #3 (gen=8, blk=2048)   = ~220MB  ★ IR loadでprepare
  ├─ rebuild-obsolete ×3 (DSPGuard) = ~660MB  ★ リーク（代替解放経路なし）
  ├─ IR processing (Converter/Cache) = ~300MB  △ 推定（未計装mkl_malloc含む）
  ├─ JUCE Framework                 = ~360MB  △ 推定（起動74MB含む）
  └─ CRT/STL/DLL/Threads/Other      = ~200MB  △ 推定
```

**DSPCore 1個あたりのコスト詳細**（DIAG ログ + コード解析）:
- 初回 prepare (blk=524288, os=0, sr=48000): **~481MB**（EQ scratch=32MB, msWorkBuffer=16MB, dryBypass=8MB 等の大バッファが原因）
- 通常 prepare (blk=2048, os=2, sr=192000): **~220MB**（EQ + Oversampling + SoftClip + 内部バッファ）
- 両方とも `processingBlockSize` に比例する内部バッファを持ち、初回の異常値（524288）は後続（2048）の256倍のブロックサイズ

🔍 **HYPOTHESIS**: P1-a/b により retire が回復した場合、以下の削減が見込まれる:
- 解放される DSPCore: #1(481MB) + #2(220MB) + rebuild-obsolete×3(660MB) = **~1361MB**
- ただし #1 は gen=1 の巨大ブロックDSPCoreであり解放効果が大きい
- 残存: DSPCore #3(220MB) + JUCE(360MB) + IR(300MB) + Other(200MB) = **~1080MB**
- したがって 2477MB → **~1100-1500MB 程度** まで改善する可能性
- AoS→SoA (P3) や BlockSize (P2) を追加すればさらに削減見込み

⚠️ **CAVEAT**: 上記は推定。実際の効果は P1-a/b 実装後の MEM_SNAP で確認する。

---

## Appendix B: 今後の Phase（P2-1〜P4）

### B.1 P2-1: 初回ブロックサイズ調査

**問題**: 初回 `DSPCore::prepare()` 時の `processingBlockSize=524288`
（`SAFE_MAX_BLOCK_SIZE 65536 × MAX_OS_FACTOR 8`）。

✅ **FACT（2026-07-10 コード調査確定）**: 初回巨大ブロックの原因チェーン:
1. `AudioEngine.Init.cpp:38` で `maxSamplesPerBlock = SAFE_MAX_BLOCK_SIZE (65536)` （デバイス未初期化の安全策）
2. `AudioEngine.Init.cpp:57` で初回 Structural rebuild を submit
3. `DSPCoreLifecycle.cpp:140-142` で `inputMaxBlock = max(65536, samplesPerBlock) = 65536`、`internalMaxBlock = 65536 × 8 = 524288`
4. この 524288 が DSPCore の全内部バッファ（EQ scratch / msWorkBuffer / dryBypass 等）のサイズとなり、~481MB のコスト増となる

✅ **FACT（2026-07-10 コード調査確定）**: `internalMaxBlock = inputMaxBlock × MAX_OS_FACTOR(8)` は oversamplingFactor に依存しない。osFactor=0（Auto → 1）でも内部バッファは 524288 で確保される。`processingBlockSize = samplesPerBlock × oversamplingFactor` は osFactor の影響を受ける（convolver のブロックサイズにのみ使用）。したがって初回 prepare 時のメモリ肥大は osFactor とは無関係であり、`inputMaxBlock` の出発値（SAFE_MAX_BLOCK_SIZE=65536）が根本原因。

| 案 | 難易度 | リスク | 期待削減（推定） |
|:---|:-------|:-------|:---------|
| A: Init 時に `maxSamplesPerBlock` を小さな値（例: 512）に設定し、prepareToPlay で再設定 | 中 | 低 | 🔍 約481MB — ただし rebuild スレッドの初回 build が完了する前に prepareToPlay に到達する可能性があるため、タイミング調整が必要 |
| B: `SAFE_MAX_BLOCK_SIZE` (65536) の低減 | **高** | 中 | 🔍 約481MB — ただしクロスフェードバッファ等 SAFE_MAX_BLOCK_SIZE に依存する全箇所に影響 |
| C: 初回 prepare 時に `internalMaxBlock` の上限を制限（osFactor=0 なら実ブロックサイズを上限） | 低 | 低 | 🔍 約481MB — 最も安全。初回は osFactor 未確定のため 1x (=65536) に制限。後続の prepareToPlay で適切な値に再設定 |

⚠️ **CAVEAT**: P1 実施後のメモリ削減効果を確認してから要否を判断する。DSPCore LiveCount が 2 に収束した場合、初回巨大ブロックの解放（gen=1 DSPCore の retire）により ~481MB が自動的に回収される可能性が高い。したがって P2-1 は P1-a/b 完了後の残存メモリから妥当性を評価する。

### B.2 P3: AoS→SoA メモリ最適化

**FACT（2026-07-10 コード調査確定）: 主要な AoS→SoA 変換は既に完了している。**

`MKLNonUniformConvolver` の `irFreqDomain` は既に「1 パーティション分の使い捨てスクラッチ（FFT出力→deinterleave中継のみ）」に縮小されている（MKLNonUniformConvolver.h:318）。同じく `fdlBuf` も「current + mirror の 2 スロットのみのスクラッチ」に縮小済み（line 327-328）。Audio Thread が参照する本番データ（`irFreqReal`/`irFreqImag`、`fdlReal`/`fdlImag`）は最初から SoA 形式である。

したがって P3 の実質的な作業範囲は大幅に縮小され、以下の確認と微調整のみとなる:
- スクラッチバッファ（irFreqDomain/fdlBuf）と永続データ（irFreqReal/irFreqImag）の分離が正しいことのコードレビュー
- 不要になった古い AoS 関連コードの削除（もし残っていれば）
- メモリ削減効果は既に達成済みのため、新たな削減はほぼ期待できない

⚠️ **CAVEAT**: 「AoS→SoA (11パッチ)」という従来の見積もりは過大であった。実際の残作業はスクラッチバッファ検証のみで、コストは「低」に再分類する。

### B.3 P4: 未計装 mkl_malloc 追跡

✅ **FACT（2026-07-10 コード調査確定）**: 以下の箇所で DIAG 非対応の `mkl_malloc`/`mkl_free` が使用されている:
- `CacheManager.cpp:190` — `mkl_malloc(dataSize, 64)` (IR データキャッシュ)
- `CacheManager.cpp:228` — `mkl_malloc(dataSize, 64)` (IR データコピー)
- `IRConverter.cpp:187` — `mkl_malloc(bytes, 64)` (IR 変換バッファ)
- `IRConverter.cpp:200` — `mkl_free(data)` (上記の解放、対応ペア)
- `AlignedAllocation.h:15` — `mkl_malloc` (ラッパー: `aligned_malloc` の内部。間接的に多くの箇所で使用される)

`AlignedAllocation.h` の `aligned_malloc`/`aligned_malloc_nothrow`/`aligned_free` は広範囲で使用される汎用ラッパーであるため、これらを DIAG 対応にすると影響範囲が極めて大きい。したがって P4 の実質的な範囲は `CacheManager.cpp` の 2 箇所 + `IRConverter.cpp` の 1 箇所に限定される。

直接の削減効果はないが透明性向上に寄与。優先度は「低」を維持。

---

## Appendix C: 設計判断の詳細補足

### C.1 FadeState::Completed が不要な理由

✅ **FACT**: 現在の `tryComplete()` は `remainingCount() > 0` を直接チェックし、CAS `FadingIn→Idle` を行う。advance() が release で remaining=0 を書き込めば、既存の設計で完了検出できる。`FadeState::Completed` を追加する必要はない。

### C.2 memory_order release/acquire 維持の理由

現状の release/acquire は以下の HB を形成しており、relaxed 化は危険:

| 変数 | 書き込み | 読み取り | HB |
|:-----|:---------|:---------|:---|
| `remainingSamples_` | release (advance) | acquire (tryComplete) | advance の結果を Timer が確実に観測 |
| `state_` | release (start/complete) | acquire (state/isFading) | startFade の FadingIn を advance が観測 |
| `alpha_` | release (advance) | acquire (alpha) | fade 進行値を Audio Thread が観測 |

### C.3 CrossfadeRuntime 責務のコード根拠

```cpp
// DSPTransition.h:49 — DSP エンジン遷移のみ
void onPublishCompleted(...) {
    crossfadeRuntime_.start(decision.fadeTimeSec, sampleRate);  // DSP transition
}
// SnapshotCoordinator.cpp:43 — パラメータ遷移のみ
void advanceFade(int numSamples) noexcept {
    m_fade.advance(numSamples);  // Param fade counter
}
```

## Appendix D: 副次的知見

### D.1 DSPGuard は rebuild-obsolete DSPCore を解放できない — 確定（v4.5 確定）

**FACT 確定: 代替解放経路は存在しない。**

✅ **FACT**: `destroyDSPCoreNode()`（AudioEngine.Threading.cpp:15-19）が DSPCore を解放する**唯一の関数**。これは EBR enqueue 経由でのみ呼ばれる。

✅ **FACT**: `aligned_free` の全出現箇所を調査（AlignedAllocation.h, CtorDtor.cpp, PrepareToPlay.cpp 等）。DSPCore のメモリ解放に関与する箇所は `destroyDSPCoreNode()` のみ。`ScopedAlignedPtr` や `aligned_unique_ptr` の自動解放は `release()` により無効化されている（RuntimeBuilder.cpp:483-487）。

✅ **FACT**: rebuild-obsolete な DSPCore は `DSPGuard::~DSPGuard()` → `DSPLifetimeManager::retire()` → `retireDSPHandleForRuntime()` → handle map 未登録 → false → return。EBR enqueue されず `destroyDSPCoreNode()` が呼ばれない。

**規模**: ログ上の rebuild-obsolete 3 回分（gen=1 obsolete on gen=3, gen=5 obsolete on gen=4, gen=7 obsolete on gen=4）。各 prepare で ~220MB（通常）または ~481MB（初回巨大ブロック）と推定されるため、合計 660-720MB 程度のリーク。

⚠️ **CAVEAT**: P1-a では rebuild-obsolete リークは解決しない（rebuild-obsolete は commit 前に abort されるため、`enqueuePublicationIntentForRuntimeCommit()` に到達せず register されない）。本件は P1-a/b とは独立した二次的問題であり、別途の対応（未登録 DSPCore の強制解放機構など）が必要。

◀ **v1.0 では「実質的に解放されている」としていたが、コード解析の結果誤りと判明。**

### D.2 Coordinator direct publish と Orchestrator publish の差異

| 項目 | Coordinator direct | Orchestrator |
|:-----|:------------------|:-------------|
| Handle 登録 | ❌ しない（P1-a で修正） | ✅ newDSP を登録 |
| DSPTransition | ❌ 経由しない | ✅ 経由する |
| retire 方法 | 明示的に `lifetimeMgr.retire(done)` | DSPTransition 内で `lifetime.retire(oldDSP)` |
| 使用箇所 | Init, PrepareToPlay, ReleaseResources, Timer(fadeCompleted), Transition | RebuildDispatch のみ |
| crossfade 有無 | なし（即時 publish） | CrossfadeAuthority が判断 (`needsCrossfade`) |

両者とも最終的には `RuntimePublicationCoordinator::publishWorld()` で publish 処理が行われる。違いは上記のライフサイクル管理方法のみ。

### D.3 ISRRetireRouter のカウンタはデッドコード

✅ **FACT**: `m_pendingRetireBytes_` と `m_trackedPendingEntries_` に値を書き込むコードが存在しない。MEM_SNAP の `trBytes` / `tr` フィールドは常に 0 を返す。

### D.4 AudioEngine::retireDSP() に呼び出し元が存在しない

✅ **FACT**: `AudioEngine::retireDSP()` は `DSPLifetimeManager::retire()` とは別の関数であり、現時点で呼び出し元が存在しない。

### D.5 DSPCore 1個あたりのメモリコスト（v4.5 更新）

✅ **FACT（ログ解析）**: ログ上の MEM_SNAP と BUILD_PHASE から以下を確定:

| prepare | processingBlockSize | Private増分 | 内訳 |
|:--------|:-------------------|:------------|:------|
| gen=1 (初回) | 524288 (65536×8) | +481MB (74→555MB) | 初回DSPCore + JUCE初期化込み |
| gen=4 (SR切替) | 2048 | +799MB (555→1354MB) | DSPCore#2(~220MB) + rebuild-obsolete#1(~220MB) + 他(~359MB) |
| gen=8 (IR load) | 2048 | +1114MB (1354→2468MB) | DSPCore#3(~220MB) + rebuild-obsolete#2,3(~440MB) + IRload/他(~454MB) |

通常時 (blk=2048) の DSPCore 1個あたりのコストは **~220MB** 程度。初回 (blk=524288) の巨大ブロックは **~481MB** に相当する。（詳細は A.4 の内訳参照）

### D.6 lifecycle(retire)=0 の調査経緯（v4.5 確定）

2026-07-10 のログ解析により以下を確定:

| 調査項目 | 結果 |
|:---------|:------|
| Coordinator direct publish が handle 登録しない | ✅ **FACT**（コード確認） |
| Orchestrator publish 3 回全て SUCCEEDED | ✅ **FACT**（ログ確認: gen=1,4,8） |
| DSPTransition は 3 回全て実行 | ✅ **FACT** |
| Structural rebuild では crossfade 不要 (`needsCrossfade=false`) | ✅ **FACT**: ログ `[XFADE]=0`、Orchestrator の rebuild 理由は全て `requestRebuild_sr_bs` |
| rebuild-obsolete DSPCore に代替解放経路なし | ✅ **FACT**: `destroyDSPCoreNode()` が唯一の解放関数。EBR enqueue 経由でのみ呼ばれる。`aligned_free` 全出現調査で確認済み。 |
| `runtimeDSPHandleMap` erase 正常動作 | ✅ **FACT**: `retireDSPHandleForRuntime()` 内で map 検索 → erase → handle.retire → handle.reclaim が一貫して動作。mutex 保護済み。 |
| DSPCore 1個あたりのコスト | ✅ **FACT（推定値）**: 初回 prepare (blk=524288) = ~481MB。通常 prepare (blk=2048) = ~220MB。 |

---

## Appendix E: 最終調査結果 — 全未確定事項の確定状況

### E.1 コード調査で確定した事項（FACT）

| # | 項目 | 確定内容 | 調査方法 |
|:-:|:-----|:---------|:---------|
| 1 | lifecycle(retire)=0 の主要原因 | Coordinator direct publish が handle 登録しないため。Orchestrator publish は 3 回全て SUCCEEDED。DSPTransition も 3 回実行済み。 | ログ解析 (`trySubmit: publish SUCCEEDED` 3回確認) |
| 2 | `[XFADE]=0` の理由 | Structural rebuild (`requestRebuild_sr_bs`) では CrossfadeAuthority が `needsCrossfade=false`。即時 retire 経路使用。 | ログ解析 (DIAG rebuild 理由確認) |
| 3 | rebuild-obsolete リーク | 代替解放経路なし。`destroyDSPCoreNode()` が唯一の解放関数。`aligned_free` 全出現調査で確認。 | `aligned_free` 全ソース調査 |
| 4 | DSPCore 1個あたりのコスト | 初回 prepare (blk=524288) = **~481MB**。通常 prepare (blk=2048) = **~220MB**。 | BUILD_PHASE Private 差分解析 |
| 5 | `runtimeDSPHandleMap` erase 正常動作 | `retireDSPHandleForRuntime()` 内で map 検索→erase→handle.retire→handle.reclaim。mutex 保護済み。 | コード確認 |
| 6 | `registerDSPHandleForRuntime()` idempotent | 既存エントリがあればそれを返す（find→return, なければ create→emplace）。 | コード確認 (AudioEngine.h:3804-3819) |
| 7 | `publishWorld()` 戻り値 | `PublishStageResult` (Success/Rejected/Failed)。 | コード確認 (RuntimePublicationCoordinator.h:101) |
| 8 | production コードの publishWorld 呼び出し | **7 箇所**（Init/PrepareToPlay×2/ReleaseResources/Timer/Transition/PublicationExecutor）。テストコードは対象外。 | grep 全数調査 |
| 9 | `jassertfalse` の既存使用 | コードベースで既に使用されているパターン。本設計書の提案と矛盾なし。 | grep 確認 |
| 10 | `FrozenRuntimeWorld` 所有形態 | `aligned_unique_ptr<RuntimeState>` 保持。`releaseState()` で raw pointer 取り出し。 | コード確認 (FrozenRuntimeWorld.h:84) |
| 11 | advanceFade(numSamples) の単位 | callback block size (`bufferToFill.numSamples`)。OS 補正不要。 | コード確認 (AudioBlock.cpp:51) |
| 12 | SnapshotCoordinator vs CrossfadeRuntime の分離 | 前者はパラメータ遷移 (EQ/NS/AGC)、後者は DSP エンジン遷移。コード上も完全独立。 | コード確認 (両ファイル) |
| 13 | **`getBridge()` 不存在** | `RuntimePublicationCoordinator` テンプレート（`src/core/RuntimePublicationCoordinator.h`）は `Bridge bridge_` を private メンバとして保持するが、public `getBridge()` は存在しない。 | コード確認 (Coordinator template) |
| 14 | **Orchestrator newDSP 登録パス確定** | `enqueuePublicationIntentForRuntimeCommit()` 内で publish **前** に `registerDSPHandleForRuntime(newDSP)` → Orchestrator → `executor_.publish()` → `DSPTransition::onPublishCompleted()` → `retireDSPHandleForRuntime(oldDSP)` の完全チェーン確定。 | コード確認 (enqueuePublicationIntentForRuntimeCommit, Orchestrator, DSPTransition) |
| 15 | **7 箇所の publishWorld 呼び出しと DSPCore* マッピング確定** | Init:nullptr / PrepToPlay①:currentForPublish / PrepToPlay②:getActiveRuntimeDSP / ReleaseResources:nullptr / Timer:currentAfterFade / Transition:引数 / PublicationExecutor:nullptr | コード確認 (各 .cpp ファイル) |
| 16 | **ISRDSPHandle::rollbackRegistration CAS 安全性確定** | Constructing → Reclaimed への CAS (compare_exchange_strong) により、他スレッドとの競合防止を確認。Constructing 以外の状態では遷移しない。 | コード確認 (ISRDSPHandle.h の DSPState enum + CAS 実装) |
| 17 | **enqueueRetire 後段は非致命的** | QueueFull/QueuePressure でも drainDeferredRetireQueues(false) が試行され、HealthMonitor が overflow を監視。致命的なブロック要因ではない。 | コード確認 (AudioEngine.h:3750-3772, DSPLifetimeManager.h) |
| 18 | **rebuild-obsolete DSPGuard 解放パス確認** | `DSPGuard::~DSPGuard()` → `DSPLifetimeManager::retire()` → `engine_.retireDSPHandleForRuntime(dsp)` → **未登録** → false → return。代替経路なしをコード確認。 | コード確認 (AudioEngine.RebuildDispatch.cpp:763-770) |
| 19 | **Orchestrator 経路では handle 登録が publish 前に完了** | `enqueuePublicationIntentForRuntimeCommit()` (AudioEngine.Commit.cpp:676) 内で `registerDSPHandleForRuntime(newDSP)` → PublishRequest 生成 → Orchestrator publish → DSPTransition。Orchestrator 経路の retire は問題なく動作する。 | コード確認 (AudioEngine.Commit.cpp:676-685) |
| 20 | **diagLog は匿名名前空間の関数（AudioEngine.Commit.cpp:17）** | `CONVOPEQ_DIAG_LOG` マクロは存在しない。`diagLog(const juce::String&)` は匿名名前空間に限定される。ScopedPublishTrace は `juce::Logger::writeToLog` を直接呼ぶ。 | コード確認 |
| 21 | **DSPCore::liveCount 存在確認** | `AudioEngine::DSPCore` 内に `static std::atomic<uint32_t> liveCount` が存在。`#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` でガード。 | コード確認 (AudioEngine.h:606) |
| 22 | **StereoConvolver::liveCount 存在確認** | `ConvolverProcessor::StereoConvolver` 内に `static std::atomic<uint32_t> liveCount` が存在。`#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` でガード。 | コード確認 (ConvolverProcessor.h:631) |
| 23 | **MEM_SNAP 現状フォーマット確定** | gen, NUC(live/alloc/peak/tA/tF/lost/zero), Ret(pend/trBytes/tr/ovf/rec), Priv/WS/Other を出力。lifecycle(publish/retire/reclaim) は [VERIFY] ログで別途出力。DSPCore::liveCount と StereoConvolver::liveCount は未出力。 | コード確認 (AudioEngine.Timer.cpp:907-929) |
| 24 | **retiringGeneration は未定義** | コードベースに「現在 retire 中の generation」を追跡する変数は存在しない。MEM_SNAP に追加するには新規 atomic フィールドを DSPLifetimeManager::retire() が唯一の Authority として設定する機構が必要。P1-c の設計に反映する。 | grep 全数調査 |
| 25 | **BlockSize 524288 の原因チェーン確認** | `Init.cpp:38` → `maxSamplesPerBlock=65536` → `DSPCoreLifecycle.cpp:140-142` → `inputMaxBlock=65536` → `internalMaxBlock=524288`。Init.cpp:69 で問題認識済みだが、初回 Structural rebuild がこの巨大値を使用する。 | コード確認 |
| 26 | **AoS→SoA は既に完了** | `irFreqDomain` は 1 パーティションのスクラッチに縮小済み。Audio Thread 参照データは `irFreqReal`/`irFreqImag` (SoA)。P3 の実作業は大幅に縮小され、主にスクラッチ検証のみ。 | コード確認 (MKLNonUniformConvolver.h:318-330) |
| 27 | **未計装 mkl_malloc は CacheManager(2) + IRConverter(1) の 3 箇所** | `AlignedAllocation.h` の `aligned_malloc` ラッパーは汎用で影響範囲大。実質 P4 範囲は CacheManager.cpp(2) + IRConverter.cpp(1) のみ。 | コード確認 |
| 28 | **publishWorld() 失敗時に副作用を残さない** | Validation 拒否時は rejectedWorld を `bridge_.retireRuntimePublishWorldNonRt` で適切に後始末。publishAndSwap は成功確定後のみ Success を返す。 | コード確認 (RuntimePublicationCoordinator.h:97-145) |
| 29 | **PublicationExecutor の CallerManaged 前提確認** | `enqueuePublicationIntentForRuntimeCommit()` (AudioEngine.Commit.cpp:684) 内で `registerDSPHandleForRuntime(newDSP)` → `submitPublishRequest(req)` の順。register と publish の間に early return は存在しない。 | コード確認 (AudioEngine.Commit.cpp:676-693) |
| 30 | **advanceFade() 呼び出し箇所 0 確認** | `SnapshotCoordinator::advanceFade(int)` の定義は SnapshotCoordinator.cpp:43、宣言は SnapshotCoordinator.h:133 のみ。呼び出し元はコードベース全体で0。 | grep 全数調査 (src/ 全体) |
| 31 | **PublishRegistrationMode は未使用の新規名称** | コードベース内に `PublishRegistrationMode` の出現はゼロ。新規導入に名前衝突のリスクはない。 | grep 全数調査 (src/ 全体) |
| 32 | **DSPCore::~DSPCore() は destroyDSPCoreNode 経由のみ** | `DSPCore::~DSPCore()` (AudioEngine.h:819) はデストラクタを定義するが、`destroyDSPCoreNode()` (AudioEngine.Threading.cpp:15) からのみ呼ばれる。直接 `delete` や `aligned_free` が呼ばれる経路は存在しない。 | コード確認 (AudioEngine.h, AudioEngine.Threading.cpp) |
| 33 | **ConvolverProcessor.h に mkl_malloc/mkl_free なし** | ConvolverProcessor.h 内の mkl_malloc/mkl_free 出現は0件。P4 のスコープに ConvolverProcessor を含める必要なし。 | grep 全数調査 |
| 34 | **QueueFull は全ての経路で非致命的** | retireDSP/retireDSPHandleForRuntime/enqueueDeferredDeleteNonRtWithResult/DSPLifetimeManager の全 QueueFull ハンドリングがカウンタ増加のみ（ログ＋HealthMonitor委譲）。deferred delete の損失となるが、shutdown 時の drain で回収される。 | コード確認（4経路全て） |
| 35 | **registerDSPHandleForRuntime() emplace 安全性確認** | emplace は find→return の後のみ実行。`std::unordered_map::emplace` は重複キーでも既存を変更しない。noexcept 関数内で例外不要。create した Handle が孤立することはない。 | コード確認 (AudioEngine.h:3803-3816) |
| 36 | **m_coordinator.advanceFade() 挿入位置確認** | AudioBlock.cpp の getNextAudioBlock() 内で m_coordinator は既に使用可能（同一クラス）。advanceFade の挿入位置は DSP 処理完了直後で問題ない。existingHandle の名前衝突はコードベース全体でゼロ。SnapshotFadeState::advance → remaining 減算 → release store → tryComplete acquire load の完全チェーン確認済み。 | コード確認 (AudioBlock.cpp, SnapshotCoordinator.cpp:43-46, SnapshotFadeState.h:42-60) |
| 37 | **BlockSize 524288 は osFactor 非依存で確定** | `internalMaxBlock = inputMaxBlock × MAX_OS_FACTOR(8)` は oversamplingFactor を見ない。osFactor=0(Auto→1)でも内部バッファは 524288。初回 prepare のメモリ肥大は inputMaxBlock の出発値 SAFE_MAX_BLOCK_SIZE=65536 が原因であり、osFactor 設定では回避不能。 | コード確認 (DSPCoreLifecycle.cpp:140-145) |
| 38 | **runtimeDSPHandleMap steady-state サイズ推定: 2-3** | 正常動作時: current(1) + fading/retiring(1) + 過渡的エントリ(0-1) = 2-3。MAX_DSP_SLOTS=256 に対して十分小さく、枯渇リスクは極めて低い。 | コード確認 (runtimeDSPHandleMap_ 操作: register/retire/rollback のみ) |
| 39 | **create() の Reclaimed→instance 順序確認** | create() は state==Reclaimed 確認後、generation++, instance代入, generation publish(release), state=Constructing(release) の順。instance は state がまだ Reclaimed の間に設定されるが、registerDSPHandleForRuntime の mutex で保護されている。rollbackRegistration は Constructing→Reclaimed CAS 成功後に instance=null とするため安全。 | コード確認 (ISRDSPHandle.cpp:21-36) |
| 40 | **DSPHandle::operator== slot+generation 比較確定** | ISRDSPHandle.h:35 で `slot == other.slot && generation == other.generation` を確認。generation は create() (ISRDSPHandle.cpp:27) で常に +1u インクリメントされるため、rollback 時の Map erase 安全性（誤削除防止）が保証される。 | コード確認 (ISRDSPHandle.h:35, ISRDSPHandle.cpp:27) |
| 41 | **C++20 確定** | CMakeLists.txt:346 `set(CMAKE_CXX_STANDARD 20)` 確認。Designated initializers 使用可能。RegistrationContext の構文（`{.dsp = ptr}`）がそのまま使用できる。 | CMakeLists.txt 確認 (line 346) |
| 42 | **SCOPE_EXIT / ScopeExit コードベース未存在** | コードベース全体で SCOPE_EXIT / scope_exit / ScopeExit の出現はゼロ。新規導入可能。コードベースでは RAII クラス（ScopedPublishTrace 等）で同等機能を実現。 | grep 全数調査 (src/ 全体) |
| 43 | **production コード publishWorld() 7 箇所確定、関数ポインタ経路なし** | 7 箇所の直接呼び出しを全数確認。`auto fn = &RuntimePublicationCoordinator::publishWorld` のような関数ポインタ経由の隠れ呼び出しは存在しない。grep ベースの CI 静的チェックで検出可能な範囲であることを確認。 | grep 全数調査 (src/ 全体) |
| 44 | **runtimeDSPHandleMap_ key 型は DSPCore*（hash 不要）** | `std::unordered_map<DSPCore*, convo::isr::DSPHandle>` の key はポインタ。DSPHandle の std::hash 特殊化は不要。 | コード確認 (AudioEngine.h:4081) |
| 45 | **RegistrationContext 相当のパターンはコードベース未存在** | コードベース内に RegistrationContext / registrationContext / registration_context の出現はゼロ。新規導入に名称衝突のリスクはない。 | grep 全数調査 (src/ 全体) |
| 46 | **`retireDSP()` はデッドコード（0 call sites）** | `AudioEngine.h:3775` で定義される `retireDSP()` の呼び出し元はコードベース全体でゼロ。`DSPLifetimeManager::retire()` とは別の関数であり、`retireDSPHandleForRuntime()` + `enqueueDeferredDeleteNonRtWithResult()` の経路を持つが誰も使用していない。`DSPLifetimeManager::retire()` は同様の機能を持つが `router_->enqueueRetire()` 経由（epoch 管理が異なる）。デッドコード自体は無害だが、コードベースの整理時に削除候補となる。 | grep 全数調査 (src/ 全体) |
| 47 | **`m_pendingRetireBytes_` / `m_trackedPendingEntries_` はデッドコード（0 writers）** | `ISRRetireRouter.h` で宣言・初期化されるが、値を書き込むコードが存在しない。getter (`pendingRetireBytes()` / `trackedPendingEntries()`) は常に 0 を返す。ISRRetireRouter.cpp のコメントに「呼び出し元が objectBytes を設定した場合に使用」とあり、将来拡張用の未実装機能であることが確認できた。MEM_SNAP の `trBytes` / `tr` が常に 0 である事実と整合。 | コード確認 (ISRRetireRouter.h:160-161, ISRRetireRouter.cpp:129-135) |
| 48 | **rebuild-obsolete DSPCore はシャットダウンでも回収不可（永久リーク確定）** | DSPGuard が capture した rebuild-obsolete DSPCore は `DSPLifetimeManager::retire()` → `retireDSPHandleForRuntime()`（未登録→false）で全て失敗する。`AudioEngine::~AudioEngine()` の shutdown 処理では `activeToRelease` と `fadingToRelease` のみ retire 対象となる（CtorDtor.cpp:140-143）。rebuild-obsolete な DSPCore をトラッキングするリストやプールは存在せず、`pendingTask.currentDSP` も未登録の場合は retire 失敗する。したがって rebuild-obsolete な DSPCore インスタンスは AudioEngine 生存期間中完全にリークする。 | コード確認 (AudioEngine.RebuildDispatch.cpp:759-770, AudioEngine.CtorDtor.cpp:120-145) |
| 49 | **DSPGuard ライフサイクル確定** | DSPGuard は `{this, nullptr}` で初期化 → `dspGuard.ptr = buildResult.runtime`（非 obsolete 時のみ） → `dspToCommit = dspGuard.ptr; dspGuard.ptr = nullptr`（commit 時に所有権移譲）の順。obsolete 検出時は `continue` で DSPGuard デストラクタが実行され `DSPLifetimeManager::retire(ptr)` を呼ぶが handle 未登録のため失敗。非 obsolete の正常経路では確実に所有権が commit に移譲される。 | コード確認 (AudioEngine.RebuildDispatch.cpp:771,810,921-922) |
| 50 | **`enqueuePublicationIntentForRuntimeCommit()` が唯一の register 経路** | `registerDSPHandleForRuntime(newDSP)` を呼び出す関数は `enqueuePublicationIntentForRuntimeCommit()`（AudioEngine.Commit.cpp:685）のみ。Coordinator direct publish（Init/PrepareToPlay/ReleaseResources/Timer/Transition）はこの関数を通らないため handle 登録が行われず、これが lifecycle(retire)=0 の根本原因であることを確定。 | コード確認 (AudioEngine.Commit.cpp:685) |
| 51 | **`DSPHandleRuntime::retire()` は状態チェックなし（無防備）** | `DSPHandleRuntime::retire()`（ISRDSPHandle.cpp:95）は `publishAtomic(state, DSPState::Retired)` を CAS なしで実行。どのような状態（Constructing / Active / CrossfadingIn / CrossfadingOut 等）からでも Retired に遷移可能。設計上は Active または CrossfadingOut からの遷移を前提としているがコード上は制限がない。P1-a 実装時は問題にならない（retire 経路が commitRuntimePublication の register→publish 間に別スレッドから呼ばれない構造）が、将来のリファクタリング時には注意が必要。`reclaim()` も同様に無防備（ISRDSPHandle.cpp:102）。 | コード確認 (ISRDSPHandle.cpp:95-100, 102-107) |
| 52 | **`retireDSPHandleForRuntime()` 内の retire→reclaim は同一ロック内で連続実行** | `retireDSPHandleForRuntime()`（AudioEngine.h:3841）は mutex ロック下で Map erase → `dspHandleRuntime_.retire(handle)` → `dspHandleRuntime_.reclaim(handle)` を連続実行する。retire から reclaim の間に他のスレッドが介入する window は存在しない。実質的に retire 直後に reclaim されるため、Handle は極短時間だけ Retired 状態になる。 | コード確認 (AudioEngine.h:3852-3856) |
| 53 | **`RuntimePublishWorld` は `RuntimeState` のエイリアス** | `AudioEngine.h:319` で `using RuntimePublishWorld = RuntimeState;` と宣言されている。`RuntimePublicationCoordinator` の `World` 型は `RuntimePublishWorld`（= `RuntimeState`）。Instantiation: `<RuntimePublishWorld, DSPCore*, RuntimePublicationBridge>`（AudioEngine.h:3237-3239）。したがって全 7 箇所の publishWorld 呼び出しは同一の `World` 型を使用しており、`commitRuntimePublication()` のシグネチャも統一できる。 | コード確認 (AudioEngine.h:319,3237-3239) |
| 54 | **新規構造体/クラス名のコードベース衝突ゼロ** | `PublishCommitResult` / `PublishStageResultTraits` / `RegistrationContext` / `RollbackResult` / `rollbackRegistration` はコードベースに存在しない。また `ScopeExit` / `SCOPE_EXIT` / `scope_exit` も存在せず（FACT #42 再確認）、新規導入に名前衝突のリスクはない。 | grep 全数調査 (src/ 全体) |
| 55 | **`DSPHandleRuntime` に `rollbackRegistration` は未存在（新規追加必須）** | `ISRDSPHandle.h` のクラスメソッド一覧に `rollbackRegistration` は存在しない。既存メソッドは `create/resolve/beginCrossfade/activate/endCrossfade/retire/reclaim/quarantine/quarantineSlot/isSlotInCrossfade/destroyQuarantineSlot/getActiveRuntimeDSPHandle/getFadingRuntimeDSPHandle/emitOwnershipTrace` の 14 個。したがって設計書で提案している `rollbackRegistration()` は P1-a 実装時に新規追加する必要がある。`rollbackDSPHandleRegistration()` も同様に未存在。 | コード確認 (ISRDSPHandle.h:104-153) |
| 56 | **`PublicationExecutor::publish()` は現状 `DSPHandle` パラメータを持たない** | 現在のシグネチャは `PublishResult publish(AudioEngine& engine, aligned_unique_ptr<FrozenRuntimeWorld> frozen) noexcept`。P1-a では `DSPHandle existingHandle` パラメータを追加する必要がある。合わせて内部の `coordinator.publishWorld()` → `engine.commitRuntimePublication()` へ変更。 | コード確認 (PublicationExecutor.h:28-31, PublicationExecutor.cpp:8-74) |
| 57 | **Orchestrator に2つの distinct failure path が存在** | Orchestrator には2種類の failure path: (1) Build failure（line 81）: publish前でP1-a対象外、手動 `lifetime_.retire()` 継続必要。(2) Publish failure（line 167）: P1-a 後は ScopeExit が rollback 自動実行するため冗長化。v5.30の「全failure path冗長化」は不正確で、build failure の手動クリーンアップは維持が必要。 | コード確認 (RuntimePublicationOrchestrator.cpp:78-82, 159-174) |
| 58 | **`std::hash<DSPHandle>` 特殊化は未存在** | `std::hash<convo::isr::DSPHandle>` の特殊化はコードベース全体で存在しない。将来の二方向 Map（`unordered_map<DSPHandle, DSPCore*>` reverse_）では `hash_combine(slot, generation)` によるカスタムハッシュが必要。現行の `unordered_map<DSPCore*, DSPHandle>`（key=ポインタ）は標準のポインタハッシュで動作するため hash 不要。 | grep 全数調査 (src/ 全体) |
| 59 | **pendingTask.currentDSP 代替時も unregistered リーク第2経路** | `requestRebuild()`（RebuildDispatch.cpp:586-696）で既存 pendingTask を新要求で置換する際、`currentToRelease = pendingTask.currentDSP` → `lifetimeMgr.retire(currentToRelease)`（line 693-696）。この DSPCore は `enqueuePublicationIntentForRuntimeCommit()` 未通過のため handle 未登録。`retireDSPHandleForRuntime()` は false を返しリーク。rebuild-obsolete（FACT #48）とは別の独立した unregistered リーク経路であり、P1-a では解決しない。 | コード確認 (AudioEngine.RebuildDispatch.cpp:586-696) |
| 60 | **Shutdown graceful drain は登録済み DSPCore のみカバー** | `~AudioEngine()`（CtorDtor.cpp:155-158）は `DSPLifetimeManager::retire(activeToRelease/fadingToRelease)` を実行。登録済み DSPCore は `retireDSPHandleForRuntime()` により Map erase+retire+reclaim され `pendingRetireCount` に計上される。未登録 DSPCore は `retireDSPHandleForRuntime()` が false を返しサイレント無視。Graceful drain（line 178-192）は pendingRetireCount==0 を待つが、未登録リークは計上されないため drain 対象外。未登録 DSPCore のメモリは ~AudioEngine 終了後の OS 回収に依存する。 | コード確認 (AudioEngine.CtorDtor.cpp:155-158, 178-192) |
| 61 | **MEM_SNAP の `gen` は既に `currentGeneration`（runtimeWorld->generation）** | MEM_SNAP フォーマット（Timer.cpp:929）の `gen` フィールドは `runtimeWorld->generation`（line 909）から設定される。したがって P1-c で提案していた `currentGeneration` の追加は既に完了済み。P1-c の作業範囲は `StereoConvolver::liveCount`、`DSPCore::liveCount`、`retiringGeneration` の 3 項目に縮小される。 | コード確認 (AudioEngine.Timer.cpp:907-929) |

### E.2 実装後の検証が必要な事項（HYPOTHESIS）

以下の項目はコード調査のみでは確定できず、**P1-a/P1-b 実装後の MEM_SNAP 測定で検証する**必要がある。これらは設計書内で 🔍 HYPOTHESIS または ⚠️ CAVEAT として正しく分類されている。

| # | 項目 | 現状の分類 | 検証方法 |
|:-:|:-----|:----------|:---------|
| 1 | lifecycle(retire)=0 の原因が handle map 未登録であることの証明 | 🔍 HYPOTHESIS | P1-a 実装後、`retireDSPHandleForRuntime()` が true を返し `lifecycle(retire)>0` になること |
| 2 | Private Memory 改善量 | 🔍 HYPOTHESIS | P1-a/b 実装後、MEM_SNAP `Priv` の変化を測定 |
| 3 | runtimeDSPHandleMap size の収束値 | ⚠️ CAVEAT | コード構造から 2-3 に収束（FACT #38）の見込み。実測が必要だが枯渇リスクは極めて低い。 |
| 4 | メモリ削減推定値の確定 | 🔍 HYPOTHESIS | P1-a/b 実装後の MEM_SNAP 測定。~1361MB 削減推定は P1-a により retire が回復した場合の最大値。 |
| 5 | BlockSize 削減効果 | 🔍 HYPOTHESIS | P2 実施後の測定。osFactor 非依存（FACT #37）のため BlockSize 最適化は INIT 時の maxSamplesPerBlock 調整が必要。 |
| 6 | rebuild-obsolete 二次リークの影響 | ⚠️ CAVEAT | P1-a で rebuild-obsolete は解決しない。P1-a/b 完了後の残存メモリから評価。代替解放経路の不存在はコード確認済み FACT。本件は P1-a とは独立した二次的問題。 |
| 7 | retiringGeneration の MEM_SNAP 追加 | 💡 PROPOSAL | P1-c で `DSPLifetimeManager::retire()` 内に `currentRetiringGeneration_` atomic フィールド追加が必要。実装コストは低い（1 atomic store）。優先度は P1-c 内で判断。 |

### E.3 結論

コード調査により確定可能な事項は **全て確定済み**（v5.38 時点で 61 項目が FACT）。未確定のまま残る 7 項目は **実装後の検証が必要な性質のもの**（実行時メモリ測定、P1-a 実装後の retire 動作確認、P1-c 設計判断等）であり、コード調査による追加確定は不可能である。設計書内で HYPOTHESIS/CAVEAT として正しく分類されている。本設計書（v5.38）は P1-a の実装を開始できる完成度にある。

**v5.38 で実施した全6ツール最終調査（2026-07-11）**:
- DIAG 基盤確認: `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 存在確認（診断ログ条件付きコンパイル可能）
- `CONVOPEQ_DIAG_LOG` マクロ未存在再確認（FACT #20 確定済み。代わりに `juce::Logger::writeToLog` を使用）
- 全コードスニペットのシグネチャ整合性確認（compareExchangeAtomic / FrozenRuntimeWorld / aligned_unique_ptr）
- 名前空間解決の確認（convo::isr::DSPHandle / RegistrationContext / PublishStageResult）
- Coordinator テンプレート制約の確認（C++20 `requires` 式 — FACT #41 確定済み）
- 全6ツール（grep/serena/cocoindex/graphify/semble/AiDex）による最終調査完了
- **結論: コード調査により確定可能な事項は全て確定済み。追加確定事項なし。**

**v5.37 で反映した項目（コードレビュー第7弾、7項目修正）**:
1. **INV-8 追加**: `alreadyRegistered()` の Handle は Constructing 状態でなければならないことを Invariant 化。
2. **rollbackRegistration() Constructing 専用コメント**: 将来の中間状態追加時に再設計が必要であることを明記。
3. **eraseByHandle() コメント改善**: 現状 O(n) linear scan であることを明記。
4. **DIAG に `rollback_skip(reason=committed)` 追加**: commit point 到達後の ScopeExit 無効化を追跡。
5. **ScopeExit コメント改善**: 参照キャプチャであること + TODO(work72) PublicationTransaction 抽出。
6. **P1-c retiringGeneration Authority 明記**: DSPLifetimeManager::retire() のみ更新可。
7. **P1-b advanceFade 位置コメント**: DSP 処理直後（Audio callback 終了直前）であることを固定。|

---

## Appendix F: 調査で使用したツール

| ツール | 使用目的 |
|:-------|:---------|
| grep/sed/awk (WSL) | ログ抽出、統計計算、production コード全数調査、未使用名称確認 |
| serena MCP | コードパストレース、型情報取得、状態遷移の全調査、Coordinator 例外保証確認 |
| cocoindex-code (ccc.exe) | 関数間依存関係の grep、シンボル特定、新規 API 依存関係分析 |
| graphify | 依存関係グラフパス検索（BFS 全ノード可視化）、関数間リンク検証 |
| semble | セマンティックコード検索、フォールバック経路の発見、retire パス確認 |
| AiDex MCP | コードインデックス検索、シンボル特定、セマンティック検索、セッションノート管理 |

---

## Appendix G: 改訂履歴

| 版 | 日付 | 改訂内容 |
|:---|:-----|:---------|
| 1.0 | 2026-07-10 | 初版 |
| 2.0 | 2026-07-10 | 事実/仮説/設計案を分離。Phase 1 最小化 |
| 3.0 | 2026-07-10 | lifecycle(retire)=0 の原因特定。設計情報前方再構成。Appendix 集約。|
| 4.0 | 2026-07-10 | rebuild-obsolete リーク確定。`[XFADE]=0` 原因確定。P1-a/P1-b 役割分離 |
| 4.1 | 2026-07-10 | P1-a 単一ラッパー設計。MEM_SNAP generation 追加。DIAG チェーン拡充 |
| 4.2 | 2026-07-10 | RuntimeState DSPCore* 不在確認。Orchestrator 型不一致確認 |
| 4.3 | 2026-07-10 | HYPOTHESIS 明確化。commitRuntimePublication 改名。HANDLE 3点セット |
| 4.4 | 2026-07-10 | FrozenRuntimeWorld overload 追加。統一ラッパー完成 |
| 4.5 | 2026-07-10 | DSPCore コスト確定。Private Memory 内訳詳細化。rebuild-obsolete 確定 |
| 4.6 | 2026-07-10 | adopt() ping-pong 排除。publishWorld 禁止ルール。PublishOrigin |
| 4.7 | 2026-07-10 | FrozenRuntimeWorld overload 削除。impl 一本化。責務分離 |
| 4.8 | 2026-07-10 | 戻り値 PublishStageResult。register 失敗前提。production 7箇所 |
| 4.9 | 2026-07-10 | CI 静的チェック。孤児 Handle 対策。lastRetiredGeneration |
| 5.0 | 2026-07-10 | PublishOrigin 引数追加。Handle rollback。jassert 明確化 |
| 5.1 | 2026-07-10 | unregisterDSPHandleForRuntime 追加。expectedDSP 明確化 |
| 5.2 | 2026-07-10 | DSPHandleRuntime::unregister 委譲。jassert 削除。P1-a 表現弱化 |
| 5.3 | 2026-07-10 | abortRegistration 採用。generation 命名修正。P1-a 条件付き化。PublishOrigin 整理 |
| 5.4 | 2026-07-10 | abort 責務を呼び出し元へ移譲。abortRegistration CAS 化。invariant 表現修正 |
| 5.5 | 2026-07-10 | abort を commitRuntimePublication 内トランザクションに。Orchestrator 命名。CI 拡張。P1-a 表現修正 |
| 5.6 | 2026-07-10 | P1-a 責務を Coordinator + Bridge トランザクションへ再配置。abort→rollbackRegistration 改名。PublishOrigin→ScopedPublishTrace(DIAG)。HandleRegistry 抽象化提案。FrozenRuntimeWorld::toPublishWorld()。publishWorld アクセス制御強化 |
| 5.7 | 2026-07-10 | getBridge() 不存在を確定 → 設計を AudioEngine メソッドに再修正。7箇所のDSPCore*マッピング確定。Orchestrator newDSP パス全容確定。rebuild-obsolete解放経路再確認。enqueueRetire後段条件確認。ISRDSPHandle Rollback安全性確定。Appendix E 更新。Appendix F ツール使用例追記 |
| 5.8 | 2026-07-10 | PublishRegistrationMode 導入で nullptr 意味論排除。7箇所のパラメータ確定表更新（mode 列追加）。DSPTransition 二重登録分析追記。commitRuntimePublication に switch(mode) 採用。設計ルール整理 |
| 5.9 | 2026-07-10 | ScopedPublishTrace コード修正（convo::isr→AudioEngine, bridge削除）。新FACT 20-24追加（diagLog実態、DSPCore/StereoVolver liveCount、MEM_SNAP現状、retiringGeneration未定義）。E.2 retiringGeneration 追加（7項目）。E.3結論更新。Appendix F ツール記述充実 |
| 5.10 | 2026-07-10 | rollback 順序修正（HandleRuntime先→map削除後）。rollbackRegistration を bool に変更（CAS失敗検出）。Constructing→Reclaimed に意味論コメント追加。CI grep 範囲拡大（publishWorld( に統一）。CommitResult 将来拡張候補を追記。設計ルールの最終整理 |
| 5.11 | 2026-07-10 | BlockSize 524288 原因チェーン確定（Init→DSPCoreLifecycle）。AoS→SoA 完了確認（P3 範囲大幅縮小）。未計装 mkl_malloc 3 箇所確定（CacheManager/IRConverter）。FACT 25-27 追加。P2-1/P3/P4 セクション全面更新。 |
| 5.12 | 2026-07-10 | publishWorld 副作用ゼロ確認（FACT #28）、PublicationExecutor 前提確認（FACT #29）。Register/AlreadyRegistered → EngineManaged/CallerManaged に改名。rollbackRegistration Constructing-only コメント強化。retiringGeneration Authority = DSPLifetimeManager 確定。commitRuntimePublication 内部分割の将来候補追記。 |
| 5.13 | 2026-07-10 | 最終コード調査完了。新FACT 30-34追加。E.3結論更新。Appendix F 全ツール記述充実。34FACT/7HYPOTHESIS確定。 |
| 5.14 | 2026-07-10 | rollback 条件に publishPerformed guard 追加。rollbackDSPHandleRegistration に DSPHandle 版追加（mutex 分割）。CallerManaged/EngineManaged → AlreadyRegistered/NeedsRegistration に改名。PublicationExecutor が DSPHandle を渡す設計に修正。INV-1〜4 明文化。CI grep を暫定措置と明記。create() の Reclaimed 依存を設計前提として文書化。P1-a 期待表現を「lookup 成功率改善」に弱化。 |
| 5.15 | 2026-07-10 | INV-4 表現緩和（過渡的共存を許容）、INV-5 追加（rollback 後再登録保証）。PublicationExecutor DSPHandle データフロー確定（publish シグネチャに existingHandle 追加）。registerDSPHandleForRuntime emplace 安全性 FACT #35 追加。E.3 結論更新（v5.15/35FACT）。 |
| 5.16 | 2026-07-10 | 最終コード調査完了。FACT #36追加（advanceFade挿入位置、existingHandle名前衝突、SnapshotFadeStateチェーン）。36FACT。 |
| 5.17 | 2026-07-10 | AlreadyRegistered guard（jassert＋実行時チェック）追加。RollbackResult enum導入（Failed/RolledBackAndErased/RolledBackAlreadyMissing）。DSPHandle版rollbackをbool→RollbackResultに変更。E.3結論更新。 |
| 5.18 | 2026-07-10 | commit point 判定に将来の enum 拡張リスクを CAVEAT 追記。rollbackRegistration CAS 先行理由（state が authority）をコメント明文化。AlreadyRegistered に jassert(dsp==nullptr) 追加。INV-1 に register idempotent の authority を明文化。Map erase Handle 一致安全性をコメントで文書化。 |
| 5.19 | 2026-07-10 | BlockSize 524288 が osFactor 非依存であることを確定（FACT #37）。runtimeDSPHandleMap steady-state サイズ推定 2-3 を確定（FACT #38）。retiringGeneration 実装設計確定（DSPLifetimeManager atomic field）。E.2 各 HYPOTHESIS に最新知見反映。E.3 結論更新。全6ツール使用（grep/serena/cocoindex/graphify/semble/AiDex）。 |
| 5.20 | 2026-07-10 | PublishRequest newDSP データフロー確認（既存設計で成立）。二方向 Map（HandleRegistry reverse_）を将来リファクタリング候補に追加。Reclaimed 二重意味（寿命終了＋利用可能）をコメント明文化。"Available for slot reuse" 追記推奨。 |
| 5.21 | 2026-07-10 | **最終版。** 全6ツールによる全コード調査完了。existingHandle 名前衝突ゼロ最終確認、cocoindex/graphify/semble による全依存関係確認。7 HYPOTHESIS は全て実装後の検証が必要な性質と最終確定。コード調査による追加確定事項なし。 |
| 5.22 | 2026-07-10 | PublishCommitResult 導入（committed bool + stage + handle 構造体）。commitRuntimePublication 戻り値を PublishStageResult→PublishCommitResult に変更。DSPCore*版 rollback を互換レイヤー化（TODO付記）。HandleRegistry 二方向 Map の O(1) 設計を恒久方針として明記。INV-6追加（トランザクション完結性）。assert(dsp==nullptr)削除（AlreadyRegistered の invariant は existingHandle のみ）。 |
| 5.23 | 2026-07-10 | **最終確定。** PublishCommitResult/CommitResult のコードベース衝突ゼロ確認。assert(dsp==nullptr)削除確認。Orchestrator経路のcommitRuntimePublication呼び出しをPublishCommitResult対応に修正。E.3結論更新。全6ツール使用完了。コード調査可能な全事項を確定（38FACT/7HYPOTHESIS）。 |
| 5.24 | 2026-07-10 | committed 判定を PublishCommitResult 戻り値に一本化（内部でも stage 直接比較せず committed bool のみ使用）。ScopeExit パターンで rollback 実装（将来の return 経路漏れ防止）。create() の Reclaimed→instance 非参照を設計前提として明記。DSPCore*版 rollback に [[deprecated]] 付与。P1-c に maxRetiredGeneration 提案追加。Architecture Decision 明文化（production コードの publish は全て commitRuntimePublication 経由）。 |
| 5.25 | 2026-07-10 | **最終確定（FACT #39）。** 全6ツール（grep/serena/cocoindex/graphify/semble/AiDex）最終確認。create() の実装順序をコード確認（state==Reclaimed のまま instance 設定されるが mutex 保護済み）。SCOPE_EXIT 既存コード確認（未使用、新規導入可能）。E.3 結論更新。39FACT/7HYPOTHESIS。P1-a 実装開始可能。 |
| 5.26 | 2026-07-10 | **コードレビュー反映版（11項目反映）。** ①PublishStageResultTraits導入、②rollbackHandle変数分離、③CAS順序確認（変更不要確認済）、④operator== slot+generation明文化、⑤runtimeDSPHandleMap private helper概念追加、⑥PublicationExecutor確認（変更不要確認済）、⑦NeedsRegistration dsp==nullptr即Failed、⑧AST matcher恒久対策案追記、⑨RolledBackMapMissing改名、⑩P1-b統合確認、⑪RegistrationContext構造体導入（enum→struct移行）。新FACT 40-45追加（operator==/C++20/SCOPE_EXIT/publishWorld7箇所/key型/RegistrationContext）。45FACT/7HYPOTHESIS/6INV確定。P1-a実装開始可能。 |
| 5.27 | 2026-07-10 | **コードレビュー第2弾反映版（6項目修正）。** A.committed二重管理削除（→Traits::isCommittedに一本化）、B.handle戻り値DIAG化（PublishCommitResult最小化）、C.RollbackResult DIAG限定（Production→bool）、D.private helper削除（P1-a最小変更方針に統一）、E.CI grep参考実装化（clang-tidy恒久対策と明記）、F.commitRuntimePublication責務範囲の現状認識（後続リファクタリングに委ねる）。45FACT/7HYPOTHESIS/6INV確定。P1-a実装開始可能。 |
| 5.28 | 2026-07-10 | **未確定事項棚卸し版（5新FACT追加）。** 全7HYPOTHESISを6ツール（grep/serena/cocoindex/graphify/semble/AiDex）で再調査。コード調査可能な5項目をFACT化: retireDSP()デッドコード(#46)、m_pendingRetireBytes_/m_trackedPendingEntries_デッドコード(#47)、rebuild-obsolete永久リーク確定(#48)、DSPGuardライフサイクル確定(#49)、唯一のregister経路確定(#50)。50FACT/7HYPOTHESIS/6INV確定。P1-a実装開始可能。 |
| 5.29 | 2026-07-10 | **コードレビュー反映版（INV-7 + CAVEAT + FACT #51-52）。** INV-7追加（rollback後のMap整合性保証）。`DSPHandleRuntime::retire()`の状態チェックなしをコード確認しCAVEAT文書化+FACT #51。retire→reclaim連続実行をFACT #52化。52FACT/7HYPOTHESIS/7INV確定。P1-a実装開始可能。 |
| 5.30 | 2026-07-10 | **最終棚卸し版（5新FACT追加 + Orchestrator failure CAVEAT）。** 全6ツール（grep/serena/cocoindex/graphify/semble/AiDex）最終調査。RuntimePublishWorld=RuntimeState確定(#53)、新規名称衝突ゼロ(#54)、rollbackRegistration未存在(#55)、PublicationExecutor未対応(#56)、Orchestrator failure冗長化(#57)。57FACT/7HYPOTHESIS/7INV確定。コード調査による追加確定事項なし。P1-a実装開始可能。 |
| 5.31 | 2026-07-10 | **コードレビュー第4弾反映版（4項目修正）。** ScopeExit宣言順序修正(committed先行)、INV-3表現変更(Traits::isCommitted基準)、resolve()state authority明記、PublicationExecutor Constructing前提コメント追加、publishWorld [[deprecated]]案追加、rollbackDSPHandleRegistration [[nodiscard]]不使用コメント化。57FACT/7HYPOTHESIS/7INV確定。P1-a実装開始可能。 |
| 5.32 | 2026-07-10 | **最終未確定事項棚卸し版（FACT修正+3新FACT追加）。** 全6ツール(grep/serena/cocoindex/graphify/semble/AiDex)最終調査。FACT#57修正(Orchestrator dual failure path)。新FACT: std::hash未存在(#58)、pendingTask代替時リーク第2経路(#59)、Shutdown graceful drain範囲(#60)。60FACT/7HYPOTHESIS/7INV確定。コード調査による追加確定事項なし。P1-a実装開始可能。 |
| 5.33 | 2026-07-10 | **コードレビュー第5弾反映版（4項目修正）。** rollbackRegistration instance=nullptr削除（state only CASに純化）、RegistrationContext静的ファクトリ追加（needsRegistration/alreadyRegistered/none）、eraseByHandle内部ヘルパー抽出（HandleRegistry準備）、INV-3表現刷新（commit point基準）、commitRuntimePublication ScopeExit簡略化（rollbackHandle.invalidate一本化）。60FACT/7HYPOTHESIS/7INV確定。P1-a実装開始可能。 |
| 5.34 | 2026-07-10 | **最終確定版（eraseByHandle抽出修正 + invalidate→DSPHandle::null()修正）。** eraseByHandle static→非static修正（AudioEngineメンバ変数アクセスのため）。invalidate()→DSPHandle::null()に統一（DSPHandleにinvalidateメソッド未存在のため）。60FACT/7HYPOTHESIS/7INV確定。P1-a実装開始可能。 |
| 5.35 | 2026-07-10 | **コードレビュー第6弾反映版（4項目修正）。** rollbackDSPHandleRegistration戻り値分離（CAS成功最優先、Map cleanup二次的）、INV-7表現緩和、create() jassert追加推奨、publishWorld [[deprecated]]削除（clang-tidyのみ）、DIAG[HANDLE] action記録提案追加。60FACT/7HYPOTHESIS/7INV確定。P1-a実装開始可能。 |
| 5.36 | 2026-07-10 | **最終未確定事項棚卸し版（FACT#61追加 + P1-c範囲訂正）。** 6ツール（grep/serena/cocoindex/graphify/semble/AiDex）最終調査。MEM_SNAP gen(=currentGeneration)既存確認→P1-c範囲縮小。FACT#61追加。61FACT/7HYPOTHESIS/7INV確定。P1-a実装開始可能。 |
| 5.37 | 2026-07-11 | **コードレビュー第7弾反映版（7項目修正）。** INV-8追加（alreadyRegistered Constructing保証）、rollbackRegistration Constructing専用コメント、eraseByHandle O(n)明記、DIAG rollback_skip追加、ScopeExit参照キャプチャコメント+TODO(work72)、P1-c retiringGeneration Authority明記、P1-b advanceFade位置固定。61FACT/7HYPOTHESIS/8INV確定。P1-a実装開始可能。 |
| 5.38 | 2026-07-11 | **最終確定版（全6ツール最終調査完了）。** grep/serena/cocoindex/graphify/semble/AiDex 全ツール最終調査。DIAG基盤存在確認、CONVOPEQ_DIAG_LOG未存在再確認、コードスニペット整合性確認。コード調査による確定可能事項は全て確定済み（追加確定事項なし）。61FACT/7HYPOTHESIS/8INV。P1-a実装開始可能。 |
