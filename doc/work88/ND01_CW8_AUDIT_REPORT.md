# ND-01 — CW-8 実装前 read-only 契約監査（Work Report）

```text
Production source: 0 / Test source: 0 / CMake: 0 / Build: 0 / CTest: 0 / stress: 0（完全 read-only）
コード基準: ConvoPeq.md Generated 2026-09-01 16:28:11（--check FRESH・NEWER_SRC_COUNT=0 実測）
詳細: evidence/ND01_CW8_PRE_IMPL_CONTRACT_AUDIT.md
```

## 総合判定

> ## **ND-01 = GO（Case A — 最小修正）・X4 topology 再開は不要**

**最重要発見: CW-8 の要求（world N + identity N+1 の混在不可・単一 linearized 読取）は、現行 topology ですでに構造的に成立している。** identity は独立 atomic として存在せず、**`RuntimeState::publication`（PublicationSemantic: sequenceId / epoch / mappedRuntimeGeneration / previousSequenceId）として world 内部に publish 前 bake・sealed 後不変**で格納されている（AudioEngine.h:204 / ISRRuntimeSemanticSchema.h:252-258 / MutablePrePublish + sealRecursively）。したがって `RuntimeStore::current`（`atomic<RuntimeState*>`・**読取 path 上の atomic は 1 個のみ**）の 1 回 acquire load が `{world, identity}` ペアの同時確定になる。

## 報告 10 項目（要約）

1. **world pointer source**: `RuntimeStore::current` 単一（INV-X4-7/A/B）— `observePublishedWorld()` / `consumeWorldHandle()` 経由、caller 12 件実測。
2. **identity source**: 独立 source 無し — world 内部の `publication` member（bake 済み）。`RuntimePublicationIdentity` は reserve 時の一時ハンドルのみ。
3. **atomic object 数**: 読取 path で **1**（`RuntimeStore::current`）。別 atomic による torn-read 窓は存在しない。
4. **publication transaction の実体**: `sealRecursively()` → `publish()`（commit = bake + monotonicity check → release fence → `publishAndSwap` acq_rel exchange = 単一 LP）→ oldWorld を 1 回だけ返却して retire。Faulted なら swap しない（原子性）。
5. **Reader/Epoch protection 境界**: EBR（`enqueueDeferredDeleteNonRt(DeletionEntryType::World)` → `minReaderEpoch()` 進行後 drain・INV-EPOCH-1/2）。`ReadToken` は opaque 空トークン（epoch 保護は持たない・保護は retire 側 EBR）。observation は非所有 borrow のため境界を変更しない。
6. **CW-8 を満たす最小データ構造**: 方式 **A**（immutable record + 単一 acquire load）を採用 — ただし「新規に作る」のではなく **現行構造の型による明示化**。方式 B（別 atomic）は torn-read 窓の新規導入となるため不採用、方式 C（seqlock）は topology 変更となるため不採用。
7. **X4 ownership topology への踏み込み**: **なし（Case A）** — RuntimeStore / Owner / WriteAccess / publish authority / retire authority に触れない。Case B（topology 変更 = X4 再開）の根拠は不成立。
8. **INV-ISR-06 / CW-5 / CW-8 の関係**: CW-5 は現行で構造的に恒真（別 store identity が存在しない）。CW-8 は新 publication semantic ではなく **既存 invariant の read-side 型による強化・明示化**。新 ownership authority の追加は発生しない → NO-GO 条件不該当。
9. **実装対象ファイル**: `src/audioengine/RuntimeWorldAuthority.h` のみ（`PublishedWorldObservation` 型 + `observePublishedObservation(ReadToken)` factory 1 件・+30 行前後）。identity は `&world->publication` からのみ得られ、外部捏造不能な型にする（factory 以外の構築を delete）。
10. **targeted test**: T-CW8-1〜7 を設計済み。最重要の **T-CW8-7 は値比較ではなく型レベル**（`PublishedWorldObservation` の identity が world と独立に構築できないことを static_assert / delete-ctor で表明）— 既存のソース文字列検査型テスト（ObservePathSingleSourceTests 等）と差別化し、将来の topology 変更時に混在可能構造へ変わったら落とすガードとする。

## 遷移

```text
ND-01 GO（Case A）   ← 本報告
   ↓
ND-02 T-CW8-1〜7 テスト契約確定（実装はまだ）
   ↓
ND-03 最小実装（RuntimeWorldAuthority.h 1 ファイル）
   ↓
ND-04 targeted CW-8 + 既存 3 suite + 全体 CTest（stress は判断保留）
   ↓
ND-06 BuildError retry contract audit → CR-α 実装
```

CR-α（BuildError retry backoff）は本監査の完了まで着手しない（指示どおり）。
