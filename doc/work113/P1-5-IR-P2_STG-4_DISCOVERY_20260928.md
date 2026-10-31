# STG-4 Concrete Defect Discovery

```text
Authority HEAD : de07ba54301697fc877d005a11b07eac8313c8e8
Authority file : ConvoPeq.md 5,568,326 B
SHA-256        : EC8B4F0637F8FDCBCCAEDF46D3BD853F0169AE11ECEBFA0FF18BF7B460583856
Generated      : 2026-09-28 07:06:17（ConvoPeq.md を直接参照・hash 照合済み）
Phase          : Discovery only（production 0 / test 0 / commit 禁止）
worktree       : de07ba54 clean（pre-existing M AGENTS.md / headroom-proxy-start.ps1 のみ・不干渉）
```

## A. Loader / IR lifecycle → STG-4-1（Concrete Defect）

### Source locations（current authority）

```text
新規要求（sync）：ConvolverProcessor.LoadPipeline.cpp L230-319 loadIR()
  L244 bumpGeneration → L256-305 sync convert（cancellation lambda 付き）→ L261/274/303 applyComputedIR
新規要求（async）：同 L17-95 loadImpulseResponse()
  L51-55 旧 loader に exit signal＋trash／新 LoaderThread 作成
旧 loader 側：LoaderThread.cpp L634-647（myGen は load 途中サンプリング）／L239-309 queueFinalize
適用（無 guard）：LoadPipeline.cpp L635-704 finalizeNUCEngineOnMessageThread
  → L706-778 applyNewState → L796-856 executePendingCommit（staleness check なし・Phase 3 無条件 swap）
preset 入口：AudioEngine.Parameters.cpp L212-215 requestConvolverPreset → loadIR（guard なし）
```

### Trigger sequence

```text
1. loadImpulseResponse(A) → async LoaderThread(A) が flight 中（isLoading=true）。
   （量産経路：Lifecycle L219 同一 file rebuild／Rebuild.cpp L17 SR 変更 rebuild。
    いずれも public API 到達可能。）
2. その完了前に requestConvolverPreset(B) → loadIR(B)：
   bump(G2) → sync convert(B) → applyComputedIR(B, G2) → B 適用確定。
   ※ loadIR は in-flight async loader への cancel／trash を一切行わない。
3. LoaderThread(A) が継続（誰も exit signal しない）→ myGen を bump 後に
   サンプリングした場合は G2 一致、resample 不要時は gen 照合自体を素通し
   → finalize → executePendingCommit（無条件 swap、isRebuild のため currentIrFile は B のまま）。
```

### Actual behavior

```text
engine（DSP 音）= A に巻き戻る一方、currentIrFile／UI／保存 state = B。
file/engine の silent divergence。B 適用が無言で失われる。
```

### Expected invariant

```text
到着順に要求を適用し、stale async completion が newer state を上書きしないこと。
同一原則の guard は applyComputedIR（L391 generation 照合）・cache 経路（L219）・
convert cancellation（L286-289）に既に存在するが、finalize→commit 経路にだけ存在しない。
```

### Impact

```text
observable：選択した IR と異なる IR が鳴る＋state 不一致。次回 load まで残留。
persistence：file=B／engine=A の不一致 state が保存され得る。
```

### Deterministic reproduction path（未実行・test GO 待ち）

```text
1. loadImpulseResponse(A_big：低速・Minimum phase 等）→ isLoading==true を poll 確認。
2. loadIR(B_small）→ 同期確定（currentIrFile==B）。
3. message pump＋settling 後に active engine／state が B であることを assert。
   修正前：A に巻き戻り → FAIL。B→B なら setup 不成立として再試行判定。
   既存 seam：getConvolverProcessor().loadImpulseResponse（BassBuzz L2276 で使用実績）
   ＋ requestConvolverPreset。既存 STG-1/STG-2 checks は不変。
```

### B-6 との関係

```text
LoaderThread.cpp L41-42「B-6: Generation check is pending」は load loop 内の
  早期打切りの話であり、本件（finalize/commit の currency 欠落）とは別物。
  B-6 を根拠に本件を closed 扱いしないこと。
```

## B. Bulk restore / deferred rebuild → concrete なし

```text
requestLoadState は mode 直接 publish（Step1）→ setter clamp（Step2）の正順。
STG-2 revalidate は Step1 を経由しないため load 意味不変。
bulk 内 setter は各々 submit（既存設計）。最終 publish は D-guard 群（下記）が
  monotonicity を強制。stale build／generation 逆転／完了前 stale publish の
  concrete 経路なし。P1Polyphase／BassBuzz の bulk 順序も staging 終端で正当。
```

## C. Failure / retry / recovery → concrete なし（D105 再開なし）

```text
LoaderThread FlagResetter／error path（callAsync 失敗時 atomic 整合）は読取り済み。
RetryScheduler／shouldRetryWarmupFailure（isLoadingIR 限定）は既存テスト PASS。
D105 系への counter-evidence なし。再開しない。
```

## D. Generation / stale-result protection → guarded（defect なし）

```text
publish commit は多段 monotonicity guard：
  Commit.cpp L191 previousSequenceId／L203 generation_not_monotonic／
  L209 sequence_not_monotonic＋frozen/sealed 検証。
RebuildDispatch L651-676 sameAsPending duplicate collapse＋L1183 isRebuildObsolete。
G2-build／G1-build 逆転時の stale publish は admission で拒否される構造を確認。
「generation が存在する」だけでなく stale が publish されないことを source で確認済み。
```

## 分類

```text
Concrete Defect：STG-4-1（上記 A）
Observation：B-6 pending（早期打切り・correctness 非関連）／二重 submit 群（defer 設計内）
Already closed：STG-1／STG-2／H-01〜03／C-01／D105 系／N2-OPEN-1／p1_5 §13.4（いずれも再開せず）
```

## Phase gate

```text
Production changes: NONE
Test-only changes: NONE（reproduction GO 未発行）
commit: NOT AUTHORIZED
push: NOT AUTHORIZED
```
