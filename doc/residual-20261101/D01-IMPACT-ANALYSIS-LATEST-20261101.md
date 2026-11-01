# D-01 影響調査（最新ソース基準・2026-11-01）

- 対象: 「D-01 の製品 2 行」＝ `PublicationAdmission::evaluate()` の fading 判定に
  テスト専用フック `engine.testFadingRuntimePresent()` を戻す変更
- 制約: **製品コードは変更していない**（Owner 判断: HOLD、影響調査のみ許可）
- 基準: `main` = `origin/main` = **`fd92c6e7`**（最新）
- 参照文書: `doc/residual-20261101/D01-STG8-ANALYSIS-20261101.md`（基準 `a12e05f2`）

## 1. 調査結果（要旨）

**最新ソースでは、当該フックは既に存在し、テストバイナリでは有効である。**
「製品 2 行」を適用してもテスト側の挙動は変わらず、変わるのは製品バイナリだけである。

| # | 事実 | 証拠 |
|---|---|---|
| 1 | `evaluate()` の fading 判定にフックが**既にある** | `src/audioengine/PublicationAdmission.cpp:51-56` |
| 2 | その形（`#if` ガード付き）は **2026-08-07 の `f0d6406f`** で導入された | `git blame -L 50,58 src/audioengine/PublicationAdmission.cpp` |
| 3 | `AudioEngineHarness` は `CONVOPEQ_UNIT_TESTS=1` でコンパイルされる | `CMakeLists.txt:2278-2280`、`build/AudioEngineHarness.vcxproj` の `PreprocessorDefinitions`（8 箇所一致） |
| 4 | `PublicationAdmission.cpp` はハーネスに**直接コンパイルされる**（共有ライブラリ経由ではない） | `build/AudioEngineHarness.vcxproj` の `<ClCompile Include="...PublicationAdmission.cpp">`、`build/AudioEngineHarness.dir/Release/PublicationAdmission.obj` |
| 5 | friend とメンバは **ODR 回避のため無条件**で存在し、テスト限定なのは**評価分岐だけ** | `src/audioengine/AudioEngine.h:3984-3988`（コメント）、同 4005-4009 |
| 6 | テストはフックを実際に多用している | `STG8RecoveryObligationTests.cpp` に 32 箇所、`DeferredFlowIntegrationTests.cpp` に 4 箇所 |

現行のコード（`fd92c6e7`）:

```cpp
    // 5. Fading active check → defer
    const bool hasFading =
#if defined(CONVOPEQ_UNIT_TESTS)
        engine.testFadingRuntimePresent() ||
#endif
        engine.hasFadingRuntimeInWorld(
            engine.makeRuntimeReadHandle(ctx));
    if (hasFading)
        return Decision::DeferredFadingActive;
```

## 2. 旧分析（`D01-STG8-ANALYSIS-20261101.md`）との差異

旧分析 §2 RC-1 は「getter の呼び出し元は 0」「その分岐は現存しない」と結論していた。
**これは最新ソースとは一致しない。**

- `git show a12e05f2:src/audioengine/PublicationAdmission.cpp` を実行すると、
  当該フック（`#if` ガード付き）は**その時点でも存在していた**。
- 旧分析の根拠は「情報源から取得できた ConvoPeq のソースが 10 月 10 日版まで」であり、
  その版と現行ソースの差分を反映していなかった可能性が高い。

**したがって「フックが死んでいる」という前提は成立しない。** 前提が崩れると、
「製品 2 行で STG-8-D2 の retention 検証が実質化する」という結論も成立しない。

## 3. 2 行を適用した場合の影響（適用はしていない）

| 対象 | 影響 |
|---|---|
| `AudioEngineHarness`（テスト） | **変化なし**。既に `CONVOPEQ_UNIT_TESTS=1` で分岐が生きている |
| `ConvoPeq`（製品アプリ） | `#if` / `#endif` を外すと、admission 評価ごとに `atomic<bool>` の acquire load が 1 回増える。`evaluate()` は非 RT 経路（publish 評価時）でのみ呼ばれる |
| 判定結果 | **不変**。フラグの書き手は `src/tests/AudioEngineHarness/DeferredPublicationTestAccess.h` のみで、製品ターゲットには含まれない。既定値は `false` |
| 設計上の意味 | テスト専用フックが**製品の判定ロジックに常駐**する。`AudioEngine.h:3984-3988` の「Production ビルドではこの評価分岐は存在しない」という記述と矛盾する |
| バイナリサイズ・性能 | **未測定**（適用しないため測定していない） |

## 4. 推奨（Owner 判断用）

1. **「製品 2 行」は適用しない**。テストの前提は既に成立しており、目的（STG-8-D2 の
   retention 検証の実質化）は現行コードで達成されている。
2. `D01-STG8-ANALYSIS-20261101.md` の RC-1 は**訂正が必要**（本節 2）。
   旧分析の RC-2（R8-2 の単一書き込み者契約違反）は独立した指摘であり、この訂正の影響を受けない。
3. STG-8-D2 に残る断続失敗があるなら、原因は「フックの欠落」ではなく
   **タイミング依存**（B と同種）として扱う。その調査は B の再現条件収集と共通化できる。

## 5. 未確定事項

- 旧分析が「10 月 10 日版」のどの文書を根拠にしたかの特定（`ConvoPeq.md` の版と
  `f0d6406f` の前後関係）。次回 `ConvoPeq.md` を更新した際に、該当箇所の有無を再確認する。
- 製品バイナリでの当該分岐の有無を、`dumpbin` で確認する余地はある（今回は不要と判断）。
