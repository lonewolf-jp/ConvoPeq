# B プローブ計測・修正案 A の差分・A の測定設計詳細（2026-11-10）

## §0 承認条件への対応

| 条件（Owner） | 対応 | 証拠 |
|---|---|---|
| 1. 対象は計画書の 5 ファイルと P1〜P5 に限定 | 実際に変更したのは **2 ファイル**（`src/InputInspection.cpp`、`src/DeviceSettings.cpp`）。計画書の「5 ファイル」は誤記で、対象関数 6 つはこの 2 ファイルにある。範囲は狭い側のため逸脱ではない | §1.1、§1.4 |
| 2. 判定結果・実行順序・データ構造を変えない | 追加したのは時刻取得・整数カウンタ・`fprintf` のみ。判定式・分岐・コンテナは変更していない | §1.2 |
| 3. バイトコピーと SHA-256 で復元確認 | 復元後に SHA-256 が編集前と一致 | §1.4 |
| 4. 計測コードと計測結果を分離し、恒久変更に混入させない | プローブは `CONVOPEQ_DIAG_PROBE=1` のときだけ stderr に 1 行出力。復元後は `PROBE` 文字列 0 件 | §1.4 |
| 5. 復元後に対象ファイルのハッシュと Git 差分を照合 | 両ファイルの SHA-256 一致、`git diff` のハッシュも編集前と一致 | §1.4 |
| 6. 対象外の既存未コミット変更を変更・破棄しない | 復元後の `git diff src/DeviceSettings.cpp` は 2,082 バイトで編集前と同一ハッシュ | §1.4 |

修正実装・テスト追加・閾値変更・設定変更・commit は行っていない。

## §1 B: 関数計測プローブ（条件付き承認の範囲で実施）

### 1.1 対象（実際に変更した範囲）

| ファイル | 追加した計測 |
|---|---|
| `src/InputInspection.cpp` | `computeMaxElementDepth` の総時間と反復回数、`inspectXmlFile` の読み込み・DOCTYPE 検査・深さ検査の 3 分割、`parseXmlFile` の検証・パース分離 |
| `src/DeviceSettings.cpp` | `saveNoiseShaperState` のパース・180 要素構築・書き出し分割、`loadNoiseShaperState` と `saveSettings` の総時間（RAII スコープ） |

計画で挙げた 6 関数（`computeMaxElementDepth`、`inspectXmlFile`、`parseXmlFile`、`saveNoiseShaperState`、`loadNoiseShaperState`、`saveSettings`）はすべて計測した。追加したファイルはない（ヘルパは各 .cpp の無名名前空間に置いた）。

### 1.2 出力形式

`CONVOPEQ_DIAG_PROBE=1` のときだけ、`stderr` に 1 行ずつ出力する。

```
[PROBE] <name> ms=<経過> a=<第1値> b=<第2値>
```

### 1.3 実測（3 回、`--cli-run --cli-exit-ms 2000`）

| 計測点 | 1 回目 | 2 回目 | 3 回目 |
|---|---|---|---|
| `computeMaxElementDepth`（状態ファイル、深さ 2・反復 3,071） | 12,697.27 ms | 12,771.53 ms | 12,858.03 ms |
| `loadNoiseShaperState` 総時間 | 12,711.17 ms | 12,781.84 ms | 12,869.45 ms |
| `saveNoiseShaper.parse` | 12,754.93 ms | 12,961.48 ms | 12,924.13 ms |
| `saveNoiseShaper.build`（180 要素） | 8.79 ms | 9.20 ms | 9.59 ms |
| `saveNoiseShaper.write` | 15.67 ms | 11.65 ms | 13.62 ms |
| `saveSettings` 総時間（P1） | 12,818.54 ms | 12,988.21 ms | 12,953.79 ms |
| `inspect.load`（状態ファイル） | 1.08 ms | 0.67 ms | 1.09 ms |
| `inspect.doctype`（状態ファイル） | 8.60 ms | 5.19 ms | 5.29 ms |
| `parse.xml`（状態ファイル） | 1.41 ms | 1.51 ms | 1.39 ms |
| プロセス総経過 | 31,882 ms | 31,781 ms | 31,769 ms |

比率は次のとおり。

- 深さ検査が `saveSettings` に占める割合: **99.05% / 98.32% / 99.26%**
- 深さ検査が `loadNoiseShaperState` に占める割合: **99.89% / 99.92% / 99.91%**

`computeMaxElementDepth` の反復回数は 3,071 回である。ファイルは 278,331 バイトなので、反復回数より**タグ内の属性文字を読む回数**が支配的である（内側の `text[j]` 走査）。これが O(n²) の実体である。

### 1.4 復元と検証

| 項目 | 結果 |
|---|---|
| `src/InputInspection.cpp` の SHA-256 | 編集前 `A4F4F01A…F7663C` と一致 |
| `src/DeviceSettings.cpp` の SHA-256 | 編集前 `D76203D6…25E54D` と一致 |
| `git diff src/InputInspection.cpp` | 0 バイト（編集前と同じ） |
| `git diff src/DeviceSettings.cpp` | 2,082 バイト、ハッシュ `7DDB12CAC3480C0B…` で編集前と一致 |
| `PROBE` 文字列の残存 | 0 件 |
| 復元後の再ビルド | `build_exit=0`（プローブ無しのバイナリに戻した） |
| 復元後の実行 | `--cli-run --cli-exit-ms 2000` が正常終了 |
| 再ビルド後の再確認 | `touch`（mtime 更新）と再ビルドの後にも SHA-256 と `git diff` が一致（`A4F4F01A31F2558B` / `D76203D622CB65A4`、diff 0 / 2,082 バイト・`7DDB12CAC3480C0B`） |

バイトコピーは `tmp/probe-backup/` に残している（`tmp/` は git 管理外）。

### 1.5 検証結果と、計画の判定基準に対する訂正

計画では「P1 が外部測定の `step 6`→`step 7` と ±100 ms で一致すること」を検証条件にしたが、**この条件は達成できなかった**。理由は環境の速度変動である。

| 測定 | 値 |
|---|---|
| プローブ実行時の P1（3 回） | 12,818 / 12,988 / 12,954 ms |
| 復元後のプローブ無し実行の `step 6`→`step 7` | 9,894 ms |
| 復元後のプローブ無し実行のプロセス総経過 | 26,499 ms |
| 同じ関数を単体で計測（リポジトリ変更なし、3 回） | 12,453 / 12,594 / 12,572 ms |

同一ファイル・同一関数でも 9.9〜13.0 秒の幅がある。プローブが追加する処理は「呼び出しごとに時計 2 回＋1 行出力」と「反復ごとの整数インクリメント 1 回（3,071 回）」だけで、機械的には 1 ms 未満である。したがって 3 秒の差は、プローブのオーバーヘッドとは断定できない。

代わりに、**同一実行内の整合**で検証した。

- `saveSettings` 12,818.54 ms = パース 12,754.93 + 構築 8.79 + 書き出し 15.67 + 残り 39.15 ms
- `loadNoiseShaperState` 12,711.17 ms − 深さ検査 12,697.27 ms = 13.90 ms
- 単体計測（12,453〜12,594 ms）と、プローブが測った深さ（12,697〜12,858 ms）は同水準

以上から、**「P1 と外部窓の ±100 ms 一致」は環境変動のため判定に使えない**。以後の受入れは「同一実行内の内訳合計が一致すること」と「深さ検査の占有率が 98% 以上であること」で判定するのが妥当である（計画の判定基準の訂正として提案する）。

## §2 B 修正案 A の差分と境界テスト（提示のみ・未適用）

### 2.1 差分（`src/InputInspection.cpp` の `computeMaxElementDepth` を置換）

方針は「文字位置参照（`text[i]`）をやめ、UTF-8 のバイト列をポインタで走査する」。判定条件・分岐順序・戻り値は現行と同一にした。

```cpp
// 提案（未適用）: 現行 computeMaxElementDepth の置換案
int computeMaxElementDepth (const juce::String& text)
{
    const char* d = text.toRawUTF8();
    const int n = text.getNumBytesAsUTF8();
    int depth = 0;
    int maxDepth = 0;
    int i = 0;
    const auto at = [d] (int k) { return (unsigned char) d[k]; };

    while (i < n)
    {
        if (at (i) != '<') { ++i; continue; }

        if (i + 3 < n && at (i + 1) == '!' && at (i + 2) == '-' && at (i + 3) == '-')
        {
            int end = -1;
            for (int k = i + 4; k + 2 < n; ++k)
                if (at (k) == '-' && at (k + 1) == '-' && at (k + 2) == '>') { end = k; break; }
            if (end < 0) return -1;
            i = end + 3;
            continue;
        }

        if (i + 8 < n)
        {
            static const char* kCdata = "<![CDATA[";
            bool isCdata = true;
            for (int k = 0; k < 9; ++k)
            {
                unsigned char c = at (i + k);
                const unsigned char t = (unsigned char) kCdata[k];
                if (c >= (unsigned char) 'a' && c <= (unsigned char) 'z') c = (unsigned char) (c - 'a' + 'A');
                if (c != t) { isCdata = false; break; }
            }
            if (isCdata)
            {
                int end = -1;
                for (int k = i + 9; k + 2 < n; ++k)
                    if (at (k) == ']' && at (k + 1) == ']' && at (k + 2) == '>') { end = k; break; }
                if (end < 0) return -1;
                i = end + 3;
                continue;
            }
        }

        if (i + 1 < n && (at (i + 1) == '!' || at (i + 1) == '?'))
        {
            int end = -1;
            for (int k = i + 1; k < n; ++k) if (at (k) == '>') { end = k; break; }
            if (end < 0) return -1;
            i = end + 1;
            continue;
        }

        const bool isClosing = (i + 1 < n && at (i + 1) == '/');
        unsigned char quote = 0;
        int tagEnd = -1;

        for (int j = i + 1; j < n; ++j)
        {
            const unsigned char c = at (j);
            if (quote != 0) { if (c == quote) quote = 0; continue; }
            if (c == '"' || c == '\'') { quote = c; continue; }
            if (c == '>') { tagEnd = j; break; }
        }

        if (tagEnd < 0) return -1;

        bool selfClosing = false;
        if (! isClosing)
        {
            int k = tagEnd - 1;
            while (k > i && (at (k) == ' ' || at (k) == '\t' || at (k) == '\n' || at (k) == '\r')) --k;
            selfClosing = (at (k) == '/');
        }

        if (isClosing)
        {
            if (depth > 0) --depth;
        }
        else if (! selfClosing)
        {
            ++depth;
            if (depth > maxDepth) maxDepth = depth;
        }

        i = tagEnd + 1;
    }

    return maxDepth;
}
```

呼び出し側と `inspectXmlFile` は変更しない。

### 2.2 境界テスト（26 ケース、全一致）

リポジトリの現行関数を `tmp` 外の一時ハーネスにそのまま取り込み（`orig_depth.inc`）、提案版と全ケースで比較した。

| ケース | 現行 | 提案 | 一致 | 現行の時間 | 提案の時間 |
|---|---|---|---|---|---|
| 空文字列 | 0 | 0 | OK | 0.00 ms | 0.000 ms |
| 自己終了タグ `<a/>` | 0 | 0 | OK | 0.00 ms | 0.000 ms |
| 通常の対 `<a></a>` | 1 | 1 | OK | 0.00 ms | 0.000 ms |
| 3 階層 | 2 | 2 | OK | 0.00 ms | 0.000 ms |
| 閉じタグのみ | 0 | 0 | OK | 0.00 ms | 0.000 ms |
| 余分な閉じタグ | 1 | 1 | OK | 0.00 ms | 0.000 ms |
| コメント | 0 | 0 | OK | 0.00 ms | 0.000 ms |
| 未終端コメント | -1 | -1 | OK | 0.00 ms | 0.000 ms |
| CDATA | 0 | 0 | OK | 0.00 ms | 0.000 ms |
| 小文字 CDATA | 0 | 0 | OK | 0.00 ms | 0.000 ms |
| 未終端 CDATA | -1 | -1 | OK | 0.00 ms | 0.000 ms |
| DOCTYPE | 0 | 0 | OK | 0.00 ms | 0.000 ms |
| XML 宣言 | 0 | 0 | OK | 0.00 ms | 0.000 ms |
| 未終端の処理命令 | -1 | -1 | OK | 0.00 ms | 0.000 ms |
| 引用符内の `>` | 0 | 0 | OK | 0.00 ms | 0.000 ms |
| 単一引用符 | 1 | 1 | OK | 0.00 ms | 0.000 ms |
| 未終端タグ | -1 | -1 | OK | 0.00 ms | 0.000 ms |
| 孤立した `<` | -1 | -1 | OK | 0.00 ms | 0.000 ms |
| UTF-8 属性 | 0 | 0 | OK | 0.00 ms | 0.000 ms |
| UTF-8 タグ名 | 1 | 1 | OK | 0.00 ms | 0.000 ms |
| UTF-8 テキスト | 1 | 1 | OK | 0.00 ms | 0.000 ms |
| BOM 付き | 0 | 0 | OK | 0.00 ms | 0.000 ms |
| 前後の空白 | 0 | 0 | OK | 0.00 ms | 0.000 ms |
| 1,000 階層 | 1000 | 1000 | OK | 17.51 ms | 0.008 ms |
| 20 KB のテキスト | 1 | 1 | OK | 50.73 ms | 0.008 ms |
| 40 KB の属性 | 0 | 0 | OK | 290.35 ms | 0.022 ms |

合計: **26 件すべて一致**。

### 2.3 途中で見つけた誤り（自己終了タグ）

最初に書いた提案版は、自己終了タグを `text[tagEnd - 1] == '/'` と単純に判定していた。この版では `<a/>` の深さが 1 になり、現行の 0 と食い違った（26 件中 13 件が不一致）。現行実装は次の 2 点を守っている。

- 自己終了判定の前に、`tagEnd - 1` から空白（スペース・タブ・改行）を読み飛ばす
- 閉じタグは `depth > 0` のときだけ減算し、負にしない

§2.1 の差分はこの 2 点を反映したものである。**境界テストを先に用意しなければ、この誤りは実ファイル 2 件の比較では見つからなかった**（実ファイルの `State` 要素は自己終了ではないため、深さが偶然一致していた）。Owner の指摘どおり、2 ファイルの一致だけでは同等性の証明にならない。

### 2.4 性能

| 入力 | 現行 | 提案 |
|---|---|---|
| 1,000 階層 | 17.51 ms | 0.008 ms |
| 20 KB テキスト | 50.73 ms | 0.008 ms |
| 40 KB 属性 | 290.35 ms | 0.022 ms |
| 278 KB（実ファイル、単体計測） | 12,453〜12,594 ms | 未計測（同じ走査なので同水準と推定） |

現行の伸びは入力長の 2 乗に比例する（40 KB で 0.29 秒、278 KB で 12.5 秒は比 6.95² × 0.29 ≈ 14 秒と整合）。提案版は線形である。

### 2.5 実装時の順序（Owner 指定）

1. 修正前の結果を固定する（本報告の 26 ケースと、実ファイル 2 件の depth）
2. 新実装を追加する
3. 同一入力で旧実装・新実装の結果を比較する（本報告のハーネスをそのまま使う）
4. 既存テストと追加の境界テストを実行する
5. 最後に起動・終了時間を再測定する

## §3 A: D1 測定設計の詳細（要求 ID・完了 ID・有効観測数）

### 3.1 識別子（epoch をキーにしない）

Owner の指摘どおり、epoch だけでは同一 epoch に複数の要求が来た場合に区別できない。次の識別子を使う。

| 識別子 | 型 | 付与位置 | 説明 |
|---|---|---|---|
| `requestId` | `std::uint64_t` | 退役要求点（`AudioEngine.Commit.cpp:485` 付近） | 単調増加。1 要求につき 1 回だけ払い出す |
| `requestTsNs` | `std::uint64_t` | 同上 | `steady_clock` の ns |
| `epoch` | `std::uint64_t` | 同上 | `world->generation`（対応付けには使わない。記録用） |
| `completeTsNs` | `std::uint64_t` | 解放点（`terminalReclaim` の同期経路、`TerminalReclaimAuthority::drain` の 2 経路） | 要求と同じ時計 |
| `path` | 列挙 | 解放点 | `sync` / `drain` / `drainAll`（終了時の強制解放） |

`requestId` は解放待ちのエントリ（`TerminalReclaimAuthority::Entry`）に持たせ、解放時にそのまま取り出す。**対応付けのキーは `requestId` のみ**とする。

### 3.2 記録する内容

- 要求イベント: `requestId`, `epoch`, `requestTsNs`
- 完了イベント: `requestId`, `epoch`, `completeTsNs`, `path`
- 集計: `requests`, `completions`, `duplicates`, `orphans`（要求の無い完了）, `N_valid`, `maxLatencyNs`, `p50`, `p99`

### 3.3 有効観測数 `N_valid` の定義

```
V = { r ∈ requests | 完了イベントがちょうど 1 件あり、completeTsNs >= requestTsNs }
N_valid = |V|
```

- 完了が 0 件の要求: 有効に数えない（遅延 0 として扱わない）
- 完了が 2 件以上: その要求は無効。`duplicates` に計上
- 要求の無い完了: `orphans` に計上し、無効
- `drainAll`（終了時の強制解放）の完了: `path=drainAll` として別集計し、`N_valid` には含めない
- 実行終了時に未完了の要求が 1 件でもあれば `insufficient`

### 3.4 `N_req` の統計量と信頼度

- 統計量: 有効観測 1 件ごとの遅延 `L_i`。判定は「すべての `L_i` が 50 ms 以下」。
- 目標: 「50 ms を超える率が 5% 未満であることを 95% 信頼で示す」。
- 0 件の超過が観測された場合に必要な件数は `n >= ln(0.05)/ln(0.95) = 58.4` なので、**`N_req = 59`**。
- 合格条件は `N_valid >= N_req` かつ `max(L_i) <= 50 ms`。
- 前提: 観測が独立同分布であること。実際の退役は負荷・epoch・スケジューリングの影響を受けて相関するため、この 59 件は**下限**として扱う。相関がある場合の実効信頼度は 95% より低い。
- したがって、`N_req = 59` を提案値とし、採用は Owner の承認による（数値確定は保留との判断を尊重する）。負荷条件を変えた 2 群（例: 通常と IR リロード併用）でそれぞれ 59 件を満たすことを推奨する。

### 3.5 判定

| 条件 | result |
|---|---|
| `N_valid` < `N_req`、または未完了の要求がある、または `orphans` > 0 | `insufficient` |
| `max(L_i)` > 50 ms | `fail` |
| 上記以外 | `pass` |

`insufficient` は pass に読み替えない。50 ms は変更しない。

## §4 CI 案 1 の互換性確認（実行結果）

証跡の実体を使わず、`isr-verify-v5.ps1` と `isr-verify-common.ps1` を一時ディレクトリへ複製し、`evidence/` に検証用の JSON を置いて実行した（リポジトリの `evidence/` は変更していない）。

| # | 入力 | 結果 |
|---|---|---|
| C1 | 既存 v1（追加フィールドなし）で `pass` | 受理（回帰なし） |
| C2 | `result=fail` + `reason` + `observations`/`required` | 拒否（`HB reorder simulation failed`） |
| C3 | `result=pass` + `observations=0`/`required=3` | **受理（穴）** |
| C4 | `result=pass` + `reason="insufficient_observations"` | **受理（穴）** |
| C5 | `result="insufficient"` | 拒否（許容値が `pass,fail` のみ） |
| C6 | 未知フィールド（トップ・要素） | 受理（後方互換） |

### 4.1 見つかった穴と、v5 への追加（提案・未適用）

案 1 では、アプリが `fail` を出せば v5 は確実に落ちる（C2）。一方、**アプリの不具合で `pass` と 0 件観測が同時に出た場合、v5 は止められない**（C3・C4）。`insufficient` を `pass` にしない要件を CI 側でも担保するには、v5 に次の 2 検査を足す必要がある。

```powershell
# 提案（未適用）: isr-verify-v5.ps1 の scenarioResults ループ内、result 検査の後
if ($scenario.PSObject.Properties.Name -contains 'reason' -and
    [string]$scenario.reason -eq 'insufficient_observations') {
    throw "HB scenario '$($scenario.name)' is insufficient (observations were not established)"
}

if (($scenario.PSObject.Properties.Name -contains 'observations') -and
    ($scenario.PSObject.Properties.Name -contains 'required') -and
    ([int64]$scenario.observations -lt [int64]$scenario.required)) {
    throw "HB scenario '$($scenario.name)' has insufficient observations. observations=$($scenario.observations) required=$($scenario.required)"
}
```

この 2 検査を**一時コピーに適用して再実行**した結果は次のとおりで、C3・C4 の穴は閉じた（リポジトリの v5 は変更していない。v5 の mtime は 2026-07-30 のままで、今回書き込んでいない）。

| # | 2 検査を入れた v5 | 結果 |
|---|---|---|
| C1 | 既存 v1 pass | 受理（回帰なし） |
| C2 | fail + reason | 拒否（`HB reorder simulation failed`） |
| C3 | pass + observations=0 | **拒否**（`observations=0 required=3`） |
| C4 | pass + reason=insufficient | **拒否**（`is insufficient`） |
| C5 | result=insufficient | 拒否（許容値外） |
| C6 | 未知フィールド pass | 受理 |

案 1 の設計はそのままで 2 検査だけを足せば足りる。案 2（v2 化）は不要と判断できる。

### 4.2 実行環境の注意

リポジトリの `.github/scripts/*.ps1` は UTF-8（BOM なし）である。Windows PowerShell 5.1 で実行すると日本語コメントの解釈が崩れて構文エラーになる（実測: `isr-verify-common.ps1:136` で `}` の予期しないトークン）。**PowerShell 7（`pwsh`）で実行すれば通る**（実測: 6 ケースすべて判定どおり）。CI の実行系が 5.1 でないことを確認する必要がある。

## §5 `AGENTS.md`

指示に従い、**変更していない**（追記の拡張・関連ドキュメントへの展開も行っていない）。原因関数が確定したことに伴う更新は、承認後に次の 1 行の置換だけを行う予定である（今回は未適用）。

- 現在: 「原因の関数は未特定（`saveNoiseShaperState` の既存 XML パース／`loadNoiseShaperState` が候補）」
- 置換案: 「原因は `computeMaxElementDepth`（`src/InputInspection.cpp:13`）の O(n²) 文字走査。`juce::String::operator[]` を 1 文字ずつ使うため、277 KB で 12 秒」

## §6 未確認事項

- 修正案 A は未適用であり、実装後の起動・終了時間は測っていない（推定: 起動 12.5 秒 → 約 2 秒）。
- v5 への 2 検査は未適用。
- CI の PowerShell 実行系（5.1 か 7 か）は未確認。
- 環境の速度変動が大きく（同一関数で 9.9〜13.0 秒）、単発の比較では ±100 ms の一致を確認できない。
- `N_req = 59` は独立同分布を仮定した下限であり、相関を考慮した実効信頼度は未評価。
- `drainAll` 経路の完了を `N_valid` から除外する設計は、終了時の強制解放が多い実行では観測数が減る。妥当性は実装後の実測で確認する。
- TSan／LSan、実機負荷時の retire 遅延、聴感確認は引き続き未確認。

## §7 成果物

- 本報告書: `doc/work113/20261110-1418_プローブ計測/report.md`
- 一時ハーネス（リポジトリ外）: `C:\Users\user\AppData\Local\Temp\opencode\b0612\` 配下（`probe\`、`equiv\`、`v5test\`）
- バイトコピー: `tmp/probe-backup/`（git 管理外）
- 計測ログ: `C:\Users\user\AppData\Local\Temp\opencode\b0612\probe\runs\run1..3\probe.log`
