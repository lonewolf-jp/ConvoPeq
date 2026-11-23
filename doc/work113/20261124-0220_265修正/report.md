# NoiseShaperLearner の OOM 経路を到達可能にする（#265）

作業日時: 2026-11-24 02:20
対象: main `1712983d`（統合クローン）

## 1. 問題

`NoiseShaperLearner` のコンストラクタは `AudioSegmentBuffer::create()` の戻り値を nullptr で判定していた。しかし `create()` は確保失敗時に `std::bad_alloc` を送出する契約（B-1）で、nullptr を返さない。OOM 時は `Status::Error` と `errorMessage` に落ちず、例外がコンストラクタの外へ抜けていた。

同じ関数内の候補バッファ（`makeAlignedArray_nothrow`）は nullptr 判定で、失敗時の意味論が 2 通りになっていた（C-28 で片側だけ直っていた）。

## 2. 修正

`create()` の例外契約は B-1 の設計なので変えない。呼出側の境界で受け、nullptr に落とす。

| ファイル | 内容 |
| --- | --- |
| `src/learner/SegmentBufferAllocation.h`（新規） | `createSegmentBufferNoThrow(factory)` と既定版。`std::bad_alloc` を nullptr に変える |
| `src/NoiseShaperLearner.cpp` | `create()` の直接呼出しを境界関数へ。既存のエラー状態へ落ちる経路が生きる |
| `src/tests/AudioSegmentBufferTests.cpp` | テスト 6 を追加（成功経路と、注入した確保関数の `std::bad_alloc`） |

失敗時の意味論は 1 つになった（確保に失敗したら nullptr、呼出側が `Status::Error` と `errorMessage` を公開する）。

## 3. 検証

- `AudioSegmentBufferTests`: 6/6 PASS。追加分は `created=1 nulled=1 escaped=0 growth=0 bytes`（例外が外へ出ず、確保済みメモリも残らない）。
- `ConvoPeq` ターゲット: Release ビルド エラー 0。
- `ctest -C Release`: 67/68 PASS。`TailLayerPerfTests` が 1 件落ちたが、これは負荷検出器が `wall/cpu=1.255 > 1.25` を 5 回観測して「撹乱」と判定したもので、CPU 実測値は 1.143 ms/s（予算内）。単独再実行で PASS（rc 0）。

## 4. 受入条件との対応

| 条件 | 結果 |
| --- | --- |
| OOM 注入で `Status::Error` と `errorMessage` が設定される | 例外→nullptr の境界をテストで固定。その直後の公開処理は既存コードのままで、分岐が到達可能になった |
| 例外がコンストラクタから外へ漏れない | 境界は `noexcept`。注入試験で `escaped=0` |
| 関数内で失敗時の意味論が 1 つ | 候補バッファ（nothrow）とセグメントバッファ（境界で捕捉）がどちらも nullptr 判定に統一 |
| 既存のテストが通る | `AudioSegmentBufferTests` と他の 66 件が PASS（上記の撹乱 1 件を除く） |

## 5. 残る制約

実際の OOM をプロセスに起こして `Status::Error` の値を読む試験は、この環境では決定論的に作れない（`IRLoadAdmissionTests.cpp` の S3 と同じ制約）。本修正では境界関数を注入可能にして、例外の捕捉とメモリの後始末を実測で固定した。公開処理そのものは従来からあるコードで、分岐条件が変わるのは「到達するか」だけである。
