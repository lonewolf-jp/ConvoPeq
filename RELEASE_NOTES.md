# Release notes

## Unreleased

### Oversampling gain correction is now the default (Part 8 E)

**English.** Oversampling gain correction is now enabled by default. The round-trip gain is unity; earlier versions applied 0.75 / 0.5625 / 0.421875 at 2× / 4× / 8× (−2.50 / −5.00 / −7.50 dB). Existing sessions play 2.5–7.5 dB louder at unchanged settings. Preset values are unchanged. Re-check output make-up gain and limiter activity after updating. `-DCONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF` restores the previous behaviour.

**日本語.** オーバーサンプリングの利得補正を既定で有効にしました。ラウンドトリップ利得は 1.0 になり、従来は 2×/4×/8× で 0.75 / 0.5625 / 0.421875（−2.50 / −5.00 / −7.50 dB）でした。既存のセッションは同じ設定のまま 2.5〜7.5 dB 大きく再生されます。プリセットの数値は変わりません。更新後は出力メイクアップとリミッタの作動を確認してください。旧挙動は `-DCONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF` でビルドできます。
