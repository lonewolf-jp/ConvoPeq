```text
ConvoPeq/
|-- src/ # メインC++ソースコード（DSP、エンジン、UI）
| |-- core/ # スナップショット/RCUの基盤とスレッドセーフな状態引き継ぎ
| |-- AudioEngine.* # オーディオ処理コア
| |-- EQProcessor.* # 20バンドパラメトリックEQ
| |-- ConvolverProcessor.* # IR畳み込み処理
| `-- MainApplication.* # アプリケーションのエントリ/ランタイム配線
|-- manual/ # ユーザーマニュアル (英語/日本語)
|-- resources/ # アプリケーションのリソース (アイコン、アセット)
|-- sampledata/ # サンプルIR/EQファイル
|-- JUCE/ # JUCEフレームワークのソース (外部依存関係)
|-- r8brain-free-src/ # r8brainのソース (外部依存関係)
|-- build/ # 生成されたビルド出力 (CMake/Ninja)
|-- README.md
|-- ARCHITECTURE.md
|-- SOUND_PROCESSING.md
|-- BUILD_GUIDE_WINDOWS.md
`-- HOW_TO_USE.md
```