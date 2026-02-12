# nanobuddy

Lightweight wake word detection — streaming inference, training, and tooling.

Built on [nanowakeword](https://github.com/arcosoph/nanowakeword) by [arcosoph](https://github.com/arcosoph) and [hey-buddy](https://github.com/dscripka/hey-buddy) by [dscripka](https://github.com/dscripka).

## Install

```bash
pip install nanobuddy              # inference only (numpy + onnxruntime)
pip install "nanobuddy[detect]"    # + sounddevice for mic detection
pip install "nanobuddy[tools]"     # full CLI: training, collection, evaluation
```

## Usage

```python
from nanobuddy import WakeEngine

engine = WakeEngine()  # uses bundled default model

score = engine.predict(audio_chunk)  # int16, 16kHz, 80ms chunks
if score > 0.5:
    print("wake word detected")
```

### Background detection

```python
from nanobuddy import WakeDetector

detector = WakeDetector(
    threshold=0.5,
    patience=4,
    on_detected=lambda: print("detected!"),
)
detector.start()  # listens on default mic
```

### CLI

```
nanobuddy collect clarvis          # record wake word samples
nanobuddy validate ./collected     # check sample quality
nanobuddy stats ./collected        # show collection stats
nanobuddy train -c config.yaml     # train a model
nanobuddy evaluate --model m.onnx --positive-dir pos/ --negative-dir neg/
```

## Architecture

- **Inference**: incremental mel-spectrogram → speaker embeddings → ONNX classifier, with Silero VAD gating and patience-based confirmation
- **Training**: 11 architectures (DNN, CNN, LSTM, GRU, CRNN, TCN, RNN, QuartzNet, Transformer, Conformer, E-Branchformer), YAML config with dot-notation overrides
- **Collection**: interactive terminal recorder with keyboard marking or hands-free auto-capture, WebRTC-style quality validation via Silero VAD

## License

MIT
