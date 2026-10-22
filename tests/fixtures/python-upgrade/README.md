# Public regression audio (not client recordings)

- `en-jfk`: John F. Kennedy's 1961 inaugural speech excerpt, public-domain US federal government work, from [whisper.cpp samples](https://github.com/ggml-org/whisper.cpp/tree/master/samples). Converted WAV to FLAC with ffmpeg.
- `ru-fleurs-0`, `ru-fleurs-5`, `es-fleurs-0`, `es-fleurs-1`: [FLEURS](https://huggingface.co/datasets/google/fleurs), Google, **CC BY 4.0**, [license](https://creativecommons.org/licenses/by/4.0/), Russian `ru_ru` and Latin American Spanish `es_419`, test split, row indices in filenames. Audio converted from dataset WAV bytes to FLAC using ffmpeg; references from `transcription` and `raw_transcription`. This is an attribution notice, not an endorsement by Google or the speakers.

Dataset: Conneau et al., *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech* (2022). [Dataset card](https://huggingface.co/datasets/google/fleurs/blob/main/README.md).

Source parquet files retrieved 2026-10-22:
- https://huggingface.co/datasets/google/fleurs/resolve/refs%2Fconvert%2Fparquet/ru_ru/test/0000.parquet
- https://huggingface.co/datasets/google/fleurs/resolve/refs%2Fconvert%2Fparquet/es_419/test/0000.parquet

`manifest.json` records references, dataset IDs, provenance, and SHA-256 for the actual fixture bytes. Five short utterances test regressions; they do not establish corpus-level ASR accuracy. Number formatting and proper names require manual inspection alongside WER/CER.
