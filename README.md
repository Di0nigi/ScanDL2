# ScanDL 2.0: A Generative Model of Eye Movements in Reading Synthesizing Scanpaths and Fixation Durations

This repository contains ScanDL 2.0, described in [ScanDL 2.0: A Generative Model of Eye Movements in Reading Synthesizing Scanpaths and Fixation Durations](https://doi.org/10.1145/3725830), together with pretrained weights for paragraph-level and sentence-level scanpath generation.

The model, pretrained inference API, and research functionality originate from the [authors' implementation](https://github.com/DiLi-Lab/ScanDL-2.0). In this project, the repository has been reorganized under the `ScanDL2` package, and `handler.py` and a Gradio interface in `app.py` have been added. The reorganization does not introduce a new model architecture.

## Setup

Run the commands and Python examples below from the project root—the directory containing `ScanDL2/`. This ensures package imports and the relative model paths resolve correctly.

### Install requirements

The code uses PyTorch and Hugging Face libraries.

```bash
python -m pip install -r ScanDL2/requirements.txt
```

Install a PyTorch build appropriate for your platform, and install Gradio to use the web interface; neither is included in this requirements file.

```bash
python -m pip install torch gradio
```

Use a separate environment from Eyettention because their dependency versions differ. The requirements contain older version pins and require a compatible Python version. For GPU inference, install CUDA-enabled PyTorch. The local implementation selects CUDA when available and otherwise CPU, where diffusion inference can be slow. Its distributed initialization requires a local socket. BERT and GPT-2 assets must be available from Hugging Face or the local cache.

## Using pre-trained ScanDL 2.0

The authors pretrained ScanDL 2.0 on EMTeC for paragraph-level generation and CELER for sentence-level generation. Both versions generate fixation locations and fixation durations without training from scratch.

To obtain missing weights, download `models.zip` from the [upstream releases](https://github.com/DiLi-Lab/ScanDL-2.0/releases). Place the model directories under `ScanDL2/models/` and ensure that the paths in [PATHS.py](PATHS.py) match their locations:

```text
ScanDL2/models/
├── sentence/
│   ├── scandl-module/
│   │   ├── ema_0.9999_080000.pt
│   │   └── training_args.json
│   └── fixdur-module/
│       ├── seq2seq_fixdur.pt
│       ├── hyperparameters.json
│       └── min_max_scaler.pkl
└── paragraph/
    ├── scandl-module/     # same filenames as above
    └── fixdur-module/     # same filenames as above
```

### Python example

```python
import torch
from ScanDL2 import ScanDL2

model = ScanDL2(
    text_type="sentence",
    bsz=2,
    save=None,
    filename=None,
)
model.eval()
with torch.no_grad():
    output = model(texts=["The quick brown fox jumps over the lazy dog."])

print(output)
```

Set `text_type="paragraph"` to use the paragraph model.

### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `text_type` | `"sentence"` | Either `"sentence"` or `"paragraph"`; selects the corresponding pretrained modules |
| `bsz` | `2` | Inference batch size; adjust to available memory |
| `save` | `None` | Optional directory in which to save the output as JSON |
| `filename` | `None` | Optional output filename; defaults to `scandl2_outputs.json` when `save` is set |

For example, setting `save="outputs"` and `filename="example.json"` saves results to `outputs/example.json`. Repeated calls with the same output name overwrite that file.

### Input and output

The model accepts a single string or a list of strings. Use nonempty sentences or paragraphs appropriate for the selected checkpoint. Input and scanpath lengths are bounded by the model configuration; split longer documents before inference.

The returned dictionary contains:

| Key | Contents |
| --- | --- |
| `predicted_sp_words` | Predicted scanpaths as lists of words in fixation order |
| `predicted_sp_ids` | Corresponding word-position indices |
| `original_sn` | Each original input as a list of words |
| `predicted_fix_durs` | Predicted fixation durations in milliseconds |
| `unique_idx` | An identifier for each input within the current call |

Word indices originate from the model's sequence including special tokens; do not assume they are zero-based offsets into `original_sn`. Use the returned word, index, and duration lists together. Generation is stochastic.

## Gradio interface

The added [app.py](app.py) provides a web interface to the existing model API.

```bash
python -m ScanDL2.app
```

Open `http://localhost:7860`, enter text, select **sentence** or **paragraph**, and click **Run**. Each nonempty line is processed as a separate input in either mode, so keep each paragraph on one line. Results include a fixation table and raw JSON output. The model is loaded and cached when first requested.

The app uses port 7860 by default, configurable through `PORT`, and binds to `0.0.0.0`.

## Endpoint handler

The added [handler.py](handler.py) defines an `EndpointHandler` adapter intended to accept requests with text in `inputs` and inference settings in `parameters`:

```json
{
  "inputs": ["The quick brown fox jumps over the lazy dog."],
  "parameters": {
    "text_type": "sentence",
    "bsz": 2
  }
}
```

The adapter is intended to select the sentence or paragraph model and return the model's output dictionary. It does not itself start an HTTP server.

If omitted, `text_type` defaults to `"sentence"` and `bsz` to `2`. The batch size must be a positive integer and is applied to both model components.

## Training, inference, and evaluation

The original research workflow trains the ScanDL module for fixation locations and the fixation-duration module for durations, then evaluates their combined predictions. The following describes the corresponding data and code in the reorganized repository.

### Download the data

- **CELER:** follow the instructions in the [dataset repository](https://github.com/berzak/celer).
- **ZuCo:** download from the [OSF repository](https://osf.io/q3zws/). The dataset requires substantial storage.
- **Beijing Sentence Corpus (BSC):** download from the [OSF repository](https://osf.io/vr3k8/).
- **EMTeC:** download from the [OSF repository](https://osf.io/ajqze/) or use the authors' [Python download utility](https://github.com/DiLi-Lab/EMTeC/blob/main/get_et_data.py).

Adapt dataset and output paths in [CONSTANTS.py](CONSTANTS.py). Check the expected filenames and directory spelling in [sp_load_celer_zuco.py](scandl_module/scripts/sp_load_celer_zuco.py).

### Preprocess the training and test data

Preprocessing takes time, so save processed datasets and reuse the same split definitions across comparable experiments. The data-loading and processing utilities are in `scandl_module/scripts/sp_load_celer_zuco.py`; [create_data.py](create_data.py) contains preprocessing for training the sentence and paragraph models on CELER and EMTeC.

Some internal paths still need alignment with the rearranged repository: `create_data.py` reads configuration from an old directory and writes beneath `scandl2_pkg`. Its BSC branch is not implemented. Review those paths before using it for a new training run.

### ScanDL module

The fixation-location training code is in [sp_train.py](scandl_module/scripts/sp_train.py), with its launcher in [sp_run_train.py](scandl_module/scripts/sp_run_train.py). Configure the appropriate `SCANDL_MODULE_TRAIN_PATH*` and `SCANDL_MODULE_INF_PATH*` values in `CONSTANTS.py` for the chosen dataset and experiment.

The launcher retains working-directory and distributed-environment assumptions from the original layout. Review those alongside your processed-data and GPU settings before training. Pretrained location prediction is also exposed by `ScanDLModule` in [model.py](model.py).

### Fixation-duration module

The duration-training code is in [train_seq2seq.py](fix_dur_module/train_seq2seq.py). Configure the corresponding `FIXDUR_MODULE_TRAIN_PATH*` and `FIXDUR_MODULE_INF_PATH*` values in `CONSTANTS.py`. Keep the fitted duration scaler, model weights, and hyperparameters together. Pretrained duration prediction is exposed by `FixdurModule` in `model.py` and is applied automatically by `ScanDL2`.

### Sentence-level and paragraph-level training

The sentence version uses CELER and the paragraph version uses EMTeC. Their full-data training output locations are configured through the `COMPLETE_SCANDL_MODULE_TRAIN_PATH_*` and `COMPLETE_FIXDUR_MODULE_TRAIN_PATH_*` constants. Train the location and duration components with matching data and configurations, then point `PATHS.py` at their output directories for inference.

### Evaluation and the diffusion-only ablation

The paper includes evaluation against human scanpaths and a diffusion-only duration ablation, ScanDL diff-dur. Refer to the [original repository](https://github.com/DiLi-Lab/ScanDL-2.0) for those experiment definitions and evaluation tools; their original evaluation modules are not present under those names in this checkout.

Local checks can be run from the project root:

```bash
python -m unittest discover -s ScanDL2/tests
```

These include real inference and require dependencies, model assets, and local socket access. They check execution and output structure rather than reproducing the paper's reported metrics.

## Citation

```bibtex
@article{bolliger2025scandl2,
	author = {Bolliger, Lena S. and Reich, David R. and J\"{a}ger, Lena A.},
	title = {ScanDL 2.0: A Generative Model of Eye Movements in Reading Synthesizing Scanpaths and Fixation Durations},
	year = {2025},
	issue_date = {May 2025},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	volume = {9},
	number = {ETRA5},
	url = {https://doi.org/10.1145/3725830},
	doi = {10.1145/3725830},
	abstract = {Eye movements in reading have become a vital tool for investigating the cognitive mechanisms involved in language processing. They are not only used within psycholinguistics but have also been leveraged within the field of NLP to improve the performance of language models on downstream tasks. However, the scarcity and limited generalizability of real eye-tracking data present challenges for data-driven approaches. In response, synthetic scanpaths have emerged as a promising alternative. Despite advances, however, existing machine learning-based methods, including the state-of-the-art ScanDL (Bolliger et al. 2023), fail to incorporate fixation durations into the generated scanpaths, which are crucial for a complete representation of reading behavior. We therefore propose a novel model, denoted ScanDL 2.0, which synthesizes both fixation locations and durations. It sets a new benchmark in generating human-like synthetic scanpaths, demonstrating superior performance across various evaluation settings. Furthermore, psycholinguistic analyses confirm its ability to emulate key phenomena in human reading. Our code as well as pre-trained model weights are available via https://github.com/DiLi-Lab/ScanDL-2.0.},
	journal = {Proceedings of the ACM on Human-Computer Interaction},
	month = may,
	articleno = {5},
	numpages = {30},
	keywords = {neural networks, scanpath generation, eye movements, reading, diffusion models}
}
```

## Related paper

The fixation-location module builds on **ScanDL: A Diffusion Model for Generating Synthetic Scanpaths on Texts** (Bolliger et al., EMNLP 2023). [Paper and citation metadata](https://aclanthology.org/2023.emnlp-main.960/).

```bibtex
@inproceedings{bolliger2023scandl,
  title={ScanDL: A Diffusion Model for Generating Synthetic Scanpaths on Texts},
  author={Bolliger, Lena S. and Reich, David R. and Haller, Patrick and Jakobi, Deborah N. and Prasse, Paul and Jäger, Lena A.},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  year={2023},
  pages={15513--15538},
  doi={10.18653/v1/2023.emnlp-main.960},
  url={https://aclanthology.org/2023.emnlp-main.960/}
}
```

## License

The included [LICENSE](LICENSE) is **CC0 1.0 Universal**, a public-domain dedication with a fallback license. The full file contains the applicable terms and disclaimers. Preserve provenance and cite the research when using it in scientific work. External datasets, pretrained language-model assets, and dependencies retain their own terms; this README does not assign them a new license.
