import os
import sys
import json
import tempfile
from typing import Union, List, Dict, Any

import gradio as gr
import torch



from ScanDL2 import ScanDL2 

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))




_MODELS: Dict[str, ScanDL2] = {}


def get_model(text_type: str) -> ScanDL2:
    """Load and cache a ScanDL2 model for the given text type."""
    if text_type not in _MODELS:
        # `save=None` -> we handle saving ourselves in the Gradio app
        _MODELS[text_type] = ScanDL2(text_type=text_type, save=None)
    return _MODELS[text_type]


def predict(
    text: str,
    text_type: str,
    progress=gr.Progress(track_tqdm=True),
) -> Dict[str, Any]:
    """
    Run ScanDL2 on the input text and return a structured result.
    """
    if text is None or text.strip() == "":
        raise gr.Error("Please provide some input text.")

   
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    if len(lines) == 0:
        raise gr.Error("Input text is empty after cleaning.")

    progress(0.05, desc="Loading ScanDL2 model...")
    model = get_model(text_type)

    progress(0.15, desc="Running ScanDL + FixDur modules...")
    with torch.no_grad():
        output = model(texts=lines)

    return output


def format_output(output: Dict[str, Any]) -> str:
    """Pretty-print the ScanDL2 output for display in the UI."""
    if output is None:
        return ""

    n = len(output.get("original_sn", []))
    lines: List[str] = []

    for i in range(n):
        original_sn = output["original_sn"][i]
        sp_words = output["predicted_sp_words"][i]
        sp_ids = output["predicted_sp_ids"][i]
        fix_durs = output["predicted_fix_durs"][i]

        lines.append(f"### Example {i + 1}")
        lines.append("")
        lines.append(f"**Original text:** {' '.join(original_sn)}")
        lines.append("")

        # Build a readable scanpath table
        lines.append("| # | Word | Word index | Fixation duration (ms) |")
        lines.append("|---|------|-----------|------------------------|")
        for step, (w, wid, dur) in enumerate(zip(sp_words, sp_ids, fix_durs), start=1):
            lines.append(f"| {step} | {w} | {wid} | {dur} |")
        lines.append("")

    return "\n".join(lines)


def format_json(output: Dict[str, Any]) -> str:
    """Return the raw JSON string of the output."""
    if output is None:
        return "{}"
    return json.dumps(output, indent=2, ensure_ascii=False)



DESCRIPTION = """
# ScanDL 2.0

**ScanDL 2.0** predicts human-like **eye-movement scanpaths** (which words are fixated, in what order)
and their **fixation durations** (in milliseconds) directly from text.

This Space wraps two jointly-trained modules:

1. **ScanDL module** — a discrete diffusion model that generates fixation *locations* (a scanpath) over the input text.
2. **FixDur module** — a sequence-to-sequence model that predicts the *duration* of each fixation.

### How to use
1. Paste your text in the box below.
   - In **sentence** mode, each line is treated as a separate sentence.
   - In **paragraph** mode, each line is treated as a separate paragraph.
2. Choose the text type (must match the model checkpoint you want to use).
3. Click **Run**.

### Output
- A human-readable scanpath table with predicted fixation durations per word.
- The raw JSON output (fixated words, word indices, and durations).
"""

EXAMPLES = [
    [
        "The quick brown fox jumps over the lazy dog.",
        "sentence",
    ],
    [
        "Researchers have long been interested in how humans process written language.\n"
        "Eye-tracking studies reveal where and for how long readers fixate on words.",
        "paragraph",
    ],
]


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="ScanDL 2.0 — Eye-Movement Scanpath Prediction",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=3):
                text_in = gr.Textbox(
                    label="Input text",
                    placeholder="Paste a sentence or paragraph here...",
                    lines=8,
                )
                text_type_in = gr.Radio(
                    choices=["sentence", "paragraph"],
                    value="sentence",
                    label="Text type",
                    info="Must match an available ScanDL2 checkpoint.",
                )
                with gr.Row():
                    run_btn = gr.Button("Run", variant="primary")
                    clear_btn = gr.Button("Clear")

            with gr.Column(scale=4):
                table_out = gr.Markdown(
                    label="Predicted scanpath",
                    value="_Results will appear here._",
                )
                json_out = gr.Code(
                    label="Raw JSON output",
                    language="json",
                    value="{}",
                )

        gr.Examples(examples=EXAMPLES, inputs=[text_in, text_type_in])

        def _run(text, text_type):
            output = predict(text, text_type)
            return format_output(output), format_json(output)

        run_btn.click(
            fn=_run,
            inputs=[text_in, text_type_in],
            outputs=[table_out, json_out],
        )
        clear_btn.click(
            fn=lambda: ("", "sentence", "_Results will appear here._", "{}"),
            inputs=None,
            outputs=[text_in, text_type_in, table_out, json_out],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue(max_size=16).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
    )