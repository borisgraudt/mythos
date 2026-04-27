"""
app.py — Gradio demo for Mythos.

Deployed on HuggingFace Spaces:
    https://huggingface.co/spaces/bgraudt/mythos

To deploy:
    1. Create a Space at huggingface.co/new-space (SDK = Gradio)
    2. Clone it locally:  git clone https://huggingface.co/spaces/bgraudt/mythos mythos-space
    3. Copy this file + requirements.txt into the Space repo
    4. Add to the Space's requirements.txt:
         transformers>=4.46
         torch>=2.5
         gradio>=4.44
    5. git push → Space auto-builds

Run locally:
    pip install gradio transformers torch
    python app.py
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "bgraudt/mythos"

print(f"Loading {MODEL_ID}…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model.eval()
device = next(model.parameters()).device
print(f"Model ready on {device}.")


@torch.inference_mode()
def generate(prompt: str, max_new_tokens: int, temperature: float, top_p: float, seed: int):
    if not prompt.strip():
        return "Please enter a prompt."

    if seed > 0:
        torch.manual_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


DESCRIPTION = """
# 🏛️ Mythos

A decoder-only language model **built from scratch** — LLaMA-style architecture
(GQA + SwiGLU + RoPE + RMSNorm), implemented in pure PyTorch with no
`transformers` inheritance.

Source: [github.com/borisgraudt/mythos](https://github.com/borisgraudt/mythos)
"""

EXAMPLES = [
    ["The history of artificial intelligence begins with", 120, 0.8, 0.9, 0],
    ["A transformer is a neural network that", 120, 0.7, 0.9, 0],
    ["In the early 20th century, physicists discovered", 150, 0.8, 0.9, 0],
    ["The key insight about attention mechanisms is", 120, 0.7, 0.9, 0],
]


with gr.Blocks(title="Mythos", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Start typing…",
                lines=4,
            )
            with gr.Row():
                max_new = gr.Slider(16, 512, value=128, step=8, label="Max new tokens")
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
            with gr.Row():
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
            run = gr.Button("Generate", variant="primary")

        with gr.Column(scale=4):
            output = gr.Textbox(label="Output", lines=14, show_copy_button=True)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[prompt, max_new, temperature, top_p, seed],
        outputs=output,
        fn=generate,
        cache_examples=False,
    )

    run.click(generate, [prompt, max_new, temperature, top_p, seed], output)


if __name__ == "__main__":
    demo.launch()
