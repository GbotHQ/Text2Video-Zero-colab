import gradio as gr
import torch

from model import Model

from app_canny import create_demo as create_demo_canny


model = Model(device="cuda", dtype=torch.float16)

with gr.Blocks(css="style.css") as demo:
    create_demo_canny(model)

_, _, link = demo.launch(file_directories=["temporal"], debug=True, share=True)
print(link)
