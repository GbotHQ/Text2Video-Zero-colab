import gradio as gr
import torch

from model import Model
from main import process_controlnet_canny


def create_demo(model: Model):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown("## Text and Canny-Edge Conditional Video Generation")

        with gr.Row():
            with gr.Column():
                input_video = gr.Video(
                    label="Input Video", source="upload", format="mp4", visible=True
                ).style(height="auto")
            with gr.Column():
                prompt = gr.Textbox(label="Prompt")
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                )
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    chunk_size = gr.Slider(
                        label="Chunk size", minimum=2, maximum=8, value=8, step=1
                    )
                    controlnet_conditioning_scale = gr.Slider(
                        label="ControlNet Strength",
                        minimum=0,
                        maximum=2,
                        value=1,
                        step=0.1,
                    )
            with gr.Column():
                result = gr.Video(label="Generated Video").style(height="auto")

        inputs = [
            model,
            input_video,
            prompt,
            negative_prompt,
            chunk_size,
            controlnet_conditioning_scale,
        ]

        run_button.click(
            fn=process_controlnet_canny,
            inputs=inputs,
            outputs=result,
        )
    return demo


if __name__ == "__main__":
    model = Model(device="cuda", dtype=torch.float16)

    _, _, link = create_demo(model).launch(
        file_directories=["temporal"], debug=True, share=True
    )
    print(link)
