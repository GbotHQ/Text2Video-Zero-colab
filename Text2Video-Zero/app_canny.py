import gradio as gr
from model import Model

def create_demo(model: Model):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Text and Canny-Edge Conditional Video Generation')
        with gr.Row():
            gr.HTML(
                """
                <div style="text-align: left; auto;">
                <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
                    Description: For performance purposes, our current preview release supports any input videos but caps output videos to no longer than 15 seconds and the input videos are scaled down before processing.
                </h3>
                </div>
                """)

        with gr.Row():
            with gr.Column():
                input_video = gr.Video(
                    label="Input Video", source='upload', format="mp4", visible=True).style(height="auto")
            with gr.Column():
                prompt = gr.Textbox(label='Prompt')
                run_button = gr.Button(label='Run')
                with gr.Accordion('Advanced options', open=False):
                    watermark = gr.Radio(["Picsart AI Research", "Text2Video-Zero", "None"], label="Watermark", value='Picsart AI Research')
                    chunk_size = gr.Slider(label="Chunk size", minimum=2, maximum=8, value=8, step=1)
            with gr.Column():
                result = gr.Video(label="Generated Video").style(height="auto")

        inputs = [
            input_video,
            prompt,
            chunk_size,
            watermark,
        ]

        run_button.click(fn=model.process_controlnet_canny,
                         inputs=inputs,
                         outputs=result,)
    return demo
