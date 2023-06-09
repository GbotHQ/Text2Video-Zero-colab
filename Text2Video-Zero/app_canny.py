import gradio as gr
from model import Model

def create_demo(model: Model):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Text and Canny-Edge Conditional Video Generation')

        with gr.Row():
            with gr.Column():
                input_video = gr.Video(
                    label="Input Video", source='upload', format="mp4", visible=True).style(height="auto")
            with gr.Column():
                prompt = gr.Textbox(label='Prompt')
                run_button = gr.Button(label='Run')
                with gr.Accordion('Advanced options', open=False):
                    chunk_size = gr.Slider(label="Chunk size", minimum=2, maximum=8, value=8, step=1)
            with gr.Column():
                result = gr.Video(label="Generated Video").style(height="auto")

        inputs = [
            input_video,
            prompt,
            chunk_size,
        ]

        run_button.click(fn=model.process_controlnet_canny,
                         inputs=inputs,
                         outputs=result,)
    return demo
