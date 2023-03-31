import gradio as gr

from main import controlnet_video_to_video, resample_video


def canny_video_to_video():
    with gr.Blocks() as demo:
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
                        value=0.6,
                        step=0.1,
                    )
            with gr.Column():
                result = gr.Video(label="Generated Video").style(height="auto")

        inputs = [
            input_video,
            prompt,
            negative_prompt,
            chunk_size,
            controlnet_conditioning_scale,
        ]

        run_button.click(
            fn=controlnet_video_to_video,
            inputs=inputs,
            outputs=result,
        )
    return demo


def resampling():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                input_video = gr.Video(
                    label="Input Video", source="upload", format="mp4", visible=True
                ).style(height="auto")
            with gr.Column():
                resolution = gr.Number(
                    label="Resolution", minimum=256, maximum=1024, value=512, precision=0
                )
                fps = gr.Number(
                    label="FPS", minimum=0, maximum=30, value=4, precision=0
                )
                run_button = gr.Button(label="Run")
            with gr.Column():
                result = gr.Video(label="Generated Video").style(height="auto")

        inputs = [
            input_video,
            resolution,
            fps,
        ]

        run_button.click(
            fn=resample_video,
            inputs=inputs,
            outputs=result,
        )
    return demo


def main():
    with gr.Blocks() as demo:
        with gr.Tab("Canny Video to Video"):
            canny_video_to_video()
        with gr.Tab("Video FPS and Resolution"):
            resampling()


if __name__ == "__main__":
    _, _, link = main().launch(
        file_directories=["temporal"], debug=True, share=True
    )
    print(link)
