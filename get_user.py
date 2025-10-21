import json
import numpy as np
from process import preprocess_image
import gradio as gr

def gradio_wrapper(username, image_path):
    # ✅ Ensure username is a string (not a component object)
    if not isinstance(username, str):
        try:
            username = username.value  # try to access .value if it's a gradio.Textbox
        except AttributeError:
            username = str(username)

    # ✅ Load existing data (or create empty list if file doesn't exist)
    try:
        with open("user_data.json", "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    # ✅ Your image preprocessing logic
    name, array = preprocess_image(username, image_path)

    # ✅ Add new data
    new_data = {str(username): array.tolist()}
    data.append(new_data)

    # ✅ Save to file
    with open("user_data.json", "w") as file:
        json.dump(data, file, indent=4)

    return f"✅ Successfully added data for user: {username}"


# if __name__ == "__main__":
with gr.Blocks() as demo:
    gr.Markdown("Image Array Extraction using FaceNet")
    gr.Markdown("Upload an face image.")
    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Name", placeholder="Enter your name")
            image_input = gr.Image(type="filepath", label="Upload Image")
            predict_button = gr.Button("Add")
        with gr.Column():
            output_label = gr.Label()
    predict_button.click(fn=gradio_wrapper, inputs=[name,image_input], outputs=output_label)
      # print(name,image_input)

demo.launch(share=True)


