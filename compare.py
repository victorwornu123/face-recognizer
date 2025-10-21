from process import preprocess_image, cosine_similarity
import json
import numpy as np
import gradio as gr

def gradio_wrapper(image_path):
    # Load user database each time to ensure updates are read
    with open("user_data.json", "r") as file:
        data = json.load(file)

    # Get embedding for uploaded image
    name, new_array = preprocess_image(None, image_path)
    new_array = np.array(new_array, dtype=float).flatten()

    # Compare with all stored users
    for entry in data:
        stored_name = list(entry.keys())[0]
        stored_embedding = np.array(entry[stored_name], dtype=float).flatten()

        cos = cosine_similarity(new_array, stored_embedding)
        print(f"Comparing with {stored_name}: {cos:.4f}")

        if cos > 0.8:
            return f"✅ Welcome back, {stored_name}!"

    # If no match found
    return f"❌ User not recognized.{cos:.4f}"


# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("### Face Recognition Login")
    gr.Markdown("Upload a face image to verify identity.")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload Image")
            predict_button = gr.Button("Recognize Face")
        with gr.Column():
            output_label = gr.Label(label="Result")

    predict_button.click(fn=gradio_wrapper, inputs=image_input, outputs=output_label)

demo.launch(share=True)


# print(np.array(list(data[0].keys())[0])) # get username
# print(np.array(list(data[0].values())[0][0])) # get user array
