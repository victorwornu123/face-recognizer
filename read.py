from process import preprocess_image, cosine_similarity, build_facenet_model


n1, val1 = preprocess_image("Victor","WhatsApp Image 2025-10-15 at 16.23.15_891405be.jpg")
n2, val2 = preprocess_image("Victor","WhatsApp Image 2025-10-15 at 16.23.15_891405be.jpg")
# n2, val2 = preprocess_image("Angela", "WhatsApp Image 2025-10-15 at 16.23.15_0c7d36d1.jpg")

print(cosine_similarity(val1.flatten(),val2.flatten()))



model = build_facenet_model()
model.save("face_embedding_model.h5")
print("✅ Model saved successfully.")
