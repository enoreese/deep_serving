import tensorflow as tf

model_path = "path to saved model"
export_path = "serving/"

new_model = tf.keras.models.load_model(model_path)

# Check its architecture
new_model.summary()

tf.saved_model.save(new_model, export_path)
