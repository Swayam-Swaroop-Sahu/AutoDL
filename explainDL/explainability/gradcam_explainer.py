"""
gradcam_explainer.py
--------------------
Generates Grad-CAM heatmaps for CNN-based image models.
Supports EfficientNet, MobileNet, and custom CNNs.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model

def generate_gradcam(model, img_array, layer_name=None, colormap=plt.cm.jet):
    """
    Generates a Grad-CAM heatmap for a given model and image.

    Parameters
    ----------
    model : keras.Model
        Trained CNN model.
    img_array : np.ndarray
        Input image array of shape (1, H, W, 3) normalized between [0, 1].
    layer_name : str or None
        Specific layer to visualize. If None, automatically picks last Conv2D layer.
    colormap : matplotlib colormap
        Colormap used for heatmap overlay.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure with Grad-CAM visualization.
    """

    # 1️⃣ Find last convolutional layer if not provided
    if layer_name is None:
        layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

    if layer_name is None:
        raise ValueError("No convolutional layer found in model for Grad-CAM.")

    conv_layer = model.get_layer(layer_name)

    # 2️⃣ Build gradient model: inputs -> (conv outputs, predictions)
    grad_model = Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])

    # 3️⃣ Compute gradients of top predicted class wrt conv outputs
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    # 4️⃣ Weighted average of conv maps
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8  # normalize

    # 5️⃣ Resize heatmap to match image
    import cv2
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = colormap(heatmap)
    heatmap = np.delete(heatmap, 3, axis=2)  # remove alpha

    # 6️⃣ Overlay on original image
    img = img_array[0]
    img = (img - img.min()) / (img.max() - img.min())
    overlay = 0.6 * heatmap[:, :, :3] + 0.4 * img

    # 7️⃣ Plot result
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(overlay)
    ax.axis('off')
    ax.set_title(f"Grad-CAM ({layer_name})")
    plt.tight_layout()

    return fig
