# src/explainability/gradcam_explainer.py
"""
Grad-CAM explainability for CNN-based image models.

Function:
    generate_gradcam(model, img_array, layer_name=None, colormap=plt.cm.jet) -> matplotlib.figure.Figure

Notes:
- img_array should be shaped (1, H, W, 3) and scaled to [0,1] (or original pixel range).
- The function selects the last Conv2D layer by default if layer_name is None.
- Returns a matplotlib Figure containing the overlay image.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model

def _find_last_conv_layer(model):
    """
    Return name of last Conv2D layer in the model or None if none exists.
    """
    for layer in reversed(model.layers):
        # Use keras.layers.Conv2D class check
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def generate_gradcam(model, img_array, layer_name=None, colormap=plt.cm.jet):
    """
    Generate Grad-CAM overlay for a single image.

    Parameters
    ----------
    model : keras.Model
        A Keras model with convolutional layers.
    img_array : np.ndarray
        Input image as a numpy array with shape (1, H, W, 3).
    layer_name : str|None
        Name of conv layer to use. If None, pick last conv layer.
    colormap : matplotlib colormap

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    # Validate input
    if not hasattr(model, "layers"):
        raise ValueError("Provided model is not a Keras model.")

    if img_array.ndim != 4 or img_array.shape[0] != 1:
        raise ValueError("img_array must be shape (1, H, W, 3).")

    if layer_name is None:
        layer_name = _find_last_conv_layer(model)

    if layer_name is None:
        raise ValueError("No Conv2D layer found in model for Grad-CAM. Provide a conv layer name.")

    # Build a model maps input -> (conv_outputs, model_output)
    conv_layer = model.get_layer(layer_name)
    grad_model = Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])

    # Compute gradient of the predicted class w.r.t. conv output
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # predicted class index
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradients of class_channel wrt conv_outputs
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError("GradientTape returned None. Check model or input.")

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output_np = conv_outputs[0].numpy()
    pooled_grads_np = pooled_grads.numpy()

    # Weight conv feature maps by gradients
    for i in range(pooled_grads_np.shape[-1]):
        conv_output_np[:, :, i] *= pooled_grads_np[i]

    # Create heatmap
    heatmap = np.mean(conv_output_np, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap) if np.max(heatmap) != 0 else 1e-8
    heatmap /= max_val

    # Resize heatmap to image size
    try:
        import cv2
    except Exception:
        # Use numpy-based simple resize fallback (nearest neighbor)
        target_h, target_w = img_array.shape[1], img_array.shape[2]
        heatmap_resized = np.array(
            np.kron(heatmap, np.ones((int(np.ceil(target_h / heatmap.shape[0])), int(np.ceil(target_w / heatmap.shape[1])))))
        )
        heatmap_resized = heatmap_resized[:target_h, :target_w]
    else:
        h, w = img_array.shape[1], img_array.shape[2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert heatmap to RGB using colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = colormap(heatmap_uint8)
    heatmap_color = heatmap_color[..., :3]  # drop alpha

    # Prepare image
    img = img_array[0]
    # If image outside [0,1], normalize for display
    img_disp = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Overlay
    overlay = 0.5 * heatmap_color + 0.5 * img_disp
    overlay = np.clip(overlay, 0, 1)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(overlay)
    ax.axis("off")
    ax.set_title(f"Grad-CAM ({layer_name})")
    plt.tight_layout()

    return fig
