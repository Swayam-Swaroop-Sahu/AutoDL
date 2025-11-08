"""
auto_model_selector.py
----------------------
Selects and constructs an appropriate deep learning model automatically
based on dataset type and number of classes.
"""

from explainDL.model_selection.tabular_models import build_mlp
from explainDL.model_selection.image_models import build_cnn
from explainDL.model_selection.text_models import build_text_lstm


def select_model(data_type: str, input_shape=None, num_classes=2):
    """
    Chooses and returns a model suited for the given dataset type.

    Parameters
    ----------
    data_type : str
        One of 'tabular', 'image', or 'text'
    input_shape : tuple, optional
        Shape of the model input
    num_classes : int
        Number of target classes

    Returns
    -------
    model : keras.Model
    model_name : str
    """

    # decide output configuration
    if num_classes == 2:
        # force 2 outputs to match categorical generators
        output_units = 2
        activation = 'softmax'
        loss = 'categorical_crossentropy'
    elif num_classes > 2:
        output_units = num_classes
        activation = 'softmax'
        loss = 'categorical_crossentropy'
    else:
        output_units = 1
        activation = 'sigmoid'
        loss = 'binary_crossentropy'

    if data_type == "tabular":
        model = build_mlp(input_shape, output_units=output_units, activation=activation, loss=loss)
        model_name = f"MLP (Tabular, {num_classes}-class)"
        return model, model_name

    elif data_type == "image":
        model = build_cnn(input_shape=input_shape, output_units=output_units, activation=activation, loss=loss)
        model_name = f"CNN (Image, {num_classes}-class)"
        return model, model_name

    elif data_type == "text":
        model = build_text_lstm(input_shape=input_shape, output_units=output_units, activation=activation, loss=loss)
        model_name = f"LSTM (Text, {num_classes}-class)"
        return model, model_name

    else:
        raise ValueError(f"Unsupported data type: {data_type}")
