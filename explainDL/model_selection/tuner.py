"""
tuner.py
---------
Handles automated hyperparameter tuning using Keras Tuner or Optuna.
"""

from keras_tuner import RandomSearch
from explainDL.model_selection.tabular_models import build_mlp

def tune_tabular_model(X_train, y_train, input_shape, num_classes=2, max_trials=10):
    """
    Example tuner for tabular data using KerasTuner RandomSearch.
    """

    def model_builder(hp):
        from tensorflow.keras import layers, models
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Dense(
            units=hp.Int('units1', min_value=64, max_value=256, step=64),
            activation='relu'))
        model.add(layers.Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))
        model.add(layers.Dense(
            units=hp.Int('units2', min_value=32, max_value=128, step=32),
            activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid'))

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    tuner = RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=max_trials,
        overwrite=True,
        project_name='ExplainDL_Tuning'
    )

    tuner.search(X_train, y_train, epochs=5, validation_split=0.2)
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model
