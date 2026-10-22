"""Task 2.2 / 2.3 — model builders.

Every model takes float32 pixels in [0, 255] (the pipeline output) and does its own normalisation inside the
graph, so the saved .keras file is self-contained and serving only decodes + resizes.
"""

from __future__ import annotations

import keras
from keras import layers

# Backbone registry: constructor, in-graph preprocessing, and the first layer of its LAST block.
#  - EfficientNetB0 ships its own Rescaling + Normalization layers and expects raw [0, 255] input.
#  - MobileNetV2's preprocess_input maps [0, 255] → [-1, 1]; Rescaling(1/127.5, offset=-1) is the same
#    formula as a serialisable layer (a Lambda would not load in Keras' safe mode).
BACKBONES = {
    "efficientnetb0": {
        "ctor": keras.applications.EfficientNetB0,
        "preprocess": lambda: None,
        "last_block": "block7a_expand_conv",
    },
    "mobilenetv2": {
        "ctor": keras.applications.MobileNetV2,
        "preprocess": lambda: layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess"),
        "last_block": "block_16_expand",
    },
}


def build_cnn(img_size: int = 224, dropout: float = 0.3, base_filters: int = 32) -> keras.Model:
    """Small CNN from scratch: 4 conv blocks (conv → BN → ReLU → max-pool), GAP head, sigmoid output."""
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale_0_1")(inputs)
    for i, mult in enumerate((1, 2, 4, 8)):
        x = layers.Conv2D(base_filters * mult, 3, padding="same", use_bias=False, name=f"conv{i + 1}")(x)
        x = layers.BatchNormalization(name=f"bn{i + 1}")(x)
        x = layers.Activation("relu", name=f"relu{i + 1}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i + 1}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return keras.Model(inputs, outputs, name="cnn_baseline")


def build_transfer(backbone: str = "efficientnetb0", img_size: int = 224,
                   dropout: float = 0.3) -> tuple[keras.Model, keras.Model]:
    """Frozen ImageNet backbone (no classifier) + GAP → Dropout → Dense(1, sigmoid). Returns (model, base)."""
    spec = BACKBONES[backbone]
    base = spec["ctor"](include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
    base.trainable = False
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = inputs
    pre = spec["preprocess"]()
    if pre is not None:
        x = pre(x)
    # training=False keeps BatchNorm in inference mode even after unfreezing, so fine-tuning with a
    # small batch does not wreck the ImageNet running statistics.
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return keras.Model(inputs, outputs, name=f"transfer_{backbone}"), base


def unfreeze_top(base: keras.Model, backbone: str, n_layers: int | None = None) -> list[str]:
    """Stage 2: make only the top of the backbone trainable.

    n_layers=None → unfreeze the last block (from BACKBONES[...]['last_block'] to the end);
    n_layers=k   → unfreeze the last k layers (used by the tuner). BatchNorm layers always stay frozen.
    Returns the names of the trainable backbone layers.
    """
    base.trainable = True
    names = [layer.name for layer in base.layers]
    start = names.index(BACKBONES[backbone]["last_block"]) if n_layers is None else max(0, len(names) - n_layers)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= start and not isinstance(layer, layers.BatchNormalization)
    return [layer.name for layer in base.layers if layer.trainable and layer.weights]


def layer_table(model: keras.Model) -> list[dict]:
    """Per-layer output shape and parameter count (Task 2.2 evidence table)."""
    rows = []
    for layer in model.layers:
        out = layer.output
        shape = tuple(out.shape) if hasattr(out, "shape") else str(out)
        rows.append({"layer": layer.name, "type": type(layer).__name__, "output_shape": str(shape),
                     "params": layer.count_params(), "trainable": layer.trainable})
    return rows
