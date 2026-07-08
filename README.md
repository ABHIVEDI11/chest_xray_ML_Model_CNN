# Chest X-Ray Pneumonia Classification — VGG19 Transfer Learning vs. Custom CNN

A binary image classifier that reads a chest X-ray and predicts **NORMAL** or
**PNEUMONIA**, built and evaluated two ways so you can see *why* the choice of
architecture matters on a small medical-imaging dataset:

1. **VGG19 transfer learning** — a pretrained ImageNet backbone with a small custom head
2. **A custom CNN** — the same problem, solved with a network trained from scratch

Both models are trained on the exact same 200 images and evaluated on the exact same
45-image test set, so the comparison is apples-to-apples. This README walks through
the full workflow end-to-end — data → model → training → evaluation → interpretability
— with the actual code and the actual results, so it doubles as a working example of
how to structure and evaluate an image classification project properly.

---

## TL;DR Results

| Model | Test Accuracy | Precision (pneumonia) | Recall (pneumonia) | F1 (pneumonia) | ROC-AUC |
|---|---|---|---|---|---|
| **VGG19 (transfer learning)** | **96.7%** (87/90) | 1.00 | 0.93 | 0.97 | **0.988** |
| Custom CNN (from scratch) | 85.6% (77/90) | 0.86 | 0.84 | 0.85 | not computed¹ |

¹ The notebook only builds an ROC curve for the VGG19 model (see [Section 5](#5-roc-curve--auc)).
To add one for the custom CNN, reuse the same `roc_curve`/`auc` cell with `Y_pred_c` in
place of `Y_pred`.

**Takeaway:** transfer learning wins decisively here — 11 percentage points higher
accuracy, and it never confuses a NORMAL scan for PNEUMONIA (0 false positives),
whereas the from-scratch CNN misreads 13 of the 90 test images. This is the expected
outcome on a 200-image training set: VGG19 starts from features learned on ~1.2M
ImageNet images and only has to fine-tune a small head, while the custom CNN has to
learn every filter — edges, textures, shapes — from 100 images per class, which just
isn't enough data to do that reliably. See [Results & Interpretation](#results--interpretation)
below for the full breakdown.

---

## Project Structure

```
.
├── notebooks/
│   └── VGG19_and_Custom_Model_Training.ipynb   # main notebook — run this top to bottom
├── models/            # saved .keras model files (gitignored — see below)
├── outputs/           # generated plots (confusion matrix, ROC curve, Grad-CAM, etc.)
├── data/              # dataset goes here locally (gitignored — not committed)
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

Chest X-ray images, 2 classes (`normal`, `pneumonia`), pre-split into:

```
data/chest_x_ray/
├── traindata/{normal,pneumonia}       # 100 / 100 images
├── validationdata/{normal,pneumonia}  #  30 /  30 images
└── testdata/{normal,pneumonia}        #  45 /  45 images
```

<img src="outputs/class_balance.png" alt="Class balance across train/val/test splits">

Both classes are perfectly balanced in every split, which matters — with an imbalanced
dataset, accuracy alone would be misleading (a model could just always predict the
majority class and still score well). Balanced classes mean plain accuracy is a fair
metric here.

The dataset is **not** committed to this repo (image datasets don't belong in git —
see `.gitignore`). Download/unzip it locally into `data/` before running the notebook,
or mount it from Google Drive if running in Colab.

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place the dataset under `data/chest_x_ray/` (see structure above), then open
`notebooks/VGG19_and_Custom_Model_Training.ipynb` in Jupyter or Colab. The notebook
auto-detects whether it's running in Colab (and mounts Drive at
`/content/drive/MyDrive/Colab Notebooks/chest_x_ray`) or locally (and uses
`./data/chest_x_ray`), so no path edits should be needed either way.

---

## Workflow

The notebook runs top to bottom as twelve sections. Each one below shows the actual
code, what it's for, and — where the notebook produces a plot — the actual output.

### 1. Data generators

```python
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
)
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

traindata = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(224, 224), batch_size=16,
    class_mode="categorical", shuffle=True, seed=42,
)
testdata = val_test_datagen.flow_from_directory(
    TEST_DIR, target_size=(224, 224), batch_size=16,
    class_mode="categorical", shuffle=False,   # <-- required for correct evaluation
)
```

Three things here matter more than they look like they should:

- **`preprocessing_function=preprocess_input`** — VGG19 expects ImageNet-style
  mean-subtracted, BGR-ordered input, not raw 0–255 pixel values. Skip this and the
  model trains on data it's not calibrated for, which quietly tanks accuracy.
- **Augmentation is applied to the training set only.** With 100 images per class,
  small random rotations/shifts/flips act as a cheap way to show the model more
  "versions" of the same 100 X-rays and reduce overfitting. Validation and test data
  are never augmented — you want to evaluate on realistic, unmodified images.
- **`shuffle=False` on the test generator.** This one is easy to get wrong and hard to
  notice when you do: if the test generator shuffles, `testdata.classes` (the true
  labels in on-disk order) no longer lines up with the order `predict()` returns
  predictions in. Every downstream metric — confusion matrix, classification report,
  ROC curve — would silently be scored against the wrong labels.

<img src="outputs/sample_images.png" alt="Sample training images per class">

### 2. VGG19 transfer-learning model

```python
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False          # freeze the pretrained convolutional base

x = base_model.output
x = GlobalAveragePooling2D()(x)      # far fewer params than Flatten
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
predictions = Dense(2, activation="softmax")(x)

model_final = Model(inputs=base_model.input, outputs=predictions)
```

`include_top=False` drops VGG19's original 1000-class ImageNet head so a custom head
can be attached instead. Freezing every layer in the base means training only updates
the new head — the ~20M parameters of pretrained convolutional filters (edge
detectors, texture detectors, shape detectors) stay exactly as ImageNet trained them.
`GlobalAveragePooling2D` instead of `Flatten` keeps the head small: `Flatten` here
would create tens of millions of extra parameters, which would overfit almost
immediately on only 200 training images.

### 3. Compile & train (VGG19)

```python
optimizer = optimizers.Adam(learning_rate=1e-4)   # low LR — only the head is training
model_final.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("models/vgg19_best.keras", monitor="val_accuracy", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
]

history = model_final.fit(traindata, epochs=30, validation_data=validationdata, callbacks=callbacks)
```

Three callbacks doing three different jobs: `EarlyStopping` halts training once
validation loss stops improving for 5 epochs and restores the best-performing weights
(protects against overfitting past the point of diminishing returns);
`ModelCheckpoint` separately saves the best validation-accuracy checkpoint to disk;
`ReduceLROnPlateau` shrinks the learning rate when validation loss plateaus, letting
the optimizer take smaller, more careful steps as it converges.

<img src="outputs/vgg19_training_curves.png" alt="VGG19 training and validation accuracy/loss curves">

**Reading this chart:** validation accuracy jumps above 90% within the first few
epochs and stays there — this is the transfer-learning effect in action, since the
head only has to learn to *re-combine* already-useful ImageNet features, not discover
them from scratch. Validation loss drops sharply and flattens out at a low value.
Training stopped early once validation loss stopped improving, well before the 30-epoch
cap, which is exactly what `EarlyStopping` is there to catch.

### 4. Evaluate on the test set

```python
Y_pred = model_final.predict(testdata)
y_pred = np.argmax(Y_pred, axis=1)
y_true = testdata.classes                       # correct only because shuffle=False

print(classification_report(y_true, y_pred, target_names=target_names))

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=target_names).plot(cmap="Blues")
```

<img src="outputs/vgg19_confusion_matrix.png" alt="VGG19 confusion matrix">

**Reading a confusion matrix:** rows are the true label, columns are what the model
predicted. A perfect model would have numbers only on the diagonal (top-left to
bottom-right) and zeros everywhere else. Here:

- **45/45 NORMAL scans correctly identified as NORMAL** — zero false positives. The
  model never raised a false pneumonia alarm on a healthy scan.
- **42/45 PNEUMONIA scans correctly identified as PNEUMONIA**, with **3 pneumonia
  cases misread as NORMAL** (false negatives).
- Overall: **87/90 correct = 96.7% test accuracy.**

In a clinical screening context, false negatives (missed pneumonia) are the more
costly error type — a missed diagnosis delays treatment, whereas a false positive
just triggers a follow-up look. This model's error profile (0 false positives, 3
false negatives) is the safer direction of the two to be biased toward, though 3
missed cases out of 45 is still a real gap you'd want to close before this went
anywhere near an actual clinical setting — see the [Disclaimer](#disclaimer).

### 5. ROC curve & AUC

```python
fpr, tpr, _ = roc_curve(y_true, Y_pred[:, 1])   # class index 1 = PNEUMONIA
roc_auc = auc(fpr, tpr)
```

<img src="outputs/vgg19_roc_curve.png" alt="VGG19 ROC curve, AUC = 0.988">

The confusion matrix above reflects one specific decision threshold (0.5). The ROC
curve shows what happens as that threshold slides — trading off the true positive
rate (catching pneumonia) against the false positive rate (raising false alarms) at
every possible threshold. A curve that hugs the top-left corner is what you want; the
diagonal dashed line represents a model no better than a coin flip. **AUC = 0.988**
means that if you picked one random NORMAL and one random PNEUMONIA image, the model
would rank the pneumonia scan as more suspicious 98.8% of the time — a strong result,
though on a 90-image test set even a couple of flipped predictions move this number
noticeably, so treat it as indicative rather than precise.

### 6. Misclassified examples

```python
wrong_idx = np.where(y_pred != y_true)[0][:8]
```

<img src="outputs/vgg19_misclassified.png" alt="VGG19 misclassified test images with true vs predicted labels">

Looking at the actual images the model got wrong is one of the most useful diagnostic
steps in medical imaging — it tells you whether the errors are random noise or a
systematic blind spot (e.g. the model consistently missing early-stage or
low-opacity cases, which would be a meaningful finding, not just statistical noise).

### 7. Custom CNN — the from-scratch baseline

```python
custom_model = Sequential([
    Conv2D(16, (3,3), activation="relu", padding="same", input_shape=(224,224,3)),
    Conv2D(16, (3,3), activation="relu", padding="same"),
    MaxPooling2D(2,2),

    Conv2D(32, (3,3), activation="relu", padding="same"),
    Conv2D(32, (3,3), activation="relu", padding="same"),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu", padding="same"),
    Conv2D(64, (3,3), activation="relu", padding="same"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu", padding="same"),
    Conv2D(128, (3,3), activation="relu", padding="same"),
    MaxPooling2D(2,2),

    GlobalAveragePooling2D(),
    Dense(512, activation="relu"),
    Dropout(0.4),
    Dense(2, activation="softmax"),
])
```

Four conv blocks (16 → 32 → 64 → 128 filters, two conv layers per block), trained
entirely from random initialization — no ImageNet weights, no frozen layers. Every
filter in this network has to be learned from the same 100 images per class the VGG19
head was fine-tuned on, which is a much harder ask. It's included specifically as a
baseline: the gap between this model and VGG19 *is* the demonstration of why transfer
learning helps on small datasets.

```python
custom_model.compile(loss="categorical_crossentropy",
                      optimizer=optimizers.Adam(learning_rate=1e-3), metrics=["accuracy"])
custom_history = custom_model.fit(traindata, epochs=30,
                                   validation_data=validationdata, callbacks=custom_callbacks)
```

<img src="outputs/custom_cnn_confusion_matrix.png" alt="Custom CNN confusion matrix">

**Reading this one against the VGG19 matrix above:** 39/45 NORMAL correct (6 false
positives) and 38/45 PNEUMONIA correct (7 false negatives) — **77/90 = 85.6%
accuracy**, with errors spread across both classes instead of concentrated in one
direction. That "errors in both directions" pattern is consistent with a model that
hasn't converged on a clean decision boundary yet, rather than one with a specific,
identifiable blind spot.

### 8. Model comparison

<img src="outputs/model_comparison.png" alt="VGG19 vs custom CNN validation accuracy and loss comparison">

Plotting both models' validation curves side by side makes the transfer-learning
advantage concrete rather than just a number in a table: VGG19 (blue) reaches high
validation accuracy within the first handful of epochs and stays stable there. The
custom CNN (orange) takes longer to improve, oscillates more between epochs — a sign
of a noisier, less certain optimization landscape when learning from scratch on
limited data — and converges to both a lower accuracy and a higher, less stable
validation loss.

### 9. Grad-CAM — interpretability check

```python
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(model.inputs,
                    [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.maximum(tf.squeeze(heatmap), 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()
```

<img src="outputs/gradcam_example.png" alt="Grad-CAM heatmap overlay on a chest X-ray">

A high test accuracy doesn't by itself prove the model is looking at the right thing —
it could, in principle, be keying off an artifact like a corner marker or border
rather than actual lung tissue. Grad-CAM answers that by backpropagating gradients
from the predicted class back to the last convolutional layer, producing a heatmap of
which pixels most influenced the decision. Warm (red/yellow) regions over the lung
fields, as seen here, are a basic sanity check that the model's attention is where a
radiologist's would be.

### 10. Save models

```python
model_final.save("models/vgg19_final.keras")
custom_model.save("models/custom_cnn_final.keras")
```

`ModelCheckpoint` already saved the best-epoch weights during training
(`models/vgg19_best.keras`, `models/custom_cnn_best.keras`); this saves the final
in-memory models explicitly on top of that.

---

## Results & Interpretation

Putting the two models' numbers side by side:

| Metric | VGG19 (transfer learning) | Custom CNN (from scratch) |
|---|---|---|
| Test accuracy | **96.7%** (87/90) | 85.6% (77/90) |
| False positives (NORMAL → PNEUMONIA) | 0 | 6 |
| False negatives (PNEUMONIA → NORMAL) | 3 | 7 |
| Pneumonia precision | 1.00 | 0.86 |
| Pneumonia recall | 0.93 | 0.84 |
| Pneumonia F1 | 0.97 | 0.85 |
| ROC-AUC | 0.988 | not computed¹ |

**Why the gap exists:** with only 100 training images per class, a network has to
learn everything about what an X-ray "looks like" — edges, textures, bone vs. tissue
contrast, lung field boundaries — from a dataset that's tiny by deep-learning
standards. VGG19 sidesteps almost all of that: its convolutional filters were already
trained on ~1.2M general images, so the 200-image training set here only has to teach
the model how to *recombine* existing visual features into a NORMAL/PNEUMONIA
decision, which is a dramatically easier learning problem. The custom CNN has no such
head start, and the result — lower accuracy, errors spread across both classes,
noisier training curves — is the textbook signature of a model that's data-starved
for the capacity it has.

**What this project demonstrates**, if you're using it as a learning reference:
- How to correctly set up data pipelines for a pretrained model (`preprocess_input`,
  `shuffle=False` on eval data, train-only augmentation)
- How to attach and train a custom head on a frozen pretrained backbone
- How to build a fair from-scratch baseline to actually justify the transfer-learning
  claim, instead of asserting it
- A complete, correct evaluation toolkit for a binary classifier: confusion matrix,
  classification report, ROC/AUC, misclassified-example inspection, and Grad-CAM for
  interpretability

---

## Requirements

See `requirements.txt`. Core: `tensorflow`, `scikit-learn`, `matplotlib`, `numpy`.

## Disclaimer

This is an educational/portfolio project, not a validated diagnostic tool. It's
trained on a very small dataset (200 training images, 90 test images), the reported
metrics come from a single train/test split rather than cross-validation, and it has
not been validated against clinical ground truth or a held-out hospital population.
It should not be used for actual medical decision-making.

## License

MIT (or your choice — add a LICENSE file).
