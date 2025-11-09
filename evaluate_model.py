import argparse, os, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

p = argparse.ArgumentParser()
p.add_argument("--model_path", required=True)
p.add_argument("--test_dir", required=True)
p.add_argument("--img_size", type=int, default=224)
p.add_argument("--batch", type=int, default=32)
args = p.parse_args()

model = load_model(args.model_path)
gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    args.test_dir,
    target_size=(args.img_size, args.img_size),
    batch_size=args.batch,
    class_mode="categorical",
    shuffle=False,
)

preds = model.predict(gen, steps=np.ceil(gen.samples / args.batch))
y_pred = preds.argmax(axis=1)
y_true = gen.classes
labels = list(gen.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=labels, digits=4))
