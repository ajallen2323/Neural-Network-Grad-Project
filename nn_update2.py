import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import datasets
import transformers

# creat custom global sum pool that Sums all values across the time/sequence dimension
class GlobalSumPooling1D(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

# Tokenizer from baseline
tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 10

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN, padding="max_length")

# loading dataset
def load_datasets(train_path="train.csv", dev_path="dev.csv"):
    # train/dev split
    hf_dataset = datasets.load_dataset("csv", data_files={"train": train_path, "validation": dev_path})
    labels = hf_dataset["train"].column_names[1:]

    def gather_labels(example):
        return {"labels": [float(example[l]) for l in labels]}

    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns="input_ids",
        label_cols="labels",
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns="input_ids",
        label_cols="labels",
        batch_size=BATCH_SIZE
    )

    return train_dataset, dev_dataset, labels


def build_model(num_labels):
    # Adds L2 weight decay to reduce overfitting
    regularizer = tf.keras.regularizers.L2(0.0005)

    # inputs defines model input
    inputs = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")

    # Embedding converts token IDs into dense vectors
    x = tf.keras.layers.Embedding(
        input_dim=tokenizer.vocab_size,
        output_dim=128,
        input_length=MAX_LEN,
        mask_zero=True,
        embeddings_regularizer=regularizer
    )(inputs)

    # drops entire word embedding vectors to prevent overfitting
    x = tf.keras.layers.SpatialDropout1D(0.2)(x)

    # CNN extracts local n-gram features and downsamples
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Bidirectional Gru processes text forward and backward to capture context
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            64,
            dropout=0.3,
            recurrent_dropout=0.3,
            return_sequences=True
        )
    )(x)

    # Attention applies an attention mechanism that assigns importance weights to each token, 
    # reweights token representations accordingly, and aggregates them into a single sequence-level feature vector.
    score = tf.keras.layers.Dense(1)(x)
    weights = tf.keras.layers.Softmax(axis=1)(score)
    x = tf.keras.layers.Multiply()([x, weights])
    x = GlobalSumPooling1D()(x)

    # Dense head applies a fully connected layer to refine learned features and uses
    # dropout for regularization to reduce overfitting.
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # output one sigmoid neuron per label for multi-label classification
    outputs = tf.keras.layers.Dense(num_labels, activation="sigmoid")(x)

    # Build model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Uses the Adam optimizer with an exponentially decaying learning rate and 
    # gradient clipping to ensure stable and efficient training
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=2000,
            decay_rate=0.95
        ),
        clipnorm=1.0
    )

    # Compiles the model using the Adam optimizer, binary cross-entropy loss, 
    # and a micro-averaged F1 score metric with a fixed threshold.
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.F1Score(average="micro", threshold=0.4)]
    )

    return model

def train(model_path="model", train_path="train.csv", dev_path="dev.csv"):
    train_dataset, dev_dataset, labels = load_datasets(train_path, dev_path)
    model = build_model(len(labels))

    model.fit(
        train_dataset,
        validation_data=dev_dataset,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint( #Saves the best model
                filepath=model_path + ".keras",
                monitor="val_f1",
                mode="max",
                save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping( #Stops training early if validation F1 stops improving
                monitor="val_f1",
                patience=3,
                mode="max",
                restore_best_weights=True
            )
        ]
    )

# close to the same as baseline just update to fit mine
def predict(model_path="model.keras", input_path="test-ref.csv"):
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"GlobalSumPooling1D": GlobalSumPooling1D})
    df = pd.read_csv(input_path)
    labels = df.columns[1:]

    enc = tokenizer(df["text"].tolist(), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="tf")
    tf_dataset = tf.data.Dataset.from_tensor_slices(enc["input_ids"]).batch(BATCH_SIZE)

    preds = model.predict(tf_dataset)
    preds = (preds > 0.4).astype(int)  # Converts predicted probabilities into binary class labels using a 0.4 decision threshold

    df.iloc[:, 1:] = preds
    df.to_csv("submission.zip", index=False, compression=dict(
        method='zip', archive_name='submission.csv'
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict"})
    args = parser.parse_args()
    globals()[args.command]()
