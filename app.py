import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json

def parse_leafsnap_metadata(metadata_path, dataset_root):
    """Parse metadata with proper path validation"""
    df = pd.read_csv(metadata_path, sep='\t')
    
    valid_records = []
    for _, row in df.iterrows():
        # Check both image paths
        img_path = os.path.join(dataset_root, row['image_path'])
        seg_path = os.path.join(dataset_root, row['segmented_path'])
        
        if os.path.exists(img_path):
            valid_records.append({'path': img_path, 'species': row['species']})
        elif os.path.exists(seg_path):
            valid_records.append({'path': seg_path, 'species': row['species']})
    
    return pd.DataFrame(valid_records)

def load_and_preprocess_image(path, label):
    """TensorFlow-native image processing without Python conditionals"""
    try:
        # Read and decode image
        img = tf.io.read_file(path)
        
        # Use TensorFlow string ops to check extension
        is_jpeg = tf.strings.regex_full_match(path, ".*\.jpe?g")
        image = tf.cond(
            is_jpeg,
            lambda: tf.image.decode_jpeg(img, channels=3),
            lambda: tf.image.decode_png(img, channels=3)
        )
        
        # Convert and preprocess
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_with_pad(image, 224, 224)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        
        return image, label
    except Exception as e:
        # You might log the exception here for debugging purposes
        return tf.zeros([224, 224, 3], tf.float32), -1

def create_dataset(df, batch_size=32, shuffle=True):
    """Create dataset with known cardinality"""
    # Create label mapping
    species = sorted(df['species'].unique())
    label_map = {s: i for i, s in enumerate(species)}
    df['label'] = df['species'].map(label_map).astype(np.int32)
    
    # 1. EXPLICITLY FILTER INVALID ENTRIES FIRST
    df = df[df['label'].notna()]
    
    # 2. CALCULATE DATASET LENGTH BEFORE CREATION
    num_samples = len(df)
    
    # Create dataset from numpy arrays
    ds = tf.data.Dataset.from_tensor_slices((df['path'].values, df['label'].values))
    
    if shuffle:
        ds = ds.shuffle(num_samples)
    
    # 3. USE KNOWN CARDINALITY
    ds = ds.apply(tf.data.experimental.assert_cardinality(num_samples))
    
    ds = ds.map(load_and_preprocess_image, 
                num_parallel_calls=tf.data.AUTOTUNE)
    
    # 4. BATCH WITH DROP_REMAINDER TO MAINTAIN KNOWN SIZE
    ds = ds.batch(batch_size, drop_remainder=True)
    
    # Prefetch for performance improvement
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds, num_samples // batch_size

def build_model(num_classes):
    """Create MobileNetV2 model with proper initialization"""
    base = MobileNetV2(input_shape=(224, 224, 3), 
                       include_top=False, 
                       weights='imagenet')
    base.trainable = False
    
    inputs = Input((224, 224, 3))
    x = base(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    # Configuration
    dataset_root = '/Users/drago/plant_detector/leafsnap-dataset'
    metadata_path = f"{dataset_root}/leafsnap-dataset-images.txt"
    
    try:
        # 1. Load data
        print("Loading metadata...")
        df = parse_leafsnap_metadata(metadata_path, dataset_root)
        print(f"Loaded {len(df)} valid images")
        
        # 2. Split data
        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=df['species'], random_state=42
        )
        
        # 3. Create datasets
        print("Creating datasets...")
        train_ds, train_steps = create_dataset(train_df, shuffle=True)
        val_ds, val_steps = create_dataset(val_df, shuffle=False)
        
        # Verify dataset sizes
        print(f"Training batches: {train_steps}")
        print(f"Validation batches: {val_steps}")
        
        # 4. Build and train model
        print("Building model...")
        model = build_model(len(df['species'].unique()))
        
        print("Training...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=15,
            callbacks=[
                EarlyStopping(patience=3, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.1, patience=2)
            ]
        )
        
        # 5. Save model (you can also save in SavedModel format if preferred)
        model.save("plant_model.h5")
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main()
