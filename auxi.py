import os

def count_images(directory):
    total = 0
    for root, _, files in os.walk(directory):
        total += len([f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
    return total


train_dir = 'C:/Users/User/pe_novo/data/train'
validation_dir = 'C:/Users/User/pe_novo/data/validation'

train_count = count_images(train_dir)
val_count = count_images(validation_dir)
print(f"Training images: {train_count}")
print(f"Validation images: {val_count}")
