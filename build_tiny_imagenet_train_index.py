import json

# Read the imagenet class index to get the first 10 classes
with open('imagenet_class_index.json', 'r') as f:
    class_index = json.load(f)

# Get the class IDs for the first 10 classes (0-9)
target_class_ids = set()
for i in range(10):
    class_id = class_index[str(i)][0]  # Get the class ID (e.g., "n01440764")
    target_class_ids.add(class_id)

# Read the overall imagenet train index
with open('imagenet_train_index_overall.json', 'r') as f:
    train_index = json.load(f)

# Filter images that belong to the first 10 classes
tiny_train_index = []
for index in train_index:
    class_id = index['name'].split('/')[0].removesuffix('.tar')
    if class_id in target_class_ids:
        tiny_train_index.append(index)

# Write the filtered index to tiny_imagenet_train_index.json
with open('tiny_imagenet_train_index.json', 'w') as f:
    json.dump(tiny_train_index, f, indent=0)

print(f"Created tiny_imagenet_train_index.json with {len(tiny_train_index)} images from {len(target_class_ids)} classes")
