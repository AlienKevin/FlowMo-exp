import json

# Read the imagenet class index to get the first n classes
with open('imagenet_class_index.json', 'r') as f:
    class_index = json.load(f)

# # Get the class IDs for the first n classes
# target_class_ids = set()
# for i in range(100):
#     class_id = class_index[str(i)][0]  # Get the class ID (e.g., "n01440764")
#     target_class_ids.add(class_id)

# 120 dog breeds in imagenet
target_class_ids = ['n02110627', 'n02088094', 'n02116738', 'n02096051', 'n02093428', 'n02107908', 'n02096294', 'n02110806', 'n02088238', 'n02088364', 'n02093647', 'n02107683', 'n02089078', 'n02086646', 'n02088466', 'n02088632', 'n02106166', 'n02093754', 'n02090622', 'n02096585', 'n02106382', 'n02108089', 'n02112706', 'n02105251', 'n02101388', 'n02108422', 'n02096177', 'n02113186', 'n02099849', 'n02085620', 'n02112137', 'n02101556', 'n02102318', 'n02106030', 'n02099429', 'n02096437', 'n02115913', 'n02115641', 'n02107142', 'n02089973', 'n02100735', 'n02102040', 'n02108000', 'n02109961', 'n02099267', 'n02108915', 'n02106662', 'n02100236', 'n02097130', 'n02099601', 'n02101006', 'n02109047', 'n02111500', 'n02107574', 'n02105056', 'n02091244', 'n02100877', 'n02093991', 'n02102973', 'n02090721', 'n02091032', 'n02085782', 'n02112350', 'n02105412', 'n02093859', 'n02105505', 'n02104029', 'n02099712', 'n02095570', 'n02111129', 'n02098413', 'n02110063', 'n02105162', 'n02085936', 'n02113978', 'n02107312', 'n02113712', 'n02097047', 'n02111277', 'n02094114', 'n02091467', 'n02094258', 'n02105641', 'n02091635', 'n02086910', 'n02086079', 'n02113023', 'n02112018', 'n02110958', 'n02090379', 'n02087394', 'n02106550', 'n02109525', 'n02091831', 'n02111889', 'n02104365', 'n02097298', 'n02092002', 'n02095889', 'n02105855', 'n02086240', 'n02110185', 'n02097658', 'n02098105', 'n02093256', 'n02113799', 'n02097209', 'n02102480', 'n02108551', 'n02097474', 'n02113624', 'n02087046', 'n02100583', 'n02089867', 'n02092339', 'n02102177', 'n02098286', 'n02091134', 'n02095314', 'n02094433']

# Read the overall imagenet train index
with open('imagenet_train_index_overall.json', 'r') as f:
    train_index = json.load(f)

# Filter images that belong to the target classes
tiny_train_index = []
for index in train_index:
    class_id = index['name'].split('/')[0].removesuffix('.tar')
    if class_id in target_class_ids:
        tiny_train_index.append(index)

# Write the filtered index to tiny_imagenet_train_index.json
with open('tiny_imagenet_train_index.json', 'w') as f:
    json.dump(tiny_train_index, f, indent=0)

print(f"Created tiny_imagenet_train_index.json with {len(tiny_train_index)} images from {len(target_class_ids)} classes")
