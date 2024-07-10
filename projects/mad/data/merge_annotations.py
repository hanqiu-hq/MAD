import json

det_data = json.load(open("./datasets/coco/annotations/instances_train2017.json", 'r'))
keypoint_data = json.load(open("./datasets/coco/annotations/person_keypoints_train2017.json", 'r'))

keypoint_anno = keypoint_data['annotations']

anns = dict()
for ann_i, ann in enumerate(det_data['annotations']):
    anns[ann['id']] = ann_i

for kpt_dict in keypoint_anno:
    if kpt_dict['id'] in anns:
        det_data['annotations'][anns[kpt_dict['id']]].update(kpt_dict)
    else:
        det_data['annotations'].append(kpt_dict)

det_data['categories'][0].update(keypoint_data['categories'][0])

json.dump(det_data, open("./datasets/coco/annotations/instances_train2017_keypoint.json", 'w'))
print("merge keypoint anno into instance_train2017.json and save in instances_train2017_keypoint.json")
