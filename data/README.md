# Data

No data is stored in this repository. Download the datasets from their original sources and upload the zip files when the notebook asks for them (Step 2).

| Dataset | Format | Used for | Source |
|---|---|---|---|
| Road-line boxes | 50 photos + Pascal VOC `.xml` (class `road_line`, 172 boxes) | YOLOv8 marking detector | `[add source link]` |
| KITTI Road (224×224 copy) | 289 photos + colour masks (`image_2/`, `gt_image_2/`) plus an unlabelled `testing/` folder | U-Net road / lane-area segmenter | http://www.cvlibs.net/datasets/kitti/eval_road.php |

Expected content of the uploaded zips (names may differ; the notebook detects the type automatically):

```
road_line_images/…jpg|png            road_line_annotation/…xml
data_road_224/training/image_2/…png  data_road_224/training/gt_image_2/gt_…png
```

Respect each dataset's licence (KITTI is released for non-commercial use under CC BY-NC-SA 3.0).
