# Smart Ordering
A smart application to track multiple person and arrange them according to the time of arrival based on [efficientdet]: (https://github.com/google/automl/tree/master/efficientdet)

[1] Abd al-Rahman al-Ktefane, Adel Kaboul, Ammar Abo Azan, Oday Mourad. Smart Ordering: Scalable and Efficient Person Tracker and Sorter 2020.
[Paper Link]: (https://drive.google.com/file/d/1rZ4MeCHHzCDzIDaZSA_OVspeo8w7IczO/view?usp=sharing)

**Quick install dependencies: ```pip install -r requirements.txt```**

## 1. install dependencies

```pip install -r requirements.txt```

## 2. Export efficientDet0 SavedModel, frozen graph.

Run the following command line to export models:
ps: exclamation point "!" used for run command's in colab cell, if you run it on regular shell please remove it.

    !rm  -rf resources/savedmodeldir
    !rm  -rf resources/efficientdet-d0
    !tar -zxvf resources/efficientdet-d0.tar.gz
    !python model_inspect.py --runmode=saved_model --model_name=efficientdet-d0 \
      --ckpt_path=resources/efficientdet-d0 --saved_model_dir=resources/savedmodeldir

Then you will get:

 - saved model under `resources/savedmodeldir/`
 - frozen graph with name `resources/savedmodeldir/efficientdet-d0_frozen.pb`


## 3. Export Feature Extractor frozen graph.

    !rm  -rf resources/networks
    !tar -zxvf resources/deep_association.tar.gz

## 4. Run The Tracker.
    !python smart_ordering.py --tracker_model_name=deep_sort --image_size=512x512
Check `python smart_ordering.py -h` for an overview of available options.


## 5. Package Diagram.
```bash
├── resources
│   ├── deep_association.tar.gz
│   ├── deep sort.odt
│   ├── deep_sort.pdf
│   ├── efficientdet-d0.tar.gz
│   ├── efficientdet.pdf
│   ├── MOT_class_diagram.mdj
│   ├── Smart_Real_Time_Ordering_Paper.docx
│   ├── Smart_Real_Time_Ordering_Paper.pdf
│   └── sort.pdf
│
├── detectors
│   ├── detection.py
│   ├── detector.py
│   └── eff_det_0.py
│
├── trackers
│   ├── deep_tracker.py
│   ├── kalman_filter.py
│   ├── sort_tracker.py
│   ├── tracker.py
│   └── track.py
│
├── utils
│   ├── features_util.py
│   ├── hyper_params.py
│   ├── nn_matching.py
│   ├── overlay_util.py
│   └── util.py
│
├── smart_ordering.py
└── model_inspect.py


```
## 6. Highlevel overview of source files
In the top-level directory are executable scripts to execute, evaluate, and
visualize the tracker. The main entry point is in `smart_ordering.py`.
This file runs the program with front camera of laptop.
In package `detectors` is the main detecting code:
* `detector.py`: Detector base class that represent a blueprint for other detectors
   models and should adopt it.
* `detection.py`: Detection base class.
* `eff_det_0.py`: child of `detector.py` and our offical tracker.

In package `trackers` is the main tracking code:
* `tracker.py`: Tracker base class that represent a blueprint for other trackers
   models and should adopt it.
* `track.py`: The track class contains single-target track data such as Kalman
  state, number of hits, misses, hit streak, associated feature vectors, etc.
* `kalman_filter.py`: A Kalman filter implementation and concrete
   parametrization for image space filtering.
* `sort_tracker.py`: implementation of SORT algorithm for tracking.
* `deep_tracker.py`: implementation of Deep_SORT algorithm for tracking.

In package `utils` is the main helper tool's code:
* `features_util.py`: This module contains code helps  in build and run feature extractor model
   that used in deep_SORT algorithm.
* `hyper_params.py`: This module contains code for default parameter value.
* `nn_matching.py`: A module for a nearest neighbor matching metric.
* `overlay_util.py`: A module contains a low level drawing functions for drawing
   overlay bounding boxes over orignal image.
* `util.py`: This module contains helper code for min cost matching problem solving and
   the matching cascade algorithm and linear algebra operations.
