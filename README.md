# Smart Ordering

[1] Abd al-Rahman al-Ktefane, Adel Kaboul, Ammar Abo Azan, Oday Mourad. Smart Ordering: Scalable and Efficient Person Tracker and Sorter 2020.
paper link: https://arxiv.org/abs/1911.09070

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

