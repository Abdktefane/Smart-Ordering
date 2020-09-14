default_params = {
    "max_boxes_to_draw": 20,                                # the max tracking object
    "image_size": "512x512",                                # the size of input image
    "min_score_thresh": 0.6,                                # the minimum threshold for accepting detection
    "use_xla": False,                                       # use xla accelerator
    "line_thickness": 4,                                    # line thickness for draw bounding box
    "max_cosine_distance": 0.2,                             # threshold for cosine similarity
    "nn_budget": 100,                                       # size of tracks gallery
    "max_age": 10,                                          # age of tracks
    "nn_init": 3,                                           # tentative time of tracks
    "detector_model_name": "efficientdet-d0",               # detector model type
    "detector_saved_model_dir": "resources/savedmodeldir",            # path to detector model graph file
    "eff_det_0_model_path": "resources/efficientdet-d0",              # path to efficient det 0 model checkpoint
    "tracker_model_name": "sort",                           # tracker model type
    "feature_model_path": "resources/networks/mars-small128.pb",      # path to feature extractor model checkpoint
}
