# Copyright (c) OpenMMLab. All rights reserved.
from .inference import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from .test import multi_gpu_test, single_gpu_test
from .train import (get_root_logger, init_random_seed, set_random_seed,
                    train_detector)

from .test import single_gpu_test_detection, single_gpu_test_reid, multi_gpu_test_detection, multi_gpu_test_reid

__all__ = [
    'get_root_logger', 'set_random_seed', 'train_detector', 'init_detector',
    'async_inference_detector', 'inference_detector', 'show_result_pyplot',
    'multi_gpu_test', 'single_gpu_test', 'init_random_seed',
    'single_gpu_test_detection', 'single_gpu_test_reid',
    'multi_gpu_test_detection', 'multi_gpu_test_reid'
]
