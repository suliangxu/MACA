# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings

from mmcv import ConfigDict


def compat_cfg(cfg):
    """This function would modify some filed to keep the compatibility of
    config.

    For example, it will move some args which will be deprecated to the correct
    fields.
    """
    cfg = copy.deepcopy(cfg)
    cfg = compat_imgs_per_gpu(cfg)
    cfg = compat_loader_args(cfg)
    cfg = compat_runner_args(cfg)
    return cfg


def compat_runner_args(cfg):
    if 'runner' not in cfg:
        cfg.runner = ConfigDict({
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        })
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs
    return cfg


def compat_imgs_per_gpu(cfg):
    cfg = copy.deepcopy(cfg)
    if 'imgs_per_gpu' in cfg.data:
        warnings.warn('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                      'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            warnings.warn(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            warnings.warn('Automatically set "samples_per_gpu"="imgs_per_gpu"='
                          f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
    return cfg


def compat_loader_args(cfg):
    """Deprecated sample_per_gpu in cfg.data."""

    cfg = copy.deepcopy(cfg)
    if 'train_dataloader' not in cfg.data:
        cfg.data['train_dataloader'] = ConfigDict()
    if 'val_dataloader' not in cfg.data:
        cfg.data['val_dataloader'] = ConfigDict()
    if 'test_gallery_dataloader' not in cfg.data:
        cfg.data['test_gallery_dataloader'] = ConfigDict()
    if 'test_query_dataloader' not in cfg.data:
        cfg.data['test_query_dataloader'] = ConfigDict()

    # special process for train_dataloader
    if 'samples_per_gpu' in cfg.data:

        samples_per_gpu = cfg.data.pop('samples_per_gpu')
        assert 'samples_per_gpu' not in \
               cfg.data.train_dataloader, ('`samples_per_gpu` are set '
                                           'in `data` field and ` '
                                           'data.train_dataloader` '
                                           'at the same time. '
                                           'Please only set it in '
                                           '`data.train_dataloader`. ')
        cfg.data.train_dataloader['samples_per_gpu'] = samples_per_gpu

    if 'persistent_workers' in cfg.data:

        persistent_workers = cfg.data.pop('persistent_workers')
        assert 'persistent_workers' not in \
               cfg.data.train_dataloader, ('`persistent_workers` are set '
                                           'in `data` field and ` '
                                           'data.train_dataloader` '
                                           'at the same time. '
                                           'Please only set it in '
                                           '`data.train_dataloader`. ')
        cfg.data.train_dataloader['persistent_workers'] = persistent_workers

    if 'workers_per_gpu' in cfg.data:

        workers_per_gpu = cfg.data.pop('workers_per_gpu')
        cfg.data.train_dataloader['workers_per_gpu'] = workers_per_gpu
        cfg.data.val_dataloader['workers_per_gpu'] = workers_per_gpu
        cfg.data.test_gallery_dataloader['workers_per_gpu'] = workers_per_gpu
        cfg.data.test_query_dataloader['workers_per_gpu'] = workers_per_gpu

    # special process for val_dataloader
    if 'samples_per_gpu' in cfg.data.val:
        # keep default value of `sample_per_gpu` is 1
        assert 'samples_per_gpu' not in \
               cfg.data.val_dataloader, ('`samples_per_gpu` are set '
                                         'in `data.val` field and ` '
                                         'data.val_dataloader` at '
                                         'the same time. '
                                         'Please only set it in '
                                         '`data.val_dataloader`. ')
        cfg.data.val_dataloader['samples_per_gpu'] = \
            cfg.data.val.pop('samples_per_gpu')
    # special process for val_dataloader

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test_gallery, dict):
        if 'samples_per_gpu' in cfg.data.test_gallery:
            assert 'samples_per_gpu' not in \
                   cfg.data.test_gallery_dataloader, ('`samples_per_gpu` are set '
                                                      'in `data.test_gallery` field and ` '
                                                      'data.test_gallery_dataloader` '
                                                      'at the same time. '
                                                      'Please only set it in '
                                                      '`data.test_gallery_dataloader`. ')

            cfg.data.test_gallery_dataloader['samples_per_gpu'] = \
                cfg.data.test_gallery.pop('samples_per_gpu')

    elif isinstance(cfg.data.test_gallery, list):
        for ds_cfg in cfg.data.test_gallery:
            if 'samples_per_gpu' in ds_cfg:
                assert 'samples_per_gpu' not in \
                       cfg.data.test_gallery_dataloader, ('`samples_per_gpu` are set '
                                                          'in `data.test_gallery` field and ` '
                                                          'data.test_gallery_dataloader` at'
                                                          ' the same time. '
                                                          'Please only set it in '
                                                          '`data.test_gallery_dataloader`. ')
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test_gallery])
        cfg.data.test_gallery_dataloader['samples_per_gpu'] = samples_per_gpu

        # in case the test dataset is concatenated
        if isinstance(cfg.data.test_query, dict):
            if 'samples_per_gpu' in cfg.data.test_query:
                assert 'samples_per_gpu' not in \
                       cfg.data.test_query_dataloader, ('`samples_per_gpu` are set '
                                                        'in `data.test_query` field and ` '
                                                        'data.test_query_dataloader` '
                                                        'at the same time. '
                                                        'Please only set it in '
                                                        '`data.test_query_dataloader`. ')

                cfg.data.test_query_dataloader['samples_per_gpu'] = \
                    cfg.data.test_query.pop('samples_per_gpu')

        elif isinstance(cfg.data.test_query, list):
            for ds_cfg in cfg.data.test_query:
                if 'samples_per_gpu' in ds_cfg:
                    assert 'samples_per_gpu' not in \
                           cfg.data.test_query_dataloader, ('`samples_per_gpu` are set '
                                                            'in `data.test_query` field and ` '
                                                            'data.test_query_dataloader` at'
                                                            ' the same time. '
                                                            'Please only set it in '
                                                            '`data.test_query_dataloader`. ')
            samples_per_gpu = max(
                [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test_query])
            cfg.data.test_query_dataloader['samples_per_gpu'] = samples_per_gpu

    return cfg
