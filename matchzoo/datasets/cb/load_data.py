"""CB(SuperGLUE version) data loader."""

import typing
from pathlib import Path

import pandas as pd
import json
import keras

import matchzoo

_url = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/CB.zip"


def load_data(
    stage: str = 'train',
    task: str = 'classification',
    target_label: str = 'entailment',
    return_classes: bool = False
) -> typing.Union[matchzoo.DataPack, tuple]:
    """
    Load CB data.

    :param stage: One of `train`, `val`, and `test`. (default: `train`)
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance. (default: `ranking`)
    :param target_label: If `ranking`, chose one of `entailment`,
        `contradiction`, `neutral` as the positive label.
        (default: `entailment`)
    :param return_classes: `True` to return classes for classification task,
        `False` otherwise.

    :return: A DataPack unless `task` is `classificiation` and `return_classes`
        is `True`: a tuple of `(DataPack, classes)` in that case.
    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")
    if stage == 'dev':
        stage = 'val'
    data_root = _download_data()
    file_path = data_root.joinpath(f'{stage}.jsonl')
    data_pack = _read_data(file_path)

    if task == 'ranking':
        task = matchzoo.tasks.Ranking()
    if task == 'classification':
        task = matchzoo.tasks.Classification()

    if isinstance(task, matchzoo.tasks.Ranking):
        if target_label not in ['entailment', 'contradiction', 'neutral']:
            raise ValueError(f"{target_label} is not a valid target label."
                             f"Must be one of `entailment`, `contradiction`, "
                             f"`neutral`.")
        binary = (data_pack.relation['label'] == target_label).astype(float)
        data_pack.relation['label'] = binary
        return data_pack
    elif isinstance(task, matchzoo.tasks.Classification):
        classes = ['entailment', 'contradiction', 'neutral']
        label = data_pack.relation['label'].apply(classes.index)
        data_pack.relation['label'] = label
        data_pack.one_hot_encode_label(num_classes=3, inplace=True)
        if return_classes:
            return data_pack, classes
        else:
            return data_pack
    else:
        raise ValueError(f"{task} is not a valid task."
                         f"Must be one of `Ranking` and `Classification`.")


def _download_data():
    ref_path = keras.utils.data_utils.get_file(
        'CB_data', _url, extract=True,
        cache_dir=matchzoo.USER_DATA_DIR,
        cache_subdir='CB'
    )
    return Path(ref_path).parent.joinpath('CB')


def _read_data(path):
    premise_list =[]
    hypothesis_list=[]
    label_list=[]
    idx_list =[]
    for line in open(path, "r", encoding="utf-8"):
        line = json.loads(line)
        premise_list.append(line["premise"])
        hypothesis_list.append(line["hypothesis"])
        label_list.append(line["label"])
        idx_list.append(line["idx"])

    df = pd.DataFrame({
        'text_left': premise_list,
        'text_right': hypothesis_list,
        'label': label_list
    })
    print(df)
    df = df.dropna(axis=0, how='any').reset_index(drop=True)
    return matchzoo.pack(df)

if __name__ == "__main__":
    _read_data("/Users/ksboy/Datasets/SuperGLUE/CB/val.jsonl")
