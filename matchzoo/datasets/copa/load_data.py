"""COPA(SuperGLUE Version) data loader."""

import typing
from pathlib import Path

import keras
import pandas as pd
import json
import matchzoo

_url = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/COPA.zip"

def load_data(
    stage: str = 'train',
    task: str = 'ranking',
    return_classes: bool = False
) -> typing.Union[matchzoo.DataPack, tuple]:
    """
    Load WikiQA data.

    :param stage: One of `train`, `val`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :param filtered: Whether remove the questions without correct answers.
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
        return data_pack
    elif isinstance(task, matchzoo.tasks.Classification):
        data_pack.one_hot_encode_label(task.num_classes, inplace=True)
        if return_classes:
            return data_pack, [False, True]
        else:
            return data_pack
    else:
        raise ValueError(f"{task} is not a valid task."
                         f"Must be one of `Ranking` and `Classification`.")


def _download_data():
    ref_path = keras.utils.data_utils.get_file(
        'COPA_data', _url, extract=True,
        cache_dir=matchzoo.USER_DATA_DIR,
        cache_subdir='COPA'
    )
    return Path(ref_path).parent.joinpath('COPA')


def _read_data(path):
    premise_question_list = []
    choice_list = []
    label_list = []
    idx_list = []
    choice_idx_list=[]
    for line in open(path, "r", encoding="utf-8"):
        line = json.loads(line)
        if line["question"] == "cause":
            question = "What was the cause of this?"
        else:
            question = "What happened as a result?"
        premise_question_list.append(line["premise"] + " " + question)
        premise_question_list.append(line["premise"] + " " + question)
        choice_list.append(line["choice1"])
        choice_list.append(line["choice2"])
        if int(line["label"])==0:
            label_list.append(1)
            label_list.append(0)
        else:
            label_list.append(0)
            label_list.append(1)
        idx_list.append(line["idx"])
        idx_list.append(line["idx"])

        choice_idx_list.append(str(line["idx"]) + "_0")
        choice_idx_list.append(str(line["idx"]) + "_1")

    df = pd.DataFrame({
        'text_left': premise_question_list,
        'text_right': choice_list,
        'id_left': idx_list,
        'id_right': choice_idx_list,
        'label': label_list
    })
    print(df)
    df = df.dropna(axis=0, how='any').reset_index(drop=True)
    return matchzoo.pack(df)

if __name__ == "__main__":
    _read_data("/Users/ksboy/Datasets/SuperGLUE/COPA/val.jsonl")