#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import pandas as pd
import numpy as np
import matchzoo as mz
import json

print('matchzoo version', mz.__version__)

print('data loading ...')
train_pack_raw = mz.datasets.copa.load_data('train', task='ranking')
dev_pack_raw = mz.datasets.copa.load_data('dev', task='ranking')
# test_pack_raw = mz.datasets.copa.load_data('test', task='ranking')
print('data loaded as `train_pack_raw` `dev_pack_raw` `test_pack_raw`')


print("loading embedding ...")
glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)
print("embedding loaded as `glove_embedding`")



preprocessor = mz.preprocessors.DSSMPreprocessor()
train_pack_processed = preprocessor.fit_transform(train_pack_raw)
valid_pack_processed = preprocessor.transform(dev_pack_raw)
# test_pack_processed = preprocessor.transform(test_pack_raw)


preprocessor.context


ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=4))
# ranking_task.metrics = [
#     mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
#     mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
#     mz.metrics.MeanAveragePrecision()
# ]


model = mz.models.DSSM()
model.params['input_shapes'] = preprocessor.context['input_shapes']
model.params['task'] = ranking_task
model.params['mlp_num_layers'] = 3
model.params['mlp_num_units'] = 300
model.params['mlp_num_fan_out'] = 128
model.params['mlp_activation_func'] = 'relu'
model.guess_and_fill_missing_params()
model.build()
model.compile()
model.backend.summary()


# In[8]:


pred_x, pred_y = valid_pack_processed[:].unpack()
# print(pred_x,pred_y)
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=pred_x, y=pred_y, batch_size=len(pred_x))


train_generator = mz.PairDataGenerator(train_pack_processed, num_dup=1, num_neg=4, batch_size=32, shuffle=True)
len(train_generator)



history = model.fit_generator(train_generator, epochs=100, callbacks=[evaluate], workers=5, use_multiprocessing=False)




