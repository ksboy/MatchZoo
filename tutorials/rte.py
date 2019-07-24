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
train_pack_raw = mz.datasets.rte.load_data('train', task='classification')
dev_pack_raw = mz.datasets.rte.load_data('dev', task='classification')
# test_pack_raw = mz.datasets.cb.load_data('test', task='classification')
print('data loaded as `train_pack_raw` `dev_pack_raw` ')


classification_task = mz.tasks.Classification( num_classes =2)

print("loading embedding ...")
glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)
print("embedding loaded as `glove_embedding`")


preprocessor = mz.preprocessors.DSSMPreprocessor()
train_pack_processed = preprocessor.fit_transform(train_pack_raw)
valid_pack_processed = preprocessor.transform(dev_pack_raw)
# test_pack_processed = preprocessor.transform(test_pack_raw)


preprocessor.context



model = mz.models.DSSM()
model.params['input_shapes'] = preprocessor.context['input_shapes']
model.params['task'] = classification_task
model.params['mlp_num_layers'] = 3
model.params['mlp_num_units'] = 300
model.params['mlp_num_fan_out'] = 128
model.params['mlp_activation_func'] = 'relu'
model.guess_and_fill_missing_params()
model.build()
model.compile()
model.backend.summary()


# In[8]:
x, y = train_pack_processed[:].unpack()
pred_x, pred_y = valid_pack_processed[:].unpack()
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=pred_x, y=pred_y, batch_size=len(pred_x))


history = model.fit(x, y, epochs=50, callbacks=[evaluate])

