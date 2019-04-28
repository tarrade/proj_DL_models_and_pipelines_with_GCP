import sys
import os
import pathlib

workingdir=os.getcwd()
sys.path.insert(0, workingdir)

from subprocess import Popen, PIPE
import pandas as pd

process = Popen(['gsutil', 'ls', 'gs://axa-ch-machine-learning-poc-dev/data/mnist/images_test'], stdout=PIPE,
                stderr=PIPE)
stdout, stderr = process.communicate()

d = []
for i in stdout.decode().split('\n'):
    if i != '':
        name = i.split('_')
        d.append({'url': i, 'type': name[2], 'label': int(name[5].split('.')[0])})

process = Popen(['gsutil', 'ls', 'gs://axa-ch-machine-learning-poc-dev/data/mnist/images_train'], stdout=PIPE,
                stderr=PIPE)
stdout, stderr = process.communicate()

for i in stdout.decode().split('\n'):
    if i != '':
        name = i.split('_')
        d.append({'url': i, 'type': name[2], 'label': int(name[5].split('.')[0])})

df = pd.DataFrame(d, columns=['url', 'type', 'label'])

df.to_csv('data/mnist/raw/mnist_url.csv', encoding='utf-8', index=False)