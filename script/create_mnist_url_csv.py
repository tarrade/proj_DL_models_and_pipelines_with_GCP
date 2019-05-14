import sys
import os
import pathlib
import random

workingdir=os.getcwd()
sys.path.insert(0, workingdir)

from subprocess import Popen, PIPE
import pandas as pd

#process = Popen(['gsutil', 'ls', 'gs://axa-ch-machine-learning-poc-dev/data/mnist/images_test'], stdout=PIPE, stderr=PIPE)
process = Popen(['gsutil', 'ls', 'gs://axa-ch-machine-learning-dev-vcm/mnist/images_test'], stdout=PIPE, stderr=PIPE)
stdout, stderr = process.communicate()

d = []
for i in stdout.decode().split('\n'):
    if i != '':
        name = i.split('_')
        #d.append({'url': i, 'type': name[2], 'label': int(name[5].split('.')[0])})
        d.append({'type': name[2].upper(), 'url': i, 'label': int(name[5].split('.')[0])})

#process = Popen(['gsutil', 'ls', 'gs://axa-ch-machine-learning-poc-dev/data/mnist/images_train'], stdout=PIPE,stderr=PIPE)
process = Popen(['gsutil', 'ls', 'gs://axa-ch-machine-learning-dev-vcm/mnist/images_train'], stdout=PIPE,stderr=PIPE)
stdout, stderr = process.communicate()

for i in stdout.decode().split('\n'):
    if i != '':
        rand=random.random()
        name = i.split('_')
        #d.append({'url': i, 'type': name[2], 'label': int(name[5].split('.')[0])})
        if rand<0.1:
            #print('validation', rand)
            d.append({'type': 'VALIDATION', 'url': i, 'label': int(name[5].split('.')[0])})
        d.append({'type': name[2].upper(), 'url': i, 'label': int(name[5].split('.')[0])})
#df = pd.DataFrame(d, columns=['url', 'type', 'label'])
df = pd.DataFrame(d, columns=['type', 'url', 'label'])

#df.to_csv('data/mnist/raw/mnist_url.csv', encoding='utf-8', index=False)
df.to_csv('data/mnist/raw/mnist_url.csv', encoding='utf-8', index=False, header=False)