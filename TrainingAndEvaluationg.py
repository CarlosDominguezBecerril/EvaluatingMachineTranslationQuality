# -*- coding: utf-8 -*-
####


# IMPORTANT: The original code is based on a Jupyter Notebook. This code might not properly display the output.


####
"""
<strong>Authors: </strong><a href="mailto:cdominguez019@ikasle.ehu.eus">Carlos Domínguez Becerril</a>, <a href="mailto:xzuazo002@ikasle.ehu.eus">Xabier de Zuazo Oteiza</a><br/>
<strong>Institution: </strong><a href="https://www.ehu.eus/">University of the Basque Country - UPV/EHU</a>
<strong>Paper:</strong> https://drive.google.com/file/d/1-k9fWSGnF6_mCxiQehW9GprciOYaWKO9/view?usp=share_link

The main task to predict Translation Error Rate (TER) using deep learning models based on transformers.

This notebook main goal of this notebook is to search for the best seeds for a BERT based regression model.

When using BERT for regression tasks, the resulting models suffer for some inestability and their learning speed and final results can vary depending on the following things:

* Seed use for randomization.
* Dataset distribution.
* Weights intialization algorithm.
* Others.

Related articles describing this phenomenon of instability:

* [What exactly happens when we fine-tune BERT?](https://towardsdatascience.com/what-exactly-happens-when-we-fine-tune-bert-f5dc32885d76)
* [What Happens To BERT Embeddings During Fine-tuning?](https://arxiv.org/abs/2004.14448)
* [To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks](https://arxiv.org/abs/1903.05987)

Until today, the proposed solutions for tackle the instability are the following:

* Use the [BertAdam](https://huggingface.co/docs/transformers/migration#optimizers-bertadam-openaiadam-are-now-adamw-schedules-are-standard-pytorch-schedules) optimizer.
* Fine-tune for 20 epochs.
* Try different seeds: seeds with good initial results have better results later*.

Related articles searching for solutions:

* [On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines](https://arxiv.org/abs/2006.04884)
* [Revisiting Few-sample BERT Fine-tuning](https://arxiv.org/abs/2006.05987)
* [Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping](https://arxiv.org/abs/2002.06305)*

Therefore, different set ups need to be tried. We called *Seed Germination* to the process of searching seeds by brute force, training them and selecting the best based on the learning process and final scores.

To summarize, in this notebook, many different setups can be tried using multiple seeds, getting the mean results, their standard deviation and final scores. Among others, the following parameters can be adjusted:

* Number of seeds to try, which seeds, ...
* The input to use: Spanish-Basque, only Basque, include post edit, ...
* Pre-trained model checkpoint to use for training.
* Epochs, learning rate, dropout, batch size, weights initialization algorithm, ...

To sum up the most common options:

* To try different **models**, change `checkpoint` configuration.
* To change the input, set `input_type` option to `basque
* For **+post edit** set `teacher_forcing_rate` to `0.5`.
* For **+features** set `extra_features` to `True`.

Go to the [Training Section](#Training) to do you set up.

**Note:** If you do not see the interactive plots, try to open it with Jupyter Notebook (the original platform used to do the training).

## Prepare the Environment (Colab, Kaggle, Jupyter)
"""

import os

try:
    from google.colab import drive
    drive.mount('/content/drive')
except:
    pass

if os.path.isdir('/content/drive'): # Google Drive
    save_dir = '/content/drive/MyDrive/WiP/models'
    platform = 'colab'
    corpus_folder = '/content/drive/My Drive/WiP/data/4th-lap-quality-estimation/'
elif os.path.isdir('/kaggle/working'): # Kaggle
    save_dir = '/kaggle/working'
    corpus_folder = '../input/4th-lap-quality-estimation/'
    platform = 'kaggle'
else: # Others
    save_dir = './'
    platform = 'notebook'
    corpus_folder = 'data/4th-lap-quality-estimation/'

"""## Install Requirements"""

!pip3 install -q transformers==4.18.0 plotly==5.7.0 pyyaml==5.4.1
!pip install -q -U kaleido sentencepiece

"""## Import Libraries"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import set_seed
from transformers import AdamW
from transformers import AutoTokenizer, AutoModel, get_scheduler
from transformers import logging as transformer_logging
import pandas as pd
from collections import defaultdict
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import math
import random
from scipy import stats
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import logging
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy import signal
from IPython.core.display import display, HTML
import plotly.io as pio

# Plotly set up for notebooks.
# Fix some problems when saving Plotly figures on notebooks and set default theme.
# For more information: https://plotly.com/python/renderers/
pio.renderers.default = platform
pio.templates.default = 'simple_white'

# Logging level:
# logging.basicConfig(level=logging.DEBUG) # uncomment this for more verbose output

# Silence warning: Some weights of the model checkpoint at ... were not used when initializing BertModel
transformer_logging.set_verbosity_error()

# Fixed random seed (optional, usually set below again through the config)
set_seed(0)

"""## Configuration Manager

This configuration object used by the classes above to manage some global configuration parameters used in diverse places. It comes with some sensitive default values.

Usage example:

```python
config = Config({'checkpoint': 'MarcBrun/ixambert-finetuned-squad', 'epochs': 5})

class MyClassExample(nn.Module):
    def __init__(self, config={}):
        self.config = config
        # [...]
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['checkpoint'])
        # [...]
        self.model = AutoModel.from_pretrained(self.config['checkpoint'])
        

model = AutoClassifier(config=config)
```
"""

class Config(dict):
    """Configuration manager."""
    def __init__(self, init={}):
        """Configuration managers constructor.
        
        Parameters
        ----------
        init : dict, optional
            Dictionary with some configuration values to change from the default ones.
        """
        self['seed'] = 0 # for reproducibility
        self['checkpoint'] = 'ixa-ehu/ixambert-base-cased' # the hugging face model to use
        self['batch_size'] = 6
        self['learning_rate'] = 3e-5
        self['dropout'] = 0.2
        self['epochs'] = 5
        self['eps'] = 1e-8 # very small number to prevent any division by zero
        self['pearson_epoch'] = float('inf') # From this epoch in advance, use Pearson as loss
        self['optimizer'] = 'CarlosAdam' # 'CarlosAdam', 'BertAdam', Other torch.optim.AdamW(...)
        self['init_weights'] = torch.nn.init.xavier_uniform_ # torch.nn.init.xavier_normal_, torch.nn.init.kaiming_uniform_, torch.nn.init.kaiming_normal_, ...
        self['augmentation_frac'] = 0.1 # how much to augment the dataset with 0.0 TER value examples
        self['teacher_forcing_rate'] = 0.0
            # Add the post_edit sometimes to help learning.
            # if extra_features enables adds all the features (0 to disable)
        self['patience'] = 10 # maximum number of epochs allowed without improvements before stopping
        self['input_type'] = 'text_basque' # features to use as input
        # Input types:
        # 'text_spanish', 'text_basque', 'text_spanish_basque', 'text_basque+post_edit'
        # 'text_spanish_basque+post_edit', 'ter_score', 'F_measures',
        # 'Post-editedTarget', 'insertions', 'deletions', 'substitutions', 'shifts', 'WdSh', 
        # 'error number', 'word number', 'Post-editTime(ms)', 'all_extra_features'
        self['extra_features'] = False # Whether to add the extra features
        self['paper_size_plots'] = True # Do smaller plots for papers
        self['pearson_learning_rate'] = None # Use a different learning rate for pearson loss
        self['all_outputs'] = False # Whether to use all transformer outputs or just the CLS
        self['loss1'] = None
        self['loss2'] = None
        self['loss3'] = None
        self.update(init)

"""## Dataset Class"""

class Quality(Dataset):
    """Dataset reading class."""

    def __init__(self, dataset_path, name=False, train=False, config={}):
        self.config = config
        # Give a name to the dataset
        if name:
            logging.info(f'Initializing {name} dataset')
        self.name = name
        self.train = train
        
        # Create the BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['checkpoint'])

        # Read the dataset
        self.dataset_path = dataset_path
        df = pd.read_csv(self.dataset_path)

        self.example_ids, self.information = [], defaultdict(lambda: defaultdict(str))

        for row_id, row_info in df.iterrows():
            self.example_ids.append(row_id)
            
            self.information[row_id]['SourceSegment'] = str(row_info['SourceSegment'].lower())
            self.information[row_id]['MTTargetSegment'] = str(row_info['MTTargetSegment'].lower())
            self.information[row_id]['TER'] = float(row_info['TER7'])
            self.information[row_id]["F"] = [float(row_info['F' + str(i)]) for i in range(1, 18)]
            
            if self.train:
                self.information[row_id]['Post-editedTarget'] = str(row_info['Post-editedTarget'].lower())
                self.information[row_id]['insertions'] = int(row_info['insertions7'])
                self.information[row_id]['deletions'] = int(row_info['deletions7'])
                self.information[row_id]['substitutions'] = int(row_info['substitutions7'])
                self.information[row_id]['shifts'] = int(row_info['shifts7'])
                self.information[row_id]['WdSh'] = int(row_info['WdSh7'])
                self.information[row_id]['error number'] = int(row_info['error number7'])
                self.information[row_id]['word number'] = int(row_info['word number7'])
                self.information[row_id]['Post-editTime(ms)'] = int(row_info['Post-editTime(ms)'])
            

    def __len__(self):
        """Gets number of examples.
        
        Returns
        -------
        int
            Number of examples.
        """
        return len(self.example_ids)
    
    def __getitem__(self, idx):
        """Creates one example for training."""
        
        max_length_padding = 512
        
        # Obtain the argument id
        example_id = self.example_ids[idx]
        
        spanish = self.information[example_id]['SourceSegment']
        basque = self.information[example_id]['MTTargetSegment']
        # We always have this information in the dataset.
        
         
        output_dict = {
            "ter_score": torch.FloatTensor([self.information[example_id]["TER"]]),
        }
        
        # IMPORTANT
        # Generate just the tokenizer that we will use later.
        # This will save memory and the improve the allowed batch size and speed.
        if self.config['input_type'] == 'text_spanish':
            output_dict[self.config['input_type']] = self.tokenizer(spanish, padding='max_length', max_length=max_length_padding, truncation=True, return_tensors="pt")
        elif self.config['input_type'] == 'text_basque':
            output_dict['text_basque'] = self.tokenizer(basque, padding='max_length', max_length=max_length_padding, truncation=True, return_tensors="pt")
        elif self.config['input_type'] == 'text_spanish_basque':
            output_dict['text_spanish_basque'] = self.tokenizer(spanish, basque, padding='max_length', max_length=max_length_padding, truncation=True, return_tensors="pt")
        elif self.config['input_type'] == 'F_measures':
            output_dict['F_measures'] = torch.FloatTensor(self.information[example_id]["F"])
        elif self.config['input_type'] == 'Post-editedTarget':
            output_dict['Post-editedTarget'] = self.tokenizer(self.information[example_id]['Post-editedTarget'] if self.train else "", padding='max_length', max_length=256, truncation=True, return_tensors="pt")
        if self.config['input_type'] == 'text_basque+post_edit' or (self.config['teacher_forcing_rate'] > 0 and self.config['input_type'] == 'text_basque'):
            output_dict['text_basque+post_edit'] = self.tokenizer(basque, self.information[example_id]['Post-editedTarget'] if self.train else "", padding='max_length', max_length=256, truncation=True, return_tensors="pt")
        if self.config['input_type'] == 'text_basque+empty_post_edit':
            output_dict['text_basque+empty_post_edit'] = self.tokenizer(basque, "", padding='max_length', max_length = 256, truncation=True, return_tensors="pt")
        if self.config['input_type'] == 'text_spanish_basque+post_edit' or (self.config['teacher_forcing_rate'] > 0 and self.config['input_type'] == 'text_spanish_basque'):
            output_dict['text_spanish_basque+post_edit'] = self.tokenizer(spanish, self.information[example_id]['Post-editedTarget'] if self.train else "", padding='max_length', max_length=256, truncation=True, return_tensors="pt")
      
        all_features = []
        # Another feature: "Post-editTime(ms)". Lo he quitado porque el resultado es muy grande y afecta mucho al entrenar.
        for name in ['insertions', 'deletions', 'substitutions', 'shifts', 'WdSh', 'error number', 'word number']:
            if self.train:
                output_dict[name] = torch.LongTensor([self.information[example_id][name]])
                all_features.append(self.information[example_id][name])
            else:
                output_dict[name] = torch.LongTensor([-1])
                all_features.append(-1)
        
        output_dict['all_extra_features'] = torch.LongTensor(all_features)
        
        return output_dict

"""## PyTorch Classifier Module"""

class AutoClassifier(nn.Module):
    """Regression model for TER prediction with a regression task."""

    def __init__(self, dropout=None, checkpoint=None, config={}):

        if checkpoint is None:
            checkpoint = config['checkpoint']
        if dropout is None:
            dropout = config['dropout']
        self.config = config
        super(AutoClassifier, self).__init__()
        
        # Create the bert model
        self.model = AutoModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(dropout)
        
        # Create the classification layer
        if self.config['all_outputs']:
            input_size = self.model.config.max_position_embeddings * self.model.config.hidden_size
        else:
            input_size = self.model.config.hidden_size
        extra = 7 if self.config['extra_features'] else 0
        self.linear = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.linear.apply(self.init_weights)
        
        self.linear2 = nn.Linear(512 + extra, 256)
        self.relu2 = nn.ReLU()
        self.linear2.apply(self.init_weights)
        
        self.linear3 = nn.Linear(256, 1)
        self.relu3 = nn.ReLU()
        self.linear3.apply(self.init_weights)

    def forward(self, input_id, mask, features=None):
        input_id = input_id.squeeze(1)
        last_hidden_state, pooler_output = self.model(input_ids=input_id, attention_mask=mask, return_dict=False)
        # last_hidden_state: torch.Size([16, 512, 768])
        # pooler_output: torch.Size([16, 768])
        if self.config['all_outputs']:
            # Merge all the last hidden state dimensions:
            last_hidden_state = torch.reshape(last_hidden_state, (last_hidden_state.size()[0], -1))
            output = self.dropout(last_hidden_state)
        else:
            output = self.dropout(pooler_output)
        output = self.relu(self.linear(output))
        if features is not None:
            output = torch.cat((output, features), dim=1)
        output = self.relu2(self.linear2(output))
        output = self.relu3(self.linear3(output))

        return output

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            if 'init_weights' in self.config:
                self.config['init_weights'](m.weight)
            else:
                torch.nn.init.xavier_uniform_(m.weight) # default initializer
            m.bias.data.fill_(0.01)
    
    def build_mlp(self, dim_list, activation='relu', batch_norm='none',
              dropout=0, final_nonlinearity=True):
        layers = []
        for i in range(len(dim_list) - 1):
            dim_in, dim_out = dim_list[i], dim_list[i + 1]
            layers.append(nn.Linear(dim_in, dim_out))
            final_layer = (i == len(dim_list) - 2)
            if not final_layer or final_nonlinearity:
                if batch_norm == 'batch':
                    layers.append(nn.BatchNorm1d(dim_out))
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'leakyrelu':
                    layers.append(nn.LeakyReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)

"""## PearsonLoss Module"""

class PearsonLoss(nn.Module):
    """Pearson loss implementation."""
    def __init__(self):
        super(PearsonLoss, self).__init__()
        
    def forward(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        loss = 1. - cost
        return torch.nan_to_num(loss)

"""## Device Helper Functions"""

def get_default_device():
    """Pick GPU if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device."""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device."""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device."""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches."""
        return len(self.dl)

# Obtain the device
device = get_default_device()
print(device)

"""## Model Training Class

This is the main class that creates the PyTorch model, trains it, evaluates it and generates plots.
"""

class ModelTask:
    """Creates a task to train an specific model.

    It helps doing the complete process that includes:

    - Loading the dataset.
    - Creating and training the model class.
    - Evaluating the model
    - Generating plot figures.
    """

    def __init__(self, config):
        """Model task constructor.
        
        Parameters
        ----------
        config : Config
            Global configuration object.
        """
        self.config = config
        self.figures = {} # Array where plots will be saved for later
        set_seed(self.config['seed'])
    
    def generate_data_plots(self, train, dev, test):
        """Creates a plot with information of the dataset,
        showing the distribution of values in the different splits.
        
        Parameters
        ----------
        train : pandas.DataFrame
        dev : pandas.DataFrame
        test : pandas.DataFrame
            
        Returns
        -------
        self
        """
        golds = [train['TER7'].values, dev['TER7'].values, test['TER7'].values]
        try:
            fig = ff.create_distplot(golds, ['Train', 'Dev', 'Test'], bin_size=5)
            # fig.update_layout(title='Dataset Gold Values Density Plot')
            fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
            fig.update_layout(xaxis_range=[0, 140])
            self.figures['gold density'] = fig
        except np.linalg.LinAlgError:
            pass
        return self
    
    @staticmethod
    def show_figures(figures, paper_size_fotos=False, save=False):
        """Shows the dictionary of figures (plots) from the model.
        This method is a static method to be called from outside.
        
        Parameters
        ----------
        figures : list
            A list of Plotly or matplotlib figures to draw.
        paper_size_fotos : bool
            If true, it will draw smaller plots more appropriate for articles.
        """
        for name, fig in figures.items():
            if isinstance(fig, go.Figure): # plotly
                display(HTML(f'<h4>Plot id: {name}</h4>'))
                if paper_size_fotos:
                    fig.update_layout(width=500, height=400)
                else:
                    fig.update_layout(autosize=True)
                if save:
                    for ext in ['svg', 'png', 'pdf', 'eps']:
                        fig.write_image(f"images/{name}.{ext}")
                    fig.show()
            else: # matplotlib (not used anymore, left just in case)
                if paper_size_fotos:
                    plt.figure(figsize=(5, 7))
                else:
                    plt.figure(figsize=(15, 10))
                plt.show(fig)
    
    def show(self):
        """Shows the different plots saved on different steps of the model.
            
        Returns
        -------
        self
        """
        ModelTask.show_figures(self.figures, self.config['paper_size_plots'])
        self.figures = {} # do not show the same plots twice if called multiple times
        return self

    @staticmethod
    def clean_figures(figures):
        """Cleans the plots to save memory."""
        for name, fig in figures.items():
            if isinstance(fig, go.Figure): # plotly
                fig.data = []
                fig.layout = {}
            else: # matplotlib (not used anymore, left just in case)
                plt.close(fig)
            
    def load_data(self):
        """Loads the three splits of the dataset.
        It saves them in self.train_dl, self.dev_dl and self.test_dl.
        
        It also does the training split augmentation.
            
        Returns
        -------
        self
        """
        df = pd.read_csv(corpus_folder + "train_utf8.csv")
        self.train_df, self.dev_df = train_test_split(df, test_size=0.2) # 20% for development split
        self.test_df = pd.read_csv(corpus_folder + "test2020_utf8.csv")
        # Generate dataset plot before augmenting it:
        self.generate_data_plots(self.train_df, self.dev_df, self.test_df)
        self.train_df = self.augment(self.train_df)
        self.train_df.to_csv("train2.csv")
        self.dev_df.to_csv("dev2.csv")
        self.test_df.to_csv("test2.csv")
        # Create the datasets
        train_ds = Quality("./train2.csv", name="training", train=True, config=self.config)
        dev_ds = Quality("./dev2.csv", name="development", config=self.config)
        test_ds = Quality("./test2.csv", name="testing", config=self.config)
        self.batch_size = self.config['batch_size']
        self.train_dl = DataLoader(train_ds, self.batch_size, shuffle=True)
        self.dev_dl = DataLoader(dev_ds, self.batch_size)
        self.test_dl = DataLoader(test_ds, self.batch_size)
        return self
        
    def augment(self, train):
        """Augments the dataset using the Post-edited value as input feature with a TER of zero.
        
        Parameters
        ----------
        train : pandas.DataFrame
            Training split of the dataset.
        
        Returns
        -------
        train : pandas.DataFrame
            The augmented dataset.
        """
        train_ter_0 = train.copy()
        train_ter_0['MTTargetSegment'] = train_ter_0['Post-editedTarget']
        train_ter_0['TER7'] = 0.0

        # I add these, which are the ones that are used at 0, that is, there are no changes (before they were at another value)
        for name in ['insertions7', 'deletions7', 'substitutions7', 'shifts7', 'WdSh7', 'error number7', 'word number7', 'Post-editTime(ms)']:
            train_ter_0[name] = 0

        # These are not used
        for i in range(0, 17):
            train_ter_0[f'F{i + 1}'] = -1

        # Sample a percentage of the split that adds 0.0 ter since it may overfit otherwise.
        augmentation_frac = self.config['augmentation_frac']
        train = pd.concat([train, train_ter_0.sample(frac=augmentation_frac)])
        train = train.sample(frac=1)
        return train

    def train(self, model, train_dl, dev_dl, loss1, loss2, loss3, optimizer, epochs, print_every=-1):
        """Implements the main model training loop.
        
        This internally sets the self.history attribute with the metric progress during the training.
        It is a dictionary with train and dev keys, containing list of the different scores.
        For example:
        
        ```python
        self.history['train'] =[
            {'Epoch': 1, 'RMSE': 99, 'MAE': 98, 'Pearson': 0.1, 'Spearman': 0.2},
            {'Epoch': 2, 'RMSE': 70, 'MAE': 80, 'Pearson': 0.5, 'Spearman': 0.5},
            # ...
        ]
        ```
        
        Parameters
        ----------
        model : nn.Module
            PyTorch model to train.
        train_dl : list
            Training split data.
        dev_dl : list
            Development split data.
        loss1 : nn.Module
            Usually MSE loss. This is the loss used by default.
        loss2 : nn.Module
            Usually L1Loss.
        loss3 : nn.Module
            Usually Pearson Loss. This loss will be used when epoch >= config['pearson_epoch'].
        optimizer : Optimizer
            Optimizer to use, usually an instance of transformers.AdamW or torch.optim.AdamW.
        epochs : int
            Number of epoch to train.
        print_every : int, optional
            Will print information about the progress every x number of steps (outputs to logging level INFO).
            
        Returns
        -------
        self
        """
        # Store the accuracies, losses and the best epoch in development
        history = {'train': [], 'dev': []}
        self.best_epoch, self.best_loss = 0, float('inf')

        # Create the scheduler
        num_training_steps = epochs * len(train_dl)
        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        # train_size, dev_size = len(train_dl) * self.batch_size, len(dev_dl) * self.batch_size

        # Training loop
        for epoch in tqdm(range(epochs), desc='Training', leave=False):
            logging.info(f'Starting epoch {epoch+1}')
            use_pearson = (epoch+1 >= self.config['pearson_epoch'])
            # First epoch with Pearson:
            if epoch+1 == self.config['pearson_epoch']:
                # If we want a different learning rate for pearson loss:
                if self.config['pearson_learning_rate']:
                    lr = self.config['pearson_learning_rate']
                else:
                    lr = self.config['learning_rate']
                logging.info(f'Pearson learning rate: {lr}')
                optimizer = self.optimizer(model.parameters(), lr=lr)
                lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

            model.train()

            y_gold, y_pred = np.array([]), np.array([])

            # Check each batch
            examples, i = 0, 0
            with tqdm(total=len(train_dl), leave=False, desc='Batch') as pbar:
                for batch in train_dl:
                    information_dict = batch
                    # KEYS
                    # ['text_spanish', 'text_basque', 'text_spanish_basque', 'ter_score', 'F_measures', 
                    # 'Post-editedTarget', 'insertions', 'deletions', 'substitutions', 'shifts', 'WdSh',
                    # 'error number', 'word number', 'Post-editTime(ms)', 'all_extra_features', 'text_basque+post_edit'
                    # 'text_basque+empty_post_edit']


                    batch_size = information_dict['ter_score'].shape[0]
                    examples += batch_size

                    # move the inputs and the expected label to GPU (if there is one)
                    # basque = information_dict['text_basque']
                    # basque = information_dict[self.config['input_type']]
                    # random.random() returns [0.0, 1.0)
                    # If teacher_forcing == 0.0, then use_everything is always false
                    use_everything = random.random() >= (1 - self.config['teacher_forcing_rate'])
                    if use_everything:
                        basque = information_dict[self.config['input_type'] + '+post_edit'].to(device)
                    else: # no teacher forcing
                        basque = information_dict[self.config['input_type']].to(device)
                    
                    text_input_ids = basque['input_ids'].to(device)
                    text_attention_mask = basque['attention_mask'].to(device)

                    ter = information_dict['ter_score'].to(device)

                    if self.config['extra_features']:
                    # When training, put some examples with the real features and others as if the information is not available (-1) 
                        if use_everything:
                            extra_features = information_dict['all_extra_features'].to(device)
                        else:
                            extra_features = torch.ones((batch_size, 7), dtype=torch.long).to(device) * -1
                            # When the feature does not exist we put -1
                    else:
                        extra_features = None

                    # Apply the model
                    output = model(text_input_ids, text_attention_mask, extra_features)

                    # get the loss
                    logging.debug(output)
                    logging.debug(ter)
                    batch_loss = loss3(output, ter) if use_pearson else loss1(output, ter)

                    # Save the outputs
                    y_gold = np.concatenate((y_gold, torch.reshape(ter.cpu(), (-1,))))
                    y_pred = np.concatenate((y_pred, np.reshape(output.detach().cpu().numpy(), (-1,))))

                    if i % print_every == 0:
                        y_pred = np.nan_to_num(y_pred)
                        train_RMSE = np.sqrt(np.mean((y_pred-y_gold)**2))
                        train_MAE = np.mean(np.abs(y_pred - y_gold))
                        train_Pearson = stats.pearsonr(y_gold, y_pred)[0]
                        train_Spearman = stats.spearmanr(y_gold, y_pred)[0]
                        logging.info(f"STEP: {i}/{len(train_dl)}. Total examples seen {examples}. RMSE: {train_RMSE}, MAE: {train_MAE}, Pearson: {train_Pearson}, Spearman: {train_Spearman}")

                    i += 1
                    model.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    pbar.update(1)

            y_pred = np.nan_to_num(y_pred)

            train_RMSE = np.sqrt(np.mean((y_pred-y_gold)**2))
            train_MAE = np.mean(np.abs(y_pred - y_gold))
            train_Pearson = stats.pearsonr(y_gold, y_pred)[0]
            train_Spearman = stats.spearmanr(y_gold, y_pred)[0]
            # Print and store accuracy and loss for training
            logging.info("Final results for training split:")
            logging.info(f"Final RMSE: {train_RMSE}, Final MAE: {train_MAE}, Final Pearson: {train_Pearson}, Final Spearman: {train_Spearman}")
            history['train'].append({
                'Epoch': epoch+1,
                'RMSE': train_RMSE,
                'MAE': train_MAE,
                'Pearson': train_Pearson,
                'Spearman': train_Spearman,
            })

            # Calculate the accuracy and loss for development
            dev_RMSE, dev_MAE, dev_Pearson, dev_Spearman = self.evaluate(model, dev_dl)

            # If this new epoch is better store the information
            # dev_loss = (1 - abs(dev_Pearson)) if use_pearson else dev_RMSE
            dev_loss = (1 - abs(dev_Pearson))
            if dev_loss < self.best_loss:
                logging.info(f"Best new epoch: {epoch+1}")
                self.best_loss = dev_loss
                self.best_epoch = epoch+1

            # Print and store accuracy and loss for developement
            logging.info("Final results for development split:")
            logging.info(f"Final RMSE: {dev_RMSE}, Final MAE: {dev_MAE}, Final Pearson: {dev_Pearson}, Final Spearman: {dev_Spearman}")
            history['dev'].append({
                'Epoch': epoch+1,
                'RMSE': dev_RMSE,
                'MAE': dev_MAE,
                'Pearson': dev_Pearson,
                'Spearman': dev_Spearman,
            })

            # Store the epoch
            if not os.path.exists(f"{save_dir}/checkpoints/"):
                os.mkdir(f"{save_dir}/checkpoints/")

            torch.save(model.state_dict(), f"{save_dir}/checkpoints/" + str(epoch+1) + ".pth")
            logging.info(f"Ending epoch {epoch+1}\n")

            patience = self.config['patience']
            if self.best_epoch - (epoch+1) >= patience:
                logging.info(f'Early stopping: no improvement for {str(patience)} epochs.')
                break

        self.history = {
            'train': pd.DataFrame(history['train']).replace([np.inf, -np.inf], np.nan).fillna(0),
            'dev': pd.DataFrame(history['dev']).replace([np.inf, -np.inf], np.nan).fillna(0),
        }
        return self
    
    def generate_loss_curves(self, splits=['train', 'dev']):
        """Generates a training curve plots with the RMSE, MAE,
        Pearson and Spearman scores in 4 different plots.
        
        The plots are not printed, they are added to the self.figures
        attribute that can be printed later using show() or show_figures() methods.
        
        Parameters
        ----------
        splits : list, optional
            Name of splits to plot, train and dev by default.
            
        Returns
        -------
        self
        """
        for metric in ['RMSE', 'MAE', 'Pearson', 'Spearman']:
            fig = go.Figure()
            for split in splits:
                fig.add_trace(go.Scatter(
                    x=self.history[split]['Epoch'], y=self.history[split][metric],
                    mode='lines', name=split
                ))
            fig.update_layout(xaxis_title='Epoch')
            # title=f"Evolution of the {metric}",
            # Position the legend in a usually empty place
            if metric in ['Pearson', 'Spearman']:
                fig.update_layout(legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99))
            else: # RMSE, MAE
                fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
            self.figures[f'{metric} loss'] = fig

        return self


    @torch.no_grad()
    def predict(self, model, val_dl):
        """Predicts values using the model.
        
        Parameters
        ----------
        model : nn.Module
            The model to evaluate.
        val_dl : list
            The dataset split to use as validation split.
        
        Returns
        -------
        y_gold : list
            Real values.
        y_pred : list
            Predicted values.
        """
        model.eval()

        y_gold, y_pred = np.array([]), np.array([])
        # Check each batch
        for batch in val_dl:
            information_dict = batch
            # KEYS
            # ['text_spanish', 'text_basque', 'text_spanish_basque', 'ter_score', 'F_measures', 
            # 'Post-editedTarget', 'insertions', 'deletions', 'substitutions', 'shifts', 'WdSh', 
            # 'error number', 'word number', 'Post-editTime(ms)', 'all_extra_features']

            batch_size = information_dict['ter_score'].shape[0]

            # move the inputs and the expected label to GPU (if there is one)
            basque = information_dict[self.config['input_type']]
            text_input_ids = basque['input_ids'].to(device)
            text_attention_mask = basque['attention_mask'].to(device)


            ter = information_dict['ter_score'].to(device)
            
            if self.config['extra_features']:
                extra_features = torch.ones((batch_size, 7), dtype=torch.long).to(device) * -1
                # When the feature does not exist we put -1
            else:
                extra_features = None

            # Apply the model
            output = model(text_input_ids, text_attention_mask, extra_features)

            # Save the outputs
            y_gold = np.concatenate((y_gold, torch.reshape(ter.cpu(), (-1,))))
            y_pred = np.concatenate((y_pred, torch.reshape(output.cpu(), (-1,))))

        y_pred = np.nan_to_num(y_pred)
        return y_gold, y_pred

    @torch.no_grad()
    def evaluate(self, model, val_dl):
        """Evaluates a model into a split.
        
        Parameters
        ----------
        model : nn.Module
            The model to evaluate.
        val_dl : list
            The dataset split to use as validation split.
        
        Returns
        -------
        RMSE : float
        MAE : float
        Pearson : float
        Spearman : float
        """
        logging.info("Evaluating the model.")
        y_gold, y_pred = self.predict(model, val_dl)
        rmse = np.sqrt(np.mean((y_pred - y_gold)**2))
        mae = np.mean(np.abs(y_pred - y_gold))
        pearson = stats.pearsonr(y_gold, y_pred)[0]
        spearman = stats.spearmanr(y_gold, y_pred)[0]
        return rmse, mae, pearson, spearman
    
    def optimizer(self, parameters, lr=None):
        """Generates the optimizer used for training.
        
        Parameters
        ----------
        parameters : list
            Iterable of parameters to optimize or dicts defining parameter groups.
        
        Returns
        -------
        optimizer : Optimizer
        """
        if not lr:
            lr = self.config['learning_rate']
        eps = self.config['eps']
        optimizer_name = self.config['optimizer']
        if optimizer_name == 'CarlosAdam':
            optimizer = torch.optim.AdamW(parameters, lr=lr, eps=eps)
        elif optimizer_name == 'BertAdam':
            optimizer = AdamW(parameters, lr=lr, correct_bias=False, eps=eps)
        else:
            optimizer = optimizer_name # Creating another optimizer from outside
        return optimizer
    
    def evaluate_epochs(self, model, test_dl):
        """Evaluates the model in the different epochs.
        
        It calculates the RMSE, MAE, Pearson and Spearman scores.
        
        Parameters
        ----------
        model : nn.Module
            The model used to evaluate.
        test_dl : list
            The dataset used for evaluation, usually the test set.
        
        Returns
        -------
        df : pandas.DataFrame
            The table with the scores, each row includes: Epoch, RMSE, MAE, Pearson and Spearman.
        """
        epochs = self.config['epochs']
        data = []
        for epoch in tqdm(np.arange(epochs) + 1, leave=False, desc='Evaluating'):
            path = f"{save_dir}/checkpoints/{int(epoch)}.pth"
            if os.path.exists(path):
                row = {}
                row['Epoch'] = epoch
                model.load_state_dict(torch.load(path))
                row['RMSE'], row['MAE'], row['Pearson'], row['Spearman'] = self.evaluate(model, test_dl)
                data.append(row)
        df = pd.DataFrame(data).replace([np.inf, -np.inf], np.nan).fillna(0)
        return df
    
    def clean(self):
        """Cleans the saved checkpoint files from the mode.
        Recommended when we already finished and wanted to train another model.
        
        Returns
        -------
        self
        """
        for filePath in glob.glob(f"{save_dir}/checkpoints/*.pth"):
            try:
                os.remove(filePath)
            except:
                logging.warning(f"Error while deleting file : {filePath}")
        return self
    
    def fit(self):
        """Trains the model.
        
        Its main tasks are:
        - Generates the PyTorch model.
        - Generates the losses.
        - Move everything to the GPU.
        - Generates the optimizer.
        - Trains eht model.
        - Generates the loss curves over the train and dev set.
        
        Returns
        -------
        model : nn.Module
            The PyTorch module trained. Try to clean it if not used for saving memory.
        """
        # Clean previous runs
        self.clean()
        # Create the model
        # Note: it is important not to save this in an attribute to avoid being using GPU memory unnecessarily
        model = AutoClassifier(config=self.config)
        # Create the loss
        loss1 = self.config['loss1'] if self.config['loss1'] else nn.MSELoss()
        loss2 = self.config['loss2'] if self.config['loss2'] else nn.L1Loss()
        loss3 = self.config['loss3'] if self.config['loss3'] else PearsonLoss()
        # Move the model to GPU (if there is one)
        model = to_device(model, device)
        # Move the loss to GPU (if there is one)
        loss1 = to_device(loss1, device)
        loss2 = to_device(loss2, device)
        loss3 = to_device(loss3, device)
        epochs = self.config['epochs']
        optimizer = self.optimizer(model.parameters())
        self.train(
            model, self.train_dl, self.dev_dl, loss1, loss2, loss3, optimizer, epochs,
            print_every=50
        )
        # Generate some training related plots
        self.generate_loss_curves()
        return model
    
    def epoch_predictions(self, model, epoch):
        """Predicts train, dev and test set values using a specific epoch.
        
        Function for plotting purposes in plot_predictions method.
        
        Returns both the predictions and the gold values.
        
        Parameters
        ----------
        model : nn.Module
            The model used to evaluate.
        epoch : int
            Epoch of the model to test. It will be loaded from disk.
            
        Returns
        -------
        predictions : dict
            A dictionary with the following keys: Gold, Pred. Both of them lists with the values.
        """
        splits = {
            'Train': self.train_dl,
            'Dev': self.dev_dl,
            'Test': self.test_dl,
        }

        model.load_state_dict(torch.load(f"{save_dir}/checkpoints/{int(epoch)}.pth"))  

        predictions = defaultdict(dict)
        for name, split_dl in tqdm(splits.items(), leave=False, desc='Predictions in best epoch'):
            logging.info(f'Predicting {name} split...')
            y_gold, y_pred = self.predict(model, split_dl)
            predictions[name]['Gold'] = y_gold
            predictions[name]['Pred'] = y_pred
        return predictions
    
    def human_predictions(self, split):
        """Get human predictions using the average TER for each segment.
        
        Parameters
        ----------
        split : str
            Name of the split like "train", "dev" or "test".
        
        Returns
        -------
        preds : list
            List of human predictions.
        """
        splits = {
            'Train': self.train_df,
            'Dev': self.dev_df,
            'Test': self.test_df,
        }
        df = splits[split]
        preds = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc=f'Human prediction in {split}'):
            segment_idx = df['MTTargetSegment'] == row['MTTargetSegment']
            pred = df.loc[segment_idx]['TER7'].mean()
            preds.append(pred)
        return preds
    
    def normalize_predictions(self, predictions):
        """Sorts and normalizes the predictions using the mean and the standard deviation.
        
        The normalization formula is: (x / stdev) - min
        
        Parameters
        ----------
        predictions : dict
            Dict of {str: pandas.DataFrame}, withe the predictions on each split.
        
        Returns
        -------
        df : pandas.DataFrame
            Returns the predictions sorted in ascending order and normalized. Columns: Gold, Pred.
        """
        df_norms = {}
        for name, df in tqdm(predictions.items(), leave=False, desc='Normalizing'):
            df_norms[name] = df.copy()
            for column in df_norms[name].columns:
                # normalize with std: x / std
                df_norms[name][column] = df_norms[name][column] / df_norms[name][column].std()
                # normalize with mean and std: (x - mean) / std
                # df_norms[name][column] = (df_norms[name][column] - df_norms[name][column].mean()) / df_norms[name][column].std()
                # Minimum to zero: x - min
                df_norms[name][column] = df_norms[name][column] - df_norms[name][column].min()
                pass
            df_norms[name] = df_norms[name].sort_values(by=['Gold']).reset_index()
            df_norms[name] = df_norms[name].replace([np.inf, -np.inf], np.nan).fillna(0)
        return df_norms
    
    def generate_plot_predictions(self, predictions, epoch):
        """Generates plots with the normalized TER predictions in each split.
        It also generates density plots of those predictions.
        
        The plots are not printed, they are added to the self.figures
        attribute that can be printed later using show() or show_figures() methods.
        
        Parameters
        ----------
        predictions : dict
            The prediction in dictionary format, usually returend by the epoch_predictions() method.
        epoch : int
            The epoch used to generate the predictions passed (for documentation purposes only).
            
        Returns
        -------
        self
        """
        colors = { # Colors from matplotlib: https://stackoverflow.com/a/42091037
            'Gold': '#ff7f0e', # orange
            'Pred': '#1f77b4', # blue
            'Human': '#2ca02c', # green
        }
        # Normalization
        dfs = {} # convert predictions to dataframes
        for name, split_dl in predictions.items():
            dfs[name] = pd.DataFrame(split_dl)
            dfs[name]['Human'] = self.human_predictions(split=name)
        df_norms = self.normalize_predictions(dfs)
        # TER predictions plot (using normalized data)
        for name, df in tqdm(df_norms.items(), leave=False, desc='Generating plot predictions'):
            fig = go.Figure()
            x = np.arange(df.shape[0]) + 1
            model_name = os.path.basename(self.config['checkpoint'])
            fig.add_trace(go.Scatter(
                x=x, y=df['Gold'], mode='lines', name='Gold', line_color=colors['Gold']
            ))
            fig.add_trace(go.Scatter(
                x=x, y=df['Human'], mode='lines', name='Human', showlegend=False, line_color=colors['Human'],
                opacity=0.2,
            ))
            fig.add_trace(go.Scatter( # predictions smoothed
                x=x,
                y=signal.savgol_filter(
                    df['Human'],
                    53, # window size used for filtering
                    3
                ), # order of fitted polynomial
                mode='lines', name='Human', 
                marker=dict(size=1, color=colors['Human'], symbol='circle'),
            ))
            fig.add_trace(go.Scatter(
                x=x, y=df['Pred'], mode='lines', name=model_name, showlegend=False, line_color=colors['Pred'],
                opacity=0.2,
            ))
            fig.add_trace(go.Scatter( # predictions smoothed
                x=x,
                y=signal.savgol_filter(
                    df['Pred'],
                    53, # window size used for filtering
                    3
                ), # order of fitted polynomial
                mode='lines', name=model_name,
                marker=dict(size=1, color=colors['Pred'], symbol='circle'),
            ))
            # fig.update_layout(title=f'{name} Predictions Normalized (Epoch {epoch})')
            fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

            self.figures[f'predictions split={name}'] = fig
        # Predictions density plots (no normalized data)
        for split, df in tqdm(dfs.items(), leave=False, desc='Generating density plots'):
            columns = ['Pred', 'Gold', 'Human']
            try:
                fig = ff.create_distplot(df[columns].values.T, columns, bin_size=5)
                # fig.update_layout(title=f'{split} Predictions Density Plot (Epoch {epoch})')
                fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
                fig.update_layout(xaxis_range=[0, 140])
                self.figures[f'{split} predictions density'] = fig
            except np.linalg.LinAlgError:
                pass
        return self
    
    def generate_evaluation_plots(self, model, epoch):
        """Generates some plots after training and finishing the evaluation.
        
        This method uses epoch to do some predictions and plots them in ascending order.
        
        The plots are not printed, they are added to the self.figures
        attribute that can be printed later using show() or show_figures() methods.
        
        Parameters
        ----------
        model : nn.Module
            The model used for the predictions.
        epoch : int
            The epoch used for the predictions to plot.
        """
        # Generates a plot with the normalized TER predictions in each split
        predictions = self.epoch_predictions(model, epoch)
        self.generate_plot_predictions(predictions, epoch)
        return predictions
    
    def run(self):
        """Loads the data, trains the model and generates all the plots.
        
        This is the function usually used from outside to do the complete task of training a model and evaluating it.
        
        Returns
        -------
        history : dict
            The training history, including train, dev and test splits with scores. Check the self.train() method for a better description.
        """
        self.load_data()
        model = self.fit()
        self.history['test'] = self.evaluate_epochs(model, self.test_dl)
        best_epoch = int(self.history['test'].iloc[self.history['test']['Pearson'].argmax()]['Epoch']) # test epoch
        self.generate_evaluation_plots(model, best_epoch)
        return self.history, best_epoch, self.figures

"""<a name="Training"></a>
## Training

The idea is to create many models, and to save from them just the results and some plots.

Here we will test the results using different input types.

Later we will use those results to disaply the best results and some plots.

This may take some time to run depending on your GPU and disk speed.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# # Seeds to try:
# seeds = np.arange(5)
# 
# configs = {
#     'ixambert+basque': {
#         'input_type': 'text_basque',
#         'checkpoint': 'ixa-ehu/ixambert-base-cased',
#     },
#     'ixambert+basque_spanish': {
#         'input_type': 'text_spanish_basque',
#         'checkpoint': 'ixa-ehu/ixambert-base-cased',
#     },
#     'ixambert+basque+post_edit': {
#         'input_type': 'text_basque',
#         'checkpoint': 'ixa-ehu/ixambert-base-cased',
#         'teacher_forcing_rate': 0.5, # post edit
#     },
#     'ixambert+basque+features': {
#         'input_type': 'text_basque',
#         'checkpoint': 'ixa-ehu/ixambert-base-cased',
#         'extra_features': True,
#     },
#     'ixambert+basque+post_edit+features': {
#         'input_type': 'text_basque',
#         'checkpoint': 'ixa-ehu/ixambert-base-cased',
#         'teacher_forcing_rate': 0.5, # post edit
#         'extra_features': True,
#     },
# #     'berteus+basque': { # 0.673466
# #         'input_type': 'text_basque',
# #         'checkpoint': 'ixa-ehu/berteus-base-cased',
# #     },
# #     'berteus+basque_spanish': {
# #         'input_type': 'text_basque_spanish',
# #         'checkpoint': 'ixa-ehu/berteus-base-cased',
# #     },
# #     'berteus+basque+post_edit': {
# #         'input_type': 'text_basque',
# #         'checkpoint': 'ixa-ehu/berteus-base-cased',
# #         'teacher_forcing_rate': 0.5, # post edit
# #     },
# #     'berteus+basque+features': {
# #         'input_type': 'text_basque',
# #         'checkpoint': 'ixa-ehu/berteus-base-cased',
# #         'extra_features': True,
# #     },
# #     'berteus+basque+post_edit+features': {
# #         'input_type': 'text_basque',
# #         'checkpoint': 'ixa-ehu/berteus-base-cased',
# #         'teacher_forcing_rate': 0.5, # post edit
# #         'extra_features': True,
# #     },
# #     'roberta-eus-cc100+basque': {
# #         'input_type': 'text_basque',
# #         'checkpoint': 'ixa-ehu/roberta-eus-cc100-base-cased',
# #     },
# #     'roberta-eus-euscrawl+basque': {
# #         'input_type': 'text_basque',
# #         'checkpoint': 'ixa-ehu/roberta-eus-euscrawl-base-cased',
# #     },
# #     'roberta-eus-euscrawl-large+basque': {
# #         'input_type': 'text_basque',
# #         'checkpoint': 'ixa-ehu/roberta-eus-euscrawl-large-cased',
# #     },
# #     'roberta-eus-mc4+basque': {
# #         'input_type': 'text_basque',
# #         'checkpoint': 'ixa-ehu/roberta-eus-mc4-base-cased',
# #     },
# }
# 
# # Save all data frame results for later inspections
# results = defaultdict(lambda: defaultdict(list))
# # And save the model with the best results for each input type
# best = defaultdict(lambda: defaultdict(int))
# for config_name, new_config in tqdm(configs.items(), leave=False, desc='Config'):
#     display(HTML(f'<h3>Name: {config_name}</h3>'))
#     for seed in tqdm(seeds, leave=False, desc='Seed'):
#         config = Config(new_config)
#         config['seed'] = seed
#         # Correct batch depending on model size:
#         if not 'batch_size' in new_config:
#             config['batch_size'] = 8 if 'large' in config['checkpoint'] else 16
#         
#         history, best_epoch, figures = ModelTask(config).run()
#         pearson = float(history['test'][history['test']['Epoch'] == best_epoch]['Pearson'])
#         if pearson > best[config_name]['pearson']:
#             if 'figures' in best[config_name]: # clean previously best model figures from memory
#                   ModelTask.clean_figures(best[config_name]['figures'])
#             best[config_name]['type'] = config_name # may be considered redundant, but used below
#             best[config_name]['seed'] = seed
#             best[config_name]['epoch'] = best_epoch
#             best[config_name]['pearson'] = pearson
#             best[config_name]['figures'] = figures
#             best[config_name]['history'] = history
#         else:
#             ModelTask.clean_figures(figures) # save memory
#             figures = None
#         display(history['test'])
#         # Save the history results
#         for split in history.keys():
#             results[config_name][split].append(history[split])

"""## Average Results

We calculate the averages and standard deviation between all seeds and epochs:
"""

for split in results[list(results.keys())[0]].keys():
    display(HTML(f'<h3>Split: {split}</h3>'))
    for config_name in tqdm(configs.keys(), leave=False, desc='Type'):
        display(HTML(f'<h4>Type: {config_name} ({split})</h4>'))
        df_type = pd.DataFrame()
        for result in results[config_name][split]:
            df_type = pd.concat((df_type, result), axis=0)
        # Remove zero values (NaNs)
        df_type = df_type[~(df_type['Pearson'] <= 0)]
        df_type.drop('Epoch', inplace=True, axis=1)
        df_min = pd.Series(df_type.iloc[df_type['Pearson'].argmin()], name='Min')
        df_max = pd.Series(df_type.iloc[df_type['Pearson'].argmax()], name='Max')
        df_mean = pd.Series(df_type.mean(axis=0), name='Mean')
        df_std = pd.Series(df_type.std(axis=0), name='Std')
        df = pd.DataFrame([df_min, df_max, df_mean, df_std])
        display(df)
    display(HTML(f'<hr/>'))

"""## Best Results

We display best result tables for each input type.
"""

best_type = None
best_type_pearson = 0
for config_name in tqdm(best.keys(), leave=False, desc='Type'):
    display(HTML(f'<h3>Type: {config_name}</h3>'))
    # Save the best model for plotting on the next code section
    if best[config_name]['pearson'] > best_type_pearson:
        best_type = config_name
        best_type_pearson = best[config_name]['pearson']
        
    # Print tables
    print(f'Seed:       ' + str(best[config_name]['seed']))
    print(f'Epoch:      ' + str(best[config_name]['epoch']))
    print(f'Pearson:    ' + str(best[config_name]['pearson']))
    display(best[config_name]['history']['test'])

"""## Best Model Plots

Now we will draw the plots of the best winner model, only one between all the input types.

Plots consume a lot of memory and processing time on the notebook. So we do not want to draw too many of them.
"""

print(f'Type:       ' + best[best_type]['type']) # This is the same as printing just best_type
print(f'Seed:       ' + str(best[best_type]['seed']))
print(f'Epoch:      ' + str(best[best_type]['epoch']))
print(f'Pearson:    ' + str(best[best_type]['pearson']))
ModelTask.show_figures(best[best_type]['figures'], config['paper_size_plots'], True)