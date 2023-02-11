# Evaluating Machine Translation Quality

**Note**: We strongly recommend reading the <a href="https://drive.google.com/file/d/1-k9fWSGnF6_mCxiQehW9GprciOYaWKO9/view?usp=share_link">paper</a> to understand better the work.

<strong>Authors: </strong><a href="mailto:cdominguez019@ikasle.ehu.eus">Carlos Domínguez Becerril</a>, <a href="mailto:xzuazo002@ikasle.ehu.eus">Xabier de Zuazo Oteiza</a><br/>
<strong>Institution: </strong><a href="https://www.ehu.eus/">University of the Basque Country - UPV/EHU</a><br/>
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

Go to the **Training** to do you set up.

**Note:** If you do not see the interactive plots, try to open it with Jupyter Notebook (the original platform used to do the training).