# CS224N default final project (2022 IID SQuAD track)

# References:
1. Official Bidaf implementation in [TensorFlow](https://github.com/allenai/bi-att-flow/blob/49004549e9a88b78c359b31481afa7792dbb3f4a/basic/model.py#L128) 
2. char CNN: [blogpost](https://towardsdatascience.com/the-definitive-guide-to-bidaf-part-2-word-embedding-character-embedding-and-contextual-c151fc4f05bb), slides from some [course](https://nlp.seas.harvard.edu/slides/aaai16.pdf)

# Things to implement:
1. include character embeddings
2. Co-Attention module implementation, and see if it produces better result.
3. multi layer co-attention.
4. Self attention (R-Net)
5. Data augmentation, in case we have many parameters.
6. Ensembling some models For final submission (Not sure, exactly how this works)

# Progress:
1. Setup done, baseline results are obtained, and they look similar to the one present in handout.
2. Char embedding implementation is done. -> Test & verify is pending.

# Experiments
1. bidaf-baseline-03: Its the baseline Bidaf model.
2. bidaf-char-02: its the first attempt at char embeddings. kernel size:5, #Filters: 50, no Relu,BatchNorm,Dropout inside characterEmbed layer. word embeddings are getting projected to 50, char embeddings are coming as 50, concat and return.
3. bidaf-char-03: some changes on bidaf-char-02, kernel_size: 3, Relu, Dropout,BatchNorm in the Sequential() for character embed layer.
4. bidaf-char-04: 100 filters, filter size: 3, 100 dimension char embedding output, along with 300 dim word embeddings, getting projected to 100 dim together. Dropout just after reading the pretrained word and char embeddings.
5. (ToDo) bidaf-char-04: Experiments with how to mix & match word & Char embeddings:
    4.1 keep word embeddings 100 size intact, add char embedding 50 dimension. total 150 dimension from embedding layer. (or we can go higher)
    4.2 keep word, char embedding together then project to 100 then return. 
    4.3 
6.(ToDo) HyperParameter Tuning for bidaf-char: mainly, dropoutprob: 0.15, LR: 0.4, 0.6 (or with exponential decay, using the lambda function, do we need it?)
7. (ToDo) Char-CNN Parameter tuning: (#Filters, kernel_size) we can try combining some filters for 3, 5 kernel size, and see, if it makes any difference.


# Results

| Name            | Params  | HyperParams  | Best F1 | Best EM | AvNA | Dev NLL|
| :---:           |     :-: | :-:          | :-:     | :-:     | :-:  |  :-: |
| bidaf-baseline-03         | Baseline params | LR 0.5, drop_prob 0.2 | 61.29 | 57.84| 68.01| 3.08|
| bidaf-char-01         | conv1D with filterSize: 3, out_channels=100, relu,batchnorm, dropout, char_emb + word_emb projected to 100  | LR 0.5, drop_prob 0.2 | 65.64 | 62.58| 71.90| 2.64|
| bidaf-char-02         | conv1D with filterSize: 3,5 out_channels=100+100, relu,batchnorm, dropout, word_emb + char_emb projected to 100| LR 0.5, drop_prob 0.2 | 67.37 | 64.22| 72.88| 2.59|   
| bidaf-char01-coattention (baseline-4)         | char-emb + coattention on bi-directinoal attention | LR 0.5, drop_prob 0.2 |  66.08 | 62.54 | 72.73 | 2.81 |



## Setup

1. Make sure you have [Miniconda](https://conda.io/docs/user-guide/install/index.html#regular-installation) installed
    1. Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
    2. Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)

2. cd into src, run `conda env create -f environment.yml`
    1. This creates a Conda environment called `squad`

3. Run `conda activate squad`
    1. This activates the `squad` environment
    2. Do this each time you want to write/test your code

4. Run `python setup.py`
    1. This downloads SQuAD 2.0 training and dev sets, as well as the GloVe 300-dimensional word vectors (840B)
    2. This also pre-processes the dataset for efficient data loading
    3. For a MacBook Pro on the Stanford network, `setup.py` takes around 30 minutes total  

5. Browse the code in `train.py`
    1. The `train.py` script is the entry point for training a model. It reads command-line arguments, loads the SQuAD dataset, and trains a model.
    2. You may find it helpful to browse the arguments provided by the starter code. Either look directly at the `parser.add_argument` lines in the source code, or run `python train.py -h`.
