# BebopNet

This repository contains PyTorch implementation and a trained model for our paper:

> [BebopNet: Deep Neural Models for Personalized Jazz Improvisations]()   
>
> Shunit Haviv Hakimi, Nadav Bhonker, Ran El-Yaniv
> Computer Science Department, Technion – Israel Institute of Technology  
>
> Published in Proceedings of 21st International Society of Music Information Retrieval Conference, ISMIR 2020



Listen to samples and find supplementary material here:  
[https://shunithaviv.github.io/bebopnet/](https://shunithaviv.github.io/bebopnet/)


------------------------

This repository includes:
- Code for the complete pipeline for generating personalized jazz improvisations presented in our paper.
- BebopNet: A trained model for generating jazz improvisations, trained on a dataset of Bebop giants.
- XMLs and backing tracks for generating jazz improvisations on with BebopNet.
    - All jazz backing tracks were generated with [Band In A Box](https://www.pgmusic.com/).

------------------------

Run all scripts with `bebopnet-code` as root.  
If you are only interested in generating improvisations, skip to step 2.

------------------------


### Step 0: Data Preparation
Collect the dataset from xml files into a network-friendly format and pickle it:
```
python jazz_rnn/0_data_prep/gather_data_from_xml.py
```
- Place xml files in `resources/dataset/train` and `resources/dataset/test`: 
```
└───resources
    └───dataset
        └───train
        |   └───artist11
        │   │   │   file111.xml
        │   │   │   file112.xml
        │   │   │   ...
        │   └───artist12
        │   │   │   file121.xml
        │   │   │   file122.xml
        │   │   │   ...
        │   │   ...
        └───test
            └───artist21
            │   │   file211.xml
            │   │   file212.xml
            │   │   ...
            └───artist22
            │   │   file221.xml
            │   │   file222.xml
            │   │   ...
            │   ...
```

- Important parameters:
     - `--xml_dir` path to dataset folder.
     - `--out_dir` path to output pickles.
     - `--labels_file` keep empty (`''`). Usage explained in step 4.
     - `--check_time_signature` throws exception if time signature is not 4/4.
     - `--no_eos` True if you desire to train LSTM. For transformer training use defaults.
- The script outputs 3 pickle files:
    - `train.pkl` training dataset.
    - `val.pkl` validation dataset.
    - `converter_and_duration.pkl` music converter (for example, information about duration decoding).

### Step 1: Train BebopNet - Next-Note Prediction

#### LSTM Based BebopNet
Based on: [pytorch/examples/word_language_model](https://github.com/pytorch/examples/tree/master/word_language_model)  
Train a neural network to predict the next note of the dataset:  
```
python jazz_rnn/B_next_note_prediction/train.py
```
- Model is defined in `jazz_rnn/B_next_note_prediction/model.py` 
- Notice: step 0 should be run with `--no_eos`
- Data and saving arguments:
    - `--data-pkl` path to the directory of the dataset and converter pickles.
    - `--save-dir` path for saving the trained model
- Convergence graphs can be seen in an HTML file in `save-dir`.

#### Transformer Based BebopNet
Based on: [transformer-xl](https://github.com/kimiyoung/transformer-xl)  
Train a neural network to predict the next note of the dataset:  
```
python jazz_rnn/B_next_note_prediction/transformer/train.py --config path_to_yml
```
- Model is defined in `jazz_rnn/B_next_note_prediction/transformer/mem_transformer.py` 
- Notice: step 0 should be run **without** `--no_eos`
- Configurations should be placed in a designated yaml file.
    - `--config` path to ymk file with arguments (`./configs/train_nnp.yml`).
    - Data and saving arguments (to be defined in config file):
        - `--data-pkl` path to the directory of the dataset and converter pickles.
        - `--work_dir` path for saving the trained model
- Convergence graphs can be seen using [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html) in `work-dir`.

### Step 2: Generate samples with BebopNet
```
python jazz_rnn/B_next_not_prediction/generate_from_xml.py
```
- You should manually update `jazz_rnn/B_next_not_prediction/generation_utils.py` with the following details:
    - Paths to xml file of jazz standard (or any song) for BebopNet to improvise over.
    - Paths to mp3 files of backing track of the jazz standard. Notice its tempo should match the xml file.
    - Further details are explained in `generation_utils.py`. 
        Also, take a look at `set_args_by_song_name()` in `generate_from_xml.py`. 
        This is where the we use the defined dictionaries. 
    - The song to improvise over may be ********************
- Song, model and saving arguments:
    - `--song` name of song to improvise over as defined in `generation_utils.py`
    - `--model_dir` path to directory of the model for generating improvisations
    - `--checkpoint` name of desired model (`model.pt` file)
    - `--score_model` keep empty (`''`). Usage explained in step 4. When None is specified, beam search is based on likelihood.
    - `--save_dir` path for saving the generated improvisation:
        - Notes in `xml` format
        - Improvisation with backing track in `mp3` format
- Search options:
    - `--non-stochstic-search` will trigger a greedy generation, choosing only the most probable note at each step.
    - `--beam_search` defines beam search depth units (notes, measures or None)
    - `--beam_depth` defines number of units as beam depth (Number of notes or measures as defined in `beam_search`)
    - `--beam_width` defined number of k options to choose after `beam_depth` steps.  
- `--verbose` flag will print details of xml to mp3 process for debugging process.

Further beam-search related details can be found in the paper.

**If you desire to use our pre-trained model, run with:**  
**`--model_dir training_results/transformer/model --checkpoint model.pt`** 
 
#### Harmony Guided Improvisation
Using `--score_model harmony` with step 2 will trigger a generation based on harmony coherency preference.
When using beam search, the next notes to be preferred will be coherent with the scale of the chord currently being played.
You can read more in the supplementary material PDF.

### Step 3: User Labels Elicitation
```
python jazz_rnn/C_reward_induction/online_tagger_gauge.py
```
Details on the labeling process can be found in the supplementary material PDF.
- You should manually update `online_tagger_gauge.py` with the following details:
    - `song_name_dict` Dictionary of song names (one word to full name)
    - `song_head_mp3_dict` Dictionary of jazz standard mp3 file (with no improvisation)
    - Further details are explained in `online_tagger_gauge.py`.
- Data and saving arguments:
    - `--dir` directory of mp3 and xml files to label (assuming the mp3 files containing only improvisations, without the head melody. i.e, generated with `--remove_head_from_mp3`)
        This directory should have two sub directories, `train` and `test`, that store the generated improvisations for training and validating the user preference metric respectively.
    - `--labels_save_dir` path to directory to save labels in.
    
### Step 4: Train User Preference Metric
Prepare dataset for user preference metric training:
```
python jazz_rnn/A_data_prep/gather_data_from_xml.py --labels_file path_to_user_labels --out_dir results/dataset_user --cached_converter results/dataset_pkls/converter_and_duration.pkl
```
- Important arguments:
    - `--xml_dir` path to the xmls of the labeled improvisations (same directory used in step 3).
        You should put the xmls with_chords in a sub-folder. The directories should look as follows:
    ```
    └───resources
        └───dataset
            └───train
            │  └───xml_with_chords
            │       └───artist11
            │       │   │   file111.xml
            │       │   │   file112.xml
            │       │   │   ...
            │       └───artist12
            │       │   │   file121.xml
            │       │   │   file122.xml
            │       │   │   ...
            │       │   ...
            └───test
                └───xml_with_chords
                    └───artist21
                    │   │   file211.xml
                    │   │   file212.xml
                    │   │   ...
                    └───artist22
                    │   │   file221.xml
                    │   │   file222.xml
                    │   │   ...
                    │   ...
    ```
    
    - `--labels_file` path to user labels saved in step 3
    - `--cached_converter` path to converter used to train the generative model (created in step 0)
    - `--out_dir` NOTICE! define a different folder than the one in step 0 to save your pkl files.



Train regression model:

```
python jazz_rnn/C_reward_induction/train_reward_reg.py
```
**NOTICE: The current version of user model training currently only supports the LSTM generative model.**

- Important arguments:
    - `--pretrained_path` path to the pre-trained generative model (`.pt` file) from step 1.
    - `--data-pkl` path folder containing the gathered train and test pickles. 
    - `--save-dir` path for saving the preference model
    - `--save` saving name for the prefernce model


### Step 5: Generate Personalized Model
 ```
python jazz_rnn/B_next_not_prediction/generate_from_xml.py --score_model path_to_user_metric
```

Use the same parameters as step 2, with addition:
- `--score_model` path to the user preference metric for the beam search to optimize.

-------------------------------

# Citation

If you found our code helpful, please consider citing us:

```
@inproceedings{haviv2020bebopnet,
title={BebopNet: Deep Neural Models for Personalized Jazz Improvisations},
author={Shunit Haviv Hakimi and Nadav Bhonker and Ran El-Yaniv},
year={2020},
booktitle={Proceedings of the 21st International Society for Music Information Retrieval Conference (ISMIR)}
}
```
