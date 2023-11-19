# ChatBot Model Comparing


## Project Summary

Our project focuses on exploring the integration of a Seq2Seq LSTM model with an attention layer, comparing its performance with the GPT-2 model. The SQuAD 2.0 dataset will be utilized for training and evaluation. We aim to assess the relative performance of these models and identify strategies to enhance the Seq2Seq model's capabilities within the existing framework. 

The project will culminate in a comprehensive analysis of model interactions and potential improvements to the Seq2Seq architecture.


### Prepare data 

Build  vocabulary from a corpus of language data. link of  the original source of the dataset: https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/

The detailed format is described in the notebook。

### Build Model 

Seq2Seq LSTM model with an attention layer: Build Attenion, Encoder, Decoder, and larger Sequence to Sequence pattern in PyTorch. 

GPT-2 model: Transformer Architecture, Multi-Head Self-Attention Mechanism, Parameter Scale and Large-Scale pretraining

### Train Model 

Our training procedure involves utilizing Negative Log Likelihood (NLL) loss for parameter evaluation. The dataset is split into training and validation sets.

Both the Seq2Seq LSTM with attention and GPT-2 models are trained, and their evaluation metrics are plotted for a comparative analysis. The trained models are saved upon achieving satisfactory accuracy levels.

### Evaluate & Interact w/ Model 

After comparison, due to defects in the attention mechanism itself and inherent deficiencies in data aggregation, the GPT-2 model performed better.
