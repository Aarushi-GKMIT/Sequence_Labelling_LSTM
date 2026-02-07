# Sequence Labelling with LSTM (BIO Tagging)

This project implements a **Recurrent Neural Network (RNN)** for a **sequence labelling task**, where the model predicts **BIO (Beginning, Inside, Outside) tags** for each token in an input sequence.

The implementation covers the full pipeline: data loading, vocabulary creation, model training, evaluation, and inference.

---

## Problem Statement

Given a sequence of tokens, the task is to assign a label to **each token** using the BIO tagging scheme:

- **B-XXX** → Beginning of an entity  
- **I-XXX** → Inside an entity  
- **O** → Outside any entity  