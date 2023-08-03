# DCGAN-DTA
Predicting Drug-Target Binding Affinity with Deep Convolutional Generative Adversarial Networks.

## Data
We utilized two DTA datasets including BindingDB, and PDBBind refine set. BindingDB, and PDBBind datasets were downloaded from [here](https://tdcommons.ai/multi_pred_tasks/dti/), and [here](http://www.pdbbind.org.cn/download.php), respectively. It should be noted that you should register and login before downloading data files from the PDBBind repositories.
<br/>
Each dataset folder includes binding affinity (i.e. Y), protein sequences (i.e. proteins.txt), protein sequences for training gan (i.e. proteins_train.txt), drug SMILES (i.e. ligands_can.txt), drug SMILES for training gan (i.e. ligands_train.txt) and protein vectors (i.e. protein_feature_vec.json and protein_feature_vecblsm.json), and a folder includes the train and test folds settings (i.e. folds).

## Requirements
Python <br/>
Tensorflow <br/>
Keras <br/>
Numpy <br/>
matplotlib 3.5.2 <br/>

## Usage
For training and evaluation of the method, you can run the following script for PDBBind dataset:
```
python run_experiments.py --num_windows 128 32 \
                          --smi_window_lengths 4 8 16 \
                          --seq_window_lengths 4 8 16 \
                          --batch_size 256 \
                          --num_epoch 300 \
                          --max_seq_len 2000 \
                          --max_smi_len 200 \
                          --dataset_path 'data/pdb/' \
                          --problem_type 1 \
                          --is_log 0 \
                          --log_dir 'logs/' \
                          --model 'A' \
```
And you can run the following script for BindingDB dataset:
```
python run_experiments.py --num_windows 128 32 \
                          --smi_window_lengths 4 8 16 \
                          --seq_window_lengths 4 8 16 \
                          --batch_size 256 \
                          --num_epoch 300 \
                          --max_seq_len 2000 \
                          --max_smi_len 200 \
                          --dataset_path 'data/bindingdb/' \
                          --problem_type 1 \
                          --is_log 2 \
                          --log_dir 'logs/' \
                          --model 'A' \
```

Since we have three different types of models, there is a parameter named 'model' that controls which type of model is used.
| Model | Proteins    | Drugs    |
| :---:   | :---: | :---: 
| A | Label encoding, embedding layer, DCGAN, and 1DCNN   | Label encoding, embedding layer, DCGAN, and 1DCNN   |
| B | BLOSUM encoding, DCGAN, and 1DCNN   | Label encoding, embedding layer, DCGAN, and 1DCNN   |
| C | BLOSUM encoding, and 1DCNN   | Label encoding, embedding layer, DCGAN, and 1DCNN   |

## Cold-start
Under the constraints of cold-start, DCGAN-DTA can predict binding affinity based on the physio-chemical properties of compound molecules. These properties include logP values computed with Open Babel logP and XLOGP3 tools.
To test based on the Open Babel logP, change the value of 'problem_type' to 2. To test based on XLOGP3, change it to 3.
```
python run_experiments.py --num_windows 128 32 \
                          --smi_window_lengths 4 8 16 \
                          --seq_window_lengths 4 8 16 \
                          --batch_size 256 \
                          --num_epoch 300 \
                          --max_seq_len 2000 \
                          --max_smi_len 200 \
                          --dataset_path 'data/pdb/' \
                          --problem_type 2 \
                          --is_log 0 \
                          --log_dir 'logs/' \
                          --model 'A' \
```
