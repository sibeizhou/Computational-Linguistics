## Neural MT Homework

### I. Instructions:
`project_baseline.ipynb` ontains the baseline method with Logistic Regression

`project.ipynb` contains methods with Simple RNN, Bi-directional GRU and Bi-directional LSTM. 

`project_uniDirectinalRNNs.ipynb` contains methods with Uni-directional GRU and Uni-directional LSTM.

If there is no a file name `allData.csv` in `data/`:

Before you can run the `project.ipynb`, make sure you either download 
the database from `http://ai.stanford.edu/~amaas/data/sentiment/` into
the data dictionary and run the `dataPreprocess.py` to transfer the 
database to one single .csv file to be better loaded into the program
using command:

    > python3 dataPreprocess.py


#### 1. Baseline: Logistic Regression

The method with model of Logistic Regression is in the `project_baseline.ipynb`, this program run with 4 different hyper-parameters c for the Logistic Regression and got the highest accuracy at 0.88432.

#### 2. Improvements 1: LSTM and GRU

The method with model of Uni-directional GRU and Uni-directional LSTM are in the `project_uniDirectinalRNNs.ipynb`, this program run with Uni-directional LSTM and Uni-directional RNN seperately got the the accuracy with variable epoch.

There are some hyper-parameter we use in the LSTM and GRU:

1. embedding_dim = 256

2. hidden_units = 512

3. target_size = 2   # 2 emotions: 0-negative, 1-positive

4. num_layers = 2

5. dropout = 0.75

6. loss function = nn.CrossEntropyLoss()

7. optimizer = optim.Adam(model.parameters(),lr=0.001)


#### 3. Improvements 2: Simple RNN, Bi-LSTM, Bi-GRU

The method with model of Simple RNN, Bi-directional GRU and Bi-directional LSTM are in the `project.ipynb`, this program run with simple RNN, Bi-directional LSTM and Bi-directional RNN seperately got the the accuracy with variable epoch.

There are some hyper-parameter we use in the Bi-LSTM and Bi-GRU:

1. embedding_dim = 256

2. hidden_units = 512

3. target_size = 2   # 2 emotions: 0-negative, 1-positive

4. num_layers = 2

5. dropout = 0.5

6. bidirectional=True

7. loss function = nn.CrossEntropyLoss()

8. optimizer = optim.Adam(model.parameters(),lr=0.001)


### II. Useful Tool

Due to the large amount of calculations required by the program, under normal circumstances, 
it takes over 20 hours to run with the CPU, so we use the GPU for operations to improve 
efficiency when running the code.

How to run code with GPU:

1. Download CUDA at https://developer.nvidia.com/cuda-toolkit-archive
2. Download cuDNN at https://developer.nvidia.com/rdp/cudnn-archive#a-collapse51b then rename cudnn-11.3-windows-x64-v8.2.0.53 to the cudnn
3. Go to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3 and put the cudnn into it.
4. Add the system environment 
5. Download https://pytorch.org/get-started/locally/ 
6. Then in terminal, type 'python' and import torch and try type torch.cuda.is_available() if it is true then it will be ok.
7. If you want to using pychram, please using the right pyhthon interpreter.

