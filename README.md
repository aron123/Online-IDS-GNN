# Online IDS GNN

Core codebase of the paper "Online Intrusion Detection in Computer Networks Using Edge-Aware Attentive Graph Neural Network".

## Installation
Under Ubuntu, you can use the following commands to install dependencies to a new Python virtual environment:

```sh
python3 -m venv .venv/
source .venv/bin/activate
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install scikit-learn==1.6.1
pip install matplotlib==3.10.9
```

## Usage

`main.py` accepts the following arguments:

```python
parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
parser.add_argument("--data_folder", type=str, required=True, help="Path to data folder")
parser.add_argument("--experiment_label", type=str, default="", help="Label of the experiment")
    
parser.add_argument("--reconstruct", action="store_true", help="Perform graph reconstruction")
parser.add_argument("--train", action="store_true", help="Run training phase")
parser.add_argument("--test", action="store_true", help="Run testing phase")
parser.add_argument("--binary", action="store_true", help="Perform binary classification")

parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
parser.add_argument("--num_layers", type=int, default=3, help="Number of GNN layers")
parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden graph dimension")
parser.add_argument("--dropout", type=float, default=0.0, help="Attention dropout rate of GAT layers")

parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay (L2 regularization)")
parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
```

For `explain.py`, the value of `p` should be passed as an argument.

Please consult the source code for further details.

## Citation

Kiss Á, Nehéz K, Hornyák O. Online intrusion detection in computer networks using edge-aware attentive graph neural network. Intelligent Data Analysis: An International Journal. 2026;0(0). doi:10.1177/1088467X261488186

```bib
@article{Kiss2026,
  title = {Online intrusion detection in computer networks using edge-aware attentive graph neural network},
  ISSN = {1571-4128},
  url = {http://dx.doi.org/10.1177/1088467X261488186},
  DOI = {10.1177/1088467x261488186},
  journal = {Intelligent Data Analysis: An International Journal},
  publisher = {SAGE Publications},
  author = {Kiss,  Áron and Nehéz,  Károly and Hornyák,  Olivér},
  year = {2026},
  month = Sept 
}
```

## Contact

Email: aron.kiss@uni-miskolc.hu
