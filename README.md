# Online IDS GNN

Core codebase of the paper "Online Intrusion Detection in Computer Networks Using Edge-Aware Attentive Graph Neural Network".

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

Publication is in progress.

## Contact

Email: aron.kiss@uni-miskolc.hu
