# LoE
Unofficial Pytorch implementation of Nvidia's "Implicit Neural Representations with Levels-of-Experts" (LoE) submitted to NIPS-2022

The structure of the LoE model is partly inspired by the official implementation of "MINER: Multiscale Implicit Neural Representation" at [MINER](https://github.com/vishwa91/MINER/blob/main/modules/siren.py)

To implement the position-dependent weights, we go through the following preprocessing steps:
1. Obtain the weight indices of all layers for each entry of x.
2. Obtain the masks for each weight tile in each layer.
3. Tile the input data based on the masks of each weight tile.
4. Treat each tile as a separate channel and run inference on each layer.
5. Untile the output of each layer back to it's original order.
6. Run inference on the last layer.
