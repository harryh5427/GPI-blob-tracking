# Tracking Blobs in the Turbulent Edge Plasma of a Tokamak Fusion Device
[![DOI](https://zenodo.org/badge/501857667.svg)](https://zenodo.org/badge/latestdoi/501857667)

Data, models, and code for the paper "Tracking Blobs in the Turbulent Edge Plasma of a Tokamak Fusion Device" by Han et al., Sci Rep 12, 18142 (2022).

GPI-blob-tracking is a package implementing image recognition frameworks from RAFT, GMA, Flow Walk, and Mask R-CNN to track turbulent structures, called blobs, in the Gas-Puff Imaging (GPI) data from tokamak boundary. Users can use the existing models trained with synthetic data for their own GPI data. Also, the training pipeline and the data files can be implemented for users' custom models and evaluate their performance as described in Han et al., Sci Rep 12, 18142 (2022).

![](data/real_gpi/teaser_raft.gif)

# Baseline repos 
[RAFT](https://github.com/princeton-vl/RAFT), [GMA](https://github.com/zacjiang/GMA), [Mask R-CNN](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html), and [Flow Walk](https://github.com/jasonbian97/flowwalk)

# Acknowledgements
The support from the US Department of Energy, Fusion Energy Sciences, awards DE-SC0014264 and DE-SC0020327, are gratefully acknowledged. Also, this work was supported in part by the Swiss National Science Foundation. This work has been carried out within the framework of the EUROfusion Consortium and has received funding from the Euratom research and training programme 2014--2018 and 2019--2020 under grant agreement No 633053. The views and opinions expressed herein do not necessarily reflect those of the European Commission.
