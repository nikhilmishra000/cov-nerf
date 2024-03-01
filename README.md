
## Code for `Closing the Visual Sim-to-Real Gap with Object-Composable NeRFs`

This repo contains code pertaining to our ICRA 2024 paper:

```
"Closing the Visual Sim-to-Real Gap with Object-Composable NeRFs".
Nikhil Mishra, Maximilan Sieb, Pieter Abbeel, Xi Chen.
In the proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2024.
```

The object-centric NeRF model we proposed, COV-NeRF, is implemented in `cov_nerf/model.py`.

The Jupyter notebook `example.ipynb` shows how to render and manipulate scenes with a pretrained COV-NeRF model we have provided.

Run `download.py` to fetch the checkpoint and data from Google Drive. 


### BibTeX

```
 @inproceedings{
     mishra2024closing,
     title={Closing the Visual Sim-to-Real Gap with Object-Composable NeRFs},
     author={Nikhil Mishra and  Maximilian Sieb and Pieter Abbeel and Xi Chen},
     year={2024},
     booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
}
```

### License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work, including the paper, code, weights, and dataset, is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
