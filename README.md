# Hierarchical-GAN

![Alt text](/architecture.svg)

## License
This code is licensed under the MIT license. Feel free to use all or portions for your research or related projects so long as you provide the following citation information:

Chen W, Fuge M. Synthesizing Designs With Inter-Part Dependencies Using Hierarchical Generative Adversarial Networks. ASME. J. Mech. Des. 2019. (Accepted)

    @article{chen2019hgan,
        author={Chen, Wei and Fuge, Mark},
        title={Synthesizing Designs with Inter-part Dependencies Using Hierarchical Generative Adversarial Networks},
        journal={Journal of Mechanical Design},
        volume={},
        number={},
        pages={},
        year={2019},
        publisher={American Society of Mechanical Engineers}
    }

## Required packages

- tensorflow-1.6.0
- numpy
- matplotlib

## Usage Example

### Generate the dataset of AHH:

```bash
cd AHH
python build_data.py
```

### Train/evaluate HGAN:

```bash
python run_<n>parts.py
```

positional arguments:
    
```
mode	startover or evaluate
data	dataset
```

optional arguments:

```
-h, --help            	show this help message and exit
--sample_size		sample size
--save_interval 	number of intervals for saving the trained model and plotting results
```

Example: train HGAN on AHH:

```bash
python run_3parts.py startover AHH --sample_size=10000 --save_interval=500
```

### Train/evaluate InfoGAN:

```bash
python run_infogan.py
```

positional arguments:
    
```
mode	startover or evaluate
data	dataset
```

optional arguments:

```
-h, --help            	show this help message and exit
--sample_size		sample size
--save_interval 	number of intervals for saving the trained model and plotting results
```

