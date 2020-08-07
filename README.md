# Evil Eye: Generative Adversarial Network

![GitHub language count](https://img.shields.io/github/languages/count/Brunopaes/grzmot-gan.svg)
![GitHub top language](https://img.shields.io/github/languages/top/Brunopaes/grzmot-gan.svg)
![GitHub repo size](https://img.shields.io/github/repo-size/Brunopaes/grzmot-gan.svg)
![GitHub](https://img.shields.io/github/license/Brunopaes/grzmot-gan.svg)

<small>_Optimized for python 3.6_</small>

This project aims in implementing a image GAN. Ideally will be used to learn and generate artists/cartoonists painting styles or in generation of new unique landscapes/scenes.

----------------------

## Dependencies

For installing the requirements, in your ___venv___ or ___anaconda env___, 
just run the following command:

```shell script
pip install -r requirements.txt
```
----------------

## Project's Structure

```bash 
.
└── grzmot-gan
    ├── data
    │   ├── generated_data
    │   │   └── generated_data
    │   │       └── 2020-04-02
    │   │           ├── gan_generated_1.png
    │   │           ├── ...
    │   │           └── gan_generated_20.png
    │   └── train_data
    │       ├── img_0.jpg
    │       ├── ...
    │       └── img_128.jpg
    ├── docs
    │   └── CREDITS
    ├── src
    │   ├── __init__.py
    │   └── settings.json
    ├── tests
    │   └── unittests
    │       └── __init__.py
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    └── requirements.txt
```

#### Directory description

- __data:__ The data dir. Group of non-script support files.
- __docs:__ The documentation dir.
- __src:__ The scripts & source code dir.
- __tests:__ The unittests dir.

----------------

## Usage Notes

Section aimed on clarifying some running issues.

### Running

For running it, at the `~/src` directory just run:

```shell script
python fitting.py
``` 

or, if importing it as a module, just run:
````python
from fitting import training

if __name__ == '__main__':
    training(100, 128)
````

---------------
