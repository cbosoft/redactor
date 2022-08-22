# Redactor

Script for redacting text from an image.

Machine learning (Faster R-CNN) based processing searches for instances of text on an image and redacts by placing a box over the text on the image.

This can be used for removing potentially sensitive information from images prior to sharing publicly. The model is intended to be used to remove identifying information from microscope images, however it could be applied elsewhere e.g. to remove number plates from photographs, date/time/location information from CCTV images, etc.

# Examples

![example 1](https://raw.githubusercontent.com/cbosoft/redactor/master/example_data/redacted_1.png?raw=true)

Image used above is from the "crystal-microscopics-images-and-annotations" dataset available on [Kaggle](http://www.kaggle.com/datasets/usamaimdad/crystalmicroscopicsimagesandannotations).

![example 2](https://raw.githubusercontent.com/cbosoft/redactor/master/example_data/redacted_2.png?raw=true)

Image used above is from the "POLLEN20L-det" dataset available on [Kaggle](https://www.kaggle.com/datasets/nataliakhanzhina/pollen20ldet).

# Installation
Start by cloning this git repo. From a command line:

```shell
git clone https://github.com/cbosoft/redactor
```

Then navigate into the repo

```shell
cd redactor
```

Optionally, create a new python environment:

```shell
python -m venv .env
source .env/bin/activate
```

Install requirements:

```shell
pip install -r requirements.txt
```

# Usage
Strictly speaking, `redactor` is mainly a library for training a model to pick up text and for using that trained model to perform redaction. The former is controlled via `yaml` config files, the latter by a simple script.

## Training
(TODO)

## Redaction

The [`redact.py`](redact.py) script shows how the redactor should be configued for use. Ideally, you should **copy this file, and edit as necessary**.

There is also a Jupyter notebook provided, which contains the same script as `redact.py`: [`redact.ipynb`](redact.ipynb).

Nevertheless, I'll go through the process of creating your own redact script.

Start by importing the redactor:

```python
from redactor.redact import Redactor
```

Then we need to set it up. It is easiest to point the redactor at a zip file containing the necessary config and model information. There are examples in the ["releases" section of this github repo](https://github.com/cbosoft/redactor/releases). If you do not have one, contact the developer (via email, or on github) for assistance.

```python
ZIP_FILE_LOCATION = r'C:\path\to\zip_file.zip'
redactor = Redactor.from_zip(ZIP_FILE_LOCATION)
```

Next, we need to tell it what images to redact. This is fairly flexible, you can give it a directory of images, or a list of paths or a path to one specific image or a glob pattern.

```python
IMAGES_PATHS = r'C:\path\to\image'
redactor.redact(
    IMAGES_PATHS,
    output_dir=None,
    redact_filename=False
)
```

The `output_dir` and `redact_filename` arguments above are optional. `output_dir` specifies where to put the redacted files. By default (with value `None`), a new output directory is created with format 'REDACTED_{DATE}_{TIME}'. `redact_filename` specifies whether output filenames should be the same as input. If this is set to true, output filenames will just be an incrementing number (useful if the filenames themselves contain sensitive information).

That's all there is to it! Save and run your generated script and you should see redacted images in your specified output directory (or in the default location if not set).

# Redacted images still contain some text! What can I do?

If the redacted images still contain some text then it is likely the machine learning model is getting a bit confused. The provided models were trained on a variety of text sizes and fonts, but possibly a different font or very different font size is affecting it. Contact the developer for assistance, the model may need further training.
