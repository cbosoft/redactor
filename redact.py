from redactor.redact import Redactor

if __name__ == '__main__':

    """
    Load machine learning model parameters.
    
    Specify a machine learning model by path to a zip file (provided by developer, or created as a result of training).
    """
    ML_MODEL_STATE_ZIP = 'zoo/2022-08-17_sugar.zip'
    redactor = Redactor.from_zip(ML_MODEL_STATE_ZIP)

    """
    Redact specified images.

    Specified images will not be modified, redacted copies will be placed in $output_dir.
    
    What images should be redacted? Can be:
      (1) a path to a single image,
      (2) a list of paths to images,
      (3) a directory, or
      (4) a glob pattern specifying images.
    """
    IMAGES_TO_REDACT = r'example_data/input_1.png'
    redactor.redact(
        IMAGES_TO_REDACT,
        output_dir=None,  # the default, will create a dir with format 'REDACTED_{DATE}_{TIME}'
        redact_filename=False  # set to true to remove original filenames too (replace with incrementing number)
    )
