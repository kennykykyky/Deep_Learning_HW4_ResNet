# Subtle R&D Internship - Coding Assignment

In this challenge, I started by writing command line tools for our deep learning team that deals with DICOM I/O. To test the whole inference pipeline, I then wrote a small mock algorithm module that performs a blurring operation on the images, a simple inference pipeline and finally a command line interface defining the data processing workflow.

## Project Structure

python files in folder './configs' control the command line interface; python files in folder './exps' are all implemented inside class or function; python files at outer level are scripts to call class/function inside './exps'; cfg files in folder './cfgs' are used for code test.

The folder structure of the project is as follows:

    .
    ├── cfgs                            # all configuration files for testing
    │   ├── dicom_input_sec1-1.cfg              # cfg for testing task 1-1
    │   ├── dicom_input_sec1-2.cfg              # cfg for testing task 1-2
    │   ├── inference_sec2-2.cfg                # cfg for testing task 2-2
    │   ├── workflow_sec2-3.cfg                 # cfg for testing task 2-3
    │   └── workflow_sec2-3.json                # configuration file for dicom filter (used for testing task 2-3)
    ├── configs                         # python files for controling configuration CLI system
    │   ├── __init__.py                         # control to enter '*_configs.py' file in below
    │   ├── inference_configs.py                # control inference pipeline configuration
    │   ├── input_configs.py                    # control dicom input configuration
    │   ├── output_configs.py                   # control dicom output configuration
    │   └── workflow_configs.py                 # control CLI workflow configuration
    ├── data                            # original data extracted from zip file
    │   ├── 01_BreastMriNactPilot               # Breast MRI DICOM data
    │   └── 16_PETCT_Brain                      # Brain PET-CT DICOM data
    ├── exps                            # experimenters and functions
    │   ├── __init__.py                         
    │   ├── dicom_input.py                      # dicom_input operator (class)
    │   ├── dicom_output.py                     # dicom_output operator (class)
    │   └── functions.py                        # include dicom filter, gaussian blurry module, and other functions
    ├── results                         # output folder
    │   └── ...   
    ├── in_out_test.py                  # python script for testing dicom I/O funct$$ions
    ├── inference_pipeline.py           # python script for testing inference pipeline function
    ├── workflow.py                     # python script for testing workflow CLI
    ├── readme.md                       # markdown file for project introduction

The simple design idea is that image processing functions and command line interface functions are independent and separate from each other, under './configs' and './exps' respectively. The outer python scripts (i.e. in_out_test.py) combined the functions in both './configs' and './exps'. The 'dicom_input.py' and 'dicom_output.py' under './exps' include 'DICOM_Input' and 'DICOM_Output' classes to do operations about DICOM I/O. They can be tested separately or they can also be inserted in the inference pipeline / CLI workflow with different input choices. Other functions such as dicom filter and gaussian blurry module are written inside 'functions.py' to fulfill the requirements of the assignment. They can be tested separately and also can work well when used in the inference pipeline / CLI workflow.

## Usage

Just type `python ***.py @[options]` in the command line to execute the script, where `@[options]` indicates several command line options inside that cfg file. Or else you can also input all command lines by yourself instead of directly calling the cfg file. The python scripts and the '.cfg' files have matching relationships.

'in_out_test.py' --> 'dicom_input_sec1-1.cfg' & 'dicom_output_sec1-2.cfg'
'inference_pipeline.py' --> 'inference_sec2-3.cfg'
'workflow.py' --> 'workflow_sec2-3.cfg'

You can also create your own '.cfg' files or type directly in command line to test different modules.

To note: you can find all configuration information inside 'configs' folder (including the default values for each argument).

### Basic options

#### `--operator`

This option is not necessary for testing the code because the default value is assigned in each testing python script. This option will lead you to add corresponding arguments for DICOM input test, DICOM output test, inference pipeline test, or workflow CLI test.

#### `--input-dicom, -i`

Path to input DICOM directory.

#### `--output-npy, -n`

Path to output numpy file.

#### `--output-json, -j`

Path to output json file.

#### `--input-npy, -n`

Path to input numpy file.

#### `--output-dicom, -o`

Path to output DICOM directory.

#### `--dicom-name`

Filename for output dicom folder.

#### `--save-npy`

To save intermediate numpy result or not

#### `--save-json`

To save intermediate numpy result or not

#### `--npy-name`

Filename for npy file.

#### `--json-name`

Filename for json file.

#### `--sigma`

A float indicating size of the Gaussian kernel in mm.

#### `--config, -c`

Path to the configuration file.

## Example

Note: for Linux/MacOS user, it is recommanded to write a command in a bash/cfg file like `dicom_input_sec1-1.cfg`, them run the test by simply executing the bash/cfg file with corresponding python script, which is more convenient than writing commands directly in the command line; for Windows user, there should be a similar way to execute a command from a file.

### DICOM Input Test (Sec 1-1)

```bash
python in_out_test.py @./cfgs/dicom_input_sec1-1.cfg
```

or

```bash
python in_out_test.py \
    --input-dicom ./data/16_PETCT_Brain/ \
    --output-npy ./results/npy/ \
    --output-json ./results/json/ \
    --save-npy \
    --save-json \
    --npy-name source_folder \
    --json-name source_folder
```

Then the results will be saved in `./results/npy` and `./results/json`.

### DICOM Output Test (Sec 1-2)

```bash
python in_out_test.py @./cfgs/dicom_output_sec1-2.cfg
```

or

```bash
python in_out_test.py \
    --input-npy ./results/npy/01_BreastMriNactPilot.npy \
    --input-dicom ./data/01_BreastMriNactPilot/ \
    --output-dicom ./results/dcm/ \
    --dicom-name source_folder
```

Then the results will be saved in `./results/dcm`.

### Inference Test (Sec 2-2)

```bash
python inference_pipeline.py @./cfgs/inference_sec2-2.cfg$$
```

or

```bash
python inference_pipeline.py \
    --input-dicom ./data/01_BreastMriNactPilot/ \
    --output-dicom ./results/inference_dcm/ \
    --sigma 10
```

Then the results will be saved in `./results/inference_dcm`.

### Workflow CLI Test (Sec 2-3)

```bash
python workflow.py @./cfgs/workflow_sec2-3.cfg
```

or

```bash
python workflow.py \
    --input-dicom ./data/ \
    --config ./cfgs/workflow_sec2-3.json \
    --output-dicom ./results/workflow_dcm/ \
    --sigma 5
```

Then the results will be saved in `./results/workflow_dcm`.

To note: the configuration file that stored dicom filter keyword is named 'workflow_sec2-3.json'. There is currently one filter keyword 'Manufacturer' and with the value of 'GE MEDICAL SYSTEMS'. Both dataset fulfill this filter criterion and therefore the workflow will return two dicom series and processed them separately. If you want to modify or add filter keyword and value in the '.json' file, feel free to try it.
