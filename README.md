# PhonoVQA
## Setup

1. Clone the repository:
    ```
    git clone https://github.com/phong-lt/PhonoVQA
    ```
2. Install the required packages:
    ```
    pip install -r /PhonoVQA/requirements.txt
    ```

## Usage

To run the main script:
```bash
python PhonoVQA/run.py \
	# config file path
	--config-file PhonoVQA/config/latr.yaml \
 
	# mode: train - to train models, eval - to evaluate models, predict - to predict trained models
	--mode train \

	# evaltype: last - evaluate lattest saved model, best - evaluate best-score saved model 
	--evaltype last \
	
	# predicttype: last - predict lattest saved model, best - predict best-score saved model 
	--predicttype best \
```