# Automated-Recovery-of-Damaged-Audio-Files-Using-Deep-Neural-Networks

## Getting started
Collect your own data under the path './DB', and run each script sequentially.

```
python 00-refine_db.py

python 01-train_model.py

python 02-decoding_case.py
```

OR

Download small refined data, and run each script sequentially.

```
./prepare_small_data.sh

python 01-train_model.py

python 02-decoding_case.py
```

## Requirements

python 3.X, tensorflow, keras, numpy
  
## Test environments

OS	 		: ubunbu 16.04LTS

python		: 3.5.2

tensorflow	: 1.7.0

keras		: 2.2.4

ngc			: Docker version 18.06.0-ce, build 0ffa825

