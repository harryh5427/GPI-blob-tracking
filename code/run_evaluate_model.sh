#!/bin/bash
python evaluate_model.py --model ../models/raft-synblobs.pth --mixed_precision --mode training --metric epe
python evaluate_model.py --model ../models/raft-synblobs.pth --mixed_precision --mode validation --metric epe
python evaluate_model.py --model ../models/raft-synblobs.pth --mixed_precision --mode testing --metric epe

python evaluate_model.py --model ../models/raft-synblobs.pth --mixed_precision --mode training --metric viou
python evaluate_model.py --model ../models/raft-synblobs.pth --mixed_precision --mode validation --metric viou
python evaluate_model.py --model ../models/raft-synblobs.pth --mixed_precision --mode testing --metric viou

python evaluate_model.py --model ../models/gma-synblobs.pth --mixed_precision --mode training --metric epe
python evaluate_model.py --model ../models/gma-synblobs.pth --mixed_precision --mode validation --metric epe
python evaluate_model.py --model ../models/gma-synblobs.pth --mixed_precision --mode testing --metric epe

python evaluate_model.py --model ../models/gma-synblobs.pth --mixed_precision --mode training --metric viou
python evaluate_model.py --model ../models/gma-synblobs.pth --mixed_precision --mode validation --metric viou
python evaluate_model.py --model ../models/gma-synblobs.pth --mixed_precision --mode testing --metric viou

python evaluate_model.py --model ../models/mrcnn-synblobs.pth --mode training --metric viou
python evaluate_model.py --model ../models/mrcnn-synblobs.pth --mode validation --metric viou
python evaluate_model.py --model ../models/mrcnn-synblobs.pth --mode testing --metric viou

python evaluate_model.py --model ../models/flowwalk-synblobs.pth --mixed_precision --mode training --metric epe
python evaluate_model.py --model ../models/flowwalk-synblobs.pth --mixed_precision --mode validation --metric epe
python evaluate_model.py --model ../models/flowwalk-synblobs.pth --mixed_precision --mode testing --metric epe

python evaluate_model.py --model ../models/flowwalk-synblobs.pth --mixed_precision --mode training --metric viou
python evaluate_model.py --model ../models/flowwalk-synblobs.pth --mixed_precision --mode validation --metric viou
python evaluate_model.py --model ../models/flowwalk-synblobs.pth --mixed_precision --mode testing --metric viou
