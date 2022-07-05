#!/bin/bash

# Makes predictions for all images and saves them to con and png.
# Run from root of dataset.

conda activate cmr-seg-pig

# predictions to con
for i in inputs/cmr-cine-sscrofa/data/raw/masks/obs0_rep0/*
do
	a=$(basename $i .con)
	python code/preds2con.py --id $a --model model/dice_70_50-final.pkl --indir inputs/cmr-cine-sscrofa/data/raw/images/$a --outdir outputs/prediction/con/dice_70_50 --con $i
done

# predictions to png
for i in inputs/cmr-cine-sscrofa/data/raw/masks/obs0_rep0/*
do
	a=$(basename $i .con)
	python code/preds2png.py --id $a --model model/dice_70_50-final.pkl --indir inputs/cmr-cine-sscrofa/data/raw/images/$a --outdir outputs/prediction/png/dice_70_50
done
