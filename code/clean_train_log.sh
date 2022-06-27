#!/bin/bash

# This script removes lines from lr_find and additional headers from fastai training log
# Call as
# clean_train_log.sh raw_log.csv
perl -ne 'if($h){print if y/,//>2 && /0/} else {print; $h=1}' $1
