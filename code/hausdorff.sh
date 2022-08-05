for i in inputs/cmr-cine-sscrofa/data/png/masks/* outputs/prediction/png/dice_70_50
do
	for j in inputs/cmr-cine-sscrofa/data/png/masks/* outputs/prediction/png/dice_70_50
	do
		if [[ $i != $j ]]
		then
			python code/hausdorff.py $i $j outputs/prediction/comparison/$(basename $i)-$(basename $j)-hausdorff.tsv
		fi
	done
done
