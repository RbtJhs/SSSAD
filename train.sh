for FOLD in 1 2 3 4 5;
do
  bash tools/dist_train_partially.sh semi ${FOLD} 50 1
done
