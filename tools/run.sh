#!/bin/sh
while true
do
	res=`ps -aux | grep 99092 | grep -v grep`
	if [ "$res" != "" ];
	then 
		echo 'process exist'
		sleep 1m
	else
		echo 'process finished'
		#conda activate myenv
		PORT=20020 CUDA_VISIBLE_DEVICES=2,3,4,5 nohup ./dist_train.sh ../configs/text-based_person_search/MACA_PRW_faster_rcnn.py 4 >log.txt 2>&1 &
		#nohup ./test.sh &> mytest.txt &
		echo 'process begin'
		break;
	fi
done

