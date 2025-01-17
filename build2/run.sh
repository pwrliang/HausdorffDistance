dist=0.0001
echo "Method: Early break"
/home/yuan.645/HausdorffDistance/build2/bin/hd_exec \
	-input1 /local/storage/shared/hd_datasets/dtl_cnty.wkt \
	-input2 /local/storage/shared/hd_datasets/dtl_cnty.wkt \
	-serialize /local/storage/shared/hd_datasets/ser \
	-limit 10000 \
	-move_offset $dist \
	-v=1 \
       	-repeat=1 \
	-variant eb


echo "Method: Zorder"
/home/yuan.645/HausdorffDistance/build2/bin/hd_exec \
	-input1 /local/storage/shared/hd_datasets/dtl_cnty.wkt \
	-input2 /local/storage/shared/hd_datasets/dtl_cnty.wkt \
	-serialize /local/storage/shared/hd_datasets/ser \
	-limit 10000 \
	-move_offset $dist \
	-v=1 \
       	-repeat=1 \
	-variant zorder


echo "Method: Yuan"
/home/yuan.645/HausdorffDistance/build2/bin/hd_exec \
	-input1 /local/storage/shared/hd_datasets/dtl_cnty.wkt \
	-input2 /local/storage/shared/hd_datasets/dtl_cnty.wkt \
	-serialize /local/storage/shared/hd_datasets/ser \
	-limit 10000 \
	-move_offset $dist \
	-v=1 \
       	-repeat=1 \
	-variant yuan
