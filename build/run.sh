dist=0.001
echo "Method: Early break"
/local/storage/shared/HausdorffDistance/build/bin/hd_exec \
	-input1 /local/storage/shared/hd_datasets/dtl_cnty.wkt \
	-input2 /local/storage/shared/hd_datasets/dtl_cnty.wkt \
	-serialize /local/storage/shared/hd_datasets/ser \
	-limit 10000 \
	-move_offset $dist \
	-v=1 \
       	-repeat=1 \
	-variant eb


echo "Method: Zorder"
/local/storage/shared/HausdorffDistance/build/bin/hd_exec \
	-input1 /local/storage/shared/hd_datasets/dtl_cnty.wkt \
	-input2 /local/storage/shared/hd_datasets/dtl_cnty.wkt \
	-serialize /local/storage/shared/hd_datasets/ser \
	-limit 10000 \
	-move_offset $dist \
	-v=1 \
       	-repeat=1 \
	-variant zorder


echo "Method: Yuan"
/local/storage/shared/HausdorffDistance/build/bin/hd_exec \
	-input1 /local/storage/shared/hd_datasets/dtl_cnty.wkt \
	-input2 /local/storage/shared/hd_datasets/dtl_cnty.wkt \
	-serialize /local/storage/shared/hd_datasets/ser \
	-limit 10000 \
	-move_offset $dist \
	-v=1 \
       	-repeat=1 \
	-variant yuan
