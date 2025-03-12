# python train.py -s "../../dataset/image/real/sihun_brow_train" \
# "../../dataset/image/child/local_child" -v "ch_child" -ch "child" --save_ckpt

# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" \
# "../../dataset/image/piers/local_piers_refined" -v "ch_piers" -ch "piers" --save_ckpt

# python train_brow.py -s "../../dataset/image/real/sihun_brow_train" \
# "../../dataset/image/piers/local_piers_brow" -v "ch_piers" -ch "piers" --save_ckpt

# python train_brow.py -s "../../dataset/image/real/sihun_brow_train" \
# "../../dataset/image/piers/local_piers_brow" -v "ch_piers" -ch "piers" \
# --load_ckpt_name "2023_09_08_17_32_09-ch_piers" --load_iter '010000' --save_ckpt


# python train_brow.py -s "../../dataset/image/real/sihun_brow_train" \
# "../../dataset/image/child/local_child_brow" -v "ch_child" -ch "child" --save_ckpt

# python train_brow.py -s "../../dataset/image/real/sihun_brow_train" \
# "../../dataset/image/girl/local_girl_brow" -v "ch_girl" -ch "girl" --save_ckpt \
# --load_ckpt_name "2023_09_16_17_39_11-ch_girl" --load_iter '070000' --save_ckpt

### EG2024 refined
# child maloclm mery girl(render) piers(align)

# 2408 10542
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" \
# "../../dataset/image/malcolm/local_malcolm_refined" -v "ch_malcolm" -ch "malcolm" --save_ckpt

# 2408 5421
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" \
# "../../dataset/image/child/local_child_refined" -v "ch_child" -ch "child" --save_ckpt

# 2408 5840
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" \
# "../../dataset/image/mery/local_mery_refined" -v "ch_mery" -ch "mery" --save_ckpt

# 2408 2900
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" \
# "../../dataset/image/piers/local_piers_REM" -v "ch_piers" -ch "piers" --save_ckpt

# TODO
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" \
# "../../dataset/image/piers/local_girl_REM" -v "ch_girl" -ch "girl" --save_ckpt


### new

# child 5421
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" \
# "../../dataset/image/child/local_child_REM_5420" -v "ch_child" -ch "child" \
# --load_ckpt_name "2023_09_22_08_59_34-ch_child" --load_iter "110000" --save_ckpt

# girl 2900 ?????
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" \
# "../../dataset/image/girl/local_girl_REM_2900" -v "ch_girl" -ch "girl" --save_ckpt
# --load_ckpt_name "2023_09_22_09_15_58-ch_girl" --load_iter "030000" --save_ckpt


# TODO
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" \
# "../../dataset/image/mery/local_mery_REM_5840" -v "ch_mery" -ch "mery" --save_ckpt

# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" \
# "../../dataset/image/malcolm/local_malcolm_REM_10542" -v "ch_malcolm" -ch "malcolm" --save_ckpt

# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" \
# "../../dataset/image/piers/local_piers_REM_2900" -v "ch_piers" -ch "piers" --save_ckpt

# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" \
# "../../dataset/image/girl/local_girl_REM_2900" -v "ch_girl" -ch "girl" --save_ckpt


# python train_brow.py -s "../../dataset/image/real/sihun_train_2408_brow" \
# "../../dataset/image/child/local_child_brow_REM_5420" -v "ch_child" -ch "child" \
# --load_ckpt_name "2023_09_26_09_48_13-ch_child" --load_iter "050000" --save_ckpt


# python train_brow.py -s "../../dataset/image/real/sihun_train_2408_brow" \
# "../../dataset/image/mery/local_mery_brow_REM_5840" -v "ch_mery" -ch "mery" \
# --load_ckpt_name "2023_09_26_09_49_40-ch_mery" --load_iter "050000" --save_ckpt

# python train_brow.py -s "../../dataset/image/real/sihun_train_2408_brow" \
# "../../dataset/image/malcolm/local_malcolm_brow_REM_10542" -v "ch_malcolm" -ch "malcolm" \
# --load_ckpt_name "2023_09_26_09_53_10-ch_malcolm" --load_iter "050000" --save_ckpt

# python train_brow.py -s "../../dataset/image/real/sihun_train_2408_brow" \
# "../../dataset/image/girl/local_girl_brow_REM_2900" -v "ch_girl" -ch "girl" --save_ckpt



# python train_brow.py -s "../../dataset/image/real/sihun_train_2408_brow" \
# "../../dataset/image/piers/local_piers_brow_REM_2900" -v "ch_piers" -ch "piers" --save_ckpt


### EG ys

# python train_brow.py -s "../../dataset/image/real/ys_train_1985" \
# "../../dataset/image/child/local_child_brow_REM_5420" -v "ch_child" -ch "child" --save_ckpt

# python train_brow.py -s "../../dataset/image/real/ys_train_1985" \
# "../../dataset/image/mery/local_mery_brow_REM_5840" -v "ch_mery" -ch "mery" --save_ckpt

# python train_brow.py -s "../../dataset/image/real/ys_train_1985" \
# "../../dataset/image/malcolm/local_malcolm_brow_REM_10542" -v "ch_malcolm" -ch "malcolm" --save_ckpt


# python train_brow.py -s "../../dataset/image/real/ys_train_1985" \
# "../../dataset/image/girl/local_girl_brow_REM_2900" -v "ch_girl" -ch "girl" --save_ckpt

# python train_brow.py -s "../../dataset/image/real/ys_train_1985" \
# "../../dataset/image/piers/local_piers_brow_REM_2900" -v "ch_piers" -ch "piers" --save_ckpt



### EG sh - girl with new brow 3patch
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" "../../dataset/image/girl/local_girl_REM_2900_NEW_lipsBig" -v "ch_girl" -ch "girl" --save_ckpt
# python train.py -s "../../dataset/image/real/sihun_brow_train_refined" "../../dataset/image/girl/local_girl_REM_2900_NEW" -v "ch_girl" -ch "girl" --save_ckpt



### EG girl with old brow 5patch (Fig10)
# python train_brow.py -s "../../dataset/image/real/sihun_train_2408_brow" \
# "../../dataset/image/girl/local_girl_brow_REM_2900_oldBrow" -v "ch_girl" -ch "girl" \
# --load_ckpt_name "2023_10_05_09_06_49-ch_girl" --load_iter "010000" --save_ckpt


### CGF Revision
    ## for better closing eyes and train wo rotated source image
python train.py -s "../../dataset/image/real/local_real_original_woRot" "../../dataset/image/child/local_child_REM_5420" --load_ckpt_name "2024_04_10_17_51_37-ch_child" --load_iter "110000" -v "ch_child" -ch "child" --save_ckpt

python train.py -s "../../dataset/image/real/local_real_original_woRot" "../../dataset/image/malcolm/local_malcolm_REM_10542" --load_ckpt_name "2024_04_10_10_57_55-ch_malcolm" -r "130000" -v "ch_malcolm" -ch "malcolm" --save_ckpt

# C2C
# python train.py -s "../../dataset/image/malcolm/local_malcolm_REM_10542" "../../dataset/image/girl/local_girl_REM_2900" -v "c2c_m2g" -ch "girl" --save_ckpt
# python train.py -s "../../dataset/image/girl/local_girl_REM_2900" "../../dataset/image/malcolm/local_malcolm_REM_10542" -v "c2c_g2m" -ch "malcolm" --save_ckpt

# mery-malcolm
# python train.py -s "../../dataset/image/mery/local_mery_REM_5840_to_malcolm" "../../dataset/image/malcolm/local_malcolm_REM_10542_to_mery" -v "c2c_me2mal" -ch "malcolm" --save_ckpt
# python train.py -s "../../dataset/image/malcolm/local_malcolm_REM_10542_to_mery" "../../dataset/image/mery/local_mery_REM_5840_to_malcolm" -v "c2c_mal2me" -ch "mery" --save_ckpt

python train.py -s "../../dataset/image/real/sihun_train_2408_brow" "../../dataset/image/metahuman/local_metahuman_refined" -v "real_metahuman" -ch "metahuman" --save_ckpt



### ICCV 2025 YAT

python train.py -s "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/source/m03" "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target/Victor" -v "ch_Victor" -ch "Victor" --save_ckpt
    python train.py --load_ckpt_name "2025_02_23_10_57_22-ch_Victor" --load_iter "050000" -s "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/source/m03" "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target/Victor" -v "ch_Victor" -ch "Victor" --save_ckpt

python train.py -s "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/source/m03" "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target/Emily" -v "ch_Emily" -ch "Emily" --save_ckpt
    python train.py --load_ckpt_name "2025_02_23_11_14_23-ch_Emily" --load_iter "050000" -s "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/source/m03" "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target/Emily" -v "ch_Emily" -ch "Emily" --save_ckpt

python train.py -s "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/source/m03" "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target/Morphy" -v "ch_Morphy" -ch "Morphy" --save_ckpt
python train.py -s "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/source/m03" "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target/VMan" -v "ch_VMan" -ch "VMan" --save_ckpt
python train.py -s "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/source/m03" "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target/Malcolm" -v "ch_Malcolm" -ch "Malcolm" --save_ckpt
