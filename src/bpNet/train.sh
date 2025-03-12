# python train.py -d "../../dataset/image/child/local_child" -a "../../dataset/animation/rom02.json" --save_ckpt -v "train_BPNet"
# python train.py -d "../../dataset/image/piers/local_piers_refined" -a "../../dataset/animation/piers/piers_refined.pth" --save_ckpt -v "train_WEM"

# 0904 brow 0909
# python train_brow.py -d "../../dataset/image/piers/local_piers_brow" -a "../../dataset/animation/piers/piers_refined.pth" --save_ckpt -v "train_WEM"

# python train_brow.py -d "../../dataset/image/piers/local_piers_brow" -a "../../dataset/animation/piers/piers_refined.pth" \
# --load_ckpt_name '2023_09_08_17_40_03-ch_None' --load_iter '040000' --save_ckpt -v "train_WEM"

# python train_brow.py -d "../../dataset/image/piers/local_piers_brow" -a "../../dataset/animation/piers/piers_refined.pth" --save_ckpt -v "train_WEM"

# python train_brow.py -d "../../dataset/image/child/local_child_brow" -a "../../dataset/animation/child/train_local.pth" --save_ckpt \
# --load_ckpt_name '2023_09_16_17_12_28-ch_child' --load_iter '340000' --save_ckpt -v "train_WEM"


# python train_brow.py -d "../../dataset/image/girl/local_girl_brow" -a "../../dataset/animation/girl/girl_refined_v2.pth" --save_ckpt \
# --load_ckpt_name '2023_09_16_17_43_42-ch_girl' --load_iter '300000' --save_ckpt -v "train_WEM"


### EG2024 refined
# python train.py -d "../../dataset/image/child/local_child_WEM_19215" -a "../../dataset/animation/child/train_local_19215.pth" \
# --load_ckpt_name "2023_09_22_09_06_23-ch_child" --load_iter '230000' --save_ckpt -v "train_WEM"

# python train.py -d "../../dataset/image/girl/local_girl_WEM_2700" -a "../../dataset/animation/girl/girl_refined_v2_2700.pth" \
# --load_ckpt_name "2023_09_22_16_07_42-ch_girl" --load_iter '120000' --save_ckpt -v "train_WEM"

# TODO 안한듯?
# python train.py -d "../../dataset/image/mery/local_mery_WEM_18810" -a "../../dataset/animation/mery/train_local_18810.pth" --save_ckpt -v "train_WEM"
# python train.py -d "../../dataset/image/malcolm/local_malcolm_WEM_18810" -a "../../dataset/animation/malcolm/train_local_18810.pth" --save_ckpt -v "train_WEM"
# python train.py -d "../../dataset/image/piers/local_piers_WEM_2700" -a "../../dataset/animation/piers/piers_refined_2700.pth" --save_ckpt -v "train_WEM"

### EG2024 brow
# python train_brow.py -d "../../dataset/image/child/local_child_brow_WEM_19215" -a "../../dataset/animation/child/train_local_19215.pth" -v "train_WEM" --char_name "child" \
# --load_ckpt_name "2023_09_26_09_55_47-ch_child" --load_iter '120000' --save_ckpt -v "train_WEM"

# python train_brow.py -d "../../dataset/image/mery/local_mery_brow_WEM_18810" -a "../../dataset/animation/mery/train_local_18810.pth" -v "train_WEM" --char_name "mery" --save_ckpt
# python train_brow.py -d "../../dataset/image/malcolm/local_malcolm_brow_WEM_18810" -a "../../dataset/animation/malcolm/train_local_18810.pth" -v "train_WEM" --char_name "malcolm" --save_ckpt

# python train_brow.py -d "../../dataset/image/girl/local_girl_brow_WEM_2700" -a "../../dataset/animation/girl/girl_refined_v2_2700.pth" -v "train_WEM" --char_name "girl" \
# --load_ckpt_name "2023_09_28_15_27_57-ch_girl" --load_iter '020000' --save_ckpt -v "train_WEM"
# python train_brow.py -d "../../dataset/image/piers/local_piers_brow_WEM_2700" -a "../../dataset/animation/piers/piers_refined_2700.pth" -v "train_WEM" --char_name "piers" \
# --load_ckpt_name "2023_09_28_14_21_58-ch_piers" --load_iter '020000' --save_ckpt -v "train_WEM"


### EG 3 patch girl piers WEM rom not considered new. REM은 걍 ckpt 대체
# python train.py -d "../../dataset/image/girl/local_girl_WEM_2700" -a "../../dataset/animation/girl/girl_refined_v2_2700.pth" --save_ckpt -v "train_WEM" --char_name "girl"
# python train.py -d "../../dataset/image/piers/local_piers_WEM_2700" -a "../../dataset/animation/piers/piers_refined_2700.pth" --save_ckpt -v "train_WEM" --char_name "piers"


### EG girl with new brow 3patch
# python train.py -d "../../dataset/image/girl/local_girl_WEM_2700_NEW" -a "../../dataset/animation/girl/girl_refined_v2_2700.pth" --save_ckpt -v "train_WEM" --char_name "girl"
# python train.py -d "../../dataset/image/girl/local_girl_WEM_2700_NEW_lipsBig" -a "../../dataset/animation/girl/girl_refined_v2_2700.pth" --save_ckpt -v "train_WEM" --char_name "girl"


### EG girl with old brow 5patch (Fig10)
# python train_brow.py -d "../../dataset/image/girl/local_girl_brow_WEM_2700_oldBrow" -a "../../dataset/animation/girl/girl_refined_v2_2700.pth" -v "train_WEM" --char_name "girl"
# --load_ckpt_name "2023_10_05_08_56_28-ch_girl" --load_iter '020000' --save_ckpt -v "train_WEM"

### CGF revision bshp vs PCA vs PCA (no norm, which is ours)
    ## bshp
        ## child
# python train.py -d "../../dataset/image/child/local_child_WEM_19215" -a "../../dataset/animation/child/child_19215.json" -ch "child" --save_ckpt -v "concat_aug_color"
# python train.py -d "../../dataset/image/child/local_child_WEM_19215" -a "../../dataset/animation/child/child_19215.json" -ck "2024_04_24_06_08_19-ch_child" -r "260000" -ch "child" --save_ckpt -v "concat_aug_color"
        ## mery 
# python train.py -d "../../dataset/image/mery/local_mery_WEM_18810" -a "../../dataset/animation/mery/mery_18810.json" -ch "mery" --save_ckpt -v "concat_aug_color"
# python train.py -d "../../dataset/image/mery/local_mery_WEM_18810" -a "../../dataset/animation/mery/mery_18810.json" -ck "2024_04_24_07_54_53-ch_mery" -r "280000" -ch "mery" --save_ckpt -v "concat_aug_color"
        ## malcolm
# python train.py -d "../../dataset/image/malcolm/local_malcolm_WEM_18810" -a "../../dataset/animation/malcolm/malcolm_18810.json" -ch "malcolm" --save_ckpt -v "concat_aug_color"
# python train.py -d "../../dataset/image/malcolm/local_malcolm_WEM_18810" -a "../../dataset/animation/malcolm/malcolm_18810.json" -ck "2024_04_24_07_55_08-ch_malcolm" -r "290000" -ch "malcolm" --save_ckpt -v "concat_aug_color"
  

    ## pca(normalized)
        ## child
# python train.py -d "../../dataset/image/child/local_child_WEM_19215" -a "../../dataset/animation/child/train_local_19215.pth" -ch "child" --save_ckpt -v "concat_aug_color"
# python train.py -d "../../dataset/image/child/local_child_WEM_19215" -a "../../dataset/animation/child/train_local_19215.pth" -ck "2024_04_24_06_40_54-ch_child" -r "250000" -ch "child" --save_ckpt -v "concat_aug_color"
        ## mery 
# python train.py -d "../../dataset/image/mery/local_mery_WEM_18810" -a "../../dataset/animation/mery/train_local_18810.pth" -ch "mery" --save_ckpt -v "concat_aug_color"
        ## malcolm
# python train.py -d "../../dataset/image/malcolm/local_malcolm_WEM_18810" -a "../../dataset/animation/malcolm/train_local_18810.pth" -ch "malcolm" --save_ckpt -v "concat_aug_color"

# c2c
# python train.py -d "../../dataset/image/mery/local_mery_WEM_18810_malcolm" -a "../../dataset/animation/mery/train_local_18810.pth" --save_ckpt -v "train_WEM" --char_name "mery"
# python train.py -d "../../dataset/image/malcolm/local_malcolm_WEM_18810_mery" -a "../../dataset/animation/malcolm/train_local_18810.pth" --save_ckpt -v "train_WEM" --char_name "malcolm"

# python train.py -d "../../dataset/image/malcolm/local_malcolm_WEM_18810_mery" -a "../../dataset/animation/malcolm/train_local_18810.pth" -v "train_WEM" --char_name "malcolm" --load_ckpt_name "2024_05_05_07_50_14-ch_malcolm" --load_iter '040000' --save_ckpt -v "train_WEM"



## PG2023 
## child
# python train.py -d "../../dataset/image/child/local_child" -a "../../dataset/animation/child/train_local.pth" -ch "child" --save_ckpt -v "train_BPNet"
# python train.py -d "../../dataset/image/child/local_child" -a "../../dataset/animation/child/train_local.pth" -ch "child" --save_ckpt -v "concat_aug_color"
# python train.py -d "../../dataset/image/child/local_child" -a "../../dataset/animation/child/train_local.pth" -ck "2023_07_11_11_35_47-ch_child" -r "380000" -ch "child" --save_ckpt -v "concat_aug_color"

## girl
# python train.py -d "../../dataset/image/girl/local_girl_refined_v2" -a "../../dataset/animation/girl/girl_refined_v2.pth" -ck "2023_07_11_11_34_31-ch_girl" -r "300000" -ch "girl" --save_ckpt -v "concat_aug_color"
# python train.py -d "../../dataset/image/girl/local_girl_refined_v2" -a "../../dataset/animation/girl/girl_refined_v2.pth" -ch "girl" --save_ckpt -v "concat_aug_color"
# python train.py -d "../../dataset/image/girl/local_girl_refined_v2" -a "../../dataset/animation/girl/girl_refined_v2.pth" -ch "girl" --save_ckpt -v "train_BPNet"
# python train.py -d "../../dataset/image/girl/local_girl" -a "../../dataset/animation/girl/train_local.pth" -ch "girl" --save_ckpt -v "train_BPNet"
# python train.py -d "../../dataset/image/girl/local_girl_refined_v2" -a "../../dataset/animation/girl/girl_refined_v2.pth" -ck "2023_07_08_12_00_45-ch_girl" -r "040000" --save_ckpt -v "train_BPNet"

## piers
# python train.py -d "../../dataset/image/piers/local_piers_refined" -a "../../dataset/animation/piers/piers_refined.pth" -ch "piers" --save_ckpt -v "train_BPNet"
# python train.py -d "../../dataset/image/piers/local_piers_refined" -a "../../dataset/animation/piers/piers_refined.pth" -ck "2023_07_11_07_59_19-ch_piers" -r "400000" -ch "piers" --save_ckpt -v "train_BPNet"
# python train.py -d "../../dataset/image/piers/local_piers_refined" -a "../../dataset/animation/piers/piers_refined.pth" -ch "piers" --save_ckpt -v "concat_aug_color"
# python train.py -d "../../dataset/image/piers/local_piers_refined" -a "../../dataset/animation/piers/piers_refined.pth" -ck "2023_07_11_11_37_07-ch_piers" -r "300000" -ch "piers" --save_ckpt -v "concat_aug_color"
# python train.py -d "../../dataset/image/piers/local_piers_refined" -a "../../dataset/animation/piers/piers_refined.pth" -ch "piers" --save_ckpt -v "concat_aug_color"

## mery
# python train.py -d "../../dataset/image/mery/local_mery" -a "../../dataset/animation/mery/train_local.pth" -ch "mery" --save_ckpt -v "train_BPNet"
# python train.py -d "../../dataset/image/mery/local_mery" -a "../../dataset/animation/mery/train_local.pth" -ch "mery" --save_ckpt -v "concat_aug_color"
# python train.py -d "../../dataset/image/mery/local_mery" -a "../../dataset/animation/mery/train_local.pth" -ck "2023_07_15_14_55_22-ch_mery-wColorAug" -r "490000" -ch "mery" --save_ckpt -v "concat_aug_color"

## malcolm
# python train.py -d "../../dataset/image/malcolm/local_malcolm" -a "../../dataset/animation/malcolm/train_local.pth" -ch "malcolm" --save_ckpt -v "train_BPNet"
# python train.py -d "../../dataset/image/malcolm/local_malcolm" -a "../../dataset/animation/malcolm/train_local.pth" -ch "malcolm" --save_ckpt -v "concat_aug_color"

## metahuman
# python train.py -d "../../dataset/image/metahuman/local_metahuman_refined" -a "../../dataset/animation/metahuman/train_local.pth" -ch "metahuman" --save_ckpt -v "train_BPNet"
# python train.py -d "../../dataset/image/metahuman/local_metahuman_refined" -a "../../dataset/animation/metahuman/train_local.pth" -ch "metahuman" --save_ckpt -v "concat_aug_color"
# python train.py -d "../../dataset/image/metahuman/metasihun/" -a "../../dataset/animation/real/train_local.pth" -ch "metasihun" --save_ckpt -v "concat_aug_color"

## metasihun
# python train.py -d "../../dataset/image/metahuman/metasihun/" -a "../../dataset/animation/metasihun/weight.pth" -ch "metasihun" --save_ckpt -v "concat_aug_color"


## CGF revision
    ## child
        ## bshp param
# python train.py -d "../../dataset/image/child/local_child_WEM_19215" -a "../../dataset/animation/child/child_19215.json" -ck "2024_04_10_17_51_37-ch_child" -ch "child" --save_ckpt -v "concat_aug_color"
    ## mery_deformed
# python train.py -d "../../dataset/image/mery_deformed/local_mery_deformed" -a "../../dataset/animation/mery_deformed/weight.pth" -ch "mery_deformed" --save_ckpt -v "concat_aug_color"


## CFG 2nd revision
# python train.py -d "../../dataset/image/metahuman/local_metahuman_refined" -a "../../dataset/animation/metahuman/train_local.pth" --char_name "metahuman" --save_ckpt -v "train_WEM"


## ICCV 2025 comparison

# python train.py --save_ckpt -ch "Emily" -d "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target/Emily" -a "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/blendshapes/bs_weight_Emily.pt" -v "concat_aug_color"
# python train.py --save_ckpt -ch "Victor" -d "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target/Victor" -a "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/blendshapes/bs_weight_Victor.pt" -v "concat_aug_color"
# python train.py --save_ckpt -ch "VMan" -d "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target/VMan" -a "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/blendshapes/bs_weight_VMan.pt" -v "concat_aug_color"
# python train.py --save_ckpt -ch "Morphy" -d "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target/Morphy" -a "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/blendshapes/bs_weight_Morphy.pt" -v "concat_aug_color"
# python train.py --save_ckpt -ch "Malcolm" -d "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target/Malcolm" -a "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/blendshapes/bs_weight_Malcolm.pt" -v "concat_aug_color"

## with reduced dataset version for training BN
python train.py --save_ckpt -ch "Emily" -d "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target_BN/Emily_BN" -a "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/blendshapes_BN/bs_weight_Emily_BN.pt" -v "concat_aug_color"
python train.py -ck "2025_02_24_11_47_25-ch_Emily" -r "030000" --save_ckpt -ch "Emily" -d "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target_BN/Emily_BN" -a "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/blendshapes_BN/bs_weight_Emily_BN.pt" -v "concat_aug_color"

python train.py --save_ckpt -ch "Victor" -d "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target_BN/Victor_BN" -a "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/blendshapes_BN/bs_weight_Victor_BN.pt" -v "concat_aug_color"
python train.py -ck "2025_02_24_12_36_42-ch_Victor" -r "020000" --save_ckpt -ch "Victor" -d "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target_BN/Victor_BN" -a "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/blendshapes_BN/bs_weight_Victor_BN.pt" -v "concat_aug_color"
python train.py -ck "2025_02_24_12_36_42-ch_Victor" -r "040000" --save_ckpt -ch "Victor" -d "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target_BN/Victor_BN" -a "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/blendshapes_BN/bs_weight_Victor_BN.pt" -v "concat_aug_color"

python train.py --save_ckpt -ch "VMan" -d "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target_BN/VMan_BN" -a "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/blendshapes_BN/bs_weight_VMan_BN.pt" -v "concat_aug_color"
python train.py -ck "2025_02_24_12_36_45-ch_VMan" -r "030000" --save_ckpt -ch "VMan" -d "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target_BN/VMan_BN" -a "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/blendshapes_BN/bs_weight_VMan_BN.pt" -v "concat_aug_color"
python train.py -ck "2025_02_24_12_36_45-ch_VMan" -r "042500" --save_ckpt -ch "VMan" -d "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target_BN/VMan_BN" -a "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/blendshapes_BN/bs_weight_VMan_BN.pt" -v "concat_aug_color"


python train.py --save_ckpt -ch "Morphy" -d "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target_BN/Morphy_BN" -a "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/blendshapes_BN/bs_weight_Morphy_BN.pt" -v "concat_aug_color"
python train.py -ck "2025_02_24_12_37_15-ch_Morphy" -r "025000" --save_ckpt -ch "Morphy" -d "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target_BN/Morphy_BN" -a "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/blendshapes_BN/bs_weight_Morphy_BN.pt" -v "concat_aug_color"
python train.py -ck "2025_02_24_12_37_15-ch_Morphy" -r "045000" --save_ckpt -ch "Morphy" -d "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target_BN/Morphy_BN" -a "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/blendshapes_BN/bs_weight_Morphy_BN.pt" -v "concat_aug_color"

python train.py --save_ckpt -ch "Malcolm" -d "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target_BN/Malcolm_BN" -a "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/blendshapes_BN/bs_weight_Malcolm_BN.pt" -v "concat_aug_color"
python train.py -ck "2025_02_24_12_37_19-ch_Malcolm" -r "025000" --save_ckpt -ch "Malcolm" -d "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target_BN/Malcolm_BN" -a "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/blendshapes_BN/bs_weight_Malcolm_BN.pt" -v "concat_aug_color"
python train.py -ck "2025_02_24_12_37_19-ch_Malcolm" -r "040000" --save_ckpt -ch "Malcolm" -d "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/images/target_BN/Malcolm_BN" -a "/input/inyup/PG2023/deeplearning/dataset/Your_Avatar_Talks/data/blendshapes_BN/bs_weight_Malcolm_BN.pt" -v "concat_aug_color"
