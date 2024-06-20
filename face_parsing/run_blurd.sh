source .env;
export DATE=$(date +"%Y-%m-%d-%H%M")
python3 -u main.py --batch_size 8 --imsize 512 --version $1   \
    --img_path "$BLURD_DIR"                                                        \
    --label_path "$DATA_DIR/train_label"                                                    \
    --val_img_path "$DATA_DIR/val_img"                                                      \
    --val_label_path "$DATA_DIR/val_label"                                                  \
    --test_image_path "$DATA_DIR/test_img"                                                  \
    --log_path "$DATA_DIR/runs/$DATE/logs"                                                  \
    --model_save_path "$DATA_DIR/runs/$DATE/models"                                         \
    --sample_path "$DATA_DIR/runs/$DATE/samples"                                            \
    --model_save_step 0.1 --num_workers 4 --top_save_num 20 --sample_step 300
