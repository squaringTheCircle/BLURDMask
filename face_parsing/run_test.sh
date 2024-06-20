source .env;
export DATE="2024-03-03-1602"
python -u main.py --batch_size 4 --imsize 512 --version $1 --train False  \
    --model_save_path "$DATA_DIR/runs/$DATE/models"                             \
    --model_name "top_00_score_1.29_step_0001059.pth"                          \
    --test_image_path "$DATA_DIR/test_img"                                             \
    --test_label_path "$DATA_DIR/test_label"                                    \
    --test_results_path "$DATA_DIR/runs/$DATE/test_results"                     \
    --test_color_label_path "$DATA_DIR/runs/$DATE/test_color_visualize"





