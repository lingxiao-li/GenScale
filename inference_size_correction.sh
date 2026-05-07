export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -z "${GOOGLE_API_KEY}" ]; then
  echo "ERROR: GOOGLE_API_KEY is not set."
  echo "Run: export GOOGLE_API_KEY=your_real_key"
  exit 1
fi

if [ -z "${IMAGE_PATH}" ]; then
  echo "ERROR: IMAGE_PATH is not set."
  echo "Run: IMAGE_PATH=/path/to/input.png bash $0"
  exit 1
fi

# IMPORTANT (SCC/Slurm/SGE):
# Do NOT manually set CUDA_VISIBLE_DEVICES here.
# Let the scheduler expose assigned GPU(s) automatically.

python "$(dirname "$0")/inference_size_correction.py" \
  --image_path "${IMAGE_PATH:-}" \
  --output_dir "${OUTPUT_DIR:-$(dirname "$0")/../outputs/size_correction_T1}" \
  --feedback_json_path "${FEEDBACK_JSON_PATH:-}" \
  --use_feedback_planner 1 \
  --feedback_lookup_task_id "" \
  --feedback_pair_index 0 \
  --multi_round 1 \
  --multi_round_scene_prefilter 1 \
  --max_round_objects 4 \
  --anchor_object "" \
  --bg_overremove_check 1 \
  --bg_overremove_model "gemini-3-flash-preview" \
  --min_scale 0.3333333333333333 \
  --max_scale 3.0 \
  --min_scale_change 0.08 \
  --weights_dir "${INSERTANYTHING_WEIGHTS_DIR:-$(dirname "$0")/weights/499000}" \
  --gpu_gen 0 \
  --gpu_tools 0 \
  --cache_models_cpu 1 \
  --crop_ratio 2.5 \
  --mask_dilate_kernel 25 \
  --mask_dilate_iter 3 \
  --mask_dilate_boost 1.35 \
  --mask_dilate_iter_boost 1.25 \
  --mask_dilate_depth_aware 1 \
  --mask_dilate_ratio 0.30 \
  --mask_inpaint_regularize close_convex_hull \
  --mask_regularize_close_ksz 15 \
  --mask_regularize_close_iter 2 \
  --mask_crop_fill_holes 1 \
  --mask_crop_post_dilate 1 \
  --ref_upscale_threshold 128 \
  --ref_upscale_target 338 \
  --blend_feather_border_px 16 \
  --blend_alpha_blur_sigma 5.0 \
  --disable_blend_feather 0 \
  --bg_removal_mode gemini \
  --bg_remove_max_retries 2 \
  --bg_remove_diff_thresh 0.065 \
  --bg_remove_changed_ratio_thresh 0.20 \
  --bg_remove_pixel_diff_cutoff 0.06 \
  --bg_preserve_diff_mean_max 0.05 \
  --bg_preserve_changed_ratio_max 0.20 \
  --enforce_bg_preserve 0 \
  --prefer_gemini_cleanup 0 \
  --gemini_cleanup_only 0 \
  --use_gemini_cleanup_fallback 1 \
  --use_seg_bbox_for_transform 1 \
  --num_steps 30 \
  --guidance_scale 30.0 \
  --controlnet_scale 0.7 \
  --controlnet_end 0.6 \
  --seed 42 \
  --presence_diff_thresh 0.03 \
  --nodepth_prefer_ratio 2.0 \
  --enable_nodepth_compare 0 \
  --force_no_depth_control 0 \
  --bottom_center_depth_bias 50.0 \
  --depth_coverage_nodepth_gate 0.60
