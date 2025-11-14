export CUDA_VISIBLE_DEVICES=0,1,2,3
python sample_video.py \
    --dit-weight checkpoint/diffusion_pytorch_model.safetensors \
    --video-length 121 \
    --infer-steps 50 \
    --seed 42 \
    --embedded-cfg-scale 1.0 \
    --cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --save-path ./long_video_results_121_frames \
    --video /scratch3/yan204/yxp/VideoX_Fun/data/test_json/long_video_new.json 