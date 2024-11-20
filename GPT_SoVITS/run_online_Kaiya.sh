export CUDA_VISIBLE_DEVICES=4
nohup uvicorn main_online_v2_Kaiya:app --host '0.0.0.0' --port 7002 --log-level warning >server_online_7002.log 2>&1 &
