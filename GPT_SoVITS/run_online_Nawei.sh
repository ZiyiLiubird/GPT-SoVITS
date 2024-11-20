export CUDA_VISIBLE_DEVICES=4
nohup uvicorn main_online_v2_Nawei:app --host '0.0.0.0' --port 7004 --log-level warning >server_online_7004.log 2>&1 &
