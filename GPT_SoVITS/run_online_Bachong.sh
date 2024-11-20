export CUDA_VISIBLE_DEVICES=4
nohup uvicorn main_online_v2_Bachong:app --host '0.0.0.0' --port 7003 --log-level warning >server_online_7003.log 2>&1 &
