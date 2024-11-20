export CUDA_VISIBLE_DEVICES=5
nohup uvicorn main_online_v2_sanyueqi:app --host '0.0.0.0' --port 7001 --log-level warning >server_online_7001.log 2>&1 &
# uvicorn main_online_v2_sanyueqi:app --host '0.0.0.0' --port 7012
