export CUDA_VISIBLE_DEVICES=6
uvicorn main_online_v2_sanyueqi:app --host '0.0.0.0' --port 7002 --log-level warning >server_online_7002.log 2>&1 &
