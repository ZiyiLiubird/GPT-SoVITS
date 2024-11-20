export CUDA_VISIBLE_DEVICES=4
nohup uvicorn main_online_v2_Zhongli:app --host '0.0.0.0' --port 7005 --log-level warning >server_online_7005.log 2>&1 &
