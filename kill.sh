ps -ef | grep sample_video.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep inference.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep test.sh | grep -v grep | awk '{print $2}' | xargs kill -9