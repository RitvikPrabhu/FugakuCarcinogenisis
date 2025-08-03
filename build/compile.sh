#bitwise
mpiFCCpx -Kfast -std=c++11 -DENABLE_PROFILE -DNUMHITS=4 -DCHUNK_SIZE=2 -I../include ../src/main.cpp ../src/readFile.cpp ../src/multiHit.cpp -o run_4hit

