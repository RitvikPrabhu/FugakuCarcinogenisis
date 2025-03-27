#bitwise
mpiFCCpx -Kparallel -Kfast -std=c++11 -DENABLE_TIMING -I../include ../src/main.cpp ../src/readFile.cpp ../src/fourHit.cpp -o run_bit

#CPP set
#mpiFCCpx -Kparallel -Kfast -std=c++11 -DUSE_CPP_SET -I../include ../src/main.cpp ../src/readFile.cpp ../src/fourHit.cpp -o run_set_normal
