#bitwise
mpiFCCpx -Kparallel -Kfast -std=c++11 -I../include ../src/main.cpp ../src/readFile.cpp ../src/fourHit.cpp -o run_bit

#CPP set
mpiFCCpx -Kparallel -Kfast -std=c++11 -DUSE_CPP_SET=ON -I../include ../src/main.cpp ../src/readFile.cpp ../src/fourHit.cpp -o run_bit
