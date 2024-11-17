mpiFCC -Nclang -std=c++11 -Ofast -o graphSparsity_3hit_loadBalance graphSparsity_3hit_loadBalance.cpp && mpirun ./graphSparsity_3hit_loadBalance sample.txt  && cat output.txt
