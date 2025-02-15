#ifndef COMMONS_H
#define COMMONS_H

#ifdef USE_CPP_SET 
    #include <set>
    #include <vector>
    #include <cstddef>

    typedef std::set<int> SET;
    typedef std::vector<std::set<int>> SET_COLLECTION;
    

#else
    typedef uint64_t unit_t;
    typedef unit_t SET;
    typedef unit_t* SET_COLLECTION;


#endif