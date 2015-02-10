#include <iostream>
#include "float2.hpp"

int main(int argc, char *argv[])
{
    double2 x, y, one(1.,0.), I(0.,1.);
    x = 3.*one + 2.*I;
    y = double2(3.,4.);
    std::cout << "hello"
    <<" "<<one
    <<" "<<I
    <<" "<<x
    <<" "<<y
    <<" "<<(x+y)
    <<" "<<dot(x,y)
    <<" "<<length(x)
    <<std::endl;

    return 0;
}