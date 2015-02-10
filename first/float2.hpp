// length 2 vector arithmetic in c++, matching openCL syntax
// Can't do V.yx to return a copy of V with x and y swapped
// Only a partial implementation used to test that my cl_complex.h works properly.
#include <cmath>
#include <iostream>

template <class T>
class V2 {
public:
       T x,y;

       // Construct/destructor from empty, scalar, scalar pair or vector
       // Not using copy-and-swap idiom because I don't understand it
       V2() {} // uninitialized x,y if declared without initial value
       V2(T x) : x(x), y(x) {}
       V2(T x, T y) : x(x), y(y) {}
       V2(const V2<T>& other) : x(other.x), y(other.y) {}
       V2& operator=(const T& rhs) { this->x=rhs; this->y=rhs; return *this; }
       V2& operator=(const V2<T>& rhs) { if (this != &rhs) {this->x=rhs.x; this->y=rhs.y;} return *this; }
       ~V2() {}

       // in-place binary operators
       V2& operator+=(const V2<T>& other) { this->x+=other.x; this->y+=other.y; return *this; }
       V2& operator-=(const V2<T>& other) { this->x-=other.x; this->y-=other.y; return *this; }
       V2& operator*=(const V2<T>& other) { this->x*=other.x; this->y*=other.y; return *this; }
       V2& operator/=(const V2<T>& other) { this->x/=other.x; this->y/=other.y; return *this; }

       // shuffle properties
       const V2& xy() { return V2<T>(*this); }
       const V2& yx() { return V2<T>(this->y,this->x); }
} ;

// binary arithmetic: vector OP vector, scalar OP vector, vector OP scalar
template <class T> inline V2<T> operator+(V2<T> lhs, const V2<T> &rhs) { lhs+=rhs; return lhs; }
template <class T> inline V2<T> operator+(const V2<T> &lhs, const T &rhs) { return lhs+V2<T>(rhs); }
template <class T> inline V2<T> operator+(const T &lhs, const V2<T> &rhs) { return V2<T>(lhs)+rhs; }

template <class T> inline V2<T> operator-(V2<T> lhs, const V2<T> &rhs) { lhs-=rhs; return lhs; }
template <class T> inline V2<T> operator-(const V2<T> &lhs, const T &rhs) { return lhs-V2<T>(rhs); }
template <class T> inline V2<T> operator-(const T &lhs, const V2<T> &rhs) { return V2<T>(lhs)-rhs; }
// unary minus
template <class T> inline V2<T> operator-(const V2<T> &lhs) { return T(0) - lhs; }
template <class T> inline std::ostream& operator<<(std::ostream& os, const V2<T>& obj) { return os<<"("<<obj.x<<","<<obj.y<<")"; }

template <class T> inline V2<T> operator*(V2<T> lhs, const V2<T> &rhs) { lhs*=rhs; return lhs; }
template <class T> inline V2<T> operator*(const V2<T> &lhs, const T &rhs) { return lhs*V2<T>(rhs); }
template <class T> inline V2<T> operator*(const T &lhs, const V2<T> &rhs) { return V2<T>(lhs)*rhs; }

template <class T> inline V2<T> operator/(V2<T> lhs, const V2<T> &rhs) { lhs/=rhs; return lhs; }
template <class T> inline V2<T> operator/(const V2<T> &lhs, const T &rhs) { return lhs/V2<T>(rhs); }
template <class T> inline V2<T> operator/(const T &lhs, const V2<T> &rhs) { return V2<T>(lhs)/rhs; }

// A complete implementation would include the functions in cmath applied component-wise
// ...

// vector math
template <class T> inline T dot(const V2<T> &p1, const V2<T> &p2) { return p1.x*p2.x + p1.y*p2.y; }
template <class T> inline T length(const V2<T> &p1) { return sqrt(dot(p1,p1)); }
template <class T> inline T distance(const V2<T> &p1, const V2<T> &p2) { return length(p2-p1); }
template <class T> inline T normalize(const V2<T> &p1) { return p1/length(p1); }

template <class T> inline int vec_step(const V2<T> &lhs) { return 2; }
//template <class T, class U> inline V2<T> shuffle(const V2<T> &lhs, const V2<U> &mask) { }

typedef V2<float> float2;
typedef V2<double> double2;
typedef V2<int> int2;

