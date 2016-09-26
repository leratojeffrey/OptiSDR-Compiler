#ifndef __cppDeliteArraydouble__
#define __cppDeliteArraydouble__

using namespace std;
class cppDeliteArraydouble {
public:
  double  *data;
  int length;

  cppDeliteArraydouble(int _length) {
    data = new double [_length]();
    length = _length;
    //printf("allocated cppDeliteArraydouble, size %d, %p\n",_length,this);
  }

  cppDeliteArraydouble(double  *_data, int _length) {
    data = _data;
    length = _length;
  }

  double  apply(int idx) {
    return data[idx];
  }

  void update(int idx, double  val) {
    data[idx] = val;
  }

  void print(void) {
    printf("length is %d\n", length);
  }
};

struct cppDeliteArraydoubleD {
  void operator()(cppDeliteArraydouble *p) {
    //printf("cppDeliteArraydouble: deleting %p\n",p);
    delete[] p->data;
  }
};

#endif