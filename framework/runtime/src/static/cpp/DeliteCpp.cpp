#include "DeliteCpp.h"

#ifdef MEMMGR_REFCNT
std::shared_ptr<cppDeliteArraystring> string_split(string str, string pattern) {
#else
cppDeliteArraystring *string_split(string str, string pattern) {
#endif
  //TODO: current g++ does not fully support c++11 regex, 
  //      so below code does not work.
  /*
  std::string s(str);
  std::regex e(pattern.c_str());
  std::vector<std::string> *elems = new std::vector<std::string>();

  const std::sregex_token_iterator endOfSequence;
  std::sregex_token_iterator token(s.begin(), s.end(), e, -1);
  while(token != endOfSequence) {
    elems->push_back(*token++);
    std::cout << *token++ << std::endl;
  }

  cppDeliteArray<string> *ret = new cppDeliteArray<string>(elems->size());
  for(int i=0; i<elems->size(); i++)
    ret->update(i,elems->at(i));
  return ret;
  */

  //Since regex is not working, we currently only support 
  assert((pattern.compare("\\s+")==0 || pattern.compare(" ")==0) && "Currently only regex \\s+ is supported for C++ target");
  string token;
  stringstream ss(str); 
  vector<string> elems;
  while (ss >> token)
    elems.push_back(token);
  
  //construct cppDeliteArray from vector
#ifdef MEMMGR_REFCNT
  std::shared_ptr<cppDeliteArraystring> ret(new cppDeliteArraystring(elems.size()), cppDeliteArraystringD());
#else
  cppDeliteArraystring *ret = new cppDeliteArraystring(elems.size());
#endif
  for(int i=0; i<elems.size(); i++)
    ret->update(i,elems.at(i));
  return ret;
}

int32_t string_toInt(string str) {
  return atoi(str.c_str());
}

float string_toFloat(string str) {
  return strtof(str.c_str(),NULL);
}

double string_toDouble(string str) {
  return strtod(str.c_str(),NULL);
}

bool string_toBoolean(string str) {
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  if (str.compare("true") == 0)
    return true;
  else if (str.compare("false") == 0)
    return false;
  else
    assert(false && "Cannot parse boolean string");
}

// Code from http://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
string &ltrim(string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

string &rtrim(string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
  return s;
}

string string_trim(string str) {
  string ret = str;
  return ltrim(rtrim(ret));
}

int8_t string_charAt(string str, int idx) {
  return str.at(idx);
}

bool string_startsWith(string str, string substr) {
  if (str.compare(0,substr.length(),substr) == 0)
    return true;
  else
    return false;
}

string string_plus(string str1, string str2) {
  return str1 + str2;
}


#ifdef MEMMGR_REFCNT
std::shared_ptr<cppDeliteArraystring> cppArgsGet(int num, ...) {
  std::shared_ptr<cppDeliteArraystring> cppArgs(new cppDeliteArraystring(num), cppDeliteArraystringD());
#else
cppDeliteArraystring *cppArgsGet(int num, ...) {
  cppDeliteArraystring *cppArgs = new cppDeliteArraystring(num);
#endif
  va_list arguments;
  va_start(arguments, num);
  for(int i=0; i<num; i++) {
    char *pathname = va_arg(arguments, char *);
    cppArgs->data[i] = string(pathname);
  }
  va_end(arguments);
  return cppArgs;
}

template<class T> string convert_to_string(T in) {
  ostringstream convert;
  convert << in;
  return convert.str();
}

// Explicit instantiation of template functions to enable separate compilation
template string convert_to_string<bool>(bool);
template string convert_to_string<int8_t>(int8_t);
template string convert_to_string<uint16_t>(uint16_t);
template string convert_to_string<int16_t>(int16_t);
template string convert_to_string<int32_t>(int32_t);
template string convert_to_string<int64_t>(int64_t);
template string convert_to_string<float>(float);
template string convert_to_string<double>(double);
template string convert_to_string<string>(string);
template string convert_to_string<void*>(void*);

string readFirstLineFile(string filename) {
  ifstream fs(filename.c_str());
  string line;
  if (fs.good()) {
    getline(fs, line);
  }
  fs.close();
  return line;
}

/* helper methods and data structures only required for execution with Delite */
#ifndef __DELITE_CPP_STANDALONE__
pthread_mutex_t lock_objmap = PTHREAD_MUTEX_INITIALIZER;
std::map<int,jobject> *JNIObjectMap = new std::map<int,jobject>();
jobject JNIObjectMap_find(int key) {
  pthread_mutex_lock (&lock_objmap);
  jobject ret = JNIObjectMap->find(key)->second;
  pthread_mutex_unlock (&lock_objmap);
  return ret;
}
void JNIObjectMap_insert(int key, jobject value) {
  pthread_mutex_lock (&lock_objmap);
  std::map<int,jobject>::iterator it = JNIObjectMap->find(key);
  if(it != JNIObjectMap->end()) 
    it->second = value;
  else
    JNIObjectMap->insert(std::pair<int,jobject>(key,value));
  pthread_mutex_unlock (&lock_objmap);
}
#endif
